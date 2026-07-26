"""Deterministic, resumable Hugging Face Dataset Viewer row loading."""

from __future__ import annotations

import http.client
import json
import os
import random
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO
from urllib.parse import quote, urlencode, urljoin, urlsplit

from .provenance import canonical_json_bytes, sha256_bytes

VIEWER_PAGE_SIZE = 100
CACHE_SCHEMA_VERSION = 1
HUB_DATASET_API = "https://huggingface.co/api/datasets"
DATASET_VIEWER_ROWS_API = "https://datasets-server.huggingface.co/rows"
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class DatasetViewerSource(Protocol):
    """Fields required from a pinned dataset source."""

    dataset: str
    subset: str
    split: str
    revision: str

    @property
    def key(self) -> str: ...


class HTTPGetter(Protocol):
    """Injectable byte transport used to test the loader without a network."""

    def __call__(
        self,
        url: str,
        *,
        connect_timeout: float,
        read_timeout: float,
    ) -> bytes: ...


class DatasetViewerError(RuntimeError):
    """A Dataset Viewer response or transport could not be trusted."""


class DatasetRevisionDriftError(DatasetViewerError):
    """The dataset default branch no longer matches the frozen revision."""


class DatasetViewerCacheError(DatasetViewerError):
    """A cached page failed its identity or content-integrity check."""


class _HTTPStatusError(OSError):
    def __init__(self, status: int, url: str) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.retryable = status in {408, 425, 429} or 500 <= status < 600


def _http_get_bytes(
    url: str,
    *,
    connect_timeout: float,
    read_timeout: float,
    redirects_remaining: int = 3,
) -> bytes:
    """GET bytes with separate connection and socket-read timeouts."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"unsupported HTTP URL: {url}")
    connection_class = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_class(
        parsed.hostname,
        parsed.port,
        timeout=connect_timeout,
    )
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    try:
        connection.connect()
        if connection.sock is None:
            raise OSError("HTTP connection did not expose a socket")
        connection.sock.settimeout(read_timeout)
        connection.request(
            "GET",
            path,
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "cliffquant-dataset-viewer/0.1",
            },
        )
        response = connection.getresponse()
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise DatasetViewerError(f"HTTP response exceeded {_MAX_RESPONSE_BYTES} bytes: {url}")
        if response.status in _REDIRECT_STATUSES:
            location = response.getheader("Location")
            if not location or redirects_remaining <= 0:
                raise DatasetViewerError(f"unusable HTTP redirect for {url}")
            return _http_get_bytes(
                urljoin(url, location),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                redirects_remaining=redirects_remaining - 1,
            )
        if response.status != 200:
            raise _HTTPStatusError(response.status, url)
        return body
    finally:
        connection.close()


def _strict_json(data: bytes, *, context: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        parsed = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetViewerError(f"invalid JSON from {context}") from exc
    if not isinstance(parsed, Mapping):
        raise DatasetViewerError(f"expected a JSON object from {context}")
    return parsed


def _request_json(
    url: str,
    *,
    http_get: HTTPGetter,
    connect_timeout: float,
    read_timeout: float,
    max_retries: int,
    retry_backoff: float,
    progress: TextIO,
    label: str,
) -> Mapping[str, Any]:
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            payload = http_get(
                url,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            if not isinstance(payload, bytes):
                raise TypeError("HTTP getter must return bytes")
            return _strict_json(payload, context=label)
        except _HTTPStatusError as exc:
            if not exc.retryable:
                raise DatasetViewerError(str(exc)) from exc
            last_error = exc
        except (DatasetViewerError, OSError, TimeoutError) as exc:
            last_error = exc
        if attempt == max_retries:
            break
        print(
            f"[cliffquant] retry {attempt + 1}/{max_retries}: {label}",
            file=progress,
            flush=True,
        )
        if retry_backoff:
            time.sleep(retry_backoff * (2**attempt))
    raise DatasetViewerError(f"{label} failed after {max_retries + 1} attempts") from last_error


def _source_identity(source: DatasetViewerSource) -> dict[str, Any]:
    return {
        "config": source.subset,
        "dataset": source.dataset,
        "revision": source.revision,
        "split": source.split,
    }


def _derived_seed(
    source: DatasetViewerSource,
    *,
    environment: str,
    seed: int,
) -> int:
    material = canonical_json_bytes(
        {
            "environment": environment,
            "seed": seed,
            "source": _source_identity(source),
        }
    )
    return int(sha256_bytes(material)[:16], 16)


def _page_request(source: DatasetViewerSource, page_index: int) -> dict[str, Any]:
    return {
        **_source_identity(source),
        "length": VIEWER_PAGE_SIZE,
        "offset": page_index * VIEWER_PAGE_SIZE,
        "page_index": page_index,
    }


def _cache_source_directory(cache_dir: Path, source: DatasetViewerSource) -> Path:
    identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "page_size": VIEWER_PAGE_SIZE,
        "source": _source_identity(source),
    }
    return cache_dir / "pages" / sha256_bytes(canonical_json_bytes(identity))


def _read_cached_page(
    cache_dir: Path,
    *,
    source: DatasetViewerSource,
    page_index: int,
) -> tuple[Mapping[str, Any], str, Path] | None:
    source_dir = _cache_source_directory(cache_dir, source)
    candidates = sorted(source_dir.glob(f"page-{page_index:08d}-*.json"))
    if not candidates:
        return None

    valid: list[tuple[Mapping[str, Any], str, Path]] = []
    expected_request = _page_request(source, page_index)
    for path in candidates:
        try:
            envelope = _strict_json(path.read_bytes(), context=str(path))
            if envelope.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                raise DatasetViewerCacheError(f"cache schema mismatch: {path}")
            if envelope.get("request") != expected_request:
                raise DatasetViewerCacheError(f"cache request identity mismatch: {path}")
            response = envelope.get("response")
            response_sha256 = envelope.get("response_sha256")
            if not isinstance(response, Mapping) or not isinstance(response_sha256, str):
                raise DatasetViewerCacheError(f"malformed cache envelope: {path}")
            actual_hash = sha256_bytes(canonical_json_bytes(response))
            if actual_hash != response_sha256:
                raise DatasetViewerCacheError(f"cache response hash mismatch: {path}")
            if path.name != f"page-{page_index:08d}-{response_sha256}.json":
                raise DatasetViewerCacheError(f"cache filename hash mismatch: {path}")
            valid.append((response, response_sha256, path))
        except OSError as exc:
            raise DatasetViewerCacheError(f"could not read cached page: {path}") from exc

    hashes = {item[1] for item in valid}
    if len(hashes) != 1:
        raise DatasetViewerCacheError(
            f"multiple cached contents for source page {page_index}: {sorted(hashes)}"
        )
    return valid[0]


def _write_cached_page(
    cache_dir: Path,
    *,
    source: DatasetViewerSource,
    page_index: int,
    response: Mapping[str, Any],
) -> tuple[str, Path]:
    source_dir = _cache_source_directory(cache_dir, source)
    source_dir.mkdir(parents=True, exist_ok=True)
    response_sha256 = sha256_bytes(canonical_json_bytes(response))
    destination = source_dir / f"page-{page_index:08d}-{response_sha256}.json"
    if destination.is_file():
        return response_sha256, destination
    envelope = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "request": _page_request(source, page_index),
        "response": response,
        "response_sha256": response_sha256,
    }
    payload = canonical_json_bytes(envelope)
    temporary = source_dir / f".tmp-{os.getpid()}-{uuid.uuid4().hex}.json"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return response_sha256, destination


def _rows_url(source: DatasetViewerSource, page_index: int) -> str:
    query = urlencode(
        {
            "dataset": source.dataset,
            "config": source.subset,
            "split": source.split,
            "offset": page_index * VIEWER_PAGE_SIZE,
            "length": VIEWER_PAGE_SIZE,
        }
    )
    return f"{DATASET_VIEWER_ROWS_API}?{query}"


def _verify_current_revision(
    source: DatasetViewerSource,
    *,
    http_get: HTTPGetter,
    connect_timeout: float,
    read_timeout: float,
    max_retries: int,
    retry_backoff: float,
    progress: TextIO,
    label_suffix: str,
) -> str:
    metadata_url = f"{HUB_DATASET_API}/{quote(source.dataset, safe='/')}"
    metadata = _request_json(
        metadata_url,
        http_get=http_get,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        progress=progress,
        label=f"{source.dataset} Hub revision {label_suffix}",
    )
    current_sha = metadata.get("sha")
    if not isinstance(current_sha, str) or _COMMIT_SHA.fullmatch(current_sha) is None:
        raise DatasetViewerError(f"Hub metadata did not include a SHA for {source.dataset}")
    if current_sha != source.revision:
        raise DatasetRevisionDriftError(
            f"dataset revision drift for {source.dataset}: "
            f"expected {source.revision}, current {current_sha}"
        )
    return current_sha


def _load_page(
    source: DatasetViewerSource,
    *,
    page_index: int,
    cache_dir: Path,
    http_get: HTTPGetter,
    connect_timeout: float,
    read_timeout: float,
    max_retries: int,
    retry_backoff: float,
    progress: TextIO,
) -> tuple[Mapping[str, Any], str, Path, bool]:
    cached = _read_cached_page(cache_dir, source=source, page_index=page_index)
    if cached is not None:
        response, response_sha256, path = cached
        return response, response_sha256, path, True
    response = _request_json(
        _rows_url(source, page_index),
        http_get=http_get,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        progress=progress,
        label=f"{source.dataset}/{source.subset}/{source.split} page {page_index}",
    )
    response_sha256, path = _write_cached_page(
        cache_dir,
        source=source,
        page_index=page_index,
        response=response,
    )
    return response, response_sha256, path, False


def _total_rows(response: Mapping[str, Any]) -> int:
    value = response.get("num_rows_total")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetViewerError("Dataset Viewer returned an invalid num_rows_total")
    return value


def _validated_page_rows(
    response: Mapping[str, Any],
    *,
    page_index: int,
    total_rows: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    response_total = _total_rows(response)
    if response_total != total_rows:
        raise DatasetViewerError(
            f"Dataset Viewer total changed within one run: {total_rows} -> {response_total}"
        )
    if response.get("partial") is not False:
        raise DatasetViewerError(
            f"Dataset Viewer page {page_index} did not explicitly declare partial=false"
        )
    entries = response.get("rows")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        raise DatasetViewerError(f"Dataset Viewer page {page_index} has invalid rows")

    offset = page_index * VIEWER_PAGE_SIZE
    expected_count = max(0, min(VIEWER_PAGE_SIZE, total_rows - offset))
    if len(entries) != expected_count:
        raise DatasetViewerError(
            f"Dataset Viewer page {page_index} has {len(entries)} rows; expected {expected_count}"
        )

    validated: list[tuple[int, Mapping[str, Any]]] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise DatasetViewerError(f"Dataset Viewer page {page_index} has a malformed row")
        row_index = entry.get("row_idx")
        expected_index = offset + position
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or row_index != expected_index
        ):
            raise DatasetViewerError(
                f"Dataset Viewer row index mismatch: expected {expected_index}, got {row_index!r}"
            )
        truncated_cells = entry.get("truncated_cells")
        if not isinstance(truncated_cells, Sequence) or isinstance(
            truncated_cells, (str, bytes, bytearray)
        ):
            raise DatasetViewerError(
                f"Dataset Viewer row {row_index} has invalid truncated_cells metadata"
            )
        if truncated_cells:
            raise DatasetViewerError(f"Dataset Viewer row {row_index} contains truncated cells")
        row = entry.get("row")
        if not isinstance(row, Mapping):
            raise DatasetViewerError(f"Dataset Viewer row {row_index} is not an object")
        validated.append((row_index, row))
    return validated


def load_dataset_viewer_rows(
    source: DatasetViewerSource,
    *,
    environment: str,
    seed: int,
    max_valid_records: int,
    cache_dir: str | Path,
    is_valid: Callable[[Mapping[str, Any], int], bool],
    audit: MutableMapping[str, Any] | None = None,
    connect_timeout: float = 10.0,
    read_timeout: float = 30.0,
    max_retries: int = 3,
    retry_backoff: float = 0.5,
    http_get: HTTPGetter | None = None,
    progress: TextIO | None = None,
) -> list[Mapping[str, Any]]:
    """Load a deterministic bounded sample through cached 100-row Viewer pages."""

    if max_valid_records <= 0:
        raise ValueError("max_valid_records must be positive")
    if _COMMIT_SHA.fullmatch(source.revision) is None:
        raise ValueError("source revision must be a lowercase 40-character commit SHA")
    if connect_timeout <= 0 or read_timeout <= 0:
        raise ValueError("connect_timeout and read_timeout must be positive")
    if max_retries < 0 or retry_backoff < 0:
        raise ValueError("max_retries and retry_backoff must be nonnegative")
    cache_root = Path(cache_dir)
    output = progress if progress is not None else sys.stderr
    transport = http_get if http_get is not None else _http_get_bytes

    current_sha = _verify_current_revision(
        source,
        http_get=transport,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        progress=output,
        label_suffix="before rows",
    )
    print(
        f"[cliffquant] verified {source.dataset}@{current_sha[:12]} ({environment})",
        file=output,
        flush=True,
    )

    first_response, first_hash, first_path, first_cached = _load_page(
        source,
        page_index=0,
        cache_dir=cache_root,
        http_get=transport,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        progress=output,
    )
    total_rows = _total_rows(first_response)
    if total_rows == 0:
        raise DatasetViewerError(f"pinned source has no rows: {source.key}")
    _validated_page_rows(first_response, page_index=0, total_rows=total_rows)

    page_count = (total_rows + VIEWER_PAGE_SIZE - 1) // VIEWER_PAGE_SIZE
    derived_seed = _derived_seed(source, environment=environment, seed=seed)
    page_order = list(range(page_count))
    random.Random(derived_seed).shuffle(page_order)
    print(
        f"[cliffquant] {source.dataset}/{source.split}: {total_rows} rows, "
        f"{page_count} pages, target {max_valid_records}",
        file=output,
        flush=True,
    )

    metadata_page = {
        "cache_file": first_path.relative_to(cache_root).as_posix(),
        "cache_hit": first_cached,
        "page_index": 0,
        "response_sha256": first_hash,
    }
    selection_pages: list[dict[str, Any]] = []
    rows: list[Mapping[str, Any]] = []
    for position, page_index in enumerate(page_order, start=1):
        if page_index == 0:
            response, response_hash, cache_path, cache_hit = (
                first_response,
                first_hash,
                first_path,
                first_cached,
            )
        else:
            response, response_hash, cache_path, cache_hit = _load_page(
                source,
                page_index=page_index,
                cache_dir=cache_root,
                http_get=transport,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                progress=output,
            )
        valid_before = len(rows)
        for row_index, row in _validated_page_rows(
            response,
            page_index=page_index,
            total_rows=total_rows,
        ):
            enriched = dict(row)
            enriched["__cliffquant_source_index__"] = row_index
            if not is_valid(enriched, row_index):
                continue
            rows.append(enriched)
            if len(rows) == max_valid_records:
                break
        selection_pages.append(
            {
                "accepted_valid_records": len(rows) - valid_before,
                "cache_file": cache_path.relative_to(cache_root).as_posix(),
                "page_index": page_index,
                "response_sha256": response_hash,
            }
        )
        print(
            f"[cliffquant] page {position}/{page_count} index={page_index} "
            f"{'cache' if cache_hit else 'fetched'} valid={len(rows)}/{max_valid_records}",
            file=output,
            flush=True,
        )
        if len(rows) == max_valid_records:
            break

    if not rows:
        raise DatasetViewerError(f"pinned source produced no valid records: {source.key}")
    ending_sha = _verify_current_revision(
        source,
        http_get=transport,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        progress=output,
        label_suffix="after rows",
    )
    if ending_sha != current_sha:  # pragma: no cover - both must equal the frozen pin
        raise DatasetRevisionDriftError(
            f"dataset revision changed during collection: {current_sha} -> {ending_sha}"
        )
    if audit is not None:
        audit.clear()
        audit.update(
            {
                "accepted_valid_records": len(rows),
                "backend": "hugging-face-dataset-viewer/rows",
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "derived_page_seed": derived_seed,
                "max_valid_records": max_valid_records,
                "metadata_page": {
                    key: value for key, value in metadata_page.items() if key != "cache_hit"
                },
                "page_count": page_count,
                "page_order_sha256": sha256_bytes(canonical_json_bytes(page_order)),
                "page_size": VIEWER_PAGE_SIZE,
                "selection_pages": selection_pages,
                "source": _source_identity(source),
                "total_rows": total_rows,
                "verified_current_sha": current_sha,
            }
        )
    return rows
