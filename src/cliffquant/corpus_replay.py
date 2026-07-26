"""Offline, byte-strict replay verification for Experiment 001 corpus bundles."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .corpus import (
    CALIBRATION_SOURCES,
    CALIBRATION_WINDOWS,
    ENVIRONMENTS,
    HELDOUT_SOURCES,
    HELDOUT_WINDOWS,
    MAX_VALID_RECORDS,
    MODEL_ID,
    MODEL_REVISION,
    SEED,
    WINDOW_TOKENS,
    DatasetSource,
    Phase,
    build_phase_manifest,
    collection_contract,
    expected_tokenizer_metadata,
    render_record,
)
from .dataset_viewer import (
    CACHE_SCHEMA_VERSION,
    VIEWER_PAGE_SIZE,
    _cache_source_directory,
    _derived_seed,
    _page_request,
    _source_identity,
    _validated_page_rows,
)
from .provenance import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    tokenizer_metadata,
)

_SOURCE_METHOD = (
    "Hub-SHA-verified Dataset Viewer /rows; deterministic "
    "100-row page shuffle; content-addressed local cache"
)
_TOKENIZER_FILES = tuple(sorted(expected_tokenizer_metadata()["files"]))


@dataclass(frozen=True, slots=True)
class _CachedPage:
    source_key: str
    page_index: int
    relative_path: str
    response_sha256: str
    total_rows: int
    rows: tuple[tuple[int, Mapping[str, Any]], ...]


@dataclass(frozen=True, slots=True)
class _SourceReplay:
    rows: tuple[Mapping[str, Any], ...]
    audit: dict[str, Any]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_canonical_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict {label}: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical JSON: {path.name}")
    return value, payload


def _required_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _required_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _safe_cache_path(cache_root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    if "\\" in relative or ":" in relative:
        raise ValueError(f"{label} must use a safe POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} must use a safe POSIX relative path")
    if pure.as_posix() != relative:
        raise ValueError(f"{label} is not a canonical POSIX path")
    path = cache_root.joinpath(*pure.parts)
    if not path.is_file():
        raise ValueError(f"{label} does not exist in the adjacent cache")
    resolved_root = cache_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the adjacent cache") from exc
    return resolved_path


def _expected_sources(phase: Phase) -> Mapping[str, tuple[DatasetSource, ...]]:
    return CALIBRATION_SOURCES if phase == "calibration" else HELDOUT_SOURCES


def _load_cached_page(
    *,
    cache_root: Path,
    source: DatasetSource,
    descriptor: Mapping[str, Any],
    expected_page_index: int,
    expected_total_rows: int | None,
    page_cache: dict[tuple[str, int], _CachedPage],
    label: str,
) -> _CachedPage:
    if set(descriptor) not in (
        {"cache_file", "page_index", "response_sha256"},
        {"accepted_valid_records", "cache_file", "page_index", "response_sha256"},
    ):
        raise ValueError(f"{label} fields are invalid")
    page_index = _required_int(descriptor.get("page_index"), label=f"{label}.page_index")
    if page_index != expected_page_index:
        raise ValueError(f"{label} page index disagrees with the seeded page order")
    response_sha256 = _required_sha256(
        descriptor.get("response_sha256"),
        label=f"{label}.response_sha256",
    )
    expected_name = f"page-{page_index:08d}-{response_sha256}.json"
    expected_path = _cache_source_directory(cache_root, source) / expected_name
    path = _safe_cache_path(
        cache_root,
        descriptor.get("cache_file"),
        label=f"{label}.cache_file",
    )
    if path != expected_path.resolve():
        raise ValueError(f"{label} cache path does not match the frozen source identity")

    candidates = sorted(expected_path.parent.glob(f"page-{page_index:08d}-*.json"))
    if [candidate.resolve() for candidate in candidates] != [path]:
        raise ValueError(f"{label} has conflicting cached contents for one source page")

    cache_key = (source.key, page_index)
    cached = page_cache.get(cache_key)
    if cached is not None:
        if (
            cached.relative_path != descriptor["cache_file"]
            or cached.response_sha256 != response_sha256
        ):
            raise ValueError(f"{label} disagrees with the previously verified page")
        return cached

    envelope, _ = _load_canonical_object(path, label="Dataset Viewer cache envelope")
    if set(envelope) != {
        "cache_schema_version",
        "request",
        "response",
        "response_sha256",
    }:
        raise ValueError(f"{label} cache envelope fields are invalid")
    if envelope["cache_schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError(f"{label} cache schema version drift")
    if envelope["request"] != _page_request(source, page_index):
        raise ValueError(f"{label} cache request identity mismatch")
    response = envelope["response"]
    if not isinstance(response, Mapping):
        raise ValueError(f"{label} cache response must be an object")
    actual_response_sha256 = sha256_bytes(canonical_json_bytes(response))
    if envelope["response_sha256"] != response_sha256 or actual_response_sha256 != response_sha256:
        raise ValueError(f"{label} cache response hash mismatch")
    total_rows = response.get("num_rows_total")
    total_rows = _required_int(total_rows, label=f"{label}.num_rows_total", minimum=1)
    if expected_total_rows is not None and total_rows != expected_total_rows:
        raise ValueError(f"{label} total row count drift")
    rows = tuple(
        _validated_page_rows(
            response,
            page_index=page_index,
            total_rows=total_rows,
        )
    )
    relative_path = path.relative_to(cache_root.resolve()).as_posix()
    cached = _CachedPage(
        source_key=source.key,
        page_index=page_index,
        relative_path=relative_path,
        response_sha256=response_sha256,
        total_rows=total_rows,
        rows=rows,
    )
    page_cache[cache_key] = cached
    return cached


def _replay_source(
    *,
    cache_root: Path,
    source: DatasetSource,
    environment: str,
    audit: Mapping[str, Any],
    page_cache: dict[tuple[str, int], _CachedPage],
) -> _SourceReplay:
    expected_fields = {
        "accepted_valid_records",
        "backend",
        "cache_schema_version",
        "derived_page_seed",
        "max_valid_records",
        "metadata_page",
        "page_count",
        "page_order_sha256",
        "page_size",
        "selection_pages",
        "source",
        "total_rows",
        "verified_current_sha",
    }
    if set(audit) != expected_fields:
        raise ValueError(f"source audit fields drift for {source.key}")
    if audit["backend"] != "hugging-face-dataset-viewer/rows":
        raise ValueError(f"source audit backend drift for {source.key}")
    if (
        type(audit["cache_schema_version"]) is not int
        or audit["cache_schema_version"] != CACHE_SCHEMA_VERSION
    ):
        raise ValueError(f"source audit cache schema drift for {source.key}")
    if (
        type(audit["max_valid_records"]) is not int
        or audit["max_valid_records"] != MAX_VALID_RECORDS
    ):
        raise ValueError(f"source audit record limit drift for {source.key}")
    if type(audit["page_size"]) is not int or audit["page_size"] != VIEWER_PAGE_SIZE:
        raise ValueError(f"source audit page size drift for {source.key}")
    if audit["source"] != _source_identity(source):
        raise ValueError(f"source audit identity drift for {source.key}")
    if audit["verified_current_sha"] != source.revision:
        raise ValueError(f"source audit revision drift for {source.key}")

    metadata_page = audit["metadata_page"]
    if not isinstance(metadata_page, Mapping):
        raise ValueError(f"source audit metadata page is invalid for {source.key}")
    first = _load_cached_page(
        cache_root=cache_root,
        source=source,
        descriptor=metadata_page,
        expected_page_index=0,
        expected_total_rows=None,
        page_cache=page_cache,
        label=f"{source.key}.metadata_page",
    )
    total_rows = _required_int(audit["total_rows"], label=f"{source.key}.total_rows", minimum=1)
    if first.total_rows != total_rows:
        raise ValueError(f"source audit metadata total drift for {source.key}")
    page_count = (total_rows + VIEWER_PAGE_SIZE - 1) // VIEWER_PAGE_SIZE
    if type(audit["page_count"]) is not int or audit["page_count"] != page_count:
        raise ValueError(f"source audit page count drift for {source.key}")

    derived_seed = _derived_seed(source, environment=environment, seed=SEED)
    if type(audit["derived_page_seed"]) is not int or audit["derived_page_seed"] != derived_seed:
        raise ValueError(f"source audit seed drift for {source.key}")
    page_order = list(range(page_count))
    random.Random(derived_seed).shuffle(page_order)
    page_order_sha256 = sha256_bytes(canonical_json_bytes(page_order))
    if audit["page_order_sha256"] != page_order_sha256:
        raise ValueError(f"source audit page-order hash drift for {source.key}")

    raw_selection_pages = audit["selection_pages"]
    if (
        not isinstance(raw_selection_pages, Sequence)
        or isinstance(raw_selection_pages, (str, bytes, bytearray))
        or not raw_selection_pages
        or len(raw_selection_pages) > page_count
    ):
        raise ValueError(f"source audit selection pages are invalid for {source.key}")

    accepted_rows: list[Mapping[str, Any]] = []
    replayed_pages: list[dict[str, Any]] = []
    for position, raw_descriptor in enumerate(raw_selection_pages):
        if not isinstance(raw_descriptor, Mapping):
            raise ValueError(f"source audit selection page is invalid for {source.key}")
        page = _load_cached_page(
            cache_root=cache_root,
            source=source,
            descriptor=raw_descriptor,
            expected_page_index=page_order[position],
            expected_total_rows=total_rows,
            page_cache=page_cache,
            label=f"{source.key}.selection_pages[{position}]",
        )
        before = len(accepted_rows)
        for source_index, raw_row in page.rows:
            enriched = dict(raw_row)
            enriched["__cliffquant_source_index__"] = source_index
            if not render_record(environment, enriched, source, source_index):
                continue
            accepted_rows.append(enriched)
            if len(accepted_rows) == MAX_VALID_RECORDS:
                break
        page_accepted = len(accepted_rows) - before
        recorded_page_accepted = _required_int(
            raw_descriptor.get("accepted_valid_records"),
            label=f"{source.key}.selection_pages[{position}].accepted_valid_records",
        )
        if recorded_page_accepted != page_accepted:
            raise ValueError(f"source audit accepted-row count drift for {source.key}")
        replayed_pages.append(
            {
                "accepted_valid_records": page_accepted,
                "cache_file": page.relative_path,
                "page_index": page.page_index,
                "response_sha256": page.response_sha256,
            }
        )
        if len(accepted_rows) == MAX_VALID_RECORDS:
            if position != len(raw_selection_pages) - 1:
                raise ValueError(
                    f"source audit continued after reaching its limit for {source.key}"
                )
            break

    if len(accepted_rows) < MAX_VALID_RECORDS and len(replayed_pages) != page_count:
        raise ValueError(f"source audit stopped before exhausting {source.key}")
    recorded_total_accepted = _required_int(
        audit["accepted_valid_records"],
        label=f"{source.key}.accepted_valid_records",
        minimum=1,
    )
    if recorded_total_accepted != len(accepted_rows):
        raise ValueError(f"source audit total accepted-row drift for {source.key}")

    replayed_audit = {
        "accepted_valid_records": len(accepted_rows),
        "backend": "hugging-face-dataset-viewer/rows",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "derived_page_seed": derived_seed,
        "max_valid_records": MAX_VALID_RECORDS,
        "metadata_page": {
            "cache_file": first.relative_path,
            "page_index": 0,
            "response_sha256": first.response_sha256,
        },
        "page_count": page_count,
        "page_order_sha256": page_order_sha256,
        "page_size": VIEWER_PAGE_SIZE,
        "selection_pages": replayed_pages,
        "source": _source_identity(source),
        "total_rows": total_rows,
        "verified_current_sha": source.revision,
    }
    if dict(audit) != replayed_audit:
        raise ValueError(f"source audit replay mismatch for {source.key}")
    return _SourceReplay(rows=tuple(accepted_rows), audit=replayed_audit)


def _replay_sources(
    manifest: Mapping[str, Any],
    *,
    phase: Phase,
    root: Path,
) -> tuple[dict[str, dict[str, tuple[Mapping[str, Any], ...]]], dict[str, Any]]:
    source_loading = manifest.get("source_loading")
    if not isinstance(source_loading, Mapping) or set(source_loading) != {
        "cache_directory",
        "contract",
        "max_records_per_split",
        "method",
        "page_size",
        "sources",
    }:
        raise ValueError("source_loading fields drift")
    if source_loading["cache_directory"] != "dataset-viewer-cache":
        raise ValueError("source_loading cache directory drift")
    if source_loading["contract"] != collection_contract():
        raise ValueError("source_loading collection contract drift")
    if source_loading["max_records_per_split"] != MAX_VALID_RECORDS:
        raise ValueError("source_loading record limit drift")
    if source_loading["method"] != _SOURCE_METHOD:
        raise ValueError("source_loading method drift")
    if source_loading["page_size"] != VIEWER_PAGE_SIZE:
        raise ValueError("source_loading page size drift")
    raw_audits = source_loading["sources"]
    if not isinstance(raw_audits, Mapping):
        raise ValueError("source_loading.sources must be an object")

    source_table = _expected_sources(phase)
    expected_keys = {source.key for sources in source_table.values() for source in sources}
    if set(raw_audits) != expected_keys:
        raise ValueError("source_loading audit set does not match every pinned source")

    cache_root = (root / "dataset-viewer-cache").resolve()
    if not cache_root.is_dir():
        raise ValueError("adjacent Dataset Viewer cache is missing")
    rows: dict[str, dict[str, tuple[Mapping[str, Any], ...]]] = {}
    replayed_audits: dict[str, Any] = {}
    for environment in ENVIRONMENTS:
        rows[environment] = {}
        for source in source_table[environment]:
            raw_audit = raw_audits[source.key]
            if not isinstance(raw_audit, Mapping):
                raise ValueError(f"source audit must be an object for {source.key}")
            replay = _replay_source(
                cache_root=cache_root,
                source=source,
                environment=environment,
                audit=raw_audit,
                page_cache={},
            )
            rows[environment][source.key] = replay.rows
            replayed_audits[source.key] = replay.audit
    replayed_loading = {
        "cache_directory": "dataset-viewer-cache",
        "contract": collection_contract(),
        "max_records_per_split": MAX_VALID_RECORDS,
        "method": _SOURCE_METHOD,
        "page_size": VIEWER_PAGE_SIZE,
        "sources": replayed_audits,
    }
    return rows, replayed_loading


def _first_difference(expected: Any, observed: Any, *, path: str = "$") -> str | None:
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        expected_keys = set(expected)
        observed_keys = set(observed)
        if expected_keys != observed_keys:
            return (
                f"{path} keys: expected {sorted(expected_keys)}, observed {sorted(observed_keys)}"
            )
        for key in sorted(expected_keys):
            difference = _first_difference(
                expected[key],
                observed[key],
                path=f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if (
        isinstance(expected, Sequence)
        and isinstance(observed, Sequence)
        and not isinstance(expected, (str, bytes, bytearray))
        and not isinstance(observed, (str, bytes, bytearray))
    ):
        if len(expected) != len(observed):
            return f"{path} length: expected {len(expected)}, observed {len(observed)}"
        for index, (expected_item, observed_item) in enumerate(
            zip(expected, observed, strict=True)
        ):
            difference = _first_difference(
                expected_item,
                observed_item,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if expected != observed:
        return f"{path}: expected {expected!r}, observed {observed!r}"
    return None


def _archive_descriptor(root: Path, manifest: Mapping[str, Any], *, phase: Phase) -> dict[str, Any]:
    raw = manifest.get("token_archive")
    if not isinstance(raw, Mapping) or set(raw) != {"file", "sha256", "size_bytes"}:
        raise ValueError("token archive descriptor fields drift")
    expected_name = f"{phase}.tokens.npz"
    if raw["file"] != expected_name:
        raise ValueError("token archive filename drift")
    path = root / expected_name
    if not path.is_file():
        raise ValueError("token archive is missing")
    descriptor = {
        "file": expected_name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if dict(raw) != descriptor:
        raise ValueError("token archive descriptor does not match its bytes")
    return descriptor


def _verify_sidecar(
    root: Path,
    *,
    phase: Phase,
    manifest_path: Path,
    manifest_payload: bytes,
    archive_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / f"{phase}.sha256.json"
    sidecar, payload = _load_canonical_object(path, label="corpus hash sidecar")
    expected = {
        "manifest": {
            "file": manifest_path.name,
            "sha256": sha256_bytes(manifest_payload),
            "size_bytes": len(manifest_payload),
        },
        "tokens": dict(archive_descriptor),
    }
    if sidecar != expected:
        raise ValueError("corpus hash sidecar does not match the bundle")
    return {
        "file": path.name,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _expected_token_arrays(
    windows: Mapping[str, Sequence[Any]],
) -> dict[str, NDArray[np.generic]]:
    arrays: dict[str, NDArray[np.generic]] = {}
    for environment in ENVIRONMENTS:
        arrays[f"{environment}__input_ids"] = np.asarray(
            [window.input_ids for window in windows[environment]],
            dtype=np.int64,
        )
        arrays[f"{environment}__attention_mask"] = np.asarray(
            [window.attention_mask for window in windows[environment]],
            dtype=np.uint8,
        )
    return arrays


def _verify_token_arrays(
    root: Path,
    *,
    phase: Phase,
    expected: Mapping[str, NDArray[np.generic]],
) -> None:
    path = root / f"{phase}.tokens.npz"
    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot open the token archive") from exc
    with archive_context as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(expected):
            raise ValueError("token archive array set drift")
        for key in sorted(expected):
            observed = np.asarray(archive[key])
            replayed = expected[key]
            if observed.dtype.str != replayed.dtype.str or observed.shape != replayed.shape:
                raise ValueError(f"token archive dtype or shape drift for {key}")
            if not np.array_equal(observed, replayed):
                raise ValueError(f"token archive replay mismatch for {key}")


def verify_corpus_replay(
    directory: str | Path,
    *,
    phase: Phase,
    tokenizer: Any,
) -> dict[str, Any]:
    """Replay one real corpus bundle from its exact cached Viewer responses."""

    if phase not in ("calibration", "heldout"):
        raise ValueError("phase must be calibration or heldout")
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("corpus bundle directory is missing")
    manifest_path = root / f"{phase}.manifest.json"
    manifest, manifest_payload = _load_canonical_object(
        manifest_path,
        label="corpus manifest",
    )
    expected_top_level = {
        "dry_run",
        "environments",
        "model",
        "phase",
        "rendering",
        "runtime",
        "sampling",
        "schema",
        "source_loading",
        "token_archive",
        "tokenizer",
        "window_count_per_environment",
        "window_tokens",
    }
    if set(manifest) != expected_top_level:
        raise ValueError("corpus manifest fields drift")
    if manifest["schema"] != "cliffquant.corpus-manifest.v1":
        raise ValueError("corpus manifest schema drift")
    if manifest["phase"] != phase or manifest["dry_run"] is not False:
        raise ValueError("corpus manifest phase or dry-run state drift")
    if manifest["model"] != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("corpus manifest model identity drift")
    if manifest["tokenizer"] != expected_tokenizer_metadata():
        raise ValueError("corpus manifest tokenizer identity drift")
    if not isinstance(manifest["runtime"], Mapping):
        raise ValueError("corpus manifest runtime provenance is missing")

    expected_count = CALIBRATION_WINDOWS if phase == "calibration" else HELDOUT_WINDOWS
    if (
        manifest["window_count_per_environment"] != expected_count
        or manifest["window_tokens"] != WINDOW_TOKENS
    ):
        raise ValueError("corpus manifest window contract drift")

    archive_descriptor = _archive_descriptor(root, manifest, phase=phase)
    sidecar_descriptor = _verify_sidecar(
        root,
        phase=phase,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        archive_descriptor=archive_descriptor,
    )
    rows, source_loading = _replay_sources(
        manifest,
        phase=phase,
        root=root,
    )
    replayed_manifest, windows = build_phase_manifest(
        phase=phase,
        tokenizer=tokenizer,
        rows=rows,
        tokenizer_info=expected_tokenizer_metadata(),
        runtime_info=manifest["runtime"],
        count=expected_count,
        window_size=WINDOW_TOKENS,
        seed=SEED,
        dry_run=False,
    )
    replayed_manifest["source_loading"] = source_loading
    replayed_manifest["token_archive"] = archive_descriptor
    difference = _first_difference(replayed_manifest, manifest)
    if difference is not None:
        raise ValueError(f"corpus manifest replay mismatch: {difference}")
    if canonical_json_bytes(replayed_manifest) != manifest_payload:
        raise ValueError("corpus manifest replay bytes differ")

    expected_arrays = _expected_token_arrays(windows)
    _verify_token_arrays(root, phase=phase, expected=expected_arrays)
    unique_pages: dict[tuple[str, int], str] = {}
    for source_key, audit in source_loading["sources"].items():
        descriptors = [audit["metadata_page"], *audit["selection_pages"]]
        for descriptor in descriptors:
            identity = (source_key, descriptor["page_index"])
            response_sha256 = descriptor["response_sha256"]
            previous = unique_pages.setdefault(identity, response_sha256)
            if previous != response_sha256:  # pragma: no cover - replay already rejects this
                raise RuntimeError("verified source audit contains conflicting page hashes")
    page_descriptors = [
        {
            "page_index": page_index,
            "response_sha256": response_sha256,
            "source": source_key,
        }
        for (source_key, page_index), response_sha256 in sorted(unique_pages.items())
    ]
    used_record_count = sum(
        len(manifest["environments"][environment]["records"]) for environment in ENVIRONMENTS
    )
    window_count = sum(len(windows[environment]) for environment in ENVIRONMENTS)
    return {
        "accepted_source_rows": sum(
            len(source_rows)
            for environment_rows in rows.values()
            for source_rows in environment_rows.values()
        ),
        "cache": {
            "page_count": len(page_descriptors),
            "pages_sha256": sha256_bytes(canonical_json_bytes(page_descriptors)),
        },
        "manifest": {
            "file": manifest_path.name,
            "sha256": sha256_bytes(manifest_payload),
            "size_bytes": len(manifest_payload),
        },
        "phase": phase,
        "schema": "cliffquant.corpus-replay.v1",
        "sidecar": sidecar_descriptor,
        "token_archive": archive_descriptor,
        "tokenizer_sha256": sha256_bytes(canonical_json_bytes(expected_tokenizer_metadata())),
        "used_rendered_records": used_record_count,
        "window_count": window_count,
    }


def load_pinned_tokenizer() -> Any:
    """Load the exact hash-frozen tokenizer used by Experiment 001."""

    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional CLI dependencies
        raise RuntimeError("install transformers and huggingface-hub for corpus replay") from exc
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(_TOKENIZER_FILES),
        )
    )
    expected = expected_tokenizer_metadata()
    files: dict[str, Path] = {}
    for name, descriptor in expected["files"].items():
        path = snapshot / name
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["size_bytes"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError(f"pinned tokenizer file identity drift: {name}")
        files[name] = path
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    observed = tokenizer_metadata(
        tokenizer,
        model_revision=MODEL_REVISION,
        explicit_files=files,
        logical_name_or_path=MODEL_ID,
    )
    if observed != expected:
        raise ValueError("loaded tokenizer metadata differs from the frozen identity")
    return tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("calibration", "heldout"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = load_pinned_tokenizer()
    result = verify_corpus_replay(
        args.directory,
        phase=args.phase,
        tokenizer=tokenizer,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
