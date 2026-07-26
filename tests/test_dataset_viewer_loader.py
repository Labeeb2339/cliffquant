from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from cliffquant.corpus import CALIBRATION_SOURCES, load_pinned_rows
from cliffquant.dataset_viewer import (
    VIEWER_PAGE_SIZE,
    DatasetRevisionDriftError,
    DatasetViewerCacheError,
    DatasetViewerError,
    _validated_page_rows,
)
from cliffquant.provenance import canonical_json_bytes, sha256_bytes


@dataclass
class FakeHTTP:
    rows: list[dict[str, Any]]
    sha: str
    later_hub_shas: list[str] = field(default_factory=list)
    row_failures_remaining: int = 0
    deny_rows: bool = False
    truncated_indices: set[int] = field(default_factory=set)
    calls: list[tuple[str, float, float]] = field(default_factory=list)

    def __call__(
        self,
        url: str,
        *,
        connect_timeout: float,
        read_timeout: float,
    ) -> bytes:
        self.calls.append((url, connect_timeout, read_timeout))
        parsed = urlsplit(url)
        if parsed.hostname == "huggingface.co":
            sha = self.later_hub_shas.pop(0) if self.later_hub_shas else self.sha
            return canonical_json_bytes({"sha": sha})
        if parsed.hostname != "datasets-server.huggingface.co":
            raise AssertionError(f"unexpected host: {parsed.hostname}")
        if self.deny_rows:
            raise AssertionError("row endpoint should not be called")
        if self.row_failures_remaining:
            self.row_failures_remaining -= 1
            raise TimeoutError("simulated Viewer timeout")

        query = parse_qs(parsed.query)
        assert int(query["length"][0]) == VIEWER_PAGE_SIZE
        offset = int(query["offset"][0])
        selected = self.rows[offset : offset + VIEWER_PAGE_SIZE]
        entries = [
            {
                "row_idx": row_index,
                "row": row,
                "truncated_cells": (["text"] if row_index in self.truncated_indices else []),
            }
            for row_index, row in enumerate(selected, start=offset)
        ]
        return canonical_json_bytes(
            {
                "num_rows_per_page": VIEWER_PAGE_SIZE,
                "num_rows_total": len(self.rows),
                "partial": False,
                "rows": entries,
            }
        )

    @property
    def hub_calls(self) -> list[tuple[str, float, float]]:
        return [call for call in self.calls if urlsplit(call[0]).hostname == "huggingface.co"]

    @property
    def row_calls(self) -> list[tuple[str, float, float]]:
        return [
            call
            for call in self.calls
            if urlsplit(call[0]).hostname == "datasets-server.huggingface.co"
        ]


def _general_rows(count: int) -> list[dict[str, str]]:
    return [{"text": f"valid row {index}"} for index in range(count)]


def test_viewer_loader_is_deterministic_filters_then_caches_and_resumes(
    tmp_path: Path,
) -> None:
    source = CALIBRATION_SOURCES["general"][0]
    raw_rows = _general_rows(225)
    for index in range(0, len(raw_rows), 2):
        raw_rows[index] = {"text": " "}

    first_http = FakeHTTP(raw_rows, source.revision)
    first_audit: dict[str, Any] = {}
    first = load_pinned_rows(
        source,
        environment="general",
        max_records=20,
        cache_dir=tmp_path,
        audit=first_audit,
        connect_timeout=1.25,
        read_timeout=4.5,
        max_retries=0,
        http_get=first_http,
        progress=StringIO(),
    )
    assert len(first) == 20
    assert all(row["text"].strip() for row in first)
    assert all(row["__cliffquant_source_index__"] % 2 == 1 for row in first)
    assert first_audit["verified_current_sha"] == source.revision
    assert first_audit["total_rows"] == 225
    assert first_audit["page_size"] == 100
    assert first_audit["accepted_valid_records"] == 20
    assert first_audit["selection_pages"]
    assert all(call[1:] == (1.25, 4.5) for call in first_http.calls)

    cache_files = sorted((tmp_path / "pages").rglob("page-*.json"))
    assert cache_files
    for path in cache_files:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        content_hash = sha256_bytes(canonical_json_bytes(envelope["response"]))
        assert envelope["response_sha256"] == content_hash
        assert path.name.endswith(f"-{content_hash}.json")

    resumed_http = FakeHTTP(raw_rows, source.revision, deny_rows=True)
    resumed_audit: dict[str, Any] = {}
    resumed = load_pinned_rows(
        source,
        environment="general",
        max_records=20,
        cache_dir=tmp_path,
        audit=resumed_audit,
        connect_timeout=1.25,
        read_timeout=4.5,
        max_retries=0,
        http_get=resumed_http,
        progress=StringIO(),
    )
    assert resumed == first
    assert resumed_audit == first_audit
    assert len(resumed_http.hub_calls) == 2
    assert resumed_http.row_calls == []


def test_viewer_loader_retries_with_bounds_and_emits_progress(tmp_path: Path) -> None:
    source = CALIBRATION_SOURCES["general"][0]
    transport = FakeHTTP(_general_rows(12), source.revision, row_failures_remaining=1)
    progress = StringIO()
    rows = load_pinned_rows(
        source,
        environment="general",
        max_records=3,
        cache_dir=tmp_path,
        connect_timeout=2.0,
        read_timeout=7.0,
        max_retries=1,
        retry_backoff=0,
        http_get=transport,
        progress=progress,
    )
    assert len(rows) == 3
    assert len(transport.row_calls) == 2
    assert all(call[1:] == (2.0, 7.0) for call in transport.calls)
    assert "retry 1/1" in progress.getvalue()
    assert "valid=3/3" in progress.getvalue()


def test_viewer_loader_stops_after_the_bounded_attempt_count(tmp_path: Path) -> None:
    source = CALIBRATION_SOURCES["general"][0]
    transport = FakeHTTP(_general_rows(12), source.revision, row_failures_remaining=10)
    with pytest.raises(DatasetViewerError, match="failed after 3 attempts"):
        load_pinned_rows(
            source,
            environment="general",
            max_records=3,
            cache_dir=tmp_path,
            max_retries=2,
            retry_backoff=0,
            http_get=transport,
            progress=StringIO(),
        )
    assert len(transport.row_calls) == 3


def test_revision_drift_refuses_cached_rows_before_viewer_access(tmp_path: Path) -> None:
    source = CALIBRATION_SOURCES["general"][0]
    initial = FakeHTTP(_general_rows(8), source.revision)
    load_pinned_rows(
        source,
        environment="general",
        max_records=2,
        cache_dir=tmp_path,
        max_retries=0,
        http_get=initial,
        progress=StringIO(),
    )

    drifted = FakeHTTP(_general_rows(8), "0" * 40, deny_rows=True)
    with pytest.raises(DatasetRevisionDriftError, match="dataset revision drift"):
        load_pinned_rows(
            source,
            environment="general",
            max_records=2,
            cache_dir=tmp_path,
            max_retries=0,
            http_get=drifted,
            progress=StringIO(),
        )
    assert len(drifted.hub_calls) == 1
    assert drifted.row_calls == []


def test_revision_drift_after_page_collection_discards_the_run(tmp_path: Path) -> None:
    source = CALIBRATION_SOURCES["general"][0]
    transport = FakeHTTP(
        _general_rows(8),
        source.revision,
        later_hub_shas=[source.revision, "0" * 40],
    )
    with pytest.raises(DatasetRevisionDriftError, match="dataset revision drift"):
        load_pinned_rows(
            source,
            environment="general",
            max_records=2,
            cache_dir=tmp_path,
            max_retries=0,
            http_get=transport,
            progress=StringIO(),
        )
    assert len(transport.hub_calls) == 2
    assert len(transport.row_calls) == 1


def test_viewer_loader_rejects_truncated_cells(tmp_path: Path) -> None:
    source = CALIBRATION_SOURCES["general"][0]
    transport = FakeHTTP(
        _general_rows(8),
        source.revision,
        truncated_indices={0},
    )
    with pytest.raises(DatasetViewerError, match="contains truncated cells"):
        load_pinned_rows(
            source,
            environment="general",
            max_records=2,
            cache_dir=tmp_path,
            max_retries=0,
            http_get=transport,
            progress=StringIO(),
        )


def test_page_must_explicitly_declare_that_it_is_complete() -> None:
    with pytest.raises(DatasetViewerError, match="partial=false"):
        _validated_page_rows(
            {
                "num_rows_total": 1,
                "rows": [{"row_idx": 0, "row": {"text": "complete"}, "truncated_cells": []}],
            },
            page_index=0,
            total_rows=1,
        )


def test_viewer_loader_rejects_tampered_cache(tmp_path: Path) -> None:
    source = CALIBRATION_SOURCES["general"][0]
    initial = FakeHTTP(_general_rows(8), source.revision)
    load_pinned_rows(
        source,
        environment="general",
        max_records=2,
        cache_dir=tmp_path,
        max_retries=0,
        http_get=initial,
        progress=StringIO(),
    )
    cache_file = next((tmp_path / "pages").rglob("page-*.json"))
    envelope = json.loads(cache_file.read_text(encoding="utf-8"))
    envelope["response"]["rows"][0]["row"]["text"] = "tampered"
    cache_file.write_text(json.dumps(envelope), encoding="utf-8")

    cached = FakeHTTP(_general_rows(8), source.revision, deny_rows=True)
    with pytest.raises(DatasetViewerCacheError, match="cache response hash mismatch"):
        load_pinned_rows(
            source,
            environment="general",
            max_records=2,
            cache_dir=tmp_path,
            max_retries=0,
            http_get=cached,
            progress=StringIO(),
        )
