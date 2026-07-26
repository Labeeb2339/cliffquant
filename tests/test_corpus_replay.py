from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from cliffquant.corpus import (
    CALIBRATION_SOURCES,
    ENVIRONMENTS,
    MAX_VALID_RECORDS,
    SEED,
    build_phase_manifest,
    collection_contract,
    expected_tokenizer_metadata,
    render_record,
    save_phase_bundle,
)
from cliffquant.corpus_replay import verify_corpus_replay
from cliffquant.dataset_viewer import (
    CACHE_SCHEMA_VERSION,
    VIEWER_PAGE_SIZE,
    _derived_seed,
    _source_identity,
    _write_cached_page,
)
from cliffquant.provenance import canonical_json_bytes, sha256_bytes


class _ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


def _row(environment: str, index: int) -> dict:
    repeated = f"{environment}-{index}-" * 35
    if environment == "general":
        return {"id": f"general-{index}", "text": repeated}
    if environment == "code":
        return {
            "id": f"code-{index}",
            "solution": f"def f():\n    return {index}\n{repeated}",
            "task": f"Return the number {index}. {repeated}",
            "test_list": [f"assert f() == {index}", repeated],
        }
    if environment == "math":
        return {
            "answer": f"The answer is {index}. {repeated}",
            "id": f"math-{index}",
            "question": f"What is {index} plus zero? {repeated}",
        }
    language = ("ar", "es", "hi", "zh")[index % 4]
    return {
        "id": f"multilingual-{index}",
        "hypothesis": f"{language} hypothesis {repeated}",
        "language": language,
        "premise": f"{language} premise {repeated}",
    }


def _source_audit(
    *,
    cache_root: Path,
    environment: str,
    source,
    rows: list[Mapping],
) -> tuple[dict, list[Mapping]]:
    response = {
        "num_rows_total": len(rows),
        "partial": False,
        "rows": [
            {
                "row": dict(row),
                "row_idx": index,
                "truncated_cells": [],
            }
            for index, row in enumerate(rows)
        ],
    }
    response_hash, cache_path = _write_cached_page(
        cache_root,
        source=source,
        page_index=0,
        response=response,
    )
    enriched: list[Mapping] = []
    for index, row in enumerate(rows):
        value = dict(row)
        value["__cliffquant_source_index__"] = index
        if render_record(environment, value, source, index):
            enriched.append(value)
    assert len(enriched) < MAX_VALID_RECORDS
    relative = cache_path.relative_to(cache_root).as_posix()
    derived_seed = _derived_seed(source, environment=environment, seed=SEED)
    page_order = [0]
    page = {
        "accepted_valid_records": len(enriched),
        "cache_file": relative,
        "page_index": 0,
        "response_sha256": response_hash,
    }
    audit = {
        "accepted_valid_records": len(enriched),
        "backend": "hugging-face-dataset-viewer/rows",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "derived_page_seed": derived_seed,
        "max_valid_records": MAX_VALID_RECORDS,
        "metadata_page": {
            "cache_file": relative,
            "page_index": 0,
            "response_sha256": response_hash,
        },
        "page_count": 1,
        "page_order_sha256": sha256_bytes(canonical_json_bytes(page_order)),
        "page_size": VIEWER_PAGE_SIZE,
        "selection_pages": [page],
        "source": _source_identity(source),
        "total_rows": len(rows),
        "verified_current_sha": source.revision,
    }
    return audit, enriched


def _build_bundle(root: Path) -> _ByteTokenizer:
    tokenizer = _ByteTokenizer()
    cache_root = root / "dataset-viewer-cache"
    source_audits: dict[str, dict] = {}
    rows: dict[str, dict[str, list[Mapping]]] = {}
    for environment in ENVIRONMENTS:
        rows[environment] = {}
        for source in CALIBRATION_SOURCES[environment]:
            raw_rows = [_row(environment, index) for index in range(VIEWER_PAGE_SIZE)]
            audit, enriched = _source_audit(
                cache_root=cache_root,
                environment=environment,
                source=source,
                rows=raw_rows,
            )
            source_audits[source.key] = audit
            rows[environment][source.key] = enriched

    manifest, windows = build_phase_manifest(
        phase="calibration",
        tokenizer=tokenizer,
        rows=rows,
        tokenizer_info=expected_tokenizer_metadata(),
        runtime_info={"fixture": True},
        dry_run=False,
    )
    manifest["source_loading"] = {
        "cache_directory": "dataset-viewer-cache",
        "contract": collection_contract(),
        "max_records_per_split": MAX_VALID_RECORDS,
        "method": (
            "Hub-SHA-verified Dataset Viewer /rows; deterministic "
            "100-row page shuffle; content-addressed local cache"
        ),
        "page_size": VIEWER_PAGE_SIZE,
        "sources": source_audits,
    }
    save_phase_bundle(
        root,
        phase="calibration",
        manifest=manifest,
        windows=windows,
    )
    return tokenizer


def _rewrite_manifest(root: Path, manifest: dict) -> None:
    manifest_path = root / "calibration.manifest.json"
    payload = canonical_json_bytes(manifest)
    manifest_path.write_bytes(payload)
    sidecar_path = root / "calibration.sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["manifest"] = {
        "file": manifest_path.name,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))


def test_replay_verifier_reconstructs_exact_bundle(tmp_path: Path) -> None:
    tokenizer = _build_bundle(tmp_path)
    result = verify_corpus_replay(
        tmp_path,
        phase="calibration",
        tokenizer=tokenizer,
    )
    assert result["schema"] == "cliffquant.corpus-replay.v1"
    assert result["phase"] == "calibration"
    assert result["accepted_source_rows"] == 4 * VIEWER_PAGE_SIZE
    assert result["cache"]["page_count"] == 4
    assert result["window_count"] == 4 * 32
    assert result["used_rendered_records"] > 0


def test_replay_rejects_self_consistent_record_relabel(tmp_path: Path) -> None:
    tokenizer = _build_bundle(tmp_path)
    manifest_path = tmp_path / "calibration.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environments"]["general"]["records"][0]["source_index"] += 1
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="corpus manifest replay mismatch"):
        verify_corpus_replay(
            tmp_path,
            phase="calibration",
            tokenizer=tokenizer,
        )


def test_replay_rejects_rehashed_cache_content_drift(tmp_path: Path) -> None:
    tokenizer = _build_bundle(tmp_path)
    manifest_path = tmp_path / "calibration.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = CALIBRATION_SOURCES["general"][0]
    audit = manifest["source_loading"]["sources"][source.key]
    old_relative = audit["metadata_page"]["cache_file"]
    old_path = tmp_path / "dataset-viewer-cache" / Path(old_relative)
    envelope = json.loads(old_path.read_text(encoding="utf-8"))
    used_index = manifest["environments"]["general"]["records"][0]["source_index"]
    envelope["response"]["rows"][used_index]["row"]["text"] += " changed"
    response_hash = sha256_bytes(canonical_json_bytes(envelope["response"]))
    envelope["response_sha256"] = response_hash
    new_path = old_path.with_name(f"page-00000000-{response_hash}.json")
    new_path.write_bytes(canonical_json_bytes(envelope))
    old_path.unlink()
    new_relative = new_path.relative_to(tmp_path / "dataset-viewer-cache").as_posix()
    audit["metadata_page"]["cache_file"] = new_relative
    audit["metadata_page"]["response_sha256"] = response_hash
    audit["selection_pages"][0]["cache_file"] = new_relative
    audit["selection_pages"][0]["response_sha256"] = response_hash
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="corpus manifest replay mismatch"):
        verify_corpus_replay(
            tmp_path,
            phase="calibration",
            tokenizer=tokenizer,
        )


def test_replay_rejects_noncanonical_cache_envelope(tmp_path: Path) -> None:
    tokenizer = _build_bundle(tmp_path)
    manifest = json.loads((tmp_path / "calibration.manifest.json").read_text(encoding="utf-8"))
    source = CALIBRATION_SOURCES["general"][0]
    relative = manifest["source_loading"]["sources"][source.key]["metadata_page"]["cache_file"]
    cache_path = tmp_path / "dataset-viewer-cache" / Path(relative)
    cache_path.write_bytes(cache_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="not canonical JSON"):
        verify_corpus_replay(
            tmp_path,
            phase="calibration",
            tokenizer=tokenizer,
        )
