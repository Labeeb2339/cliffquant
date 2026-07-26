from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from cliffquant.activation_stats import (
    ActivationStatistics,
    EnvironmentDiagonal,
    save_activation_statistics,
)
from cliffquant.corpus import (
    CALIBRATION_SOURCES,
    CALIBRATION_WINDOWS,
    HELDOUT_SOURCES,
    HELDOUT_WINDOWS,
    MAX_VALID_RECORDS,
    MODEL_ID,
    MODEL_REVISION,
    SEED,
    WINDOW_TOKENS,
    collection_contract,
    expected_tokenizer_metadata,
)
from cliffquant.dataset_viewer import CACHE_SCHEMA_VERSION, VIEWER_PAGE_SIZE
from cliffquant.experiment001_io import (
    EXPERIMENT_001_ENVIRONMENTS,
    ModuleInventoryEntry,
    deterministic_group_coordinates,
    load_diagonal_artifact,
    stratified_module_quotas,
    validate_diagonal_pair,
)
from cliffquant.model_snapshot import MODEL_SNAPSHOT_SCHEMA
from cliffquant.provenance import (
    canonical_json_bytes,
    runtime_metadata,
    sha256_bytes,
    sha256_file,
)
from cliffquant.qwen35_modules import QWEN35_GPTQMODEL_TREE_REVISION, QuantizedModule


def _fixture_diagonal_runtime() -> dict:
    runtime = runtime_metadata(device="cpu")
    runtime.update(
        {
            "accelerator": {
                "available": True,
                "capability": [12, 0],
                "cublas_workspace_config": ":4096:8",
                "cuda_runtime": "12.8",
                "cudnn_allow_tf32": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cudnn_version": 91_000,
                "deterministic_algorithms": True,
                "deterministic_algorithms_warn_only": False,
                "device_index": 0,
                "float32_matmul_precision": "highest",
                "matmul_allow_tf32": False,
                "name": "fixture GPU",
                "total_memory_bytes": 8_000_000_000,
            },
            "device": "cuda",
            "model_parameter_dtypes": ["torch.bfloat16"],
            "seed": SEED,
        }
    )
    return runtime


def _fixture_model_source() -> dict:
    payload = {
        "config": {
            "file": "config.json",
            "sha256": "a" * 64,
            "size_bytes": 100,
        },
        "id": MODEL_ID,
        "index": {
            "file": "model.safetensors.index.json",
            "sha256": "b" * 64,
            "size_bytes": 200,
        },
        "revision": MODEL_REVISION,
        "schema": MODEL_SNAPSHOT_SCHEMA,
        "shards": [
            {
                "file": "model-00001-of-00001.safetensors",
                "sha256": "c" * 64,
                "size_bytes": 300,
            }
        ],
        "tensor_count": 1,
    }
    return {
        **payload,
        "fingerprint_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _small_inventory() -> tuple[ModuleInventoryEntry, ...]:
    return (
        ModuleInventoryEntry(
            name="model.language_model.layers.0.self_attn.q_proj",
            layer=0,
            branch="self_attn.q_proj",
            in_features=128,
            out_features=2,
        ),
        ModuleInventoryEntry(
            name="model.language_model.layers.0.self_attn.k_proj",
            layer=0,
            branch="self_attn.k_proj",
            in_features=256,
            out_features=2,
        ),
        ModuleInventoryEntry(
            name="model.language_model.layers.0.mlp.gate_proj",
            layer=0,
            branch="mlp.gate_proj",
            in_features=128,
            out_features=4,
        ),
    )


def test_stratified_sample_is_exact_deterministic_and_covers_every_module() -> None:
    modules = _small_inventory()
    quotas = stratified_module_quotas(modules, sample_size=7)
    assert sum(quotas.values()) == 7
    assert all(quota >= 1 for quota in quotas.values())

    first = deterministic_group_coordinates(modules, sample_size=7, seed=2339)
    second = deterministic_group_coordinates(modules, sample_size=7, seed=2339)
    assert first == second
    assert len(first) == 7
    assert len({coordinate.group_id for coordinate in first}) == 7
    assert {coordinate.module for coordinate in first} == {module.name for module in modules}


def _source_identity(source) -> dict[str, str]:
    return {
        "config": source.subset,
        "dataset": source.dataset,
        "revision": source.revision,
        "split": source.split,
    }


def _source_audit(source, environment: str) -> dict:
    identity = _source_identity(source)
    derived_seed = int(
        sha256_bytes(
            canonical_json_bytes(
                {
                    "environment": environment,
                    "seed": SEED,
                    "source": identity,
                }
            )
        )[:16],
        16,
    )
    page_order = [0]
    random.Random(derived_seed).shuffle(page_order)
    response_hash = sha256_bytes(canonical_json_bytes({"source": identity}))
    cache_file = f"pages/{response_hash}/page-00000000-{response_hash}.json"
    page = {
        "accepted_valid_records": VIEWER_PAGE_SIZE,
        "cache_file": cache_file,
        "page_index": 0,
        "response_sha256": response_hash,
    }
    return {
        "accepted_valid_records": VIEWER_PAGE_SIZE,
        "backend": "hugging-face-dataset-viewer/rows",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "derived_page_seed": derived_seed,
        "max_valid_records": MAX_VALID_RECORDS,
        "metadata_page": {
            "cache_file": cache_file,
            "page_index": 0,
            "response_sha256": response_hash,
        },
        "page_count": 1,
        "page_order_sha256": sha256_bytes(canonical_json_bytes(page_order)),
        "page_size": VIEWER_PAGE_SIZE,
        "selection_pages": [page],
        "source": identity,
        "total_rows": VIEWER_PAGE_SIZE,
        "verified_current_sha": source.revision,
    }


def _write_corpus_artifact(
    output: Path,
    *,
    phase: str,
    wrong_dataset_revision: bool = False,
    shared_rendered_hash: str | None = None,
) -> Path:
    sources = CALIBRATION_SOURCES if phase == "calibration" else HELDOUT_SOURCES
    count = CALIBRATION_WINDOWS if phase == "calibration" else HELDOUT_WINDOWS
    phase_offset = 0 if phase == "calibration" else 1_000_000
    arrays: dict[str, np.ndarray] = {}
    environments: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    for environment_index, environment in enumerate(EXPERIMENT_001_ENVIRONMENTS):
        environment_sources = sources[environment]
        records: list[dict] = []
        for source_index, source in enumerate(environment_sources):
            rendered_hash = sha256_bytes(
                canonical_json_bytes(
                    {
                        "environment": environment,
                        "phase": phase,
                        "source": source.key,
                    }
                )
            )
            if shared_rendered_hash is not None and environment == "general":
                rendered_hash = shared_rendered_hash
            records.append(
                {
                    "dataset": source.dataset,
                    "rendered_sha256": rendered_hash,
                    "revision": source.revision,
                    "row_id": f"{phase}-{environment}-{source_index}",
                    "source_index": source_index,
                    "split": source.split,
                    "subset": source.subset,
                    "token_count": WINDOW_TOKENS,
                    "variant": None,
                }
            )
            audits[source.key] = _source_audit(source, environment)

        ids = np.arange(count * WINDOW_TOKENS, dtype=np.int64).reshape(
            count,
            WINDOW_TOKENS,
        )
        ids = ids + phase_offset + environment_index * 100_000
        mask = np.ones_like(ids, dtype=np.uint8)
        arrays[f"{environment}__input_ids"] = ids
        arrays[f"{environment}__attention_mask"] = mask
        windows: list[dict] = []
        for window_index in range(count):
            record = records[window_index % len(records)]
            windows.append(
                {
                    "index": window_index,
                    "language": (
                        ("ar", "es", "hi", "zh")[window_index % 4]
                        if environment == "multilingual"
                        else None
                    ),
                    "segments": [
                        {
                            "record_token_end": WINDOW_TOKENS,
                            "record_token_start": 0,
                            "rendered_sha256": record["rendered_sha256"],
                            "row_id": record["row_id"],
                            "window_token_end": WINDOW_TOKENS,
                            "window_token_start": 0,
                        }
                    ],
                    "token_count": WINDOW_TOKENS,
                    "token_sha256": sha256_bytes(canonical_json_bytes(ids[window_index].tolist())),
                }
            )
        dataset_table = [
            {
                "dataset": source.dataset,
                "revision": source.revision,
                "split": source.split,
                "subset": source.subset,
            }
            for source in environment_sources
        ]
        if wrong_dataset_revision and environment == "general":
            dataset_table[0]["revision"] = "0" * 40
        environments[environment] = {
            "datasets": dataset_table,
            "records": records,
            "windows": windows,
        }

    archive_path = output / f"{phase}.tokens.npz"
    np.savez_compressed(archive_path, **arrays)
    manifest = {
        "dry_run": False,
        "environments": environments,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "phase": phase,
        "rendering": {
            environment: f"frozen-{environment}" for environment in EXPERIMENT_001_ENVIRONMENTS
        },
        "runtime": runtime_metadata(device="cpu"),
        "sampling": {
            "deduplication": "first seeded-rank occurrence of each rendered SHA256",
            "non_overlapping": True,
            "ordering": "SHA256(seed, environment, phase, source, row ID, rendered SHA256)",
            "seed": SEED,
            "split_boundary_policy": "discard incomplete tail before the next split",
        },
        "schema": "cliffquant.corpus-manifest.v1",
        "source_loading": {
            "cache_directory": "dataset-viewer-cache",
            "contract": collection_contract(),
            "max_records_per_split": MAX_VALID_RECORDS,
            "method": (
                "Hub-SHA-verified Dataset Viewer /rows; deterministic "
                "100-row page shuffle; content-addressed local cache"
            ),
            "page_size": VIEWER_PAGE_SIZE,
            "sources": audits,
        },
        "token_archive": {
            "file": archive_path.name,
            "sha256": sha256_file(archive_path),
            "size_bytes": archive_path.stat().st_size,
        },
        "tokenizer": expected_tokenizer_metadata(),
        "window_count_per_environment": count,
        "window_tokens": WINDOW_TOKENS,
    }
    manifest_path = output / f"{phase}.manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    sidecar = {
        "manifest": {
            "file": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "tokens": manifest["token_archive"],
    }
    (output / f"{phase}.sha256.json").write_bytes(canonical_json_bytes(sidecar))
    return manifest_path


def _write_small_diagonal_artifact(
    output: Path,
    *,
    phase: str = "calibration",
    wrong_dataset_revision: bool = False,
    shared_rendered_hash: str | None = None,
) -> Path:
    manifest_path = _write_corpus_artifact(
        output,
        phase=phase,
        wrong_dataset_revision=wrong_dataset_revision,
        shared_rendered_hash=shared_rendered_hash,
    )
    module = QuantizedModule(
        name="model.language_model.layers.0.self_attn.q_proj",
        layer=0,
        branch="self_attn.q_proj",
        in_features=128,
        out_features=1,
        module=None,
    )
    raw = np.linspace(0.5, 1.5, 128, dtype=np.float64)
    normalization = float(np.mean(raw, dtype=np.float64))
    normalized = raw / normalization
    token_rows = (
        CALIBRATION_WINDOWS if phase == "calibration" else HELDOUT_WINDOWS
    ) * WINDOW_TOKENS
    values = {
        (environment, module.name): EnvironmentDiagonal(
            mean_input_square=raw,
            normalized=normalized,
            normalization_constant=normalization,
            token_rows=token_rows,
        )
        for environment in EXPERIMENT_001_ENVIRONMENTS
    }
    save_activation_statistics(
        output,
        statistics=ActivationStatistics(
            modules=(module,),
            environments=EXPERIMENT_001_ENVIRONMENTS,
            values=values,
        ),
        phase=phase,
        provenance={
            "corpus_manifest_sha256": sha256_file(manifest_path),
            "gptqmodel_module_tree_revision": QWEN35_GPTQMODEL_TREE_REVISION,
            "model": _fixture_model_source(),
            "runtime": _fixture_diagonal_runtime(),
            "tokenizer": expected_tokenizer_metadata(),
        },
    )
    return output / f"{phase}.diagonals.json"


def _rewrite_diagonal_metadata(path: Path, mutate) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    mutate(metadata)
    path.write_bytes(canonical_json_bytes(metadata))
    sidecar_path = path.with_name(f"{path.name[:-5]}.sha256.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"] = {
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))


def test_diagonal_loader_validates_hashes_and_normalization(tmp_path: Path) -> None:
    metadata_path = _write_small_diagonal_artifact(tmp_path)
    artifact = load_diagonal_artifact(
        metadata_path,
        expected_phase="calibration",
        require_frozen_inventory=False,
    )
    assert artifact.environments == EXPERIMENT_001_ENVIRONMENTS
    assert artifact.normalized[artifact.modules[0].name].shape == (4, 128)


def test_diagonal_loader_rejects_tampered_archive(tmp_path: Path) -> None:
    metadata_path = _write_small_diagonal_artifact(tmp_path)
    archive_path = tmp_path / "calibration.diagonals.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        changed = {name: np.asarray(archive[name]).copy() for name in archive.files}
    first_key = sorted(changed)[0]
    changed[first_key][0] += 1.0
    np.savez_compressed(archive_path, **changed)

    with pytest.raises(ValueError, match=r"archive (size|hash) mismatch"):
        load_diagonal_artifact(
            metadata_path,
            expected_phase="calibration",
            require_frozen_inventory=False,
        )


def test_diagonal_loader_rejects_wrong_model_revision_with_consistent_hashes(
    tmp_path: Path,
) -> None:
    metadata_path = _write_small_diagonal_artifact(tmp_path)
    _rewrite_diagonal_metadata(
        metadata_path,
        lambda metadata: metadata["provenance"]["model"].update({"revision": "0" * 40}),
    )
    with pytest.raises(ValueError, match="model snapshot descriptor identity drift"):
        load_diagonal_artifact(
            metadata_path,
            expected_phase="calibration",
            require_frozen_inventory=False,
        )


def test_corpus_loader_rejects_changed_dataset_revision_with_consistent_hashes(
    tmp_path: Path,
) -> None:
    metadata_path = _write_small_diagonal_artifact(
        tmp_path,
        wrong_dataset_revision=True,
    )
    with pytest.raises(ValueError, match="dataset revision table drift"):
        load_diagonal_artifact(
            metadata_path,
            expected_phase="calibration",
            require_frozen_inventory=False,
        )


def test_corpus_loader_rejects_collection_contract_drift(tmp_path: Path) -> None:
    metadata_path = _write_small_diagonal_artifact(tmp_path)
    manifest_path = tmp_path / "calibration.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_loading"]["contract"]["sources"]["protocol"]["sha256"] = "0" * 64
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    sidecar_path = tmp_path / "calibration.sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["manifest"] = {
        "file": manifest_path.name,
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))

    with pytest.raises(ValueError, match="collection contract drift"):
        load_diagonal_artifact(
            metadata_path,
            expected_phase="calibration",
            require_frozen_inventory=False,
        )


def test_corpus_loader_rejects_tokenizer_identity_drift(tmp_path: Path) -> None:
    metadata_path = _write_small_diagonal_artifact(tmp_path)
    manifest_path = tmp_path / "calibration.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"]["class"] = "DifferentTokenizer"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    sidecar_path = tmp_path / "calibration.sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["manifest"] = {
        "file": manifest_path.name,
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))

    with pytest.raises(ValueError, match="tokenizer identity drift"):
        load_diagonal_artifact(
            metadata_path,
            expected_phase="calibration",
            require_frozen_inventory=False,
        )


def test_diagonal_loader_rejects_warn_only_determinism(tmp_path: Path) -> None:
    metadata_path = _write_small_diagonal_artifact(tmp_path)
    _rewrite_diagonal_metadata(
        metadata_path,
        lambda metadata: metadata["provenance"]["runtime"]["accelerator"].update(
            {"deterministic_algorithms_warn_only": True}
        ),
    )
    with pytest.raises(ValueError, match="deterministic CUDA settings drift"):
        load_diagonal_artifact(
            metadata_path,
            expected_phase="calibration",
            require_frozen_inventory=False,
        )


def test_diagonal_loader_rejects_wrong_corpus_manifest_hash(tmp_path: Path) -> None:
    metadata_path = _write_small_diagonal_artifact(tmp_path)
    _rewrite_diagonal_metadata(
        metadata_path,
        lambda metadata: metadata["provenance"].update({"corpus_manifest_sha256": "0" * 64}),
    )
    with pytest.raises(ValueError, match="corpus manifest hash mismatch"):
        load_diagonal_artifact(
            metadata_path,
            expected_phase="calibration",
            require_frozen_inventory=False,
        )


def test_pair_rejects_different_model_bytes(tmp_path: Path) -> None:
    calibration_path = _write_small_diagonal_artifact(
        tmp_path,
        phase="calibration",
    )
    heldout_path = _write_small_diagonal_artifact(
        tmp_path,
        phase="heldout",
    )

    def change_model_bytes(metadata: dict) -> None:
        descriptor = metadata["provenance"]["model"]
        descriptor["shards"][0]["sha256"] = "d" * 64
        payload = {key: value for key, value in descriptor.items() if key != "fingerprint_sha256"}
        descriptor["fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(payload))

    _rewrite_diagonal_metadata(heldout_path, change_model_bytes)
    calibration = load_diagonal_artifact(
        calibration_path,
        expected_phase="calibration",
        require_frozen_inventory=False,
    )
    heldout = load_diagonal_artifact(
        heldout_path,
        expected_phase="heldout",
        require_frozen_inventory=False,
    )
    with pytest.raises(ValueError, match="model-byte provenance differs"):
        validate_diagonal_pair(
            calibration,
            heldout,
            require_frozen_inventory=False,
        )


def test_pair_rejects_rendered_or_window_overlap(tmp_path: Path) -> None:
    shared_hash = "a" * 64
    calibration_path = _write_small_diagonal_artifact(
        tmp_path,
        phase="calibration",
        shared_rendered_hash=shared_hash,
    )
    heldout_path = _write_small_diagonal_artifact(
        tmp_path,
        phase="heldout",
        shared_rendered_hash=shared_hash,
    )
    calibration = load_diagonal_artifact(
        calibration_path,
        expected_phase="calibration",
        require_frozen_inventory=False,
    )
    heldout = load_diagonal_artifact(
        heldout_path,
        expected_phase="heldout",
        require_frozen_inventory=False,
    )
    with pytest.raises(ValueError, match="calibration/held-out overlap"):
        validate_diagonal_pair(
            calibration,
            heldout,
            require_frozen_inventory=False,
        )
