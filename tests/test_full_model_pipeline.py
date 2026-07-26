from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cliffquant.full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    FROZEN_ENVIRONMENTS,
    CalibrationDiagonals,
    FrozenModuleSpec,
    MappingWeightSource,
    _parse_module,
    load_scale_run,
    resolve_hf_snapshot_file,
    validate_corpus_replay_pair,
    verify_frozen_corpus_pair,
)
from cliffquant.full_model_solve import (
    build_certified_engine_identity,
    solve_modules_to_artifacts,
)
from cliffquant.full_model_verify import TensorCheckpoint
from cliffquant.gptqmodel_export import fp16_values_from_bits, quantize_rows
from cliffquant.provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from cliffquant.qwen35_modules import QWEN35_GPTQMODEL_TREE_REVISION


def _tiny_module() -> FrozenModuleSpec:
    return FrozenModuleSpec(
        name="model.language_model.layers.0.self_attn.q_proj",
        layer=0,
        branch="self_attn.q_proj",
        in_features=128,
        out_features=8,
    )


def _test_engine_identity() -> dict[str, object]:
    return {
        "certificate": {
            "aggregate_fingerprint_sha256": "a" * 64,
            "completed_cases": 10_000,
            "file": "certificate-v2.json",
            "protocol_gate_passed": True,
            "schema": "cliffquant-solver-certification-v2",
            "sha256": "b" * 64,
            "size_bytes": 1,
            "sources": {},
        },
        "engine_source_sha256": "c" * 64,
        "engine_sources": {"full_model_solve.py": "c" * 64},
        "engine_version": "test-engine-v2",
        "solver_source_sha256": {
            "breakpoint.py": "d" * 64,
            "certification.py": "4" * 64,
            "contracts.py": "e" * 64,
            "fp16_grid.py": "f" * 64,
            "objective.py": "1" * 64,
            "provenance.py": "2" * 64,
            "SOLVER_CERTIFICATION_001_PROTOCOL.md": "5" * 64,
            "solver.py": "3" * 64,
            "solver_certificate.py": "6" * 64,
        },
    }


def _fake_replay_pair() -> dict[str, object]:
    def replay(phase: str) -> dict[str, object]:
        return {
            "accepted_source_rows": 4,
            "cache": {"page_count": 4, "pages_sha256": "7" * 64},
            "manifest": {
                "file": f"{phase}.manifest.json",
                "sha256": ("8" if phase == "calibration" else "9") * 64,
                "size_bytes": 1,
            },
            "phase": phase,
            "schema": "cliffquant.corpus-replay.v1",
            "sidecar": {
                "file": f"{phase}.sha256.json",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            "token_archive": {
                "file": f"{phase}.tokens.npz",
                "sha256": ("b" if phase == "calibration" else "c") * 64,
                "size_bytes": 1,
            },
            "tokenizer_sha256": "d" * 64,
            "used_rendered_records": 4,
            "window_count": 128 if phase == "calibration" else 256,
        }

    identity = {
        "calibration": replay("calibration"),
        "heldout": replay("heldout"),
    }
    return {
        **identity,
        "schema": "cliffquant.corpus-replay-pair.v1",
        "sha256": sha256_bytes(canonical_json_bytes(identity)),
    }


def _write_calibration_bundle(root: Path) -> Path:
    module = _tiny_module()
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, object]] = []
    for environment_index, environment in enumerate(FROZEN_ENVIRONMENTS):
        raw_key = f"m000__e{environment_index:02d}__raw"
        normalized_key = f"m000__e{environment_index:02d}__normalized"
        arrays[raw_key] = np.full(128, environment_index + 1, dtype=np.float64)
        arrays[normalized_key] = np.ones(128, dtype=np.float64)
        entries.append(
            {
                "environment": environment,
                "module": module.name,
                "normalization_constant": float(environment_index + 1),
                "normalized": {
                    "array_key": normalized_key,
                    "dtype": "<f8",
                    "sha256": canonical_array_sha256(arrays[normalized_key]),
                    "shape": [128],
                },
                "raw": {
                    "array_key": raw_key,
                    "dtype": "<f8",
                    "sha256": canonical_array_sha256(arrays[raw_key]),
                    "shape": [128],
                },
                "token_rows": 4,
            }
        )
    archive = root / "calibration.diagonals.npz"
    np.savez_compressed(archive, **arrays)
    metadata = {
        "arrays": entries,
        "environments": list(FROZEN_ENVIRONMENTS),
        "modules": [
            {
                "branch": module.branch,
                "in_features": module.in_features,
                "layer": module.layer,
                "name": module.name,
                "out_features": module.out_features,
            }
        ],
        "module_tree": {
            "implementation": "GPTQModel qwen3_5.py",
            "revision": QWEN35_GPTQMODEL_TREE_REVISION,
        },
        "phase": "calibration",
        "provenance": {"model": {"id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION}},
        "schema": "cliffquant.activation-diagonals.v1",
        "statistics_archive": {
            "file": archive.name,
            "sha256": sha256_file(archive),
            "size_bytes": archive.stat().st_size,
        },
    }
    metadata_path = root / "calibration.diagonals.json"
    metadata_path.write_bytes(canonical_json_bytes(metadata))
    hashes = {
        "metadata": {
            "file": metadata_path.name,
            "sha256": sha256_file(metadata_path),
            "size_bytes": metadata_path.stat().st_size,
        },
        "statistics": metadata["statistics_archive"],
    }
    (root / "calibration.diagonals.sha256.json").write_bytes(canonical_json_bytes(hashes))
    return metadata_path


def test_scale_pipeline_resumes_and_canonicalizes_all_zero_export_scale(
    tmp_path: Path,
) -> None:
    module = _tiny_module()
    weights = np.zeros((8, 128), dtype=np.float32)
    weights[0, 0] = 1.0
    source = MappingWeightSource({module.weight_key: weights})
    curvature = np.ones((4, 128), dtype=np.float64)
    calibration = CalibrationDiagonals(
        metadata_path=tmp_path / "calibration.diagonals.json",
        archive_path=tmp_path / "calibration.diagonals.npz",
        metadata_sha256="1" * 64,
        archive_sha256="2" * 64,
        modules=(module,),
        environments=FROZEN_ENVIRONMENTS,
        normalized={module.name: curvature},
        corpus_replay=_fake_replay_pair(),
    )
    output = tmp_path / "scales"

    solve_modules_to_artifacts(
        output,
        source=source,
        calibration=calibration,
        row_chunk_size=2,
        solver_chunk_size=128,
        workers=1,
        engine_identity=_test_engine_identity(),
    )
    first_manifest_bytes = (output / "scale-run.json").read_bytes()
    first_manifest_sha256 = sha256_file(output / "scale-run.json")
    first = load_scale_run(output, strict_inventory=False)
    # A second run must validate and reuse every row checkpoint.
    resumed = solve_modules_to_artifacts(
        output,
        source=source,
        calibration=calibration,
        row_chunk_size=2,
        solver_chunk_size=128,
        workers=1,
        engine_identity=_test_engine_identity(),
    )
    second = load_scale_run(output, strict_inventory=False)

    assert (output / "scale-run.json").read_bytes() == first_manifest_bytes
    assert sha256_file(output / "scale-run.json") == first_manifest_sha256
    assert resumed == json.loads(first_manifest_bytes)
    np.testing.assert_array_equal(
        first.artifacts[module.name].scale_bits,
        second.artifacts[module.name].scale_bits,
    )
    artifact = second.artifacts[module.name]
    assert artifact.scale_bits[1, 0] == 0x0001
    assert artifact.export_scale_bits[1, 0] == 0x3C00
    assert artifact.objectives[1, 0] == 0.0
    assert artifact.scale_bits[0, 0] != 0
    assert second.metadata["corpus_replay"] == _fake_replay_pair()
    chunk = next(output.rglob("rows-*.json"))
    chunk_metadata = json.loads(chunk.read_text(encoding="utf-8"))
    assert chunk_metadata["input"]["corpus_replay_sha256"] == _fake_replay_pair()["sha256"]


def test_full_model_engine_binds_shared_v2_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    certificate = tmp_path / "certificate-v2.json"
    descriptor = dict(_test_engine_identity()["certificate"])
    descriptor["sources"] = {
        name: {
            "file": filename,
            "sha256": str(index) * 64,
            "size_bytes": 1,
        }
        for index, (name, filename) in enumerate(
            (
                ("breakpoint", "breakpoint.py"),
                ("certification", "certification.py"),
                ("contracts", "contracts.py"),
                ("fp16_grid", "fp16_grid.py"),
                ("objective", "objective.py"),
                ("protocol", "SOLVER_CERTIFICATION_001_PROTOCOL.md"),
                ("solver", "solver.py"),
            ),
            start=1,
        )
    }
    monkeypatch.setattr(
        "cliffquant.full_model_solve.validate_solver_certificate",
        lambda supplied: descriptor if Path(supplied) == certificate else None,
    )

    identity = build_certified_engine_identity(certificate)

    assert identity["certificate"]["schema"] == "cliffquant-solver-certification-v2"
    assert identity["certificate"]["completed_cases"] == 10_000
    assert identity["certificate"]["protocol_gate_passed"] is True
    assert identity["certificate"] == descriptor
    assert str(certificate.parent.resolve()) not in json.dumps(identity)
    assert set(identity["engine_sources"]) == {
        "corpus.py",
        "corpus_replay.py",
        "dataset_viewer.py",
        "experiment001_io.py",
        "full_model_artifacts.py",
        "full_model_solve.py",
        "model_snapshot.py",
        "provenance.py",
        "qwen35_modules.py",
        "solver_certificate.py",
    }
    assert {
        "breakpoint.py",
        "certification.py",
        "contracts.py",
        "fp16_grid.py",
        "objective.py",
        "provenance.py",
        "SOLVER_CERTIFICATION_001_PROTOCOL.md",
        "solver.py",
        "solver_certificate.py",
    } == set(identity["solver_source_sha256"])


def test_scale_pipeline_uses_windows_spawn_workers(tmp_path: Path) -> None:
    module = _tiny_module()
    source = MappingWeightSource(
        {module.weight_key: np.zeros((module.out_features, module.in_features))}
    )
    calibration = CalibrationDiagonals(
        metadata_path=tmp_path / "calibration.diagonals.json",
        archive_path=tmp_path / "calibration.diagonals.npz",
        metadata_sha256="1" * 64,
        archive_sha256="2" * 64,
        modules=(module,),
        environments=FROZEN_ENVIRONMENTS,
        normalized={module.name: np.ones((4, 128), dtype=np.float64)},
    )

    solve_modules_to_artifacts(
        tmp_path / "spawn-scales",
        source=source,
        calibration=calibration,
        row_chunk_size=2,
        solver_chunk_size=128,
        workers=2,
        engine_identity=_test_engine_identity(),
    )

    artifact = load_scale_run(
        tmp_path / "spawn-scales",
        strict_inventory=False,
    ).artifacts[module.name]
    assert np.all(artifact.scale_bits == 0x0001)
    assert np.all(artifact.export_scale_bits == 0x3C00)


def test_scale_pipeline_rejects_calibration_from_different_model_bytes(
    tmp_path: Path,
) -> None:
    module = _tiny_module()
    calibration = CalibrationDiagonals(
        metadata_path=tmp_path / "calibration.diagonals.json",
        archive_path=tmp_path / "calibration.diagonals.npz",
        metadata_sha256="1" * 64,
        archive_sha256="2" * 64,
        modules=(module,),
        environments=FROZEN_ENVIRONMENTS,
        normalized={module.name: np.ones((4, 128), dtype=np.float64)},
        source={"model": {"fingerprint_sha256": "f" * 64}},
    )

    with pytest.raises(ValueError, match="different model bytes"):
        solve_modules_to_artifacts(
            tmp_path / "scales",
            source=MappingWeightSource(
                {module.weight_key: np.zeros((module.out_features, module.in_features))}
            ),
            calibration=calibration,
            engine_identity=_test_engine_identity(),
        )


def test_absmax_scale_run_is_genuine_and_immutably_tagged(tmp_path: Path) -> None:
    module = _tiny_module()
    weights = np.zeros((8, 128), dtype=np.float64)
    weights[0, 0] = 7.0
    weights[0, 1] = -8.0
    calibration = CalibrationDiagonals(
        metadata_path=tmp_path / "calibration.diagonals.json",
        archive_path=tmp_path / "calibration.diagonals.npz",
        metadata_sha256="1" * 64,
        archive_sha256="2" * 64,
        modules=(module,),
        environments=FROZEN_ENVIRONMENTS,
        normalized={module.name: np.ones((4, 128), dtype=np.float64)},
    )
    output = tmp_path / "absmax-scales"

    solve_modules_to_artifacts(
        output,
        source=MappingWeightSource({module.weight_key: weights}),
        calibration=calibration,
        row_chunk_size=4,
        workers=1,
        policy="absmax",
        engine_identity=_test_engine_identity(),
    )
    run = load_scale_run(output, strict_inventory=False)

    assert run.policy == "absmax"
    assert run.artifacts[module.name].scale_bits[0, 0] == 0x3C00
    assert run.artifacts[module.name].export_scale_bits[0, 0] == 0x3C00
    chunk = next((output / "modules").rglob("rows-*.json"))
    metadata = json.loads(chunk.read_text(encoding="utf-8"))
    assert metadata["policy"] == "absmax"
    assert metadata["input"]["policy"] == "absmax"
    assert metadata["input"]["solver_chunk_size"] == 4096
    assert metadata["input"]["engine"]["engine_version"] == "test-engine-v2"
    assert set(metadata["input"]["engine"]["solver_source_sha256"]) == {
        "breakpoint.py",
        "certification.py",
        "contracts.py",
        "fp16_grid.py",
        "objective.py",
        "provenance.py",
        "SOLVER_CERTIFICATION_001_PROTOCOL.md",
        "solver.py",
        "solver_certificate.py",
    }


def test_scale_run_rejects_top_level_policy_relabelling(tmp_path: Path) -> None:
    module = _tiny_module()
    calibration = CalibrationDiagonals(
        metadata_path=tmp_path / "calibration.diagonals.json",
        archive_path=tmp_path / "calibration.diagonals.npz",
        metadata_sha256="1" * 64,
        archive_sha256="2" * 64,
        modules=(module,),
        environments=FROZEN_ENVIRONMENTS,
        normalized={module.name: np.ones((4, 128), dtype=np.float64)},
    )
    output = tmp_path / "scales"
    solve_modules_to_artifacts(
        output,
        source=MappingWeightSource(
            {module.weight_key: np.ones((module.out_features, module.in_features))}
        ),
        calibration=calibration,
        engine_identity=_test_engine_identity(),
    )
    manifest_path = output / "scale-run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy"] = "absmax"
    manifest["quantizer"]["selection"] = "absmax_signed_range_nearest_fp16"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="module policy mismatch"):
        load_scale_run(output, strict_inventory=False)


def test_resume_rejects_solver_or_policy_identity_drift(tmp_path: Path) -> None:
    module = _tiny_module()
    calibration = CalibrationDiagonals(
        metadata_path=tmp_path / "calibration.diagonals.json",
        archive_path=tmp_path / "calibration.diagonals.npz",
        metadata_sha256="1" * 64,
        archive_sha256="2" * 64,
        modules=(module,),
        environments=FROZEN_ENVIRONMENTS,
        normalized={module.name: np.ones((4, 128), dtype=np.float64)},
    )
    source = MappingWeightSource(
        {module.weight_key: np.zeros((module.out_features, module.in_features))}
    )
    output = tmp_path / "immutable-scales"
    solve_modules_to_artifacts(
        output,
        source=source,
        calibration=calibration,
        solver_chunk_size=128,
        engine_identity=_test_engine_identity(),
    )

    with pytest.raises(ValueError, match="input drift"):
        solve_modules_to_artifacts(
            output,
            source=source,
            calibration=calibration,
            solver_chunk_size=256,
            engine_identity=_test_engine_identity(),
        )
    with pytest.raises(ValueError, match="input drift"):
        solve_modules_to_artifacts(
            output,
            source=source,
            calibration=calibration,
            solver_chunk_size=128,
            policy="absmax",
            engine_identity=_test_engine_identity(),
        )
    with pytest.raises(ValueError, match="chunk checkpoint inventory drift"):
        solve_modules_to_artifacts(
            output,
            source=source,
            calibration=calibration,
            row_chunk_size=2,
            solver_chunk_size=128,
            engine_identity=_test_engine_identity(),
        )


def test_fp16_bits_and_frozen_quantizer_semantics() -> None:
    scales = fp16_values_from_bits(np.asarray([[0x3C00]], dtype=np.uint16))
    weights = np.zeros((1, 128), dtype=np.float64)
    weights[0, :6] = [-8.5, -7.5, -6.5, 6.5, 7.5, 8.5]

    codes = quantize_rows(weights, scales)

    assert scales.dtype == np.float16
    assert scales[0, 0] == 1.0
    np.testing.assert_array_equal(codes[0, :6], [-8, -8, -6, 6, 7, 7])


def test_fp16_bits_reject_nonpositive_and_infinite_encodings() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        fp16_values_from_bits(np.asarray([[0]], dtype=np.uint16))
    with pytest.raises(ValueError, match="positive finite"):
        fp16_values_from_bits(np.asarray([[0x7C00]], dtype=np.uint16))


def test_mapping_source_fingerprint_is_stable() -> None:
    module = _tiny_module()
    source = MappingWeightSource(
        {module.weight_key: np.zeros((module.out_features, module.in_features))}
    )
    first = sha256_bytes(canonical_json_bytes(dict(source.provenance)))
    second = sha256_bytes(canonical_json_bytes(dict(source.provenance)))
    assert first == second
    assert json.loads(canonical_json_bytes(dict(source.provenance))) == {
        "kind": "in-memory-test-source"
    }


def test_module_metadata_branch_must_match_name() -> None:
    with pytest.raises(ValueError, match="branch metadata"):
        _parse_module(
            {
                "branch": "mlp.up_proj",
                "in_features": 128,
                "layer": 0,
                "name": "model.language_model.layers.0.self_attn.q_proj",
                "out_features": 8,
            }
        )


def test_snapshot_path_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes pinned snapshot"):
        resolve_hf_snapshot_file(tmp_path, "../outside.safetensors", label="test shard")


def test_tensor_checkpoint_rejects_unindexed_shard_keys(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    root = tmp_path / "checkpoint"
    root.mkdir()
    shard = root / "model-00001-of-00001.safetensors"
    save_file(
        {
            "indexed": np.zeros(1, dtype=np.float32),
            "unindexed": np.zeros(1, dtype=np.float32),
        },
        shard,
    )
    (root / "model.safetensors.index.json").write_bytes(
        canonical_json_bytes(
            {
                "weight_map": {
                    "indexed": shard.name,
                }
            }
        )
    )

    with pytest.raises(ValueError, match="index coverage mismatch"):
        TensorCheckpoint(root)


def test_corpus_replay_pair_fingerprint_is_immutable() -> None:
    replay = _fake_replay_pair()
    assert validate_corpus_replay_pair(replay) == replay
    replay["heldout"]["cache"]["pages_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_corpus_replay_pair(replay)


def test_corpus_pair_replay_loads_one_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cliffquant.corpus_replay as corpus_replay

    tokenizer = object()
    loads: list[object] = []
    phases: list[tuple[str, object]] = []
    expected = _fake_replay_pair()

    def fake_load() -> object:
        loads.append(tokenizer)
        return tokenizer

    def fake_verify(_directory: Path, *, phase: str, tokenizer: object) -> dict:
        phases.append((phase, tokenizer))
        return expected[phase]

    monkeypatch.setattr(corpus_replay, "load_pinned_tokenizer", fake_load)
    monkeypatch.setattr(corpus_replay, "verify_corpus_replay", fake_verify)

    observed = verify_frozen_corpus_pair(tmp_path)

    assert observed == expected
    assert loads == [tokenizer]
    assert phases == [
        ("calibration", tokenizer),
        ("heldout", tokenizer),
    ]
