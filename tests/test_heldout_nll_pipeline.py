from __future__ import annotations

import json
import struct
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cliffquant.heldout_nll as heldout_nll
from cliffquant.full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    FROZEN_ENVIRONMENTS,
)
from cliffquant.provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from cliffquant.qwen35_modules import QWEN35_GPTQMODEL_TREE_REVISION


def _fake_model_source() -> dict:
    payload = {
        "config": {
            "file": "config.json",
            "sha256": "6" * 64,
            "size_bytes": 1,
        },
        "id": BASE_MODEL_ID,
        "index": {
            "file": "model.safetensors.index.json",
            "sha256": "7" * 64,
            "size_bytes": 1,
        },
        "revision": BASE_MODEL_REVISION,
        "schema": "cliffquant.model-snapshot.v1",
        "shards": [
            {
                "file": "model-00001-of-00001.safetensors",
                "sha256": "8" * 64,
                "size_bytes": 1,
            }
        ],
        "tensor_count": 488,
    }
    return {
        **payload,
        "fingerprint_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _fake_replay_pair() -> dict:
    def replay(phase: str, *, manifest_sha256: str, archive_sha256: str) -> dict:
        return {
            "accepted_source_rows": 4,
            "cache": {
                "page_count": 4,
                "pages_sha256": "4" * 64,
            },
            "manifest": {
                "file": f"{phase}.manifest.json",
                "sha256": manifest_sha256,
                "size_bytes": 1,
            },
            "phase": phase,
            "schema": "cliffquant.corpus-replay.v1",
            "sidecar": {
                "file": f"{phase}.sha256.json",
                "sha256": "3" * 64,
                "size_bytes": 1,
            },
            "token_archive": {
                "file": f"{phase}.tokens.npz",
                "sha256": archive_sha256,
                "size_bytes": 1,
            },
            "tokenizer_sha256": "5" * 64,
            "used_rendered_records": 4,
            "window_count": 128 if phase == "calibration" else 256,
        }

    identity = {
        "calibration": replay(
            "calibration",
            manifest_sha256="a" * 64,
            archive_sha256="b" * 64,
        ),
        "heldout": replay(
            "heldout",
            manifest_sha256="2" * 64,
            archive_sha256="1" * 64,
        ),
    }
    return {
        **identity,
        "schema": "cliffquant.corpus-replay-pair.v1",
        "sha256": sha256_bytes(canonical_json_bytes(identity)),
    }


def _fake_runtime_metadata(*, device: str, package_names: tuple[str, ...]) -> dict:
    assert device == "cuda:0"
    assert package_names == tuple(heldout_nll._NLL_RUNTIME_PACKAGE_VERSIONS)
    return {
        "accelerator": {
            "available": True,
            "capability": [12, 0],
            "cublas_workspace_config": heldout_nll.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "cuda_runtime": "12.8",
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_version": 91002,
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": False,
            "device_index": 0,
            "float32_matmul_precision": "highest",
            "matmul_allow_tf32": False,
            "name": "Test GPU",
            "total_memory_bytes": 8_000_000_000,
        },
        "device": device,
        "implementation": "CPython",
        "machine": "AMD64",
        "numpy": heldout_nll._NLL_RUNTIME_PACKAGE_VERSIONS["numpy"],
        "numpy_build": {
            "blas": {
                "found": True,
                "name": "test-blas",
                "openblas configuration": "test",
                "version": "1",
            },
            "host": {"system": "windows"},
            "lapack": {
                "found": True,
                "name": "test-lapack",
                "openblas configuration": "test",
                "version": "1",
            },
            "simd": {"baseline": [], "found": []},
        },
        "packages": dict(heldout_nll._NLL_RUNTIME_PACKAGE_VERSIONS),
        "platform": "Windows-test",
        "python": "3.11.15",
        "python_executable": "python.exe",
    }


def _write_export_stub(
    root: Path,
    *,
    policy: str,
    scale_run_sha256: str,
    model_payload: bytes,
) -> None:
    corpus_replay = _fake_replay_pair()
    model_source = _fake_model_source()
    root.mkdir()
    model_path = root / "model.safetensors"
    model_path.write_bytes(model_payload)
    quantize_config_path = root / "quantize_config.json"
    quantize_config_path.write_bytes(
        canonical_json_bytes(
            {
                "bits": 4,
                "format": "gptq",
                "group_size": 128,
                "meta": {
                    "cliffquant": {
                        "base_model": BASE_MODEL_ID,
                        "base_revision": BASE_MODEL_REVISION,
                        "base_snapshot_sha256": model_source["fingerprint_sha256"],
                        "corpus_replay_sha256": corpus_replay["sha256"],
                        "gptqmodel_revision": QWEN35_GPTQMODEL_TREE_REVISION,
                        "policy": policy,
                        "scale_run_sha256": scale_run_sha256,
                    }
                },
                "method": "gptq",
                "quant_method": "gptq",
                "sym": True,
            }
        )
    )
    descriptors = [
        {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (model_path, quantize_config_path)
    ]
    (root / "cliffquant_export.json").write_bytes(
        canonical_json_bytes(
            {
                "base_model": {
                    "id": BASE_MODEL_ID,
                    "revision": BASE_MODEL_REVISION,
                    "source": model_source,
                },
                "corpus_replay": corpus_replay,
                "files": descriptors,
                "gptq": {
                    "backend": "GPTQ_TORCH",
                    "bits": 4,
                    "desc_act": False,
                    "format": "gptq",
                    "group_size": 128,
                    "method": "gptq",
                    "pack_dtype": "int32",
                    "sym": True,
                    "zero": 8,
                },
                "gptqmodel": {
                    "revision": QWEN35_GPTQMODEL_TREE_REVISION,
                    "version": "7.3.4",
                },
                "policy": policy,
                "scale_run": {
                    "manifest_sha256": scale_run_sha256,
                    "path": "manifest.json",
                },
                "schema": "cliffquant.gptq-export.v1",
            }
        )
    )


def _write_heldout_bundle(root: Path) -> Path:
    arrays: dict[str, np.ndarray] = {}
    environments: dict[str, object] = {}
    for environment_index, environment in enumerate(FROZEN_ENVIRONMENTS):
        ids = np.arange(64 * 256, dtype=np.int64).reshape(64, 256) + environment_index * 100_000
        mask = np.ones_like(ids, dtype=np.uint8)
        arrays[f"{environment}__input_ids"] = ids
        arrays[f"{environment}__attention_mask"] = mask
        environments[environment] = {
            "datasets": [],
            "records": [],
            "windows": [
                {
                    "index": index,
                    "language": None,
                    "segments": [],
                    "token_count": 256,
                    "token_sha256": sha256_bytes(canonical_json_bytes(ids[index].tolist())),
                }
                for index in range(64)
            ],
        }
    archive = root / "heldout.tokens.npz"
    np.savez_compressed(archive, **arrays)
    manifest = {
        "dry_run": False,
        "environments": environments,
        "model": {"id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION},
        "phase": "heldout",
        "schema": "cliffquant.corpus-manifest.v1",
        "token_archive": {
            "file": archive.name,
            "sha256": sha256_file(archive),
            "size_bytes": archive.stat().st_size,
        },
        "window_count_per_environment": 64,
        "window_tokens": 256,
    }
    manifest_path = root / "heldout.manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (root / "heldout.sha256.json").write_bytes(
        canonical_json_bytes(
            {
                "manifest": {
                    "file": manifest_path.name,
                    "sha256": sha256_file(manifest_path),
                    "size_bytes": manifest_path.stat().st_size,
                },
                "tokens": manifest["token_archive"],
            }
        )
    )
    return manifest_path


def _write_valid_nll_result(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cliffquant_mean: float = 1.005,
) -> tuple[Path, dict]:
    corpus_replay = _fake_replay_pair()
    monkeypatch.setattr(heldout_nll, "_configure_deterministic_evaluation", lambda _device: None)
    monkeypatch.setattr(heldout_nll, "runtime_metadata", _fake_runtime_metadata)
    monkeypatch.setattr(
        heldout_nll,
        "verify_frozen_corpus_pair",
        lambda _directory: corpus_replay,
    )
    checkpoints = {}
    for role in ("absmax", "cliffquant"):
        checkpoint = root / role
        _write_export_stub(
            checkpoint,
            policy="absmax" if role == "absmax" else "cliffquant_minimax",
            scale_run_sha256=role[0] * 64,
            model_payload=role.encode(),
        )
        checkpoints[role] = checkpoint
    manifest = {
        "_validated": {
            "archive_file": "heldout.tokens.npz",
            "archive_sha256": "1" * 64,
            "manifest_file": "heldout.manifest.json",
            "manifest_sha256": "2" * 64,
            "sidecar_sha256": "3" * 64,
        }
    }
    tokens = {
        environment: (
            np.zeros((64, 256), dtype=np.int64),
            np.ones((64, 256), dtype=np.uint8),
        )
        for environment in FROZEN_ENVIRONMENTS
    }
    monkeypatch.setattr(
        heldout_nll,
        "load_heldout_tokens",
        lambda _path: (manifest, tokens),
    )

    def fake_evaluate(*, checkpoint: Path, **_kwargs):
        mean = 1.0 if checkpoint.name == "absmax" else cliffquant_mean
        metrics = {}
        arrays = {}
        for environment in FROZEN_ENVIRONMENTS:
            token_nll = np.full((64, 255), mean, dtype=np.float64)
            target_mask = np.ones((64, 255), dtype=np.uint8)
            window_sums = np.sum(token_nll, axis=1, dtype=np.float64)
            window_counts = np.sum(target_mask, axis=1, dtype=np.int64)
            total_nll = float(np.sum(window_sums, dtype=np.float64))
            total_tokens = int(np.sum(window_counts, dtype=np.int64))
            metrics[environment] = {
                "mean_nll": total_nll / total_tokens,
                "target_tokens": total_tokens,
                "total_nll": total_nll,
                "windows": 64,
            }
            arrays[f"{environment}__target_mask"] = target_mask
            arrays[f"{environment}__token_nll"] = token_nll
            arrays[f"{environment}__window_nll_sum"] = window_sums
            arrays[f"{environment}__window_target_tokens"] = window_counts
        metrics["macro_mean_nll"] = float(
            np.mean([metrics[environment]["mean_nll"] for environment in FROZEN_ENVIRONMENTS])
        )
        return metrics, arrays

    monkeypatch.setattr(heldout_nll, "_evaluate_checkpoint", fake_evaluate)
    output = root / "evidence"
    report = heldout_nll.compare_heldout_nll(
        cliffquant_checkpoint=checkpoints["cliffquant"],
        absmax_checkpoint=checkpoints["absmax"],
        heldout_manifest=root / "ignored.json",
        gptqmodel_source=root / "gptqmodel",
        output_dir=output,
    )
    return output, report


def _rewrite_report_and_sidecar(output: Path, report: dict) -> None:
    report_path = output / "heldout-nll.json"
    report_path.write_bytes(canonical_json_bytes(report))
    sidecar_path = output / "heldout-nll.sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["report"] = {
        "file": report_path.name,
        "sha256": sha256_file(report_path),
        "size_bytes": report_path.stat().st_size,
    }
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))


def _rewrite_raw_and_outer_hashes(
    output: Path,
    report: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    raw_path = output / "heldout-nll.raw.npz"
    temporary = output / "mutated.raw.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(raw_path)
    report["raw_evidence"]["sha256"] = sha256_file(raw_path)
    report["raw_evidence"]["size_bytes"] = raw_path.stat().st_size
    _rewrite_report_and_sidecar(output, report)
    sidecar_path = output / "heldout-nll.sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw"] = {
        "file": raw_path.name,
        "sha256": sha256_file(raw_path),
        "size_bytes": raw_path.stat().st_size,
    }
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))


def _rehash_raw_and_outer_evidence(output: Path, report: dict) -> None:
    raw_path = output / "heldout-nll.raw.npz"
    report["raw_evidence"]["sha256"] = sha256_file(raw_path)
    report["raw_evidence"]["size_bytes"] = raw_path.stat().st_size
    _rewrite_report_and_sidecar(output, report)
    sidecar_path = output / "heldout-nll.sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw"] = {
        "file": raw_path.name,
        "sha256": sha256_file(raw_path),
        "size_bytes": raw_path.stat().st_size,
    }
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))


def _rewrite_raw_zip_inventory(output: Path, mutation: str) -> None:
    raw_path = output / "heldout-nll.raw.npz"
    with zipfile.ZipFile(raw_path, mode="r") as archive:
        entries = [(info.filename, archive.read(info)) for info in archive.infolist()]
    if mutation == "duplicate":
        entries.append(entries[0])
    elif mutation == "extra":
        entries.append(("unexpected.npy", entries[0][1]))
    elif mutation == "missing":
        entries.pop()
    else:
        raise AssertionError(f"unknown ZIP mutation: {mutation}")
    temporary = output / "mutated.raw.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
    temporary.replace(raw_path)


def test_heldout_loader_uses_strict_corpus_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_heldout_bundle(tmp_path)
    archive_path = tmp_path / "heldout.tokens.npz"
    sidecar_path = tmp_path / "heldout.sha256.json"
    calls: list[tuple[Path, str]] = []

    def fake_strict_loader(directory: Path, *, phase: str):
        calls.append((Path(directory), phase))
        return SimpleNamespace(
            archive_path=archive_path.resolve(),
            archive_sha256=sha256_file(archive_path),
            manifest_path=manifest_path.resolve(),
            manifest_sha256=sha256_file(manifest_path),
            sidecar_sha256=sha256_file(sidecar_path),
        )

    monkeypatch.setattr(heldout_nll, "load_corpus_artifact", fake_strict_loader)

    manifest, tokens = heldout_nll.load_heldout_tokens(manifest_path)

    assert calls == [(tmp_path.resolve(), "heldout")]
    assert manifest["_validated"]["manifest_sha256"] == sha256_file(manifest_path)
    assert tuple(tokens) == FROZEN_ENVIRONMENTS
    assert tokens["general"][0].shape == (64, 256)


def test_heldout_loader_propagates_strict_corpus_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_heldout_bundle(tmp_path)
    monkeypatch.setattr(
        heldout_nll,
        "load_corpus_artifact",
        lambda _directory, *, phase: (_ for _ in ()).throw(
            ValueError(f"{phase} corpus manifest phase mismatch")
        ),
    )

    with pytest.raises(ValueError, match="corpus manifest phase mismatch"):
        heldout_nll.load_heldout_tokens(manifest_path)


def test_paired_nll_writes_raw_evidence_and_applies_frozen_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_replay = _fake_replay_pair()
    monkeypatch.setattr(heldout_nll, "_configure_deterministic_evaluation", lambda _device: None)
    monkeypatch.setattr(heldout_nll, "runtime_metadata", _fake_runtime_metadata)
    monkeypatch.setattr(
        heldout_nll,
        "verify_frozen_corpus_pair",
        lambda _directory: corpus_replay,
    )
    checkpoints = {}
    for policy in ("absmax", "cliffquant"):
        checkpoint = tmp_path / policy
        immutable_policy = "absmax" if policy == "absmax" else "cliffquant_minimax"
        _write_export_stub(
            checkpoint,
            policy=immutable_policy,
            scale_run_sha256=policy[0] * 64,
            model_payload=policy.encode(),
        )
        checkpoints[policy] = checkpoint
    manifest = {
        "_validated": {
            "archive_file": "heldout.tokens.npz",
            "archive_sha256": "1" * 64,
            "manifest_file": "heldout.manifest.json",
            "manifest_sha256": "2" * 64,
            "sidecar_sha256": "3" * 64,
        }
    }
    tokens = {
        environment: (
            np.asarray([[1, 2]], dtype=np.int64),
            np.asarray([[1, 1]], dtype=np.uint8),
        )
        for environment in FROZEN_ENVIRONMENTS
    }
    monkeypatch.setattr(
        heldout_nll,
        "load_heldout_tokens",
        lambda _path: (manifest, tokens),
    )

    def fake_evaluate(*, checkpoint: Path, **_kwargs):
        mean = 1.0 if checkpoint.name == "absmax" else 1.005
        metrics = {}
        arrays = {}
        for environment in FROZEN_ENVIRONMENTS:
            token_nll = np.full((64, 255), mean, dtype=np.float64)
            target_mask = np.ones((64, 255), dtype=np.uint8)
            window_sums = np.sum(token_nll, axis=1, dtype=np.float64)
            window_counts = np.sum(target_mask, axis=1, dtype=np.int64)
            total_nll = float(np.sum(window_sums, dtype=np.float64))
            total_tokens = int(np.sum(window_counts, dtype=np.int64))
            metrics[environment] = {
                "mean_nll": total_nll / total_tokens,
                "target_tokens": total_tokens,
                "total_nll": total_nll,
                "windows": 64,
            }
            arrays[f"{environment}__target_mask"] = target_mask
            arrays[f"{environment}__token_nll"] = token_nll
            arrays[f"{environment}__window_nll_sum"] = window_sums
            arrays[f"{environment}__window_target_tokens"] = window_counts
        metrics["macro_mean_nll"] = float(
            np.mean([metrics[environment]["mean_nll"] for environment in FROZEN_ENVIRONMENTS])
        )
        return metrics, arrays

    monkeypatch.setattr(heldout_nll, "_evaluate_checkpoint", fake_evaluate)
    output = tmp_path / "evidence"
    report = heldout_nll.compare_heldout_nll(
        cliffquant_checkpoint=checkpoints["cliffquant"],
        absmax_checkpoint=checkpoints["absmax"],
        heldout_manifest=tmp_path / "ignored.json",
        gptqmodel_source=tmp_path / "gptqmodel",
        output_dir=output,
    )

    assert report["status"] == "pass"
    assert report["gates"]["publishable_model_nll_gate"] is True
    assert report["comparison"]["macro_delta_cliffquant_minus_absmax"] == pytest.approx(0.005)
    assert (output / "heldout-nll.raw.npz").is_file()
    hashes = json.loads((output / "heldout-nll.sha256.json").read_text(encoding="utf-8"))
    assert hashes["report"]["sha256"] == sha256_file(output / "heldout-nll.json")
    serialized = (output / "heldout-nll.json").read_text(encoding="utf-8")
    assert str(tmp_path.resolve()) not in serialized
    assert heldout_nll.validate_heldout_nll_result(output) == report


def test_paired_nll_rejects_swapped_or_identical_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(heldout_nll, "_configure_deterministic_evaluation", lambda _device: None)
    monkeypatch.setattr(heldout_nll, "runtime_metadata", _fake_runtime_metadata)
    monkeypatch.setattr(
        heldout_nll,
        "verify_frozen_corpus_pair",
        lambda _directory: _fake_replay_pair(),
    )
    checkpoint = tmp_path / "one-checkpoint"
    _write_export_stub(
        checkpoint,
        policy="cliffquant_minimax",
        scale_run_sha256="c" * 64,
        model_payload=b"one",
    )
    monkeypatch.setattr(
        heldout_nll,
        "load_heldout_tokens",
        lambda _path: (
            {
                "_validated": {
                    "archive_file": "heldout.tokens.npz",
                    "archive_sha256": "1" * 64,
                    "manifest_file": "heldout.manifest.json",
                    "manifest_sha256": "2" * 64,
                    "sidecar_sha256": "3" * 64,
                }
            },
            {},
        ),
    )

    with pytest.raises(ValueError, match="policy mismatch"):
        heldout_nll.compare_heldout_nll(
            cliffquant_checkpoint=checkpoint,
            absmax_checkpoint=checkpoint,
            heldout_manifest=tmp_path / "ignored.json",
            gptqmodel_source=tmp_path / "gptqmodel",
            output_dir=tmp_path / "evidence",
        )


def test_nll_checkpoint_identity_rejects_symlinked_payload_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export_stub(
        checkpoint,
        policy="cliffquant_minimax",
        scale_run_sha256="c" * 64,
        model_payload=b"model",
    )
    model = checkpoint / "model.safetensors"
    outside = tmp_path / "outside-model.safetensors"
    model.replace(outside)
    try:
        model.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="regular non-symlink file"):
        heldout_nll._checkpoint_result_identity(
            checkpoint,
            role="cliffquant",
        )


def test_paired_nll_rejects_relabelled_identical_model_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(heldout_nll, "_configure_deterministic_evaluation", lambda _device: None)
    monkeypatch.setattr(heldout_nll, "runtime_metadata", _fake_runtime_metadata)
    monkeypatch.setattr(
        heldout_nll,
        "verify_frozen_corpus_pair",
        lambda _directory: _fake_replay_pair(),
    )
    absmax = tmp_path / "absmax"
    cliffquant = tmp_path / "cliffquant"
    _write_export_stub(
        absmax,
        policy="absmax",
        scale_run_sha256="a" * 64,
        model_payload=b"identical-model",
    )
    _write_export_stub(
        cliffquant,
        policy="cliffquant_minimax",
        scale_run_sha256="c" * 64,
        model_payload=b"identical-model",
    )
    monkeypatch.setattr(
        heldout_nll,
        "load_heldout_tokens",
        lambda _path: (
            {
                "_validated": {
                    "archive_file": "heldout.tokens.npz",
                    "archive_sha256": "1" * 64,
                    "manifest_file": "heldout.manifest.json",
                    "manifest_sha256": "2" * 64,
                    "sidecar_sha256": "3" * 64,
                }
            },
            {},
        ),
    )

    with pytest.raises(ValueError, match="model payloads must be distinct"):
        heldout_nll.compare_heldout_nll(
            cliffquant_checkpoint=cliffquant,
            absmax_checkpoint=absmax,
            heldout_manifest=tmp_path / "ignored.json",
            gptqmodel_source=tmp_path / "gptqmodel",
            output_dir=tmp_path / "evidence",
        )


def test_saved_nll_recomputes_metrics_after_outer_hashes_are_forged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _write_valid_nll_result(tmp_path, monkeypatch)
    forged = json.loads(canonical_json_bytes(report))
    forged["policies"]["absmax"]["general"]["mean_nll"] += 0.25
    _rewrite_report_and_sidecar(output, forged)

    with pytest.raises(ValueError, match="absmax/general mean NLL drift"):
        heldout_nll.validate_heldout_nll_result(output)


def test_saved_nll_rejects_forged_evaluator_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _write_valid_nll_result(tmp_path, monkeypatch)
    evaluator = report["evaluator"]
    evaluator["files"][0]["sha256"] = "0" * 64
    evaluator["sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "files": evaluator["files"],
                "schema": evaluator["schema"],
            }
        )
    )
    _rewrite_report_and_sidecar(output, report)

    with pytest.raises(ValueError, match="differs from current source bytes"):
        heldout_nll.validate_heldout_nll_result(output)


def test_saved_nll_accepts_valid_negative_evidence_and_rejects_forged_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _write_valid_nll_result(
        tmp_path,
        monkeypatch,
        cliffquant_mean=1.03,
    )
    assert report["status"] == "fail"
    assert heldout_nll.validate_heldout_nll_result(output) == report

    report["status"] = "pass"
    _rewrite_report_and_sidecar(output, report)
    with pytest.raises(ValueError, match="status disagrees with frozen gates"):
        heldout_nll.validate_heldout_nll_result(output)


def test_saved_nll_recomputes_raw_window_totals_after_all_hashes_are_forged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _write_valid_nll_result(tmp_path, monkeypatch)
    raw_path = output / "heldout-nll.raw.npz"
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    name = "cliffquant__math__token_nll"
    arrays[name][0, 0] += 0.5
    report["raw_evidence"]["arrays"][name]["sha256"] = canonical_array_sha256(arrays[name])
    _rewrite_raw_and_outer_hashes(output, report, arrays)

    with pytest.raises(ValueError, match="cliffquant/math window NLL totals drift"):
        heldout_nll.validate_heldout_nll_result(output)


@pytest.mark.parametrize(
    "forged_path",
    (
        r"C:\private\machine",
        r"C:private\machine",
        r"\private\machine",
        r"\\??\C:\private\machine",
        "file:///C:/private/machine",
    ),
)
def test_saved_nll_rejects_unsafe_path_after_report_hash_is_forged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forged_path: str,
) -> None:
    output, report = _write_valid_nll_result(tmp_path, monkeypatch)
    report["runtime"]["platform"] = forged_path
    _rewrite_report_and_sidecar(output, report)

    with pytest.raises(ValueError, match="contains an absolute path"):
        heldout_nll.validate_heldout_nll_result(output)


def test_saved_nll_rejects_impossible_cuda_runtime_and_package_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _write_valid_nll_result(tmp_path, monkeypatch)
    impossible = json.loads(canonical_json_bytes(report))
    impossible["runtime"]["device"] = "cuda:not-a-device"
    _rewrite_report_and_sidecar(output, impossible)
    with pytest.raises(ValueError, match="runtime device drift"):
        heldout_nll.validate_heldout_nll_result(output)

    impossible = json.loads(canonical_json_bytes(report))
    impossible["runtime"]["packages"]["torch"] = None
    _rewrite_report_and_sidecar(output, impossible)
    with pytest.raises(ValueError, match="package inventory drift"):
        heldout_nll.validate_heldout_nll_result(output)


def test_saved_nll_rejects_symlinked_parent_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    output, _report = _write_valid_nll_result(real_root, monkeypatch)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink or reparse-point components"):
        heldout_nll.validate_heldout_nll_result(alias / output.name)


def test_compare_rehashes_checkpoints_after_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_identity = heldout_nll._checkpoint_result_identity
    calls = {"absmax": 0, "cliffquant": 0}

    def drifting_identity(checkpoint: Path, *, role: str) -> dict:
        calls[role] += 1
        identity = real_identity(checkpoint, role=role)
        if role == "absmax" and calls[role] == 2:
            identity = json.loads(canonical_json_bytes(identity))
            identity["model_payload_sha256"] = "0" * 64
        return identity

    monkeypatch.setattr(
        heldout_nll,
        "_checkpoint_result_identity",
        drifting_identity,
    )
    with pytest.raises(ValueError, match="checkpoint changed during held-out NLL evaluation"):
        _write_valid_nll_result(tmp_path, monkeypatch)


def test_saved_nll_live_binding_requires_and_checks_all_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _write_valid_nll_result(tmp_path, monkeypatch)
    manifest_path = tmp_path / "heldout.manifest.json"
    manifest_path.write_bytes(canonical_json_bytes({}))
    validated = heldout_nll.validate_heldout_nll_result(
        output,
        heldout_manifest=manifest_path,
        absmax_checkpoint=tmp_path / "absmax",
        cliffquant_checkpoint=tmp_path / "cliffquant",
    )
    assert validated == report

    with pytest.raises(ValueError, match="requires the manifest and both checkpoints"):
        heldout_nll.validate_heldout_nll_result(
            output,
            heldout_manifest=manifest_path,
        )


def test_saved_nll_rejects_extra_and_symlinked_result_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _report = _write_valid_nll_result(tmp_path, monkeypatch)
    extra = output / "notes.txt"
    extra.write_text("not evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="file coverage mismatch"):
        heldout_nll.validate_heldout_nll_result(output)
    extra.unlink()

    raw_path = output / "heldout-nll.raw.npz"
    outside = tmp_path / "outside.raw.npz"
    raw_path.replace(outside)
    try:
        raw_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")
    with pytest.raises(ValueError, match="regular non-symlink files"):
        heldout_nll.validate_heldout_nll_result(output)


def test_saved_nll_rejects_noncanonical_json_even_with_matching_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _write_valid_nll_result(tmp_path, monkeypatch)
    report_path = output / "heldout-nll.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sidecar_path = output / "heldout-nll.sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["report"]["sha256"] = sha256_file(report_path)
    sidecar["report"]["size_bytes"] = report_path.stat().st_size
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))

    with pytest.raises(ValueError, match="report is not canonical JSON"):
        heldout_nll.validate_heldout_nll_result(output)


def test_compare_cleans_staging_directory_after_builder_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "evidence"
    seen_stage: list[Path] = []

    def fail_after_partial_write(**kwargs):
        stage = Path(kwargs["output_dir"])
        seen_stage.append(stage)
        (stage / "partial.json").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(
        heldout_nll,
        "_compare_heldout_nll_into_directory",
        fail_after_partial_write,
    )
    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        heldout_nll.compare_heldout_nll(
            cliffquant_checkpoint=tmp_path / "cliffquant",
            absmax_checkpoint=tmp_path / "absmax",
            heldout_manifest=tmp_path / "heldout.manifest.json",
            gptqmodel_source=tmp_path / "gptqmodel",
            output_dir=output,
        )

    assert seen_stage and seen_stage[0].parent == output.parent
    assert not output.exists()
    assert not seen_stage[0].exists()
    assert list(tmp_path.glob(".evidence.staging-*")) == []


def test_compare_refuses_existing_destination_without_touching_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        heldout_nll,
        "_compare_heldout_nll_into_directory",
        lambda **_kwargs: pytest.fail("builder must not run"),
    )

    with pytest.raises(FileExistsError, match="must be absent"):
        heldout_nll.compare_heldout_nll(
            cliffquant_checkpoint=tmp_path / "cliffquant",
            absmax_checkpoint=tmp_path / "absmax",
            heldout_manifest=tmp_path / "heldout.manifest.json",
            gptqmodel_source=tmp_path / "gptqmodel",
            output_dir=output,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_compare_preserves_racing_destination_and_cleans_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "evidence"
    staged: list[Path] = []
    report = {"complete": True}

    def fake_build(**kwargs):
        stage = Path(kwargs["output_dir"])
        staged.append(stage)
        (stage / "complete.txt").write_text("complete", encoding="utf-8")
        return report

    def collide(_source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "other-writer.txt").write_text("other", encoding="utf-8")
        raise FileExistsError("injected publication race")

    monkeypatch.setattr(heldout_nll, "_compare_heldout_nll_into_directory", fake_build)
    monkeypatch.setattr(heldout_nll, "validate_heldout_nll_result", lambda _stage: report)
    monkeypatch.setattr(heldout_nll, "_atomic_rename_directory_no_replace", collide)

    with pytest.raises(FileExistsError, match="publication race"):
        heldout_nll.compare_heldout_nll(
            cliffquant_checkpoint=tmp_path / "cliffquant",
            absmax_checkpoint=tmp_path / "absmax",
            heldout_manifest=tmp_path / "heldout.manifest.json",
            gptqmodel_source=tmp_path / "gptqmodel",
            output_dir=output,
        )
    assert (output / "other-writer.txt").read_text(encoding="utf-8") == "other"
    assert staged and not staged[0].exists()


def test_compare_does_not_publish_when_staged_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "evidence"
    staged: list[Path] = []

    def fake_build(**kwargs):
        stage = Path(kwargs["output_dir"])
        staged.append(stage)
        (stage / "incomplete.txt").write_text("incomplete", encoding="utf-8")
        return {"unvalidated": True}

    monkeypatch.setattr(heldout_nll, "_compare_heldout_nll_into_directory", fake_build)
    monkeypatch.setattr(
        heldout_nll,
        "validate_heldout_nll_result",
        lambda _stage: (_ for _ in ()).throw(ValueError("injected staged validation failure")),
    )
    monkeypatch.setattr(
        heldout_nll,
        "_atomic_rename_directory_no_replace",
        lambda *_args: pytest.fail("invalid stage must not be published"),
    )

    with pytest.raises(ValueError, match="injected staged validation failure"):
        heldout_nll.compare_heldout_nll(
            cliffquant_checkpoint=tmp_path / "cliffquant",
            absmax_checkpoint=tmp_path / "absmax",
            heldout_manifest=tmp_path / "heldout.manifest.json",
            gptqmodel_source=tmp_path / "gptqmodel",
            output_dir=output,
        )
    assert not output.exists()
    assert staged and not staged[0].exists()


def test_atomic_directory_publication_never_replaces_existing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stage"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source", encoding="utf-8")
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("destination", encoding="utf-8")

    with pytest.raises(FileExistsError):
        heldout_nll._atomic_rename_directory_no_replace(source, destination)
    assert source.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "destination"


@pytest.mark.parametrize("mutation", ("duplicate", "extra", "missing"))
def test_saved_nll_preflights_exact_raw_npz_entry_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    output, report = _write_valid_nll_result(tmp_path, monkeypatch)
    _rewrite_raw_zip_inventory(output, mutation)
    _rehash_raw_and_outer_evidence(output, report)

    with pytest.raises(
        ValueError,
        match="entry inventory has duplicate, extra, or missing entries",
    ):
        heldout_nll.validate_heldout_nll_result(output)


def test_saved_nll_rejects_huge_declared_raw_npz_member_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _write_valid_nll_result(tmp_path, monkeypatch)
    raw_path = output / "heldout-nll.raw.npz"
    payload = bytearray(raw_path.read_bytes())
    central_header = payload.find(b"PK\x01\x02")
    assert central_header >= 0
    payload[central_header + 24 : central_header + 28] = struct.pack("<I", 1_000_000_000)
    raw_path.write_bytes(payload)
    _rehash_raw_and_outer_evidence(output, report)
    monkeypatch.setattr(
        heldout_nll.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("np.load must not run before ZIP preflight"),
    )

    with pytest.raises(ValueError, match="raw NPZ entry metadata drift"):
        heldout_nll.validate_heldout_nll_result(output)


def test_saved_nll_leaf_check_rejects_mocked_windows_reparse_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _report = _write_valid_nll_result(tmp_path, monkeypatch)
    original = heldout_nll._is_symlink_or_reparse_point

    def mark_raw_as_reparse(path: Path) -> bool:
        return Path(path).name == "heldout-nll.raw.npz" or original(Path(path))

    monkeypatch.setattr(heldout_nll, "_is_symlink_or_reparse_point", mark_raw_as_reparse)
    with pytest.raises(ValueError, match="reparse points are forbidden"):
        heldout_nll.validate_heldout_nll_result(output)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("cublas_workspace_config", None),
        ("cudnn_allow_tf32", True),
        ("cudnn_benchmark", True),
        ("cudnn_deterministic", False),
        ("deterministic_algorithms", False),
        ("deterministic_algorithms_warn_only", True),
        ("float32_matmul_precision", "high"),
        ("matmul_allow_tf32", True),
    ),
)
def test_runtime_rejects_each_deterministic_cuda_contract_drift(
    field: str,
    forged_value: object,
) -> None:
    runtime = {
        **_fake_runtime_metadata(
            device="cuda:0",
            package_names=tuple(heldout_nll._NLL_RUNTIME_PACKAGE_VERSIONS),
        ),
        "batch_size": 1,
        "teacher_forcing": True,
    }
    runtime["accelerator"][field] = forged_value

    with pytest.raises(ValueError, match="deterministic CUDA contract drift"):
        heldout_nll._validate_runtime(runtime)


def test_evaluate_checkpoint_uses_shifted_targets_masks_batches_and_float64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    class NextTokenModel:
        def eval(self) -> None:
            return None

        def __call__(self, *, input_ids, attention_mask, **_kwargs):
            calls.append(
                (
                    input_ids.detach().cpu().numpy().copy(),
                    attention_mask.detach().cpu().numpy().copy(),
                )
            )
            logits = torch.full(
                (*input_ids.shape, 16),
                -3.0,
                dtype=torch.float16,
                device=input_ids.device,
            )
            logits[:, :-1, :].scatter_(2, input_ids[:, 1:].unsqueeze(-1), 3.0)
            return SimpleNamespace(logits=logits)

    wrapper = SimpleNamespace(model=NextTokenModel())
    monkeypatch.setattr(
        heldout_nll,
        "load_fresh_gptq_torch",
        lambda **_kwargs: ({"contract": "test"}, wrapper),
    )
    input_ids = np.asarray(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=np.int64,
    )
    attention_mask = np.asarray(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    tokens = {
        environment: (input_ids.copy(), attention_mask.copy())
        for environment in FROZEN_ENVIRONMENTS
    }

    metrics, arrays = heldout_nll._evaluate_checkpoint(
        checkpoint=tmp_path / "checkpoint",
        gptqmodel_source=tmp_path / "gptqmodel",
        tokens=tokens,
        device="cpu",
        batch_size=1,
    )

    assert len(calls) == len(FROZEN_ENVIRONMENTS) * 2
    assert all(ids.shape == (1, 4) and mask.shape == (1, 4) for ids, mask in calls)
    for environment in FROZEN_ENVIRONMENTS:
        prefix = f"{environment}__"
        token_nll = arrays[f"{prefix}token_nll"]
        target_mask = arrays[f"{prefix}target_mask"]
        window_sums = arrays[f"{prefix}window_nll_sum"]
        window_counts = arrays[f"{prefix}window_target_tokens"]
        assert token_nll.dtype == np.dtype("<f8")
        assert target_mask.dtype == np.dtype("|u1")
        assert window_sums.dtype == np.dtype("<f8")
        assert window_counts.dtype == np.dtype("<i8")
        assert np.array_equal(target_mask, attention_mask[:, 1:])
        assert np.array_equal(window_counts, np.asarray([2, 1], dtype=np.int64))
        assert np.all(token_nll < 0.1)
        assert np.array_equal(
            window_sums,
            np.sum(token_nll * target_mask, axis=1, dtype=np.float64),
        )
        assert metrics[environment]["windows"] == 2
        assert metrics[environment]["target_tokens"] == 3


def test_evaluate_checkpoint_rejects_nonfinite_token_losses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class NonfiniteModel:
        def eval(self) -> None:
            return None

        def __call__(self, *, input_ids, **_kwargs):
            return SimpleNamespace(
                logits=torch.full(
                    (*input_ids.shape, 16),
                    float("nan"),
                    dtype=torch.float32,
                    device=input_ids.device,
                )
            )

    monkeypatch.setattr(
        heldout_nll,
        "load_fresh_gptq_torch",
        lambda **_kwargs: ({"contract": "test"}, SimpleNamespace(model=NonfiniteModel())),
    )
    tokens = {
        environment: (
            np.asarray([[1, 2, 3]], dtype=np.int64),
            np.ones((1, 3), dtype=np.uint8),
        )
        for environment in FROZEN_ENVIRONMENTS
    }

    with pytest.raises(ValueError, match="non-finite held-out token NLL"):
        heldout_nll._evaluate_checkpoint(
            checkpoint=tmp_path / "checkpoint",
            gptqmodel_source=tmp_path / "gptqmodel",
            tokens=tokens,
            device="cpu",
            batch_size=1,
        )
