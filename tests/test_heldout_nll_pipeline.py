from __future__ import annotations

import json
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
from cliffquant.provenance import canonical_json_bytes, sha256_bytes, sha256_file
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
                    "bits": 4,
                    "format": "gptq",
                    "group_size": 128,
                    "method": "gptq",
                    "sym": True,
                    "zero": 8,
                },
                "gptqmodel": {"revision": QWEN35_GPTQMODEL_TREE_REVISION},
                "policy": policy,
                "scale_run": {"manifest_sha256": scale_run_sha256},
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
        metrics = {
            environment: {
                "mean_nll": mean,
                "target_tokens": 1,
                "total_nll": mean,
                "windows": 1,
            }
            for environment in FROZEN_ENVIRONMENTS
        }
        metrics["macro_mean_nll"] = mean
        arrays = {
            f"{environment}__token_nll": np.asarray([[mean]], dtype=np.float64)
            for environment in FROZEN_ENVIRONMENTS
        }
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


def test_paired_nll_rejects_swapped_or_identical_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_nll_export_identity_rejects_symlinked_payload_file(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="regular non-symlink files"):
        heldout_nll._export_identity(
            checkpoint,
            expected_policy="cliffquant_minimax",
        )


def test_paired_nll_rejects_relabelled_identical_model_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
