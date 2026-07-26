from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from cliffquant.clean_environment import (
    _build_environment_report,
    _build_source_report,
    _package_map,
    _wheelhouse_identity,
    run_clean_verification,
)
from cliffquant.full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BITS,
    GROUP_SIZE,
)
from cliffquant.full_model_verify import (
    CLEAN_BUILD_SOURCE_SCHEMA,
    CLEAN_ENVIRONMENT_SCHEMA,
    HUB_RELEASE_ENVELOPE_FILES,
    RUNTIME_VERIFICATION_SCHEMA,
    VERIFICATION_LOCK_SHA256,
    VERIFICATION_SCHEMA,
    _check_generation,
    _verifier_identity,
    _write_canonical_report,
    describe_exported_checkpoint,
    load_clean_environment_identity,
    load_verification_report,
    run_image_generation_smoke,
    run_text_generation_smoke,
    validate_clean_build_source_identity,
    validate_clean_environment_identity,
    validate_exported_checkpoint_identity,
)
from cliffquant.gptq_pack import INT4_LOGICAL_ZERO
from cliffquant.gptqmodel_export import EXPORT_SCHEMA
from cliffquant.model_snapshot import MODEL_SNAPSHOT_SCHEMA
from cliffquant.provenance import canonical_json_bytes, sha256_bytes, sha256_file
from cliffquant.qwen35_modules import (
    QWEN35_08B_EXPECTED_QUANTIZED_MODULES,
    QWEN35_GPTQMODEL_TREE_REVISION,
)


def _descriptor(name: str, marker: str = "a") -> dict[str, Any]:
    return {
        "file": name,
        "sha256": marker * 64,
        "size_bytes": 1,
    }


def _runtime(device: str = "cpu") -> dict[str, Any]:
    packages = {
        "accelerate": "1.14.0",
        "datasets": "5.0.0",
        "gptqmodel": "7.3.4",
        "huggingface-hub": "1.24.0",
        "numpy": "2.2.6",
        "pillow": "12.3.0",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
        "torch": "2.11.0+cu128",
        "torchao": "0.17.0",
        "torchvision": "0.26.0+cu128",
        "transformers": "5.14.1",
    }
    return {
        "device": device,
        "implementation": "CPython",
        "machine": "AMD64",
        "numpy": "2.2.6",
        "numpy_build": {},
        "packages": packages,
        "platform": "Windows-test",
        "python": "3.11.15",
        "python_executable": "python.exe",
    }


def _snapshot() -> dict[str, Any]:
    payload = {
        "config": _descriptor("config.json", "1"),
        "id": BASE_MODEL_ID,
        "index": _descriptor("model.safetensors.index.json", "2"),
        "revision": BASE_MODEL_REVISION,
        "schema": MODEL_SNAPSHOT_SCHEMA,
        "shards": [_descriptor("model-00001-of-00001.safetensors", "3")],
        "tensor_count": 488,
    }
    return {
        **payload,
        "fingerprint_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _replay_pair() -> dict[str, Any]:
    def phase(name: str, marker: str) -> dict[str, Any]:
        return {
            "accepted_source_rows": 4,
            "cache": {"page_count": 4, "pages_sha256": marker * 64},
            "manifest": _descriptor(f"{name}.manifest.json", marker),
            "phase": name,
            "schema": "cliffquant.corpus-replay.v1",
            "sidecar": _descriptor(f"{name}.sha256.json", marker),
            "token_archive": _descriptor(f"{name}.tokens.npz", marker),
            "tokenizer_sha256": "d" * 64,
            "used_rendered_records": 4,
            "window_count": 128 if name == "calibration" else 256,
        }

    phases = {
        "calibration": phase("calibration", "4"),
        "heldout": phase("heldout", "5"),
    }
    return {
        **phases,
        "schema": "cliffquant.corpus-replay-pair.v1",
        "sha256": sha256_bytes(canonical_json_bytes(phases)),
    }


def _write_export(root: Path) -> None:
    root.mkdir()
    snapshot = _snapshot()
    replay = _replay_pair()
    scale_sha256 = "6" * 64
    quantize_config = {
        "format": "gptq",
        "meta": {
            "cliffquant": {
                "base_model": BASE_MODEL_ID,
                "base_revision": BASE_MODEL_REVISION,
                "base_snapshot_sha256": snapshot["fingerprint_sha256"],
                "corpus_replay_sha256": replay["sha256"],
                "gptqmodel_revision": QWEN35_GPTQMODEL_TREE_REVISION,
                "policy": "cliffquant_minimax",
                "scale_run_sha256": scale_sha256,
            }
        },
        "quant_method": "gptq",
    }
    (root / "quantize_config.json").write_bytes(canonical_json_bytes(quantize_config))
    (root / "model.safetensors").write_bytes(b"packed")
    (root / "processor_config.json").write_bytes(canonical_json_bytes({}))
    (root / "tokenizer_config.json").write_bytes(canonical_json_bytes({}))
    files = [
        {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.iterdir())
    ]
    manifest = {
        "base_model": {
            "id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
            "source": snapshot,
        },
        "corpus_replay": replay,
        "files": files,
        "gptq": {
            "backend": "GPTQ_TORCH",
            "bits": BITS,
            "desc_act": False,
            "format": "gptq",
            "group_size": GROUP_SIZE,
            "method": "gptq",
            "pack_dtype": "int32",
            "sym": True,
            "zero": INT4_LOGICAL_ZERO,
        },
        "gptqmodel": {
            "revision": QWEN35_GPTQMODEL_TREE_REVISION,
            "version": "7.3.4",
        },
        "modules": [],
        "policy": "cliffquant_minimax",
        "runtime": {},
        "scale_run": {
            "manifest_sha256": scale_sha256,
            "path": "manifest.json",
        },
        "schema": EXPORT_SCHEMA,
        "status": "packed-unverified",
    }
    (root / "cliffquant_export.json").write_bytes(canonical_json_bytes(manifest))


def _write_hub_release_envelope(root: Path) -> None:
    assets = root / "assets"
    assets.mkdir()
    (root / "README.md").write_text("# Hub model card\n", encoding="utf-8")
    (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    (root / ".gitattributes").write_text("*.safetensors filter=lfs\n", encoding="utf-8")
    (assets / "proxy-policy-comparison.png").write_bytes(b"proxy-graph")
    (assets / "heldout-nll-comparison.png").write_bytes(b"nll-graph")


def _structural_report(checkpoint: Path) -> dict[str, Any]:
    identity = describe_exported_checkpoint(checkpoint)
    live_files = {item["file"]: item for item in identity["checkpoint_files"]}
    modules = [
        {
            "dequant_values_checked": 8,
            "g_idx_sha256": "7" * 64,
            "module": f"model.layers.{index:03d}.linear",
            "qweight_sha256": "8" * 64,
            "qzeros_sha256": "9" * 64,
            "scales_sha256": "a" * 64,
            "signed_codes_checked": 16,
        }
        for index in range(QWEN35_08B_EXPECTED_QUANTIZED_MODULES)
    ]
    preserved = [{"key": "model.embed_tokens.weight", "sha256": "b" * 64}]
    return {
        "auxiliary_files": [
            {
                "base_sha256": live_files[name]["sha256"],
                "file": name,
                "output_sha256": live_files[name]["sha256"],
            }
            for name in ("processor_config.json", "tokenizer_config.json")
        ],
        "base_checkpoint_files": [_descriptor("base.safetensors", "c")],
        "checkpoint_files": [live_files["model.safetensors"]],
        "checkpoint_identity": identity,
        "corpus_replay": identity["corpus_replay"],
        "export_manifest_sha256": identity["export_manifest"]["sha256"],
        "modules": modules,
        "policy": identity["policy"],
        "preserved_non_target_tensor_count": len(preserved),
        "preserved_non_target_tensors": preserved,
        "runtime": _runtime(),
        "scale_run_sha256": identity["scale_run"]["manifest_sha256"],
        "schema": VERIFICATION_SCHEMA,
        "status": "pass",
        "summary": {
            "dense_target_weights_present": 0,
            "module_buffer_sets": len(modules),
            "non_target_tensors_exact": len(preserved),
            "signed_codes_checked": sum(item["signed_codes_checked"] for item in modules),
        },
        "verifier": _verifier_identity(),
    }


def _patch_smoke_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import cliffquant.full_model_verify as verification

    class Contract:
        revision = QWEN35_GPTQMODEL_TREE_REVISION
        version = "7.3.4"

    class Tokenizer:
        decode = None

        def __call__(self, *_args: Any, **_kwargs: Any) -> dict[str, object]:
            return {"input_ids": object()}

    class Processor:
        tokenizer = Tokenizer()

        def __call__(self, *_args: Any, **_kwargs: Any) -> dict[str, object]:
            return {"input_ids": object(), "pixel_values": object()}

    wrapper = types.SimpleNamespace(tokenizer=Tokenizer(), processor=Processor())
    monkeypatch.setattr(
        verification,
        "load_fresh_gptq_torch",
        lambda **_kwargs: (Contract(), wrapper),
    )
    monkeypatch.setattr(verification, "_move_inputs", lambda values, _device: dict(values))
    monkeypatch.setattr(verification, "_generate_once", lambda *_args, **_kwargs: (None, []))
    monkeypatch.setattr(
        verification,
        "_check_generation",
        lambda *_args: {
            "generated_tokens": 2,
            "score_steps": 2,
            "sequence_sha256": "7" * 64,
            "scores_sha256": ["8" * 64, "9" * 64],
        },
    )
    monkeypatch.setattr(
        verification,
        "_verification_runtime",
        _runtime,
    )


def test_generation_evidence_counts_only_new_tokens() -> None:
    torch = pytest.importorskip("torch")
    sequence = torch.tensor([[10, 11, 12, 13, 14, 15, 16]])
    scores = [
        torch.tensor([[0.25, 0.75]]),
        torch.tensor([[0.50, 0.50]]),
    ]

    evidence = _check_generation((sequence, scores), (sequence.clone(), list(scores)))

    assert evidence["generated_tokens"] == 2
    assert evidence["score_steps"] == 2


def _write_clean_environment_report(root: Path) -> tuple[Path, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    cliffquant_wheel = root / "cliffquant-0.1.0.whl"
    gptqmodel_wheel = root / "gptqmodel-7.3.4.whl"
    lock = root / "verification-cu128.txt"
    for path in (cliffquant_wheel, gptqmodel_wheel):
        path.write_bytes(b"x")
    repository_lock = (
        Path(__file__).resolve().parents[1] / "requirements" / "verification-cu128.txt"
    )
    lock.write_bytes(repository_lock.read_bytes())
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "Numpy-2.2.6-cp311-cp311-win_amd64.whl").write_bytes(b"numpy")
    (wheelhouse / "aiohttp-3.14.3-cp311-cp311-win_amd64.whl").write_bytes(b"aiohttp")
    hashed_lock = root / "verification-cu128-hashed.txt"
    hashed_lock.write_text(
        "numpy==2.2.6 --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    dependencies = _wheelhouse_identity(
        wheelhouse,
        hashed_lock=hashed_lock,
        hashed_lock_logical_name="requirements/verification-cu128-hashed.txt",
    )
    source_build = _build_source_report(
        cliffquant_identity={"commit": "1" * 40, "tree": "2" * 40},
        cliffquant_wheel=cliffquant_wheel,
        dependency_identity=dependencies,
        gptqmodel_identity={
            "commit": QWEN35_GPTQMODEL_TREE_REVISION,
            "tree": "4" * 40,
        },
        gptqmodel_wheel=gptqmodel_wheel,
    )
    locked = [
        line.strip()
        for line in repository_lock.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("--")
    ]
    packages = sorted(
        [
            *locked,
            "cliffquant==0.1.0",
            "GPTQModel==7.3.4",
            "pip==26.1.2",
            "setuptools==81.0.0",
            "wheel==0.47.0",
        ],
        key=str.casefold,
    )
    report = _build_environment_report(
        cliffquant_wheel=cliffquant_wheel,
        gptqmodel_wheel=gptqmodel_wheel,
        lock_path=lock,
        packages=packages,
        pip_check="No broken requirements found.",
        python_identity={"implementation": "CPython", "version": "3.11.15"},
        source_build=source_build,
    )
    report_path = root / "clean-environment.json"
    _write_canonical_report(report, report_path)
    return report_path, report


def _rehash_report(report: dict[str, Any]) -> None:
    payload = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = sha256_bytes(canonical_json_bytes(payload))


def test_checkpoint_identity_binds_every_exported_byte(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)

    identity = describe_exported_checkpoint(checkpoint)

    assert identity["export_manifest"]["sha256"] == sha256_file(
        checkpoint / "cliffquant_export.json"
    )
    assert (
        identity["base_model"]["source"]["fingerprint_sha256"] == _snapshot()["fingerprint_sha256"]
    )
    assert identity["corpus_replay"]["sha256"] == _replay_pair()["sha256"]
    assert [item["file"] for item in identity["checkpoint_files"]] == [
        "cliffquant_export.json",
        "model.safetensors",
        "processor_config.json",
        "quantize_config.json",
        "tokenizer_config.json",
    ]
    assert validate_exported_checkpoint_identity(identity) == identity

    (checkpoint / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="export file hash mismatch"):
        describe_exported_checkpoint(checkpoint)


def test_checkpoint_identity_rejects_export_manifest_symlink_escape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    manifest = checkpoint / "cliffquant_export.json"
    outside = tmp_path / "outside-export.json"
    manifest.replace(outside)
    try:
        manifest.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="regular file inside the checkpoint"):
        describe_exported_checkpoint(checkpoint)


def test_checkpoint_identity_rejects_symlinked_payload_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    model = checkpoint / "model.safetensors"
    outside = tmp_path / "outside-model.safetensors"
    model.replace(outside)
    try:
        model.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="regular non-symlink file"):
        describe_exported_checkpoint(checkpoint)


def test_hub_release_envelope_preserves_immutable_payload_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    report = _structural_report(checkpoint)
    strict_identity = report["checkpoint_identity"]
    report_path = tmp_path / "structural.json"
    _write_canonical_report(report, report_path)
    _write_hub_release_envelope(checkpoint)

    with pytest.raises(ValueError, match="export file coverage mismatch"):
        describe_exported_checkpoint(checkpoint)

    hub_identity = describe_exported_checkpoint(checkpoint, hub_release=True)
    assert hub_identity == strict_identity

    result = load_verification_report(
        report_path,
        checkpoint_dir=checkpoint,
        hub_release=True,
    )
    assert result["status"] == "model-payload-verified"
    assert result["report"] == report
    assert result["publication_envelope"] == {
        "files": list(HUB_RELEASE_ENVELOPE_FILES),
        "identity_scope": "model-payload-only",
        "path_policy": "required-regular-non-symlink-in-root",
        "publication_metadata_bytes_verified": False,
        "status": "path-policy-verified",
    }

    (checkpoint / "README.md").write_text("# Changed Hub model card\n", encoding="utf-8")
    assert describe_exported_checkpoint(checkpoint, hub_release=True) == strict_identity


@pytest.mark.parametrize("missing_file", HUB_RELEASE_ENVELOPE_FILES)
def test_hub_release_requires_every_envelope_file(
    tmp_path: Path,
    missing_file: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    _write_hub_release_envelope(checkpoint)
    (checkpoint / missing_file).unlink()

    with pytest.raises(FileNotFoundError, match="required Hub publication envelope file"):
        describe_exported_checkpoint(checkpoint, hub_release=True)


def test_hub_release_rejects_every_other_extra_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    _write_hub_release_envelope(checkpoint)
    (checkpoint / "untracked-notes.txt").write_text("not allowlisted\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"extra=\['untracked-notes.txt'\]"):
        describe_exported_checkpoint(checkpoint, hub_release=True)


def test_hub_release_rejects_symlinked_envelope_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    _write_hub_release_envelope(checkpoint)
    readme = checkpoint / "README.md"
    outside = tmp_path / "outside-readme.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    readme.unlink()
    try:
        readme.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="regular non-symlink file"):
        describe_exported_checkpoint(checkpoint, hub_release=True)


def test_report_parser_exposes_explicit_hub_release_flag() -> None:
    from scripts.verify_full_model_gptq import _parser

    args = _parser().parse_args(
        [
            "report",
            "--report",
            "structural.json",
            "--checkpoint",
            "hub-stage",
            "--hub-release",
        ]
    )

    assert args.hub_release is True


def test_structural_report_sidecar_preserves_all_identity_bindings(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    report = _structural_report(checkpoint)
    report_path = tmp_path / "structural.json"
    _write_canonical_report(report, report_path)

    assert load_verification_report(report_path, checkpoint_dir=checkpoint) == report

    drifted = {**report, "policy": "absmax"}
    drifted_path = tmp_path / "structural-drifted.json"
    _write_canonical_report(drifted, drifted_path)
    with pytest.raises(ValueError, match="checkpoint binding drift"):
        load_verification_report(drifted_path, checkpoint_dir=checkpoint)


def test_report_validation_requires_live_checkpoint_and_current_verifier(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    report = _structural_report(checkpoint)

    forged_verifier = json.loads(canonical_json_bytes(report["verifier"]))
    forged_verifier["files"][0]["sha256"] = "0" * 64
    verifier_payload = {
        "files": forged_verifier["files"],
        "schema": forged_verifier["schema"],
    }
    forged_verifier["sha256"] = sha256_bytes(canonical_json_bytes(verifier_payload))
    forged = {**report, "verifier": forged_verifier}
    forged_path = tmp_path / "forged-verifier.json"
    _write_canonical_report(forged, forged_path)
    with pytest.raises(ValueError, match="different verifier source bytes"):
        load_verification_report(forged_path, checkpoint_dir=checkpoint)

    valid_path = tmp_path / "valid-structural.json"
    _write_canonical_report(report, valid_path)
    (checkpoint / "model.safetensors").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="export file hash mismatch"):
        load_verification_report(valid_path, checkpoint_dir=checkpoint)


def test_structural_report_requires_complete_evidence_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    report = _structural_report(checkpoint)
    report.pop("modules")
    report_path = tmp_path / "missing-modules.json"
    _write_canonical_report(report, report_path)

    with pytest.raises(ValueError, match="report fields drift"):
        load_verification_report(report_path, checkpoint_dir=checkpoint)


def test_text_smoke_persists_bound_canonical_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    _patch_smoke_runtime(monkeypatch)
    report_path = tmp_path / "evidence" / "text-smoke.json"

    report = run_text_generation_smoke(
        checkpoint_dir=checkpoint,
        gptqmodel_source=tmp_path,
        report_path=report_path,
        prompt="test prompt",
        device="cpu",
        max_new_tokens=2,
    )

    assert report["schema"] == RUNTIME_VERIFICATION_SCHEMA
    assert report["checkpoint_identity"] == describe_exported_checkpoint(checkpoint)
    assert report_path.read_bytes() == canonical_json_bytes(report)
    assert load_verification_report(report_path, checkpoint_dir=checkpoint) == report
    with pytest.raises(ValueError, match="not bound to a clean-environment"):
        load_verification_report(
            report_path,
            checkpoint_dir=checkpoint,
            require_clean_environment=True,
        )
    sidecar = json.loads((report_path.parent / "text-smoke.sha256.json").read_bytes())
    assert sidecar["report"]["sha256"] == sha256_file(report_path)

    run_text_generation_smoke(
        checkpoint_dir=checkpoint,
        gptqmodel_source=tmp_path,
        report_path=report_path,
        prompt="test prompt",
        device="cpu",
        max_new_tokens=2,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_text_generation_smoke(
            checkpoint_dir=checkpoint,
            gptqmodel_source=tmp_path,
            report_path=report_path,
            prompt="different prompt",
            device="cpu",
            max_new_tokens=2,
        )


def test_runtime_report_requires_generation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    _patch_smoke_runtime(monkeypatch)
    generated_path = tmp_path / "generated.json"
    report = run_text_generation_smoke(
        checkpoint_dir=checkpoint,
        gptqmodel_source=tmp_path,
        report_path=generated_path,
        prompt="test",
        device="cpu",
        max_new_tokens=2,
    )
    mismatched = {**report, "generated_tokens": report["score_steps"] + 1}
    mismatched_path = tmp_path / "mismatched-generation-count.json"
    _write_canonical_report(mismatched, mismatched_path)
    with pytest.raises(ValueError, match="runtime smoke-test evidence drift"):
        load_verification_report(mismatched_path, checkpoint_dir=checkpoint)

    report.pop("scores_sha256")
    missing_path = tmp_path / "missing-generation-evidence.json"
    _write_canonical_report(report, missing_path)

    with pytest.raises(ValueError, match="report fields drift"):
        load_verification_report(missing_path, checkpoint_dir=checkpoint)


def test_image_smoke_persists_input_and_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    _patch_smoke_runtime(monkeypatch)
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"image")

    class OpenImage:
        def __enter__(self) -> OpenImage:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def convert(self, _mode: str) -> OpenImage:
            return self

    fake_pillow = types.ModuleType("PIL")
    fake_pillow.Image = types.SimpleNamespace(open=lambda _path: OpenImage())
    monkeypatch.setitem(sys.modules, "PIL", fake_pillow)
    report_path = tmp_path / "evidence" / "image-smoke.json"

    report = run_image_generation_smoke(
        checkpoint_dir=checkpoint,
        gptqmodel_source=tmp_path,
        image_path=image_path,
        report_path=report_path,
        prompt="describe",
        device="cpu",
        max_new_tokens=2,
    )

    assert report["image"] == {
        "file": "fixture.png",
        "sha256": sha256_file(image_path),
        "size_bytes": image_path.stat().st_size,
    }
    assert load_verification_report(report_path, checkpoint_dir=checkpoint) == report


def test_smoke_report_embeds_validated_clean_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    _patch_smoke_runtime(monkeypatch)
    environment_path, environment = _write_clean_environment_report(tmp_path)
    report_path = tmp_path / "evidence" / "text-smoke.json"

    report = run_text_generation_smoke(
        checkpoint_dir=checkpoint,
        gptqmodel_source=tmp_path,
        report_path=report_path,
        environment_report=environment_path,
        prompt="test",
        device="cpu",
    )

    assert report["clean_environment"] == environment
    assert (
        load_verification_report(
            report_path,
            checkpoint_dir=checkpoint,
            require_clean_environment=True,
        )
        == report
    )


def test_reports_cannot_mutate_immutable_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    _patch_smoke_runtime(monkeypatch)

    with pytest.raises(ValueError, match="outside the immutable checkpoint"):
        run_text_generation_smoke(
            checkpoint_dir=checkpoint,
            gptqmodel_source=tmp_path,
            report_path=checkpoint / "text-smoke.json",
            prompt="test",
            device="cpu",
        )


def test_clean_environment_report_is_self_hashing_and_sidecar_bound(tmp_path: Path) -> None:
    report_path, report = _write_clean_environment_report(tmp_path)

    assert report["schema"] == CLEAN_ENVIRONMENT_SCHEMA
    assert report["source_build"]["schema"] == CLEAN_BUILD_SOURCE_SCHEMA
    assert validate_clean_build_source_identity(report["source_build"]) == report["source_build"]
    assert report["cliffquant_wheel"] == report["source_build"]["cliffquant"]["wheel"]
    assert report["gptqmodel_wheel"] == report["source_build"]["gptqmodel"]["wheel"]
    assert load_clean_environment_identity(report_path) == report
    sidecar_path = tmp_path / "clean-environment.sha256.json"
    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar["report"]["sha256"] = "0" * 64
    sidecar_path.write_bytes(canonical_json_bytes(sidecar))
    with pytest.raises(ValueError, match="sidecar mismatch"):
        load_clean_environment_identity(report_path)


def test_clean_environment_rejects_source_build_identity_drift(tmp_path: Path) -> None:
    cases = (
        ("commit", "GPTQModel clean-build source identity drift"),
        ("tree", "CliffQuant clean-build Git identity drift"),
        ("wheel", "source-build binding drift"),
        ("wheelhouse", "wheelhouse aggregate mismatch"),
        ("source_hash", "source report fingerprint mismatch"),
    )
    for case, message in cases:
        _path, original = _write_clean_environment_report(tmp_path / case)
        report = json.loads(canonical_json_bytes(original))
        source = report["source_build"]
        if case == "commit":
            source["gptqmodel"]["commit"] = "0" * 40
            _rehash_report(source)
        elif case == "tree":
            source["cliffquant"]["tree"] = "invalid"
            _rehash_report(source)
        elif case == "wheel":
            source["cliffquant"]["wheel"]["sha256"] = "0" * 64
            _rehash_report(source)
        elif case == "wheelhouse":
            source["dependencies"]["files"][0]["sha256"] = "0" * 64
            _rehash_report(source)
        else:
            source["sha256"] = "0" * 64
        _rehash_report(report)

        with pytest.raises(ValueError, match=message):
            validate_clean_environment_identity(report)


def test_clean_environment_requires_exact_frozen_package_inventory(tmp_path: Path) -> None:
    _report_path, report = _write_clean_environment_report(tmp_path)
    report["packages"] = [
        item for item in report["packages"] if not item.casefold().startswith("requests==")
    ]
    payload = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = sha256_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="differs from the frozen lock"):
        validate_clean_environment_identity(report)


def test_clean_environment_must_match_runtime_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_export(checkpoint)
    _patch_smoke_runtime(monkeypatch)
    environment_path, _environment = _write_clean_environment_report(tmp_path)
    generated_path = tmp_path / "generated-clean.json"
    report = run_text_generation_smoke(
        checkpoint_dir=checkpoint,
        gptqmodel_source=tmp_path,
        report_path=generated_path,
        environment_report=environment_path,
        prompt="test",
        device="cpu",
    )
    report["runtime"]["packages"]["transformers"] = "0.0.0"
    mismatch_path = tmp_path / "runtime-mismatch.json"
    _write_canonical_report(report, mismatch_path)

    with pytest.raises(ValueError, match="differs from clean environment"):
        load_verification_report(
            mismatch_path,
            checkpoint_dir=checkpoint,
            require_clean_environment=True,
        )


def test_clean_runner_rejects_preexisting_workspace_before_commands(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "output"
    workspace = tmp_path / "already-there"
    workspace.mkdir()
    python = tmp_path / "python.exe"
    lock = tmp_path / "lock.txt"
    python.write_bytes(b"x")
    lock.write_bytes(b"x")

    with pytest.raises(FileExistsError, match="must not exist"):
        run_clean_verification(
            checkpoint_dir=checkpoint,
            output_directory=output,
            work_directory=workspace,
            python_executable=python,
            lock_path=lock,
            prompt="test",
            device="cpu",
            max_new_tokens=1,
        )


def test_clean_runtime_lock_contains_only_exact_versions() -> None:
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "requirements" / "verification-cu128.txt"
    lines = [
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("--")
    ]

    assert sha256_file(lock_path) == VERIFICATION_LOCK_SHA256
    assert lines
    assert all("==" in line and " @ " not in line for line in lines)
    assert "torch==2.11.0+cu128" in lines
    assert "torchao==0.17.0" in lines
    assert "transformers==5.14.1" in lines
    package_map = _package_map(lines)
    assert package_map["torch"] == "2.11.0+cu128"
    assert package_map["huggingface-hub"] == "1.24.0"
