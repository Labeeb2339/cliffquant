from __future__ import annotations

import copy
import json
import os
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import cliffquant.publication_release as release
from cliffquant.provenance import canonical_json_bytes, sha256_bytes

_TEST_BINDINGS = {
    "evidence": {
        "gate_recomputed": True,
        "identity": "1" * 64,
    }
}


def _fake_nll_report() -> dict[str, Any]:
    accelerator = {
        "cublas_workspace_config": release.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "float32_matmul_precision": "highest",
        "matmul_allow_tf32": False,
    }
    return {
        "checkpoints": {"absmax": {"sha256": "1" * 64}},
        "comparison": {"macro": 0.0},
        "corpus_replay": {"sha256": "2" * 64},
        "evaluator": {"sha256": "3" * 64},
        "gates": {"publishable_model_nll_gate": True},
        "heldout": {"sha256": "4" * 64},
        "policies": {"absmax": {"macro_mean_nll": 1.0}},
        "raw_evidence": {
            "arrays": {
                "absmax_token_nll": {
                    "dtype": "<f8",
                    "sha256": "5" * 64,
                    "shape": [1, 2],
                }
            },
            "file": "heldout-nll.raw.npz",
            "sha256": "6" * 64,
            "size_bytes": 100,
        },
        "runtime": {
            "accelerator": accelerator,
            "batch_size": 1,
            "dequantized_weight_cache": True,
            "device": "cuda:0",
            "teacher_forcing": True,
        },
        "schema": "cliffquant.heldout-nll.v1",
        "status": "pass",
    }


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(data, zlib.crc32(kind)) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _rgba_png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixels = zlib.compress(b"\x00\x20\x40\x60\xff")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


def _proxy_policy(
    *,
    policy: str,
    calibration_loss: float,
    heldout_loss: float,
) -> dict[str, Any]:
    codes = [0, 0]
    method = {
        "absmax": "range-derived-absmax-nearest-positive-fp16",
        "cliffquant_minimax": "exact-breakpoint-four-environment-minimax",
        "pooled_wmse": "exact-breakpoint-pooled-wmse",
    }[policy]
    selection: dict[str, Any]
    if policy == "absmax":
        selection = {
            "method": method,
            "objective": "none",
            "rounding_tie": "lower-fp16-bit-pattern",
        }
    else:
        selection = {
            "method": method,
            "objective": calibration_loss,
            "solver_provenance": {},
            "solver_returned_objective": calibration_loss,
        }
    return {
        "calibration_loss_by_environment": {"general": calibration_loss},
        "codes": codes,
        "codes_sha256": release.canonical_array_sha256(release.np.asarray(codes, dtype="<i2")),
        "heldout_loss_by_environment": {"general": heldout_loss},
        "heldout_max_loss": heldout_loss,
        "runtime_seconds": 0.0,
        "scale": release.bits_to_scale(0x3C00),
        "scale_bits": 0x3C00,
        "selection": selection,
    }


def _small_proxy_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any]]:
    monkeypatch.setattr(release, "EXPERIMENT_001_ENVIRONMENTS", ("general",))
    monkeypatch.setattr(release, "EXPERIMENT_001_GROUP_SIZE", 2)
    monkeypatch.setattr(release, "EXPERIMENT_001_SAMPLE_SIZE", 2)
    monkeypatch.setattr(release, "QWEN35_08B_EXPECTED_QUANTIZED_MODULES", 2)
    root = tmp_path / "proxy"
    root.mkdir()

    entries: list[dict[str, Any]] = []
    for index in range(2):
        identity = {
            "branch": "mlp.down_proj",
            "group": 0,
            "input_start": 0,
            "input_stop": 2,
            "layer": index,
            "module": f"model.language_model.layers.{index}.mlp.down_proj",
            "row": 0,
            "sample_index": index,
        }
        entries.append(
            {
                **identity,
                "group_id": sha256_bytes(canonical_json_bytes(identity)),
                "weight_sha256": f"{index + 3:064x}",
            }
        )
    manifest = {
        "entries": entries,
        "group_size": 2,
        "sample_size": 2,
        "sampling": {
            "allocation": "one-per-module-then-capacity-proportional-largest-remainder",
            "positions": "sha256-derived-coprime-modular-traversal",
            "seed": release.EXPERIMENT_001_SEED,
        },
        "schema": "cliffquant.experiment-001.sample.v1",
        "weight_sha256_encoding": "canonical-array-float64-little-endian",
    }
    manifest_path = root / "sample.manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    sample_descriptor = {
        "file": "sample.manifest.json",
        "sha256": release.sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }
    source = {
        "environments": ["general"],
        "sample_size": 2,
        "seed": release.EXPERIMENT_001_SEED,
    }
    source_fingerprint = sha256_bytes(canonical_json_bytes(source))
    input_fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "sample_manifest_sha256": sample_descriptor["sha256"],
                "source_fingerprint": source_fingerprint,
            }
        )
    )
    run_inputs = {
        **source,
        "input_fingerprint": input_fingerprint,
        "sample_manifest": sample_descriptor,
        "schema": "cliffquant.experiment-001.run-inputs.v1",
    }
    (root / "run.inputs.json").write_bytes(canonical_json_bytes(run_inputs))

    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        absmax_loss = 3.0 + index
        pooled_loss = 2.0 + index
        cliff_loss = 1.0
        results.append(
            {
                "identity": {key: value for key, value in entry.items() if key != "weight_sha256"},
                "input_fingerprint": input_fingerprint,
                "paired_baseline_minus_cliff": {
                    "absmax": absmax_loss - cliff_loss,
                    "pooled_wmse": pooled_loss - cliff_loss,
                },
                "policies": {
                    "absmax": _proxy_policy(
                        policy="absmax",
                        calibration_loss=absmax_loss,
                        heldout_loss=absmax_loss,
                    ),
                    "cliffquant_minimax": _proxy_policy(
                        policy="cliffquant_minimax",
                        calibration_loss=cliff_loss,
                        heldout_loss=cliff_loss,
                    ),
                    "pooled_wmse": _proxy_policy(
                        policy="pooled_wmse",
                        calibration_loss=pooled_loss,
                        heldout_loss=pooled_loss,
                    ),
                },
                "runtime_seconds": 0.1 + 0.1 * index,
                "schema": "cliffquant.experiment-001.group-result.v1",
                "weight_sha256": entry["weight_sha256"],
            }
        )
    (root / "groups.jsonl").write_bytes(
        b"".join(canonical_json_bytes(result) for result in results)
    )
    aggregate = release.aggregate_group_results(
        results,
        environments=("general",),
    )
    summary = {
        **aggregate,
        "input_fingerprint": input_fingerprint,
        "runtime": {
            "group_runtime_seconds_sum": float(
                release.np.sum(
                    [result["runtime_seconds"] for result in results],
                    dtype=release.np.float64,
                )
            ),
            "host": {},
            "wall_seconds": 1.0,
            "workers": 1,
        },
    }
    return root, summary


def _fake_validated_inputs(tmp_path: Path) -> dict[str, Any]:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    report_sources: dict[str, Path] = {}
    for role in release._REPORT_LOCATIONS:
        path = source_root / f"{role}.json"
        path.write_bytes(
            canonical_json_bytes(
                {
                    "role": role,
                    "schema": "test.verification.v1",
                    "status": "pass",
                }
            )
        )
        report_sources[role] = path

    report_payload_paths = {
        *{str(path) for path in release._REPORT_LOCATIONS.values()},
        *{
            str(path.with_name(f"{path.stem}.sha256.json"))
            for path in release._REPORT_LOCATIONS.values()
        },
    }
    sources: dict[str, Path] = {}
    for index, relative in enumerate(sorted(release._STATIC_PAYLOAD_PATHS)):
        if relative in report_payload_paths:
            continue
        path = source_root / f"source-{index:03d}.bin"
        path.write_bytes(f"publication source: {relative}\n".encode())
        sources[relative] = path
    return {
        "absmax_checkpoint": source_root,
        "absmax_identity": {"role": "absmax"},
        "cliffquant_checkpoint": source_root,
        "cliffquant_identity": {"role": "cliffquant"},
        "report_sources": report_sources,
        "sources": sources,
    }


def _build_fake_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    validated = _fake_validated_inputs(tmp_path)
    monkeypatch.setattr(release, "_validate_live_inputs", lambda **_kwargs: validated)
    monkeypatch.setattr(
        release,
        "_validate_bundled_semantics",
        lambda _root: {"validated": True},
    )
    monkeypatch.setattr(
        release,
        "_semantic_bindings",
        lambda _validated: _TEST_BINDINGS,
    )
    checkpoint_identities = iter((validated["absmax_identity"], validated["cliffquant_identity"]))
    monkeypatch.setattr(
        release,
        "describe_exported_checkpoint",
        lambda _checkpoint: next(checkpoint_identities),
    )
    destination = tmp_path / "research" / "results" / "experiment-001"
    unused = tmp_path / "unused"
    result = release.build_publication_release(
        absmax_checkpoint=unused,
        cliffquant_checkpoint=unused,
        absmax_structural_report=unused,
        cliffquant_structural_report=unused,
        cliffquant_clean_text_report=unused,
        nll_result_dir=unused,
        proxy_result_dir=unused,
        solver_certificate=unused,
        calibration_manifest=unused,
        calibration_tokens=unused,
        calibration_sidecar=unused,
        heldout_manifest=unused,
        heldout_tokens=unused,
        heldout_sidecar=unused,
        absmax_scale_run_manifest=unused,
        cliffquant_scale_run_manifest=unused,
        figure_manifest=unused,
        proxy_policy_figure=unused,
        proxy_paired_figure=unused,
        heldout_nll_figure=unused,
        gptqmodel_source=unused,
        destination=destination,
    )
    assert result["status"] == "pass"
    return destination


def _rewrite_release_sidecar(root: Path) -> None:
    manifest_path = root / release._RELEASE_MANIFEST
    payload = manifest_path.read_bytes()
    sidecar = {
        "manifest": {
            "file": release._RELEASE_MANIFEST,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        },
        "schema": release.RELEASE_SIDECAR_SCHEMA,
    }
    (root / release._RELEASE_SIDECAR).write_bytes(canonical_json_bytes(sidecar))


def _rewrite_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / release._RELEASE_MANIFEST).write_bytes(canonical_json_bytes(manifest))
    _rewrite_release_sidecar(root)


def test_builds_exact_atomic_release_and_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = _build_fake_release(tmp_path, monkeypatch)

    result = release.validate_publication_release(destination)

    assert result["bindings"] == _TEST_BINDINGS
    assert {entry["path"] for entry in result["files"]} == release._STATIC_PAYLOAD_PATHS
    assert not any(
        path.name.startswith(".experiment-001.") for path in destination.parent.iterdir()
    )


def test_release_rejects_payload_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = _build_fake_release(tmp_path, monkeypatch)
    target = destination / "proxy" / "summary.json"
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="file hash drift"):
        release.validate_publication_release(destination)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_release_rejects_missing_or_extra_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    destination = _build_fake_release(tmp_path, monkeypatch)
    if mutation == "missing":
        (destination / "proxy" / "groups.jsonl").unlink()
    else:
        (destination / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(ValueError, match="release file coverage mismatch"):
        release.validate_publication_release(destination)


def test_release_rejects_symlinked_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = _build_fake_release(tmp_path, monkeypatch)
    target = destination / "proxy" / "summary.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    try:
        os.symlink(outside, target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ValueError, match="contains a symlink"):
        release.validate_publication_release(destination)


def test_release_rejects_rehashed_status_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = _build_fake_release(tmp_path, monkeypatch)
    manifest = json.loads((destination / release._RELEASE_MANIFEST).read_bytes())
    manifest["status"] = "fail"
    _rewrite_manifest(destination, manifest)

    with pytest.raises(ValueError, match="schema, experiment, or status drift"):
        release.validate_publication_release(destination)


def test_release_rejects_rehashed_gate_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = _build_fake_release(tmp_path, monkeypatch)
    manifest = json.loads((destination / release._RELEASE_MANIFEST).read_bytes())
    manifest["bindings"]["evidence"]["gate_recomputed"] = False
    _rewrite_manifest(destination, manifest)

    with pytest.raises(ValueError, match="bindings drift"):
        release.validate_publication_release(destination)


def test_release_rejects_unsafe_inventory_path_even_when_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = _build_fake_release(tmp_path, monkeypatch)
    manifest = json.loads((destination / release._RELEASE_MANIFEST).read_bytes())
    manifest["files"][0]["path"] = "../escape"
    _rewrite_manifest(destination, manifest)

    with pytest.raises(ValueError, match="unsafe"):
        release.validate_publication_release(destination)


def test_builder_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = _build_fake_release(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        release.build_publication_release(
            **{
                name: tmp_path / "unused"
                for name in (
                    "absmax_checkpoint",
                    "cliffquant_checkpoint",
                    "absmax_structural_report",
                    "cliffquant_structural_report",
                    "cliffquant_clean_text_report",
                    "nll_result_dir",
                    "proxy_result_dir",
                    "solver_certificate",
                    "calibration_manifest",
                    "calibration_tokens",
                    "calibration_sidecar",
                    "heldout_manifest",
                    "heldout_tokens",
                    "heldout_sidecar",
                    "absmax_scale_run_manifest",
                    "cliffquant_scale_run_manifest",
                    "figure_manifest",
                    "proxy_policy_figure",
                    "proxy_paired_figure",
                    "heldout_nll_figure",
                    "gptqmodel_source",
                )
            },
            destination=destination,
        )


def test_frozen_nll_gate_is_recomputed_not_status_only() -> None:
    report = {
        "gates": {
            "environment": {"pass": True},
            "macro": {"pass": False},
            "publishable_model_nll_gate": True,
        },
        "status": "pass",
    }

    with pytest.raises(ValueError, match="every frozen publication gate"):
        release._require_publication_nll_gate(report)


def test_release_nll_gate_requires_all_live_origin_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    passing = {
        "gates": {
            "environment": {"pass": True},
            "macro": {"pass": True},
            "publishable_model_nll_gate": True,
        },
        "status": "pass",
    }

    def validate(result_dir: Path, **kwargs: Any) -> dict[str, Any]:
        observed["result_dir"] = result_dir
        observed.update(kwargs)
        return passing

    monkeypatch.setattr(release, "validate_heldout_nll_result", validate)
    result = release._validate_live_nll_evidence(
        result_dir=tmp_path / "nll",
        heldout_manifest=tmp_path / "heldout.manifest.json",
        absmax_checkpoint=tmp_path / "absmax",
        cliffquant_checkpoint=tmp_path / "cliffquant",
    )

    assert result is passing
    assert observed == {
        "absmax_checkpoint": tmp_path / "absmax",
        "cliffquant_checkpoint": tmp_path / "cliffquant",
        "heldout_manifest": tmp_path / "heldout.manifest.json",
        "result_dir": tmp_path / "nll",
    }


def test_checkpoint_pair_requires_distinct_model_and_scale_identities() -> None:
    shared_files = [
        {
            "file": "model.safetensors",
            "sha256": "1" * 64,
            "size_bytes": 1,
        }
    ]
    absmax = {
        "base_model": {"same": True},
        "checkpoint_files": shared_files,
        "corpus_replay": {"same": True},
        "gptq": {"same": True},
        "gptqmodel": {"same": True},
        "policy": release.POLICY_ABSMAX,
        "scale_run": {"manifest_sha256": "2" * 64},
        "sha256": "3" * 64,
    }
    cliffquant = {
        **absmax,
        "policy": release.POLICY_CLIFFQUANT,
        "sha256": "4" * 64,
    }

    with pytest.raises(ValueError, match="model payload identities must be distinct"):
        release._require_checkpoint_pair(
            absmax_identity=absmax,
            cliffquant_identity=cliffquant,
        )


def test_independent_nll_rerun_compares_arrays_not_npz_container_hash() -> None:
    supplied = _fake_nll_report()
    rerun = copy.deepcopy(supplied)
    rerun["raw_evidence"]["sha256"] = "a" * 64
    rerun["raw_evidence"]["size_bytes"] = 999

    release._compare_independent_nll_rerun(supplied, rerun)

    rerun["raw_evidence"]["arrays"]["absmax_token_nll"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="raw array identities differ"):
        release._compare_independent_nll_rerun(supplied, rerun)


def test_independent_nll_rerun_rejects_semantic_policy_drift() -> None:
    supplied = _fake_nll_report()
    rerun = copy.deepcopy(supplied)
    rerun["policies"]["absmax"]["macro_mean_nll"] = 0.5

    with pytest.raises(ValueError, match="semantic evidence differs"):
        release._compare_independent_nll_rerun(supplied, rerun)


def test_independent_nll_rerun_uses_explicit_live_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = _fake_nll_report()
    observed: dict[str, Any] = {}

    def compare(**kwargs: Any) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(release, "compare_heldout_nll", compare)
    monkeypatch.setattr(
        release,
        "_validate_live_nll_evidence",
        lambda **_kwargs: copy.deepcopy(supplied),
    )
    release._rerun_and_compare_nll(
        supplied_report=supplied,
        heldout_manifest=tmp_path / "heldout.manifest.json",
        absmax_checkpoint=tmp_path / "absmax",
        cliffquant_checkpoint=tmp_path / "cliffquant",
        gptqmodel_source=tmp_path / "gptqmodel",
        device="cuda:0",
        batch_size=1,
    )

    assert observed["heldout_manifest"] == tmp_path / "heldout.manifest.json"
    assert observed["absmax_checkpoint"] == tmp_path / "absmax"
    assert observed["cliffquant_checkpoint"] == tmp_path / "cliffquant"
    assert observed["gptqmodel_source"] == tmp_path / "gptqmodel"
    assert observed["device"] == "cuda:0"
    assert observed["batch_size"] == 1
    assert observed["output_dir"].name == "result"


def test_publication_runtime_requires_every_deterministic_cuda_setting() -> None:
    report = _fake_nll_report()
    release._require_deterministic_cuda_runtime(
        report,
        device="cuda:0",
        batch_size=1,
    )

    for field in tuple(report["runtime"]["accelerator"]):
        mutated = copy.deepcopy(report)
        current = mutated["runtime"]["accelerator"][field]
        mutated["runtime"]["accelerator"][field] = (
            "wrong" if isinstance(current, str) else not current
        )
        with pytest.raises(ValueError, match="deterministic CUDA contract"):
            release._require_deterministic_cuda_runtime(
                mutated,
                device="cuda:0",
                batch_size=1,
            )


def test_structural_tensors_must_differ_in_qweight_or_scales() -> None:
    modules = [
        {
            "module": f"model.language_model.layers.0.mlp.fake_{index}",
            "qweight_sha256": "1" * 64,
            "scales_sha256": "2" * 64,
        }
        for index in range(release.QWEN35_08B_EXPECTED_QUANTIZED_MODULES)
    ]
    absmax = {"modules": copy.deepcopy(modules)}
    cliffquant = {"modules": copy.deepcopy(modules)}

    with pytest.raises(ValueError, match="identical canonical qweight/scales"):
        release._require_distinct_structural_tensors(absmax, cliffquant)

    cliffquant["modules"][0]["qweight_sha256"] = "3" * 64
    identities = release._require_distinct_structural_tensors(absmax, cliffquant)
    assert identities["absmax"]["sha256"] != identities["cliffquant"]["sha256"]


def test_png_validation_reads_complete_checked_pixel_stream(tmp_path: Path) -> None:
    valid = tmp_path / "valid.png"
    valid.write_bytes(_rgba_png())
    assert release._png_dimensions(valid) == (1, 1)

    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(valid.read_bytes()[:24])
    with pytest.raises(ValueError, match="complete PNG"):
        release._png_dimensions(truncated)

    corrupt = bytearray(valid.read_bytes())
    corrupt[-5] ^= 1
    damaged = tmp_path / "damaged.png"
    damaged.write_bytes(corrupt)
    with pytest.raises(ValueError, match="checksum mismatch"):
        release._png_dimensions(damaged)


def test_figure_logical_names_cannot_be_swapped(tmp_path: Path) -> None:
    manifest = tmp_path / "figure-manifest.json"
    manifest.write_bytes(canonical_json_bytes({}))
    policy = tmp_path / "proxy-policy-comparison.png"
    paired = tmp_path / "proxy-paired-evidence.png"
    nll = tmp_path / "heldout-nll-comparison.png"
    for path in (policy, paired, nll):
        path.write_bytes(_rgba_png())

    with pytest.raises(ValueError, match="logical names"):
        release._validate_figure_sources(
            figure_manifest=manifest,
            policy_figure=paired,
            paired_figure=policy,
            nll_figure=nll,
            proxy_root=tmp_path / "proxy",
            nll_root=tmp_path / "nll",
        )


def test_fresh_figure_regeneration_rejects_forged_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = {
        logical_name: tmp_path / path.name
        for logical_name, path in release._FIGURE_LOCATIONS.items()
    }
    for path in supplied.values():
        path.write_bytes(_rgba_png())
    supplied_manifest = {"schema": "supplied"}

    def generate(
        _proxy_root: Path,
        output: Path,
        *,
        nll_input_dir: Path,
    ) -> dict[str, Any]:
        assert nll_input_dir == tmp_path / "nll"
        output.mkdir()
        for path in release._FIGURE_LOCATIONS.values():
            (output / path.name).write_bytes(_rgba_png() + b"forged")
        return dict(supplied_manifest)

    generated_files = {
        logical_name: tmp_path / "generated" / path.name
        for logical_name, path in release._FIGURE_LOCATIONS.items()
    }

    def validate(**kwargs: Any) -> tuple[Path, dict[str, Path], dict[str, Any]]:
        generated_root = kwargs["figure_manifest"].parent
        files = {
            logical_name: generated_root / path.name
            for logical_name, path in release._FIGURE_LOCATIONS.items()
        }
        generated_files.update(files)
        return kwargs["figure_manifest"], files, dict(supplied_manifest)

    monkeypatch.setattr(
        release,
        "_load_figure_helpers",
        lambda: SimpleNamespace(generate_figures=generate),
    )
    monkeypatch.setattr(release, "_validate_figure_sources", validate)
    with pytest.raises(ValueError, match="regenerated figure bytes differ"):
        release._regenerate_and_compare_figures(
            supplied_manifest=supplied_manifest,
            supplied_figures=supplied,
            proxy_root=tmp_path / "proxy",
            nll_root=tmp_path / "nll",
        )


def test_release_rechecks_hashes_after_semantic_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = _build_fake_release(tmp_path, monkeypatch)
    target = destination / "proxy" / "summary.json"
    changed = False

    def mutate_during_semantics(_root: Path) -> dict[str, Any]:
        nonlocal changed
        if not changed:
            target.write_bytes(target.read_bytes() + b"changed")
            changed = True
        return {"validated": True}

    monkeypatch.setattr(release, "_validate_bundled_semantics", mutate_during_semantics)
    with pytest.raises(ValueError, match="file hash drift"):
        release.validate_publication_release(destination)


def test_destination_must_be_disjoint_from_inputs(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()

    with pytest.raises(ValueError, match="overlaps validated input"):
        release._require_disjoint_destination(
            source / "research" / "results" / "experiment-001",
            (source,),
        )


def test_proxy_summary_is_recomputed_from_every_canonical_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, summary = _small_proxy_evidence(tmp_path, monkeypatch)
    release._recompute_proxy_summary(
        root,
        summary,
        require_checkpoints=False,
    )

    fabricated = copy.deepcopy(summary)
    fabricated["comparisons"]["absmax"]["paired_mean_improvement"] += 1.0
    with pytest.raises(ValueError, match="recomputed frozen aggregation"):
        release._recompute_proxy_summary(
            root,
            fabricated,
            require_checkpoints=False,
        )


def test_proxy_group_inventory_rejects_missing_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, summary = _small_proxy_evidence(tmp_path, monkeypatch)
    first_line = (root / "groups.jsonl").read_bytes().splitlines(keepends=True)[0]
    (root / "groups.jsonl").write_bytes(first_line)

    with pytest.raises(ValueError, match="inventory drift"):
        release._recompute_proxy_summary(
            root,
            summary,
            require_checkpoints=False,
        )


def test_live_proxy_jsonl_must_equal_every_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, summary = _small_proxy_evidence(tmp_path, monkeypatch)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir()
    lines = (root / "groups.jsonl").read_bytes().splitlines(keepends=True)
    for line in lines:
        record = json.loads(line)
        identity = record["identity"]
        name = f"{identity['sample_index']:04d}-{identity['group_id'][:16]}.json"
        (checkpoint_root / name).write_bytes(line)
    release._recompute_proxy_summary(
        root,
        summary,
        require_checkpoints=True,
    )

    first = next(checkpoint_root.iterdir())
    first.write_bytes(first.read_bytes() + b" ")
    with pytest.raises(ValueError, match="differs from its live checkpoint"):
        release._recompute_proxy_summary(
            root,
            summary,
            require_checkpoints=True,
        )


def test_publication_atomic_rename_never_replaces_existing_leaf(tmp_path: Path) -> None:
    source = tmp_path / "stage"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        release._rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert destination.is_dir()


def test_offline_scale_calibration_source_is_fully_bound() -> None:
    def descriptor(file: str, digit: str) -> dict[str, Any]:
        return {
            "file": file,
            "sha256": digit * 64,
            "size_bytes": 10,
        }

    model_source = {"fingerprint_sha256": "1" * 64}
    replay = {
        "manifest": descriptor("calibration.manifest.json", "2"),
        "sidecar": descriptor("calibration.sha256.json", "3"),
        "token_archive": descriptor("calibration.tokens.npz", "4"),
    }
    identity = {
        "base_model": {"source": model_source},
        "corpus_replay": {"calibration": replay},
    }
    calibration = {
        "archive_file": "calibration.diagonals.npz",
        "archive_sha256": "5" * 64,
        "environments": list(release.FROZEN_ENVIRONMENTS),
        "metadata_file": "calibration.diagonals.json",
        "metadata_sha256": "6" * 64,
        "source": {
            "archive": descriptor("calibration.diagonals.npz", "5"),
            "corpus": {
                "archive": replay["token_archive"],
                "manifest": replay["manifest"],
                "phase": "calibration",
                "sidecar_sha256": replay["sidecar"]["sha256"],
            },
            "metadata": descriptor("calibration.diagonals.json", "6"),
            "model": model_source,
            "phase": "calibration",
            "sidecar_sha256": "7" * 64,
        },
    }
    release._validate_scale_calibration_source(
        calibration,
        identity=identity,
        policy=release.POLICY_CLIFFQUANT,
    )

    calibration["source"] = {}
    with pytest.raises(ValueError, match="calibration source fields drift"):
        release._validate_scale_calibration_source(
            calibration,
            identity=identity,
            policy=release.POLICY_CLIFFQUANT,
        )


def test_offline_scale_module_rejects_extra_object_fields() -> None:
    module = {
        "branch": "mlp.down_proj",
        "in_features": 128,
        "layer": 0,
        "name": "model.language_model.layers.0.mlp.down_proj",
        "out_features": 8,
    }
    parsed = release._parse_scale_module(
        module,
        policy=release.POLICY_CLIFFQUANT,
    )
    assert parsed.name == module["name"]

    with pytest.raises(ValueError, match="module object fields drift"):
        release._parse_scale_module(
            {**module, "ignored": True},
            policy=release.POLICY_CLIFFQUANT,
        )
