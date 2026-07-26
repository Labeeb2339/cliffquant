from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.generate_figures import (
    FIGURE_MANIFEST_SCHEMA,
    generate_figures,
    verify_and_load_summary,
)


def _write_fixture(directory: Path) -> None:
    summary = {
        "comparisons": {
            "absmax": {
                "baseline_minus_cliff_module_macro": 0.25,
                "bootstrap_95_ci": [0.2, 0.3],
                "bootstrap_method": {"replicates": 100},
                "gate_passed": True,
                "paired_mean_improvement": 0.25,
                "wins_ties_losses": {"losses": 0, "ties": 1, "wins": 3},
            },
            "pooled_wmse": {
                "baseline_minus_cliff_module_macro": 0.05,
                "bootstrap_95_ci": [0.04, 0.06],
                "bootstrap_method": {"replicates": 100},
                "gate_passed": True,
                "paired_mean_improvement": 0.05,
                "wins_ties_losses": {"losses": 1, "ties": 1, "wins": 2},
            },
        },
        "environments": ["a", "b"],
        "group_count": 4,
        "module_count": 2,
        "policies": {
            "absmax": {"module_macro_heldout_max_loss": 1.0},
            "cliffquant_minimax": {"module_macro_heldout_max_loss": 0.75},
            "pooled_wmse": {"module_macro_heldout_max_loss": 0.8},
        },
        "proxy_gate_passed": True,
        "schema": "cliffquant.experiment-001.proxy-summary.v1",
    }
    payload = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    summary_path = directory / "summary.json"
    summary_path.write_bytes(payload)
    checksums = {
        "summary": {
            "file": "summary.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    }
    (directory / "SHA256SUMS.json").write_text(
        json.dumps(checksums, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_nll_fixture(directory: Path) -> None:
    import numpy as np

    from cliffquant import heldout_nll
    from cliffquant.full_model_artifacts import FROZEN_ENVIRONMENTS
    from cliffquant.provenance import canonical_array_sha256, canonical_json_bytes

    directory.mkdir()
    raw_path = directory / "heldout-nll.raw.npz"
    absmax_means = {
        "general": 4.2,
        "code": 3.8,
        "math": 5.6,
        "multilingual": 4.7,
    }
    cliffquant_means = {
        "general": 4.204,
        "code": 3.797,
        "math": 5.611,
        "multilingual": 4.7,
    }
    means = {"absmax": absmax_means, "cliffquant": cliffquant_means}
    arrays = {}
    for policy in ("absmax", "cliffquant"):
        for environment in FROZEN_ENVIRONMENTS:
            prefix = f"{policy}__{environment}__"
            target_mask = np.ones((64, 255), dtype=np.uint8)
            token_nll = np.full((64, 255), means[policy][environment], dtype=np.float64)
            window_nll_sum = np.sum(token_nll, axis=1, dtype=np.float64)
            window_target_tokens = np.sum(target_mask, axis=1, dtype=np.int64)
            arrays[f"{prefix}target_mask"] = target_mask
            arrays[f"{prefix}token_nll"] = token_nll
            arrays[f"{prefix}window_nll_sum"] = window_nll_sum
            arrays[f"{prefix}window_target_tokens"] = window_target_tokens
    with raw_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    policies = heldout_nll._recompute_policy_metrics(arrays)
    environment_deltas = {
        environment: (
            policies["cliffquant"][environment]["mean_nll"]
            - policies["absmax"][environment]["mean_nll"]
        )
        for environment in FROZEN_ENVIRONMENTS
    }
    macro_delta = policies["cliffquant"]["macro_mean_nll"] - policies["absmax"]["macro_mean_nll"]
    array_metadata = {
        name: {
            "dtype": value.dtype.str,
            "sha256": canonical_array_sha256(value),
            "shape": list(value.shape),
        }
        for name, value in sorted(arrays.items())
    }
    report = {
        "checkpoints": {"absmax": {}, "cliffquant": {}},
        "comparison": {
            "environment_delta_cliffquant_minus_absmax": environment_deltas,
            "macro_delta_cliffquant_minus_absmax": macro_delta,
        },
        "corpus_replay": {"sha256": "c" * 64},
        "evaluator": heldout_nll._nll_evaluator_identity(),
        "gates": {
            "environment": {"limit": 0.02, "pass": True},
            "macro": {"limit": 0.01, "pass": True},
            "publishable_model_nll_gate": True,
        },
        "heldout": {
            "archive_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        },
        "policies": policies,
        "raw_evidence": {
            "arrays": array_metadata,
            "file": raw_path.name,
            "sha256": _sha256_file(raw_path),
            "size_bytes": raw_path.stat().st_size,
        },
        "runtime": {},
        "schema": "cliffquant.heldout-nll-comparison.v1",
        "status": "pass",
    }
    report_path = directory / "heldout-nll.json"
    report_path.write_bytes(canonical_json_bytes(report))
    sidecar = {
        "inputs": {
            "corpus_replay_pair_sha256": "c" * 64,
            "heldout_manifest_sha256": "b" * 64,
            "heldout_tokens_sha256": "a" * 64,
        },
        "raw": {
            "file": raw_path.name,
            "sha256": _sha256_file(raw_path),
            "size_bytes": raw_path.stat().st_size,
        },
        "report": {
            "file": report_path.name,
            "sha256": _sha256_file(report_path),
            "size_bytes": report_path.stat().st_size,
        },
        "schema": "cliffquant.heldout-nll-sidecar.v1",
    }
    (directory / "heldout-nll.sha256.json").write_bytes(canonical_json_bytes(sidecar))


def _install_nll_validator_spy(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    from cliffquant import heldout_nll

    monkeypatch.setattr(heldout_nll, "validate_corpus_replay_pair", lambda value: value)
    monkeypatch.setattr(
        heldout_nll,
        "_validate_heldout_identity",
        lambda _value, *, corpus_replay: None,
    )
    monkeypatch.setattr(
        heldout_nll,
        "_validate_checkpoint_results",
        lambda _value, _corpus_replay: None,
    )
    monkeypatch.setattr(heldout_nll, "_validate_runtime", lambda _value: None)

    canonical_validator = heldout_nll.validate_heldout_nll_result
    calls: list[Path] = []

    def validate(result_dir: Path) -> dict:
        root = Path(result_dir).resolve()
        calls.append(root)
        return canonical_validator(root)

    monkeypatch.setattr(
        heldout_nll,
        "validate_heldout_nll_result",
        validate,
    )
    return calls


def test_checksum_verification_precedes_summary_parsing(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    summary, provenance = verify_and_load_summary(tmp_path)
    assert summary["group_count"] == 4
    assert provenance["verified_files"]["summary"]["file"] == "summary.json"

    (tmp_path / "summary.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum verification failed"):
        verify_and_load_summary(tmp_path)


def test_generation_is_byte_reproducible_and_has_no_timestamp(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    input_dir.mkdir()
    _write_fixture(input_dir)

    first = generate_figures(input_dir, first_dir)
    second = generate_figures(input_dir, second_dir)

    assert first["schema"] == FIGURE_MANIFEST_SCHEMA
    assert first == second
    assert "heldout_nll_comparison" not in first["figures"]
    assert "nll_input" not in first
    assert "timestamp" not in (first_dir / "figure-manifest.json").read_text(encoding="utf-8")
    for figure in first["figures"].values():
        first_bytes = (first_dir / figure["file"]).read_bytes()
        second_bytes = (second_dir / figure["file"]).read_bytes()
        assert first_bytes == second_bytes
        assert first_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_nll_generation_is_validated_bound_and_byte_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_input = tmp_path / "proxy-input"
    nll_input = tmp_path / "nll-input"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    proxy_input.mkdir()
    _write_fixture(proxy_input)
    _write_nll_fixture(nll_input)
    validator_calls = _install_nll_validator_spy(monkeypatch)

    first = generate_figures(proxy_input, first_dir, nll_input_dir=nll_input)
    second = generate_figures(proxy_input, second_dir, nll_input_dir=nll_input)

    assert validator_calls == [nll_input.resolve(), nll_input.resolve()]
    assert first == second
    figure = first["figures"]["heldout_nll_comparison"]
    assert figure["file"] == "heldout-nll-comparison.png"
    assert figure["width_px"] == 2400
    assert figure["height_px"] == 1450
    assert (first_dir / figure["file"]).read_bytes() == (second_dir / figure["file"]).read_bytes()
    assert figure["sha256"] == _sha256_file(first_dir / figure["file"])
    assert first["nll_input"]["report"]["sha256"] == _sha256_file(nll_input / "heldout-nll.json")
    assert first["nll_input"]["raw_evidence"]["sha256"] == _sha256_file(
        nll_input / "heldout-nll.raw.npz"
    )
    rows = first["data"]["heldout_nll"]["rows"]
    assert [row["label"] for row in rows] == [
        "Macro mean",
        "General",
        "Code",
        "Math",
        "Multilingual",
    ]
    assert [row["limit"] for row in rows] == [0.01, 0.02, 0.02, 0.02, 0.02]


def test_nll_mutation_is_rejected_by_canonical_validator_before_plotting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_input = tmp_path / "proxy-input"
    nll_input = tmp_path / "nll-input"
    output_dir = tmp_path / "output"
    proxy_input.mkdir()
    _write_fixture(proxy_input)
    _write_nll_fixture(nll_input)
    _install_nll_validator_spy(monkeypatch)
    (nll_input / "heldout-nll.raw.npz").write_bytes(b"mutated")

    with pytest.raises(ValueError, match="raw evidence live hash or size mismatch"):
        generate_figures(proxy_input, output_dir, nll_input_dir=nll_input)

    assert not output_dir.exists()
