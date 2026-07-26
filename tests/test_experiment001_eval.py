from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import cliffquant.experiment001_eval as experiment001_eval
import cliffquant.solver_certificate as solver_certificate
from cliffquant.experiment001_eval import (
    GroupEvaluationTask,
    _source_hashes,
    aggregate_group_results,
    evaluate_group,
    validate_group_checkpoint,
)
from cliffquant.experiment001_io import EXPERIMENT_001_ENVIRONMENTS, GroupCoordinate
from cliffquant.fp16_grid import bits_to_scale
from cliffquant.objective import quantize_codes
from cliffquant.provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    runtime_metadata,
    sha256_bytes,
    sha256_file,
)
from cliffquant.solver_certificate import (
    _certificate_source_descriptors,
    validate_solver_certificate,
)


def _write_solver_certificate(
    path: Path,
    *,
    elapsed_seconds: float = 1.0,
) -> tuple[dict, str]:
    family_order = (
        "normal",
        "log-uniform",
        "rne-half-ties",
        "outlier-mixture",
        "sparse",
    )
    case_results = []
    aggregate_input = bytearray()
    candidate_counts = []
    fallback_count = 0
    families = {family: 0 for family in family_order}
    for index in range(10_000):
        fingerprint = sha256_bytes(f"certificate-case-{index}".encode())
        aggregate_input.extend(bytes.fromhex(fingerprint))
        candidate_count = index % 19 + 1
        fallback = index % 7 == 0
        family = family_order[index % len(family_order)]
        candidate_counts.append(candidate_count)
        fallback_count += int(fallback)
        families[family] += 1
        case_results.append(
            {
                "candidate_scales_evaluated": candidate_count,
                "case_index": index,
                "exhaustive_fallback": fallback,
                "family": family,
                "fingerprint": fingerprint,
                "mismatch": False,
            }
        )
    aggregate = sha256_bytes(bytes(aggregate_input))
    certificate = {
        "aggregate_fingerprint_sha256": aggregate,
        "candidate_scales": {
            "maximum": max(candidate_counts),
            "median": float(np.median(candidate_counts)),
            "minimum": min(candidate_counts),
        },
        "case_results": case_results,
        "completed_cases": 10_000,
        "config": {
            "batch_size": 32,
            "cases": 10_000,
            "seed": 20_260_726,
            "workers": 1,
        },
        "elapsed_seconds": elapsed_seconds,
        "evidence_provenance": {
            "runtime": runtime_metadata(device="cpu", package_names=("numpy",)),
            "sources": _certificate_source_descriptors(),
        },
        "exhaustive_fallback_count": fallback_count,
        "families": families,
        "minimum_protocol_cases": 10_000,
        "mismatch_count": 0,
        "mismatches": [],
        "protocol_gate_passed": True,
        "schema": "cliffquant-solver-certification-v2",
        "status": "passed",
    }
    path.write_bytes(canonical_json_bytes(certificate))
    return certificate, aggregate


def _policy(loss: float) -> dict:
    return {
        "heldout_loss_by_environment": {
            environment: loss for environment in EXPERIMENT_001_ENVIRONMENTS
        },
        "heldout_max_loss": loss,
    }


def _aggregate_result(
    *,
    module: str,
    sample_index: int,
    absmax: float,
    pooled: float,
    cliff: float,
) -> dict:
    return {
        "identity": {
            "module": module,
            "sample_index": sample_index,
        },
        "policies": {
            "absmax": _policy(absmax),
            "cliffquant_minimax": _policy(cliff),
            "pooled_wmse": _policy(pooled),
        },
    }


def test_module_stratified_bootstrap_matches_hand_computable_constant_fixture() -> None:
    results = [
        _aggregate_result(
            module="module-a",
            sample_index=0,
            absmax=2.0,
            pooled=1.5,
            cliff=1.0,
        ),
        _aggregate_result(
            module="module-a",
            sample_index=1,
            absmax=3.0,
            pooled=2.5,
            cliff=2.0,
        ),
        _aggregate_result(
            module="module-b",
            sample_index=2,
            absmax=5.0,
            pooled=4.0,
            cliff=2.0,
        ),
        _aggregate_result(
            module="module-b",
            sample_index=3,
            absmax=6.0,
            pooled=5.0,
            cliff=3.0,
        ),
    ]
    summary = aggregate_group_results(
        results,
        bootstrap_replicates=100,
        bootstrap_seed=2339,
    )
    absmax = summary["comparisons"]["absmax"]
    pooled = summary["comparisons"]["pooled_wmse"]
    assert absmax["paired_mean_improvement"] == 2.0
    assert absmax["bootstrap_95_ci"] == [2.0, 2.0]
    assert pooled["paired_mean_improvement"] == 1.25
    assert pooled["bootstrap_95_ci"] == [1.25, 1.25]
    assert summary["proxy_gate_passed"] is True
    assert (
        aggregate_group_results(
            list(reversed(results)),
            bootstrap_replicates=100,
            bootstrap_seed=2339,
        )
        == summary
    )


def test_proxy_gate_failure_is_reported_without_changing_thresholds() -> None:
    results = [
        _aggregate_result(
            module="module-a",
            sample_index=0,
            absmax=3.0,
            pooled=0.5,
            cliff=1.0,
        ),
        _aggregate_result(
            module="module-b",
            sample_index=1,
            absmax=4.0,
            pooled=0.75,
            cliff=1.5,
        ),
    ]
    summary = aggregate_group_results(
        results,
        bootstrap_replicates=100,
        bootstrap_seed=2339,
    )
    assert summary["comparisons"]["absmax"]["gate_passed"] is True
    assert summary["comparisons"]["pooled_wmse"]["gate_passed"] is False
    assert summary["proxy_gate_passed"] is False


def _group_task() -> GroupEvaluationTask:
    coordinate = GroupCoordinate(
        sample_index=0,
        module="model.language_model.layers.0.self_attn.q_proj",
        layer=0,
        branch="self_attn.q_proj",
        row=0,
        group=0,
        input_start=0,
        input_stop=128,
        group_id="a" * 64,
    )
    weights = np.linspace(-0.7, 0.6, 128, dtype=np.float64)
    calibration = np.vstack(
        [np.linspace(0.5 + index * 0.1, 1.5 + index * 0.1, 128) for index in range(4)]
    )
    heldout = np.vstack(
        [np.linspace(1.5 + index * 0.1, 0.5 + index * 0.1, 128) for index in range(4)]
    )
    return GroupEvaluationTask(
        coordinate=coordinate,
        weight_sha256="b" * 64,
        weights=weights,
        calibration=calibration,
        heldout=heldout,
        environments=EXPERIMENT_001_ENVIRONMENTS,
        input_fingerprint="c" * 64,
        solver_chunk_size=4096,
    )


def test_group_evaluation_is_checkpoint_validatable_and_tamper_evident() -> None:
    task = _group_task()
    result = evaluate_group(task)
    assert validate_group_checkpoint(result, task=task) is result
    assert tuple(result["policies"]) == (
        "absmax",
        "cliffquant_minimax",
        "pooled_wmse",
    )
    assert all(
        tuple(result["policies"][policy]["heldout_loss_by_environment"])
        == EXPERIMENT_001_ENVIRONMENTS
        for policy in result["policies"]
    )

    canonical_round_trip = json.loads(json.dumps(result, sort_keys=True))
    assert validate_group_checkpoint(canonical_round_trip, task=task) is canonical_round_trip

    result["policies"]["absmax"]["heldout_loss_by_environment"]["general"] += 1.0
    with pytest.raises(ValueError, match="arithmetic mismatch"):
        validate_group_checkpoint(result, task=task)


def test_checkpoint_rejects_selection_objective_tamper() -> None:
    task = _group_task()
    result = evaluate_group(task)
    result["policies"]["pooled_wmse"]["selection"]["objective"] += 0.25
    with pytest.raises(ValueError, match="selection objective arithmetic mismatch"):
        validate_group_checkpoint(result, task=task)


def _retarget_policy(
    task: GroupEvaluationTask,
    policy: dict,
    alternate_bits: int,
) -> np.ndarray:
    alternate_scale = bits_to_scale(alternate_bits)
    alternate_codes = quantize_codes(task.weights, alternate_scale)
    residual = task.weights - alternate_scale * np.asarray(alternate_codes, dtype=np.float64)
    squared = np.square(residual)
    calibration_losses = np.sum(task.calibration * squared, axis=1, dtype=np.float64)
    heldout_losses = np.sum(task.heldout * squared, axis=1, dtype=np.float64)
    policy.update(
        {
            "calibration_loss_by_environment": {
                environment: float(calibration_losses[index])
                for index, environment in enumerate(task.environments)
            },
            "codes": list(alternate_codes),
            "codes_sha256": canonical_array_sha256(np.asarray(alternate_codes, dtype="<i2")),
            "heldout_loss_by_environment": {
                environment: float(heldout_losses[index])
                for index, environment in enumerate(task.environments)
            },
            "heldout_max_loss": float(np.max(heldout_losses)),
            "scale": alternate_scale,
            "scale_bits": alternate_bits,
        }
    )
    return calibration_losses


def test_checkpoint_rejects_self_consistent_but_nonoptimal_exact_scale() -> None:
    task = _group_task()
    result = evaluate_group(task)
    policy = result["policies"]["pooled_wmse"]
    alternate_bits = policy["scale_bits"] + 1
    calibration_losses = _retarget_policy(task, policy, alternate_bits)
    policy["selection"]["objective"] = float(np.mean(calibration_losses, dtype=np.float64))
    with pytest.raises(ValueError, match="not the current optimum"):
        validate_group_checkpoint(result, task=task)


def test_checkpoint_rejects_self_consistent_non_absmax_scale() -> None:
    task = _group_task()
    result = evaluate_group(task)
    policy = result["policies"]["absmax"]
    _retarget_policy(task, policy, policy["scale_bits"] + 1)
    with pytest.raises(ValueError, match="not the frozen selection"):
        validate_group_checkpoint(result, task=task)


def test_checkpoint_rejects_unknown_fields() -> None:
    task = _group_task()
    result = evaluate_group(task)
    result["unexpected"] = "private-or-stale-data"
    with pytest.raises(ValueError, match="top-level fields drift"):
        validate_group_checkpoint(result, task=task)

    result = evaluate_group(task)
    result["policies"]["absmax"]["unexpected"] = True
    with pytest.raises(ValueError, match="policy fields drift"):
        validate_group_checkpoint(result, task=task)


def test_execution_source_fingerprint_covers_all_numerical_dependencies() -> None:
    protocol = Path(__file__).resolve().parents[1] / "research" / "EXPERIMENT_001_PROTOCOL.md"
    hashes = _source_hashes(protocol)
    assert {
        "activation_statistics",
        "breakpoint_solver",
        "calibration_entrypoint",
        "contracts",
        "corpus_contract",
        "corpus_replay",
        "dataset_viewer_contract",
        "evaluation_engine",
        "fp16_grid",
        "input_engine",
        "model_snapshot",
        "objective",
        "provenance",
        "protocol",
        "qwen35_inventory",
        "solver_certificate_validator",
    } <= set(hashes)
    assert all(len(descriptor["sha256"]) == 64 for descriptor in hashes.values())


def test_execution_rejects_a_noncanonical_protocol_path(tmp_path: Path) -> None:
    protocol = tmp_path / "EXPERIMENT_001_PROTOCOL.md"
    protocol.write_text("lookalike", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical frozen protocol path"):
        _source_hashes(protocol)


def test_cli_returns_nonzero_when_proxy_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment001_eval,
        "run_experiment_001",
        lambda **_kwargs: {
            "artifacts": {},
            "input_fingerprint": "0" * 64,
            "proxy_gate_passed": False,
            "summary": {},
        },
    )
    assert (
        experiment001_eval.main(
            [
                "--execute",
                "--calibration-json",
                str(tmp_path / "calibration.json"),
                "--heldout-json",
                str(tmp_path / "heldout.json"),
                "--model-dir",
                str(tmp_path / "model"),
                "--output-dir",
                str(tmp_path / "output"),
                "--solver-certificate",
                str(tmp_path / "certificate.json"),
            ]
        )
        == 2
    )


def test_solver_certificate_v2_is_strictly_validated_and_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "certificate-a.json"
    _, aggregate = _write_solver_certificate(first_path, elapsed_seconds=1.0)
    monkeypatch.setattr(
        solver_certificate,
        "SOLVER_AGGREGATE_FINGERPRINT",
        aggregate,
    )
    first = validate_solver_certificate(first_path)
    assert first["schema"] == "cliffquant-solver-certification-v2"
    assert first["completed_cases"] == 10_000
    assert first["protocol_gate_passed"] is True
    assert first["sha256"] == sha256_file(first_path)
    assert first["file"] == first_path.name
    assert str(tmp_path) not in json.dumps(first)

    second_path = tmp_path / "certificate-b.json"
    _write_solver_certificate(second_path, elapsed_seconds=2.0)
    second = validate_solver_certificate(second_path)
    assert second["sha256"] != first["sha256"]
    first_fingerprint = sha256_bytes(canonical_json_bytes({"solver_certificate": first}))
    second_fingerprint = sha256_bytes(canonical_json_bytes({"solver_certificate": second}))
    assert first_fingerprint != second_fingerprint


def test_solver_certificate_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="file is missing"):
        validate_solver_certificate(tmp_path / "missing-certificate.json")


def test_solver_certificate_rejects_gate_and_case_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "certificate.json"
    certificate, aggregate = _write_solver_certificate(path)
    monkeypatch.setattr(
        solver_certificate,
        "SOLVER_AGGREGATE_FINGERPRINT",
        aggregate,
    )

    certificate["protocol_gate_passed"] = False
    path.write_bytes(canonical_json_bytes(certificate))
    with pytest.raises(ValueError, match="did not pass"):
        validate_solver_certificate(path)

    certificate["protocol_gate_passed"] = True
    certificate["case_results"][0]["fingerprint"] = "0" * 64
    path.write_bytes(canonical_json_bytes(certificate))
    with pytest.raises(ValueError, match="case aggregate fingerprint mismatch"):
        validate_solver_certificate(path)


def test_solver_certificate_rejects_mismatched_current_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "certificate.json"
    certificate, aggregate = _write_solver_certificate(path)
    monkeypatch.setattr(
        solver_certificate,
        "SOLVER_AGGREGATE_FINGERPRINT",
        aggregate,
    )
    certificate["evidence_provenance"]["sources"]["breakpoint"]["sha256"] = "0" * 64
    path.write_bytes(canonical_json_bytes(certificate))
    with pytest.raises(ValueError, match="source hashes do not match"):
        validate_solver_certificate(path)


def test_solver_certificate_rejects_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "certificate.json"
    certificate, aggregate = _write_solver_certificate(path)
    monkeypatch.setattr(
        solver_certificate,
        "SOLVER_AGGREGATE_FINGERPRINT",
        aggregate,
    )
    certificate["evidence_provenance"]["runtime"]["numpy"] = "0.0.0"
    path.write_bytes(canonical_json_bytes(certificate))
    with pytest.raises(ValueError, match="runtime does not match"):
        validate_solver_certificate(path)
