"""Policy evaluation, resumable execution, and frozen gates for Experiment 001."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cliffquant.breakpoint import solve_breakpoint_exact
from cliffquant.corpus_replay import load_pinned_tokenizer, verify_corpus_replay
from cliffquant.experiment001_io import (
    EXPERIMENT_001_ENVIRONMENTS,
    EXPERIMENT_001_GROUP_SIZE,
    EXPERIMENT_001_MODEL_REVISION,
    EXPERIMENT_001_SAMPLE_SIZE,
    EXPERIMENT_001_SEED,
    DiagonalArtifact,
    GroupCoordinate,
    SafetensorWeightStore,
    SampledGroup,
    deterministic_group_coordinates,
    load_diagonal_artifact,
    materialize_sample,
    sample_manifest,
    validate_diagonal_pair,
)
from cliffquant.fp16_grid import (
    bits_to_scale,
    positive_finite_fp16_scales,
)
from cliffquant.objective import quantize_codes
from cliffquant.provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    runtime_metadata,
    sha256_bytes,
    sha256_file,
)
from cliffquant.solver_certificate import validate_solver_certificate

EXPERIMENT_001_ENGINE_VERSION = "experiment-001-real-group-v1"
EXPERIMENT_001_BOOTSTRAP_REPLICATES = 10_000
EXPERIMENT_001_SOLVER_CHUNK_SIZE = 4096

_POLICIES = ("absmax", "pooled_wmse", "cliffquant_minimax")
_BASELINES = ("absmax", "pooled_wmse")
_SCALES = positive_finite_fp16_scales()


@dataclass(frozen=True, slots=True)
class GroupEvaluationTask:
    """Pickle-safe payload for one exact real-group comparison."""

    coordinate: GroupCoordinate
    weight_sha256: str
    weights: NDArray[np.float64]
    calibration: NDArray[np.float64]
    heldout: NDArray[np.float64]
    environments: tuple[str, ...]
    input_fingerprint: str
    solver_chunk_size: int


def _nearest_positive_fp16(target: float) -> tuple[int, float]:
    if not np.isfinite(target) or target < 0.0:
        raise ValueError("scale target must be finite and non-negative")
    insertion = int(np.searchsorted(_SCALES, target, side="left"))
    if insertion == 0:
        index = 0
    elif insertion == _SCALES.size:
        index = _SCALES.size - 1
    else:
        lower_index = insertion - 1
        upper_index = insertion
        lower_distance = target - float(_SCALES[lower_index])
        upper_distance = float(_SCALES[upper_index]) - target
        index = lower_index if lower_distance <= upper_distance else upper_index
    return index + 1, float(_SCALES[index])


def _absmax_scale(weights: NDArray[np.float64]) -> tuple[int, float]:
    positive = weights[weights > 0.0]
    negative = weights[weights < 0.0]
    positive_limit = float(np.max(positive)) / 7.0 if positive.size else 0.0
    negative_limit = float(np.max(np.abs(negative))) / 8.0 if negative.size else 0.0
    return _nearest_positive_fp16(max(positive_limit, negative_limit))


def _direct_losses(
    weights: NDArray[np.float64],
    diagonals: NDArray[np.float64],
    scale: float,
    codes: Sequence[int],
) -> NDArray[np.float64]:
    code_array = np.asarray(codes, dtype=np.int16)
    if code_array.shape != weights.shape:
        raise ValueError("codes must match the weight group")
    residual = weights - scale * code_array.astype(np.float64)
    losses = np.sum(diagonals * np.square(residual)[np.newaxis, :], axis=1, dtype=np.float64)
    if not np.all(np.isfinite(losses)):
        raise ArithmeticError("direct heldout proxy evaluation overflowed")
    return losses


def _policy_record(
    *,
    weights: NDArray[np.float64],
    calibration: NDArray[np.float64],
    heldout: NDArray[np.float64],
    environments: tuple[str, ...],
    scale_bits: int,
    codes: Sequence[int],
    selection: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    scale = bits_to_scale(scale_bits)
    code_array = np.asarray(codes, dtype="<i2")
    calibration_losses = _direct_losses(weights, calibration, scale, code_array)
    heldout_losses = _direct_losses(weights, heldout, scale, code_array)
    return {
        "calibration_loss_by_environment": {
            environment: float(calibration_losses[index])
            for index, environment in enumerate(environments)
        },
        "codes": [int(code) for code in code_array],
        "codes_sha256": canonical_array_sha256(code_array),
        "heldout_loss_by_environment": {
            environment: float(heldout_losses[index])
            for index, environment in enumerate(environments)
        },
        "heldout_max_loss": float(np.max(heldout_losses)),
        "runtime_seconds": runtime_seconds,
        "scale": scale,
        "scale_bits": scale_bits,
        "selection": selection,
    }


def evaluate_group(task: GroupEvaluationTask) -> dict[str, Any]:
    """Evaluate all three frozen policies for one group using direct arithmetic."""

    weights = np.asarray(task.weights, dtype=np.float64)
    calibration = np.asarray(task.calibration, dtype=np.float64)
    heldout = np.asarray(task.heldout, dtype=np.float64)
    expected_shape = (len(task.environments), EXPERIMENT_001_GROUP_SIZE)
    if weights.shape != (EXPERIMENT_001_GROUP_SIZE,):
        raise ValueError("task weight group must contain exactly 128 values")
    if calibration.shape != expected_shape or heldout.shape != expected_shape:
        raise ValueError("task diagonals must have one 128-value row per environment")
    if (
        not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(calibration))
        or not np.all(np.isfinite(heldout))
        or np.any(calibration < 0.0)
        or np.any(heldout < 0.0)
    ):
        raise ValueError("task arrays must be finite and diagonals non-negative")

    total_start = time.perf_counter()
    absmax_start = time.perf_counter()
    absmax_bits, absmax_scale = _absmax_scale(weights)
    absmax_codes = quantize_codes(weights, absmax_scale)
    absmax_runtime = time.perf_counter() - absmax_start
    absmax = _policy_record(
        weights=weights,
        calibration=calibration,
        heldout=heldout,
        environments=task.environments,
        scale_bits=absmax_bits,
        codes=absmax_codes,
        selection={
            "method": "range-derived-absmax-nearest-positive-fp16",
            "objective": "none",
            "rounding_tie": "lower-fp16-bit-pattern",
        },
        runtime_seconds=absmax_runtime,
    )

    pooled_start = time.perf_counter()
    pooled_diagonal = np.mean(calibration, axis=0, dtype=np.float64)[np.newaxis, :]
    pooled_result = solve_breakpoint_exact(
        weights,
        pooled_diagonal,
        chunk_size=task.solver_chunk_size,
    )
    pooled_runtime = time.perf_counter() - pooled_start
    pooled = _policy_record(
        weights=weights,
        calibration=calibration,
        heldout=heldout,
        environments=task.environments,
        scale_bits=pooled_result.scale_bits,
        codes=pooled_result.codes,
        selection={
            "method": "exact-breakpoint-pooled-wmse",
            "objective": pooled_result.objective,
            "solver_provenance": asdict(pooled_result.provenance),
            "solver_returned_objective": pooled_result.objective,
        },
        runtime_seconds=pooled_runtime,
    )
    pooled["selection"]["objective"] = float(
        np.mean(
            [
                pooled["calibration_loss_by_environment"][environment]
                for environment in task.environments
            ],
            dtype=np.float64,
        )
    )

    cliff_start = time.perf_counter()
    cliff_result = solve_breakpoint_exact(
        weights,
        calibration,
        chunk_size=task.solver_chunk_size,
    )
    cliff_runtime = time.perf_counter() - cliff_start
    cliff = _policy_record(
        weights=weights,
        calibration=calibration,
        heldout=heldout,
        environments=task.environments,
        scale_bits=cliff_result.scale_bits,
        codes=cliff_result.codes,
        selection={
            "method": "exact-breakpoint-four-environment-minimax",
            "objective": cliff_result.objective,
            "solver_provenance": asdict(cliff_result.provenance),
            "solver_returned_objective": cliff_result.objective,
        },
        runtime_seconds=cliff_runtime,
    )
    cliff["selection"]["objective"] = max(cliff["calibration_loss_by_environment"].values())

    return {
        "identity": task.coordinate.identity(),
        "input_fingerprint": task.input_fingerprint,
        "paired_baseline_minus_cliff": {
            "absmax": absmax["heldout_max_loss"] - cliff["heldout_max_loss"],
            "pooled_wmse": pooled["heldout_max_loss"] - cliff["heldout_max_loss"],
        },
        "policies": {
            "absmax": absmax,
            "cliffquant_minimax": cliff,
            "pooled_wmse": pooled,
        },
        "runtime_seconds": time.perf_counter() - total_start,
        "schema": "cliffquant.experiment-001.group-result.v1",
        "weight_sha256": task.weight_sha256,
    }


def _module_macro(
    results: Sequence[dict[str, Any]],
    *,
    policy: str,
    metric: Callable[[dict[str, Any]], float],
) -> tuple[float, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        module = result["identity"]["module"]
        values[module].append(metric(result["policies"][policy]))
    if not values:
        raise ValueError("cannot aggregate an empty result set")
    module_means = {
        module: float(np.mean(module_values, dtype=np.float64))
        for module, module_values in sorted(values.items())
    }
    return float(np.mean(tuple(module_means.values()), dtype=np.float64)), module_means


def _module_stratified_bootstrap(
    effects_by_module: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
    generator = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.zeros(replicates, dtype=np.float64)
    for module in sorted(effects_by_module):
        effects = np.asarray(effects_by_module[module], dtype=np.float64)
        if effects.ndim != 1 or effects.size == 0 or not np.all(np.isfinite(effects)):
            raise ValueError(f"invalid paired effects for module {module}")
        indices = generator.integers(
            0,
            effects.size,
            size=(replicates, effects.size),
        )
        bootstrap += np.mean(effects[indices], axis=1, dtype=np.float64)
    bootstrap /= len(effects_by_module)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975), method="linear")
    return float(lower), float(upper)


def aggregate_group_results(
    results: Sequence[dict[str, Any]],
    *,
    environments: Sequence[str] = EXPERIMENT_001_ENVIRONMENTS,
    bootstrap_replicates: int = EXPERIMENT_001_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = EXPERIMENT_001_SEED,
) -> dict[str, Any]:
    """Compute module-macro metrics and both frozen baseline gates."""

    if not results:
        raise ValueError("cannot aggregate an empty result set")
    ordered_results = tuple(
        sorted(results, key=lambda result: int(result["identity"]["sample_index"]))
    )
    sample_indices = [result["identity"]["sample_index"] for result in ordered_results]
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("result set contains duplicate sample indices")
    environment_tuple = tuple(environments)
    policy_metrics: dict[str, Any] = {}
    for policy in _POLICIES:
        macro_max, module_max = _module_macro(
            ordered_results,
            policy=policy,
            metric=lambda record: float(record["heldout_max_loss"]),
        )
        environment_macro: dict[str, float] = {}
        for environment in environment_tuple:
            value, _ = _module_macro(
                ordered_results,
                policy=policy,
                metric=lambda record, env=environment: float(
                    record["heldout_loss_by_environment"][env]
                ),
            )
            environment_macro[environment] = value
        policy_metrics[policy] = {
            "module_macro_heldout_loss_by_environment": environment_macro,
            "module_macro_heldout_max_loss": macro_max,
            "per_module_heldout_max_loss": module_max,
        }

    comparisons: dict[str, Any] = {}
    cliff_macro = policy_metrics["cliffquant_minimax"]["module_macro_heldout_max_loss"]
    for baseline_index, baseline in enumerate(_BASELINES):
        effects_by_module: dict[str, list[float]] = defaultdict(list)
        wins = 0
        ties = 0
        losses = 0
        for result in ordered_results:
            baseline_loss = float(result["policies"][baseline]["heldout_max_loss"])
            cliff_loss = float(result["policies"]["cliffquant_minimax"]["heldout_max_loss"])
            effect = baseline_loss - cliff_loss
            effects_by_module[result["identity"]["module"]].append(effect)
            if effect > 0.0:
                wins += 1
            elif effect < 0.0:
                losses += 1
            else:
                ties += 1
        module_effects = [
            float(np.mean(effects_by_module[module], dtype=np.float64))
            for module in sorted(effects_by_module)
        ]
        paired_mean = float(np.mean(module_effects, dtype=np.float64))
        comparison_seed = bootstrap_seed + baseline_index
        lower, upper = _module_stratified_bootstrap(
            effects_by_module,
            replicates=bootstrap_replicates,
            seed=comparison_seed,
        )
        baseline_macro = policy_metrics[baseline]["module_macro_heldout_max_loss"]
        checks = {
            "bootstrap_lower_bound_gt_zero": lower > 0.0,
            "cliff_macro_lower": cliff_macro < baseline_macro,
            "paired_mean_positive": paired_mean > 0.0,
            "wins_exceed_losses": wins > losses,
        }
        comparisons[baseline] = {
            "baseline_minus_cliff_module_macro": baseline_macro - cliff_macro,
            "bootstrap_95_ci": [lower, upper],
            "bootstrap_method": {
                "modules": "fixed-strata",
                "replicates": bootstrap_replicates,
                "resampling": "groups-with-replacement-within-every-module",
                "rng": "NumPy-PCG64",
                "seed": comparison_seed,
            },
            "gate_checks": checks,
            "gate_passed": all(checks.values()),
            "paired_mean_improvement": paired_mean,
            "wins_ties_losses": {
                "losses": losses,
                "ties": ties,
                "wins": wins,
            },
        }

    return {
        "comparisons": comparisons,
        "environments": list(environment_tuple),
        "group_count": len(ordered_results),
        "module_count": len({result["identity"]["module"] for result in ordered_results}),
        "policies": policy_metrics,
        "proxy_gate_passed": all(comparison["gate_passed"] for comparison in comparisons.values()),
        "schema": "cliffquant.experiment-001.proxy-summary.v1",
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_or_validate_frozen(path: Path, value: dict[str, Any]) -> str:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing frozen artifact disagrees with current inputs: {path}")
    else:
        _atomic_write(path, payload)
    return sha256_bytes(payload)


def _checkpoint_path(checkpoint_dir: Path, coordinate: GroupCoordinate) -> Path:
    return checkpoint_dir / (f"{coordinate.sample_index:04d}-{coordinate.group_id[:16]}.json")


def _validate_policy_checkpoint(
    policy: Any,
    *,
    task: GroupEvaluationTask,
    policy_name: str,
) -> None:
    if not isinstance(policy, dict):
        raise ValueError("checkpoint policy must be an object")
    if set(policy) != {
        "calibration_loss_by_environment",
        "codes",
        "codes_sha256",
        "heldout_loss_by_environment",
        "heldout_max_loss",
        "runtime_seconds",
        "scale",
        "scale_bits",
        "selection",
    }:
        raise ValueError("checkpoint policy fields drift")
    scale_bits = policy.get("scale_bits")
    if type(scale_bits) is not int:
        raise ValueError("checkpoint scale_bits must be an integer")
    scale = bits_to_scale(scale_bits)
    if policy.get("scale") != scale:
        raise ValueError("checkpoint scale does not match scale_bits")
    codes = policy.get("codes")
    if (
        not isinstance(codes, list)
        or len(codes) != EXPERIMENT_001_GROUP_SIZE
        or any(type(code) is not int or not -8 <= code <= 7 for code in codes)
    ):
        raise ValueError("checkpoint codes violate W4/G128")
    expected_codes = quantize_codes(task.weights, scale)
    if tuple(codes) != expected_codes:
        raise ValueError("checkpoint codes do not match RNE quantization")
    if canonical_array_sha256(np.asarray(codes, dtype="<i2")) != policy.get("codes_sha256"):
        raise ValueError("checkpoint code hash mismatch")
    direct_losses: dict[str, NDArray[np.float64]] = {}
    for key, diagonals in (
        ("calibration_loss_by_environment", task.calibration),
        ("heldout_loss_by_environment", task.heldout),
    ):
        raw_losses = policy.get(key)
        if not isinstance(raw_losses, dict) or set(raw_losses) != set(task.environments):
            raise ValueError(f"checkpoint {key} environment order drift")
        expected_losses = _direct_losses(task.weights, diagonals, scale, codes)
        direct_losses[key] = expected_losses
        observed = np.asarray(
            [raw_losses[environment] for environment in task.environments],
            dtype=np.float64,
        )
        if not np.array_equal(observed, expected_losses):
            raise ValueError(f"checkpoint {key} arithmetic mismatch")
    expected_max = float(np.max(direct_losses["heldout_loss_by_environment"]))
    if policy.get("heldout_max_loss") != expected_max:
        raise ValueError("checkpoint heldout_max_loss arithmetic mismatch")
    runtime_seconds = policy.get("runtime_seconds")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, (int, float))
        or not np.isfinite(runtime_seconds)
        or runtime_seconds < 0.0
    ):
        raise ValueError("checkpoint policy runtime is invalid")
    selection = policy.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("checkpoint selection provenance is missing")
    expected_method = {
        "absmax": "range-derived-absmax-nearest-positive-fp16",
        "cliffquant_minimax": "exact-breakpoint-four-environment-minimax",
        "pooled_wmse": "exact-breakpoint-pooled-wmse",
    }[policy_name]
    if selection.get("method") != expected_method:
        raise ValueError("checkpoint selection method drift")
    if policy_name == "absmax":
        expected_bits, _ = _absmax_scale(task.weights)
        if scale_bits != expected_bits:
            raise ValueError("checkpoint AbsMax scale is not the frozen selection")
        if selection != {
            "method": expected_method,
            "objective": "none",
            "rounding_tie": "lower-fp16-bit-pattern",
        }:
            raise ValueError("checkpoint AbsMax selection provenance drift")
        return

    if set(selection) != {
        "method",
        "objective",
        "solver_provenance",
        "solver_returned_objective",
    }:
        raise ValueError("checkpoint exact-solver selection fields drift")
    calibration_losses = direct_losses["calibration_loss_by_environment"]
    if policy_name == "pooled_wmse":
        expected_selection_objective = float(np.mean(calibration_losses, dtype=np.float64))
        solver_curvature = np.mean(
            task.calibration,
            axis=0,
            dtype=np.float64,
        )[np.newaxis, :]
    else:
        expected_selection_objective = float(np.max(calibration_losses))
        solver_curvature = task.calibration
    if selection["objective"] != expected_selection_objective:
        raise ValueError("checkpoint selection objective arithmetic mismatch")

    solved = solve_breakpoint_exact(
        task.weights,
        solver_curvature,
        chunk_size=task.solver_chunk_size,
    )
    if scale_bits != solved.scale_bits or tuple(codes) != solved.codes:
        raise ValueError("checkpoint exact scale is not the current optimum")
    if selection["solver_returned_objective"] != solved.objective:
        raise ValueError("checkpoint solver-returned objective mismatch")
    if selection["solver_provenance"] != asdict(solved.provenance):
        raise ValueError("checkpoint exact-solver provenance mismatch")


def validate_group_checkpoint(record: Any, *, task: GroupEvaluationTask) -> dict[str, Any]:
    """Validate a resumed result against current weights, diagonals, and code semantics."""

    if not isinstance(record, dict):
        raise ValueError("checkpoint must be a JSON object")
    if set(record) != {
        "identity",
        "input_fingerprint",
        "paired_baseline_minus_cliff",
        "policies",
        "runtime_seconds",
        "schema",
        "weight_sha256",
    }:
        raise ValueError("checkpoint top-level fields drift")
    if record.get("schema") != "cliffquant.experiment-001.group-result.v1":
        raise ValueError("checkpoint schema mismatch")
    if record.get("identity") != task.coordinate.identity():
        raise ValueError("checkpoint group identity mismatch")
    if record.get("weight_sha256") != task.weight_sha256:
        raise ValueError("checkpoint weight hash mismatch")
    if record.get("input_fingerprint") != task.input_fingerprint:
        raise ValueError("checkpoint input fingerprint mismatch")
    runtime_seconds = record.get("runtime_seconds")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, (int, float))
        or not np.isfinite(runtime_seconds)
        or runtime_seconds < 0.0
    ):
        raise ValueError("checkpoint group runtime is invalid")
    policies = record.get("policies")
    if not isinstance(policies, dict) or tuple(sorted(policies)) != tuple(sorted(_POLICIES)):
        raise ValueError("checkpoint policy set mismatch")
    for policy in _POLICIES:
        _validate_policy_checkpoint(
            policies[policy],
            task=task,
            policy_name=policy,
        )
    paired = record.get("paired_baseline_minus_cliff")
    if not isinstance(paired, dict) or set(paired) != set(_BASELINES):
        raise ValueError("checkpoint paired-effect set mismatch")
    cliff_loss = policies["cliffquant_minimax"]["heldout_max_loss"]
    for baseline in _BASELINES:
        if paired[baseline] != policies[baseline]["heldout_max_loss"] - cliff_loss:
            raise ValueError("checkpoint paired-effect arithmetic mismatch")
    return record


def _load_checkpoint(path: Path, *, task: GroupEvaluationTask) -> dict[str, Any]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate checkpoint JSON key: {key}")
            value[key] = item
        return value

    try:
        record = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite checkpoint number: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read checkpoint {path}") from exc
    return validate_group_checkpoint(record, task=task)


def _build_tasks(
    groups: Sequence[SampledGroup],
    calibration: DiagonalArtifact,
    heldout: DiagonalArtifact,
    *,
    input_fingerprint: str,
    solver_chunk_size: int,
) -> tuple[GroupEvaluationTask, ...]:
    tasks: list[GroupEvaluationTask] = []
    for group in groups:
        coordinate = group.coordinate
        start = coordinate.input_start
        stop = coordinate.input_stop
        tasks.append(
            GroupEvaluationTask(
                coordinate=coordinate,
                weight_sha256=group.weight_sha256,
                weights=group.weights,
                calibration=np.ascontiguousarray(
                    calibration.normalized[coordinate.module][:, start:stop]
                ),
                heldout=np.ascontiguousarray(heldout.normalized[coordinate.module][:, start:stop]),
                environments=calibration.environments,
                input_fingerprint=input_fingerprint,
                solver_chunk_size=solver_chunk_size,
            )
        )
    return tuple(tasks)


def _source_hashes(protocol_path: Path) -> dict[str, dict[str, Any]]:
    package_dir = Path(__file__).resolve().parent
    canonical_protocol = package_dir.parents[1] / "research" / "EXPERIMENT_001_PROTOCOL.md"
    if protocol_path.resolve() != canonical_protocol.resolve():
        raise ValueError("Experiment 001 requires the canonical frozen protocol path")
    source_paths = {
        "activation_statistics": package_dir / "activation_stats.py",
        "breakpoint_solver": Path(solve_breakpoint_exact.__code__.co_filename).resolve(),
        "calibration_entrypoint": package_dir / "calibration_cli.py",
        "contracts": package_dir / "contracts.py",
        "corpus_contract": package_dir / "corpus.py",
        "corpus_replay": package_dir / "corpus_replay.py",
        "dataset_viewer_contract": package_dir / "dataset_viewer.py",
        "evaluation_engine": Path(__file__).resolve(),
        "fp16_grid": package_dir / "fp16_grid.py",
        "input_engine": Path(__file__).with_name("experiment001_io.py").resolve(),
        "model_snapshot": package_dir / "model_snapshot.py",
        "objective": package_dir / "objective.py",
        "provenance": package_dir / "provenance.py",
        "protocol": protocol_path.resolve(),
        "qwen35_inventory": package_dir / "qwen35_modules.py",
        "solver_certificate_validator": package_dir / "solver_certificate.py",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, path in source_paths.items():
        if not path.is_file():
            raise ValueError(f"missing frozen source file: {path}")
        result[name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _result_artifacts(
    output_dir: Path,
    results: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    jsonl_path = output_dir / "groups.jsonl"
    jsonl_payload = b"".join(canonical_json_bytes(result) for result in results)
    _atomic_write(jsonl_path, jsonl_payload)
    summary_path = output_dir / "summary.json"
    _atomic_write(summary_path, canonical_json_bytes(summary))
    artifact_paths = {
        "groups": jsonl_path,
        "run_inputs": output_dir / "run.inputs.json",
        "sample_manifest": output_dir / "sample.manifest.json",
        "summary": summary_path,
    }
    artifacts = {
        name: {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in artifact_paths.items()
    }
    hashes_path = output_dir / "SHA256SUMS.json"
    _atomic_write(hashes_path, canonical_json_bytes(artifacts))
    return artifacts


def run_experiment_001(
    *,
    calibration_json: str | Path,
    heldout_json: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    protocol_path: str | Path,
    solver_certificate: str | Path,
    workers: int,
    resume: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the explicitly requested, resumable 4,096-group proxy experiment."""

    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    started = time.perf_counter()
    solver_certificate_descriptor = validate_solver_certificate(solver_certificate)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration = load_diagonal_artifact(
        calibration_json,
        expected_phase="calibration",
    )
    heldout = load_diagonal_artifact(
        heldout_json,
        expected_phase="heldout",
    )
    validate_diagonal_pair(calibration, heldout)
    tokenizer = load_pinned_tokenizer()
    calibration_replay = verify_corpus_replay(
        calibration.metadata_path.parent,
        phase="calibration",
        tokenizer=tokenizer,
    )
    heldout_replay = verify_corpus_replay(
        heldout.metadata_path.parent,
        phase="heldout",
        tokenizer=tokenizer,
    )
    if (
        calibration_replay["manifest"]["sha256"] != calibration.corpus.manifest_sha256
        or heldout_replay["manifest"]["sha256"] != heldout.corpus.manifest_sha256
    ):
        raise ValueError("corpus replay descriptors disagree with diagonal corpus artifacts")
    store = SafetensorWeightStore(model_dir)
    if store.modules != calibration.modules:
        raise ValueError("base-model safetensors inventory disagrees with diagonal inventory")
    if calibration.model_source != store.source or heldout.model_source != store.source:
        raise ValueError("diagonal model bytes disagree with the evaluation checkpoint")

    coordinates = deterministic_group_coordinates(store.modules)
    groups = materialize_sample(store, coordinates)
    source = {
        "artifacts": {
            "calibration": calibration.source_descriptor(),
            "calibration_corpus_replay": calibration_replay,
            "heldout": heldout.source_descriptor(),
            "heldout_corpus_replay": heldout_replay,
            "model": store.source,
        },
        "engine_version": EXPERIMENT_001_ENGINE_VERSION,
        "environments": list(EXPERIMENT_001_ENVIRONMENTS),
        "model_revision": EXPERIMENT_001_MODEL_REVISION,
        "policies": {
            "absmax": "range-derived-absmax-nearest-positive-fp16",
            "cliffquant_minimax": "exact-four-environment-minimax",
            "pooled_wmse": "exact-mean-normalized-calibration-diagonal",
        },
        "quantizer": {
            "group_size": EXPERIMENT_001_GROUP_SIZE,
            "qmax": 7,
            "qmin": -8,
            "rounding": "nearest-even",
            "stored_scale": "positive-finite-fp16",
        },
        "sample_size": EXPERIMENT_001_SAMPLE_SIZE,
        "seed": EXPERIMENT_001_SEED,
        "solver_chunk_size": EXPERIMENT_001_SOLVER_CHUNK_SIZE,
        "solver_certificate": solver_certificate_descriptor,
        "sources": _source_hashes(Path(protocol_path)),
        "runtime": runtime_metadata(device="cpu"),
    }
    source_fingerprint = sha256_bytes(canonical_json_bytes(source))
    manifest = sample_manifest(groups)
    manifest["source_fingerprint"] = source_fingerprint
    sample_hash = _write_or_validate_frozen(output / "sample.manifest.json", manifest)
    fingerprint_payload = {
        "sample_manifest_sha256": sample_hash,
        "source_fingerprint": source_fingerprint,
    }
    input_fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_payload))
    run_inputs = {
        **source,
        "input_fingerprint": input_fingerprint,
        "sample_manifest": {
            "file": "sample.manifest.json",
            "sha256": sample_hash,
            "size_bytes": (output / "sample.manifest.json").stat().st_size,
        },
        "schema": "cliffquant.experiment-001.run-inputs.v1",
    }
    _write_or_validate_frozen(output / "run.inputs.json", run_inputs)

    tasks = _build_tasks(
        groups,
        calibration,
        heldout,
        input_fingerprint=input_fingerprint,
        solver_chunk_size=EXPERIMENT_001_SOLVER_CHUNK_SIZE,
    )
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    completed: dict[int, dict[str, Any]] = {}
    missing: list[GroupEvaluationTask] = []
    for task in tasks:
        checkpoint = _checkpoint_path(checkpoint_dir, task.coordinate)
        if checkpoint.exists():
            if not resume:
                raise ValueError(f"checkpoint exists while resume is disabled: {checkpoint}")
            completed[task.coordinate.sample_index] = _load_checkpoint(
                checkpoint,
                task=task,
            )
        else:
            missing.append(task)
    if progress is not None:
        progress(f"validated {len(completed)} checkpoints; {len(missing)} groups remain")

    report_interval = max(1, len(tasks) // 20)

    def accept(result: dict[str, Any], task: GroupEvaluationTask) -> None:
        validated = validate_group_checkpoint(result, task=task)
        checkpoint = _checkpoint_path(checkpoint_dir, task.coordinate)
        _atomic_write(checkpoint, canonical_json_bytes(validated))
        completed[task.coordinate.sample_index] = validated
        count = len(completed)
        if progress is not None and (count == len(tasks) or count % report_interval == 0):
            progress(f"completed {count}/{len(tasks)} groups")

    if workers == 1:
        for task in missing:
            accept(evaluate_group(task), task)
    elif missing:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            future_to_task = {executor.submit(evaluate_group, task): task for task in missing}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"group worker failed for {task.coordinate.group_id}"
                    ) from exc
                accept(result, task)

    ordered_results = [completed[index] for index in range(len(tasks))]
    aggregate = aggregate_group_results(ordered_results)
    if aggregate["group_count"] != EXPERIMENT_001_SAMPLE_SIZE or aggregate["module_count"] != len(
        store.modules
    ):
        raise RuntimeError("completed result inventory does not match the frozen sample")
    summary = {
        **aggregate,
        "input_fingerprint": input_fingerprint,
        "runtime": {
            "group_runtime_seconds_sum": float(
                np.sum(
                    [result["runtime_seconds"] for result in ordered_results],
                    dtype=np.float64,
                )
            ),
            "host": runtime_metadata(device="cpu"),
            "wall_seconds": time.perf_counter() - started,
            "workers": workers,
        },
    }
    artifacts = _result_artifacts(output, ordered_results, summary)
    if progress is not None:
        progress(
            "proxy gate "
            f"{'passed' if summary['proxy_gate_passed'] else 'failed'}; "
            f"artifacts written to {output}"
        )
    return {
        "artifacts": artifacts,
        "input_fingerprint": input_fingerprint,
        "proxy_gate_passed": summary["proxy_gate_passed"],
        "summary": summary,
    }


def _default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the explicitly frozen CliffQuant Experiment 001 real-group proxy comparison."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required safety flag; importing or inspecting this CLI never starts the experiment",
    )
    parser.add_argument("--calibration-json", required=True, type=Path)
    parser.add_argument("--heldout-json", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--solver-certificate",
        required=True,
        type=Path,
        help="required v2 zero-mismatch 10,000-case exact-solver certificate",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "research" / "EXPERIMENT_001_PROTOCOL.md",
    )
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="reject any existing per-group checkpoints instead of resuming them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("refusing to run without the explicit --execute flag")
    result = run_experiment_001(
        calibration_json=args.calibration_json,
        heldout_json=args.heldout_json,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        solver_certificate=args.solver_certificate,
        workers=args.workers,
        resume=not args.fresh,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "summary"}, indent=2))
    return 0 if result["proxy_gate_passed"] else 2
