"""Reproduce phase-2 quality and one-group runtime evidence."""

from __future__ import annotations

import json
import platform
import time

import numpy as np

from cliffquant import QuantizerSpec, solve_approximate, solve_exact

QUALITY_CASES = 512
QUALITY_SEED = 23039
RUNTIME_SEED = 2339
RUNTIME_REPETITIONS = 25


def _quality_case(
    rng: np.random.Generator,
    case: int,
) -> tuple[np.ndarray, np.ndarray, QuantizerSpec, str]:
    group_size = int(rng.integers(4, 65))
    environment_count = int(rng.integers(1, 5))
    mode = case % 6
    mode_name = (
        "normal",
        "single-outlier-x50",
        "log-uniform-magnitude",
        "half-step-cluster",
        "sparse-curvature",
        "heavy-tail-environment-scaled",
    )[mode]

    if mode == 0:
        weights = rng.normal(size=group_size)
    elif mode == 1:
        weights = rng.normal(size=group_size)
        weights[int(rng.integers(group_size))] *= 50.0
    elif mode == 2:
        weights = np.sign(rng.normal(size=group_size)) * np.exp2(
            rng.uniform(-10.0, 5.0, size=group_size)
        )
    elif mode == 3:
        base = float(np.exp2(rng.uniform(-4.0, 2.0)))
        weights = (rng.integers(-9, 10, size=group_size) + 0.5) * base
    elif mode == 4:
        weights = rng.normal(size=group_size)
    else:
        weights = rng.standard_t(df=2.5, size=group_size)

    curvature = rng.uniform(0.01, 2.0, size=(environment_count, group_size))
    if mode == 4:
        curvature[rng.random(size=curvature.shape) < 0.7] = 0.0
        for environment in curvature:
            if not np.any(environment):
                environment[int(rng.integers(group_size))] = 1.0
    if mode == 5:
        curvature *= np.exp2(rng.uniform(-4.0, 4.0, size=(environment_count, 1)))

    qmin, qmax = ((-2, 1), (-4, 3), (-8, 7))[case % 3]
    return weights, curvature, QuantizerSpec(qmin=qmin, qmax=qmax), mode_name


def quality_evidence() -> dict[str, object]:
    rng = np.random.default_rng(QUALITY_SEED)
    exact_scale_matches = 0
    objective_matches = 0
    gaps: list[float] = []
    candidate_counts: list[int] = []
    start = time.perf_counter()

    for case in range(QUALITY_CASES):
        weights, curvature, spec, _ = _quality_case(rng, case)
        approximate = solve_approximate(weights, curvature, spec)
        exact = solve_exact(weights, curvature, spec)
        exact_scale_matches += approximate.scale_bits == exact.scale_bits
        gap = (approximate.objective - exact.objective) / max(abs(exact.objective), 1e-15)
        gaps.append(gap)
        objective_matches += abs(gap) <= 1e-14
        candidate_counts.append(approximate.candidates_evaluated)

    elapsed = time.perf_counter() - start
    gap_array = np.asarray(gaps)
    count_array = np.asarray(candidate_counts)
    return {
        "cases": QUALITY_CASES,
        "seed": QUALITY_SEED,
        "exact_scale_matches": exact_scale_matches,
        "exact_scale_match_rate": exact_scale_matches / QUALITY_CASES,
        "objective_matches_at_1e-14": objective_matches,
        "median_relative_objective_gap": float(np.median(gap_array)),
        "p95_relative_objective_gap": float(np.quantile(gap_array, 0.95)),
        "worst_relative_objective_gap": float(np.max(gap_array)),
        "mean_candidates": float(np.mean(count_array)),
        "max_candidates": int(np.max(count_array)),
        "elapsed_seconds": elapsed,
    }


def runtime_evidence() -> dict[str, object]:
    rng = np.random.default_rng(RUNTIME_SEED)
    weights = rng.normal(size=128)
    curvature = rng.uniform(0.0, 2.0, size=(4, 128))
    solve_approximate(weights, curvature)
    solve_exact(weights, curvature)

    approximate_times: list[float] = []
    exact_times: list[float] = []
    approximate = None
    exact = None
    for _ in range(RUNTIME_REPETITIONS):
        start = time.perf_counter()
        approximate = solve_approximate(weights, curvature)
        approximate_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        exact = solve_exact(weights, curvature)
        exact_times.append(time.perf_counter() - start)

    if approximate is None or exact is None:  # pragma: no cover
        raise RuntimeError("runtime repetitions must be positive")

    approximate_median = float(np.median(approximate_times))
    exact_median = float(np.median(exact_times))
    return {
        "seed": RUNTIME_SEED,
        "group_size": 128,
        "environments": 4,
        "repetitions": RUNTIME_REPETITIONS,
        "approximate_median_seconds": approximate_median,
        "approximate_min_seconds": min(approximate_times),
        "approximate_max_seconds": max(approximate_times),
        "exact_median_seconds": exact_median,
        "exact_min_seconds": min(exact_times),
        "exact_max_seconds": max(exact_times),
        "median_speedup": exact_median / approximate_median,
        "approximate_candidates": approximate.candidates_evaluated,
        "exact_candidates": exact.scales_evaluated,
        "same_scale": approximate.scale_bits == exact.scale_bits,
        "relative_objective_gap": (
            (approximate.objective - exact.objective) / max(abs(exact.objective), 1e-15)
        ),
    }


def main() -> None:
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "quality": quality_evidence(),
                "runtime": runtime_evidence(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
