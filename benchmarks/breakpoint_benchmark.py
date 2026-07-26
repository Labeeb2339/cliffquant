"""Reproduce one-group exact breakpoint-versus-exhaustive runtime evidence."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable

import numpy as np

from cliffquant import solve_breakpoint_exact, solve_exact

RUNTIME_SEED = 20260726
RUNTIME_REPETITIONS = 31
WARMUP_REPETITIONS = 3


def _timed_call(function: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def runtime_evidence() -> dict[str, object]:
    generator = np.random.default_rng(RUNTIME_SEED)
    weights = generator.normal(0.0, 0.2, size=128)
    curvature = generator.lognormal(0.0, 0.5, size=(4, 128))

    def exact_call() -> object:
        return solve_exact(weights, curvature)

    def breakpoint_call() -> object:
        return solve_breakpoint_exact(weights, curvature)

    for _ in range(WARMUP_REPETITIONS):
        exact_call()
        breakpoint_call()

    exact_times: list[float] = []
    breakpoint_times: list[float] = []
    exact = None
    breakpoint = None
    for repetition in range(RUNTIME_REPETITIONS):
        ordered_calls = (
            (("exact", exact_call), ("breakpoint", breakpoint_call))
            if repetition % 2 == 0
            else (("breakpoint", breakpoint_call), ("exact", exact_call))
        )
        for name, function in ordered_calls:
            result, elapsed = _timed_call(function)
            if name == "exact":
                exact = result
                exact_times.append(elapsed)
            else:
                breakpoint = result
                breakpoint_times.append(elapsed)

    if exact is None or breakpoint is None:  # pragma: no cover
        raise RuntimeError("runtime repetitions must be positive")
    if (
        breakpoint.scale_bits,
        breakpoint.codes,
        breakpoint.per_environment_loss,
        breakpoint.objective,
    ) != (
        exact.scale_bits,
        exact.codes,
        exact.per_environment_loss,
        exact.objective,
    ):
        raise AssertionError("breakpoint and exhaustive solvers returned different results")

    exact_median = float(np.median(exact_times))
    breakpoint_median = float(np.median(breakpoint_times))
    return {
        "seed": RUNTIME_SEED,
        "group_size": 128,
        "environments": 4,
        "quantizer": "W4 signed, qmin=-8, qmax=7",
        "repetitions": RUNTIME_REPETITIONS,
        "exact_median_seconds": exact_median,
        "exact_p10_seconds": float(np.quantile(exact_times, 0.10)),
        "exact_p90_seconds": float(np.quantile(exact_times, 0.90)),
        "breakpoint_median_seconds": breakpoint_median,
        "breakpoint_p10_seconds": float(np.quantile(breakpoint_times, 0.10)),
        "breakpoint_p90_seconds": float(np.quantile(breakpoint_times, 0.90)),
        "median_speedup": exact_median / breakpoint_median,
        "exact_scales_evaluated": exact.scales_evaluated,
        "breakpoint_candidates_evaluated": breakpoint.candidates_evaluated,
        "constant_code_segments": breakpoint.provenance.constant_code_segments,
        "certificate_candidates_added": (breakpoint.provenance.certificate_candidates_added),
        "exhaustive_fallback": breakpoint.provenance.exhaustive_fallback,
        "same_result": True,
    }


def main() -> None:
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "runtime": runtime_evidence(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
