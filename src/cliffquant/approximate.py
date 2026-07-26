"""Deterministic accelerated scale search with an explicit approximate boundary."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cliffquant.contracts import (
    DEFAULT_SPEC,
    ApproximateSearchProvenance,
    ApproximateSolveResult,
    PreparedProblem,
    QuantizerSpec,
    prepare_problem,
    validate_spec,
)
from cliffquant.fp16_grid import (
    FP16_MAX_FINITE_BITS,
    FP16_MIN_POSITIVE_BITS,
    bits_to_scale,
    positive_finite_fp16_scales,
)
from cliffquant.objective import _losses_prepared, _quantize_prepared

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

_METHOD = "fixed-alpha-multibasin-fp16-v1"
_COARSE_LOG2_ALPHA_MIN = -12.0
_COARSE_LOG2_ALPHA_MAX = 1.0
_COARSE_LOG2_ALPHA_STEP = 0.25
_BASIN_COUNT = 4
_REFINEMENT_LEVELS = 5
_REFINEMENT_POINTS = 9
_MAX_DENSE_BASIN_WIDTH = 2048

_TARGET_SCALES = positive_finite_fp16_scales()
_TARGET_SCALES.setflags(write=False)


def _no_clipping_scale(weights: NDArray[np.float64], spec: QuantizerSpec) -> float:
    positive = weights[weights > 0.0]
    negative = weights[weights < 0.0]
    positive_scale = float(np.max(positive)) / spec.qmax if positive.size else 0.0
    negative_scale = float(np.max(-negative)) / (-spec.qmin) if negative.size else 0.0
    return max(positive_scale, negative_scale)


def _nearest_scale_bits(value: float) -> int:
    if not math.isfinite(value) or value >= _TARGET_SCALES[-1]:
        return FP16_MAX_FINITE_BITS
    if value <= _TARGET_SCALES[0]:
        return FP16_MIN_POSITIVE_BITS

    upper_index = int(np.searchsorted(_TARGET_SCALES, value, side="left"))
    lower_index = upper_index - 1
    lower = float(_TARGET_SCALES[lower_index])
    upper = float(_TARGET_SCALES[upper_index])
    if value - lower <= upper - value:
        return lower_index + FP16_MIN_POSITIVE_BITS
    return upper_index + FP16_MIN_POSITIVE_BITS


def _score_bits(
    problem: PreparedProblem,
    spec: QuantizerSpec,
    scale_bits: Iterable[int],
) -> dict[int, float]:
    ordered_bits = np.asarray(sorted(set(scale_bits)), dtype="<u2")
    if ordered_bits.size == 0:
        return {}
    scales = ordered_bits.view("<f2").astype(np.float64)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratios = problem.weights[np.newaxis, :] / scales[:, np.newaxis]
        rounded = np.rint(ratios)
    codes = np.clip(rounded, spec.qmin, spec.qmax).astype(np.int16)
    residual = problem.weights[np.newaxis, :] - scales[:, np.newaxis] * codes.astype(np.float64)
    squared = residual * residual

    objectives = np.full(scales.size, -np.inf, dtype=np.float64)
    for diagonal in problem.curvature:
        losses = np.sum(
            squared * diagonal[np.newaxis, :],
            axis=1,
            dtype=np.float64,
        )
        objectives = np.maximum(objectives, losses)
    if not np.all(np.isfinite(objectives)):
        raise ArithmeticError("proxy evaluation overflowed; rescale or reject this input group")
    return {
        int(bits): float(objective)
        for bits, objective in zip(ordered_bits, objectives, strict=True)
    }


def _coarse_candidates(reference_scale: float) -> tuple[int, ...]:
    count = round((_COARSE_LOG2_ALPHA_MAX - _COARSE_LOG2_ALPHA_MIN) / _COARSE_LOG2_ALPHA_STEP)
    exponents = (
        _COARSE_LOG2_ALPHA_MIN + np.arange(count + 1, dtype=np.float64) * _COARSE_LOG2_ALPHA_STEP
    )
    with np.errstate(over="ignore", invalid="ignore"):
        values = reference_scale * np.exp2(exponents)
    return tuple(sorted({_nearest_scale_bits(float(value)) for value in values}))


def _coarse_local_minima(
    coarse_bits: tuple[int, ...],
    objectives: dict[int, float],
) -> tuple[int, ...]:
    values = [objectives[bits] for bits in coarse_bits]
    local_indices: list[int] = []
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and values[stop + 1] == values[start]:
            stop += 1
        left = values[start - 1] if start > 0 else float("inf")
        right = values[stop + 1] if stop + 1 < len(values) else float("inf")
        if values[start] <= left and values[start] <= right:
            local_indices.append(start)
        start = stop + 1

    ranked = sorted(local_indices, key=lambda index: (values[index], coarse_bits[index]))
    selected = ranked[:_BASIN_COUNT]

    if len(selected) < min(_BASIN_COUNT, len(coarse_bits)):
        global_ranked = sorted(
            range(len(coarse_bits)),
            key=lambda index: (values[index], coarse_bits[index]),
        )
        for index in global_ranked:
            if index in selected:
                continue
            if all(abs(index - chosen) > 1 for chosen in selected):
                selected.append(index)
            if len(selected) == min(_BASIN_COUNT, len(coarse_bits)):
                break

    if len(selected) < min(_BASIN_COUNT, len(coarse_bits)):
        for index in range(len(coarse_bits)):
            if index not in selected:
                selected.append(index)
            if len(selected) == min(_BASIN_COUNT, len(coarse_bits)):
                break

    return tuple(sorted(selected))


def _basin_bounds(coarse_bits: tuple[int, ...], seed_index: int) -> tuple[int, int]:
    lower = FP16_MIN_POSITIVE_BITS if seed_index == 0 else coarse_bits[seed_index - 1]
    if seed_index == len(coarse_bits) - 1:
        upper = FP16_MAX_FINITE_BITS
    else:
        upper = coarse_bits[seed_index + 1]
    return lower, upper


def _evenly_spaced_bits(lower: int, upper: int) -> tuple[int, ...]:
    width = upper - lower
    if width < _REFINEMENT_POINTS:
        return tuple(range(lower, upper + 1))
    denominator = _REFINEMENT_POINTS - 1
    return tuple(
        sorted(
            {
                lower + (point * width + denominator // 2) // denominator
                for point in range(_REFINEMENT_POINTS)
            }
        )
    )


def _make_result(
    problem: PreparedProblem,
    spec: QuantizerSpec,
    best_bits: int,
    objective_cache: dict[int, float],
    origins: dict[int, set[str]],
    coarse_bits: tuple[int, ...],
    basin_seed_bits: tuple[int, ...],
    basin_bounds: tuple[tuple[int, int], ...],
    reference_scale: float,
    degenerate_reason: str | None,
) -> ApproximateSolveResult:
    best_scale = bits_to_scale(best_bits)
    best_codes = _quantize_prepared(problem.weights, best_scale, spec)
    if degenerate_reason == "all-zero-curvature":
        best_losses = np.zeros(problem.curvature.shape[0], dtype=np.float64)
    else:
        best_losses = _losses_prepared(problem, best_scale, best_codes)
    coarse_best_objective = min(objective_cache[bits] for bits in coarse_bits)
    candidate_bits = tuple(sorted(objective_cache))
    provenance = ApproximateSearchProvenance(
        method=_METHOD,
        no_clipping_scale=reference_scale,
        coarse_log2_alpha_min=_COARSE_LOG2_ALPHA_MIN,
        coarse_log2_alpha_max=_COARSE_LOG2_ALPHA_MAX,
        coarse_log2_alpha_step=_COARSE_LOG2_ALPHA_STEP,
        coarse_scale_bits=coarse_bits,
        basin_seed_bits=basin_seed_bits,
        basin_initial_bounds=basin_bounds,
        refinement_levels=_REFINEMENT_LEVELS,
        refinement_points_per_level=_REFINEMENT_POINTS,
        candidate_scale_bits=candidate_bits,
        best_origins=tuple(sorted(origins[best_bits])),
        degenerate_reason=degenerate_reason,
    )
    return ApproximateSolveResult(
        scale=best_scale,
        scale_bits=best_bits,
        codes=tuple(int(code) for code in best_codes),
        per_environment_loss=tuple(float(loss) for loss in best_losses),
        objective=float(np.max(best_losses)),
        candidates_evaluated=len(candidate_bits),
        coarse_candidates_evaluated=len(coarse_bits),
        refinement_candidates_evaluated=len(candidate_bits) - len(coarse_bits),
        coarse_best_objective=coarse_best_objective,
        provenance=provenance,
    )


def solve_approximate(
    weights: ArrayLike,
    curvature: ArrayLike,
    spec: QuantizerSpec = DEFAULT_SPEC,
) -> ApproximateSolveResult:
    """Search fixed coarse alpha basins and refine them on the FP16 grid.

    This path is deterministic and directly scores the actual quantizer, but it
    is not exhaustive. Only ``solve_exact`` can verify that its result is the
    global optimum over the complete target grid.
    """

    validate_spec(spec)
    problem = prepare_problem(weights, curvature)
    reference_scale = _no_clipping_scale(problem.weights, spec)
    origins: dict[int, set[str]] = {}

    def register(bits: int, origin: str) -> None:
        origins.setdefault(bits, set()).add(origin)

    if reference_scale == 0.0:
        register(FP16_MIN_POSITIVE_BITS, "degenerate:all-zero-weights")
        cache = _score_bits(problem, spec, origins)
        return _make_result(
            problem,
            spec,
            FP16_MIN_POSITIVE_BITS,
            cache,
            origins,
            (FP16_MIN_POSITIVE_BITS,),
            (),
            (),
            reference_scale,
            "all-zero-weights",
        )

    if not np.any(problem.curvature):
        register(FP16_MIN_POSITIVE_BITS, "degenerate:all-zero-curvature")
        cache = {FP16_MIN_POSITIVE_BITS: 0.0}
        return _make_result(
            problem,
            spec,
            FP16_MIN_POSITIVE_BITS,
            cache,
            origins,
            (FP16_MIN_POSITIVE_BITS,),
            (),
            (),
            reference_scale,
            "all-zero-curvature",
        )

    coarse_bits = _coarse_candidates(reference_scale)
    for bits in coarse_bits:
        register(bits, "coarse-alpha")
    objective_cache = _score_bits(problem, spec, coarse_bits)

    seed_indices = _coarse_local_minima(coarse_bits, objective_cache)
    seed_bits = tuple(coarse_bits[index] for index in seed_indices)
    initial_bounds = tuple(_basin_bounds(coarse_bits, index) for index in seed_indices)

    for basin_number, (seed, bounds) in enumerate(zip(seed_bits, initial_bounds, strict=True)):
        initial_lower, initial_upper = bounds
        lower, upper = initial_lower, initial_upper
        center = seed
        register(center, f"basin:{basin_number}:seed")

        for level in range(_REFINEMENT_LEVELS):
            level_bits = set(_evenly_spaced_bits(lower, upper))
            level_bits.add(center)
            for bits in level_bits:
                register(bits, f"basin:{basin_number}:level:{level}")

            missing = level_bits.difference(objective_cache)
            objective_cache.update(_score_bits(problem, spec, missing))
            ordered = sorted(level_bits)
            center = min(ordered, key=lambda bits: (objective_cache[bits], bits))
            center_index = ordered.index(center)
            lower = ordered[max(0, center_index - 1)]
            upper = ordered[min(len(ordered) - 1, center_index + 1)]

        final_bits = set(range(lower, upper + 1))
        final_bits.add(center)
        for bits in final_bits:
            register(bits, f"basin:{basin_number}:final")
        missing = final_bits.difference(objective_cache)
        objective_cache.update(_score_bits(problem, spec, missing))

        if initial_upper - initial_lower <= _MAX_DENSE_BASIN_WIDTH:
            dense_bits = set(range(initial_lower, initial_upper + 1))
            for bits in dense_bits:
                register(bits, f"basin:{basin_number}:dense")
            missing = dense_bits.difference(objective_cache)
            objective_cache.update(_score_bits(problem, spec, missing))

    best_bits = min(
        objective_cache,
        key=lambda bits: (objective_cache[bits], bits),
    )
    return _make_result(
        problem,
        spec,
        best_bits,
        objective_cache,
        origins,
        coarse_bits,
        seed_bits,
        initial_bounds,
        reference_scale,
        None,
    )
