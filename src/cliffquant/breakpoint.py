"""Exact minimax search over analytic code-transition segments.

The exhaustive reference evaluates every positive finite binary16 scale.  This
module instead partitions that same ordered grid wherever any scalar RNE code
changes.  On each constant-code run, every environment loss is a convex
quadratic in the stored scale.  The minimum of their upper envelope can only
occur at a run boundary, an individual quadratic vertex, or an intersection of
two quadratics.

All generated target-scale candidates are scored again with the direct
residual implementation.  The analytic coefficients are used only to locate
candidates; they are not used for the returned losses or objective.  A
conservative roundoff certificate retains any additional scale whose lower
bound can beat the direct incumbent.  A zero/subnormal winning objective or a
direct candidate tie activates a complete grid fallback, because IEEE
underflow can otherwise create a flat zero-loss interval whose first bit is far
from the real-arithmetic minimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cliffquant.contracts import (
    DEFAULT_SPEC,
    PreparedProblem,
    QuantizerSpec,
    prepare_problem,
    validate_chunk_size,
    validate_spec,
)
from cliffquant.fp16_grid import (
    bits_to_scale,
    positive_finite_fp16_scales,
)
from cliffquant.objective import _losses_prepared, _quantize_prepared

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


_CANDIDATE_NEIGHBORHOOD_RADIUS = 3
_SHORT_RUN_EXHAUSTIVE_LIMIT = 8
_ROUNDING_BOUND_SAFETY_FACTOR = 256.0


@dataclass(frozen=True, slots=True)
class BreakpointSearchProvenance:
    """Auditable counts and any numerical fallback from one search."""

    method: str
    scalar_transitions_considered: int
    in_grid_scalar_transitions: int
    distinct_transition_indices: int
    constant_code_segments: int
    boundary_candidates_considered: int
    vertex_candidates_considered: int
    intersection_candidates_considered: int
    convex_locator_candidates_considered: int
    certificate_candidates_added: int
    candidate_scales_evaluated: int
    exhaustive_fallback: bool
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class BreakpointSolveResult:
    """Result of the analytic exact search on the binary16 target grid."""

    scale: float
    scale_bits: int
    codes: tuple[int, ...]
    per_environment_loss: tuple[float, ...]
    objective: float
    candidates_evaluated: int
    chunk_size: int
    provenance: BreakpointSearchProvenance


@dataclass(frozen=True, slots=True)
class _TransitionPartition:
    starts: tuple[int, ...]
    scalar_transitions_considered: int
    in_grid_scalar_transitions: int


@dataclass(frozen=True, slots=True)
class _CandidateConstruction:
    indices: tuple[int, ...]
    direct_objective_lower_bound: NDArray[np.float64]
    provenance: BreakpointSearchProvenance


def _transition_predicate(
    magnitude: float,
    scale: float,
    level: int,
) -> bool:
    """Return whether RNE magnitude has crossed below ``level``.

    At the exact half-integer boundary, ties-to-even assigns the tie to
    ``level - 1`` when ``level`` is odd and to ``level`` when it is even.
    A widened binary16 scale multiplied by ``level - 0.5`` is exactly
    representable in binary64 for every supported level.
    """

    threshold = scale * (level - 0.5)
    if level % 2:
        return magnitude <= threshold
    return magnitude < threshold


def _first_lower_magnitude_index(
    magnitude: float,
    level: int,
    scales: NDArray[np.float64],
) -> int:
    """Locate the first grid index whose RNE magnitude is below ``level``."""

    factor = level - 0.5
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        boundary = magnitude / factor
    side = "left" if level % 2 else "right"
    index = int(np.searchsorted(scales, boundary, side=side))

    # The quotient above is only a locator.  These exact-product comparisons
    # correct either neighboring placement if the quotient rounded onto a grid
    # value, including a true RNE half tie.
    while index > 0 and _transition_predicate(
        magnitude,
        float(scales[index - 1]),
        level,
    ):
        index -= 1
    while index < scales.size and not _transition_predicate(
        magnitude,
        float(scales[index]),
        level,
    ):
        index += 1
    return index


def _partition_grid(
    weights: NDArray[np.float64],
    spec: QuantizerSpec,
    scales: NDArray[np.float64],
) -> _TransitionPartition:
    starts = {0}
    scalar_transitions_considered = 0
    in_grid_scalar_transitions = 0

    for weight in weights:
        value = float(weight)
        if value == 0.0:
            continue
        max_magnitude = spec.qmax if value > 0.0 else -spec.qmin
        magnitude = abs(value)
        for level in range(1, max_magnitude + 1):
            scalar_transitions_considered += 1
            index = _first_lower_magnitude_index(magnitude, level, scales)
            if 0 < index < scales.size:
                in_grid_scalar_transitions += 1
                starts.add(index)

    return _TransitionPartition(
        starts=tuple(sorted(starts)),
        scalar_transitions_considered=scalar_transitions_considered,
        in_grid_scalar_transitions=in_grid_scalar_transitions,
    )


def _real_quadratic_roots_batch(
    quadratic: NDArray[np.float64],
    linear: NDArray[np.float64],
    constant: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return two vectorized root slots per quadratic, with NaN for no root."""

    first = np.full(quadratic.shape, np.nan, dtype=np.float64)
    second = np.full(quadratic.shape, np.nan, dtype=np.float64)

    linear_mask = (quadratic == 0.0) & (linear != 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        first[linear_mask] = -constant[linear_mask] / linear[linear_mask]

    quadratic_mask = quadratic != 0.0
    if not np.any(quadratic_mask):
        return first, second

    normalization = np.maximum.reduce(
        (
            np.abs(quadratic),
            np.abs(linear),
            np.abs(constant),
        )
    )
    normalized_quadratic = np.zeros_like(quadratic)
    normalized_linear = np.zeros_like(linear)
    normalized_constant = np.zeros_like(constant)
    normalized_quadratic[quadratic_mask] = quadratic[quadratic_mask] / normalization[quadratic_mask]
    normalized_linear[quadratic_mask] = linear[quadratic_mask] / normalization[quadratic_mask]
    normalized_constant[quadratic_mask] = constant[quadratic_mask] / normalization[quadratic_mask]

    squared_linear = normalized_linear * normalized_linear
    cross = 4.0 * normalized_quadratic * normalized_constant
    discriminant = squared_linear - cross
    roundoff_allowance = 64.0 * np.finfo(np.float64).eps * (squared_linear + np.abs(cross))
    real_mask = quadratic_mask & (discriminant >= -roundoff_allowance)
    clipped_discriminant = np.maximum(discriminant, 0.0)
    root_discriminant = np.sqrt(clipped_discriminant)
    stable_numerator = -0.5 * (
        normalized_linear + np.copysign(root_discriminant, normalized_linear)
    )

    stable_mask = real_mask & (stable_numerator != 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        first[stable_mask] = stable_numerator[stable_mask] / normalized_quadratic[stable_mask]
        second[stable_mask] = normalized_constant[stable_mask] / stable_numerator[stable_mask]

    repeated_mask = stable_mask & (first == second)
    second[repeated_mask] = np.nan

    zero_numerator_mask = real_mask & (stable_numerator == 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        first[zero_numerator_mask] = -normalized_linear[zero_numerator_mask] / (
            2.0 * normalized_quadratic[zero_numerator_mask]
        )
    return first, second


def _add_candidate_neighborhoods(
    candidates: NDArray[np.bool_],
    continuous_scales: NDArray[np.float64],
    run_starts: NDArray[np.int64],
    run_ends: NDArray[np.int64],
    scales: NDArray[np.float64],
) -> None:
    """Map continuous candidates to neighboring target values in their runs."""

    finite = np.isfinite(continuous_scales)
    if not np.any(finite):
        return
    insertions = np.searchsorted(scales, continuous_scales, side="left")
    for offset in range(
        -_CANDIDATE_NEIGHBORHOOD_RADIUS - 1,
        _CANDIDATE_NEIGHBORHOOD_RADIUS + 1,
    ):
        indices = insertions + offset
        valid = (
            finite
            & (indices >= run_starts)
            & (indices <= run_ends)
            & (indices >= 0)
            & (indices < scales.size)
        )
        candidates[indices[valid]] = True


def _candidate_indices(
    problem: PreparedProblem,
    spec: QuantizerSpec,
    scales: NDArray[np.float64],
    partition: _TransitionPartition,
) -> _CandidateConstruction:
    squared_weights = problem.weights * problem.weights
    constant = np.sum(
        problem.curvature * squared_weights[np.newaxis, :],
        axis=1,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(constant)):
        raise ArithmeticError("proxy coefficients overflowed; rescale or reject this input group")

    starts = np.asarray(partition.starts, dtype=np.int64)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = scales.size - 1
    segment_count = starts.size

    candidates = np.zeros(scales.size, dtype=np.bool_)
    candidates[starts] = True
    candidates[ends] = True
    boundary_count = 2 * segment_count

    start_scales = scales[starts]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratios = problem.weights[np.newaxis, :] / start_scales[:, np.newaxis]
        rounded = np.rint(ratios)
    codes = np.clip(rounded, spec.qmin, spec.qmax).astype(np.int16)
    code_values = codes.astype(np.float64)
    curvature_transpose = problem.curvature.T
    quadratic = (code_values * code_values) @ curvature_transpose
    linear_half = (code_values * problem.weights[np.newaxis, :]) @ curvature_transpose
    if not np.all(np.isfinite(quadratic)) or not np.all(np.isfinite(linear_half)):
        raise ArithmeticError("proxy coefficients overflowed; rescale or reject this input group")

    environment_count = problem.curvature.shape[0]
    run_starts_by_environment = np.repeat(starts, environment_count)
    run_ends_by_environment = np.repeat(ends, environment_count)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        vertices = np.divide(
            linear_half,
            quadratic,
            out=np.full_like(linear_half, np.nan),
            where=quadratic > 0.0,
        )
    _add_candidate_neighborhoods(
        candidates,
        vertices.ravel(),
        run_starts_by_environment,
        run_ends_by_environment,
        scales,
    )
    vertex_count = int(np.count_nonzero(quadratic > 0.0))

    intersection_count = 0
    for left in range(environment_count):
        for right in range(left + 1, environment_count):
            first_roots, second_roots = _real_quadratic_roots_batch(
                quadratic[:, left] - quadratic[:, right],
                -2.0 * (linear_half[:, left] - linear_half[:, right]),
                np.full(segment_count, constant[left] - constant[right], dtype=np.float64),
            )
            intersection_count += int(np.count_nonzero(np.isfinite(first_roots)))
            intersection_count += int(np.count_nonzero(np.isfinite(second_roots)))
            _add_candidate_neighborhoods(
                candidates,
                first_roots,
                starts,
                ends,
                scales,
            )
            _add_candidate_neighborhoods(
                candidates,
                second_roots,
                starts,
                ends,
                scales,
            )

    # Evaluate the coefficient form over the complete one-dimensional grid.
    # This is O(grid * environments), not O(grid * group_size), and gives every
    # segment an independent discrete-convexity locator if a root is ill
    # conditioned in binary64.  Returned values are still direct-rescored.
    segment_by_grid_index = np.searchsorted(starts, np.arange(scales.size), side="right") - 1
    grid_quadratic = quadratic[segment_by_grid_index]
    grid_linear_half = linear_half[segment_by_grid_index]
    with np.errstate(over="ignore", invalid="ignore"):
        grid_losses = (grid_quadratic * scales[:, np.newaxis] - 2.0 * grid_linear_half) * scales[
            :, np.newaxis
        ] + constant[np.newaxis, :]
    grid_objectives = np.max(grid_losses, axis=1)
    if not np.all(np.isfinite(grid_objectives)):
        raise ArithmeticError("proxy coefficients overflowed; rescale or reject this input group")

    # Bound the discrepancy between the expanded coefficient form and the
    # direct residual/square/multiply/sum path.  The factor deliberately
    # exceeds the standard gamma_n count for both paths.  Any scale whose lower
    # bound can still beat the directly scored incumbent is retained below.
    operation_count = 4 * problem.weights.size + 32
    unit_roundoff_bound = np.finfo(np.float64).eps
    accumulated_roundoff = operation_count * unit_roundoff_bound
    if accumulated_roundoff >= 0.5:  # pragma: no cover - impossible for practical arrays
        direct_objective_lower_bound = np.full(scales.size, -np.inf, dtype=np.float64)
    else:
        gamma = accumulated_roundoff / (1.0 - accumulated_roundoff)
        with np.errstate(over="ignore", invalid="ignore"):
            grid_magnitudes = (
                grid_quadratic * scales[:, np.newaxis] + 2.0 * grid_linear_half
            ) * scales[:, np.newaxis] + constant[np.newaxis, :]
            uncertainty = _ROUNDING_BOUND_SAFETY_FACTOR * gamma * grid_magnitudes
        environment_lower_bounds = grid_losses - uncertainty
        direct_objective_lower_bound = np.max(environment_lower_bounds, axis=1)
        unsafe_subnormal_input = np.any(
            (problem.curvature > 0.0) & (problem.curvature < np.finfo(np.float64).tiny)
        ) or np.any(
            (np.abs(problem.weights) > 0.0) & (np.abs(problem.weights) < np.finfo(np.float64).tiny)
        )
        if unsafe_subnormal_input or not np.all(np.isfinite(direct_objective_lower_bound)):
            direct_objective_lower_bound = np.full(scales.size, -np.inf, dtype=np.float64)

    convex_locator_count = 0
    for run_start, run_end in zip(starts, ends, strict=True):
        run_length = int(run_end - run_start + 1)
        if run_length <= _SHORT_RUN_EXHAUSTIVE_LIMIT:
            candidates[run_start : run_end + 1] = True
            convex_locator_count += run_length
            continue
        local_index = int(run_start + np.argmin(grid_objectives[run_start : run_end + 1]))
        lower = max(int(run_start), local_index - _CANDIDATE_NEIGHBORHOOD_RADIUS)
        upper = min(int(run_end), local_index + _CANDIDATE_NEIGHBORHOOD_RADIUS)
        candidates[lower : upper + 1] = True
        convex_locator_count += 1

    ordered_candidates = tuple(int(index) for index in np.flatnonzero(candidates))
    provenance = BreakpointSearchProvenance(
        method="analytic-breakpoint-minimax-fp16-v0",
        scalar_transitions_considered=partition.scalar_transitions_considered,
        in_grid_scalar_transitions=partition.in_grid_scalar_transitions,
        distinct_transition_indices=max(0, segment_count - 1),
        constant_code_segments=segment_count,
        boundary_candidates_considered=boundary_count,
        vertex_candidates_considered=vertex_count,
        intersection_candidates_considered=intersection_count,
        convex_locator_candidates_considered=convex_locator_count,
        certificate_candidates_added=0,
        candidate_scales_evaluated=len(ordered_candidates),
        exhaustive_fallback=False,
        fallback_reason=None,
    )
    direct_objective_lower_bound.setflags(write=False)
    return _CandidateConstruction(
        indices=ordered_candidates,
        direct_objective_lower_bound=direct_objective_lower_bound,
        provenance=provenance,
    )


def _score_candidates(
    problem: PreparedProblem,
    spec: QuantizerSpec,
    scales: NDArray[np.float64],
    candidate_indices: tuple[int, ...],
    chunk_size: int,
) -> tuple[int, float, int]:
    best_objective = float("inf")
    best_index: int | None = None
    best_tie_count = 0

    index_array = np.asarray(candidate_indices, dtype=np.int64)
    for start in range(0, index_array.size, chunk_size):
        indices = index_array[start : start + chunk_size]
        scale_chunk = scales[indices]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ratios = problem.weights[np.newaxis, :] / scale_chunk[:, np.newaxis]
            rounded = np.rint(ratios)
        codes = np.clip(rounded, spec.qmin, spec.qmax).astype(np.int16)
        residual = problem.weights[np.newaxis, :] - scale_chunk[:, np.newaxis] * codes.astype(
            np.float64
        )
        squared = residual * residual
        losses = np.empty(
            (scale_chunk.size, problem.curvature.shape[0]),
            dtype=np.float64,
        )
        for environment_index, diagonal in enumerate(problem.curvature):
            losses[:, environment_index] = np.sum(
                squared * diagonal[np.newaxis, :],
                axis=1,
                dtype=np.float64,
            )
        objectives = np.max(losses, axis=1)
        if not np.all(np.isfinite(objectives)):
            raise ArithmeticError("proxy evaluation overflowed; rescale or reject this input group")

        local_offset = int(np.argmin(objectives))
        local_objective = float(objectives[local_offset])
        local_tie_count = int(np.count_nonzero(objectives == local_objective))
        if local_objective < best_objective:
            best_objective = local_objective
            best_index = int(indices[local_offset])
            best_tie_count = local_tie_count
        elif local_objective == best_objective:
            best_tie_count += local_tie_count

    if best_index is None:  # pragma: no cover - run boundaries make the set non-empty
        raise RuntimeError("analytic candidate construction unexpectedly returned no scales")
    return best_index, best_objective, best_tie_count


def solve_breakpoint_exact(
    weights: ArrayLike,
    curvature: ArrayLike,
    spec: QuantizerSpec = DEFAULT_SPEC,
    *,
    chunk_size: int = 4096,
) -> BreakpointSolveResult:
    """Return the exact minimax scale using analytic constant-code segments.

    Target scales, quantizer semantics, direct loss arithmetic, and final
    lower-bit tie breaking are the same as :func:`cliffquant.solver.solve_exact`.
    ``chunk_size`` only bounds memory during direct candidate rescoring.
    Numerically flat candidate optima are conservatively rescored exhaustively.
    """

    validate_spec(spec)
    problem = prepare_problem(weights, curvature)
    validated_chunk_size = validate_chunk_size(chunk_size)
    scales = positive_finite_fp16_scales()

    partition = _partition_grid(problem.weights, spec, scales)
    construction = _candidate_indices(
        problem,
        spec,
        scales,
        partition,
    )
    candidate_indices = construction.indices
    provenance = construction.provenance
    best_index, best_objective, best_tie_count = _score_candidates(
        problem,
        spec,
        scales,
        candidate_indices,
        validated_chunk_size,
    )

    fallback_reason: str | None = None
    if best_objective < np.finfo(np.float64).tiny:
        fallback_reason = "zero-or-subnormal-objective"
    elif best_tie_count > 1:
        fallback_reason = "direct-candidate-objective-tie"

    if fallback_reason is None:
        certification_indices = np.flatnonzero(
            construction.direct_objective_lower_bound <= best_objective
        )
        existing = np.zeros(scales.size, dtype=np.bool_)
        existing[np.asarray(candidate_indices, dtype=np.int64)] = True
        added = certification_indices[~existing[certification_indices]]
        if added.size:
            candidate_indices = tuple(
                int(index)
                for index in np.union1d(
                    np.asarray(candidate_indices, dtype=np.int64),
                    added,
                )
            )
            best_index, best_objective, best_tie_count = _score_candidates(
                problem,
                spec,
                scales,
                candidate_indices,
                validated_chunk_size,
            )
            provenance = replace(
                provenance,
                certificate_candidates_added=int(added.size),
                candidate_scales_evaluated=len(candidate_indices),
            )
            if best_objective < np.finfo(np.float64).tiny:
                fallback_reason = "zero-or-subnormal-objective"
            elif best_tie_count > 1:
                fallback_reason = "direct-candidate-objective-tie"

    if fallback_reason is not None:
        if len(candidate_indices) < scales.size:
            candidate_indices = tuple(range(scales.size))
            best_index, best_objective, _ = _score_candidates(
                problem,
                spec,
                scales,
                candidate_indices,
                validated_chunk_size,
            )
        provenance = replace(
            provenance,
            candidate_scales_evaluated=len(candidate_indices),
            exhaustive_fallback=True,
            fallback_reason=fallback_reason,
        )

    best_bits = best_index + 1
    best_scale = bits_to_scale(best_bits)
    best_codes = _quantize_prepared(problem.weights, best_scale, spec)
    best_losses = _losses_prepared(problem, best_scale, best_codes)
    return BreakpointSolveResult(
        scale=best_scale,
        scale_bits=best_bits,
        codes=tuple(int(code) for code in best_codes),
        per_environment_loss=tuple(float(loss) for loss in best_losses),
        objective=float(np.max(best_losses)),
        candidates_evaluated=len(candidate_indices),
        chunk_size=validated_chunk_size,
        provenance=provenance,
    )
