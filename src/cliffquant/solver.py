"""Correctness-first exhaustive target-scale solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cliffquant.contracts import (
    DEFAULT_SPEC,
    QuantizerSpec,
    SolveResult,
    prepare_problem,
    validate_chunk_size,
    validate_spec,
)
from cliffquant.fp16_grid import FP16_POSITIVE_FINITE_COUNT, bits_to_scale, iter_scale_chunks
from cliffquant.objective import _losses_prepared, _quantize_prepared

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def solve_exact(
    weights: ArrayLike,
    curvature: ArrayLike,
    spec: QuantizerSpec = DEFAULT_SPEC,
    *,
    chunk_size: int = 1024,
) -> SolveResult:
    """Evaluate every positive finite FP16 scale and return the best one.

    Scales are visited in ascending bit-pattern order. The global winner changes
    only for a strictly lower float64 objective, so an exact objective tie
    deterministically keeps the smaller FP16 bit pattern.
    """

    validate_spec(spec)
    problem = prepare_problem(weights, curvature)
    validated_chunk_size = validate_chunk_size(chunk_size)

    best_objective = float("inf")
    best_bits: int | None = None

    for bit_chunk, scale_chunk in iter_scale_chunks(validated_chunk_size):
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

        local_index = int(np.argmin(objectives))
        local_objective = float(objectives[local_index])
        if local_objective < best_objective:
            best_objective = local_objective
            best_bits = int(bit_chunk[local_index])

    if best_bits is None:  # pragma: no cover - the fixed grid is non-empty
        raise RuntimeError("the FP16 scale grid was unexpectedly empty")

    best_scale = bits_to_scale(best_bits)
    best_codes = _quantize_prepared(problem.weights, best_scale, spec)
    best_losses = _losses_prepared(problem, best_scale, best_codes)

    return SolveResult(
        scale=best_scale,
        scale_bits=best_bits,
        codes=tuple(int(code) for code in best_codes),
        per_environment_loss=tuple(float(loss) for loss in best_losses),
        objective=float(np.max(best_losses)),
        scales_evaluated=FP16_POSITIVE_FINITE_COUNT,
        chunk_size=validated_chunk_size,
    )
