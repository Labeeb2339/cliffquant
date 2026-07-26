"""Actual quantization and direct diagonal-proxy evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cliffquant.contracts import (
    DEFAULT_SPEC,
    PreparedProblem,
    QuantizerSpec,
    ScaleEvaluation,
    prepare_problem,
    prepare_weights,
    validate_spec,
)
from cliffquant.fp16_grid import scale_to_bits

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _quantize_prepared(
    weights: NDArray[np.float64],
    scale: float,
    spec: QuantizerSpec,
) -> NDArray[np.int16]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratios = weights / scale
        rounded = np.rint(ratios)
    return np.clip(rounded, spec.qmin, spec.qmax).astype(np.int16)


def quantize_codes(
    weights: ArrayLike,
    scale: float,
    spec: QuantizerSpec = DEFAULT_SPEC,
) -> tuple[int, ...]:
    """Quantize one group with RNE followed by clipping."""

    validate_spec(spec)
    weight_array = prepare_weights(weights)
    scale_to_bits(scale)
    codes = _quantize_prepared(weight_array, float(scale), spec)
    return tuple(int(code) for code in codes)


def _losses_prepared(
    problem: PreparedProblem,
    scale: float,
    codes: NDArray[np.int16],
) -> NDArray[np.float64]:
    residual = problem.weights - scale * codes.astype(np.float64)
    squared = residual * residual
    losses = np.sum(problem.curvature * squared[np.newaxis, :], axis=1, dtype=np.float64)
    if not np.all(np.isfinite(losses)):
        raise ArithmeticError("proxy evaluation overflowed; rescale or reject this input group")
    return losses


def diagonal_losses(
    weights: ArrayLike,
    curvature: ArrayLike,
    scale: float,
    spec: QuantizerSpec = DEFAULT_SPEC,
) -> tuple[float, ...]:
    """Compute each environment's diagonal reconstruction loss directly."""

    return evaluate_scale(weights, curvature, scale, spec).per_environment_loss


def evaluate_scale(
    weights: ArrayLike,
    curvature: ArrayLike,
    scale: float,
    spec: QuantizerSpec = DEFAULT_SPEC,
) -> ScaleEvaluation:
    """Quantize and directly score one exactly representable stored scale."""

    validate_spec(spec)
    problem = prepare_problem(weights, curvature)
    scale_bits = scale_to_bits(scale)
    stored_scale = float(scale)
    codes = _quantize_prepared(problem.weights, stored_scale, spec)
    losses = _losses_prepared(problem, stored_scale, codes)
    return ScaleEvaluation(
        scale=stored_scale,
        scale_bits=scale_bits,
        codes=tuple(int(code) for code in codes),
        per_environment_loss=tuple(float(loss) for loss in losses),
        objective=float(np.max(losses)),
    )
