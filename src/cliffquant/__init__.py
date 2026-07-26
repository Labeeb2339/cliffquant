"""Correctness-first exhaustive FP16 scale selection."""

from cliffquant.contracts import QuantizerSpec, ScaleEvaluation, SolveResult
from cliffquant.fp16_grid import (
    FP16_MAX_FINITE_BITS,
    FP16_MIN_POSITIVE_BITS,
    FP16_POSITIVE_FINITE_COUNT,
    bits_to_scale,
    positive_finite_fp16_bits,
    positive_finite_fp16_scales,
    scale_to_bits,
)
from cliffquant.objective import diagonal_losses, evaluate_scale, quantize_codes
from cliffquant.solver import solve_exact

__all__ = [
    "FP16_MAX_FINITE_BITS",
    "FP16_MIN_POSITIVE_BITS",
    "FP16_POSITIVE_FINITE_COUNT",
    "QuantizerSpec",
    "ScaleEvaluation",
    "SolveResult",
    "bits_to_scale",
    "diagonal_losses",
    "evaluate_scale",
    "positive_finite_fp16_bits",
    "positive_finite_fp16_scales",
    "quantize_codes",
    "scale_to_bits",
    "solve_exact",
]

__version__ = "0.1.0"
