"""Public contracts and strict input validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class QuantizerSpec:
    """The fixed integer quantizer used during scale search."""

    qmin: int = -8
    qmax: int = 7
    rounding: str = "nearest_even"
    scale_dtype: str = "float16"

    def __post_init__(self) -> None:
        if type(self.qmin) is not int or type(self.qmax) is not int:
            raise TypeError("qmin and qmax must be integers, not booleans or floats")
        if not (-128 <= self.qmin < 0 < self.qmax <= 127):
            raise ValueError("quantization range must satisfy -128 <= qmin < 0 < qmax <= 127")
        if self.rounding != "nearest_even":
            raise ValueError("only round-to-nearest, ties-to-even is supported")
        if self.scale_dtype != "float16":
            raise ValueError("only positive finite float16 stored scales are supported")


DEFAULT_SPEC = QuantizerSpec()


@dataclass(frozen=True, slots=True)
class ScaleEvaluation:
    """Direct evaluation of one stored scale."""

    scale: float
    scale_bits: int
    codes: tuple[int, ...]
    per_environment_loss: tuple[float, ...]
    objective: float


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Result of a complete target-scale search."""

    scale: float
    scale_bits: int
    codes: tuple[int, ...]
    per_environment_loss: tuple[float, ...]
    objective: float
    scales_evaluated: int
    chunk_size: int


@dataclass(frozen=True, slots=True)
class ApproximateSearchProvenance:
    """Auditable description of one deterministic approximate search."""

    method: str
    no_clipping_scale: float
    coarse_log2_alpha_min: float
    coarse_log2_alpha_max: float
    coarse_log2_alpha_step: float
    coarse_scale_bits: tuple[int, ...]
    basin_seed_bits: tuple[int, ...]
    basin_initial_bounds: tuple[tuple[int, int], ...]
    refinement_levels: int
    refinement_points_per_level: int
    candidate_scale_bits: tuple[int, ...]
    best_origins: tuple[str, ...]
    degenerate_reason: str | None


@dataclass(frozen=True, slots=True)
class ApproximateSolveResult:
    """Result of the deterministic non-exhaustive search."""

    scale: float
    scale_bits: int
    codes: tuple[int, ...]
    per_environment_loss: tuple[float, ...]
    objective: float
    candidates_evaluated: int
    coarse_candidates_evaluated: int
    refinement_candidates_evaluated: int
    coarse_best_objective: float
    provenance: ApproximateSearchProvenance


@dataclass(frozen=True, slots=True)
class PreparedProblem:
    """Validated contiguous arrays for internal numerical code."""

    weights: NDArray[np.float64]
    curvature: NDArray[np.float64]


def _float64_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        source = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular collection of real numbers") from exc
    if source.dtype.kind in {"b", "c", "S", "U"}:
        raise TypeError(f"{name} must contain real numbers")
    try:
        array = source.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular collection of real numbers") from exc
    return np.ascontiguousarray(array)


def prepare_weights(weights: ArrayLike) -> NDArray[np.float64]:
    """Validate a non-empty, finite, one-dimensional weight group."""

    array = _float64_array(weights, name="weights")
    if array.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if array.size == 0:
        raise ValueError("weights must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("weights must contain only finite values")
    array.setflags(write=False)
    return array


def prepare_problem(weights: ArrayLike, curvature: ArrayLike) -> PreparedProblem:
    """Validate weights and a non-empty environment-by-weight curvature matrix."""

    weight_array = prepare_weights(weights)
    curvature_array = _float64_array(curvature, name="curvature")
    if curvature_array.ndim != 2:
        raise ValueError("curvature must be two-dimensional")
    if curvature_array.shape[0] == 0:
        raise ValueError("curvature must contain at least one environment")
    if curvature_array.shape[1] != weight_array.size:
        raise ValueError("each curvature environment must match the weight-group length")
    if not np.all(np.isfinite(curvature_array)):
        raise ValueError("curvature must contain only finite values")
    if np.any(curvature_array < 0.0):
        raise ValueError("curvature must be non-negative")
    curvature_array.setflags(write=False)
    return PreparedProblem(weights=weight_array, curvature=curvature_array)


def validate_chunk_size(chunk_size: int) -> int:
    """Validate the bounded-memory search chunk size."""

    if type(chunk_size) is not int:
        raise TypeError("chunk_size must be an integer, not a boolean or float")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return chunk_size


def validate_spec(spec: QuantizerSpec) -> QuantizerSpec:
    """Reject objects that only happen to expose similarly named attributes."""

    if not isinstance(spec, QuantizerSpec):
        raise TypeError("spec must be a QuantizerSpec")
    return spec
