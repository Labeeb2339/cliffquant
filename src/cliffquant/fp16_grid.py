"""The positive finite IEEE binary16 target-scale grid."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

FP16_MIN_POSITIVE_BITS = 0x0001
FP16_MAX_FINITE_BITS = 0x7BFF
FP16_POSITIVE_FINITE_COUNT = FP16_MAX_FINITE_BITS - FP16_MIN_POSITIVE_BITS + 1

_BITS = np.arange(
    FP16_MIN_POSITIVE_BITS,
    FP16_MAX_FINITE_BITS + 1,
    dtype="<u2",
)
_SCALES = _BITS.view("<f2").astype(np.float64)
_BITS.setflags(write=False)
_SCALES.setflags(write=False)


def positive_finite_fp16_bits() -> NDArray[np.uint16]:
    """Return all positive finite FP16 bit patterns in increasing-value order."""

    return _BITS.copy()


def positive_finite_fp16_scales() -> NDArray[np.float64]:
    """Return all positive finite FP16 values widened exactly to float64."""

    return _SCALES.copy()


def bits_to_scale(bits: int) -> float:
    """Decode one positive finite FP16 bit pattern as a Python float."""

    if type(bits) is not int:
        raise TypeError("bits must be an integer")
    if not FP16_MIN_POSITIVE_BITS <= bits <= FP16_MAX_FINITE_BITS:
        raise ValueError("bits must encode a positive finite float16 value")
    encoded = np.asarray([bits], dtype="<u2")
    return float(encoded.view("<f2")[0])


def scale_to_bits(scale: float) -> int:
    """Return the bit pattern of an exactly representable positive finite FP16 scale."""

    if isinstance(scale, (bool, np.bool_)):
        raise TypeError("scale must be a real number, not a boolean")
    try:
        value = float(scale)
    except (TypeError, ValueError) as exc:
        raise TypeError("scale must be a real number") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("scale must be positive and finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stored = np.float16(value)
    widened = float(stored)
    if not np.isfinite(widened) or widened <= 0.0 or widened != value:
        raise ValueError("scale must already be exactly representable as positive finite float16")
    return int(np.asarray([stored], dtype="<f2").view("<u2")[0])


def iter_scale_chunks(
    chunk_size: int,
) -> Iterator[tuple[NDArray[np.uint16], NDArray[np.float64]]]:
    """Yield read-only views of the target grid in ascending chunks."""

    for start in range(0, FP16_POSITIVE_FINITE_COUNT, chunk_size):
        stop = min(start + chunk_size, FP16_POSITIVE_FINITE_COUNT)
        yield _BITS[start:stop], _SCALES[start:stop]
