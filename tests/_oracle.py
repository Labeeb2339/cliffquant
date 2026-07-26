"""Independent standard-library oracle.

This module intentionally imports no CliffQuant production code.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OracleResult:
    scale: float
    scale_bits: int
    codes: tuple[int, ...]
    per_environment_loss: tuple[float, ...]
    objective: float


def _decode_half(bits: int) -> float:
    return float(struct.unpack("<e", struct.pack("<H", bits))[0])


def _code(value: float, scale: float, qmin: int, qmax: int) -> int:
    ratio = value / scale
    if ratio >= qmax + 0.5:
        return qmax
    if ratio <= qmin - 0.5:
        return qmin
    return max(qmin, min(qmax, round(ratio)))


def _score(
    weights: Sequence[float],
    curvature: Sequence[Sequence[float]],
    scale: float,
    qmin: int,
    qmax: int,
) -> tuple[tuple[int, ...], tuple[float, ...], float]:
    codes = tuple(_code(weight, scale, qmin, qmax) for weight in weights)
    losses = tuple(
        math.fsum(
            diagonal[index] * (weight - scale * codes[index]) ** 2
            for index, weight in enumerate(weights)
        )
        for diagonal in curvature
    )
    return codes, losses, max(losses)


def exhaustive_oracle(
    weights: Iterable[float],
    curvature: Iterable[Iterable[float]],
    qmin: int = -8,
    qmax: int = 7,
) -> OracleResult:
    frozen_weights = tuple(float(value) for value in weights)
    frozen_curvature = tuple(tuple(float(value) for value in row) for row in curvature)

    best_bits = 1
    best_scale = _decode_half(best_bits)
    best_codes, best_losses, best_objective = _score(
        frozen_weights,
        frozen_curvature,
        best_scale,
        qmin,
        qmax,
    )

    for bits in range(2, 0x7C00):
        scale = _decode_half(bits)
        codes, losses, objective = _score(
            frozen_weights,
            frozen_curvature,
            scale,
            qmin,
            qmax,
        )
        if objective < best_objective:
            best_bits = bits
            best_scale = scale
            best_codes = codes
            best_losses = losses
            best_objective = objective

    return OracleResult(
        scale=best_scale,
        scale_bits=best_bits,
        codes=best_codes,
        per_environment_loss=best_losses,
        objective=best_objective,
    )
