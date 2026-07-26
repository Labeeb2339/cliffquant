from __future__ import annotations

import random

import pytest

from cliffquant import QuantizerSpec, solve_exact
from tests._oracle import exhaustive_oracle


@pytest.mark.parametrize(
    ("weights", "curvature", "spec"),
    [
        ([0.5, -0.5, 1.5, -1.5], [[1.0, 1.0, 1.0, 1.0]], QuantizerSpec()),
        (
            [7.5, -7.5, 8.5, -8.5],
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            QuantizerSpec(),
        ),
        (
            [0.03125, -0.0625, 0.0],
            [[0.0, 1.0, 2.0], [3.0, 0.0, 1.0]],
            QuantizerSpec(qmin=-2, qmax=1),
        ),
    ],
)
def test_adversarial_groups_match_independent_oracle(weights, curvature, spec) -> None:
    result = solve_exact(weights, curvature, spec, chunk_size=257)
    oracle = exhaustive_oracle(weights, curvature, spec.qmin, spec.qmax)

    assert result.scale_bits == oracle.scale_bits
    assert result.codes == oracle.codes
    assert result.objective == pytest.approx(oracle.objective, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("seed", range(6))
def test_randomized_groups_match_independent_oracle(seed: int) -> None:
    rng = random.Random(seed)
    group_size = rng.randint(2, 6)
    environment_count = rng.randint(1, 3)
    weights = [rng.uniform(-2.0, 2.0) for _ in range(group_size)]
    curvature = [
        [rng.uniform(0.0, 2.0) for _ in range(group_size)] for _ in range(environment_count)
    ]
    ranges = [(-2, 1), (-4, 3), (-8, 7)]
    qmin, qmax = ranges[seed % len(ranges)]
    spec = QuantizerSpec(qmin=qmin, qmax=qmax)

    result = solve_exact(weights, curvature, spec, chunk_size=509)
    oracle = exhaustive_oracle(weights, curvature, qmin, qmax)

    assert result.scale_bits == oracle.scale_bits
    assert result.codes == oracle.codes
    assert result.objective == pytest.approx(oracle.objective, rel=1e-11, abs=1e-13)
    assert result.scales_evaluated == 31_743


def test_zero_curvature_uses_deterministic_lowest_scale_tie_break() -> None:
    result = solve_exact([1.0, -2.0], [[0.0, 0.0], [0.0, 0.0]], chunk_size=37)

    assert result.objective == 0.0
    assert result.scale_bits == 1
    assert result.scale == 2.0**-24


def test_zero_weights_use_deterministic_lowest_scale_tie_break() -> None:
    result = solve_exact([0.0, -0.0], [[1.0, 2.0]])

    assert result.objective == 0.0
    assert result.codes == (0, 0)
    assert result.scale_bits == 1
