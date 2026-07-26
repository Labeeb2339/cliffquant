from __future__ import annotations

import pytest

from cliffquant import solve_exact

WEIGHTS = [0.91, -0.37, 1.28, -1.73, 0.14]
CURVATURE = [
    [1.0, 0.5, 1.2, 0.7, 1.4],
    [0.4, 1.8, 0.6, 1.1, 0.9],
]


def test_result_is_deterministic_across_chunk_sizes() -> None:
    small_chunks = solve_exact(WEIGHTS, CURVATURE, chunk_size=17)
    large_chunks = solve_exact(WEIGHTS, CURVATURE, chunk_size=4096)

    assert small_chunks == large_chunks.__class__(
        scale=large_chunks.scale,
        scale_bits=large_chunks.scale_bits,
        codes=large_chunks.codes,
        per_environment_loss=large_chunks.per_environment_loss,
        objective=large_chunks.objective,
        scales_evaluated=large_chunks.scales_evaluated,
        chunk_size=small_chunks.chunk_size,
    )


def test_coordinate_permutation_preserves_solution() -> None:
    permutation = [3, 0, 4, 1, 2]
    permuted_weights = [WEIGHTS[index] for index in permutation]
    permuted_curvature = [
        [environment[index] for index in permutation] for environment in CURVATURE
    ]

    original = solve_exact(WEIGHTS, CURVATURE)
    permuted = solve_exact(permuted_weights, permuted_curvature)

    assert permuted.scale_bits == original.scale_bits
    assert permuted.objective == pytest.approx(original.objective, rel=1e-14, abs=1e-15)
    assert permuted.codes == tuple(original.codes[index] for index in permutation)


def test_duplicate_environment_does_not_change_solution() -> None:
    original = solve_exact(WEIGHTS, CURVATURE)
    duplicated = solve_exact(WEIGHTS, [*CURVATURE, CURVATURE[0]])

    assert duplicated.scale_bits == original.scale_bits
    assert duplicated.objective == original.objective


def test_zero_environment_does_not_change_solution() -> None:
    original = solve_exact(WEIGHTS, CURVATURE)
    with_zero = solve_exact(WEIGHTS, [*CURVATURE, [0.0] * len(WEIGHTS)])

    assert with_zero.scale_bits == original.scale_bits
    assert with_zero.objective == original.objective


def test_common_curvature_scaling_preserves_scale_and_scales_objective() -> None:
    original = solve_exact(WEIGHTS, CURVATURE)
    scaled_curvature = [[4.0 * value for value in row] for row in CURVATURE]
    scaled = solve_exact(WEIGHTS, scaled_curvature)

    assert scaled.scale_bits == original.scale_bits
    assert scaled.codes == original.codes
    assert scaled.objective == pytest.approx(4.0 * original.objective, rel=1e-14)
