from __future__ import annotations

import random

import pytest

from cliffquant import (
    QuantizerSpec,
    bits_to_scale,
    evaluate_scale,
    solve_approximate,
    solve_exact,
)


def test_known_no_clipping_scale_is_recovered_exactly() -> None:
    weights = [7.0, -8.0, 3.0, -4.0]
    curvature = [[1.0, 1.0, 1.0, 1.0]]

    approximate = solve_approximate(weights, curvature)
    exact = solve_exact(weights, curvature)

    assert approximate.scale == 1.0
    assert approximate.objective == 0.0
    assert approximate.scale_bits == exact.scale_bits


@pytest.mark.parametrize("seed", range(8))
def test_randomized_approximation_is_bounded_by_exact_reference(seed: int) -> None:
    rng = random.Random(seed + 700)
    group_size = rng.randint(3, 9)
    environment_count = rng.randint(1, 4)
    weights = [rng.uniform(-3.0, 3.0) for _ in range(group_size)]
    curvature = [
        [rng.uniform(0.0, 2.0) for _ in range(group_size)] for _ in range(environment_count)
    ]
    qmin, qmax = [(-2, 1), (-4, 3), (-8, 7)][seed % 3]
    spec = QuantizerSpec(qmin=qmin, qmax=qmax)

    approximate = solve_approximate(weights, curvature, spec)
    exact = solve_exact(weights, curvature, spec)

    tolerance = 1e-12 + 1e-10 * max(1.0, exact.objective)
    assert approximate.objective + tolerance >= exact.objective
    relative_gap = (approximate.objective - exact.objective) / max(exact.objective, 1e-15)
    assert relative_gap < 0.05


def test_asymmetric_range_and_rounding_are_directly_rescored() -> None:
    spec = QuantizerSpec(qmin=-2, qmax=1)
    weights = [-2.5, -1.5, -0.5, 0.5, 1.5, 4.0]
    curvature = [[1.0, 0.5, 2.0, 3.0, 0.25, 1.5]]

    result = solve_approximate(weights, curvature, spec)
    direct = evaluate_scale(weights, curvature, result.scale, spec)

    assert result.codes == direct.codes
    assert result.per_environment_loss == direct.per_environment_loss
    assert result.objective == direct.objective


def test_all_zero_weights_are_handled_deterministically() -> None:
    result = solve_approximate([0.0, -0.0], [[1.0, 2.0]])

    assert result.scale_bits == 1
    assert result.objective == 0.0
    assert result.provenance.degenerate_reason == "all-zero-weights"
    assert result.candidates_evaluated == 1


def test_all_zero_curvature_is_handled_deterministically() -> None:
    result = solve_approximate([1.0e300, -1.0e300], [[0.0, 0.0], [0.0, 0.0]])

    assert result.scale_bits == 1
    assert result.objective == 0.0
    assert result.provenance.degenerate_reason == "all-zero-curvature"
    assert result.candidates_evaluated == 1


def test_refinement_never_underperforms_its_own_coarse_candidates() -> None:
    weights = [0.91, -0.37, 1.28, -1.73, 0.14]
    curvature = [[1.0, 0.5, 1.2, 0.7, 1.4], [0.4, 1.8, 0.6, 1.1, 0.9]]
    result = solve_approximate(
        weights,
        curvature,
    )
    direct_coarse_objectives = [
        evaluate_scale(weights, curvature, bits_to_scale(bits)).objective
        for bits in result.provenance.coarse_scale_bits
    ]

    assert result.coarse_best_objective == min(direct_coarse_objectives)
    assert result.objective <= min(direct_coarse_objectives)
    assert result.candidates_evaluated == len(result.provenance.candidate_scale_bits)
    assert result.coarse_candidates_evaluated == len(result.provenance.coarse_scale_bits)
    assert result.refinement_candidates_evaluated >= 0
    assert len(result.provenance.basin_seed_bits) == 4
    assert len(set(result.provenance.candidate_scale_bits)) == result.candidates_evaluated
    assert result.scale_bits in result.provenance.candidate_scale_bits
    assert result.provenance.best_origins


def test_search_is_deterministic() -> None:
    weights = [0.91, -0.37, 1.28, -1.73, 0.14]
    curvature = [[1.0, 0.5, 1.2, 0.7, 1.4], [0.4, 1.8, 0.6, 1.1, 0.9]]

    first = solve_approximate(weights, curvature)
    second = solve_approximate(weights, curvature)

    assert first == second


def test_duplicate_environment_and_common_scaling_are_metamorphic() -> None:
    weights = [0.91, -0.37, 1.28, -1.73, 0.14]
    curvature = [[1.0, 0.5, 1.2, 0.7, 1.4], [0.4, 1.8, 0.6, 1.1, 0.9]]

    original = solve_approximate(weights, curvature)
    duplicated = solve_approximate(weights, [*curvature, curvature[0]])
    scaled = solve_approximate(
        weights,
        [[4.0 * value for value in environment] for environment in curvature],
    )

    assert duplicated.scale_bits == original.scale_bits
    assert duplicated.objective == original.objective
    assert scaled.scale_bits == original.scale_bits
    assert scaled.objective == pytest.approx(4.0 * original.objective, rel=1e-14)
