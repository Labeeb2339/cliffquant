from __future__ import annotations

import numpy as np
import pytest

from cliffquant import BreakpointSolveResult, solve_breakpoint_exact
from cliffquant.breakpoint import (
    _first_lower_magnitude_index,
    _partition_grid,
)
from cliffquant.contracts import DEFAULT_SPEC, QuantizerSpec, prepare_weights
from cliffquant.fp16_grid import (
    FP16_POSITIVE_FINITE_COUNT,
    bits_to_scale,
    positive_finite_fp16_scales,
)
from cliffquant.solver import solve_exact


def _assert_same_solution(
    weights: np.ndarray | list[float],
    curvature: np.ndarray | list[list[float]],
    spec: QuantizerSpec = DEFAULT_SPEC,
) -> None:
    expected = solve_exact(
        weights,
        curvature,
        spec,
        chunk_size=FP16_POSITIVE_FINITE_COUNT,
    )
    actual = solve_breakpoint_exact(
        weights,
        curvature,
        spec,
        chunk_size=FP16_POSITIVE_FINITE_COUNT,
    )

    assert actual.scale_bits == expected.scale_bits
    assert actual.scale == expected.scale
    assert actual.codes == expected.codes
    assert actual.per_environment_loss == expected.per_environment_loss
    assert actual.objective == expected.objective
    assert actual.candidates_evaluated == actual.provenance.candidate_scales_evaluated
    assert 1 <= actual.candidates_evaluated <= FP16_POSITIVE_FINITE_COUNT


@pytest.mark.parametrize(
    ("weights", "curvature", "spec"),
    [
        ([0.0, -0.0], [[0.0, 0.0]], QuantizerSpec()),
        ([0.0, -0.0], [[1.0, 2.0], [3.0, 4.0]], QuantizerSpec()),
        ([0.75, -0.75], [[0.0, 0.0], [0.0, 0.0]], QuantizerSpec()),
        ([1.0], [[1.0]], QuantizerSpec(qmin=-1, qmax=1)),
        ([-1.0], [[1.0]], QuantizerSpec(qmin=-128, qmax=1)),
        ([1.0], [[1.0]], QuantizerSpec(qmin=-1, qmax=127)),
        (
            [1.0, -1.0, 0.25, -0.25],
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            QuantizerSpec(qmin=-3, qmax=5),
        ),
        (
            [5e-324, -5e-324, 2.0**-24, -(2.0**-24)],
            [[1.0, 2.0, 3.0, 4.0]],
            QuantizerSpec(),
        ),
        (
            [65504.0 * 7.0, -65504.0 * 8.0, 1e-200, -1e-200],
            [[1e-100, 1e-100, 1e100, 1e100], [2e-100, 0.0, 2e100, 0.0]],
            QuantizerSpec(),
        ),
    ],
)
def test_adversarial_cases_match_exhaustive_reference(
    weights: list[float],
    curvature: list[list[float]],
    spec: QuantizerSpec,
) -> None:
    _assert_same_solution(weights, curvature, spec)


def test_exact_half_ties_of_both_parities_match_exhaustive_reference() -> None:
    tie_scales = [bits_to_scale(bits) for bits in (1, 0x03FF, 0x0400, 0x3555, 0x7BFF)]
    weights: list[float] = []
    for scale in tie_scales:
        for level in range(1, 9):
            tied_weight = scale * (level - 0.5)
            weights.extend((tied_weight, -tied_weight))

    weight_array = np.asarray(weights, dtype=np.float64)
    curvature = np.vstack(
        (
            np.ones_like(weight_array),
            np.linspace(0.0, 2.0, weight_array.size),
            np.linspace(2.0, 0.0, weight_array.size),
        )
    )
    _assert_same_solution(weight_array, curvature)


def test_every_supported_transition_level_places_fp16_half_tie_by_even_parity() -> None:
    scales = positive_finite_fp16_scales()
    for bits in (1, 0x03FF, 0x0400, 0x3555, 0x7BFE):
        scale = bits_to_scale(bits)
        for level in range(1, 128):
            magnitude = scale * (level - 0.5)
            actual = _first_lower_magnitude_index(magnitude, level, scales)
            expected = bits - 1 if level % 2 else bits
            assert actual == expected


def test_partition_starts_are_exactly_the_observed_vector_code_changes() -> None:
    generator = np.random.default_rng(8841)
    weights = generator.normal(0.0, 5.0, size=24)
    spec = QuantizerSpec(qmin=-128, qmax=127)
    scales = positive_finite_fp16_scales()
    partition = _partition_grid(prepare_weights(weights), spec, scales)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        codes = np.clip(
            np.rint(weights[np.newaxis, :] / scales[:, np.newaxis]),
            spec.qmin,
            spec.qmax,
        ).astype(np.int16)
    observed_starts = np.flatnonzero(np.any(codes[1:] != codes[:-1], axis=1)) + 1

    assert partition.starts == (0, *(int(index) for index in observed_starts))


def test_every_supported_positive_bound_matches_exhaustive_reference() -> None:
    weights = np.asarray([1.234375, -0.9873046875], dtype=np.float64)
    curvature = np.asarray([[1.0, 0.5], [0.2, 1.3]], dtype=np.float64)
    for qmax in range(1, 128):
        _assert_same_solution(
            weights,
            curvature,
            QuantizerSpec(qmin=-8, qmax=qmax),
        )


def test_every_supported_negative_bound_matches_exhaustive_reference() -> None:
    weights = np.asarray([1.234375, -0.9873046875], dtype=np.float64)
    curvature = np.asarray([[1.0, 0.5], [0.2, 1.3]], dtype=np.float64)
    for qmin in range(-128, 0):
        _assert_same_solution(
            weights,
            curvature,
            QuantizerSpec(qmin=qmin, qmax=7),
        )


@pytest.mark.parametrize("seed", range(40))
def test_randomized_problems_match_exhaustive_reference(seed: int) -> None:
    generator = np.random.default_rng(seed)
    group_size = int(generator.integers(1, 17))
    environments = int(generator.integers(1, 6))
    weights = generator.normal(0.0, 1.5, size=group_size)
    weights[generator.random(group_size) < 0.15] = 0.0
    curvature = generator.lognormal(0.0, 2.0, size=(environments, group_size))
    curvature[generator.random(curvature.shape) < 0.2] = 0.0
    spec = QuantizerSpec(
        qmin=-int(generator.integers(1, 129)),
        qmax=int(generator.integers(1, 128)),
    )
    _assert_same_solution(weights, curvature, spec)


def test_lower_bit_wins_an_objective_tie() -> None:
    result = solve_breakpoint_exact([4.0, -7.0], [[0.0, 0.0]])
    assert result.scale_bits == 1
    assert result.objective == 0.0


@pytest.mark.parametrize(
    ("weight", "curvature"),
    [
        (1.0, 5e-324),
        (10.0, 1e-322),
        (0.1, 1e-320),
    ],
)
def test_underflow_plateau_falls_back_to_exhaustive_lower_bit(
    weight: float,
    curvature: float,
) -> None:
    expected = solve_exact([weight], [[curvature]])
    actual = solve_breakpoint_exact([weight], [[curvature]])

    assert actual.scale_bits == expected.scale_bits
    assert actual.objective == expected.objective == 0.0
    assert actual.candidates_evaluated == FP16_POSITIVE_FINITE_COUNT
    assert actual.provenance.exhaustive_fallback is True
    assert actual.provenance.fallback_reason == "zero-or-subnormal-objective"


def test_large_constant_rounding_plateau_falls_back_to_exhaustive_lower_bit() -> None:
    weights = [1e-8, 1.0]
    curvature = [[1e300, 1e288]]
    expected = solve_exact(weights, curvature)
    actual = solve_breakpoint_exact(weights, curvature)

    assert actual.scale_bits == expected.scale_bits
    assert actual.objective == expected.objective
    assert actual.candidates_evaluated == FP16_POSITIVE_FINITE_COUNT
    assert actual.provenance.exhaustive_fallback is True
    assert actual.provenance.fallback_reason == "direct-candidate-objective-tie"


def test_typical_int4_problem_prunes_the_target_grid() -> None:
    generator = np.random.default_rng(20260726)
    weights = generator.normal(0.0, 0.2, size=128)
    curvature = generator.lognormal(0.0, 0.5, size=(4, 128))
    result = solve_breakpoint_exact(weights, curvature)

    assert isinstance(result, BreakpointSolveResult)
    assert result.provenance.constant_code_segments > 1
    assert result.candidates_evaluated < FP16_POSITIVE_FINITE_COUNT
    assert result.provenance.certificate_candidates_added == 0
    assert result.provenance.exhaustive_fallback is False
    assert result.provenance.fallback_reason is None


@pytest.mark.parametrize(
    ("weights", "curvature", "exception"),
    [
        ([], [[]], ValueError),
        ([1.0, np.nan], [[1.0, 1.0]], ValueError),
        ([1.0], [], ValueError),
        ([1.0], [[-1.0]], ValueError),
    ],
)
def test_validation_matches_public_contract(
    weights: list[float],
    curvature: list[list[float]],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        solve_breakpoint_exact(weights, curvature)


@pytest.mark.parametrize("chunk_size", [True, 0, -1, 1.5])
def test_rejects_invalid_chunk_size(chunk_size: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        solve_breakpoint_exact([1.0], [[1.0]], chunk_size=chunk_size)  # type: ignore[arg-type]
