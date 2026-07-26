from __future__ import annotations

import math

import numpy as np
import pytest

from cliffquant import QuantizerSpec, evaluate_scale, solve_exact


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"qmin": True}, TypeError),
        ({"qmax": 7.0}, TypeError),
        ({"qmin": 0}, ValueError),
        ({"qmax": 0}, ValueError),
        ({"qmin": -129}, ValueError),
        ({"qmax": 128}, ValueError),
        ({"rounding": "away"}, ValueError),
        ({"scale_dtype": "float32"}, ValueError),
    ],
)
def test_quantizer_spec_rejects_unsupported_contracts(kwargs, error) -> None:
    with pytest.raises(error):
        QuantizerSpec(**kwargs)


def test_public_functions_require_a_quantizer_spec() -> None:
    with pytest.raises(TypeError, match="QuantizerSpec"):
        solve_exact([1.0], [[1.0]], spec=None)


@pytest.mark.parametrize(
    ("weights", "curvature", "match"),
    [
        ([], [[1.0]], "must not be empty"),
        ([[1.0]], [[1.0]], "one-dimensional"),
        ([1.0, math.inf], [[1.0, 1.0]], "finite"),
        ([1.0], [], "two-dimensional"),
        ([1.0], [[]], "match"),
        ([1.0], [[1.0], [math.nan]], "finite"),
        ([1.0], [[-1.0]], "non-negative"),
    ],
)
def test_problem_validation_is_strict(weights, curvature, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        solve_exact(weights, curvature)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_size_must_be_positive(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        solve_exact([1.0], [[1.0]], chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [True, 1.5])
def test_chunk_size_must_be_an_integer(chunk_size) -> None:
    with pytest.raises(TypeError, match="integer"):
        solve_exact([1.0], [[1.0]], chunk_size=chunk_size)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan, 0.1])
def test_scale_must_already_be_positive_finite_fp16(scale: float) -> None:
    with pytest.raises(ValueError):
        evaluate_scale([1.0], [[1.0]], scale)


def test_numpy_inputs_are_accepted_without_mutation() -> None:
    weights = np.asarray([0.5, -0.25], dtype=np.float32)
    curvature = np.asarray([[1.0, 2.0]], dtype=np.float32)
    original_weights = weights.copy()
    original_curvature = curvature.copy()

    solve_exact(weights, curvature)

    np.testing.assert_array_equal(weights, original_weights)
    np.testing.assert_array_equal(curvature, original_curvature)


@pytest.mark.parametrize("weights", [[True], ["1.0"], [1.0 + 2.0j]])
def test_non_real_weight_encodings_are_rejected(weights) -> None:
    with pytest.raises(TypeError, match="real numbers"):
        solve_exact(weights, [[1.0]])
