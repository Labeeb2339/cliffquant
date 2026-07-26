from __future__ import annotations

import pytest

from cliffquant import QuantizerSpec, diagonal_losses, evaluate_scale, quantize_codes


def test_round_to_nearest_even_then_clip_is_explicit() -> None:
    weights = [-9.5, -8.5, -7.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 6.5, 7.5, 8.5]

    assert quantize_codes(weights, 1.0) == (
        -8,
        -8,
        -8,
        -2,
        -2,
        0,
        0,
        2,
        2,
        6,
        7,
        7,
    )


def test_asymmetric_w4_saturation() -> None:
    assert quantize_codes([-1.0e300, 1.0e300], 2.0**-24) == (-8, 7)


def test_other_signed_ranges_are_supported() -> None:
    spec = QuantizerSpec(qmin=-2, qmax=1)
    assert quantize_codes([-9.0, -1.5, -0.5, 0.5, 1.5, 9.0], 1.0, spec) == (
        -2,
        -2,
        0,
        0,
        1,
        1,
    )


def test_direct_multi_environment_loss() -> None:
    weights = [1.0, -0.5]
    curvature = [[1.0, 2.0], [4.0, 0.5]]
    evaluation = evaluate_scale(weights, curvature, 0.25)

    assert evaluation.codes == (4, -2)
    assert evaluation.per_environment_loss == (0.0, 0.0)
    assert evaluation.objective == 0.0
    assert diagonal_losses(weights, curvature, 0.25) == (0.0, 0.0)


def test_objective_is_worst_environment() -> None:
    evaluation = evaluate_scale([0.3], [[1.0], [4.0]], 0.25)
    assert evaluation.codes == (1,)
    assert evaluation.per_environment_loss[1] == pytest.approx(
        4.0 * evaluation.per_environment_loss[0]
    )
    assert evaluation.objective == evaluation.per_environment_loss[1]
