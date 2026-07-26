from __future__ import annotations

import pytest

from cliffquant import (
    FP16_MAX_FINITE_BITS,
    FP16_MIN_POSITIVE_BITS,
    FP16_POSITIVE_FINITE_COUNT,
    bits_to_scale,
    positive_finite_fp16_bits,
    positive_finite_fp16_scales,
    scale_to_bits,
)


def test_grid_is_complete_strictly_increasing_and_finite() -> None:
    bits = positive_finite_fp16_bits()
    scales = positive_finite_fp16_scales()

    assert len(bits) == len(scales) == FP16_POSITIVE_FINITE_COUNT == 31_743
    assert int(bits[0]) == FP16_MIN_POSITIVE_BITS == 0x0001
    assert int(bits[-1]) == FP16_MAX_FINITE_BITS == 0x7BFF
    assert scales[0] == 2.0**-24
    assert scales[-1] == 65_504.0
    assert (scales[1:] > scales[:-1]).all()


@pytest.mark.parametrize("bits", [0x0001, 0x0002, 0x03FF, 0x0400, 0x3C00, 0x7BFF])
def test_bits_and_scale_round_trip(bits: int) -> None:
    assert scale_to_bits(bits_to_scale(bits)) == bits


def test_returned_grid_arrays_do_not_mutate_the_internal_grid() -> None:
    bits = positive_finite_fp16_bits()
    scales = positive_finite_fp16_scales()
    bits[0] = 99
    scales[0] = 99.0

    assert int(positive_finite_fp16_bits()[0]) == 1
    assert positive_finite_fp16_scales()[0] == 2.0**-24


@pytest.mark.parametrize("bits", [0, 0x7C00, 0xFFFF, -1])
def test_invalid_bit_patterns_are_rejected(bits: int) -> None:
    with pytest.raises(ValueError):
        bits_to_scale(bits)
