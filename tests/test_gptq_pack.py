from __future__ import annotations

import numpy as np
import pytest

from cliffquant.gptq_pack import (
    dequantize_gptq_int4,
    make_g_idx,
    pack_gptq_v1_qzeros,
    pack_int4_qweight,
    unpack_gptq_v1_qzeros,
    unpack_int4_qweight,
)


def test_known_nibble_order_matches_gptq_layout() -> None:
    codes = np.asarray([[-8, -7, -6, -5, -4, -3, -2, -1]], dtype=np.int8)

    packed = pack_int4_qweight(codes)

    assert packed.shape == (1, 1)
    assert packed.view(np.uint32)[0, 0] == np.uint32(0x76543210)
    np.testing.assert_array_equal(unpack_int4_qweight(packed), codes)


def test_random_code_round_trip_preserves_signed_int4() -> None:
    rng = np.random.default_rng(2339)
    codes = rng.integers(-8, 8, size=(32, 128), dtype=np.int8)

    packed = pack_int4_qweight(codes)
    restored = unpack_int4_qweight(packed)

    assert packed.shape == (16, 32)
    assert packed.dtype == np.int32
    np.testing.assert_array_equal(restored, codes)


def test_symmetric_gptq_v1_zero_word_is_seven_nibbles() -> None:
    packed = pack_gptq_v1_qzeros(num_groups=3, out_features=16)

    assert np.all(packed.view(np.uint32) == np.uint32(0x77777777))
    decoded = unpack_gptq_v1_qzeros(packed)
    np.testing.assert_array_equal(decoded, np.full((3, 16), 8, dtype=np.int16))


def test_independent_dequantization_matches_direct_formula() -> None:
    rng = np.random.default_rng(23039)
    codes = rng.integers(-8, 8, size=(16, 256), dtype=np.int8)
    scales = rng.uniform(0.001, 0.2, size=(2, 16)).astype(np.float16)
    g_idx = make_g_idx(256, 128)

    reconstructed = dequantize_gptq_int4(
        pack_int4_qweight(codes),
        pack_gptq_v1_qzeros(num_groups=2, out_features=16),
        scales,
        g_idx,
    )

    expected = codes.astype(np.float32) * scales[g_idx].T.astype(np.float32)
    np.testing.assert_array_equal(reconstructed, expected)


def test_output_block_dequantization_uses_matching_scale_columns() -> None:
    rng = np.random.default_rng(23040)
    codes = rng.integers(-8, 8, size=(16, 256), dtype=np.int8)
    scales = rng.uniform(0.001, 0.2, size=(2, 16)).astype(np.float16)
    g_idx = make_g_idx(256, 128)
    start, stop = 8, 16

    reconstructed = dequantize_gptq_int4(
        pack_int4_qweight(codes)[:, start:stop],
        pack_gptq_v1_qzeros(num_groups=2, out_features=16)[:, start // 8 : stop // 8],
        scales[:, start:stop],
        g_idx,
    )

    expected = codes[start:stop].astype(np.float32) * scales[:, start:stop][g_idx].T.astype(
        np.float32
    )
    np.testing.assert_array_equal(reconstructed, expected)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: pack_int4_qweight(np.zeros((2, 7), dtype=np.int8)), "divisible by 8"),
        (lambda: pack_int4_qweight(np.asarray([[8] * 8])), "signed INT4"),
        (
            lambda: pack_gptq_v1_qzeros(num_groups=1, out_features=7),
            "divisible by 8",
        ),
        (lambda: make_g_idx(129, 128), "divisible by group_size"),
    ],
)
def test_invalid_packing_contracts_are_rejected(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
