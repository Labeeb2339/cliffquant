"""Small, dependency-light reference for standard GPTQ INT4 packing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

INT4_QMIN = -8
INT4_QMAX = 7
INT4_LOGICAL_ZERO = 8
INT4_PACK_FACTOR = 8
GPTQ_V1_STORED_ZERO = INT4_LOGICAL_ZERO - 1


def _integer_array(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    source = np.asarray(values)
    if source.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integers")
    return np.ascontiguousarray(source, dtype=np.int64)


def make_g_idx(in_features: int, group_size: int = 128) -> NDArray[np.int32]:
    """Return the standard non-activation-ordered GPTQ group index."""

    if type(in_features) is not int or type(group_size) is not int:
        raise TypeError("in_features and group_size must be integers")
    if in_features <= 0 or group_size <= 0:
        raise ValueError("in_features and group_size must be positive")
    if in_features % group_size:
        raise ValueError("in_features must be divisible by group_size")
    return (np.arange(in_features, dtype=np.int32) // group_size).astype(np.int32)


def pack_int4_qweight(codes: ArrayLike) -> NDArray[np.int32]:
    """Pack signed INT4 codes from ``[out, in]`` to GPTQ ``[in/8, out]``."""

    signed = _integer_array(codes, name="codes")
    if signed.ndim != 2:
        raise ValueError("codes must have shape [out_features, in_features]")
    out_features, in_features = signed.shape
    if out_features == 0 or in_features == 0:
        raise ValueError("codes must not have an empty dimension")
    if in_features % INT4_PACK_FACTOR:
        raise ValueError("in_features must be divisible by 8")
    if np.any((signed < INT4_QMIN) | (signed > INT4_QMAX)):
        raise ValueError("codes must be in the signed INT4 range [-8, 7]")

    unsigned = (signed.T + INT4_LOGICAL_ZERO).astype(np.uint32, copy=False)
    blocks = unsigned.reshape(in_features // INT4_PACK_FACTOR, INT4_PACK_FACTOR, out_features)
    packed = np.zeros((in_features // INT4_PACK_FACTOR, out_features), dtype=np.uint32)
    for nibble in range(INT4_PACK_FACTOR):
        packed |= blocks[:, nibble, :] << np.uint32(4 * nibble)
    return packed.view(np.int32).copy()


def unpack_int4_qweight(
    qweight: ArrayLike,
    *,
    in_features: int | None = None,
) -> NDArray[np.int8]:
    """Unpack GPTQ qweight to signed codes with shape ``[out, in]``."""

    packed = _integer_array(qweight, name="qweight")
    if packed.ndim != 2:
        raise ValueError("qweight must have shape [in_features/8, out_features]")
    if packed.shape[0] == 0 or packed.shape[1] == 0:
        raise ValueError("qweight must not have an empty dimension")
    inferred_in_features = packed.shape[0] * INT4_PACK_FACTOR
    if in_features is None:
        in_features = inferred_in_features
    if type(in_features) is not int:
        raise TypeError("in_features must be an integer")
    if in_features != inferred_in_features:
        raise ValueError("in_features does not match the packed qweight shape")

    words = packed.astype(np.int32, copy=False).view(np.uint32)
    unpacked = np.empty(
        (packed.shape[0], INT4_PACK_FACTOR, packed.shape[1]),
        dtype=np.uint8,
    )
    for nibble in range(INT4_PACK_FACTOR):
        unpacked[:, nibble, :] = ((words >> np.uint32(4 * nibble)) & np.uint32(0xF)).astype(
            np.uint8
        )
    unsigned = unpacked.reshape(in_features, packed.shape[1])
    return (unsigned.astype(np.int16) - INT4_LOGICAL_ZERO).T.astype(np.int8)


def pack_gptq_v1_qzeros(
    *,
    num_groups: int,
    out_features: int,
    logical_zero: int = INT4_LOGICAL_ZERO,
) -> NDArray[np.int32]:
    """Pack a constant logical zero point in standard GPTQ-v1 layout.

    GPTQ-v1 serializes each zero-point nibble one below its logical value. A
    compatible loader restores the one before dequantization.
    """

    if type(num_groups) is not int or type(out_features) is not int:
        raise TypeError("num_groups and out_features must be integers")
    if num_groups <= 0 or out_features <= 0:
        raise ValueError("num_groups and out_features must be positive")
    if out_features % INT4_PACK_FACTOR:
        raise ValueError("out_features must be divisible by 8")
    if type(logical_zero) is not int:
        raise TypeError("logical_zero must be an integer")
    if not 1 <= logical_zero <= 16:
        raise ValueError("logical_zero must fit GPTQ-v1 after subtracting one")

    stored = np.uint32(logical_zero - 1)
    word = np.uint32(0)
    for nibble in range(INT4_PACK_FACTOR):
        word |= stored << np.uint32(4 * nibble)
    packed = np.full(
        (num_groups, out_features // INT4_PACK_FACTOR),
        word,
        dtype=np.uint32,
    )
    return packed.view(np.int32).copy()


def unpack_gptq_v1_qzeros(
    qzeros: ArrayLike,
    *,
    out_features: int | None = None,
) -> NDArray[np.int16]:
    """Decode serialized GPTQ-v1 zero points to logical ``[groups, out]``."""

    packed = _integer_array(qzeros, name="qzeros")
    if packed.ndim != 2:
        raise ValueError("qzeros must have shape [num_groups, out_features/8]")
    if packed.shape[0] == 0 or packed.shape[1] == 0:
        raise ValueError("qzeros must not have an empty dimension")
    inferred_out_features = packed.shape[1] * INT4_PACK_FACTOR
    if out_features is None:
        out_features = inferred_out_features
    if type(out_features) is not int:
        raise TypeError("out_features must be an integer")
    if out_features != inferred_out_features:
        raise ValueError("out_features does not match the packed qzeros shape")

    words = packed.astype(np.int32, copy=False).view(np.uint32)
    decoded = np.empty(
        (packed.shape[0], packed.shape[1], INT4_PACK_FACTOR),
        dtype=np.int16,
    )
    for nibble in range(INT4_PACK_FACTOR):
        stored = (words >> np.uint32(4 * nibble)) & np.uint32(0xF)
        decoded[:, :, nibble] = stored.astype(np.int16) + 1
    return decoded.reshape(packed.shape[0], out_features)


def dequantize_gptq_int4(
    qweight: ArrayLike,
    qzeros: ArrayLike,
    scales: ArrayLike,
    g_idx: ArrayLike,
) -> NDArray[np.float32]:
    """Independently reconstruct a GPTQ INT4 matrix as ``[out, in]``."""

    codes_with_zero_eight = unpack_int4_qweight(qweight).astype(np.int16) + INT4_LOGICAL_ZERO
    zeros = unpack_gptq_v1_qzeros(qzeros)
    scale_array = np.asarray(scales)
    if scale_array.dtype.kind != "f" or scale_array.ndim != 2:
        raise TypeError("scales must be a floating [groups, out] array")
    if not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0):
        raise ValueError("scales must be positive and finite")
    groups = _integer_array(g_idx, name="g_idx")
    if groups.ndim != 1:
        raise ValueError("g_idx must be one-dimensional")

    out_features, in_features = codes_with_zero_eight.shape
    if groups.size != in_features:
        raise ValueError("g_idx length must match qweight input features")
    if scale_array.shape != zeros.shape:
        raise ValueError("scales and decoded qzeros must have the same shape")
    if scale_array.shape[1] != out_features:
        raise ValueError("scale output dimension must match qweight")
    if np.any(groups < 0) or np.any(groups >= scale_array.shape[0]):
        raise ValueError("g_idx contains an out-of-range group")

    group_indices = groups.astype(np.intp)
    unsigned_by_input = codes_with_zero_eight.T
    centered = unsigned_by_input - zeros[group_indices]
    reconstructed = scale_array[group_indices].astype(np.float32) * centered.astype(np.float32)
    return reconstructed.T
