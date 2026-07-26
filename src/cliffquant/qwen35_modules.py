"""Qwen3.5 linear-module discovery matching GPTQModel's module tree."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

QWEN35_GPTQMODEL_TREE_REVISION = "581bfd970b8b67372ed61b0ef449d88f5388d196"
QWEN35_08B_EXPECTED_QUANTIZED_MODULES = 150

_MODULE_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\."
    r"(?P<branch>"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj)"
    r")$"
)

_BRANCH_ORDER = {
    "self_attn.q_proj": 0,
    "self_attn.k_proj": 1,
    "self_attn.v_proj": 2,
    "self_attn.o_proj": 3,
    "linear_attn.in_proj_qkv": 0,
    "linear_attn.in_proj_z": 1,
    "linear_attn.out_proj": 2,
    "mlp.gate_proj": 4,
    "mlp.up_proj": 5,
    "mlp.down_proj": 6,
}


@dataclass(frozen=True, slots=True)
class QuantizedModule:
    """One concrete module selected by the frozen GPTQModel tree."""

    name: str
    layer: int
    branch: str
    in_features: int
    out_features: int
    module: Any


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def _named_modules(model: Any) -> Iterator[tuple[str, Any]]:
    method = getattr(model, "named_modules", None)
    if not callable(method):
        raise TypeError("model must expose named_modules()")
    yield from method()


def discover_qwen35_quantized_modules(model: Any) -> tuple[QuantizedModule, ...]:
    """Discover only the linear projections listed by GPTQModel Qwen3.5."""

    discovered: list[QuantizedModule] = []
    for name, module in _named_modules(model):
        match = _MODULE_PATTERN.fullmatch(name)
        if match is None:
            continue
        weight_shape = _shape(getattr(module, "weight", None))
        if weight_shape is None or len(weight_shape) != 2:
            raise TypeError(f"quantized module {name} must expose a rank-2 weight")
        out_features, in_features = weight_shape
        declared_in = getattr(module, "in_features", in_features)
        declared_out = getattr(module, "out_features", out_features)
        if int(declared_in) != in_features or int(declared_out) != out_features:
            raise ValueError(f"feature metadata disagrees with weight shape for {name}")
        discovered.append(
            QuantizedModule(
                name=name,
                layer=int(match.group("layer")),
                branch=match.group("branch"),
                in_features=in_features,
                out_features=out_features,
                module=module,
            )
        )
    if not discovered:
        raise ValueError(
            "no Qwen3.5 GPTQModel-tree projections found under model.language_model.layers"
        )
    discovered.sort(key=lambda item: (item.layer, _BRANCH_ORDER[item.branch], item.name))
    names = [item.name for item in discovered]
    if len(names) != len(set(names)):
        raise ValueError("model.named_modules() returned duplicate quantized module names")
    return tuple(discovered)
