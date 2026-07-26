from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cliffquant.activation_stats import (
    ActivationDiagonalCollector,
    save_activation_statistics,
)
from cliffquant.provenance import canonical_array_sha256, sha256_file
from cliffquant.qwen35_modules import (
    QWEN35_GPTQMODEL_TREE_REVISION,
    discover_qwen35_quantized_modules,
)


class FakeHandle:
    def __init__(self, module: FakeLinear, hook) -> None:
        self.module = module
        self.hook = hook

    def remove(self) -> None:
        self.module.hooks.remove(self.hook)


class FakeLinear:
    def __init__(self, out_features: int, in_features: int) -> None:
        self.out_features = out_features
        self.in_features = in_features
        self.weight = np.zeros((out_features, in_features), dtype=np.float32)
        self.hooks: list = []

    def register_forward_pre_hook(self, hook) -> FakeHandle:
        self.hooks.append(hook)
        return FakeHandle(self, hook)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        for hook in tuple(self.hooks):
            hook(self, (inputs,))
        return inputs


class FakeModel:
    def __init__(self) -> None:
        self.q = FakeLinear(3, 2)
        self.linear_out = FakeLinear(2, 2)
        self.gate = FakeLinear(4, 2)
        self.excluded = FakeLinear(2, 2)
        self.mtp = FakeLinear(2, 2)

    def named_modules(self):
        return iter(
            (
                ("model.language_model.layers.0.self_attn.q_proj", self.q),
                ("model.language_model.layers.1.linear_attn.out_proj", self.linear_out),
                ("model.language_model.layers.1.linear_attn.in_proj_a", self.excluded),
                ("model.language_model.layers.1.mlp.gate_proj", self.gate),
                ("mtp.layers.0.mlp.gate_proj", self.mtp),
            )
        )


def test_discovery_matches_gptqmodel_tree_and_excludes_in_proj_a() -> None:
    assert QWEN35_GPTQMODEL_TREE_REVISION == "581bfd970b8b67372ed61b0ef449d88f5388d196"
    modules = discover_qwen35_quantized_modules(FakeModel())
    assert [module.name for module in modules] == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.1.linear_attn.out_proj",
        "model.language_model.layers.1.mlp.gate_proj",
    ]
    assert [module.in_features for module in modules] == [2, 2, 2]


def test_hooks_use_only_unmasked_rows_and_normalize_each_environment() -> None:
    model = FakeModel()
    module = discover_qwen35_quantized_modules(model)[0]
    collector = ActivationDiagonalCollector([module])
    inputs = np.asarray([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    with collector.capture("general", np.asarray([[1, 1, 0]])):
        model.q(inputs)
    with collector.capture("code", np.asarray([[1, 1, 0]])):
        model.q(np.asarray([[[2.0, 2.0], [0.0, 0.0], [999.0, 999.0]]]))
    statistics = collector.finalize(["general", "code"])
    collector.close()

    general = statistics.values[("general", module.name)]
    np.testing.assert_allclose(general.mean_input_square, [5.0, 10.0])
    np.testing.assert_allclose(general.normalized, [2.0 / 3.0, 4.0 / 3.0])
    assert general.normalization_constant == 7.5
    assert general.token_rows == 2
    code = statistics.values[("code", module.name)]
    np.testing.assert_allclose(code.normalized, [1.0, 1.0])
    assert np.mean(general.normalized) == pytest.approx(1.0)
    assert np.mean(code.normalized) == pytest.approx(1.0)
    assert model.q.hooks == []


def test_zero_mean_diagonal_is_rejected() -> None:
    model = FakeModel()
    module = discover_qwen35_quantized_modules(model)[0]
    with ActivationDiagonalCollector([module]) as collector:
        with collector.capture("general", np.ones((1, 2), dtype=np.uint8)):
            model.q(np.zeros((1, 2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="zero or invalid"):
            collector.finalize(["general"])


def test_missing_module_environment_pair_is_rejected() -> None:
    model = FakeModel()
    modules = discover_qwen35_quantized_modules(model)[:2]
    with ActivationDiagonalCollector(modules) as collector:
        with collector.capture("general", np.ones((1, 2), dtype=np.uint8)):
            model.q(np.ones((1, 2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="missing activation rows"):
            collector.finalize(["general"])


def test_saved_diagonals_have_canonical_array_and_file_hashes(tmp_path: Path) -> None:
    model = FakeModel()
    module = discover_qwen35_quantized_modules(model)[0]
    with ActivationDiagonalCollector([module]) as collector:
        with collector.capture("general", np.ones((1, 2), dtype=np.uint8)):
            model.q(np.asarray([[[1.0, 2.0], [3.0, 4.0]]]))
        statistics = collector.finalize(["general"])

    hashes = save_activation_statistics(
        tmp_path,
        statistics=statistics,
        phase="calibration",
        provenance={"model_revision": "test"},
    )
    metadata_path = tmp_path / "calibration.diagonals.json"
    archive_path = tmp_path / "calibration.diagonals.npz"
    assert hashes["metadata"]["sha256"] == sha256_file(metadata_path)
    assert hashes["statistics"]["sha256"] == sha256_file(archive_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entry = metadata["arrays"][0]
    with np.load(archive_path) as arrays:
        normalized = arrays[entry["normalized"]["array_key"]]
    assert entry["normalized"]["sha256"] == canonical_array_sha256(normalized)


def test_discovery_rejects_shape_metadata_disagreement() -> None:
    model = FakeModel()
    model.q.in_features = 99
    with pytest.raises(ValueError, match="feature metadata"):
        discover_qwen35_quantized_modules(model)


def test_discovery_requires_model_tree_match() -> None:
    model = SimpleNamespace(named_modules=lambda: iter([("linear", FakeLinear(2, 2))]))
    with pytest.raises(ValueError, match=r"no Qwen3\.5"):
        discover_qwen35_quantized_modules(model)
