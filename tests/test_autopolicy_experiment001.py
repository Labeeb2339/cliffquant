from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cliffquant.autopolicy_experiment001 as experiment001
from cliffquant.autopolicy_experiment001 import (
    _evaluate_scale_rows,
    build_experiment001_candidates,
    logical_payload_bytes,
    vectorized_w8_absmax_rows,
    verify_experiment001_evidence,
)
from cliffquant.fp16_grid import positive_finite_fp16_scales
from cliffquant.full_model_artifacts import POLICY_CLIFFQUANT


def _scalar_w8_oracle(
    weights: np.ndarray,
    curvature: np.ndarray,
    *,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = positive_finite_fp16_scales()
    rows = weights.shape[0]
    groups_per_row = weights.shape[1] // group_size
    bits = np.empty((rows, groups_per_row), dtype=np.uint16)
    objectives = np.empty((rows, groups_per_row), dtype=np.float64)
    environment_totals = np.zeros(curvature.shape[0], dtype=np.float64)
    for row, group in itertools.product(range(rows), range(groups_per_row)):
        start = group * group_size
        stop = start + group_size
        values = np.asarray(weights[row, start:stop], dtype=np.float64)
        positive = values[values > 0]
        negative = values[values < 0]
        target = max(
            float(np.max(positive)) / 127.0 if positive.size else 0.0,
            float(-np.min(negative)) / 128.0 if negative.size else 0.0,
        )
        distances = np.abs(grid - target)
        scale_index = int(np.flatnonzero(distances == np.min(distances))[0])
        scale = float(grid[scale_index])
        bits[row, group] = scale_index + 1
        codes = np.clip(np.rint(values / scale), -128, 127)
        squared_error = np.square(values - scale * codes)
        losses = np.asarray(
            [
                np.sum(
                    curvature[environment, start:stop] * squared_error,
                    dtype=np.float64,
                )
                for environment in range(curvature.shape[0])
            ]
        )
        objectives[row, group] = float(np.max(losses))
        environment_totals += losses
    return bits, objectives, environment_totals


def test_logical_payload_byte_formula() -> None:
    assert logical_payload_bytes(out_features=8, in_features=128, bits=4) == {
        "codes": 512,
        "scales": 16,
        "total": 528,
    }
    assert logical_payload_bytes(out_features=8, in_features=128, bits=8) == {
        "codes": 1024,
        "scales": 16,
        "total": 1040,
    }
    assert logical_payload_bytes(out_features=8, in_features=128, bits=16) == {
        "codes": 2048,
        "scales": 0,
        "total": 2048,
    }


@pytest.mark.parametrize("seed", range(12))
def test_vectorized_w8_matches_independent_scalar_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(3, 16)).astype(np.float64)
    curvature = rng.uniform(0.01, 2.0, size=(4, 16)).astype(np.float64)

    expected = _scalar_w8_oracle(weights, curvature, group_size=8)
    actual = vectorized_w8_absmax_rows(weights, curvature, group_size=8)

    assert np.array_equal(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1], rtol=1e-14, atol=1e-18)
    assert np.allclose(actual[2], expected[2], rtol=1e-14, atol=1e-18)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(8),
        np.arange(1.0, 9.0),
        -np.arange(1.0, 9.0),
        np.asarray([-128.0, -1.0, 0.0, 1.0, 127.0, 3.5, -2.5, 0.25]),
    ],
)
def test_w8_edge_sign_patterns_are_finite(values: np.ndarray) -> None:
    bits, objectives, totals = vectorized_w8_absmax_rows(
        values.reshape(1, 8),
        np.ones((2, 8)),
        group_size=8,
    )

    assert bits.shape == (1, 1)
    assert 0 < int(bits[0, 0]) <= 0x7BFF
    assert np.all(np.isfinite(objectives))
    assert np.all(objectives >= 0.0)
    assert np.all(np.isfinite(totals))


def test_chunking_does_not_change_w8_scores() -> None:
    rng = np.random.default_rng(2339)
    weights = rng.normal(size=(7, 16))
    curvature = rng.uniform(0.1, 1.5, size=(3, 16))
    whole = vectorized_w8_absmax_rows(weights, curvature, group_size=8)
    chunks = [
        vectorized_w8_absmax_rows(weights[start:stop], curvature, group_size=8)
        for start, stop in ((0, 2), (2, 5), (5, 7))
    ]

    assert np.array_equal(np.concatenate([item[0] for item in chunks]), whole[0])
    assert np.allclose(np.concatenate([item[1] for item in chunks]), whole[1])
    assert np.allclose(sum((item[2] for item in chunks), np.zeros(3)), whole[2])


def test_direct_scale_evaluation_uses_requested_integer_range() -> None:
    weights = np.asarray([[7.6, -8.6, 0.5, -0.5, 1.5, -1.5, 0.0, 2.0]])
    curvature = np.ones((2, 8))
    one_bits = np.asarray([[0x3C00]], dtype=np.uint16)

    w4, _ = _evaluate_scale_rows(
        weights,
        curvature,
        one_bits,
        qmin=-8,
        qmax=7,
        group_size=8,
    )
    w8, _ = _evaluate_scale_rows(
        weights,
        curvature,
        one_bits,
        qmin=-128,
        qmax=127,
        group_size=8,
    )

    assert float(w8[0, 0]) < float(w4[0, 0])


@pytest.mark.parametrize(
    ("weights", "curvature", "message"),
    [
        (np.ones((1, 7)), np.ones((2, 7)), "divisible"),
        (np.ones((1, 8)), np.ones((2, 7)), "feature width"),
        (np.asarray([[float("nan")] * 8]), np.ones((2, 8)), "weights must be finite"),
        (np.ones((1, 8)), -np.ones((2, 8)), "curvature"),
    ],
)
def test_w8_rejects_invalid_inputs(
    weights: np.ndarray,
    curvature: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        vectorized_w8_absmax_rows(weights, curvature, group_size=8)


def test_candidate_builder_and_evidence_verifier_end_to_end_with_frozen_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2339)
    weights = rng.uniform(-0.75, 0.75, size=(2, 128)).astype(np.float64)
    environments = ("general", "code")
    curvature = rng.uniform(0.1, 1.2, size=(2, 128)).astype(np.float64)
    scale_bits = np.full((2, 1), 0x3800, dtype=np.uint16)
    w4_objectives, _ = _evaluate_scale_rows(
        weights,
        curvature,
        scale_bits,
        qmin=-8,
        qmax=7,
    )
    module = SimpleNamespace(
        name="model.language_model.layers.0.mlp.up_proj",
        weight_key="model.language_model.layers.0.mlp.up_proj.weight",
        out_features=2,
        in_features=128,
        layer=0,
        branch="mlp.up_proj",
    )
    modules = (module,)
    provenance = {"model_id": "frozen-test", "revision": "frozen-revision"}
    calibration = SimpleNamespace(
        environments=environments,
        modules=modules,
        normalized={module.name: curvature},
        source={"model": provenance},
        metadata_sha256="1" * 64,
        archive_sha256="2" * 64,
    )
    scale_run = SimpleNamespace(
        modules=modules,
        policy=POLICY_CLIFFQUANT,
        metadata={
            "base_model": {"source": provenance},
            "corpus_replay": {"status": "pass"},
        },
        artifacts={
            module.name: SimpleNamespace(
                scale_bits=scale_bits,
                objectives=w4_objectives,
                archive_sha256="3" * 64,
                metadata_sha256="4" * 64,
            )
        },
        manifest_sha256="5" * 64,
    )

    class FakeWeightSource:
        def __init__(self, _snapshot: str | Path) -> None:
            self.provenance = provenance
            self.keys = frozenset({module.weight_key})

        def read_rows(self, key: str, start: int, stop: int) -> np.ndarray:
            assert key == module.weight_key
            return weights[start:stop]

    monkeypatch.setattr(experiment001, "load_scale_run", lambda *_args, **_kwargs: scale_run)
    monkeypatch.setattr(
        experiment001,
        "load_calibration_diagonals",
        lambda *_args, **_kwargs: calibration,
    )
    monkeypatch.setattr(experiment001, "NumpySafetensorsWeightSource", FakeWeightSource)

    output = tmp_path / "autopolicy"
    bundle = build_experiment001_candidates(
        base_model_dir=tmp_path / "snapshot",
        calibration_metadata=tmp_path / "calibration.json",
        w4_scale_run=tmp_path / "scale-run",
        output_dir=output,
        row_chunk_size=1,
    )
    verification = verify_experiment001_evidence(output / "candidates.json", bundle)

    assert bundle["scope"]["module_count"] == 1
    assert bundle["scope"]["weight_count"] == 256
    assert bundle["totals"] == {
        "w4_cliffquant": 132,
        "w8_absmax": 260,
        "dense16": 512,
    }
    assert verification["status"] == "pass"
    assert verification["w8_archives_checked"] == 1
    assert (output / "modules" / "module-00-mlp-up_proj.w8.npz").is_file()

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        build_experiment001_candidates(
            base_model_dir=tmp_path / "snapshot",
            calibration_metadata=tmp_path / "calibration.json",
            w4_scale_run=tmp_path / "scale-run",
            output_dir=output,
        )
