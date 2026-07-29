"""Measured AutoPolicy candidates from CliffQuant's frozen Experiment 001."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .autopolicy_artifacts import CANDIDATE_SCHEMA, write_candidate_bundle
from .fp16_grid import positive_finite_fp16_scales
from .full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    GROUP_SIZE,
    POLICY_CLIFFQUANT,
    CalibrationDiagonals,
    FrozenModuleSpec,
    ScaleRun,
    load_calibration_diagonals,
    load_scale_run,
    resolve_hf_snapshot_file,
)
from .model_snapshot import describe_model_snapshot
from .provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    runtime_metadata,
    sha256_bytes,
    sha256_file,
)

W4_CANDIDATE = "w4_cliffquant"
W8_CANDIDATE = "w8_absmax"
DENSE16_CANDIDATE = "dense16"
BYTE_MODEL = "logical-codes-plus-fp16-scales-v1"
BUILDER_VERSION = "cliffquant-autopolicy-experiment001-v1"


def logical_payload_bytes(
    *,
    out_features: int,
    in_features: int,
    bits: int,
    group_size: int = GROUP_SIZE,
) -> dict[str, int]:
    """Return theoretical code/scale bytes for one symmetric weight matrix."""

    for value, label in (
        (out_features, "out_features"),
        (in_features, "in_features"),
        (group_size, "group_size"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if type(bits) is not int or bits not in {4, 8, 16}:
        raise ValueError("bits must be one of 4, 8, or 16")
    if bits in {4, 8} and in_features % group_size:
        raise ValueError("quantized in_features must be divisible by group_size")

    weight_count = out_features * in_features
    if bits == 16:
        return {
            "codes": weight_count * 2,
            "scales": 0,
            "total": weight_count * 2,
        }
    code_bytes = (weight_count * bits + 7) // 8
    scale_bytes = out_features * (in_features // group_size) * 2
    return {
        "codes": code_bytes,
        "scales": scale_bytes,
        "total": code_bytes + scale_bytes,
    }


def _nearest_positive_fp16_bits(targets: NDArray[np.float64]) -> NDArray[np.uint16]:
    if targets.dtype.kind != "f" or not np.all(np.isfinite(targets)):
        raise ValueError("scale targets must be finite floating point")
    if np.any(targets < 0.0):
        raise ValueError("scale targets must be non-negative")
    grid = positive_finite_fp16_scales()
    insertion = np.searchsorted(grid, targets, side="left")
    upper = np.minimum(insertion, grid.size - 1)
    lower = np.maximum(insertion - 1, 0)
    lower_distance = targets - grid[lower]
    upper_distance = grid[upper] - targets
    choose_lower = (insertion == grid.size) | (lower_distance <= upper_distance)
    indices = np.where((insertion > 0) & choose_lower, lower, upper)
    return np.ascontiguousarray(indices + 1, dtype=np.uint16)


def _validate_rows(
    weights: NDArray[np.generic],
    curvature: NDArray[np.generic],
    *,
    group_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    weight_array = np.ascontiguousarray(np.asarray(weights), dtype=np.float64)
    curvature_array = np.ascontiguousarray(np.asarray(curvature), dtype=np.float64)
    if weight_array.ndim != 2 or weight_array.shape[0] == 0:
        raise ValueError("weights must be a non-empty rank-two array")
    if curvature_array.ndim != 2 or curvature_array.shape[0] == 0:
        raise ValueError("curvature must be a non-empty rank-two array")
    if curvature_array.shape[1] != weight_array.shape[1]:
        raise ValueError("curvature feature width must match weights")
    if weight_array.shape[1] % group_size:
        raise ValueError("weight width must be divisible by group_size")
    if not np.all(np.isfinite(weight_array)):
        raise ValueError("weights must be finite")
    if not np.all(np.isfinite(curvature_array)) or np.any(curvature_array < 0.0):
        raise ValueError("curvature must be finite and non-negative")
    return weight_array, curvature_array


def _evaluate_scale_rows(
    weights: NDArray[np.generic],
    curvature: NDArray[np.generic],
    scale_bits: NDArray[np.generic],
    *,
    qmin: int,
    qmax: int,
    group_size: int = GROUP_SIZE,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    weight_array, curvature_array = _validate_rows(
        weights,
        curvature,
        group_size=group_size,
    )
    rows = weight_array.shape[0]
    groups_per_row = weight_array.shape[1] // group_size
    bits = np.ascontiguousarray(np.asarray(scale_bits), dtype=np.uint16)
    if bits.shape != (rows, groups_per_row):
        raise ValueError("scale_bits shape does not match grouped weights")
    if np.any(bits == 0) or np.any(bits > 0x7BFF):
        raise ValueError("scale_bits must encode positive finite FP16 values")
    scales = bits.astype("<u2", copy=False).view("<f2").astype(np.float64)
    grouped_weights = weight_array.reshape(rows, groups_per_row, group_size)
    codes = np.clip(
        np.rint(grouped_weights / scales[..., None]),
        qmin,
        qmax,
    )
    squared_error = np.square(
        grouped_weights - scales[..., None] * codes,
        dtype=np.float64,
    )
    grouped_curvature = curvature_array.reshape(
        curvature_array.shape[0],
        groups_per_row,
        group_size,
    )
    environment_group_losses = np.einsum(
        "rgi,egi->erg",
        squared_error,
        grouped_curvature,
        optimize=True,
    )
    groupwise_max = np.max(environment_group_losses, axis=0)
    per_environment = np.sum(environment_group_losses, axis=(1, 2), dtype=np.float64)
    return (
        np.ascontiguousarray(groupwise_max, dtype=np.float64),
        np.ascontiguousarray(per_environment, dtype=np.float64),
    )


def vectorized_w8_absmax_rows(
    weights: NDArray[np.generic],
    curvature: NDArray[np.generic],
    *,
    group_size: int = GROUP_SIZE,
) -> tuple[NDArray[np.uint16], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate sign-aware W8 AbsMax using stored FP16 scales."""

    weight_array, curvature_array = _validate_rows(
        weights,
        curvature,
        group_size=group_size,
    )
    rows = weight_array.shape[0]
    groups_per_row = weight_array.shape[1] // group_size
    grouped = weight_array.reshape(rows, groups_per_row, group_size)
    positive_limit = np.max(np.where(grouped > 0.0, grouped, 0.0), axis=-1) / 127.0
    negative_limit = np.max(np.where(grouped < 0.0, -grouped, 0.0), axis=-1) / 128.0
    scale_bits = _nearest_positive_fp16_bits(np.maximum(positive_limit, negative_limit))
    objectives, per_environment = _evaluate_scale_rows(
        weight_array,
        curvature_array,
        scale_bits,
        qmin=-128,
        qmax=127,
        group_size=group_size,
    )
    return scale_bits, objectives, per_environment


class NumpySafetensorsWeightSource:
    """Read BF16/FP16 model rows without importing PyTorch."""

    def __init__(self, snapshot_dir: str | Path) -> None:
        try:
            import ml_dtypes  # noqa: F401
            from safetensors import safe_open  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("install CliffQuant with the 'autopolicy-experiment' extra") from exc

        self.root = Path(snapshot_dir).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"base model snapshot does not exist: {self.root}")
        if self.root.name != BASE_MODEL_REVISION:
            raise ValueError(
                "base model directory must be the frozen Hugging Face snapshot "
                f"{BASE_MODEL_REVISION}"
            )
        self._provenance = describe_model_snapshot(
            self.root,
            model_id=BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
        )
        index_path = resolve_hf_snapshot_file(
            self.root,
            "model.safetensors.index.json",
            label="safetensors index",
        )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("safetensors index has no weight_map")
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_map.items()
        ):
            raise TypeError("safetensors weight_map must map strings to strings")
        self._weight_map = dict(raw_map)
        self._shards = {
            name: resolve_hf_snapshot_file(self.root, name, label="safetensors shard")
            for name in sorted(set(self._weight_map.values()))
        }

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self._weight_map)

    def read_rows(self, key: str, start: int, stop: int) -> NDArray[np.float64]:
        if key not in self._weight_map:
            raise KeyError(f"tensor is absent from pinned base checkpoint: {key}")
        if type(start) is not int or type(stop) is not int or start < 0 or stop <= start:
            raise ValueError("row bounds must be increasing non-negative integers")
        import ml_dtypes  # noqa: F401
        from safetensors import safe_open

        path = self._shards[self._weight_map[key]]
        with safe_open(path, framework="np") as handle:
            tensor_slice = handle.get_slice(key)
            shape = tuple(int(value) for value in tensor_slice.get_shape())
            if len(shape) != 2 or stop > shape[0]:
                raise ValueError(f"invalid row slice {start}:{stop} for {key} with shape {shape}")
            value = tensor_slice[start:stop]
        return np.ascontiguousarray(np.asarray(value), dtype=np.float64)


def _atomic_npz_new(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _module_slug(module: FrozenModuleSpec) -> str:
    return f"module-{module.layer:02d}-{module.branch.replace('.', '-')}"


def _source_identity_matches(
    source: NumpySafetensorsWeightSource,
    calibration: CalibrationDiagonals,
    scale_run: ScaleRun,
) -> None:
    scale_source = scale_run.metadata.get("base_model", {}).get("source")
    if source.provenance != scale_source:
        raise ValueError("base snapshot identity disagrees with the W4 scale run")
    calibration_source = (calibration.source or {}).get("model")
    if source.provenance != calibration_source:
        raise ValueError("base snapshot identity disagrees with calibration diagonals")
    if calibration.modules != scale_run.modules:
        raise ValueError("calibration and W4 scale-run module inventories disagree")
    if scale_run.policy != POLICY_CLIFFQUANT:
        raise ValueError("W4 scale run must use the frozen CliffQuant minimax policy")


def _candidate_entry(
    candidate_id: str,
    payload: Mapping[str, int],
    groupwise_max_distortion: float,
    per_environment: NDArray[np.float64],
    *,
    environments: tuple[str, ...],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "logical_bytes": payload["total"],
        "groupwise_max_distortion": float(groupwise_max_distortion),
        "per_environment_distortion": {
            environment: float(value)
            for environment, value in zip(environments, per_environment, strict=True)
        },
        "metadata": {
            **dict(metadata),
            "logical_code_bytes": payload["codes"],
            "logical_scale_bytes": payload["scales"],
        },
    }


def _score_module(
    output_root: Path,
    *,
    source: NumpySafetensorsWeightSource,
    calibration: CalibrationDiagonals,
    scale_run: ScaleRun,
    module: FrozenModuleSpec,
    row_chunk_size: int,
) -> dict[str, Any]:
    curvature = calibration.normalized[module.name]
    w4_artifact = scale_run.artifacts[module.name]
    groups_per_row = module.in_features // GROUP_SIZE
    w8_scale_bits = np.empty((module.out_features, groups_per_row), dtype=np.uint16)
    w8_objectives = np.empty((module.out_features, groups_per_row), dtype=np.float64)
    w4_per_environment = np.zeros(len(calibration.environments), dtype=np.float64)
    w8_per_environment = np.zeros(len(calibration.environments), dtype=np.float64)

    for start in range(0, module.out_features, row_chunk_size):
        stop = min(start + row_chunk_size, module.out_features)
        weights = source.read_rows(module.weight_key, start, stop)
        w4_objectives, w4_environment = _evaluate_scale_rows(
            weights,
            curvature,
            w4_artifact.scale_bits[start:stop],
            qmin=-8,
            qmax=7,
        )
        expected_w4 = w4_artifact.objectives[start:stop]
        if not np.allclose(w4_objectives, expected_w4, rtol=2e-12, atol=1e-18):
            maximum_delta = float(np.max(np.abs(w4_objectives - expected_w4)))
            raise ValueError(
                f"recomputed W4 objectives disagree for {module.name}; "
                f"max_abs_delta={maximum_delta}"
            )
        bits, w8_group_objectives, w8_environment = vectorized_w8_absmax_rows(
            weights,
            curvature,
        )
        w8_scale_bits[start:stop] = bits
        w8_objectives[start:stop] = w8_group_objectives
        w4_per_environment += w4_environment
        w8_per_environment += w8_environment

    module_path = output_root / "modules" / f"{_module_slug(module)}.w8.npz"
    arrays: dict[str, NDArray[np.generic]] = {
        "groupwise_max_objectives": w8_objectives,
        "scale_bits": w8_scale_bits,
    }
    _atomic_npz_new(module_path, arrays)
    relative_archive = module_path.relative_to(output_root).as_posix()
    w4_bytes = logical_payload_bytes(
        out_features=module.out_features,
        in_features=module.in_features,
        bits=4,
    )
    w8_bytes = logical_payload_bytes(
        out_features=module.out_features,
        in_features=module.in_features,
        bits=8,
    )
    dense_bytes = logical_payload_bytes(
        out_features=module.out_features,
        in_features=module.in_features,
        bits=16,
    )
    zeros = np.zeros(len(calibration.environments), dtype=np.float64)
    return {
        "unit_id": module.name,
        "weight_key": module.weight_key,
        "shape": [module.out_features, module.in_features],
        "group_count": module.out_features * groups_per_row,
        "candidates": [
            _candidate_entry(
                DENSE16_CANDIDATE,
                dense_bytes,
                0.0,
                zeros,
                environments=calibration.environments,
                metadata={
                    "bits": 16,
                    "group_size": None,
                    "scale_selection": None,
                },
            ),
            _candidate_entry(
                W4_CANDIDATE,
                w4_bytes,
                float(np.sum(w4_artifact.objectives, dtype=np.float64)),
                w4_per_environment,
                environments=calibration.environments,
                metadata={
                    "bits": 4,
                    "group_size": GROUP_SIZE,
                    "scale_selection": "cliffquant_exact_minimax_fp16",
                },
            ),
            _candidate_entry(
                W8_CANDIDATE,
                w8_bytes,
                float(np.sum(w8_objectives, dtype=np.float64)),
                w8_per_environment,
                environments=calibration.environments,
                metadata={
                    "bits": 8,
                    "group_size": GROUP_SIZE,
                    "scale_selection": "absmax_signed_range_nearest_fp16",
                },
            ),
        ],
        "evidence": {
            "w4_scale_archive_sha256": w4_artifact.archive_sha256,
            "w4_scale_metadata_sha256": w4_artifact.metadata_sha256,
            "w8_archive": {
                "file": relative_archive,
                "sha256": sha256_file(module_path),
                "arrays": {
                    name: {
                        "dtype": np.ascontiguousarray(value).dtype.str,
                        "shape": list(value.shape),
                        "sha256": canonical_array_sha256(np.ascontiguousarray(value)),
                    }
                    for name, value in arrays.items()
                },
            },
        },
    }


def build_experiment001_candidates(
    *,
    base_model_dir: str | Path,
    calibration_metadata: str | Path,
    w4_scale_run: str | Path,
    output_dir: str | Path,
    row_chunk_size: int = 8,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Build the measured W4/W8/dense candidate table for Experiment 001."""

    if type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be a positive integer")
    output_root = Path(output_dir).resolve()
    candidate_path = output_root / "candidates.json"
    if candidate_path.exists() or (output_root / "modules").exists():
        raise FileExistsError(
            f"refusing to reuse a non-empty AutoPolicy output directory: {output_root}"
        )

    scale_run = load_scale_run(w4_scale_run, strict_inventory=True)
    calibration = load_calibration_diagonals(
        calibration_metadata,
        strict_inventory=True,
        corpus_replay=scale_run.metadata.get("corpus_replay"),
    )
    source = NumpySafetensorsWeightSource(base_model_dir)
    _source_identity_matches(source, calibration, scale_run)
    for module in calibration.modules:
        if module.weight_key not in source.keys:
            raise ValueError(f"base checkpoint is missing {module.weight_key}")

    units: list[dict[str, Any]] = []
    total = len(calibration.modules)
    for index, module in enumerate(calibration.modules, start=1):
        units.append(
            _score_module(
                output_root,
                source=source,
                calibration=calibration,
                scale_run=scale_run,
                module=module,
                row_chunk_size=row_chunk_size,
            )
        )
        if progress is not None:
            progress(index, total, module.name)
    units.sort(key=lambda unit: unit["unit_id"])

    totals = {
        candidate_id: sum(
            next(
                candidate["logical_bytes"]
                for candidate in unit["candidates"]
                if candidate["candidate_id"] == candidate_id
            )
            for unit in units
        )
        for candidate_id in (W4_CANDIDATE, W8_CANDIDATE, DENSE16_CANDIDATE)
    }
    weight_count = sum(module.out_features * module.in_features for module in calibration.modules)
    group_count = sum(unit["group_count"] for unit in units)
    payload = {
        "schema": CANDIDATE_SCHEMA,
        "status": "complete",
        "profile": "experiment-001",
        "scope": {
            "module_count": len(units),
            "weight_count": weight_count,
            "group_count": group_count,
            "group_size": GROUP_SIZE,
            "byte_model": BYTE_MODEL,
            "budget_scope": "logical-target-weight-and-scale-payload",
            "checkpoint_export": False,
        },
        "source": {
            "base_model": dict(source.provenance),
            "calibration": {
                "metadata_sha256": calibration.metadata_sha256,
                "archive_sha256": calibration.archive_sha256,
            },
            "w4_scale_run": {
                "manifest_sha256": scale_run.manifest_sha256,
                "policy": scale_run.policy,
            },
        },
        "environments": list(calibration.environments),
        "distortion_metric": {
            "name": ("normalized_diagonal_squared_error_with_separate_environment_totals"),
            "groupwise_scalar": "sum_groupwise_max_environment_error",
            "robust_policy": "max_environment_sum_module_error",
            "additive_across_units": True,
            "downstream_metric": False,
        },
        "candidate_definitions": {
            W4_CANDIDATE: {
                "bits": 4,
                "group_size": GROUP_SIZE,
                "scale_selection": "cliffquant_exact_minimax_fp16",
            },
            W8_CANDIDATE: {
                "bits": 8,
                "group_size": GROUP_SIZE,
                "scale_selection": "absmax_signed_range_nearest_fp16",
            },
            DENSE16_CANDIDATE: {
                "bits": 16,
                "group_size": None,
                "scale_selection": None,
            },
        },
        "units": units,
        "totals": totals,
        "builder": {
            "name": BUILDER_VERSION,
            "row_chunk_size": row_chunk_size,
            "runtime": runtime_metadata(device="cpu"),
            "source_sha256": sha256_file(__file__),
        },
    }
    return write_candidate_bundle(candidate_path, payload)


def verify_experiment001_evidence(
    candidate_path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify every local W8 archive bound by a candidate bundle."""

    root = Path(candidate_path).resolve().parent
    checked = 0
    for unit in payload["units"]:
        descriptor = unit["evidence"]["w8_archive"]
        relative = descriptor.get("file")
        if not isinstance(relative, str):
            raise TypeError("W8 archive file must be a string")
        archive = (root / Path(relative)).resolve()
        try:
            archive.relative_to(root)
        except ValueError as exc:
            raise ValueError("W8 archive path escapes the candidate bundle") from exc
        if sha256_file(archive) != descriptor.get("sha256"):
            raise ValueError(f"W8 archive hash mismatch for {unit['unit_id']}")
        with np.load(archive, allow_pickle=False) as arrays:
            if set(arrays.files) != {"groupwise_max_objectives", "scale_bits"}:
                raise ValueError(f"unexpected W8 arrays for {unit['unit_id']}")
            for name in arrays.files:
                value = np.ascontiguousarray(arrays[name])
                expected = descriptor["arrays"].get(name)
                if (
                    not isinstance(expected, dict)
                    or expected.get("dtype") != value.dtype.str
                    or expected.get("shape") != list(value.shape)
                    or expected.get("sha256") != canonical_array_sha256(value)
                ):
                    raise ValueError(f"W8 array evidence mismatch for {unit['unit_id']}/{name}")
        checked += 1
    result = {
        "schema": "cliffquant.autopolicy-evidence-verification.v1",
        "status": "pass",
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_fingerprint_sha256": payload["fingerprint_sha256"],
        "w8_archives_checked": checked,
    }
    result["sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result
