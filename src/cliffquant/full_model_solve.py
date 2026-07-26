"""Resumable exact CliffQuant scale search for the frozen full model."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .breakpoint import solve_breakpoint_exact
from .fp16_grid import positive_finite_fp16_scales
from .full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BITS,
    GROUP_SIZE,
    POLICY_ABSMAX,
    POLICY_CLIFFQUANT,
    QMAX,
    QMIN,
    SCALE_CHUNK_SCHEMA,
    SCALE_MODULE_SCHEMA,
    SCALE_RUN_SCHEMA,
    CalibrationDiagonals,
    FrozenModuleSpec,
    SafetensorsWeightSource,
    WeightSource,
    load_calibration_diagonals,
    validate_corpus_replay_pair,
    verify_frozen_corpus_pair,
)
from .provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    runtime_metadata,
    sha256_bytes,
    sha256_file,
)
from .qwen35_modules import QWEN35_GPTQMODEL_TREE_REVISION
from .solver_certificate import validate_solver_certificate

_FP16_ONE_BITS = 0x3C00
_LOWEST_POSITIVE_FP16_BITS = 0x0001
FULL_MODEL_ENGINE_VERSION = "cliffquant-full-model-scale-engine-v3"


@dataclass(frozen=True, slots=True)
class ChunkTask:
    """Serializable, Windows-spawn-safe input for one deterministic row chunk."""

    module_name: str
    policy: str
    row_start: int
    weights: NDArray[np.float64]
    curvature: NDArray[np.float64]
    solver_chunk_size: int
    input_identity: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """Exact scale results and compact solver provenance for one row chunk."""

    module_name: str
    policy: str
    row_start: int
    row_stop: int
    scale_bits: NDArray[np.uint16]
    export_scale_bits: NDArray[np.uint16]
    objectives: NDArray[np.float64]
    provenance: Mapping[str, Any]
    runtime: Mapping[str, Any]
    input_identity: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _module_slug(name: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return f"{readable}-{sha256_bytes(name.encode())[:12]}"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def build_certified_engine_identity(
    solver_certificate: str | Path,
) -> dict[str, Any]:
    """Bind a run to the passed frozen certificate and certified solver sources."""

    certificate = validate_solver_certificate(solver_certificate)
    package_root = Path(__file__).resolve().parent
    engine_files = (
        "corpus.py",
        "corpus_replay.py",
        "dataset_viewer.py",
        "experiment001_io.py",
        "full_model_artifacts.py",
        "full_model_solve.py",
        "model_snapshot.py",
        "provenance.py",
        "qwen35_modules.py",
        "solver_certificate.py",
    )
    engine_sources = {name: sha256_file(package_root / name) for name in engine_files}
    source_hashes = {
        descriptor["file"]: descriptor["sha256"] for descriptor in certificate["sources"].values()
    }
    for name in ("provenance.py", "solver_certificate.py"):
        source_hashes[name] = sha256_file(package_root / name)
    return {
        "certificate": certificate,
        "engine_source_sha256": sha256_bytes(canonical_json_bytes(engine_sources)),
        "engine_sources": engine_sources,
        "engine_version": FULL_MODEL_ENGINE_VERSION,
        "solver_source_sha256": dict(sorted(source_hashes.items())),
    }


def _bits_to_fp16(bits: NDArray[np.uint16]) -> NDArray[np.float16]:
    canonical = np.ascontiguousarray(bits, dtype="<u2")
    return canonical.view("<f2")


def _nearest_positive_fp16(target: float) -> tuple[int, float]:
    scales = positive_finite_fp16_scales()
    insertion = int(np.searchsorted(scales, target, side="left"))
    if insertion == 0:
        index = 0
    elif insertion == scales.size:
        index = scales.size - 1
    else:
        lower_index = insertion - 1
        upper_index = insertion
        lower_distance = target - float(scales[lower_index])
        upper_distance = float(scales[upper_index]) - target
        index = lower_index if lower_distance <= upper_distance else upper_index
    return index + 1, float(scales[index])


def _absmax_result(
    weights: NDArray[np.float64],
    curvature: NDArray[np.float64],
) -> tuple[int, float]:
    positive = weights[weights > 0.0]
    negative = weights[weights < 0.0]
    positive_limit = float(np.max(positive)) / QMAX if positive.size else 0.0
    negative_limit = float(np.max(np.abs(negative))) / abs(QMIN) if negative.size else 0.0
    bits, scale = _nearest_positive_fp16(max(positive_limit, negative_limit))
    codes = np.clip(np.rint(weights / scale), QMIN, QMAX).astype(np.int16)
    residual = weights - scale * codes.astype(np.float64)
    losses = np.sum(
        curvature * np.square(residual)[np.newaxis, :],
        axis=1,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(losses)):
        raise ArithmeticError("AbsMax proxy evaluation overflowed")
    return bits, float(np.max(losses))


def _solve_chunk_worker(task: ChunkTask) -> ChunkResult:
    started_at = _utc_now()
    started = time.perf_counter()
    weights = np.ascontiguousarray(task.weights, dtype=np.float64)
    curvature = np.ascontiguousarray(task.curvature, dtype=np.float64)
    if weights.ndim != 2 or curvature.ndim != 2:
        raise ValueError("chunk weights and curvature must be rank two")
    if weights.shape[1] != curvature.shape[1]:
        raise ValueError("chunk weights and curvature must share input width")
    if weights.shape[1] % GROUP_SIZE:
        raise ValueError("chunk input width must be divisible by group size")
    groups = weights.shape[1] // GROUP_SIZE
    shape = (weights.shape[0], groups)
    scale_bits = np.empty(shape, dtype=np.uint16)
    export_bits = np.empty(shape, dtype=np.uint16)
    objectives = np.empty(shape, dtype=np.float64)
    fallback_reasons: Counter[str] = Counter()
    candidates_evaluated = 0
    exact_solver_groups = 0
    all_zero_groups = 0
    transition_counts = Counter()

    for local_row, row in enumerate(weights):
        for group_index in range(groups):
            start = group_index * GROUP_SIZE
            stop = start + GROUP_SIZE
            group = row[start:stop]
            group_curvature = curvature[:, start:stop]
            if np.count_nonzero(group) == 0:
                # Every positive finite scale has objective zero.  The research
                # tie-break optimum is bit 1; export uses 1.0 to avoid a useless
                # subnormal runtime scale while preserving identical zero codes.
                scale_bits[local_row, group_index] = _LOWEST_POSITIVE_FP16_BITS
                export_bits[local_row, group_index] = _FP16_ONE_BITS
                objectives[local_row, group_index] = 0.0
                all_zero_groups += 1
                continue

            if task.policy == POLICY_CLIFFQUANT:
                result = solve_breakpoint_exact(
                    group,
                    group_curvature,
                    chunk_size=task.solver_chunk_size,
                )
                exact_solver_groups += 1
                scale_bits[local_row, group_index] = result.scale_bits
                export_bits[local_row, group_index] = result.scale_bits
                objectives[local_row, group_index] = result.objective
                candidates_evaluated += result.candidates_evaluated
                provenance = result.provenance
                if provenance.exhaustive_fallback:
                    fallback_reasons[provenance.fallback_reason or "unspecified"] += 1
                for field in (
                    "scalar_transitions_considered",
                    "in_grid_scalar_transitions",
                    "distinct_transition_indices",
                    "constant_code_segments",
                    "certificate_candidates_added",
                ):
                    transition_counts[field] += int(getattr(provenance, field))
            elif task.policy == POLICY_ABSMAX:
                bits, objective = _absmax_result(group, group_curvature)
                scale_bits[local_row, group_index] = bits
                export_bits[local_row, group_index] = bits
                objectives[local_row, group_index] = objective
                candidates_evaluated += 1
            else:  # pragma: no cover - parent validation rejects this
                raise ValueError(f"unsupported full-model policy: {task.policy}")

    elapsed = time.perf_counter() - started
    provenance_summary = {
        "algorithm": (
            "solve_breakpoint_exact"
            if task.policy == POLICY_CLIFFQUANT
            else "absmax_signed_range_nearest_fp16"
        ),
        "all_zero_exact_tie_shortcuts": all_zero_groups,
        "candidates_evaluated": candidates_evaluated,
        "exact_solver_groups": exact_solver_groups,
        "exhaustive_fallback_count": sum(fallback_reasons.values()),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "groups": int(np.prod(shape)),
        "transition_counts": dict(sorted(transition_counts.items())),
    }
    return ChunkResult(
        module_name=task.module_name,
        policy=task.policy,
        row_start=task.row_start,
        row_stop=task.row_start + weights.shape[0],
        scale_bits=scale_bits,
        export_scale_bits=export_bits,
        objectives=objectives,
        provenance=provenance_summary,
        runtime={
            "elapsed_seconds": elapsed,
            "finished_at": _utc_now(),
            "pid": os.getpid(),
            "started_at": started_at,
        },
        input_identity=dict(task.input_identity),
    )


def _chunk_paths(module_dir: Path, start: int, stop: int) -> tuple[Path, Path]:
    stem = f"rows-{start:06d}-{stop:06d}"
    return module_dir / "chunks" / f"{stem}.npz", module_dir / "chunks" / f"{stem}.json"


def _reject_stale_chunk_inventory(
    module_dir: Path,
    bounds: Sequence[tuple[int, int]],
) -> None:
    chunk_dir = module_dir / "chunks"
    if not chunk_dir.is_dir():
        return
    expected = {
        path.name for start, stop in bounds for path in _chunk_paths(module_dir, start, stop)
    }
    actual = {
        path.name
        for path in chunk_dir.iterdir()
        if path.is_file() and (path.name.startswith("rows-") or not path.name.startswith("."))
    }
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(f"chunk checkpoint inventory drift for {module_dir.name}; extra={extra}")


def _chunk_metadata(result: ChunkResult, archive_path: Path) -> dict[str, Any]:
    arrays = {
        "objectives": result.objectives,
        "export_scale_bits": result.export_scale_bits,
        "scale_bits": result.scale_bits,
    }
    return {
        "archive_file": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "arrays": {
            name: {
                "dtype": np.asarray(value).dtype.str,
                "sha256": canonical_array_sha256(np.asarray(value)),
                "shape": list(np.asarray(value).shape),
            }
            for name, value in sorted(arrays.items())
        },
        "input": dict(result.input_identity),
        "module": result.module_name,
        "policy": result.policy,
        "provenance": dict(result.provenance),
        "row_start": result.row_start,
        "row_stop": result.row_stop,
        "runtime": dict(result.runtime),
        "schema": SCALE_CHUNK_SCHEMA,
    }


def _write_chunk(module_dir: Path, result: ChunkResult) -> dict[str, Any]:
    archive_path, metadata_path = _chunk_paths(
        module_dir,
        result.row_start,
        result.row_stop,
    )
    _atomic_npz(
        archive_path,
        {
            "objectives": np.ascontiguousarray(result.objectives, dtype=np.float64),
            "export_scale_bits": np.ascontiguousarray(
                result.export_scale_bits,
                dtype=np.uint16,
            ),
            "scale_bits": np.ascontiguousarray(result.scale_bits, dtype=np.uint16),
        },
    )
    metadata = _chunk_metadata(result, archive_path)
    _atomic_write(metadata_path, canonical_json_bytes(metadata))
    return metadata


def _load_chunk(
    module_dir: Path,
    *,
    module_name: str,
    row_start: int,
    row_stop: int,
    expected_input: Mapping[str, Any],
) -> ChunkResult | None:
    archive_path, metadata_path = _chunk_paths(module_dir, row_start, row_stop)
    if not archive_path.exists() and not metadata_path.exists():
        return None
    if not archive_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"incomplete chunk checkpoint for {module_name} rows {row_start}:{row_stop}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema") != SCALE_CHUNK_SCHEMA:
        raise ValueError(f"invalid chunk checkpoint schema: {metadata_path}")
    if (
        metadata.get("module") != module_name
        or metadata.get("row_start") != row_start
        or metadata.get("row_stop") != row_stop
    ):
        raise ValueError(f"chunk checkpoint identity mismatch: {metadata_path}")
    if metadata.get("input") != dict(expected_input):
        raise ValueError(
            f"chunk checkpoint input drift for {module_name} rows {row_start}:{row_stop}"
        )
    if sha256_file(archive_path) != metadata.get("archive_sha256"):
        raise ValueError(f"chunk archive hash mismatch: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"objectives", "export_scale_bits", "scale_bits"}:
            raise ValueError(f"unexpected chunk arrays: {archive_path}")
        loaded = {name: np.ascontiguousarray(arrays[name]) for name in arrays.files}
    for name, value in loaded.items():
        descriptor = metadata.get("arrays", {}).get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"missing chunk array descriptor: {metadata_path}")
        if canonical_array_sha256(value) != descriptor.get("sha256"):
            raise ValueError(f"chunk array hash mismatch for {name}: {archive_path}")
        if list(value.shape) != descriptor.get("shape"):
            raise ValueError(f"chunk array shape mismatch for {name}: {archive_path}")
    expected_rows = row_stop - row_start
    if loaded["scale_bits"].shape[0] != expected_rows:
        raise ValueError(f"chunk row count mismatch: {archive_path}")
    return ChunkResult(
        module_name=module_name,
        policy=str(expected_input["policy"]),
        row_start=row_start,
        row_stop=row_stop,
        scale_bits=np.ascontiguousarray(loaded["scale_bits"], dtype=np.uint16),
        export_scale_bits=np.ascontiguousarray(
            loaded["export_scale_bits"],
            dtype=np.uint16,
        ),
        objectives=np.ascontiguousarray(loaded["objectives"], dtype=np.float64),
        provenance=metadata.get("provenance", {}),
        runtime=metadata.get("runtime", {}),
        input_identity=dict(expected_input),
    )


def _task_for(
    source: WeightSource,
    calibration: CalibrationDiagonals,
    module: FrozenModuleSpec,
    start: int,
    stop: int,
    *,
    solver_chunk_size: int,
    base_fingerprint: str,
    policy: str,
    engine_identity: Mapping[str, Any],
) -> ChunkTask:
    weights = np.ascontiguousarray(
        source.read_rows(module.weight_key, start, stop),
        dtype=np.float64,
    )
    if weights.shape != (stop - start, module.in_features):
        raise ValueError(f"base tensor row shape drift for {module.weight_key}")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"base tensor contains non-finite weights: {module.weight_key}")
    curvature = calibration.normalized[module.name]
    input_identity = {
        "base_fingerprint": base_fingerprint,
        "calibration_archive_sha256": calibration.archive_sha256,
        "calibration_metadata_sha256": calibration.metadata_sha256,
        "calibration_source_sha256": sha256_bytes(
            canonical_json_bytes(dict(calibration.source or {}))
        ),
        "corpus_replay_sha256": (
            calibration.corpus_replay["sha256"] if calibration.corpus_replay is not None else None
        ),
        "curvature_sha256": canonical_array_sha256(curvature),
        "engine": dict(engine_identity),
        "policy": policy,
        "quantizer": {
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "qmax": QMAX,
            "qmin": QMIN,
            "rounding": "round-to-nearest ties-to-even then clipping",
            "stored_scale": "positive finite IEEE binary16",
        },
        "solver_chunk_size": solver_chunk_size,
        "weight_rows_sha256": canonical_array_sha256(weights),
    }
    return ChunkTask(
        module_name=module.name,
        policy=policy,
        row_start=start,
        weights=weights,
        curvature=curvature,
        solver_chunk_size=solver_chunk_size,
        input_identity=input_identity,
    )


def _progress_payload(
    *,
    completed: int,
    total: int,
    active_module: str,
    started_at: str,
    policy: str,
) -> dict[str, Any]:
    return {
        "active_module": active_module,
        "completed_chunks": completed,
        "fraction": completed / total if total else 1.0,
        "policy": policy,
        "schema": "cliffquant.scale-progress.v1",
        "started_at": started_at,
        "status": "complete" if completed == total else "running",
        "total_chunks": total,
        "updated_at": _utc_now(),
    }


def _combine_counter(target: Counter[str], value: Mapping[str, Any], field: str) -> None:
    nested = value.get(field, {})
    if isinstance(nested, Mapping):
        for key, count in nested.items():
            target[str(key)] += int(count)


def _finalize_module(
    root: Path,
    module_dir: Path,
    module: FrozenModuleSpec,
    chunks: Sequence[ChunkResult],
    *,
    calibration: CalibrationDiagonals,
    base_fingerprint: str,
    policy: str,
    engine_identity: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(chunks, key=lambda chunk: chunk.row_start)
    expected_start = 0
    for chunk in ordered:
        if chunk.row_start != expected_start:
            raise ValueError(f"missing or overlapping chunks while finalizing {module.name}")
        expected_start = chunk.row_stop
    if expected_start != module.out_features:
        raise ValueError(f"incomplete row coverage while finalizing {module.name}")
    scale_bits = np.ascontiguousarray(
        np.concatenate([chunk.scale_bits for chunk in ordered], axis=0),
        dtype=np.uint16,
    )
    export_bits = np.ascontiguousarray(
        np.concatenate([chunk.export_scale_bits for chunk in ordered], axis=0),
        dtype=np.uint16,
    )
    objectives = np.ascontiguousarray(
        np.concatenate([chunk.objectives for chunk in ordered], axis=0),
        dtype=np.float64,
    )
    expected_shape = (module.out_features, module.groups_per_row)
    if any(value.shape != expected_shape for value in (scale_bits, export_bits, objectives)):
        raise ValueError(f"assembled module shape drift for {module.name}")
    scales = _bits_to_fp16(export_bits)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError(f"assembled export scales are not positive finite FP16 for {module.name}")

    archive_path = module_dir / "module.scales.npz"
    _atomic_npz(
        archive_path,
        {
            "objectives": objectives,
            "export_scale_bits": export_bits,
            "scale_bits": scale_bits,
        },
    )
    fallback_reasons: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    for chunk in ordered:
        _combine_counter(fallback_reasons, chunk.provenance, "fallback_reasons")
        _combine_counter(transition_counts, chunk.provenance, "transition_counts")
    sums = {
        field: sum(int(chunk.provenance.get(field, 0)) for chunk in ordered)
        for field in (
            "all_zero_exact_tie_shortcuts",
            "candidates_evaluated",
            "exact_solver_groups",
            "exhaustive_fallback_count",
            "groups",
        )
    }
    elapsed = sum(float(chunk.runtime.get("elapsed_seconds", 0.0)) for chunk in ordered)
    arrays = {
        "objectives": objectives,
        "export_scale_bits": export_bits,
        "scale_bits": scale_bits,
    }
    metadata = {
        "archive_file": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "arrays": {
            name: {
                "dtype": value.dtype.str,
                "sha256": canonical_array_sha256(value),
                "shape": list(value.shape),
            }
            for name, value in sorted(arrays.items())
        },
        "base_fingerprint": base_fingerprint,
        "calibration": {
            "archive_sha256": calibration.archive_sha256,
            "metadata_sha256": calibration.metadata_sha256,
        },
        "corpus_replay_sha256": (
            calibration.corpus_replay["sha256"] if calibration.corpus_replay is not None else None
        ),
        "chunk_count": len(ordered),
        "module": module.name,
        "policy": policy,
        "provenance": {
            **sums,
            "algorithm": (
                "solve_breakpoint_exact"
                if policy == POLICY_CLIFFQUANT
                else "absmax_signed_range_nearest_fp16"
            ),
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "transition_counts": dict(sorted(transition_counts.items())),
        },
        "engine": dict(engine_identity),
        "runtime": {"worker_elapsed_seconds_sum": elapsed},
        "schema": SCALE_MODULE_SCHEMA,
    }
    metadata_path = module_dir / "module.scales.json"
    _atomic_write(metadata_path, canonical_json_bytes(metadata))
    return {
        "archive_sha256": metadata["archive_sha256"],
        "metadata_file": metadata_path.relative_to(root).as_posix(),
        "metadata_sha256": sha256_file(metadata_path),
        "module": asdict(module),
        "policy": policy,
        "provenance": metadata["provenance"],
    }


def solve_modules_to_artifacts(
    output_dir: str | Path,
    *,
    source: WeightSource,
    calibration: CalibrationDiagonals,
    modules: Sequence[FrozenModuleSpec] | None = None,
    row_chunk_size: int = 8,
    solver_chunk_size: int = 4096,
    workers: int = 1,
    policy: str = POLICY_CLIFFQUANT,
    engine_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve all requested matrices with resumable, hash-checked row chunks."""

    if type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be a positive integer")
    if type(solver_chunk_size) is not int or solver_chunk_size <= 0:
        raise ValueError("solver_chunk_size must be a positive integer")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if policy not in (POLICY_CLIFFQUANT, POLICY_ABSMAX):
        raise ValueError("policy must be cliffquant_minimax or absmax")
    if not isinstance(engine_identity, Mapping):
        raise ValueError("a validated immutable engine_identity is required")
    required_engine_fields = {
        "certificate",
        "engine_source_sha256",
        "engine_sources",
        "engine_version",
        "solver_source_sha256",
    }
    if not required_engine_fields.issubset(engine_identity):
        raise ValueError("engine_identity is missing required source/certificate fields")
    corpus_replay = (
        validate_corpus_replay_pair(calibration.corpus_replay)
        if calibration.corpus_replay is not None
        else None
    )
    selected = tuple(calibration.modules if modules is None else modules)
    if not selected:
        raise ValueError("at least one module is required")
    calibration_model = (
        calibration.source.get("model") if isinstance(calibration.source, Mapping) else None
    )
    if calibration_model is not None and calibration_model != dict(source.provenance):
        raise ValueError("calibration diagonals and dense weights have different model bytes")
    calibration_names = {module.name for module in calibration.modules}
    if any(module.name not in calibration_names for module in selected):
        raise ValueError("requested module is absent from calibration inventory")

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_fingerprint = sha256_bytes(canonical_json_bytes(dict(source.provenance)))
    for module in selected:
        if module.weight_key not in source.keys:
            raise KeyError(f"pinned base checkpoint is missing {module.weight_key}")
        if source.shape(module.weight_key) != (module.out_features, module.in_features):
            raise ValueError(f"base tensor shape drift for {module.weight_key}")

    chunk_bounds = {
        module.name: [
            (start, min(start + row_chunk_size, module.out_features))
            for start in range(0, module.out_features, row_chunk_size)
        ]
        for module in selected
    }
    total_chunks = sum(len(bounds) for bounds in chunk_bounds.values())
    progress_path = root / "progress.json"
    started_at = _utc_now()
    completed_chunks = 0
    module_descriptors: list[dict[str, Any]] = []

    for module in selected:
        module_dir = root / "modules" / _module_slug(module.name)
        _reject_stale_chunk_inventory(module_dir, chunk_bounds[module.name])
        completed: dict[tuple[int, int], ChunkResult] = {}
        pending_bounds: list[tuple[int, int]] = []
        for start, stop in chunk_bounds[module.name]:
            task = _task_for(
                source,
                calibration,
                module,
                start,
                stop,
                solver_chunk_size=solver_chunk_size,
                base_fingerprint=base_fingerprint,
                policy=policy,
                engine_identity=engine_identity,
            )
            existing = _load_chunk(
                module_dir,
                module_name=module.name,
                row_start=start,
                row_stop=stop,
                expected_input=task.input_identity,
            )
            if existing is None:
                pending_bounds.append((start, stop))
            else:
                completed[(start, stop)] = existing
                completed_chunks += 1

        _atomic_write(
            progress_path,
            canonical_json_bytes(
                _progress_payload(
                    completed=completed_chunks,
                    total=total_chunks,
                    active_module=module.name,
                    started_at=started_at,
                    policy=policy,
                )
            ),
        )

        if workers == 1:
            for start, stop in pending_bounds:
                task = _task_for(
                    source,
                    calibration,
                    module,
                    start,
                    stop,
                    solver_chunk_size=solver_chunk_size,
                    base_fingerprint=base_fingerprint,
                    policy=policy,
                    engine_identity=engine_identity,
                )
                result = _solve_chunk_worker(task)
                _write_chunk(module_dir, result)
                completed[(start, stop)] = result
                completed_chunks += 1
                _atomic_write(
                    progress_path,
                    canonical_json_bytes(
                        _progress_payload(
                            completed=completed_chunks,
                            total=total_chunks,
                            active_module=module.name,
                            started_at=started_at,
                            policy=policy,
                        )
                    ),
                )
        elif pending_bounds:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                remaining = iter(pending_bounds)
                futures: dict[Any, tuple[int, int]] = {}

                def submit_one(
                    iterator: Any = remaining,
                    current_module: FrozenModuleSpec = module,
                    pending: dict[Any, tuple[int, int]] = futures,
                ) -> bool:
                    try:
                        start, stop = next(iterator)
                    except StopIteration:
                        return False
                    task = _task_for(
                        source,
                        calibration,
                        current_module,
                        start,
                        stop,
                        solver_chunk_size=solver_chunk_size,
                        base_fingerprint=base_fingerprint,
                        policy=policy,
                        engine_identity=engine_identity,
                    )
                    pending[executor.submit(_solve_chunk_worker, task)] = (start, stop)
                    return True

                for _ in range(workers * 2):
                    if not submit_one():
                        break
                while futures:
                    done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                    for future in done:
                        bounds = futures.pop(future)
                        result = future.result()
                        if (result.row_start, result.row_stop) != bounds:
                            raise RuntimeError("worker returned the wrong deterministic row chunk")
                        _write_chunk(module_dir, result)
                        completed[bounds] = result
                        completed_chunks += 1
                        _atomic_write(
                            progress_path,
                            canonical_json_bytes(
                                _progress_payload(
                                    completed=completed_chunks,
                                    total=total_chunks,
                                    active_module=module.name,
                                    started_at=started_at,
                                    policy=policy,
                                )
                            ),
                        )
                        submit_one()

        module_descriptors.append(
            _finalize_module(
                root,
                module_dir,
                module,
                list(completed.values()),
                calibration=calibration,
                base_fingerprint=base_fingerprint,
                policy=policy,
                engine_identity=engine_identity,
            )
        )

    run_metadata = {
        "base_fingerprint": base_fingerprint,
        "base_model": {
            "id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
            "source": dict(source.provenance),
        },
        "calibration": {
            "archive_file": calibration.archive_path.name,
            "archive_sha256": calibration.archive_sha256,
            "environments": list(calibration.environments),
            "metadata_file": calibration.metadata_path.name,
            "metadata_sha256": calibration.metadata_sha256,
            "source": dict(calibration.source or {}),
        },
        "corpus_replay": corpus_replay,
        "finished_at": _utc_now(),
        "gptqmodel_revision": QWEN35_GPTQMODEL_TREE_REVISION,
        "engine": dict(engine_identity),
        "modules": module_descriptors,
        "policy": policy,
        "protocol": "research/EXPERIMENT_001_PROTOCOL.md",
        "quantizer": {
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "qmax": QMAX,
            "qmin": QMIN,
            "scale_dtype": "IEEE binary16 positive finite",
            "selection": (
                "solve_breakpoint_exact"
                if policy == POLICY_CLIFFQUANT
                else "absmax_signed_range_nearest_fp16"
            ),
        },
        "runtime": runtime_metadata(device="cpu"),
        "schema": SCALE_RUN_SCHEMA,
        "started_at": started_at,
        "status": "complete",
    }
    manifest_path = root / "scale-run.json"
    _atomic_write(manifest_path, canonical_json_bytes(run_metadata))
    _atomic_write(
        progress_path,
        canonical_json_bytes(
            _progress_payload(
                completed=total_chunks,
                total=total_chunks,
                active_module=selected[-1].name,
                started_at=started_at,
                policy=policy,
            )
        ),
    )
    return run_metadata


def run_frozen_full_model_scale_search(
    *,
    base_model_dir: str | Path,
    calibration_metadata: str | Path,
    output_dir: str | Path,
    solver_certificate: str | Path,
    policy: str = POLICY_CLIFFQUANT,
    row_chunk_size: int = 8,
    solver_chunk_size: int = 4096,
    workers: int = 1,
) -> dict[str, Any]:
    """Validate frozen inputs and execute the resumable 150-matrix scale run."""

    calibration_path = Path(calibration_metadata).resolve()
    corpus_replay = verify_frozen_corpus_pair(calibration_path.parent)
    calibration = load_calibration_diagonals(
        calibration_path,
        strict_inventory=True,
        corpus_replay=corpus_replay,
    )
    source = SafetensorsWeightSource(base_model_dir)
    engine_identity = build_certified_engine_identity(solver_certificate)
    return solve_modules_to_artifacts(
        output_dir,
        source=source,
        calibration=calibration,
        row_chunk_size=row_chunk_size,
        solver_chunk_size=solver_chunk_size,
        workers=workers,
        policy=policy,
        engine_identity=engine_identity,
    )
