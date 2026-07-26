"""Deterministic exhaustive certification campaign for the breakpoint solver."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .breakpoint import solve_breakpoint_exact
from .contracts import QuantizerSpec
from .fp16_grid import bits_to_scale
from .provenance import runtime_metadata, sha256_file
from .solver import solve_exact

_CERTIFICATE_SCHEMA = "cliffquant-solver-certification-v2"
_PROTOCOL_MINIMUM_CASES = 10_000


@dataclass(frozen=True, slots=True)
class CertificationConfig:
    """Frozen inputs that define a reproducible randomized campaign."""

    cases: int = _PROTOCOL_MINIMUM_CASES
    seed: int = 20_260_726
    workers: int = 1
    batch_size: int = 32

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class _Problem:
    weights: NDArray[np.float64]
    curvature: NDArray[np.float64]
    spec: QuantizerSpec
    family: str


def _problem(case_index: int, seed: int) -> _Problem:
    rng = np.random.default_rng(np.random.SeedSequence([seed, case_index]))
    group_size = 128 if case_index % 50 == 0 else int(rng.integers(1, 17))
    environments = int(rng.integers(1, 6))
    range_cycle = ((-8, 7), (-4, 3), (-2, 1))
    if case_index % 4 == 3:
        qmin = -int(rng.integers(1, 129))
        qmax = int(rng.integers(1, 128))
    else:
        qmin, qmax = range_cycle[case_index % len(range_cycle)]
    spec = QuantizerSpec(qmin=qmin, qmax=qmax)

    family_index = case_index % 5
    if family_index == 0:
        family = "normal"
        weights = rng.normal(0.0, 1.5, size=group_size)
    elif family_index == 1:
        family = "log-uniform"
        magnitudes = np.exp2(rng.uniform(-20.0, 12.0, size=group_size))
        weights = magnitudes * rng.choice(np.asarray([-1.0, 1.0]), size=group_size)
    elif family_index == 2:
        family = "rne-half-ties"
        scale = bits_to_scale(int(rng.integers(1, 0x7C00)))
        positive_bound = max(1, qmax)
        negative_bound = max(1, -qmin)
        signs = rng.choice(np.asarray([-1, 1]), size=group_size)
        levels = np.empty(group_size, dtype=np.int64)
        positive = signs > 0
        levels[positive] = rng.integers(1, positive_bound + 1, size=int(np.sum(positive)))
        levels[~positive] = rng.integers(1, negative_bound + 1, size=int(np.sum(~positive)))
        weights = signs.astype(np.float64) * scale * (levels.astype(np.float64) - 0.5)
    elif family_index == 3:
        family = "outlier-mixture"
        weights = rng.normal(0.0, 0.2, size=group_size)
        outliers = rng.random(group_size) < 0.15
        weights[outliers] = rng.normal(0.0, 25.0, size=int(np.sum(outliers)))
    else:
        family = "sparse"
        weights = rng.normal(0.0, 2.0, size=group_size)
        weights[rng.random(group_size) < 0.45] = 0.0

    curvature = rng.lognormal(0.0, 2.0, size=(environments, group_size))
    curvature[rng.random(curvature.shape) < 0.20] = 0.0
    if case_index % 997 == 0:
        curvature.fill(0.0)
    if case_index % 991 == 0:
        weights.fill(0.0)
    weights = np.asarray(weights, dtype=np.float64)
    curvature = np.asarray(curvature, dtype=np.float64)
    weights.setflags(write=False)
    curvature.setflags(write=False)
    return _Problem(weights=weights, curvature=curvature, spec=spec, family=family)


def _fingerprint(
    case_index: int,
    problem: _Problem,
    expected: Any,
    actual: Any,
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Qii", case_index, problem.spec.qmin, problem.spec.qmax))
    digest.update(struct.pack("<II", problem.weights.size, problem.curvature.shape[0]))
    digest.update(problem.weights.astype("<f8", copy=False).tobytes())
    digest.update(problem.curvature.astype("<f8", copy=False).tobytes())
    for result in (expected, actual):
        digest.update(struct.pack("<Hd", result.scale_bits, result.objective))
        digest.update(np.asarray(result.codes, dtype="<i2").tobytes())
        digest.update(np.asarray(result.per_environment_loss, dtype="<f8").tobytes())
    return digest.hexdigest()


def _certify_case(case_index: int, seed: int) -> dict[str, Any]:
    problem = _problem(case_index, seed)
    expected = solve_exact(problem.weights, problem.curvature, problem.spec)
    actual = solve_breakpoint_exact(problem.weights, problem.curvature, problem.spec)
    fields = {
        "scale_bits": actual.scale_bits == expected.scale_bits,
        "scale": actual.scale == expected.scale,
        "codes": actual.codes == expected.codes,
        "per_environment_loss": actual.per_environment_loss == expected.per_environment_loss,
        "objective": actual.objective == expected.objective,
    }
    mismatch = not all(fields.values())
    result: dict[str, Any] = {
        "case_index": case_index,
        "candidate_scales_evaluated": actual.candidates_evaluated,
        "exhaustive_fallback": actual.provenance.exhaustive_fallback,
        "family": problem.family,
        "fingerprint": _fingerprint(case_index, problem, expected, actual),
        "mismatch": mismatch,
    }
    if mismatch:
        result["comparison"] = fields
        result["problem"] = {
            "curvature": problem.curvature.tolist(),
            "qmax": problem.spec.qmax,
            "qmin": problem.spec.qmin,
            "weights": problem.weights.tolist(),
        }
        result["expected"] = {
            "codes": list(expected.codes),
            "objective": expected.objective,
            "per_environment_loss": list(expected.per_environment_loss),
            "scale": expected.scale,
            "scale_bits": expected.scale_bits,
        }
        result["actual"] = {
            "codes": list(actual.codes),
            "objective": actual.objective,
            "per_environment_loss": list(actual.per_environment_loss),
            "scale": actual.scale,
            "scale_bits": actual.scale_bits,
        }
    return result


def _certify_batch(payload: tuple[tuple[int, ...], int]) -> list[dict[str, Any]]:
    indices, seed = payload
    return [_certify_case(case_index, seed) for case_index in indices]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@lru_cache(maxsize=1)
def _evidence_provenance() -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    source_paths = {
        "breakpoint": package_dir / "breakpoint.py",
        "certification": package_dir / "certification.py",
        "contracts": package_dir / "contracts.py",
        "fp16_grid": package_dir / "fp16_grid.py",
        "objective": package_dir / "objective.py",
        "solver": package_dir / "solver.py",
    }
    repository_protocol = (
        package_dir.parents[1] / "research" / "SOLVER_CERTIFICATION_001_PROTOCOL.md"
    )
    if repository_protocol.is_file():
        source_paths["protocol"] = repository_protocol
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:  # pragma: no cover - package installation would be broken
        raise RuntimeError(f"solver certification source files are missing: {missing}")
    return {
        "runtime": runtime_metadata(device="cpu", package_names=("numpy",)),
        "sources": {
            name: {
                "file": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in sorted(source_paths.items())
        },
    }


def _aggregate(
    config: CertificationConfig,
    results: list[dict[str, Any]],
    *,
    elapsed_seconds: float,
    status: str,
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: int(item["case_index"]))
    aggregate = hashlib.sha256()
    for result in ordered:
        aggregate.update(bytes.fromhex(str(result["fingerprint"])))
    mismatches = [result for result in ordered if bool(result["mismatch"])]
    candidate_counts = [int(result["candidate_scales_evaluated"]) for result in ordered]
    families: dict[str, int] = {}
    for result in ordered:
        family = str(result["family"])
        families[family] = families.get(family, 0) + 1
    complete = len(ordered) == config.cases
    passed = complete and not mismatches
    return {
        "schema": _CERTIFICATE_SCHEMA,
        "status": status,
        "evidence_provenance": _evidence_provenance(),
        "config": asdict(config),
        "completed_cases": len(ordered),
        "mismatch_count": len(mismatches),
        "protocol_gate_passed": passed and config.cases >= _PROTOCOL_MINIMUM_CASES,
        "minimum_protocol_cases": _PROTOCOL_MINIMUM_CASES,
        "aggregate_fingerprint_sha256": aggregate.hexdigest(),
        "families": dict(sorted(families.items())),
        "exhaustive_fallback_count": sum(
            1 for result in ordered if bool(result["exhaustive_fallback"])
        ),
        "candidate_scales": {
            "minimum": min(candidate_counts) if candidate_counts else None,
            "median": float(np.median(candidate_counts)) if candidate_counts else None,
            "maximum": max(candidate_counts) if candidate_counts else None,
        },
        "elapsed_seconds": elapsed_seconds,
        "mismatches": mismatches[:10],
        "case_results": ordered,
    }


def run_certification(
    output_path: Path,
    config: CertificationConfig,
    *,
    progress_every: int = 250,
) -> dict[str, Any]:
    """Run or resume the deterministic campaign and atomically save its certificate."""

    config.validate()
    if isinstance(progress_every, bool) or not isinstance(progress_every, int):
        raise TypeError("progress_every must be an integer")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    existing_results: list[dict[str, Any]] = []
    prior_elapsed = 0.0
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("schema") != _CERTIFICATE_SCHEMA:
            raise ValueError("existing certificate has an incompatible schema")
        if existing.get("config") != asdict(config):
            raise ValueError("existing certificate configuration does not match this run")
        existing_results = list(existing.get("case_results", []))
        indices = [int(item["case_index"]) for item in existing_results]
        if indices != list(range(len(indices))):
            raise ValueError("existing certificate results are not a contiguous prefix")
        prior_elapsed = float(existing.get("elapsed_seconds", 0.0))

    start_index = len(existing_results)
    if start_index >= config.cases:
        return _aggregate(
            config,
            existing_results[: config.cases],
            elapsed_seconds=prior_elapsed,
            status="passed" if not any(item["mismatch"] for item in existing_results) else "failed",
        )

    batches = [
        tuple(range(start, min(start + config.batch_size, config.cases)))
        for start in range(start_index, config.cases, config.batch_size)
    ]
    results = existing_results
    started = time.perf_counter()
    last_reported = start_index

    if config.workers == 1:
        iterator = map(_certify_batch, ((batch, config.seed) for batch in batches))
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=config.workers)
        iterator = executor.map(
            _certify_batch,
            ((batch, config.seed) for batch in batches),
            chunksize=1,
        )

    try:
        for batch_results in iterator:
            results.extend(batch_results)
            elapsed = prior_elapsed + time.perf_counter() - started
            snapshot = _aggregate(config, results, elapsed_seconds=elapsed, status="in_progress")
            _atomic_json(output_path, snapshot)
            if len(results) - last_reported >= progress_every or len(results) == config.cases:
                print(
                    f"certified {len(results)}/{config.cases} cases; "
                    f"mismatches={snapshot['mismatch_count']}",
                    flush=True,
                )
                last_reported = len(results)
            if snapshot["mismatch_count"]:
                final = _aggregate(config, results, elapsed_seconds=elapsed, status="failed")
                _atomic_json(output_path, final)
                return final
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    elapsed = prior_elapsed + time.perf_counter() - started
    status = "passed" if len(results) == config.cases else "in_progress"
    final = _aggregate(config, results, elapsed_seconds=elapsed, status=status)
    _atomic_json(output_path, final)
    return final


def default_worker_count() -> int:
    """Return a conservative host default for the CPU-only campaign."""

    logical = os.cpu_count() or 1
    return max(1, min(16, logical - 1))


def certificate_summary(certificate: dict[str, Any]) -> str:
    """Return a concise stable human-readable summary."""

    elapsed = float(certificate["elapsed_seconds"])
    return (
        f"status={certificate['status']} "
        f"cases={certificate['completed_cases']} "
        f"mismatches={certificate['mismatch_count']} "
        f"gate={certificate['protocol_gate_passed']} "
        f"elapsed={math.floor(elapsed // 60)}m{elapsed % 60:.1f}s"
    )
