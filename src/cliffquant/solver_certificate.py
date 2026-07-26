"""Strict validation for the frozen exact-solver certification evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cliffquant.provenance import runtime_metadata, sha256_file

SOLVER_CERTIFICATE_SCHEMA = "cliffquant-solver-certification-v2"
SOLVER_CERTIFICATE_CASES = 10_000
SOLVER_CERTIFICATE_SEED = 20_260_726
SOLVER_AGGREGATE_FINGERPRINT = "f1ee394153ea3e31add761e6e06793a3591620a22262bd5d21b500e84ef103bd"

_FAMILY_ORDER = (
    "normal",
    "log-uniform",
    "rne-half-ties",
    "outlier-mixture",
    "sparse",
)
_SOURCE_FILENAMES = {
    "breakpoint": "breakpoint.py",
    "certification": "certification.py",
    "contracts": "contracts.py",
    "fp16_grid": "fp16_grid.py",
    "objective": "objective.py",
    "provenance": "provenance.py",
    "protocol": "SOLVER_CERTIFICATION_001_PROTOCOL.md",
    "solver": "solver.py",
}


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate {label} JSON key: {key}")
            value[key] = item
        return value

    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite {label} JSON number: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read strict {label} JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _certificate_source_descriptors() -> dict[str, dict[str, Any]]:
    package_dir = Path(__file__).resolve().parent
    repository_dir = package_dir.parents[1]
    source_paths = {
        name: (
            repository_dir / "research" / filename if name == "protocol" else package_dir / filename
        )
        for name, filename in _SOURCE_FILENAMES.items()
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"solver certification source files are missing: {missing}")
    return {
        name: {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(source_paths.items())
    }


def validate_solver_certificate(solver_certificate: str | Path) -> dict[str, Any]:
    """Validate and summarize the frozen zero-mismatch exact-solver certificate."""

    certificate_path = Path(solver_certificate).resolve()
    certificate = _strict_json_object(
        certificate_path,
        label="solver certificate",
    )
    expected_top_level = {
        "aggregate_fingerprint_sha256",
        "candidate_scales",
        "case_results",
        "completed_cases",
        "config",
        "elapsed_seconds",
        "evidence_provenance",
        "exhaustive_fallback_count",
        "families",
        "minimum_protocol_cases",
        "mismatch_count",
        "mismatches",
        "protocol_gate_passed",
        "schema",
        "status",
    }
    if set(certificate) != expected_top_level:
        raise ValueError("solver certificate top-level fields drift")
    if certificate["schema"] != SOLVER_CERTIFICATE_SCHEMA:
        raise ValueError("unsupported solver certificate schema; v2 is required")
    integer_gate_fields = (
        certificate["mismatch_count"],
        certificate["completed_cases"],
        certificate["minimum_protocol_cases"],
    )
    if any(type(value) is not int for value in integer_gate_fields):
        raise ValueError("solver certificate gate counts must be integers")
    if (
        certificate["status"] != "passed"
        or certificate["protocol_gate_passed"] is not True
        or certificate["mismatch_count"] != 0
        or certificate["mismatches"] != []
        or certificate["completed_cases"] != SOLVER_CERTIFICATE_CASES
        or certificate["minimum_protocol_cases"] != SOLVER_CERTIFICATE_CASES
    ):
        raise ValueError("solver certificate did not pass the frozen 10,000-case gate")
    if certificate["aggregate_fingerprint_sha256"] != SOLVER_AGGREGATE_FINGERPRINT:
        raise ValueError("solver certificate aggregate fingerprint drift")

    config = certificate["config"]
    if not isinstance(config, dict) or set(config) != {
        "batch_size",
        "cases",
        "seed",
        "workers",
    }:
        raise ValueError("solver certificate config fields drift")
    if (
        type(config["cases"]) is not int
        or type(config["seed"]) is not int
        or config["cases"] != SOLVER_CERTIFICATE_CASES
        or config["seed"] != SOLVER_CERTIFICATE_SEED
    ):
        raise ValueError("solver certificate frozen campaign config drift")
    for name in ("batch_size", "workers"):
        value = config[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"solver certificate {name} must be a positive integer")

    elapsed_seconds = certificate["elapsed_seconds"]
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not np.isfinite(elapsed_seconds)
        or elapsed_seconds < 0.0
    ):
        raise ValueError("solver certificate elapsed_seconds is invalid")

    evidence = certificate["evidence_provenance"]
    if not isinstance(evidence, dict) or set(evidence) != {"runtime", "sources"}:
        raise ValueError("solver certificate evidence provenance fields drift")
    expected_runtime = runtime_metadata(device="cpu", package_names=("numpy",))
    if evidence["runtime"] != expected_runtime:
        raise ValueError(
            "solver certificate runtime does not match the current Python/NumPy platform"
        )
    expected_sources = _certificate_source_descriptors()
    if evidence["sources"] != expected_sources:
        raise ValueError(
            "solver certificate source hashes do not match the current semantic files/protocol"
        )

    case_results = certificate["case_results"]
    if not isinstance(case_results, list) or len(case_results) != SOLVER_CERTIFICATE_CASES:
        raise ValueError("solver certificate must contain exactly 10,000 case results")
    aggregate = hashlib.sha256()
    candidate_counts: list[int] = []
    family_counts: dict[str, int] = defaultdict(int)
    fallback_count = 0
    expected_case_fields = {
        "candidate_scales_evaluated",
        "case_index",
        "exhaustive_fallback",
        "family",
        "fingerprint",
        "mismatch",
    }
    for expected_index, result in enumerate(case_results):
        if not isinstance(result, dict) or set(result) != expected_case_fields:
            raise ValueError(f"solver certificate case {expected_index} fields drift")
        if type(result["case_index"]) is not int or result["case_index"] != expected_index:
            raise ValueError("solver certificate case indices are not contiguous and ordered")
        expected_family = _FAMILY_ORDER[expected_index % len(_FAMILY_ORDER)]
        if result["family"] != expected_family:
            raise ValueError(f"solver certificate case {expected_index} family drift")
        if result["mismatch"] is not False:
            raise ValueError(f"solver certificate case {expected_index} is a mismatch")
        candidate_count = result["candidate_scales_evaluated"]
        if type(candidate_count) is not int or candidate_count <= 0:
            raise ValueError(f"solver certificate case {expected_index} candidate count is invalid")
        fallback = result["exhaustive_fallback"]
        if type(fallback) is not bool:
            raise ValueError(f"solver certificate case {expected_index} fallback flag is invalid")
        fingerprint = result["fingerprint"]
        if not _is_lower_sha256(fingerprint):
            raise ValueError(f"solver certificate case {expected_index} fingerprint is invalid")
        aggregate.update(bytes.fromhex(fingerprint))
        candidate_counts.append(candidate_count)
        family_counts[expected_family] += 1
        fallback_count += int(fallback)

    recomputed_aggregate = aggregate.hexdigest()
    if (
        recomputed_aggregate != certificate["aggregate_fingerprint_sha256"]
        or recomputed_aggregate != SOLVER_AGGREGATE_FINGERPRINT
    ):
        raise ValueError("solver certificate case aggregate fingerprint mismatch")
    expected_families = dict(sorted(family_counts.items()))
    if certificate["families"] != expected_families:
        raise ValueError("solver certificate family counts do not match its cases")
    if certificate["exhaustive_fallback_count"] != fallback_count:
        raise ValueError("solver certificate fallback count does not match its cases")
    candidate_scales = certificate["candidate_scales"]
    expected_candidate_scales = {
        "maximum": max(candidate_counts),
        "median": float(np.median(candidate_counts)),
        "minimum": min(candidate_counts),
    }
    if candidate_scales != expected_candidate_scales:
        raise ValueError("solver certificate candidate-scale summary does not match its cases")

    return {
        "aggregate_fingerprint_sha256": recomputed_aggregate,
        "completed_cases": SOLVER_CERTIFICATE_CASES,
        "file": certificate_path.name,
        "protocol_gate_passed": True,
        "schema": SOLVER_CERTIFICATE_SCHEMA,
        "sha256": sha256_file(certificate_path),
        "size_bytes": certificate_path.stat().st_size,
        "sources": expected_sources,
        "runtime": expected_runtime,
    }
