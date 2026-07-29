"""Strict, content-bound artifacts for CliffQuant AutoPolicy planning."""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .autopolicy import (
    DEFAULT_MAX_PARETO_COMPARISONS,
    DEFAULT_MAX_TRANSITIONS,
    AutoPolicyPlan,
    PolicyCandidate,
    QuantizableUnit,
    RobustAutoPolicyPlan,
    RobustPolicyCandidate,
    RobustQuantizableUnit,
    solve_autopolicy,
    solve_robust_autopolicy,
)
from .provenance import canonical_json_bytes, sha256_bytes, sha256_file

CANDIDATE_SCHEMA = "cliffquant.autopolicy-candidates.v1"
PLAN_SCHEMA = "cliffquant.autopolicy-plan.v1"
ROBUST_OBJECTIVE = "robust-minimax"
SCALAR_OBJECTIVE = "sum-groupwise-max"
OBJECTIVES = (ROBUST_OBJECTIVE, SCALAR_OBJECTIVE)
TRANSFORMER_BLOCK_PROFILE = "experiment-001-transformer-blocks"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRANSFORMER_BLOCK = re.compile(r"^(model\.language_model\.layers\.\d+)\.")


def _load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key in {source}: {key}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number in {source}: {token}")
            ),
        )
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {source}")
    return payload


def _without_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "fingerprint_sha256"}


def payload_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash a payload without its self-describing fingerprint field."""

    return sha256_bytes(canonical_json_bytes(_without_fingerprint(payload)))


def bind_fingerprint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy carrying a deterministic top-level fingerprint."""

    result = _without_fingerprint(payload)
    result["fingerprint_sha256"] = payload_fingerprint(result)
    return result


def _atomic_write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_candidate_bundle(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, fingerprint, and atomically create a candidate bundle."""

    bound = bind_fingerprint(payload)
    validate_candidate_bundle(bound)
    _atomic_write_new(Path(path), canonical_json_bytes(bound))
    return bound


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _require_nonempty_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _validate_candidate_entry(
    entry: Mapping[str, Any],
    *,
    environments: tuple[str, ...],
    unit_id: str,
    definition: Mapping[str, Any],
    shape: tuple[int, int],
    group_count: int,
) -> None:
    _require_exact_keys(
        entry,
        {
            "candidate_id",
            "logical_bytes",
            "groupwise_max_distortion",
            "per_environment_distortion",
            "metadata",
        },
        label=f"{unit_id} candidate",
    )
    candidate_id = _require_nonempty_string(
        entry.get("candidate_id"),
        label=f"{unit_id} candidate_id",
    )
    PolicyCandidate(
        candidate_id,
        entry.get("logical_bytes"),  # type: ignore[arg-type]
        entry.get("groupwise_max_distortion"),  # type: ignore[arg-type]
    )
    raw_per_environment = entry.get("per_environment_distortion")
    if not isinstance(raw_per_environment, dict):
        raise TypeError(f"{unit_id}/{candidate_id} per_environment_distortion must be an object")
    if set(raw_per_environment) != set(environments):
        raise ValueError(f"{unit_id}/{candidate_id} environments must exactly match the bundle")
    RobustPolicyCandidate(
        candidate_id,
        entry.get("logical_bytes"),  # type: ignore[arg-type]
        tuple(raw_per_environment[environment] for environment in environments),
    )
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"{unit_id}/{candidate_id} metadata must be an object")
    for key in ("bits", "group_size", "scale_selection"):
        if key in definition and metadata.get(key) != definition[key]:
            raise ValueError(f"{unit_id}/{candidate_id} metadata disagrees on {key}")

    has_code_bytes = "logical_code_bytes" in metadata
    has_scale_bytes = "logical_scale_bytes" in metadata
    if has_code_bytes != has_scale_bytes:
        raise ValueError(f"{unit_id}/{candidate_id} must record both logical byte components")
    if has_code_bytes:
        code_bytes = metadata["logical_code_bytes"]
        scale_bytes = metadata["logical_scale_bytes"]
        if (
            type(code_bytes) is not int
            or code_bytes < 0
            or type(scale_bytes) is not int
            or scale_bytes < 0
        ):
            raise ValueError(f"{unit_id}/{candidate_id} logical byte components are invalid")
        if code_bytes + scale_bytes != entry["logical_bytes"]:
            raise ValueError(f"{unit_id}/{candidate_id} logical byte components do not sum")
        bits = metadata.get("bits")
        if type(bits) is not int or bits not in {4, 8, 16}:
            raise ValueError(f"{unit_id}/{candidate_id} bits metadata is invalid")
        expected_code_bytes = (shape[0] * shape[1] * bits + 7) // 8
        if code_bytes != expected_code_bytes:
            raise ValueError(f"{unit_id}/{candidate_id} logical code bytes mismatch")
        expected_scale_bytes = 0 if bits == 16 else group_count * 2
        if scale_bytes != expected_scale_bytes:
            raise ValueError(f"{unit_id}/{candidate_id} logical scale bytes mismatch")


def validate_candidate_bundle(payload: Mapping[str, Any]) -> None:
    """Fail closed unless a candidate bundle satisfies the v1 contract."""

    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "profile",
            "scope",
            "source",
            "environments",
            "distortion_metric",
            "candidate_definitions",
            "units",
            "totals",
            "builder",
            "fingerprint_sha256",
        },
        label="candidate bundle",
    )
    if payload.get("schema") != CANDIDATE_SCHEMA or payload.get("status") != "complete":
        raise ValueError("unsupported or incomplete AutoPolicy candidate bundle")
    _require_nonempty_string(payload.get("profile"), label="profile")
    supplied_fingerprint = payload.get("fingerprint_sha256")
    if not isinstance(supplied_fingerprint, str) or not _SHA256.fullmatch(supplied_fingerprint):
        raise ValueError("candidate bundle fingerprint is invalid")
    if supplied_fingerprint != payload_fingerprint(payload):
        raise ValueError("candidate bundle fingerprint mismatch")

    raw_environments = payload.get("environments")
    if not isinstance(raw_environments, list) or not raw_environments:
        raise ValueError("environments must be a non-empty list")
    environments = tuple(
        _require_nonempty_string(value, label="environment") for value in raw_environments
    )
    if len(set(environments)) != len(environments):
        raise ValueError("environments must be unique")
    for label in ("scope", "source", "distortion_metric", "candidate_definitions", "builder"):
        if not isinstance(payload.get(label), dict):
            raise TypeError(f"{label} must be an object")
    scope = payload["scope"]
    _require_nonempty_string(scope.get("budget_scope"), label="scope budget_scope")
    if type(scope.get("checkpoint_export")) is not bool:
        raise TypeError("scope checkpoint_export must be a boolean")
    if scope["checkpoint_export"] is not False:
        raise ValueError("v1 candidate bundles cannot claim checkpoint export")
    distortion_metric = payload["distortion_metric"]
    if distortion_metric.get("additive_across_units") is not True:
        raise ValueError("distortion_metric additive_across_units must be true")
    candidate_definitions = payload["candidate_definitions"]
    if not candidate_definitions:
        raise ValueError("candidate_definitions must not be empty")
    expected_candidate_ids = set(candidate_definitions)
    for candidate_id, definition in candidate_definitions.items():
        _require_nonempty_string(candidate_id, label="candidate definition id")
        if not isinstance(definition, dict):
            raise TypeError(f"candidate definition {candidate_id!r} must be an object")

    raw_units = payload.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("units must be a non-empty list")
    unit_ids: list[str] = []
    for raw_unit in raw_units:
        if not isinstance(raw_unit, dict):
            raise TypeError("unit entries must be objects")
        _require_exact_keys(
            raw_unit,
            {
                "unit_id",
                "weight_key",
                "shape",
                "group_count",
                "candidates",
                "evidence",
            },
            label="unit",
        )
        unit_id = _require_nonempty_string(raw_unit.get("unit_id"), label="unit_id")
        unit_ids.append(unit_id)
        _require_nonempty_string(raw_unit.get("weight_key"), label=f"{unit_id} weight_key")
        shape = raw_unit.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise ValueError(f"{unit_id} shape must contain two positive integers")
        if type(raw_unit.get("group_count")) is not int or raw_unit["group_count"] <= 0:
            raise ValueError(f"{unit_id} group_count must be positive")
        candidates = raw_unit.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{unit_id} candidates must be a non-empty list")
        candidate_ids = [
            candidate.get("candidate_id") for candidate in candidates if isinstance(candidate, dict)
        ]
        if (
            len(candidate_ids) != len(candidates)
            or len(candidate_ids) != len(expected_candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or set(candidate_ids) != expected_candidate_ids
        ):
            raise ValueError(f"{unit_id} candidate inventory disagrees with definitions")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise TypeError(f"{unit_id} candidates must be objects")
            _validate_candidate_entry(
                candidate,
                environments=environments,
                unit_id=unit_id,
                definition=candidate_definitions[candidate["candidate_id"]],
                shape=(shape[0], shape[1]),
                group_count=raw_unit["group_count"],
            )
        if not isinstance(raw_unit.get("evidence"), dict):
            raise TypeError(f"{unit_id} evidence must be an object")
    if unit_ids != sorted(unit_ids) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("units must have unique identifiers in sorted order")
    totals = payload.get("totals")
    if not isinstance(totals, dict):
        raise TypeError("totals must be an object")
    if set(totals) != expected_candidate_ids:
        raise ValueError("totals candidate inventory disagrees with definitions")
    for candidate_id in sorted(expected_candidate_ids):
        if type(totals[candidate_id]) is not int or totals[candidate_id] <= 0:
            raise ValueError(f"totals value for {candidate_id} must be a positive integer")
        expected_total = sum(
            next(
                candidate["logical_bytes"]
                for candidate in unit["candidates"]
                if candidate["candidate_id"] == candidate_id
            )
            for unit in raw_units
        )
        if totals[candidate_id] != expected_total:
            raise ValueError(f"totals mismatch for {candidate_id}")


def load_candidate_bundle(path: str | Path) -> dict[str, Any]:
    """Load and validate a canonical candidate bundle."""

    source = Path(path)
    payload = _load_json_object(source)
    validate_candidate_bundle(payload)
    if source.read_bytes() != canonical_json_bytes(payload):
        raise ValueError("candidate bundle is not in canonical JSON form")
    return payload


def aggregate_experiment001_transformer_blocks(
    candidate_path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate measured matrix candidates into transformer-block choices."""

    validate_candidate_bundle(payload)
    if payload["profile"] != "experiment-001":
        raise ValueError("transformer-block aggregation requires the experiment-001 profile")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in payload["units"]:
        match = _TRANSFORMER_BLOCK.match(unit["unit_id"])
        if match is None:
            raise ValueError(
                f"unit is not inside a recognized transformer block: {unit['unit_id']}"
            )
        grouped[match.group(1)].append(unit)
    if not grouped:  # pragma: no cover - guarded by candidate validation
        raise ValueError("candidate bundle contains no transformer blocks")

    environments = tuple(payload["environments"])
    candidate_ids = tuple(sorted(payload["candidate_definitions"]))
    units: list[dict[str, Any]] = []
    for block_id in sorted(grouped):
        members = sorted(grouped[block_id], key=lambda unit: unit["unit_id"])
        candidates = []
        for candidate_id in candidate_ids:
            selected = [
                next(
                    candidate
                    for candidate in member["candidates"]
                    if candidate["candidate_id"] == candidate_id
                )
                for member in members
            ]
            metadata_keys = set(selected[0]["metadata"])
            if any(set(candidate["metadata"]) != metadata_keys for candidate in selected[1:]):
                raise ValueError(f"candidate metadata keys disagree within {block_id}")
            metadata: dict[str, Any] = {}
            for key in sorted(metadata_keys):
                values = [candidate["metadata"][key] for candidate in selected]
                if key in {"logical_code_bytes", "logical_scale_bytes"}:
                    if any(type(value) is not int or value < 0 for value in values):
                        raise ValueError(f"candidate byte metadata is invalid within {block_id}")
                    metadata[key] = sum(values)
                elif all(value == values[0] for value in values[1:]):
                    metadata[key] = deepcopy(values[0])
                else:
                    raise ValueError(
                        f"candidate metadata is inconsistent within {block_id}: "
                        f"{candidate_id}.{key}"
                    )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "logical_bytes": sum(candidate["logical_bytes"] for candidate in selected),
                    "groupwise_max_distortion": math.fsum(
                        candidate["groupwise_max_distortion"] for candidate in selected
                    ),
                    "per_environment_distortion": {
                        environment: math.fsum(
                            candidate["per_environment_distortion"][environment]
                            for candidate in selected
                        )
                        for environment in environments
                    },
                    "metadata": metadata,
                }
            )
        weight_count = sum(member["shape"][0] * member["shape"][1] for member in members)
        units.append(
            {
                "unit_id": block_id,
                "weight_key": f"aggregate:{block_id}",
                "shape": [1, weight_count],
                "group_count": sum(member["group_count"] for member in members),
                "candidates": candidates,
                "evidence": {
                    "aggregation": "transformer-block",
                    "member_unit_ids": [member["unit_id"] for member in members],
                },
            }
        )

    scope = deepcopy(payload["scope"])
    scope["module_count"] = len(payload["units"])
    scope["allocation_unit_count"] = len(units)
    scope["allocation_granularity"] = "transformer-block"
    aggregated = {
        "schema": CANDIDATE_SCHEMA,
        "status": "complete",
        "profile": TRANSFORMER_BLOCK_PROFILE,
        "scope": scope,
        "source": {
            "parent_candidate_bundle": {
                "file_sha256": sha256_file(candidate_path),
                "fingerprint_sha256": payload["fingerprint_sha256"],
            },
            "measurement_source": deepcopy(payload["source"]),
        },
        "environments": list(environments),
        "distortion_metric": deepcopy(payload["distortion_metric"]),
        "candidate_definitions": deepcopy(payload["candidate_definitions"]),
        "units": units,
        "totals": deepcopy(payload["totals"]),
        "builder": {
            "name": "cliffquant-autopolicy-transformer-blocks-v1",
            "parent_profile": payload["profile"],
        },
    }
    bound = bind_fingerprint(aggregated)
    validate_candidate_bundle(bound)
    return bound


def _candidate_lookup(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (unit["unit_id"], candidate["candidate_id"]): candidate
        for unit in payload["units"]
        for candidate in unit["candidates"]
    }


def scalar_units_from_bundle(payload: Mapping[str, Any]) -> tuple[QuantizableUnit, ...]:
    """Convert a validated bundle to scalar additive solver contracts."""

    validate_candidate_bundle(payload)
    return tuple(
        QuantizableUnit(
            unit["unit_id"],
            tuple(
                PolicyCandidate(
                    candidate["candidate_id"],
                    candidate["logical_bytes"],
                    candidate["groupwise_max_distortion"],
                )
                for candidate in unit["candidates"]
            ),
        )
        for unit in payload["units"]
    )


def robust_units_from_bundle(payload: Mapping[str, Any]) -> tuple[RobustQuantizableUnit, ...]:
    """Convert a validated bundle to separately scored environment contracts."""

    validate_candidate_bundle(payload)
    environments = tuple(payload["environments"])
    return tuple(
        RobustQuantizableUnit(
            unit["unit_id"],
            tuple(
                RobustPolicyCandidate(
                    candidate["candidate_id"],
                    candidate["logical_bytes"],
                    tuple(
                        candidate["per_environment_distortion"][environment]
                        for environment in environments
                    ),
                )
                for candidate in unit["candidates"]
            ),
        )
        for unit in payload["units"]
    )


def solve_candidate_bundle(
    payload: Mapping[str, Any],
    *,
    budget_bytes: int,
    objective: str = ROBUST_OBJECTIVE,
    max_frontier_states: int = 250_000,
    max_transitions: int = DEFAULT_MAX_TRANSITIONS,
    max_pareto_comparisons: int = DEFAULT_MAX_PARETO_COMPARISONS,
) -> AutoPolicyPlan | RobustAutoPolicyPlan:
    """Solve a validated bundle with one of the explicit v1 objectives."""

    validate_candidate_bundle(payload)
    if objective == ROBUST_OBJECTIVE:
        return solve_robust_autopolicy(
            robust_units_from_bundle(payload),
            environments=tuple(payload["environments"]),
            budget_bytes=budget_bytes,
            max_frontier_states=max_frontier_states,
            max_transitions=max_transitions,
            max_pareto_comparisons=max_pareto_comparisons,
        )
    if objective == SCALAR_OBJECTIVE:
        return solve_autopolicy(
            scalar_units_from_bundle(payload),
            budget_bytes=budget_bytes,
            max_frontier_states=max_frontier_states,
            max_transitions=max_transitions,
        )
    raise ValueError(f"objective must be one of {OBJECTIVES}")


def build_plan_payload(
    candidate_path: str | Path,
    candidate_bundle: Mapping[str, Any],
    *,
    objective: str,
    optimization: AutoPolicyPlan | RobustAutoPolicyPlan,
) -> dict[str, Any]:
    """Bind a solver result to the exact candidate evidence it used."""

    validate_candidate_bundle(candidate_bundle)
    lookup = _candidate_lookup(candidate_bundle)
    enriched_decisions = []
    for decision in optimization.decisions:
        candidate = lookup[(decision.unit_id, decision.candidate_id)]
        enriched_decisions.append(
            {
                "unit_id": decision.unit_id,
                "candidate_id": decision.candidate_id,
                "logical_bytes": decision.byte_size,
                "metadata": candidate["metadata"],
                "evidence": next(
                    unit["evidence"]
                    for unit in candidate_bundle["units"]
                    if unit["unit_id"] == decision.unit_id
                ),
            }
        )
    payload = {
        "schema": PLAN_SCHEMA,
        "status": "complete",
        "candidate_bundle": {
            "file_sha256": sha256_file(candidate_path),
            "fingerprint_sha256": candidate_bundle["fingerprint_sha256"],
        },
        "profile": candidate_bundle["profile"],
        "budget_scope": candidate_bundle["scope"]["budget_scope"],
        "objective": objective,
        "optimization": optimization.to_dict(),
        "selected_candidates": enriched_decisions,
        "claim_boundary": {
            "exact": ("selection for the recorded candidate distortions and logical byte sizes"),
            "not_claimed": [
                "downstream-accuracy optimum",
                "checkpoint size",
                "runtime memory",
                "latency improvement",
                "checkpoint exportability",
            ],
        },
    }
    return bind_fingerprint(payload)


def write_plan(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and atomically create a plan artifact."""

    validate_plan(payload)
    _atomic_write_new(Path(path), canonical_json_bytes(payload))
    return dict(payload)


def validate_plan(payload: Mapping[str, Any]) -> None:
    """Validate the structural and self-hash portion of a plan."""

    _require_exact_keys(
        payload,
        {
            "schema",
            "status",
            "candidate_bundle",
            "profile",
            "budget_scope",
            "objective",
            "optimization",
            "selected_candidates",
            "claim_boundary",
            "fingerprint_sha256",
        },
        label="AutoPolicy plan",
    )
    if payload.get("schema") != PLAN_SCHEMA or payload.get("status") != "complete":
        raise ValueError("unsupported or incomplete AutoPolicy plan")
    if payload.get("objective") not in OBJECTIVES:
        raise ValueError(f"plan objective must be one of {OBJECTIVES}")
    if payload.get("fingerprint_sha256") != payload_fingerprint(payload):
        raise ValueError("AutoPolicy plan fingerprint mismatch")
    descriptor = payload.get("candidate_bundle")
    if not isinstance(descriptor, dict):
        raise TypeError("candidate_bundle must be an object")
    _require_exact_keys(
        descriptor,
        {"file_sha256", "fingerprint_sha256"},
        label="candidate_bundle descriptor",
    )
    if any(
        not isinstance(value, str) or not _SHA256.fullmatch(value) for value in descriptor.values()
    ):
        raise ValueError("candidate bundle descriptor contains an invalid hash")
    if not isinstance(payload.get("optimization"), dict):
        raise TypeError("optimization must be an object")
    if not isinstance(payload.get("selected_candidates"), list):
        raise TypeError("selected_candidates must be a list")
    if not isinstance(payload.get("claim_boundary"), dict):
        raise TypeError("claim_boundary must be an object")


def load_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate a canonical plan artifact."""

    source = Path(path)
    payload = _load_json_object(source)
    validate_plan(payload)
    if source.read_bytes() != canonical_json_bytes(payload):
        raise ValueError("AutoPolicy plan is not in canonical JSON form")
    return payload


def verify_plan(
    candidate_path: str | Path,
    plan_path: str | Path,
    *,
    max_frontier_states: int = 250_000,
    max_transitions: int = DEFAULT_MAX_TRANSITIONS,
    max_pareto_comparisons: int = DEFAULT_MAX_PARETO_COMPARISONS,
) -> dict[str, Any]:
    """Recompute a plan from its bound candidate bundle and compare exactly."""

    candidate_bundle = load_candidate_bundle(candidate_path)
    plan = load_plan(plan_path)
    descriptor = plan["candidate_bundle"]
    if descriptor["file_sha256"] != sha256_file(candidate_path):
        raise ValueError("plan candidate file hash mismatch")
    if descriptor["fingerprint_sha256"] != candidate_bundle["fingerprint_sha256"]:
        raise ValueError("plan candidate fingerprint mismatch")
    optimization = plan["optimization"]
    budget = optimization.get("budget_bytes")
    recomputed = solve_candidate_bundle(
        candidate_bundle,
        budget_bytes=budget,
        objective=plan["objective"],
        max_frontier_states=max_frontier_states,
        max_transitions=max_transitions,
        max_pareto_comparisons=max_pareto_comparisons,
    )
    expected = build_plan_payload(
        candidate_path,
        candidate_bundle,
        objective=plan["objective"],
        optimization=recomputed,
    )
    if expected != plan:
        raise ValueError("AutoPolicy plan does not match exact recomputation")
    return {
        "status": "pass",
        "schema": "cliffquant.autopolicy-verification.v1",
        "candidate_sha256": sha256_file(candidate_path),
        "plan_sha256": sha256_file(plan_path),
        "objective": plan["objective"],
        "budget_bytes": budget,
    }
