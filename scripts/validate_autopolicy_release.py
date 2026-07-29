"""Build or validate the portable AutoPolicy 001 evidence manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cliffquant.autopolicy_artifacts import (
    ROBUST_OBJECTIVE,
    aggregate_experiment001_transformer_blocks,
    load_candidate_bundle,
    solve_candidate_bundle,
    verify_plan,
)
from cliffquant.autopolicy_experiment001 import (
    DENSE16_CANDIDATE,
    W4_CANDIDATE,
    W8_CANDIDATE,
    verify_experiment001_evidence,
)
from cliffquant.provenance import canonical_json_bytes, sha256_bytes, sha256_file

SCHEMA = "cliffquant.autopolicy-release.v1"
MANIFEST_NAME = "manifest.json"
MANIFEST_HASH_NAME = "manifest.sha256.json"
DATA_FILES = (
    "candidates.json",
    "candidates-transformer-blocks.json",
    "frontier.json",
    "plan-blocks-7bpp.json",
)
MAX_FRONTIER_STATES = 250_000


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number in {path}: {token}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _checked_file(root: Path, relative: str) -> Path:
    value = Path(relative)
    if not relative or value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError(f"unsafe release path: {relative!r}")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"release member is missing, unsafe, or not regular: {relative}")
    return candidate


def _validate_grouped_candidates(
    matrix_path: Path,
    matrix: Mapping[str, Any],
    grouped: Mapping[str, Any],
) -> None:
    expected = aggregate_experiment001_transformer_blocks(matrix_path, matrix)
    if canonical_json_bytes(expected) != canonical_json_bytes(grouped):
        raise ValueError(
            "transformer-block candidates do not match exact aggregation "
            "of the matrix candidate bundle"
        )


def _frontier_budgets(grouped: Mapping[str, Any]) -> list[int]:
    w4 = int(grouped["totals"][W4_CANDIDATE])
    w8 = int(grouped["totals"][W8_CANDIDATE])
    dense = int(grouped["totals"][DENSE16_CANDIDATE])
    interpolated = {
        round(w4 + fraction * (dense - w4))
        for fraction in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    }
    return sorted({w4, w8, dense, *interpolated})


def _recompute_frontier(grouped_path: Path, grouped: Mapping[str, Any]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for budget in _frontier_budgets(grouped):
        plan = solve_candidate_bundle(
            grouped,
            budget_bytes=budget,
            objective=ROBUST_OBJECTIVE,
            max_frontier_states=MAX_FRONTIER_STATES,
        )
        counts = Counter(decision.candidate_id for decision in plan.decisions)
        points.append(
            {
                "budget_bytes": budget,
                "used_bytes": plan.used_bytes,
                "remaining_bytes": plan.remaining_bytes,
                "logical_bits_per_target_weight": (
                    plan.used_bytes * 8 / grouped["scope"]["weight_count"]
                ),
                "worst_environment_distortion": plan.worst_environment_distortion,
                "per_environment_distortion": {
                    environment: value
                    for environment, value in zip(
                        plan.environments,
                        plan.per_environment_distortion,
                        strict=True,
                    )
                },
                "selection_counts": {
                    candidate_id: counts.get(candidate_id, 0)
                    for candidate_id in (
                        W4_CANDIDATE,
                        W8_CANDIDATE,
                        DENSE16_CANDIDATE,
                    )
                },
                "transitions_evaluated": plan.transitions_evaluated,
                "peak_frontier_states": plan.peak_frontier_states,
            }
        )
    baseline = points[0]["worst_environment_distortion"]
    for point in points:
        point["worst_environment_distortion_reduction_vs_all_w4"] = (
            0.0 if baseline == 0.0 else 1.0 - point["worst_environment_distortion"] / baseline
        )
    return {
        "schema": "cliffquant.autopolicy-frontier.v1",
        "status": "complete",
        "candidate_sha256": sha256_file(grouped_path),
        "candidate_fingerprint_sha256": grouped["fingerprint_sha256"],
        "candidate_profile": grouped["profile"],
        "objective": ROBUST_OBJECTIVE,
        "budget_scope": grouped["scope"]["budget_scope"],
        "allocation_granularity": grouped["scope"].get("allocation_granularity", "matrix"),
        "allocation_unit_count": len(grouped["units"]),
        "checkpoint_export": False,
        "points": points,
    }


def _validate_frontier(
    payload: Mapping[str, Any],
    *,
    grouped_path: Path,
    grouped: Mapping[str, Any],
) -> None:
    if (
        payload.get("schema") != "cliffquant.autopolicy-frontier.v1"
        or payload.get("status") != "complete"
        or payload.get("objective") != ROBUST_OBJECTIVE
        or payload.get("checkpoint_export") is not False
    ):
        raise ValueError("unsupported or incomplete AutoPolicy frontier")
    if payload.get("candidate_sha256") != sha256_file(grouped_path):
        raise ValueError("frontier candidate hash mismatch")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("frontier points are missing")
    previous_budget = -1
    for point in points:
        if not isinstance(point, dict):
            raise TypeError("frontier points must be objects")
        budget = point.get("budget_bytes")
        used = point.get("used_bytes")
        distortion = point.get("worst_environment_distortion")
        if (
            type(budget) is not int
            or type(used) is not int
            or not 0 < used <= budget
            or budget <= previous_budget
        ):
            raise ValueError("frontier byte values are invalid")
        if (
            isinstance(distortion, bool)
            or not isinstance(distortion, (int, float))
            or not math.isfinite(float(distortion))
            or distortion < 0
        ):
            raise ValueError("frontier distortion is invalid")
        previous_budget = budget
    expected = _recompute_frontier(grouped_path, grouped)
    if canonical_json_bytes(expected) != canonical_json_bytes(payload):
        raise ValueError("AutoPolicy frontier does not match exact recomputation")


def _expected_inventory(root: Path, candidate: Mapping[str, Any]) -> tuple[str, ...]:
    archives = tuple(unit["evidence"]["w8_archive"]["file"] for unit in candidate["units"])
    expected = tuple(sorted((*DATA_FILES, *archives)))
    actual = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name not in {MANIFEST_NAME, MANIFEST_HASH_NAME}
        )
    )
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"release inventory mismatch; missing={missing}, extra={extra}")
    return expected


def build_manifest(root: str | Path) -> dict[str, Any]:
    """Validate every portable artifact and return its deterministic manifest."""

    release_root = Path(root).resolve()
    if not release_root.is_dir():
        raise FileNotFoundError(f"release directory does not exist: {release_root}")
    paths = {name: _checked_file(release_root, name) for name in DATA_FILES}
    candidate = load_candidate_bundle(paths["candidates.json"])
    grouped = load_candidate_bundle(paths["candidates-transformer-blocks.json"])
    if candidate["profile"] != "experiment-001":
        raise ValueError("matrix candidate profile is not Experiment 001")
    parent = grouped["source"].get("parent_candidate_bundle", {})
    if parent.get("file_sha256") != sha256_file(paths["candidates.json"]):
        raise ValueError("grouped candidates are not bound to the matrix candidate file")
    if parent.get("fingerprint_sha256") != candidate["fingerprint_sha256"]:
        raise ValueError("grouped candidate parent fingerprint mismatch")
    _validate_grouped_candidates(paths["candidates.json"], candidate, grouped)

    evidence = verify_experiment001_evidence(paths["candidates.json"], candidate)
    frontier = _load_json(paths["frontier.json"])
    _validate_frontier(
        frontier,
        grouped_path=paths["candidates-transformer-blocks.json"],
        grouped=grouped,
    )
    plan = verify_plan(
        paths["candidates-transformer-blocks.json"],
        paths["plan-blocks-7bpp.json"],
    )
    expected = _expected_inventory(release_root, candidate)
    files = [
        {
            "path": relative,
            "size_bytes": _checked_file(release_root, relative).stat().st_size,
            "sha256": sha256_file(_checked_file(release_root, relative)),
        }
        for relative in expected
    ]
    return {
        "schema": SCHEMA,
        "status": "pass",
        "profile": "autopolicy-001",
        "candidate": {
            "file_sha256": sha256_file(paths["candidates.json"]),
            "fingerprint_sha256": candidate["fingerprint_sha256"],
            "module_count": len(candidate["units"]),
            "evidence_verification": evidence,
        },
        "transformer_block_view": {
            "file_sha256": sha256_file(paths["candidates-transformer-blocks.json"]),
            "fingerprint_sha256": grouped["fingerprint_sha256"],
            "allocation_unit_count": len(grouped["units"]),
        },
        "frontier": {
            "file_sha256": sha256_file(paths["frontier.json"]),
            "point_count": len(frontier["points"]),
            "objective": frontier["objective"],
        },
        "representative_plan": plan,
        "claim_boundary": {
            "exact": (
                "recorded additive proxy allocation at the recorded granularity "
                "under logical target-payload bytes"
            ),
            "not_claimed": [
                "mixed-bit checkpoint export",
                "downstream quality",
                "checkpoint size",
                "runtime memory",
                "latency",
                "novelty",
            ],
        },
        "files": files,
    }


def _atomic_create(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {path}")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_manifest(root: str | Path) -> dict[str, Any]:
    release_root = Path(root).resolve()
    manifest = build_manifest(release_root)
    payload = canonical_json_bytes(manifest)
    manifest_path = release_root / MANIFEST_NAME
    _atomic_create(manifest_path, payload)
    sidecar = {
        "schema": "cliffquant.autopolicy-release-manifest-hash.v1",
        "file": MANIFEST_NAME,
        "sha256": sha256_bytes(payload),
    }
    _atomic_create(release_root / MANIFEST_HASH_NAME, canonical_json_bytes(sidecar))
    return manifest


def check_manifest(root: str | Path) -> dict[str, Any]:
    release_root = Path(root).resolve()
    manifest_path = _checked_file(release_root, MANIFEST_NAME)
    sidecar_path = _checked_file(release_root, MANIFEST_HASH_NAME)
    recorded = _load_json(manifest_path)
    expected = build_manifest(release_root)
    if recorded != expected:
        raise ValueError("AutoPolicy release manifest content mismatch")
    sidecar = _load_json(sidecar_path)
    if (
        sidecar.get("schema") != "cliffquant.autopolicy-release-manifest-hash.v1"
        or sidecar.get("file") != MANIFEST_NAME
        or sidecar.get("sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("AutoPolicy release manifest sidecar mismatch")
    return recorded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = write_manifest(args.root) if args.write else check_manifest(args.root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "schema": result["schema"],
                "files": len(result["files"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
