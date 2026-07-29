"""Command-line interface for measured CliffQuant AutoPolicy planning."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .autopolicy import DEFAULT_MAX_PARETO_COMPARISONS, DEFAULT_MAX_TRANSITIONS
from .autopolicy_artifacts import (
    OBJECTIVES,
    ROBUST_OBJECTIVE,
    aggregate_experiment001_transformer_blocks,
    build_plan_payload,
    load_candidate_bundle,
    solve_candidate_bundle,
    verify_plan,
    write_candidate_bundle,
    write_plan,
)
from .autopolicy_experiment001 import (
    build_experiment001_candidates,
    verify_experiment001_evidence,
)
from .provenance import sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cliffquant autopolicy",
        description=(
            "Build, solve, and verify measured mixed-precision policy tables. "
            "This planner does not export a runnable mixed-bit checkpoint."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-experiment-001",
        help="measure W4/W8/dense candidates from the frozen Experiment 001 artifacts",
    )
    build.add_argument("--base-model", type=Path, required=True)
    build.add_argument("--calibration-metadata", type=Path, required=True)
    build.add_argument("--w4-scale-run", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--row-chunk-size", type=int, default=8)

    aggregate = subparsers.add_parser(
        "aggregate-blocks",
        help="derive content-bound transformer-block candidates from Experiment 001",
    )
    aggregate.add_argument("--candidates", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)

    solve = subparsers.add_parser(
        "solve",
        help="select one recorded candidate per unit under an exact logical-byte budget",
    )
    solve.add_argument("--candidates", type=Path, required=True)
    solve.add_argument("--budget-bytes", type=int, required=True)
    solve.add_argument("--objective", choices=OBJECTIVES, default=ROBUST_OBJECTIVE)
    solve.add_argument("--max-frontier-states", type=int, default=250_000)
    solve.add_argument("--max-transitions", type=int, default=DEFAULT_MAX_TRANSITIONS)
    solve.add_argument(
        "--max-pareto-comparisons",
        type=int,
        default=DEFAULT_MAX_PARETO_COMPARISONS,
    )
    solve.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify",
        help="verify candidate evidence and optionally recompute a bound plan",
    )
    verify.add_argument("--candidates", type=Path, required=True)
    verify.add_argument("--plan", type=Path)
    verify.add_argument("--max-frontier-states", type=int, default=250_000)
    verify.add_argument("--max-transitions", type=int, default=DEFAULT_MAX_TRANSITIONS)
    verify.add_argument(
        "--max-pareto-comparisons",
        type=int,
        default=DEFAULT_MAX_PARETO_COMPARISONS,
    )
    return parser


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _build(args: argparse.Namespace) -> dict[str, Any]:
    def progress(completed: int, total: int, module_name: str) -> None:
        print(f"[{completed:03d}/{total:03d}] {module_name}", file=sys.stderr, flush=True)

    bundle = build_experiment001_candidates(
        base_model_dir=args.base_model,
        calibration_metadata=args.calibration_metadata,
        w4_scale_run=args.w4_scale_run,
        output_dir=args.output,
        row_chunk_size=args.row_chunk_size,
        progress=progress,
    )
    candidate_path = args.output.resolve() / "candidates.json"
    return {
        "status": "complete",
        "schema": bundle["schema"],
        "candidate_file": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_fingerprint_sha256": bundle["fingerprint_sha256"],
        "module_count": bundle["scope"]["module_count"],
        "totals": bundle["totals"],
        "checkpoint_export": False,
    }


def _solve(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_candidate_bundle(args.candidates)
    optimization = solve_candidate_bundle(
        bundle,
        budget_bytes=args.budget_bytes,
        objective=args.objective,
        max_frontier_states=args.max_frontier_states,
        max_transitions=args.max_transitions,
        max_pareto_comparisons=args.max_pareto_comparisons,
    )
    plan = build_plan_payload(
        args.candidates,
        bundle,
        objective=args.objective,
        optimization=optimization,
    )
    write_plan(args.output, plan)
    return {
        "status": "complete",
        "plan_file": str(args.output.resolve()),
        "plan_sha256": sha256_file(args.output),
        "plan_fingerprint_sha256": plan["fingerprint_sha256"],
        "objective": args.objective,
        "budget_scope": plan["budget_scope"],
        "optimization": optimization.to_dict(),
        "checkpoint_export": False,
    }


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_candidate_bundle(args.candidates)
    aggregated = aggregate_experiment001_transformer_blocks(
        args.candidates,
        bundle,
    )
    written = write_candidate_bundle(args.output, aggregated)
    return {
        "status": "complete",
        "schema": written["schema"],
        "profile": written["profile"],
        "candidate_file": str(args.output.resolve()),
        "candidate_sha256": sha256_file(args.output),
        "candidate_fingerprint_sha256": written["fingerprint_sha256"],
        "allocation_unit_count": written["scope"]["allocation_unit_count"],
        "allocation_granularity": written["scope"]["allocation_granularity"],
    }


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_candidate_bundle(args.candidates)
    result: dict[str, Any] = {
        "status": "pass",
        "candidate_schema": bundle["schema"],
        "candidate_sha256": sha256_file(args.candidates),
        "candidate_fingerprint_sha256": bundle["fingerprint_sha256"],
    }
    if bundle["profile"] == "experiment-001":
        result["candidate_evidence"] = verify_experiment001_evidence(
            args.candidates,
            bundle,
        )
    if args.plan is not None:
        result["plan"] = verify_plan(
            args.candidates,
            args.plan,
            max_frontier_states=args.max_frontier_states,
            max_transitions=args.max_transitions,
            max_pareto_comparisons=args.max_pareto_comparisons,
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-experiment-001":
            result = _build(args)
        elif args.command == "aggregate-blocks":
            result = _aggregate(args)
        elif args.command == "solve":
            result = _solve(args)
        elif args.command == "verify":
            result = _verify(args)
        else:  # pragma: no cover - guarded by argparse
            parser.error(f"unsupported command: {args.command}")
    except (
        ArithmeticError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
