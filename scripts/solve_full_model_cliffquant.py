"""Run the resumable frozen Qwen3.5 full-model CliffQuant scale search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.full_model_solve import run_frozen_full_model_scale_search


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("cliffquant_minimax", "absmax"),
        required=True,
    )
    parser.add_argument("--solver-certificate", type=Path, required=True)
    parser.add_argument("--row-chunk-size", type=int, default=8)
    parser.add_argument("--solver-chunk-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_frozen_full_model_scale_search(
        base_model_dir=args.base_model,
        calibration_metadata=args.calibration_metadata,
        output_dir=args.output,
        solver_certificate=args.solver_certificate,
        policy=args.policy,
        row_chunk_size=args.row_chunk_size,
        solver_chunk_size=args.solver_chunk_size,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "modules": len(result["modules"]),
                "output": str(args.output.resolve()),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
