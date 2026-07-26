"""Recompute and validate a saved held-out NLL evidence bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.heldout_nll import validate_heldout_nll_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--heldout-manifest", type=Path)
    parser.add_argument("--absmax-checkpoint", type=Path)
    parser.add_argument("--cliffquant-checkpoint", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_heldout_nll_result(
        args.result_directory,
        heldout_manifest=args.heldout_manifest,
        absmax_checkpoint=args.absmax_checkpoint,
        cliffquant_checkpoint=args.cliffquant_checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
