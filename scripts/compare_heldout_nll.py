"""Compare CliffQuant and AbsMax on the exact frozen held-out token manifests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.heldout_nll import compare_heldout_nll


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cliffquant-checkpoint", type=Path, required=True)
    parser.add_argument("--absmax-checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--gptqmodel-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compare_heldout_nll(
        cliffquant_checkpoint=args.cliffquant_checkpoint,
        absmax_checkpoint=args.absmax_checkpoint,
        heldout_manifest=args.heldout_manifest,
        gptqmodel_source=args.gptqmodel_source,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
