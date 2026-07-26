"""Check the pinned GPTQModel packer against CliffQuant's independent reference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.gptqmodel_export import run_gptqmodel_pack_parity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gptqmodel-source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2339)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            run_gptqmodel_pack_parity(
                gptqmodel_source=args.gptqmodel_source,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
