"""Check curated Experiment 001 bundle bytes and internal consistency offline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.publication_release import (
    validate_publication_release,
    validate_publication_release_inventory,
)


def _parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "release_dir",
        type=Path,
        nargs="?",
        default=repository_root / "research" / "results" / "experiment-001",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help=(
            "check the exact file inventory, sizes, and hashes without replaying "
            "runtime-bound semantic validators"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validator = (
        validate_publication_release_inventory
        if args.inventory_only
        else validate_publication_release
    )
    result = validator(args.release_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
