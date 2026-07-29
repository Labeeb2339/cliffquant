"""Top-level CliffQuant command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cliffquant",
        description="Correctness-first quantization research tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("command", choices=("autopolicy",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "autopolicy":
        from .autopolicy_cli import main as autopolicy_main

        return autopolicy_main(arguments[1:])
    parser = _parser()
    args = parser.parse_args(arguments)
    parser.error(f"unsupported command: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
