"""Build a new pinned venv and run publication-gate GPTQ smoke checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.clean_environment import (
    default_python_executable,
    run_clean_verification,
)


def _parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The work directory must not exist. The runner creates a new venv, "
            "installs only exact locked versions, clones the frozen GPTQModel commit, "
            "runs pip check, and binds the resulting environment report into each smoke report."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--work-directory",
        type=Path,
        required=True,
        help="new disposable directory; launch is rejected if it already exists",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=default_python_executable(),
        help="CPython 3.11.15 executable used only to create the new venv",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=repository / "requirements" / "verification-cu128.txt",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        required=True,
        help=(
            "external wheel-only directory used offline with --no-index, "
            "--require-hashes, and --only-binary"
        ),
    )
    parser.add_argument(
        "--hashed-lock",
        type=Path,
        required=True,
        help="committed --require-hashes lock generated for the wheelhouse",
    )
    parser.add_argument(
        "--prompt",
        default="Explain why exact arithmetic checks matter.",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-prompt", default="Describe this image briefly.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_clean_verification(
        checkpoint_dir=args.checkpoint,
        output_directory=args.output_directory,
        work_directory=args.work_directory,
        python_executable=args.python,
        lock_path=args.lock,
        prompt=args.prompt,
        image_path=args.image,
        image_prompt=args.image_prompt,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        hashed_lock_path=args.hashed_lock,
        wheelhouse=args.wheelhouse,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
