"""Verify CliffQuant GPTQ structure or run fresh text/image smoke tests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.full_model_verify import (
    load_verification_report,
    run_image_generation_smoke,
    run_text_generation_smoke,
    verify_full_model_structure,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    structural = subparsers.add_parser(
        "structural",
        help="independently unpack and compare every exported tensor",
    )
    structural.add_argument("--base-model", type=Path, required=True)
    structural.add_argument("--scale-run", type=Path, required=True)
    structural.add_argument("--corpus-directory", type=Path, required=True)
    structural.add_argument("--checkpoint", type=Path, required=True)
    structural.add_argument("--report", type=Path, required=True)
    structural.add_argument("--row-chunk-size", type=int, default=32)
    structural.add_argument("--dequant-blocks-per-module", type=int, default=2)

    text = subparsers.add_parser(
        "text",
        help="fresh-load once, generate twice, and persist deterministic finite text evidence",
    )
    text.add_argument("--checkpoint", type=Path, required=True)
    text_gptqmodel = text.add_mutually_exclusive_group(required=True)
    text_gptqmodel.add_argument(
        "--gptqmodel-source",
        type=Path,
        help="explicit clean checkout of the pinned GPTQModel commit",
    )
    text_gptqmodel.add_argument(
        "--installed-gptqmodel",
        action="store_true",
        help="load only the pinned GPTQModel distribution installed in the active venv",
    )
    text.add_argument("--report", type=Path, required=True)
    text.add_argument(
        "--environment-report",
        type=Path,
        help="canonical clean-environment report to validate and embed",
    )
    text.add_argument("--prompt", default="Explain why exact arithmetic checks matter.")
    text.add_argument("--device", default="cuda:0")
    text.add_argument("--max-new-tokens", type=int, default=16)

    image = subparsers.add_parser(
        "image",
        help="load once, generate twice, and persist finite deterministic image-text evidence",
    )
    image.add_argument("--checkpoint", type=Path, required=True)
    image_gptqmodel = image.add_mutually_exclusive_group(required=True)
    image_gptqmodel.add_argument(
        "--gptqmodel-source",
        type=Path,
        help="explicit clean checkout of the pinned GPTQModel commit",
    )
    image_gptqmodel.add_argument(
        "--installed-gptqmodel",
        action="store_true",
        help="load only the pinned GPTQModel distribution installed in the active venv",
    )
    image.add_argument("--image", type=Path, required=True)
    image.add_argument("--report", type=Path, required=True)
    image.add_argument(
        "--environment-report",
        type=Path,
        help="canonical clean-environment report to validate and embed",
    )
    image.add_argument("--prompt", default="Describe this image briefly.")
    image.add_argument("--device", default="cuda:0")
    image.add_argument("--max-new-tokens", type=int, default=16)

    report = subparsers.add_parser(
        "report",
        help="validate a canonical report and its hash sidecar",
    )
    report.add_argument("--report", type=Path, required=True)
    report.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="live exported checkpoint whose bytes must match the report",
    )
    report.add_argument(
        "--require-clean-environment",
        action="store_true",
        help="reject runtime evidence not produced by the clean-venv runner",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "structural":
        result = verify_full_model_structure(
            base_model_dir=args.base_model,
            scale_run_dir=args.scale_run,
            corpus_directory=args.corpus_directory,
            checkpoint_dir=args.checkpoint,
            report_path=args.report,
            row_chunk_size=args.row_chunk_size,
            dequant_blocks_per_module=args.dequant_blocks_per_module,
        )
    elif args.command == "text":
        result = run_text_generation_smoke(
            checkpoint_dir=args.checkpoint,
            gptqmodel_source=args.gptqmodel_source,
            report_path=args.report,
            prompt=args.prompt,
            environment_report=args.environment_report,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.command == "image":
        result = run_image_generation_smoke(
            checkpoint_dir=args.checkpoint,
            gptqmodel_source=args.gptqmodel_source,
            image_path=args.image,
            report_path=args.report,
            prompt=args.prompt,
            environment_report=args.environment_report,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        result = load_verification_report(
            args.report,
            checkpoint_dir=args.checkpoint,
            require_clean_environment=args.require_clean_environment,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
