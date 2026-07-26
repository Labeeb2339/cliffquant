"""Verify CliffQuant GPTQ structure or run fresh text/image smoke tests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.full_model_verify import (
    run_image_generation_smoke,
    run_text_generation_smoke,
    verify_full_model_structure,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    structural = subparsers.add_parser("structural")
    structural.add_argument("--base-model", type=Path, required=True)
    structural.add_argument("--scale-run", type=Path, required=True)
    structural.add_argument("--corpus-directory", type=Path, required=True)
    structural.add_argument("--checkpoint", type=Path, required=True)
    structural.add_argument("--report", type=Path, required=True)
    structural.add_argument("--row-chunk-size", type=int, default=32)
    structural.add_argument("--dequant-blocks-per-module", type=int, default=2)

    text = subparsers.add_parser("text")
    text.add_argument("--checkpoint", type=Path, required=True)
    text.add_argument("--gptqmodel-source", type=Path, required=True)
    text.add_argument("--prompt", default="Explain why exact arithmetic checks matter.")
    text.add_argument("--device", default="cuda:0")
    text.add_argument("--max-new-tokens", type=int, default=16)

    image = subparsers.add_parser("image")
    image.add_argument("--checkpoint", type=Path, required=True)
    image.add_argument("--gptqmodel-source", type=Path, required=True)
    image.add_argument("--image", type=Path, required=True)
    image.add_argument("--prompt", default="Describe this image briefly.")
    image.add_argument("--device", default="cuda:0")
    image.add_argument("--max-new-tokens", type=int, default=16)
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
            prompt=args.prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        result = run_image_generation_smoke(
            checkpoint_dir=args.checkpoint,
            gptqmodel_source=args.gptqmodel_source,
            image_path=args.image,
            prompt=args.prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
