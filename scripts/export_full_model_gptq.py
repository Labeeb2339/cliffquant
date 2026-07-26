"""Export a completed CliffQuant scale run as standard GPTQ-v1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.gptqmodel_export import export_full_model_gptq


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--scale-run", type=Path, required=True)
    parser.add_argument("--corpus-directory", type=Path, required=True)
    parser.add_argument("--gptqmodel-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-row-chunk-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = export_full_model_gptq(
        base_model_dir=args.base_model,
        scale_run_dir=args.scale_run,
        corpus_directory=args.corpus_directory,
        gptqmodel_source=args.gptqmodel_source,
        output_dir=args.output,
        verify_row_chunk_size=args.verify_row_chunk_size,
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
