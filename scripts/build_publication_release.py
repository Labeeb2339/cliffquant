"""Build the fail-closed curated Experiment 001 publication evidence bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cliffquant.publication_release import build_publication_release


def _parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absmax-checkpoint", type=Path, required=True)
    parser.add_argument("--cliffquant-checkpoint", type=Path, required=True)
    parser.add_argument("--absmax-structural-report", type=Path, required=True)
    parser.add_argument("--cliffquant-structural-report", type=Path, required=True)
    parser.add_argument("--cliffquant-clean-text-report", type=Path, required=True)
    parser.add_argument("--nll-result-dir", type=Path, required=True)
    parser.add_argument("--proxy-result-dir", type=Path, required=True)
    parser.add_argument("--solver-certificate", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=Path, required=True)
    parser.add_argument("--calibration-sidecar", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--heldout-tokens", type=Path, required=True)
    parser.add_argument("--heldout-sidecar", type=Path, required=True)
    parser.add_argument("--absmax-scale-run-manifest", type=Path, required=True)
    parser.add_argument("--cliffquant-scale-run-manifest", type=Path, required=True)
    parser.add_argument("--figure-manifest", type=Path, required=True)
    parser.add_argument("--proxy-policy-figure", type=Path, required=True)
    parser.add_argument("--proxy-paired-figure", type=Path, required=True)
    parser.add_argument("--heldout-nll-figure", type=Path, required=True)
    parser.add_argument(
        "--gptqmodel-source",
        type=Path,
        required=True,
        help="Pinned local GPTQModel source tree used for the independent NLL rerun.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Explicit CUDA device for the independent NLL rerun (default: cuda:0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Teacher-forced NLL batch size (default: 1).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=repository_root / "research" / "results" / "experiment-001",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_publication_release(
        absmax_checkpoint=args.absmax_checkpoint,
        cliffquant_checkpoint=args.cliffquant_checkpoint,
        absmax_structural_report=args.absmax_structural_report,
        cliffquant_structural_report=args.cliffquant_structural_report,
        cliffquant_clean_text_report=args.cliffquant_clean_text_report,
        nll_result_dir=args.nll_result_dir,
        proxy_result_dir=args.proxy_result_dir,
        solver_certificate=args.solver_certificate,
        calibration_manifest=args.calibration_manifest,
        calibration_tokens=args.calibration_tokens,
        calibration_sidecar=args.calibration_sidecar,
        heldout_manifest=args.heldout_manifest,
        heldout_tokens=args.heldout_tokens,
        heldout_sidecar=args.heldout_sidecar,
        absmax_scale_run_manifest=args.absmax_scale_run_manifest,
        cliffquant_scale_run_manifest=args.cliffquant_scale_run_manifest,
        figure_manifest=args.figure_manifest,
        proxy_policy_figure=args.proxy_policy_figure,
        proxy_paired_figure=args.proxy_paired_figure,
        heldout_nll_figure=args.heldout_nll_figure,
        gptqmodel_source=args.gptqmodel_source,
        destination=args.destination,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
