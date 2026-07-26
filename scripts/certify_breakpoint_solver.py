"""Run the frozen CliffQuant breakpoint-solver certification campaign."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from cliffquant.certification import (
    CertificationConfig,
    certificate_summary,
    default_worker_count,
    run_certification,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    certificate = run_certification(
        args.output,
        CertificationConfig(
            cases=args.cases,
            seed=args.seed,
            workers=args.workers,
            batch_size=args.batch_size,
        ),
        progress_every=args.progress_every,
    )
    print(certificate_summary(certificate))
    return 0 if certificate["protocol_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
