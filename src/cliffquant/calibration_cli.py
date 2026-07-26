"""Command-line entry point for Experiment 001 corpus and diagonal collection."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .activation_stats import collect_model_windows, save_activation_statistics
from .corpus import (
    CALIBRATION_SOURCES,
    ENVIRONMENTS,
    HELDOUT_SOURCES,
    MODEL_ID,
    MODEL_REVISION,
    Phase,
    build_phase_manifest,
    load_manifest,
    load_pinned_rows,
    reject_manifest_overlap,
    save_phase_bundle,
    synthetic_rows,
)
from .provenance import canonical_json_bytes, runtime_metadata, tokenizer_metadata
from .qwen35_modules import QWEN35_GPTQMODEL_TREE_REVISION


class _DryRunTokenizer:
    """Tiny deterministic tokenizer used only by --dry-run."""

    name_or_path = "cliffquant/dry-run-tokenizer"
    vocab_size = 65536
    bos_token_id = None
    eos_token_id = None
    pad_token_id = 0
    unk_token_id = 1
    vocab_files_names: ClassVar[dict[str, str]] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("dry-run tokenizer does not add special tokens")
        return [
            2
            + (
                sum((position + 1) * ord(character) for position, character in enumerate(word))
                % 65534
            )
            for word in text.split()
        ]


def _sources(phase: Phase) -> Mapping[str, Sequence[Any]]:
    return CALIBRATION_SOURCES if phase == "calibration" else HELDOUT_SOURCES


def _load_rows(
    phase: Phase,
    *,
    max_records: int,
    shuffle_buffer: int,
) -> dict[str, dict[str, Iterable[Mapping[str, Any]]]]:
    def deferred_rows(source: Any, environment: str) -> Iterable[Mapping[str, Any]]:
        yield from load_pinned_rows(
            source,
            environment=environment,
            max_records=max_records,
            shuffle_buffer=shuffle_buffer,
        )

    result: dict[str, dict[str, Iterable[Mapping[str, Any]]]] = {}
    for environment in ENVIRONMENTS:
        result[environment] = {}
        for source in _sources(phase)[environment]:
            result[environment][source.key] = deferred_rows(source, environment)
    return result


def _load_tokenizer() -> tuple[Any, dict[str, Any]]:
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install transformers and huggingface-hub for a real collection"
        ) from exc

    tokenizer_patterns = [
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    ]
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=tokenizer_patterns,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    files = {
        path.relative_to(snapshot).as_posix(): path
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }
    return tokenizer, tokenizer_metadata(
        tokenizer,
        model_revision=MODEL_REVISION,
        explicit_files=files,
    )


def _token_arrays(windows: Mapping[str, Sequence[Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        environment: (
            np.asarray([window.input_ids for window in windows[environment]], dtype=np.int64),
            np.asarray(
                [window.attention_mask for window in windows[environment]],
                dtype=np.uint8,
            ),
        )
        for environment in ENVIRONMENTS
    }


def _load_model(device: str) -> Any:
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise RuntimeError("install transformers and PyTorch for model collection") from exc
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype="auto",
    )
    return model.to(device)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("both", "calibration", "heldout"),
        default="both",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=10_000)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        help="required for a heldout-only run so overlap can be rejected",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute a dry run or pinned real collection."""

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.phase == "heldout" and args.calibration_manifest is None:
        raise ValueError("--calibration-manifest is required for --phase heldout")

    phases: tuple[Phase, ...] = (
        ("calibration", "heldout") if args.phase == "both" else (args.phase,)
    )

    if args.dry_run:
        tokenizer: Any = _DryRunTokenizer()
        tokenizer_info = tokenizer_metadata(tokenizer, model_revision=MODEL_REVISION)
        device = "cpu-dry-run"
        count = 4
        window_size = 16
    else:
        tokenizer, tokenizer_info = _load_tokenizer()
        device = args.device
        count = None
        window_size = 256

    runtime_info = runtime_metadata(device=device)
    manifests: dict[str, dict[str, Any]] = {}
    windows_by_phase: dict[str, Mapping[str, Sequence[Any]]] = {}
    for phase in phases:
        rows = (
            synthetic_rows(phase=phase, repetitions=96)
            if args.dry_run
            else _load_rows(
                phase,
                max_records=args.max_records,
                shuffle_buffer=args.shuffle_buffer,
            )
        )
        manifest, windows = build_phase_manifest(
            phase=phase,
            tokenizer=tokenizer,
            rows=rows,
            tokenizer_info=tokenizer_info,
            runtime_info=runtime_info,
            count=count,
            window_size=window_size,
            dry_run=args.dry_run,
        )
        manifest["source_loading"] = {
            "max_records_per_split": args.max_records if not args.dry_run else 96,
            "method": (
                "synthetic"
                if args.dry_run
                else "pinned datasets stream; filter; deterministic Python buffer shuffle"
            ),
            "shuffle_buffer": args.shuffle_buffer if not args.dry_run else None,
        }
        manifests[phase] = manifest
        windows_by_phase[phase] = windows

    calibration_manifest: Mapping[str, Any] | None = manifests.get("calibration")
    if calibration_manifest is None and args.calibration_manifest is not None:
        calibration_manifest = load_manifest(args.calibration_manifest)
    if "heldout" in manifests:
        if calibration_manifest is None:
            raise ValueError("heldout collection requires a calibration manifest")
        reject_manifest_overlap(calibration_manifest, manifests["heldout"])

    results: dict[str, Any] = {"dry_run": args.dry_run, "phases": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = None if args.dry_run or args.corpus_only else _load_model(device)
    for phase in phases:
        bundle_hashes = save_phase_bundle(
            args.output_dir,
            phase=phase,
            manifest=manifests[phase],
            windows=windows_by_phase[phase],
        )
        phase_result: dict[str, Any] = {"corpus": bundle_hashes}
        if model is not None:
            statistics = collect_model_windows(
                model,
                token_arrays=_token_arrays(windows_by_phase[phase]),
                batch_size=args.batch_size,
                device=device,
            )
            phase_result["diagonals"] = save_activation_statistics(
                args.output_dir,
                statistics=statistics,
                phase=phase,
                provenance={
                    "corpus_manifest_sha256": bundle_hashes["manifest"]["sha256"],
                    "gptqmodel_module_tree_revision": QWEN35_GPTQMODEL_TREE_REVISION,
                    "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
                    "runtime": runtime_info,
                    "tokenizer": tokenizer_info,
                },
            )
            del statistics
        results["phases"][phase] = phase_result
    del model

    summary_path = args.output_dir / "collection-summary.json"
    summary_path.write_bytes(canonical_json_bytes(results))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
