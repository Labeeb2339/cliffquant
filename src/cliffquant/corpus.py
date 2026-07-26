"""Frozen corpus rendering, windowing, and manifests for Experiment 001."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .provenance import canonical_json_bytes, sha256_bytes, sha256_file

SEED = 2339
WINDOW_TOKENS = 256
CALIBRATION_WINDOWS = 32
HELDOUT_WINDOWS = 64
MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
XNLI_LANGUAGES = ("ar", "es", "hi", "zh")
ENVIRONMENTS = ("general", "code", "math", "multilingual")


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """One pinned Hub dataset source."""

    dataset: str
    subset: str
    split: str
    revision: str

    @property
    def key(self) -> str:
        return f"{self.dataset}|{self.subset}|{self.split}|{self.revision}"


WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
XNLI_REVISION = "b8dd5d7af51114dbda02c0e3f6133f332186418e"

CALIBRATION_SOURCES: dict[str, tuple[DatasetSource, ...]] = {
    "general": (
        DatasetSource("Salesforce/wikitext", "wikitext-2-raw-v1", "train", WIKITEXT_REVISION),
    ),
    "code": (DatasetSource("google-research-datasets/mbpp", "full", "train", MBPP_REVISION),),
    "math": (DatasetSource("openai/gsm8k", "main", "train", GSM8K_REVISION),),
    "multilingual": (DatasetSource("facebook/xnli", "all_languages", "train", XNLI_REVISION),),
}

HELDOUT_SOURCES: dict[str, tuple[DatasetSource, ...]] = {
    "general": (
        DatasetSource(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            "validation",
            WIKITEXT_REVISION,
        ),
    ),
    "code": (
        DatasetSource("google-research-datasets/mbpp", "full", "validation", MBPP_REVISION),
        DatasetSource("google-research-datasets/mbpp", "full", "test", MBPP_REVISION),
    ),
    "math": (DatasetSource("openai/gsm8k", "main", "test", GSM8K_REVISION),),
    "multilingual": (DatasetSource("facebook/xnli", "all_languages", "validation", XNLI_REVISION),),
}

Phase = Literal["calibration", "heldout"]


@dataclass(frozen=True, slots=True)
class RenderedRecord:
    """A valid, deterministically rendered source record."""

    source: DatasetSource
    source_index: int
    row_id: str
    variant: str | None
    text: str
    rendered_sha256: str


@dataclass(frozen=True, slots=True)
class TokenizedRecord:
    """A rendered record after special-token-free tokenization."""

    rendered: RenderedRecord
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WindowSegment:
    """The exact portion of one rendered row used by a token window."""

    row_id: str
    rendered_sha256: str
    record_token_start: int
    record_token_end: int
    window_token_start: int
    window_token_end: int


@dataclass(frozen=True, slots=True)
class TokenWindow:
    """One fixed-length, non-overlapping token window."""

    environment: str
    index: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_sha256: str
    segments: tuple[WindowSegment, ...]
    language: str | None = None


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _stable_row_id(row: Mapping[str, Any], source: DatasetSource, index: int) -> str:
    for key in ("task_id", "id", "idx", "example_id"):
        value = row.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return f"{source.dataset}:{source.subset}:{source.split}:{key}={value}"
    original_index = row.get("__cliffquant_source_index__", index)
    if isinstance(original_index, bool) or not isinstance(original_index, int):
        raise TypeError("__cliffquant_source_index__ must be an integer")
    return f"{source.dataset}:{source.subset}:{source.split}:row={original_index}"


def _tests_text(value: Any) -> str | None:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        tests = [_clean_text(item) for item in value]
        valid = [item for item in tests if item is not None]
        return "\n".join(valid) if valid else None
    return None


def _translation(value: Any, language: str) -> str | None:
    if isinstance(value, Mapping):
        direct = _clean_text(value.get(language))
        if direct is not None:
            return direct
        languages = value.get("language")
        translations = value.get("translation")
        if (
            isinstance(languages, Sequence)
            and not isinstance(languages, (str, bytes))
            and isinstance(translations, Sequence)
            and not isinstance(translations, (str, bytes))
        ):
            for candidate, translation in zip(languages, translations, strict=False):
                if candidate == language:
                    return _clean_text(translation)
    return None


def render_record(
    environment: str,
    row: Mapping[str, Any],
    source: DatasetSource,
    source_index: int,
) -> tuple[RenderedRecord, ...]:
    """Render a source row using the frozen field-labelled templates."""

    if environment not in ENVIRONMENTS:
        raise ValueError(f"unsupported environment: {environment}")
    if not isinstance(row, Mapping):
        return ()
    row_id = _stable_row_id(row, source, source_index)
    outputs: list[tuple[str | None, str]] = []

    if environment == "general":
        text = _clean_text(row.get("text"))
        if text is not None:
            outputs.append((None, text + "\n\n"))
    elif environment == "code":
        task = _clean_text(row.get("task")) or _clean_text(row.get("text"))
        solution = _clean_text(row.get("solution")) or _clean_text(row.get("code"))
        tests = _tests_text(row.get("tests")) or _tests_text(row.get("test_list"))
        if task is not None and solution is not None and tests is not None:
            outputs.append(
                (
                    None,
                    f"Task:\n{task}\n\nSolution:\n{solution}\n\nTests:\n{tests}\n\n",
                )
            )
    elif environment == "math":
        question = _clean_text(row.get("question"))
        answer = _clean_text(row.get("answer"))
        if question is not None and answer is not None:
            outputs.append((None, f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"))
    else:
        premise = row.get("premise")
        hypothesis = row.get("hypothesis")
        row_language = row.get("language")
        if isinstance(row_language, str):
            language = row_language.strip()
            if language in XNLI_LANGUAGES:
                premise_text = _clean_text(premise)
                hypothesis_text = _clean_text(hypothesis)
                if premise_text is not None and hypothesis_text is not None:
                    outputs.append(
                        (
                            language,
                            (
                                f"Language: {language}\nPremise:\n{premise_text}\n\n"
                                f"Hypothesis:\n{hypothesis_text}\n\n"
                            ),
                        )
                    )
        else:
            for language in XNLI_LANGUAGES:
                premise_text = _translation(premise, language)
                hypothesis_text = _translation(hypothesis, language)
                if premise_text is not None and hypothesis_text is not None:
                    outputs.append(
                        (
                            language,
                            (
                                f"Language: {language}\nPremise:\n{premise_text}\n\n"
                                f"Hypothesis:\n{hypothesis_text}\n\n"
                            ),
                        )
                    )

    return tuple(
        RenderedRecord(
            source=source,
            source_index=source_index,
            row_id=f"{row_id}:lang={variant}" if variant is not None else row_id,
            variant=variant,
            text=text,
            rendered_sha256=sha256_bytes(text.encode()),
        )
        for variant, text in outputs
    )


def _tokenize(tokenizer: Any, record: RenderedRecord) -> TokenizedRecord:
    try:
        values = tokenizer.encode(record.text, add_special_tokens=False)
    except TypeError:
        values = tokenizer.encode(record.text)
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("tokenizer.encode must return a sequence of token IDs")
    token_ids: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("tokenizer returned a non-integer token ID")
        token_id = int(value)
        if token_id < 0:
            raise ValueError("tokenizer returned a negative token ID")
        token_ids.append(token_id)
    return TokenizedRecord(record, tuple(token_ids))


def _rank(record: RenderedRecord, *, environment: str, phase: Phase, seed: int) -> str:
    material = (
        f"{seed}\0{environment}\0{phase}\0{record.source.key}\0"
        f"{record.row_id}\0{record.rendered_sha256}"
    ).encode()
    return sha256_bytes(material)


def _ordered_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    environment: str,
    source: DatasetSource,
    phase: Phase,
    seed: int,
    language: str | None,
) -> list[RenderedRecord]:
    candidates: list[RenderedRecord] = []
    for fallback_index, row in enumerate(rows):
        source_index = row.get("__cliffquant_source_index__", fallback_index)
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise TypeError("__cliffquant_source_index__ must be an integer")
        candidates.extend(
            rendered
            for rendered in render_record(environment, row, source, source_index)
            if language is None or rendered.variant == language
        )
    candidates.sort(
        key=lambda item: (_rank(item, environment=environment, phase=phase, seed=seed), item.row_id)
    )
    unique: list[RenderedRecord] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        if candidate.rendered_sha256 not in seen_hashes:
            seen_hashes.add(candidate.rendered_sha256)
            unique.append(candidate)
    return unique


def _emit_windows(
    records: Sequence[RenderedRecord],
    *,
    tokenizer: Any,
    environment: str,
    count: int,
    window_size: int,
    language: str | None,
) -> tuple[list[TokenWindow], list[TokenizedRecord]]:
    if count <= 0 or window_size <= 0:
        raise ValueError("count and window_size must be positive")

    windows: list[TokenWindow] = []
    used: dict[str, TokenizedRecord] = {}
    buffer: list[tuple[int, TokenizedRecord, int]] = []
    cursor = 0
    for rendered in records:
        tokenized = _tokenize(tokenizer, rendered)
        if not tokenized.token_ids:
            continue
        buffer.extend(
            (token_id, tokenized, record_position)
            for record_position, token_id in enumerate(tokenized.token_ids)
        )
        while len(buffer) - cursor >= window_size and len(windows) < count:
            chunk = buffer[cursor : cursor + window_size]
            cursor += window_size
            ids = tuple(item[0] for item in chunk)
            segments: list[WindowSegment] = []
            segment_start = 0
            while segment_start < len(chunk):
                tokenized_record = chunk[segment_start][1]
                record_start = chunk[segment_start][2]
                segment_end = segment_start + 1
                while (
                    segment_end < len(chunk)
                    and chunk[segment_end][1].rendered.rendered_sha256
                    == tokenized_record.rendered.rendered_sha256
                    and chunk[segment_end][2] == record_start + (segment_end - segment_start)
                ):
                    segment_end += 1
                segments.append(
                    WindowSegment(
                        row_id=tokenized_record.rendered.row_id,
                        rendered_sha256=tokenized_record.rendered.rendered_sha256,
                        record_token_start=record_start,
                        record_token_end=record_start + (segment_end - segment_start),
                        window_token_start=segment_start,
                        window_token_end=segment_end,
                    )
                )
                used[tokenized_record.rendered.rendered_sha256] = tokenized_record
                segment_start = segment_end

            windows.append(
                TokenWindow(
                    environment=environment,
                    index=len(windows),
                    input_ids=ids,
                    attention_mask=(1,) * window_size,
                    token_sha256=sha256_bytes(canonical_json_bytes(list(ids))),
                    segments=tuple(segments),
                    language=language,
                )
            )
        if len(windows) == count:
            break
        if cursor:
            del buffer[:cursor]
            cursor = 0
    return windows, list(used.values())


def _copy_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return list(rows)


def build_environment_windows(
    environment: str,
    *,
    phase: Phase,
    tokenizer: Any,
    rows_by_source: Mapping[str, Iterable[Mapping[str, Any]]],
    count: int | None = None,
    window_size: int = WINDOW_TOKENS,
    seed: int = SEED,
) -> tuple[list[TokenWindow], list[TokenizedRecord]]:
    """Build exact-length windows, never carrying a partial tail across splits."""

    if phase not in ("calibration", "heldout"):
        raise ValueError("phase must be calibration or heldout")
    if environment not in ENVIRONMENTS:
        raise ValueError(f"unsupported environment: {environment}")
    sources = CALIBRATION_SOURCES if phase == "calibration" else HELDOUT_SOURCES
    desired = (
        count
        if count is not None
        else (CALIBRATION_WINDOWS if phase == "calibration" else HELDOUT_WINDOWS)
    )
    if desired <= 0:
        raise ValueError("count must be positive")

    source_rows: dict[str, list[Mapping[str, Any]]] = {}
    for source in sources[environment]:
        if source.key not in rows_by_source:
            raise KeyError(f"missing rows for pinned source {source.key}")

    def rows_for(source: DatasetSource) -> list[Mapping[str, Any]]:
        if source.key not in source_rows:
            source_rows[source.key] = _copy_rows(rows_by_source[source.key])
        return source_rows[source.key]

    languages: tuple[str | None, ...]
    if environment == "multilingual":
        if desired % len(XNLI_LANGUAGES):
            raise ValueError("multilingual window count must be divisible by four")
        languages = XNLI_LANGUAGES
    else:
        languages = (None,)

    all_windows: list[TokenWindow] = []
    all_used: dict[str, TokenizedRecord] = {}
    per_language_target = desired // len(languages)
    for language in languages:
        language_windows: list[TokenWindow] = []
        for source in sources[environment]:
            remaining = per_language_target - len(language_windows)
            if remaining == 0:
                break
            ordered = _ordered_records(
                rows_for(source),
                environment=environment,
                source=source,
                phase=phase,
                seed=seed,
                language=language,
            )
            emitted, used = _emit_windows(
                ordered,
                tokenizer=tokenizer,
                environment=environment,
                count=remaining,
                window_size=window_size,
                language=language,
            )
            language_windows.extend(emitted)
            all_used.update({record.rendered.rendered_sha256: record for record in used})
            # Any tail is intentionally discarded here so no window crosses a split.
        if len(language_windows) != per_language_target:
            label = language or environment
            raise ValueError(
                f"insufficient valid tokens for {label}: built {len(language_windows)} "
                f"of {per_language_target} windows"
            )
        all_windows.extend(language_windows)

    if environment == "multilingual":
        by_language: dict[str, list[TokenWindow]] = defaultdict(list)
        for window in all_windows:
            if window.language is None:
                raise AssertionError("multilingual window lost its language label")
            by_language[window.language].append(window)
        all_windows = [
            by_language[language][position]
            for position in range(per_language_target)
            for language in XNLI_LANGUAGES
        ]

    all_windows = [
        TokenWindow(
            environment=window.environment,
            index=index,
            input_ids=window.input_ids,
            attention_mask=window.attention_mask,
            token_sha256=window.token_sha256,
            segments=window.segments,
            language=window.language,
        )
        for index, window in enumerate(all_windows)
    ]
    return all_windows, sorted(
        all_used.values(),
        key=lambda item: (item.rendered.source.key, item.rendered.row_id),
    )


def build_phase_manifest(
    *,
    phase: Phase,
    tokenizer: Any,
    rows: Mapping[str, Mapping[str, Iterable[Mapping[str, Any]]]],
    tokenizer_info: Mapping[str, Any],
    runtime_info: Mapping[str, Any],
    count: int | None = None,
    window_size: int = WINDOW_TOKENS,
    seed: int = SEED,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, list[TokenWindow]]]:
    """Build all four environments and their canonical manifest payload."""

    desired = (
        count
        if count is not None
        else (CALIBRATION_WINDOWS if phase == "calibration" else HELDOUT_WINDOWS)
    )
    source_table = CALIBRATION_SOURCES if phase == "calibration" else HELDOUT_SOURCES
    windows_by_environment: dict[str, list[TokenWindow]] = {}
    environment_manifests: dict[str, Any] = {}

    for environment in ENVIRONMENTS:
        windows, used = build_environment_windows(
            environment,
            phase=phase,
            tokenizer=tokenizer,
            rows_by_source=rows[environment],
            count=desired,
            window_size=window_size,
            seed=seed,
        )
        windows_by_environment[environment] = windows
        environment_manifests[environment] = {
            "datasets": [asdict(source) for source in source_table[environment]],
            "records": [
                {
                    "dataset": item.rendered.source.dataset,
                    "rendered_sha256": item.rendered.rendered_sha256,
                    "revision": item.rendered.source.revision,
                    "row_id": item.rendered.row_id,
                    "source_index": item.rendered.source_index,
                    "split": item.rendered.source.split,
                    "subset": item.rendered.source.subset,
                    "token_count": len(item.token_ids),
                    "variant": item.rendered.variant,
                }
                for item in used
            ],
            "windows": [
                {
                    "index": window.index,
                    "language": window.language,
                    "segments": [asdict(segment) for segment in window.segments],
                    "token_count": len(window.input_ids),
                    "token_sha256": window.token_sha256,
                }
                for window in windows
            ],
        }

    manifest = {
        "dry_run": dry_run,
        "environments": environment_manifests,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "phase": phase,
        "rendering": {
            "code": "Task/Solution/Tests labelled template, trailing blank line",
            "general": "trimmed text with trailing blank line",
            "math": "Question/Answer labelled template, trailing blank line",
            "multilingual": (
                "one Language/Premise/Hypothesis record per ar/es/hi/zh translation; "
                "equal windows per language"
            ),
        },
        "runtime": dict(runtime_info),
        "sampling": {
            "deduplication": "first seeded-rank occurrence of each rendered SHA256",
            "non_overlapping": True,
            "ordering": "SHA256(seed, environment, phase, source, row ID, rendered SHA256)",
            "seed": seed,
            "split_boundary_policy": "discard incomplete tail before the next split",
        },
        "schema": "cliffquant.corpus-manifest.v1",
        "tokenizer": dict(tokenizer_info),
        "window_count_per_environment": desired,
        "window_tokens": window_size,
    }
    return manifest, windows_by_environment


def _manifest_hash_sets(manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    rendered: set[str] = set()
    windows: set[str] = set()
    environments = manifest.get("environments")
    if not isinstance(environments, Mapping):
        raise ValueError("manifest has no environments mapping")
    for value in environments.values():
        if not isinstance(value, Mapping):
            raise ValueError("invalid environment manifest")
        for record in value.get("records", []):
            rendered.add(str(record["rendered_sha256"]))
        for window in value.get("windows", []):
            windows.add(str(window["token_sha256"]))
    return rendered, windows


def reject_manifest_overlap(
    calibration_manifest: Mapping[str, Any],
    heldout_manifest: Mapping[str, Any],
) -> None:
    """Reject exact rendered-record or token-window reuse across the boundary."""

    calibration_records, calibration_windows = _manifest_hash_sets(calibration_manifest)
    heldout_records, heldout_windows = _manifest_hash_sets(heldout_manifest)
    record_overlap = sorted(calibration_records & heldout_records)
    window_overlap = sorted(calibration_windows & heldout_windows)
    if record_overlap or window_overlap:
        details = []
        if record_overlap:
            details.append(f"{len(record_overlap)} rendered-text hashes")
        if window_overlap:
            details.append(f"{len(window_overlap)} token-window hashes")
        raise ValueError("calibration/held-out overlap detected: " + ", ".join(details))


def save_phase_bundle(
    output_dir: str | Path,
    *,
    phase: Phase,
    manifest: Mapping[str, Any],
    windows: Mapping[str, Sequence[TokenWindow]],
) -> dict[str, Any]:
    """Save exact token arrays, canonical manifest JSON, and content hashes."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    npz_path = destination / f"{phase}.tokens.npz"
    arrays: dict[str, np.ndarray] = {}
    for environment in ENVIRONMENTS:
        environment_windows = windows[environment]
        arrays[f"{environment}__input_ids"] = np.asarray(
            [window.input_ids for window in environment_windows],
            dtype=np.int64,
        )
        arrays[f"{environment}__attention_mask"] = np.asarray(
            [window.attention_mask for window in environment_windows],
            dtype=np.uint8,
        )
    np.savez_compressed(npz_path, **arrays)

    enriched = dict(manifest)
    enriched["token_archive"] = {
        "file": npz_path.name,
        "sha256": sha256_file(npz_path),
        "size_bytes": npz_path.stat().st_size,
    }
    manifest_path = destination / f"{phase}.manifest.json"
    manifest_bytes = canonical_json_bytes(enriched)
    manifest_path.write_bytes(manifest_bytes)
    hashes = {
        "manifest": {
            "file": manifest_path.name,
            "sha256": sha256_bytes(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        },
        "tokens": enriched["token_archive"],
    }
    hashes_path = destination / f"{phase}.sha256.json"
    hashes_path.write_bytes(canonical_json_bytes(hashes))
    return hashes


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a strict JSON manifest and verify its token archive when present."""

    import json

    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = manifest.get("token_archive")
    if isinstance(archive, Mapping):
        archive_path = manifest_path.parent / str(archive["file"])
        if sha256_file(archive_path) != archive["sha256"]:
            raise ValueError(f"token archive hash mismatch: {archive_path}")
    return manifest


def load_pinned_rows(
    source: DatasetSource,
    *,
    environment: str,
    seed: int = SEED,
    max_records: int = 10_000,
    shuffle_buffer: int = 10_000,
) -> list[Mapping[str, Any]]:
    """Stream a bounded, seeded pool from a pinned Hub revision.

    Streaming avoids materializing XNLI's roughly gigabyte-scale Arrow cache.
    The pool and buffer sizes are part of the caller's manifest provenance.
    """

    if max_records <= 0 or shuffle_buffer <= 0:
        raise ValueError("max_records and shuffle_buffer must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the optional 'datasets' package for Hub collection") from exc

    stream = load_dataset(
        source.dataset,
        source.subset,
        split=source.split,
        revision=source.revision,
        streaming=True,
    )
    # Filtering precedes sampling. The local buffer shuffle avoids relying on
    # a mutable Dataset Viewer response and preserves each original row offset.
    generator = iter(stream)
    buffer: list[Mapping[str, Any]] = []
    source_index = 0
    while len(buffer) < shuffle_buffer:
        try:
            row = next(generator)
        except StopIteration:
            break
        current_index = source_index
        source_index += 1
        if not isinstance(row, Mapping):
            continue
        if not render_record(environment, row, source, current_index):
            continue
        enriched = dict(row)
        enriched["__cliffquant_source_index__"] = current_index
        buffer.append(enriched)

    rng = random.Random(seed)
    rows: list[Mapping[str, Any]] = []
    while buffer and len(rows) < max_records:
        selected_index = rng.randrange(len(buffer))
        rows.append(buffer[selected_index])
        replacement: Mapping[str, Any] | None = None
        while replacement is None:
            try:
                row = next(generator)
            except StopIteration:
                break
            current_index = source_index
            source_index += 1
            if not isinstance(row, Mapping):
                continue
            if not render_record(environment, row, source, current_index):
                continue
            enriched = dict(row)
            enriched["__cliffquant_source_index__"] = current_index
            replacement = enriched
        if replacement is None:
            buffer.pop(selected_index)
        else:
            buffer[selected_index] = replacement
        if len(rows) == max_records:
            break
    if not rows:
        raise ValueError(f"pinned source produced no records: {source.key}")
    return rows


def synthetic_rows(
    *,
    phase: Phase,
    repetitions: int = 64,
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    """Create deterministic local records for CLI and unit-test dry runs."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    sources = CALIBRATION_SOURCES if phase == "calibration" else HELDOUT_SOURCES
    result: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    salt = 0 if phase == "calibration" else 10_000
    for environment in ENVIRONMENTS:
        result[environment] = {}
        for source_position, source in enumerate(sources[environment]):
            records: list[Mapping[str, Any]] = []
            for index in range(repetitions):
                unique = salt + source_position * repetitions + index
                if environment == "general":
                    row: Mapping[str, Any] = {
                        "id": unique,
                        "text": f"General synthetic record {unique} has stable words for dry run.",
                    }
                elif environment == "code":
                    row = {
                        "task_id": unique,
                        "text": f"Return the synthetic integer {unique}.",
                        "code": f"def answer():\n    return {unique}",
                        "test_list": [f"assert answer() == {unique}"],
                    }
                elif environment == "math":
                    row = {
                        "id": unique,
                        "question": f"What is {unique} plus one?",
                        "answer": str(unique + 1),
                    }
                else:
                    row = {
                        "id": unique,
                        "premise": {
                            language: f"{language} premise synthetic {unique}"
                            for language in XNLI_LANGUAGES
                        },
                        "hypothesis": {
                            language: f"{language} hypothesis synthetic {unique}"
                            for language in XNLI_LANGUAGES
                        },
                    }
                records.append(row)
            result[environment][source.key] = records
    return result
