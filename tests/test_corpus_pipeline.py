from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from cliffquant.corpus import (
    CALIBRATION_SOURCES,
    CALIBRATION_WINDOWS,
    ENVIRONMENTS,
    HELDOUT_SOURCES,
    HELDOUT_WINDOWS,
    MODEL_REVISION,
    SEED,
    WINDOW_TOKENS,
    XNLI_LANGUAGES,
    build_environment_windows,
    build_phase_manifest,
    reject_manifest_overlap,
    render_record,
    save_phase_bundle,
    synthetic_rows,
)
from cliffquant.provenance import sha256_file


class WordTokenizer:
    name_or_path = "synthetic"
    vocab_size = 100_000
    bos_token_id = None
    eos_token_id = None
    pad_token_id = 0
    unk_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [2 + sum(map(ord, word)) for word in text.split()]


def test_protocol_constants_are_frozen() -> None:
    assert SEED == 2339
    assert WINDOW_TOKENS == 256
    assert CALIBRATION_WINDOWS == 32
    assert HELDOUT_WINDOWS == 64
    assert MODEL_REVISION == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    assert CALIBRATION_SOURCES["general"][0].revision == (
        "b08601e04326c79dfdd32d625aee71d232d685c3"
    )
    assert CALIBRATION_SOURCES["code"][0].revision == ("4bb6404fdc6cacfda99d4ac4205087b89d32030c")
    assert CALIBRATION_SOURCES["math"][0].revision == ("740312add88f781978c0658806c59bc2815b9866")
    assert CALIBRATION_SOURCES["multilingual"][0].revision == (
        "b8dd5d7af51114dbda02c0e3f6133f332186418e"
    )
    assert [source.split for source in HELDOUT_SOURCES["code"]] == ["validation", "test"]


def test_renderers_filter_invalid_rows_and_expand_xnli_languages() -> None:
    tokenizer_source = CALIBRATION_SOURCES["general"][0]
    assert render_record("general", {"text": "  "}, tokenizer_source, 0) == ()

    code = render_record(
        "code",
        {"task_id": 7, "text": "Do it", "code": "pass", "test_list": ["assert True"]},
        CALIBRATION_SOURCES["code"][0],
        0,
    )
    assert code[0].text == "Task:\nDo it\n\nSolution:\npass\n\nTests:\nassert True\n\n"

    languages = list(XNLI_LANGUAGES)
    multilingual = render_record(
        "multilingual",
        {
            "premise": {
                "language": languages,
                "translation": [f"premise-{language}" for language in languages],
            },
            "hypothesis": {
                "language": languages,
                "translation": [f"hypothesis-{language}" for language in languages],
            },
        },
        CALIBRATION_SOURCES["multilingual"][0],
        4,
    )
    assert tuple(record.variant for record in multilingual) == XNLI_LANGUAGES
    assert all(record.rendered_sha256 for record in multilingual)


def test_windowing_is_deterministic_balanced_and_non_overlapping() -> None:
    rows = synthetic_rows(phase="calibration", repetitions=96)
    tokenizer = WordTokenizer()
    first, _ = build_environment_windows(
        "multilingual",
        phase="calibration",
        tokenizer=tokenizer,
        rows_by_source=rows["multilingual"],
        count=8,
        window_size=16,
    )
    second, _ = build_environment_windows(
        "multilingual",
        phase="calibration",
        tokenizer=tokenizer,
        rows_by_source=rows["multilingual"],
        count=8,
        window_size=16,
    )
    assert [window.token_sha256 for window in first] == [window.token_sha256 for window in second]
    assert [window.language for window in first] == ["ar", "es", "hi", "zh"] * 2
    assert all(len(window.input_ids) == 16 for window in first)

    intervals: dict[str, list[tuple[int, int]]] = {}
    for window in first:
        for segment in window.segments:
            intervals.setdefault(segment.rendered_sha256, []).append(
                (segment.record_token_start, segment.record_token_end)
            )
    for row_intervals in intervals.values():
        ordered = sorted(row_intervals)
        assert all(left[1] <= right[0] for left, right in pairwise(ordered))


def test_row_input_order_does_not_change_seeded_selection() -> None:
    rows = synthetic_rows(phase="calibration", repetitions=96)
    tokenizer = WordTokenizer()
    source = CALIBRATION_SOURCES["general"][0]
    original, _ = build_environment_windows(
        "general",
        phase="calibration",
        tokenizer=tokenizer,
        rows_by_source=rows["general"],
        count=4,
        window_size=16,
    )
    reversed_rows = {source.key: list(reversed(rows["general"][source.key]))}
    reversed_windows, _ = build_environment_windows(
        "general",
        phase="calibration",
        tokenizer=tokenizer,
        rows_by_source=reversed_rows,
        count=4,
        window_size=16,
    )
    assert [window.token_sha256 for window in original] == [
        window.token_sha256 for window in reversed_windows
    ]


def test_mbpp_test_split_is_not_consumed_when_validation_is_sufficient() -> None:
    validation, test = HELDOUT_SOURCES["code"]

    def forbidden_test_rows():
        raise AssertionError("test split should be lazy")
        yield

    validation_row = {
        "task_id": 1,
        "text": " ".join(["task"] * 32),
        "code": " ".join(["solution"] * 32),
        "test_list": [" ".join(["test"] * 32)],
    }
    windows, _ = build_environment_windows(
        "code",
        phase="heldout",
        tokenizer=WordTokenizer(),
        rows_by_source={
            validation.key: [validation_row],
            test.key: forbidden_test_rows(),
        },
        count=1,
        window_size=16,
    )
    assert len(windows) == 1


def test_phase_manifests_reject_record_and_window_hash_overlap() -> None:
    tokenizer = WordTokenizer()
    calibration, _ = build_phase_manifest(
        phase="calibration",
        tokenizer=tokenizer,
        rows=synthetic_rows(phase="calibration", repetitions=96),
        tokenizer_info={"name": "synthetic"},
        runtime_info={"device": "cpu"},
        count=4,
        window_size=16,
        dry_run=True,
    )
    heldout, _ = build_phase_manifest(
        phase="heldout",
        tokenizer=tokenizer,
        rows=synthetic_rows(phase="heldout", repetitions=96),
        tokenizer_info={"name": "synthetic"},
        runtime_info={"device": "cpu"},
        count=4,
        window_size=16,
        dry_run=True,
    )
    reject_manifest_overlap(calibration, heldout)
    heldout["environments"]["general"]["records"][0]["rendered_sha256"] = calibration[
        "environments"
    ]["general"]["records"][0]["rendered_sha256"]
    with pytest.raises(ValueError, match="rendered-text"):
        reject_manifest_overlap(calibration, heldout)


def test_phase_bundle_contains_exact_arrays_and_verified_hashes(tmp_path: Path) -> None:
    tokenizer = WordTokenizer()
    manifest, windows = build_phase_manifest(
        phase="calibration",
        tokenizer=tokenizer,
        rows=synthetic_rows(phase="calibration", repetitions=96),
        tokenizer_info={"name": "synthetic"},
        runtime_info={"device": "cpu"},
        count=4,
        window_size=16,
        dry_run=True,
    )
    hashes = save_phase_bundle(
        tmp_path,
        phase="calibration",
        manifest=manifest,
        windows=windows,
    )
    manifest_path = tmp_path / "calibration.manifest.json"
    archive_path = tmp_path / "calibration.tokens.npz"
    assert hashes["manifest"]["sha256"] == sha256_file(manifest_path)
    assert hashes["tokens"]["sha256"] == sha256_file(archive_path)
    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert parsed["token_archive"]["sha256"] == sha256_file(archive_path)
    with np.load(archive_path) as arrays:
        for environment in ENVIRONMENTS:
            assert arrays[f"{environment}__input_ids"].shape == (4, 16)
