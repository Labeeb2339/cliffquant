from __future__ import annotations

import json
from pathlib import Path

import pytest

from cliffquant.calibration_cli import _frozen_tokenizer_files, main


def test_cli_small_dry_run_writes_both_disjoint_phases(tmp_path: Path) -> None:
    assert main(["--dry-run", "--output-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "collection-summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert set(summary["phases"]) == {"calibration", "heldout"}
    assert (tmp_path / "calibration.manifest.json").is_file()
    assert (tmp_path / "heldout.manifest.json").is_file()


def test_real_cli_refuses_a_non_protocol_record_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="freezes --max-records at 10000"):
        main(["--output-dir", str(tmp_path), "--max-records", "99"])


def test_tokenizer_fingerprint_ignores_unrelated_snapshot_files(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"unrelated")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert _frozen_tokenizer_files(tmp_path) == {"tokenizer.json": tokenizer}
