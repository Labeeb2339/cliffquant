from __future__ import annotations

import json
from pathlib import Path

from cliffquant.calibration_cli import main


def test_cli_small_dry_run_writes_both_disjoint_phases(tmp_path: Path) -> None:
    assert main(["--dry-run", "--output-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "collection-summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert set(summary["phases"]) == {"calibration", "heldout"}
    assert (tmp_path / "calibration.manifest.json").is_file()
    assert (tmp_path / "heldout.manifest.json").is_file()
