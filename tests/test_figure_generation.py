from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.generate_figures import (
    FIGURE_MANIFEST_SCHEMA,
    generate_figures,
    verify_and_load_summary,
)


def _write_fixture(directory: Path) -> None:
    summary = {
        "comparisons": {
            "absmax": {
                "baseline_minus_cliff_module_macro": 0.25,
                "bootstrap_95_ci": [0.2, 0.3],
                "bootstrap_method": {"replicates": 100},
                "gate_passed": True,
                "paired_mean_improvement": 0.25,
                "wins_ties_losses": {"losses": 0, "ties": 1, "wins": 3},
            },
            "pooled_wmse": {
                "baseline_minus_cliff_module_macro": 0.05,
                "bootstrap_95_ci": [0.04, 0.06],
                "bootstrap_method": {"replicates": 100},
                "gate_passed": True,
                "paired_mean_improvement": 0.05,
                "wins_ties_losses": {"losses": 1, "ties": 1, "wins": 2},
            },
        },
        "environments": ["a", "b"],
        "group_count": 4,
        "module_count": 2,
        "policies": {
            "absmax": {"module_macro_heldout_max_loss": 1.0},
            "cliffquant_minimax": {"module_macro_heldout_max_loss": 0.75},
            "pooled_wmse": {"module_macro_heldout_max_loss": 0.8},
        },
        "proxy_gate_passed": True,
        "schema": "cliffquant.experiment-001.proxy-summary.v1",
    }
    payload = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    summary_path = directory / "summary.json"
    summary_path.write_bytes(payload)
    checksums = {
        "summary": {
            "file": "summary.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    }
    (directory / "SHA256SUMS.json").write_text(
        json.dumps(checksums, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_checksum_verification_precedes_summary_parsing(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    summary, provenance = verify_and_load_summary(tmp_path)
    assert summary["group_count"] == 4
    assert provenance["verified_files"]["summary"]["file"] == "summary.json"

    (tmp_path / "summary.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum verification failed"):
        verify_and_load_summary(tmp_path)


def test_generation_is_byte_reproducible_and_has_no_timestamp(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    input_dir.mkdir()
    _write_fixture(input_dir)

    first = generate_figures(input_dir, first_dir)
    second = generate_figures(input_dir, second_dir)

    assert first["schema"] == FIGURE_MANIFEST_SCHEMA
    assert first == second
    assert "timestamp" not in (first_dir / "figure-manifest.json").read_text(encoding="utf-8")
    for figure in first["figures"].values():
        first_bytes = (first_dir / figure["file"]).read_bytes()
        second_bytes = (second_dir / figure["file"]).read_bytes()
        assert first_bytes == second_bytes
        assert first_bytes.startswith(b"\x89PNG\r\n\x1a\n")
