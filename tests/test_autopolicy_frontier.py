from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from cliffquant.autopolicy_artifacts import CANDIDATE_SCHEMA, write_candidate_bundle
from scripts.generate_autopolicy_frontier import (
    build_provenance_manifest,
    generate,
    validate_frontier,
    verify_provenance_manifest,
)


def _grouped_bundle() -> dict[str, object]:
    units = []
    for index in range(2):
        units.append(
            {
                "unit_id": f"model.language_model.layers.{index}",
                "weight_key": f"aggregate:model.language_model.layers.{index}",
                "shape": [1, 400],
                "group_count": 4,
                "candidates": [
                    {
                        "candidate_id": "dense16",
                        "logical_bytes": 200,
                        "groupwise_max_distortion": 0.0,
                        "per_environment_distortion": {"general": 0.0, "code": 0.0},
                        "metadata": {"bits": 16},
                    },
                    {
                        "candidate_id": "w4_cliffquant",
                        "logical_bytes": 50,
                        "groupwise_max_distortion": 5.0 + index,
                        "per_environment_distortion": {
                            "general": 5.0 + index,
                            "code": 4.0 + index,
                        },
                        "metadata": {"bits": 4},
                    },
                    {
                        "candidate_id": "w8_absmax",
                        "logical_bytes": 100,
                        "groupwise_max_distortion": 1.0 + index,
                        "per_environment_distortion": {
                            "general": 1.0 + index,
                            "code": 0.8 + index,
                        },
                        "metadata": {"bits": 8},
                    },
                ],
                "evidence": {"aggregation": "transformer-block"},
            }
        )
    return {
        "schema": CANDIDATE_SCHEMA,
        "status": "complete",
        "profile": "experiment-001-transformer-blocks",
        "scope": {
            "weight_count": 800,
            "budget_scope": "logical-target-weight-and-scale-payload",
            "checkpoint_export": False,
            "allocation_granularity": "transformer-block",
            "allocation_unit_count": 2,
        },
        "source": {"kind": "synthetic"},
        "environments": ["general", "code"],
        "distortion_metric": {
            "name": "synthetic",
            "additive_across_units": True,
            "downstream_metric": False,
        },
        "candidate_definitions": {
            "dense16": {},
            "w4_cliffquant": {},
            "w8_absmax": {},
        },
        "units": units,
        "totals": {
            "dense16": 400,
            "w4_cliffquant": 100,
            "w8_absmax": 200,
        },
        "builder": {"name": "unit-test"},
    }


def test_frontier_generation_is_deterministic_and_self_consistent(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.json"
    write_candidate_bundle(candidate_path, _grouped_bundle())

    first = generate(candidate_path, max_frontier_states=1_000)
    second = generate(candidate_path, max_frontier_states=1_000)

    assert first == second
    validate_frontier(first)
    assert first["allocation_granularity"] == "transformer-block"
    assert first["points"][0]["selection_counts"]["w4_cliffquant"] == 2
    assert first["points"][-1]["selection_counts"]["dense16"] == 2


def test_frontier_validator_rejects_selection_count_drift(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.json"
    write_candidate_bundle(candidate_path, _grouped_bundle())
    payload = generate(candidate_path, max_frontier_states=1_000)
    tampered = deepcopy(payload)
    tampered["points"][0]["selection_counts"]["w4_cliffquant"] = 1

    with pytest.raises(ValueError, match="selection counts"):
        validate_frontier(tampered)


def test_figure_provenance_binds_input_frontier_figure_and_generator(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.json"
    frontier_path = tmp_path / "frontier.json"
    figure_path = tmp_path / "frontier.png"
    manifest_path = tmp_path / "frontier.manifest.json"
    write_candidate_bundle(candidate_path, _grouped_bundle())
    payload = generate(candidate_path, max_frontier_states=1_000)
    frontier_path.write_text("frontier\n", encoding="utf-8")
    figure_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + (640).to_bytes(4, "big")
        + (480).to_bytes(4, "big")
    )
    manifest = build_provenance_manifest(
        candidate_path=candidate_path,
        frontier_path=frontier_path,
        figure_path=figure_path,
        manifest_path=manifest_path,
        candidate_fingerprint_sha256=payload["candidate_fingerprint_sha256"],
    )
    from cliffquant.provenance import canonical_json_bytes

    manifest_path.write_bytes(canonical_json_bytes(manifest))

    assert verify_provenance_manifest(manifest_path) == manifest

    frontier_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        verify_provenance_manifest(manifest_path)


def test_checked_in_figure_provenance_revalidates() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "figures" / "autopolicy-logical-byte-frontier.manifest.json"
    if not manifest_path.is_file():
        pytest.skip("checked-in AutoPolicy figure provenance is not present")

    payload = verify_provenance_manifest(manifest_path)
    assert payload["outputs"]["figure"]["width_px"] == 2052
