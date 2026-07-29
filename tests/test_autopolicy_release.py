from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from cliffquant.autopolicy_artifacts import bind_fingerprint, load_candidate_bundle
from scripts.validate_autopolicy_release import (
    _validate_frontier,
    _validate_grouped_candidates,
    check_manifest,
)

RELEASE_ROOT = Path(__file__).resolve().parents[1] / "research" / "results" / "autopolicy-001"

pytestmark = pytest.mark.skipif(
    not (RELEASE_ROOT / "manifest.json").is_file(),
    reason="portable AutoPolicy evidence is not included in this source distribution",
)


def test_checked_in_autopolicy_release_revalidates() -> None:
    manifest = check_manifest(RELEASE_ROOT)

    assert manifest["status"] == "pass"
    assert manifest["candidate"]["module_count"] == 150
    assert manifest["candidate"]["evidence_verification"]["w8_archives_checked"] == 150
    assert manifest["transformer_block_view"]["allocation_unit_count"] == 24
    assert manifest["frontier"]["point_count"] == 10
    assert manifest["representative_plan"]["status"] == "pass"


def test_grouped_candidates_fail_if_not_exact_matrix_aggregation() -> None:
    matrix_path = RELEASE_ROOT / "candidates.json"
    matrix = load_candidate_bundle(matrix_path)
    grouped = load_candidate_bundle(RELEASE_ROOT / "candidates-transformer-blocks.json")
    tampered = deepcopy(grouped)
    candidate = tampered["units"][0]["candidates"][0]
    environment = max(
        candidate["per_environment_distortion"],
        key=candidate["per_environment_distortion"].get,
    )
    candidate["per_environment_distortion"][environment] += 1.0
    candidate["groupwise_max_distortion"] = max(candidate["per_environment_distortion"].values())
    tampered = bind_fingerprint(tampered)

    with pytest.raises(ValueError, match="exact aggregation"):
        _validate_grouped_candidates(matrix_path, matrix, tampered)


def test_frontier_fails_if_any_published_point_is_not_exact() -> None:
    grouped_path = RELEASE_ROOT / "candidates-transformer-blocks.json"
    grouped = load_candidate_bundle(grouped_path)
    frontier = json.loads((RELEASE_ROOT / "frontier.json").read_text(encoding="utf-8"))
    tampered = deepcopy(frontier)
    tampered["points"][1]["worst_environment_distortion"] += 1.0

    with pytest.raises(ValueError, match="exact recomputation"):
        _validate_frontier(tampered, grouped_path=grouped_path, grouped=grouped)
