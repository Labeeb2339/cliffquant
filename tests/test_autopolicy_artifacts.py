from __future__ import annotations

import json
from pathlib import Path

import pytest

from cliffquant.autopolicy_artifacts import (
    CANDIDATE_SCHEMA,
    ROBUST_OBJECTIVE,
    TRANSFORMER_BLOCK_PROFILE,
    aggregate_experiment001_transformer_blocks,
    bind_fingerprint,
    build_plan_payload,
    load_candidate_bundle,
    load_plan,
    solve_candidate_bundle,
    verify_plan,
    write_candidate_bundle,
    write_plan,
)
from cliffquant.provenance import canonical_json_bytes


def _sample_bundle() -> dict[str, object]:
    units = []
    for index in range(2):
        units.append(
            {
                "unit_id": f"model.language_model.layers.{index}.mlp.up_proj",
                "weight_key": f"model.language_model.layers.{index}.mlp.up_proj.weight",
                "shape": [8, 128],
                "group_count": 8,
                "candidates": [
                    {
                        "candidate_id": "dense16",
                        "logical_bytes": 2048,
                        "groupwise_max_distortion": 0.0,
                        "per_environment_distortion": {
                            "general": 0.0,
                            "code": 0.0,
                        },
                        "metadata": {"bits": 16, "group_size": None},
                    },
                    {
                        "candidate_id": "w4_cliffquant",
                        "logical_bytes": 528,
                        "groupwise_max_distortion": 5.0 + index,
                        "per_environment_distortion": {
                            "general": 4.0 + index,
                            "code": 3.0 + index,
                        },
                        "metadata": {"bits": 4, "group_size": 128},
                    },
                    {
                        "candidate_id": "w8_absmax",
                        "logical_bytes": 1040,
                        "groupwise_max_distortion": 1.0 + index,
                        "per_environment_distortion": {
                            "general": 0.8 + index,
                            "code": 0.7 + index,
                        },
                        "metadata": {"bits": 8, "group_size": 128},
                    },
                ],
                "evidence": {"kind": "unit-test"},
            }
        )
    return {
        "schema": CANDIDATE_SCHEMA,
        "status": "complete",
        "profile": "unit-test",
        "scope": {
            "budget_scope": "logical-target-weight-and-scale-payload",
            "checkpoint_export": False,
        },
        "source": {"kind": "synthetic"},
        "environments": ["general", "code"],
        "distortion_metric": {
            "name": "unit-test",
            "additive_across_units": True,
            "downstream_metric": False,
        },
        "candidate_definitions": {"dense16": {}, "w4_cliffquant": {}, "w8_absmax": {}},
        "units": units,
        "totals": {
            "dense16": 4096,
            "w4_cliffquant": 1056,
            "w8_absmax": 2080,
        },
        "builder": {"name": "unit-test"},
    }


def test_candidate_and_plan_round_trip_with_exact_recomputation(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.json"
    candidate_bundle = write_candidate_bundle(candidate_path, _sample_bundle())

    loaded = load_candidate_bundle(candidate_path)
    assert loaded == candidate_bundle
    optimization = solve_candidate_bundle(
        loaded,
        budget_bytes=2080,
        objective=ROBUST_OBJECTIVE,
    )
    plan = build_plan_payload(
        candidate_path,
        loaded,
        objective=ROBUST_OBJECTIVE,
        optimization=optimization,
    )
    plan_path = tmp_path / "plan.json"
    write_plan(plan_path, plan)

    assert load_plan(plan_path) == plan
    verification = verify_plan(candidate_path, plan_path)
    assert verification["status"] == "pass"
    assert verification["objective"] == ROBUST_OBJECTIVE
    assert verification["budget_bytes"] == 2080


def test_writers_refuse_to_overwrite(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.json"
    write_candidate_bundle(candidate_path, _sample_bundle())

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_candidate_bundle(candidate_path, _sample_bundle())


def test_candidate_loader_rejects_duplicate_keys_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_candidate_bundle(duplicate)

    noncanonical = tmp_path / "noncanonical.json"
    bound = bind_fingerprint(_sample_bundle())
    noncanonical.write_text(json.dumps(bound, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not in canonical JSON form"):
        load_candidate_bundle(noncanonical)


def test_candidate_fingerprint_and_plan_binding_detect_tampering(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.json"
    bundle = write_candidate_bundle(candidate_path, _sample_bundle())
    tampered = dict(bundle)
    tampered["profile"] = "tampered"
    candidate_path.write_bytes(canonical_json_bytes(tampered))

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_candidate_bundle(candidate_path)


def test_plan_verifier_rejects_changed_candidate_file(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.json"
    bundle = write_candidate_bundle(candidate_path, _sample_bundle())
    optimization = solve_candidate_bundle(bundle, budget_bytes=2080)
    plan_path = tmp_path / "plan.json"
    write_plan(
        plan_path,
        build_plan_payload(
            candidate_path,
            bundle,
            objective=ROBUST_OBJECTIVE,
            optimization=optimization,
        ),
    )

    replacement = _sample_bundle()
    replacement["profile"] = "replacement"
    candidate_path.unlink()
    write_candidate_bundle(candidate_path, replacement)

    with pytest.raises(ValueError, match="candidate file hash mismatch"):
        verify_plan(candidate_path, plan_path)


def test_candidate_environment_inventory_is_part_of_contract(tmp_path: Path) -> None:
    payload = _sample_bundle()
    first_candidate = payload["units"][0]["candidates"][0]  # type: ignore[index]
    first_candidate["per_environment_distortion"] = {  # type: ignore[index]
        "general": 0.0,
        "unexpected": 0.0,
    }

    with pytest.raises(ValueError, match="environments must exactly match"):
        write_candidate_bundle(tmp_path / "bad.json", payload)


def test_candidate_bundle_rejects_duplicate_candidates_and_nonadditive_metrics(
    tmp_path: Path,
) -> None:
    duplicate = _sample_bundle()
    duplicate["units"][0]["candidates"].append(  # type: ignore[index,union-attr]
        dict(duplicate["units"][0]["candidates"][0])  # type: ignore[index,arg-type]
    )
    with pytest.raises(ValueError, match="candidate inventory disagrees"):
        write_candidate_bundle(tmp_path / "duplicate.json", duplicate)

    nonadditive = _sample_bundle()
    nonadditive["distortion_metric"]["additive_across_units"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="additive_across_units must be true"):
        write_candidate_bundle(tmp_path / "nonadditive.json", nonadditive)


def test_candidate_bundle_totals_are_strict_positive_integers(tmp_path: Path) -> None:
    payload = _sample_bundle()
    payload["totals"]["dense16"] = 4096.0  # type: ignore[index]

    with pytest.raises(ValueError, match="must be a positive integer"):
        write_candidate_bundle(tmp_path / "float-total.json", payload)


def test_candidate_bundle_rejects_export_claim_and_byte_component_drift(
    tmp_path: Path,
) -> None:
    exported = _sample_bundle()
    exported["scope"]["checkpoint_export"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="cannot claim checkpoint export"):
        write_candidate_bundle(tmp_path / "exported.json", exported)

    drifted = _sample_bundle()
    candidate = drifted["units"][0]["candidates"][0]  # type: ignore[index]
    candidate["metadata"].update(  # type: ignore[index,union-attr]
        {
            "logical_code_bytes": 2047,
            "logical_scale_bytes": 0,
        }
    )
    with pytest.raises(ValueError, match="logical byte components do not sum"):
        write_candidate_bundle(tmp_path / "byte-drift.json", drifted)


def test_transformer_block_aggregation_is_content_bound_and_conserves_totals(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidates.json"
    source = _sample_bundle()
    source["profile"] = "experiment-001"
    bundle = write_candidate_bundle(candidate_path, source)

    aggregated = aggregate_experiment001_transformer_blocks(candidate_path, bundle)

    assert aggregated["profile"] == TRANSFORMER_BLOCK_PROFILE
    assert aggregated["scope"]["allocation_granularity"] == "transformer-block"
    assert aggregated["scope"]["allocation_unit_count"] == 2
    assert aggregated["totals"] == bundle["totals"]
    assert (
        aggregated["source"]["parent_candidate_bundle"]["fingerprint_sha256"]
        == (bundle["fingerprint_sha256"])
    )
    assert [unit["unit_id"] for unit in aggregated["units"]] == [
        "model.language_model.layers.0",
        "model.language_model.layers.1",
    ]
