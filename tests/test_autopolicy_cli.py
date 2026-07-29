from __future__ import annotations

import json
from pathlib import Path

from cliffquant.autopolicy_artifacts import CANDIDATE_SCHEMA, write_candidate_bundle
from cliffquant.autopolicy_cli import main as autopolicy_main
from cliffquant.cli import main as cli_main


def _bundle() -> dict[str, object]:
    return {
        "schema": CANDIDATE_SCHEMA,
        "status": "complete",
        "profile": "cli-test",
        "scope": {
            "budget_scope": "logical-target-weight-and-scale-payload",
            "checkpoint_export": False,
        },
        "source": {"kind": "synthetic"},
        "environments": ["general", "code"],
        "distortion_metric": {
            "name": "synthetic",
            "additive_across_units": True,
            "downstream_metric": False,
        },
        "candidate_definitions": {"dense16": {}, "w4": {}},
        "units": [
            {
                "unit_id": "layer.0",
                "weight_key": "layer.0.weight",
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
                        "metadata": {"bits": 16},
                    },
                    {
                        "candidate_id": "w4",
                        "logical_bytes": 528,
                        "groupwise_max_distortion": 2.0,
                        "per_environment_distortion": {
                            "general": 2.0,
                            "code": 1.0,
                        },
                        "metadata": {"bits": 4},
                    },
                ],
                "evidence": {"kind": "synthetic"},
            }
        ],
        "totals": {"dense16": 2048, "w4": 528},
        "builder": {"name": "cli-test"},
    }


def test_top_level_and_autopolicy_help(capsys) -> None:
    try:
        cli_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    top_help = capsys.readouterr().out
    assert "autopolicy" in top_help

    try:
        cli_main(["autopolicy", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    autopolicy_help = capsys.readouterr().out
    assert "build-experiment-001" in autopolicy_help
    assert "aggregate-blocks" in autopolicy_help
    assert "does not export a runnable mixed-bit checkpoint" in autopolicy_help


def test_solve_and_verify_cli_round_trip(tmp_path: Path, capsys) -> None:
    candidates = tmp_path / "candidates.json"
    plan = tmp_path / "plan.json"
    write_candidate_bundle(candidates, _bundle())

    exit_code = autopolicy_main(
        [
            "solve",
            "--candidates",
            str(candidates),
            "--budget-bytes",
            "528",
            "--output",
            str(plan),
        ]
    )
    assert exit_code == 0
    solve_output = json.loads(capsys.readouterr().out)
    assert solve_output["checkpoint_export"] is False
    assert solve_output["optimization"]["used_bytes"] == 528
    assert solve_output["optimization"]["decisions"][0]["candidate_id"] == "w4"

    exit_code = autopolicy_main(
        [
            "verify",
            "--candidates",
            str(candidates),
            "--plan",
            str(plan),
        ]
    )
    assert exit_code == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output["status"] == "pass"
    assert verify_output["plan"]["status"] == "pass"


def test_cli_reports_infeasible_budget_without_writing_plan(
    tmp_path: Path,
    capsys,
) -> None:
    candidates = tmp_path / "candidates.json"
    plan = tmp_path / "plan.json"
    write_candidate_bundle(candidates, _bundle())

    exit_code = autopolicy_main(
        [
            "solve",
            "--candidates",
            str(candidates),
            "--budget-bytes",
            "527",
            "--output",
            str(plan),
        ]
    )

    assert exit_code == 2
    assert "infeasible" in capsys.readouterr().err
    assert not plan.exists()


def test_cli_refuses_to_overwrite_plan(tmp_path: Path, capsys) -> None:
    candidates = tmp_path / "candidates.json"
    plan = tmp_path / "plan.json"
    write_candidate_bundle(candidates, _bundle())
    arguments = [
        "solve",
        "--candidates",
        str(candidates),
        "--budget-bytes",
        "528",
        "--output",
        str(plan),
    ]

    assert autopolicy_main(arguments) == 0
    capsys.readouterr()
    assert autopolicy_main(arguments) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_aggregate_blocks_cli_writes_content_bound_candidate_view(
    tmp_path: Path,
    capsys,
) -> None:
    candidates = tmp_path / "candidates.json"
    grouped = tmp_path / "grouped.json"
    payload = _bundle()
    payload["profile"] = "experiment-001"
    payload["units"][0]["unit_id"] = "model.language_model.layers.0.mlp.up_proj"  # type: ignore[index]
    payload["units"][0]["weight_key"] = (  # type: ignore[index]
        "model.language_model.layers.0.mlp.up_proj.weight"
    )
    write_candidate_bundle(candidates, payload)

    exit_code = autopolicy_main(
        [
            "aggregate-blocks",
            "--candidates",
            str(candidates),
            "--output",
            str(grouped),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["profile"] == "experiment-001-transformer-blocks"
    assert output["allocation_unit_count"] == 1
    assert grouped.exists()
