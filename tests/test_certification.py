from __future__ import annotations

import json

import pytest

from cliffquant.certification import CertificationConfig, run_certification


def test_small_campaign_is_deterministic_and_resumable(tmp_path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    config = CertificationConfig(cases=7, seed=2339, workers=1, batch_size=3)

    first = run_certification(first_path, config, progress_every=3)
    second = run_certification(second_path, config, progress_every=3)
    resumed = run_certification(first_path, config, progress_every=3)

    assert first["status"] == second["status"] == resumed["status"] == "passed"
    assert first["mismatch_count"] == second["mismatch_count"] == 0
    assert first["protocol_gate_passed"] is False
    assert first["aggregate_fingerprint_sha256"] == second["aggregate_fingerprint_sha256"]
    assert resumed["aggregate_fingerprint_sha256"] == first["aggregate_fingerprint_sha256"]
    assert json.loads(first_path.read_text(encoding="utf-8"))["completed_cases"] == 7


@pytest.mark.parametrize(
    "config",
    [
        CertificationConfig(cases=0),
        CertificationConfig(seed=0),
        CertificationConfig(workers=0),
        CertificationConfig(batch_size=0),
    ],
)
def test_invalid_config_is_rejected(tmp_path, config) -> None:
    with pytest.raises(ValueError):
        run_certification(tmp_path / "certificate.json", config)


def test_existing_configuration_must_match(tmp_path) -> None:
    path = tmp_path / "certificate.json"
    run_certification(
        path,
        CertificationConfig(cases=2, seed=1, workers=1, batch_size=1),
        progress_every=1,
    )

    with pytest.raises(ValueError, match="configuration"):
        run_certification(
            path,
            CertificationConfig(cases=2, seed=2, workers=1, batch_size=1),
            progress_every=1,
        )
