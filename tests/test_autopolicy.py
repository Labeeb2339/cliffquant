from __future__ import annotations

import itertools
import json
import random

import pytest

from cliffquant.autopolicy import (
    FrontierLimitError,
    InfeasiblePolicyError,
    PolicyCandidate,
    QuantizableUnit,
    RobustPolicyCandidate,
    RobustQuantizableUnit,
    solve_autopolicy,
    solve_robust_autopolicy,
)


def _candidate(candidate_id: str, byte_size: int, distortion: float) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=candidate_id,
        byte_size=byte_size,
        distortion=distortion,
    )


def _exhaustive_oracle(
    units: list[QuantizableUnit],
    budget_bytes: int,
) -> tuple[float, int, tuple[str, ...]]:
    canonical_units = sorted(units, key=lambda unit: unit.unit_id)
    feasible: list[tuple[float, int, tuple[str, ...]]] = []
    for combination in itertools.product(*(unit.candidates for unit in canonical_units)):
        used_bytes = sum(candidate.byte_size for candidate in combination)
        if used_bytes > budget_bytes:
            continue
        aggregate_distortion = 0.0
        for candidate in combination:
            aggregate_distortion += candidate.distortion
        feasible.append(
            (
                aggregate_distortion,
                used_bytes,
                tuple(candidate.candidate_id for candidate in combination),
            )
        )
    if not feasible:
        raise InfeasiblePolicyError
    return min(feasible)


def test_known_budget_tradeoff_selects_exact_optimum() -> None:
    units = [
        QuantizableUnit(
            "block.0",
            (
                _candidate("int4", 4, 8.0),
                _candidate("int8", 8, 1.0),
                _candidate("fp16", 16, 0.0),
            ),
        ),
        QuantizableUnit(
            "block.1",
            (
                _candidate("int4", 4, 5.0),
                _candidate("int8", 8, 2.0),
                _candidate("fp16", 16, 0.0),
            ),
        ),
    ]

    result = solve_autopolicy(units, budget_bytes=16)

    assert [(item.unit_id, item.candidate_id) for item in result.decisions] == [
        ("block.0", "int8"),
        ("block.1", "int8"),
    ]
    assert result.used_bytes == 16
    assert result.remaining_bytes == 0
    assert result.aggregate_distortion == 3.0


@pytest.mark.parametrize("seed", range(30))
def test_random_small_problems_match_exhaustive_oracle(seed: int) -> None:
    rng = random.Random(seed)
    units: list[QuantizableUnit] = []
    for unit_index in range(rng.randint(1, 4)):
        candidates = tuple(
            _candidate(
                f"choice-{candidate_index}",
                rng.randint(1, 12),
                rng.randint(0, 100) / 10.0,
            )
            for candidate_index in range(rng.randint(2, 4))
        )
        units.append(QuantizableUnit(f"unit-{unit_index}", candidates))

    minimum = sum(min(item.byte_size for item in unit.candidates) for unit in units)
    maximum = sum(max(item.byte_size for item in unit.candidates) for unit in units)
    budget = rng.randint(minimum, maximum)
    oracle = _exhaustive_oracle(units, budget)

    result = solve_autopolicy(reversed(units), budget_bytes=budget)
    actual = (
        result.aggregate_distortion,
        result.used_bytes,
        tuple(item.candidate_id for item in result.decisions),
    )

    assert actual == oracle
    assert result.used_bytes <= budget
    assert len(result.decisions) == len(units)


def test_input_order_does_not_change_plan_or_serialization() -> None:
    first = QuantizableUnit(
        "a",
        (
            _candidate("z", 3, 1.0),
            _candidate("a", 3, 1.0),
        ),
    )
    second = QuantizableUnit(
        "b",
        (
            _candidate("high", 5, 0.0),
            _candidate("low", 2, 2.0),
        ),
    )

    forward = solve_autopolicy([first, second], budget_bytes=8)
    reversed_plan = solve_autopolicy(
        [
            QuantizableUnit("b", tuple(reversed(second.candidates))),
            QuantizableUnit("a", tuple(reversed(first.candidates))),
        ],
        budget_bytes=8,
    )

    assert forward == reversed_plan
    assert forward.to_json() == reversed_plan.to_json()


def test_ties_prefer_lower_bytes_then_lexicographic_candidate_ids() -> None:
    lower_bytes = QuantizableUnit(
        "unit",
        (
            _candidate("large", 9, 1.0),
            _candidate("small", 4, 1.0),
        ),
    )
    same_bytes = QuantizableUnit(
        "other",
        (
            _candidate("z-choice", 3, 2.0),
            _candidate("a-choice", 3, 2.0),
        ),
    )

    result = solve_autopolicy([lower_bytes, same_bytes], budget_bytes=20)

    assert [(item.unit_id, item.candidate_id) for item in result.decisions] == [
        ("other", "a-choice"),
        ("unit", "small"),
    ]
    assert result.used_bytes == 7


def test_relaxing_budget_never_worsens_exact_objective() -> None:
    units = [
        QuantizableUnit(
            f"unit-{index}",
            (
                _candidate("int4", 4, float(10 - index)),
                _candidate("int8", 8, float(3 - index / 10)),
                _candidate("fp16", 16, 0.0),
            ),
        )
        for index in range(3)
    ]

    objectives = [
        solve_autopolicy(units, budget_bytes=budget).aggregate_distortion
        for budget in range(12, 49)
    ]

    assert all(later <= earlier for earlier, later in itertools.pairwise(objectives))


def test_dominated_candidate_does_not_change_plan() -> None:
    baseline_unit = QuantizableUnit(
        "unit",
        (
            _candidate("compact", 4, 2.0),
            _candidate("accurate", 8, 0.5),
        ),
    )
    with_dominated = QuantizableUnit(
        "unit",
        (
            *baseline_unit.candidates,
            _candidate("dominated", 9, 3.0),
        ),
    )

    baseline = solve_autopolicy([baseline_unit], budget_bytes=9)
    augmented = solve_autopolicy([with_dominated], budget_bytes=9)

    assert baseline.aggregate_distortion == augmented.aggregate_distortion
    assert baseline.used_bytes == augmented.used_bytes
    assert baseline.decisions[0].candidate_id == augmented.decisions[0].candidate_id


def test_result_is_json_serializable_and_self_describing() -> None:
    unit = QuantizableUnit("layer.0", (_candidate("w4", 128, 0.25),))

    result = solve_autopolicy([unit], budget_bytes=256)
    payload = result.to_dict()
    round_tripped = json.loads(json.dumps(payload, allow_nan=False))

    assert round_tripped["schema_version"] == 1
    assert round_tripped["algorithm"] == "exact_pareto_frontier"
    assert round_tripped["objective"] == "minimize_sum_distortion"
    assert round_tripped["budget_bytes"] == 256
    assert round_tripped["used_bytes"] == 128
    assert round_tripped["remaining_bytes"] == 128
    assert round_tripped["decisions"] == [
        {
            "unit_id": "layer.0",
            "candidate_id": "w4",
            "byte_size": 128,
            "distortion": 0.25,
        }
    ]
    assert json.loads(result.to_json()) == round_tripped


def test_infeasible_budget_fails_closed_with_minimum_requirement() -> None:
    units = [
        QuantizableUnit("a", (_candidate("only", 5, 0.0),)),
        QuantizableUnit("b", (_candidate("only", 7, 0.0),)),
    ]

    with pytest.raises(
        InfeasiblePolicyError,
        match=r"budget_bytes=11 is infeasible; at least 12 bytes are required",
    ):
        solve_autopolicy(units, budget_bytes=11)


@pytest.mark.parametrize("invalid_budget", [True, 1.5, "10", None])
def test_invalid_budget_type_is_rejected(invalid_budget: object) -> None:
    unit = QuantizableUnit("unit", (_candidate("only", 1, 0.0),))

    with pytest.raises(TypeError, match="budget_bytes must be an integer"):
        solve_autopolicy([unit], budget_bytes=invalid_budget)  # type: ignore[arg-type]


def test_negative_budget_and_empty_problem_are_rejected() -> None:
    unit = QuantizableUnit("unit", (_candidate("only", 1, 0.0),))

    with pytest.raises(ValueError, match="budget_bytes must be non-negative"):
        solve_autopolicy([unit], budget_bytes=-1)
    with pytest.raises(ValueError, match="units must not be empty"):
        solve_autopolicy([], budget_bytes=1)


def test_duplicate_unit_and_candidate_identifiers_are_rejected() -> None:
    candidate = _candidate("same", 1, 0.0)

    with pytest.raises(ValueError, match="candidate_id values must be unique"):
        QuantizableUnit("unit", (candidate, candidate))

    units = [
        QuantizableUnit("same", (_candidate("a", 1, 0.0),)),
        QuantizableUnit("same", (_candidate("b", 1, 0.0),)),
    ]
    with pytest.raises(ValueError, match="unit_id values must be unique"):
        solve_autopolicy(units, budget_bytes=2)


@pytest.mark.parametrize(
    ("candidate_id", "byte_size", "distortion", "exception"),
    [
        ("", 1, 0.0, ValueError),
        (" padded", 1, 0.0, ValueError),
        ("valid", True, 0.0, TypeError),
        ("valid", 0, 0.0, ValueError),
        ("valid", -1, 0.0, ValueError),
        ("valid", 1, True, TypeError),
        ("valid", 1, -0.1, ValueError),
        ("valid", 1, float("nan"), ValueError),
        ("valid", 1, float("inf"), ValueError),
    ],
)
def test_invalid_candidates_are_rejected(
    candidate_id: str,
    byte_size: int,
    distortion: float,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        PolicyCandidate(candidate_id, byte_size, distortion)


def test_invalid_units_are_rejected() -> None:
    candidate = _candidate("only", 1, 0.0)

    with pytest.raises(ValueError, match="unit_id must not be empty"):
        QuantizableUnit("", (candidate,))
    with pytest.raises(ValueError, match="candidates must not be empty"):
        QuantizableUnit("unit", ())
    with pytest.raises(TypeError, match="candidates must contain only"):
        QuantizableUnit("unit", (candidate, object()))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="units must contain only"):
        solve_autopolicy([QuantizableUnit("unit", (candidate,)), object()], budget_bytes=2)


def _robust_candidate(
    candidate_id: str,
    byte_size: int,
    *distortions: float,
) -> RobustPolicyCandidate:
    return RobustPolicyCandidate(candidate_id, byte_size, distortions)


def _robust_exhaustive_oracle(
    units: list[RobustQuantizableUnit],
    environments: tuple[str, ...],
    budget_bytes: int,
) -> tuple[float, float, int, tuple[str, ...], tuple[float, ...]]:
    canonical_units = sorted(units, key=lambda unit: unit.unit_id)
    feasible = []
    for combination in itertools.product(*(unit.candidates for unit in canonical_units)):
        used_bytes = sum(candidate.byte_size for candidate in combination)
        if used_bytes > budget_bytes:
            continue
        totals = tuple(
            sum(candidate.per_environment_distortion[index] for candidate in combination)
            for index in range(len(environments))
        )
        feasible.append(
            (
                max(totals),
                sum(totals),
                used_bytes,
                tuple(candidate.candidate_id for candidate in combination),
                totals,
            )
        )
    if not feasible:
        raise InfeasiblePolicyError
    return min(feasible)


@pytest.mark.parametrize("seed", range(20))
def test_robust_random_problems_match_exhaustive_oracle(seed: int) -> None:
    rng = random.Random(10_000 + seed)
    environments = ("general", "code", "math")
    units: list[RobustQuantizableUnit] = []
    for unit_index in range(rng.randint(1, 4)):
        units.append(
            RobustQuantizableUnit(
                f"unit-{unit_index}",
                tuple(
                    _robust_candidate(
                        f"choice-{candidate_index}",
                        rng.randint(1, 10),
                        *(rng.randint(0, 100) / 10.0 for _ in environments),
                    )
                    for candidate_index in range(rng.randint(2, 4))
                ),
            )
        )
    minimum = sum(min(item.byte_size for item in unit.candidates) for unit in units)
    maximum = sum(max(item.byte_size for item in unit.candidates) for unit in units)
    budget = rng.randint(minimum, maximum)
    oracle = _robust_exhaustive_oracle(units, environments, budget)

    result = solve_robust_autopolicy(
        reversed(units),
        environments=environments,
        budget_bytes=budget,
    )
    actual = (
        result.worst_environment_distortion,
        sum(result.per_environment_distortion),
        result.used_bytes,
        tuple(item.candidate_id for item in result.decisions),
        result.per_environment_distortion,
    )

    assert actual == oracle
    assert json.loads(result.to_json()) == result.to_dict()


def test_robust_minimax_can_differ_from_sum_of_per_unit_maxima() -> None:
    units = [
        RobustQuantizableUnit(
            "a",
            (
                _robust_candidate("balanced", 4, 4.0, 4.0),
                _robust_candidate("general-heavy", 4, 7.0, 0.0),
            ),
        ),
        RobustQuantizableUnit(
            "b",
            (
                _robust_candidate("balanced", 4, 4.0, 4.0),
                _robust_candidate("code-heavy", 4, 0.0, 7.0),
            ),
        ),
    ]

    result = solve_robust_autopolicy(
        units,
        environments=("general", "code"),
        budget_bytes=8,
    )

    assert [decision.candidate_id for decision in result.decisions] == [
        "general-heavy",
        "code-heavy",
    ]
    assert result.per_environment_distortion == (7.0, 7.0)
    assert result.worst_environment_distortion == 7.0


def test_minimum_budget_prunes_incomplete_high_byte_prefixes() -> None:
    units = [
        RobustQuantizableUnit(
            f"unit-{index:03d}",
            (
                _robust_candidate("minimum", 1, 1.0, 1.0),
                _robust_candidate("larger", 2, 0.0, 0.0),
            ),
        )
        for index in range(150)
    ]

    result = solve_robust_autopolicy(
        units,
        environments=("general", "code"),
        budget_bytes=150,
        max_frontier_states=1,
    )

    assert result.used_bytes == 150
    assert result.peak_frontier_states == 1
    assert result.transitions_evaluated == 150
    assert {decision.candidate_id for decision in result.decisions} == {"minimum"}


def test_search_transition_limit_fails_closed() -> None:
    units = [
        RobustQuantizableUnit(
            f"unit-{index}",
            (
                _robust_candidate("a", 1, 0.0, 1.0),
                _robust_candidate("b", 1, 1.0, 0.0),
            ),
        )
        for index in range(2)
    ]

    with pytest.raises(FrontierLimitError, match=r"max_transitions=2"):
        solve_robust_autopolicy(
            units,
            environments=("one", "two"),
            budget_bytes=2,
            max_transitions=2,
        )


def test_robust_pareto_work_limit_fails_closed_before_pruning() -> None:
    unit = RobustQuantizableUnit(
        "unit",
        (
            _robust_candidate("a", 1, 0.0, 1.0),
            _robust_candidate("b", 1, 1.0, 0.0),
        ),
    )

    with pytest.raises(FrontierLimitError, match=r"max_pareto_comparisons=1"):
        solve_robust_autopolicy(
            [unit],
            environments=("one", "two"),
            budget_bytes=1,
            max_pareto_comparisons=1,
        )


def test_robust_secondary_sum_overflow_fails_closed() -> None:
    unit = RobustQuantizableUnit(
        "unit",
        (
            _robust_candidate("a", 1, 1e308, 1e308),
            _robust_candidate("b", 2, 1e308, 9e307),
        ),
    )

    with pytest.raises(ArithmeticError, match="sum environment distortion overflowed"):
        solve_robust_autopolicy(
            [unit],
            environments=("one", "two"),
            budget_bytes=2,
        )


def test_robust_validation_and_frontier_limit_fail_closed() -> None:
    with pytest.raises(ValueError, match="same environment count"):
        RobustQuantizableUnit(
            "unit",
            (
                _robust_candidate("a", 1, 0.0),
                _robust_candidate("b", 2, 0.0, 0.0),
            ),
        )
    unit = RobustQuantizableUnit(
        "unit",
        (
            _robust_candidate("a", 1, 0.0, 1.0),
            _robust_candidate("b", 2, 1.0, 0.0),
        ),
    )
    with pytest.raises(ValueError, match="environment count disagrees"):
        solve_robust_autopolicy(
            [unit],
            environments=("only-one",),
            budget_bytes=2,
        )
    with pytest.raises(ValueError, match="environments must be unique"):
        solve_robust_autopolicy(
            [unit],
            environments=("same", "same"),
            budget_bytes=2,
        )
    with pytest.raises(FrontierLimitError, match="no approximate plan was returned"):
        solve_robust_autopolicy(
            [unit],
            environments=("one", "two"),
            budget_bytes=2,
            max_frontier_states=1,
        )
