"""Exact byte-budget policy search over discrete per-unit candidates.

The optimization problem is a multiple-choice knapsack with a separable proxy:

    minimize  sum(candidate.distortion for each selected candidate)
    subject to
              sum(candidate.byte_size for each selected candidate) <= budget
              exactly one candidate is selected for every quantizable unit

``distortion`` is deliberately generic. It can be a calibration loss increase,
weighted reconstruction error, or another non-negative scalar, provided every
candidate was scored on the same scale. The solver makes no claim about
interactions between units; it exactly solves the supplied additive proxy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

DEFAULT_MAX_TRANSITIONS = 1_000_000
DEFAULT_MAX_PARETO_COMPARISONS = 25_000_000


class InfeasiblePolicyError(ValueError):
    """Raised when no one-candidate-per-unit plan fits the byte budget."""


class FrontierLimitError(RuntimeError):
    """Raised instead of silently approximating an oversized exact search."""


def _validate_identifier(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    return value


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    """One measured precision/configuration choice for a quantizable unit."""

    candidate_id: str
    byte_size: int
    distortion: float

    def __post_init__(self) -> None:
        _validate_identifier(self.candidate_id, name="candidate_id")
        if type(self.byte_size) is not int:
            raise TypeError("byte_size must be an integer, not a boolean or float")
        if self.byte_size <= 0:
            raise ValueError("byte_size must be positive")
        if isinstance(self.distortion, bool) or not isinstance(self.distortion, Real):
            raise TypeError("distortion must be a real number, not a boolean")
        normalized_distortion = float(self.distortion)
        if not math.isfinite(normalized_distortion):
            raise ValueError("distortion must be finite")
        if normalized_distortion < 0.0:
            raise ValueError("distortion must be non-negative")
        object.__setattr__(self, "distortion", normalized_distortion)


@dataclass(frozen=True, slots=True)
class QuantizableUnit:
    """A named unit and every candidate the solver may choose for it."""

    unit_id: str
    candidates: tuple[PolicyCandidate, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.unit_id, name="unit_id")
        if isinstance(self.candidates, (str, bytes)):
            raise TypeError("candidates must be an iterable of PolicyCandidate objects")
        try:
            candidates = tuple(self.candidates)
        except TypeError as exc:
            raise TypeError("candidates must be an iterable of PolicyCandidate objects") from exc
        if not candidates:
            raise ValueError("candidates must not be empty")
        if any(not isinstance(candidate, PolicyCandidate) for candidate in candidates):
            raise TypeError("candidates must contain only PolicyCandidate objects")

        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"candidate_id values must be unique within unit {self.unit_id!r}")

        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id)),
        )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """The selected candidate for one quantizable unit."""

    unit_id: str
    candidate_id: str
    byte_size: int
    distortion: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-compatible decision record."""

        return {
            "unit_id": self.unit_id,
            "candidate_id": self.candidate_id,
            "byte_size": self.byte_size,
            "distortion": self.distortion,
        }


@dataclass(frozen=True, slots=True)
class AutoPolicyPlan:
    """Auditable result of the exact byte-budget search."""

    budget_bytes: int
    used_bytes: int
    remaining_bytes: int
    aggregate_distortion: float
    decisions: tuple[PolicyDecision, ...]
    transitions_evaluated: int
    peak_frontier_states: int

    def to_dict(self) -> dict[str, Any]:
        """Return a result made only of JSON-compatible primitives."""

        return {
            "schema_version": 1,
            "algorithm": "exact_pareto_frontier",
            "objective": "minimize_sum_distortion",
            "constraint": "select_exactly_one_candidate_per_unit_under_byte_budget",
            "tie_breaking": [
                "lower_aggregate_distortion",
                "lower_used_bytes",
                "lexicographically_smaller_candidate_ids_by_sorted_unit_id",
            ],
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "remaining_bytes": self.remaining_bytes,
            "aggregate_distortion": self.aggregate_distortion,
            "unit_count": len(self.decisions),
            "transitions_evaluated": self.transitions_evaluated,
            "peak_frontier_states": self.peak_frontier_states,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    def to_json(self) -> str:
        """Return a canonical compact JSON representation of this plan."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class RobustPolicyCandidate:
    """One candidate scored separately in every frozen environment."""

    candidate_id: str
    byte_size: int
    per_environment_distortion: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.candidate_id, name="candidate_id")
        if type(self.byte_size) is not int:
            raise TypeError("byte_size must be an integer, not a boolean or float")
        if self.byte_size <= 0:
            raise ValueError("byte_size must be positive")
        if isinstance(self.per_environment_distortion, (str, bytes)):
            raise TypeError("per_environment_distortion must be an iterable of real numbers")
        try:
            values = tuple(self.per_environment_distortion)
        except TypeError as exc:
            raise TypeError(
                "per_environment_distortion must be an iterable of real numbers"
            ) from exc
        if not values:
            raise ValueError("per_environment_distortion must not be empty")
        normalized: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("per_environment_distortion must contain only real numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("per_environment_distortion must contain only finite values")
            if number < 0.0:
                raise ValueError("per_environment_distortion must be non-negative")
            normalized.append(number)
        object.__setattr__(self, "per_environment_distortion", tuple(normalized))


@dataclass(frozen=True, slots=True)
class RobustQuantizableUnit:
    """A unit whose candidates retain separate environment scores."""

    unit_id: str
    candidates: tuple[RobustPolicyCandidate, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.unit_id, name="unit_id")
        if isinstance(self.candidates, (str, bytes)):
            raise TypeError("candidates must be an iterable of RobustPolicyCandidate objects")
        try:
            candidates = tuple(self.candidates)
        except TypeError as exc:
            raise TypeError(
                "candidates must be an iterable of RobustPolicyCandidate objects"
            ) from exc
        if not candidates:
            raise ValueError("candidates must not be empty")
        if any(not isinstance(candidate, RobustPolicyCandidate) for candidate in candidates):
            raise TypeError("candidates must contain only RobustPolicyCandidate objects")
        widths = {len(candidate.per_environment_distortion) for candidate in candidates}
        if len(widths) != 1:
            raise ValueError("all candidates in a unit must use the same environment count")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"candidate_id values must be unique within unit {self.unit_id!r}")
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id)),
        )


@dataclass(frozen=True, slots=True)
class RobustPolicyDecision:
    """The selected robust candidate for one quantizable unit."""

    unit_id: str
    candidate_id: str
    byte_size: int
    per_environment_distortion: tuple[float, ...]

    def to_dict(self, environments: tuple[str, ...]) -> dict[str, Any]:
        """Return one JSON-compatible decision record."""

        return {
            "unit_id": self.unit_id,
            "candidate_id": self.candidate_id,
            "byte_size": self.byte_size,
            "per_environment_distortion": {
                environment: value
                for environment, value in zip(
                    environments,
                    self.per_environment_distortion,
                    strict=True,
                )
            },
        }


@dataclass(frozen=True, slots=True)
class RobustAutoPolicyPlan:
    """Exact minimax plan over separately retained environment totals."""

    environments: tuple[str, ...]
    budget_bytes: int
    used_bytes: int
    remaining_bytes: int
    per_environment_distortion: tuple[float, ...]
    worst_environment_distortion: float
    decisions: tuple[RobustPolicyDecision, ...]
    transitions_evaluated: int
    peak_frontier_states: int

    def to_dict(self) -> dict[str, Any]:
        """Return a self-describing result made of JSON primitives."""

        return {
            "schema_version": 1,
            "algorithm": "exact_multienvironment_pareto_frontier",
            "objective": "minimize_max_environment_sum_distortion",
            "constraint": "select_exactly_one_candidate_per_unit_under_byte_budget",
            "tie_breaking": [
                "lower_worst_environment_distortion",
                "lower_sum_environment_distortion",
                "lower_used_bytes",
                "lexicographically_smaller_candidate_ids_by_sorted_unit_id",
            ],
            "environments": list(self.environments),
            "budget_bytes": self.budget_bytes,
            "used_bytes": self.used_bytes,
            "remaining_bytes": self.remaining_bytes,
            "per_environment_distortion": {
                environment: value
                for environment, value in zip(
                    self.environments,
                    self.per_environment_distortion,
                    strict=True,
                )
            },
            "worst_environment_distortion": self.worst_environment_distortion,
            "unit_count": len(self.decisions),
            "transitions_evaluated": self.transitions_evaluated,
            "peak_frontier_states": self.peak_frontier_states,
            "decisions": [decision.to_dict(self.environments) for decision in self.decisions],
        }

    def to_json(self) -> str:
        """Return a canonical compact JSON representation of this plan."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class _SearchState:
    used_bytes: int
    aggregate_distortion: float
    candidates: tuple[PolicyCandidate, ...]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class _RobustSearchState:
    used_bytes: int
    per_environment_distortion: tuple[float, ...]
    candidates: tuple[RobustPolicyCandidate, ...]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)


def _validate_units(units: Iterable[QuantizableUnit]) -> tuple[QuantizableUnit, ...]:
    if isinstance(units, (str, bytes)):
        raise TypeError("units must be an iterable of QuantizableUnit objects")
    try:
        normalized_units = tuple(units)
    except TypeError as exc:
        raise TypeError("units must be an iterable of QuantizableUnit objects") from exc
    if not normalized_units:
        raise ValueError("units must not be empty")
    if any(not isinstance(unit, QuantizableUnit) for unit in normalized_units):
        raise TypeError("units must contain only QuantizableUnit objects")

    unit_ids = [unit.unit_id for unit in normalized_units]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unit_id values must be unique")
    return tuple(sorted(normalized_units, key=lambda unit: unit.unit_id))


def _validate_robust_units(
    units: Iterable[RobustQuantizableUnit],
    *,
    environment_count: int,
) -> tuple[RobustQuantizableUnit, ...]:
    if isinstance(units, (str, bytes)):
        raise TypeError("units must be an iterable of RobustQuantizableUnit objects")
    try:
        normalized_units = tuple(units)
    except TypeError as exc:
        raise TypeError("units must be an iterable of RobustQuantizableUnit objects") from exc
    if not normalized_units:
        raise ValueError("units must not be empty")
    if any(not isinstance(unit, RobustQuantizableUnit) for unit in normalized_units):
        raise TypeError("units must contain only RobustQuantizableUnit objects")
    unit_ids = [unit.unit_id for unit in normalized_units]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unit_id values must be unique")
    for unit in normalized_units:
        if len(unit.candidates[0].per_environment_distortion) != environment_count:
            raise ValueError(
                f"candidate environment count disagrees with environments for {unit.unit_id!r}"
            )
    return tuple(sorted(normalized_units, key=lambda unit: unit.unit_id))


def _validate_max_frontier_states(max_frontier_states: int) -> int:
    if type(max_frontier_states) is not int:
        raise TypeError("max_frontier_states must be an integer")
    if max_frontier_states <= 0:
        raise ValueError("max_frontier_states must be positive")
    return max_frontier_states


def _validate_max_transitions(max_transitions: int) -> int:
    if type(max_transitions) is not int:
        raise TypeError("max_transitions must be an integer")
    if max_transitions <= 0:
        raise ValueError("max_transitions must be positive")
    return max_transitions


def _validate_max_pareto_comparisons(max_pareto_comparisons: int) -> int:
    if type(max_pareto_comparisons) is not int:
        raise TypeError("max_pareto_comparisons must be an integer")
    if max_pareto_comparisons <= 0:
        raise ValueError("max_pareto_comparisons must be positive")
    return max_pareto_comparisons


def _checked_fsum(values: Iterable[float], *, label: str) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ArithmeticError(f"{label} overflowed") from exc
    if not math.isfinite(total):
        raise ArithmeticError(f"{label} overflowed")
    return total


def _suffix_minimum_bytes(
    units: tuple[QuantizableUnit, ...] | tuple[RobustQuantizableUnit, ...],
) -> tuple[int, ...]:
    """Return the minimum bytes required from each unit index onward."""

    suffix = [0] * (len(units) + 1)
    for index in range(len(units) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + min(
            candidate.byte_size for candidate in units[index].candidates
        )
    return tuple(suffix)


def _prefer_same_size(candidate: _SearchState, incumbent: _SearchState) -> bool:
    if candidate.aggregate_distortion != incumbent.aggregate_distortion:
        return candidate.aggregate_distortion < incumbent.aggregate_distortion
    return candidate.candidate_ids < incumbent.candidate_ids


def _pareto_prune(states_by_bytes: dict[int, _SearchState]) -> dict[int, _SearchState]:
    """Drop states dominated by one using no more bytes and no more distortion."""

    frontier: dict[int, _SearchState] = {}
    best_distortion = float("inf")
    for used_bytes in sorted(states_by_bytes):
        state = states_by_bytes[used_bytes]
        if state.aggregate_distortion < best_distortion:
            frontier[used_bytes] = state
            best_distortion = state.aggregate_distortion
    return frontier


def solve_autopolicy(
    units: Iterable[QuantizableUnit],
    *,
    budget_bytes: int,
    max_frontier_states: int = 250_000,
    max_transitions: int = DEFAULT_MAX_TRANSITIONS,
) -> AutoPolicyPlan:
    """Return the exact minimum-distortion plan under ``budget_bytes``.

    Units and candidates are canonicalized by identifier. Objective ties prefer
    fewer bytes, then the lexicographically smaller candidate-ID tuple in
    sorted unit-ID order. The Pareto-frontier dynamic program is exact for the
    additive proxy while avoiding an array whose size scales with the raw byte
    budget.
    """

    if type(budget_bytes) is not int:
        raise TypeError("budget_bytes must be an integer, not a boolean or float")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    _validate_max_frontier_states(max_frontier_states)
    _validate_max_transitions(max_transitions)
    normalized_units = _validate_units(units)

    suffix_minimum_bytes = _suffix_minimum_bytes(normalized_units)
    minimum_required_bytes = suffix_minimum_bytes[0]
    if minimum_required_bytes > budget_bytes:
        raise InfeasiblePolicyError(
            f"budget_bytes={budget_bytes} is infeasible; "
            f"at least {minimum_required_bytes} bytes are required"
        )

    frontier = {
        0: _SearchState(
            used_bytes=0,
            aggregate_distortion=0.0,
            candidates=(),
        )
    }
    transitions_evaluated = 0
    peak_frontier_states = 1

    for unit_index, unit in enumerate(normalized_units):
        next_states_by_bytes: dict[int, _SearchState] = {}
        for state in (frontier[key] for key in sorted(frontier)):
            for candidate in unit.candidates:
                used_bytes = state.used_bytes + candidate.byte_size
                if used_bytes + suffix_minimum_bytes[unit_index + 1] > budget_bytes:
                    continue
                if transitions_evaluated >= max_transitions:
                    raise FrontierLimitError(
                        "exact AutoPolicy search exceeded "
                        f"max_transitions={max_transitions}; no approximate plan was returned"
                    )
                aggregate_distortion = _checked_fsum(
                    (state.aggregate_distortion, candidate.distortion),
                    label="aggregate distortion",
                )
                transitions_evaluated += 1
                next_state = _SearchState(
                    used_bytes=used_bytes,
                    aggregate_distortion=aggregate_distortion,
                    candidates=(*state.candidates, candidate),
                )
                incumbent = next_states_by_bytes.get(used_bytes)
                if incumbent is None or _prefer_same_size(next_state, incumbent):
                    next_states_by_bytes[used_bytes] = next_state

        if not next_states_by_bytes:  # pragma: no cover - guarded by the minimum-size check
            raise InfeasiblePolicyError("no complete policy fits the byte budget")
        frontier = _pareto_prune(next_states_by_bytes)
        if len(frontier) > max_frontier_states:
            raise FrontierLimitError(
                "exact AutoPolicy frontier exceeded max_frontier_states="
                f"{max_frontier_states}; no approximate plan was returned"
            )
        peak_frontier_states = max(peak_frontier_states, len(frontier))

    winner = min(
        frontier.values(),
        key=lambda state: (
            state.aggregate_distortion,
            state.used_bytes,
            state.candidate_ids,
        ),
    )
    decisions = tuple(
        PolicyDecision(
            unit_id=unit.unit_id,
            candidate_id=candidate.candidate_id,
            byte_size=candidate.byte_size,
            distortion=candidate.distortion,
        )
        for unit, candidate in zip(normalized_units, winner.candidates, strict=True)
    )
    return AutoPolicyPlan(
        budget_bytes=budget_bytes,
        used_bytes=winner.used_bytes,
        remaining_bytes=budget_bytes - winner.used_bytes,
        aggregate_distortion=winner.aggregate_distortion,
        decisions=decisions,
        transitions_evaluated=transitions_evaluated,
        peak_frontier_states=peak_frontier_states,
    )


def _robust_state_dominates(
    left: _RobustSearchState,
    right: _RobustSearchState,
) -> bool:
    if left.used_bytes > right.used_bytes:
        return False
    no_worse = all(
        left_value <= right_value
        for left_value, right_value in zip(
            left.per_environment_distortion,
            right.per_environment_distortion,
            strict=True,
        )
    )
    if not no_worse:
        return False
    strictly_better = left.used_bytes < right.used_bytes or any(
        left_value < right_value
        for left_value, right_value in zip(
            left.per_environment_distortion,
            right.per_environment_distortion,
            strict=True,
        )
    )
    if strictly_better:
        return True
    return left.candidate_ids <= right.candidate_ids


def _robust_pareto_prune(
    states: Iterable[_RobustSearchState],
) -> list[_RobustSearchState]:
    """Return the exact componentwise byte/environment Pareto frontier."""

    best_for_point: dict[tuple[int, tuple[float, ...]], _RobustSearchState] = {}
    for state in states:
        key = (state.used_bytes, state.per_environment_distortion)
        incumbent = best_for_point.get(key)
        if incumbent is None or state.candidate_ids < incumbent.candidate_ids:
            best_for_point[key] = state
    ordered = sorted(
        best_for_point.values(),
        key=lambda state: (
            state.used_bytes,
            state.per_environment_distortion,
            state.candidate_ids,
        ),
    )
    if len(ordered) <= 1:
        return ordered

    byte_sizes = np.asarray([state.used_bytes for state in ordered], dtype=np.int64)
    costs = np.asarray(
        [state.per_environment_distortion for state in ordered],
        dtype=np.float64,
    )
    dominated = np.zeros(len(ordered), dtype=np.bool_)
    chunk_size = 128
    for start in range(0, len(ordered), chunk_size):
        stop = min(start + chunk_size, len(ordered))
        right_bytes = byte_sizes[start:stop]
        right_costs = costs[start:stop]
        no_worse = (byte_sizes[:, None] <= right_bytes[None, :]) & np.all(
            costs[:, None, :] <= right_costs[None, :, :],
            axis=2,
        )
        strictly_better = (byte_sizes[:, None] < right_bytes[None, :]) | np.any(
            costs[:, None, :] < right_costs[None, :, :],
            axis=2,
        )
        dominated[start:stop] = np.any(no_worse & strictly_better, axis=0)
    return [
        state
        for state, is_dominated in zip(
            ordered,
            dominated.tolist(),
            strict=True,
        )
        if not is_dominated
    ]


def solve_robust_autopolicy(
    units: Iterable[RobustQuantizableUnit],
    *,
    environments: Iterable[str],
    budget_bytes: int,
    max_frontier_states: int = 250_000,
    max_transitions: int = DEFAULT_MAX_TRANSITIONS,
    max_pareto_comparisons: int = DEFAULT_MAX_PARETO_COMPARISONS,
) -> RobustAutoPolicyPlan:
    """Exactly minimize the worst summed environment distortion under a budget.

    This solves the supplied discrete proxy only. It assumes per-unit
    distortions are additive and does not claim to model cross-unit interactions
    or downstream language-model loss.
    """

    if type(budget_bytes) is not int:
        raise TypeError("budget_bytes must be an integer, not a boolean or float")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    _validate_max_frontier_states(max_frontier_states)
    _validate_max_transitions(max_transitions)
    _validate_max_pareto_comparisons(max_pareto_comparisons)
    if isinstance(environments, (str, bytes)):
        raise TypeError("environments must be an iterable of strings")
    try:
        normalized_environments = tuple(environments)
    except TypeError as exc:
        raise TypeError("environments must be an iterable of strings") from exc
    if not normalized_environments:
        raise ValueError("environments must not be empty")
    for environment in normalized_environments:
        _validate_identifier(environment, name="environment")
    if len(set(normalized_environments)) != len(normalized_environments):
        raise ValueError("environments must be unique")

    normalized_units = _validate_robust_units(
        units,
        environment_count=len(normalized_environments),
    )
    suffix_minimum_bytes = _suffix_minimum_bytes(normalized_units)
    minimum_required_bytes = suffix_minimum_bytes[0]
    if minimum_required_bytes > budget_bytes:
        raise InfeasiblePolicyError(
            f"budget_bytes={budget_bytes} is infeasible; "
            f"at least {minimum_required_bytes} bytes are required"
        )

    frontier = [
        _RobustSearchState(
            used_bytes=0,
            per_environment_distortion=(0.0,) * len(normalized_environments),
            candidates=(),
        )
    ]
    transitions_evaluated = 0
    peak_frontier_states = 1
    for unit_index, unit in enumerate(normalized_units):
        next_states: list[_RobustSearchState] = []
        for state in frontier:
            for candidate in unit.candidates:
                used_bytes = state.used_bytes + candidate.byte_size
                if used_bytes + suffix_minimum_bytes[unit_index + 1] > budget_bytes:
                    continue
                if transitions_evaluated >= max_transitions:
                    raise FrontierLimitError(
                        "exact robust AutoPolicy search exceeded "
                        f"max_transitions={max_transitions}; no approximate plan was returned"
                    )
                distortions = tuple(
                    _checked_fsum(
                        (left, right),
                        label="aggregate environment distortion",
                    )
                    for left, right in zip(
                        state.per_environment_distortion,
                        candidate.per_environment_distortion,
                        strict=True,
                    )
                )
                transitions_evaluated += 1
                next_states.append(
                    _RobustSearchState(
                        used_bytes=used_bytes,
                        per_environment_distortion=distortions,
                        candidates=(*state.candidates, candidate),
                    )
                )
        if not next_states:  # pragma: no cover - guarded by the minimum-size check
            raise InfeasiblePolicyError("no complete policy fits the byte budget")
        comparison_count = len(next_states) ** 2
        if comparison_count > max_pareto_comparisons:
            raise FrontierLimitError(
                "exact robust AutoPolicy Pareto prune requires "
                f"{comparison_count} pairwise comparisons, exceeding "
                f"max_pareto_comparisons={max_pareto_comparisons}; "
                "no approximate plan was returned"
            )
        frontier = _robust_pareto_prune(next_states)
        if len(frontier) > max_frontier_states:
            raise FrontierLimitError(
                "exact robust AutoPolicy frontier exceeded max_frontier_states="
                f"{max_frontier_states}; no approximate plan was returned"
            )
        peak_frontier_states = max(peak_frontier_states, len(frontier))

    winner = min(
        frontier,
        key=lambda state: (
            max(state.per_environment_distortion),
            _checked_fsum(
                state.per_environment_distortion,
                label="sum environment distortion",
            ),
            state.used_bytes,
            state.candidate_ids,
        ),
    )
    decisions = tuple(
        RobustPolicyDecision(
            unit_id=unit.unit_id,
            candidate_id=candidate.candidate_id,
            byte_size=candidate.byte_size,
            per_environment_distortion=candidate.per_environment_distortion,
        )
        for unit, candidate in zip(normalized_units, winner.candidates, strict=True)
    )
    return RobustAutoPolicyPlan(
        environments=normalized_environments,
        budget_bytes=budget_bytes,
        used_bytes=winner.used_bytes,
        remaining_bytes=budget_bytes - winner.used_bytes,
        per_environment_distortion=winner.per_environment_distortion,
        worst_environment_distortion=max(winner.per_environment_distortion),
        decisions=decisions,
        transitions_evaluated=transitions_evaluated,
        peak_frontier_states=peak_frontier_states,
    )
