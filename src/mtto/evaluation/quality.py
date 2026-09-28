"""Assess speed profiles at their recorded nodes.

Dynamic-limit audit results depend on node spacing.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from mtto.domain.safeguard import (
    SPS,
    ViolationKind,
    dynamic_limit_violation,
    dynamic_limits,
    max_speed,
)
from mtto.domain.scenario import Scenario, StopState, Task
from mtto.domain.speed_profile import SpeedProfile


class ViolationCategory(Enum):
    DELAY_RELATED = "DELAY_RELATED"
    PRE_TIMEOUT = "PRE_TIMEOUT"


class SpsEventKind(Enum):
    REQUEST_START = "REQUEST_START"
    STEP_COMPLETE = "STEP_COMPLETE"
    REQUEST_UNFINISHED = "REQUEST_UNFINISHED"


@dataclass(frozen=True, slots=True)
class AuditViolation:
    node_index: int
    position_m: float
    speed_mps: float
    kind: ViolationKind
    margin_mps: float
    category: ViolationCategory


@dataclass(frozen=True, slots=True)
class SpsEvent:
    node_index: int
    kind: SpsEventKind
    time_s: float
    position_m: float
    target_stopping_point: int


@dataclass(frozen=True, slots=True, eq=False)
class SafetyAudit:
    target_stopping_point: NDArray[np.int64]
    request_pending: NDArray[np.bool_]
    lower_limit_mps: NDArray[np.float64]
    upper_limit_mps: NDArray[np.float64]
    violations: tuple[AuditViolation, ...]
    events: tuple[SpsEvent, ...]
    min_margin_mps: float  # 全部节点 min(upper - v, v - lower)；越界时为负

    def __post_init__(self) -> None:
        for values in (
            self.target_stopping_point,
            self.request_pending,
            self.lower_limit_mps,
            self.upper_limit_mps,
        ):
            values.flags.writeable = False


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    propulsion_energy_kj: float
    levitation_energy_kj: float
    run_time_s: float
    stop_error_m: float
    comfort_tav_mps2: float
    comfort_rms_mps2: float
    comfort_exceedance_pct: float
    arrival_time_error_s: float | None

    @property
    def total_energy_kj(self) -> float:
        return self.propulsion_energy_kj + self.levitation_energy_kj


@dataclass(frozen=True, slots=True, eq=False)
class QualityReport:
    metrics: QualityMetrics
    audit: SafetyAudit
    completed: bool
    precise_stop: bool
    safe: bool
    punctual: bool | None
    feasible: bool


def audit_dynamic_limits(profile: SpeedProfile, scenario: Scenario) -> SafetyAudit:
    """Audit every node; results depend on the profile's node spacing."""
    safeguard = scenario.safeguard
    sps = SPS(
        safeguard=safeguard,
        accessible_positions_m=scenario.line.accessible_points_m,
        danger_positions_m=scenario.line.danger_points_m,
        step_delay_s=safeguard.params.step_delay_s,
    )
    state = sps.initial_state()
    targets: list[int] = []
    pending: list[bool] = []
    lower_limits: list[float] = []
    upper_limits: list[float] = []
    violations: list[AuditViolation] = []
    events: list[SpsEvent] = []

    for index, (position, speed, time) in enumerate(
        zip(profile.position_m, profile.speed_mps, profile.time_s, strict=True)
    ):
        x, v, t = float(position), float(speed), float(time)
        previous = state
        if index > 0:
            state = sps.advance(previous, position_m=x, speed_mps=v, time_s=t)
            same_target = (
                state.target_stopping_point_index
                == previous.target_stopping_point_index
            )
            if not previous.request_pending and state.request_pending and same_target:
                events.append(
                    SpsEvent(
                        index,
                        SpsEventKind.REQUEST_START,
                        t,
                        x,
                        state.target_stopping_point_index,
                    )
                )
            elif (
                previous.request_pending
                and not state.request_pending
                and state.target_stopping_point_index
                == previous.target_stopping_point_index + 1
            ):
                events.append(
                    SpsEvent(
                        index,
                        SpsEventKind.STEP_COMPLETE,
                        t,
                        x,
                        state.target_stopping_point_index,
                    )
                )

        target = state.target_stopping_point_index
        lower, upper = dynamic_limits(safeguard, x, target)
        targets.append(target)
        pending.append(state.request_pending)
        lower_limits.append(lower)
        upper_limits.append(upper)

        violation = dynamic_limit_violation(x, v, lower, upper)
        if violation is not None:
            delayed = (
                violation.kind == ViolationKind.OVER_UPPER_LIMIT
                and previous.request_pending
                and state.request_pending
                and v > max_speed(safeguard, x, previous.target_stopping_point_index)
            )
            violations.append(
                AuditViolation(
                    index,
                    x,
                    v,
                    violation.kind,
                    violation.margin_mps,
                    ViolationCategory.DELAY_RELATED
                    if delayed
                    else ViolationCategory.PRE_TIMEOUT,
                )
            )

    if state.request_pending:
        events.append(
            SpsEvent(
                profile.position_m.size - 1,
                SpsEventKind.REQUEST_UNFINISHED,
                float(profile.time_s[-1]),
                float(profile.position_m[-1]),
                state.target_stopping_point_index,
            )
        )

    lower_array = np.array(lower_limits, dtype=np.float64)
    upper_array = np.array(upper_limits, dtype=np.float64)
    min_margin = float(
        np.min(
            np.minimum(upper_array - profile.speed_mps, profile.speed_mps - lower_array)
        )
    )
    return SafetyAudit(
        np.array(targets, dtype=np.int64),
        np.array(pending, dtype=np.bool_),
        lower_array,
        upper_array,
        tuple(violations),
        tuple(events),
        min_margin,
    )


def assess(profile: SpeedProfile, scenario: Scenario, task: Task) -> QualityReport:
    acceleration = profile.segment_acceleration_mps2
    if acceleration.size:
        delta = np.abs(np.diff(acceleration, prepend=0.0))
        comfort_tav = float(np.sum(delta))
        comfort_rms = float(np.sqrt(np.sum(delta**2) / delta.size))
        comfort_exceedance = float(
            np.count_nonzero(delta > task.max_acc_change) / delta.size * 100
        )
    else:
        comfort_tav = comfort_rms = comfort_exceedance = 0.0

    stop_error = abs(task.target_position_m - float(profile.position_m[-1]))
    schedule_time = task.final_schedule_time(profile.position_m)
    arrival_error = (
        None if schedule_time is None else float(profile.time_s[-1] - schedule_time)
    )
    metrics = QualityMetrics(
        propulsion_energy_kj=float(profile.propulsion_energy_kj[-1]),
        levitation_energy_kj=float(profile.levitation_energy_kj[-1]),
        run_time_s=float(profile.time_s[-1]),
        stop_error_m=stop_error,
        comfort_tav_mps2=comfort_tav,
        comfort_rms_mps2=comfort_rms,
        comfort_exceedance_pct=comfort_exceedance,
        arrival_time_error_s=arrival_error,
    )
    audit = audit_dynamic_limits(profile, scenario)
    completed = (
        task.stop_state(float(profile.position_m[-1]), float(profile.speed_mps[-1]))
        == StopState.STOPPED_IN_ZONE
    )
    precise_stop = completed and stop_error <= task.max_stop_error_m
    safe = not audit.violations
    punctual = (
        None
        if arrival_error is None
        else precise_stop and abs(arrival_error) < task.max_arr_time_error_s
    )
    feasible = completed and precise_stop and safe and (punctual is None or punctual)
    return QualityReport(
        metrics, audit, completed, precise_stop, safe, punctual, feasible
    )


def selection_key(report: QualityReport) -> tuple[float, ...]:
    metrics = report.metrics
    energy_j = metrics.total_energy_kj * 1000.0
    if report.punctual is None:
        if report.feasible:
            return (1.0, -energy_j, 0.0, 0.0, 0.0)
        return (
            0.0,
            float(report.safe and report.completed),
            float(report.precise_stop),
            -metrics.stop_error_m,
            -energy_j,
        )
    if report.feasible:
        return (1.0, -energy_j, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        0.0,
        float(report.safe and report.completed),
        float(report.precise_stop),
        -metrics.stop_error_m,
        float(report.punctual),
        -abs(metrics.arrival_time_error_s),
        -energy_j,
    )


def best_update_reason(
    candidate: QualityReport, previous: QualityReport | None
) -> str | None:
    if previous is None:
        return "first_evaluation"
    candidate_key = selection_key(candidate)
    previous_key = selection_key(previous)
    if candidate_key <= previous_key:
        return None
    if candidate_key[0] > previous_key[0]:
        return "strict_feasibility_reached"
    if candidate_key[0] == previous_key[0] == 1.0:
        return "lower_energy_among_feasible"
    if candidate.completed and not previous.completed:
        return "safe_success_reached"
    return "better_constraint_fallback"
