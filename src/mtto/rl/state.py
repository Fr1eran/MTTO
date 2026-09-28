"""Immutable values for an RL environment step."""

from dataclasses import dataclass
from enum import IntEnum

from mtto.domain.kinematics import Motion
from mtto.domain.safeguard import SafeguardViolation, SPSState


class TerminationReason(IntEnum):
    STOPPED_IN_ZONE = 1
    STOPPED_SHORT = 2
    OVERRAN = 3
    UNDER_LOWER_LIMIT = 4
    OVER_UPPER_LIMIT = 5
    OVER_SRTSP = 6


@dataclass(frozen=True, slots=True)
class State:
    s_m: float
    v_mps: float
    commanded_acceleration_mps2: float
    t_s: float
    propulsion_energy_kj: float
    levitation_energy_kj: float
    sps: SPSState
    schedule_time_s: float
    step: int
    schedule_changed: bool
    slope_permille: float
    stop_error_m: float
    lower_limit_mps: float
    upper_limit_mps: float
    srtsp_limit_mps: float
    slack_time_s: float

    @property
    def max_speed_mps(self) -> float:
        return min(self.srtsp_limit_mps, self.upper_limit_mps)

    @property
    def total_energy_kj(self) -> float:
        return self.propulsion_energy_kj + self.levitation_energy_kj


@dataclass(frozen=True, slots=True)
class StepResult:
    step_end_state: State
    next_state: State
    commanded_acceleration_mps2: float
    motion: Motion
    propulsion_delta_kj: float
    levitation_delta_kj: float
    termination_reason: TerminationReason | None
    violation: SafeguardViolation | None
