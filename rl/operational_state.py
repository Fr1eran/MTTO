"""Shared runtime values exchanged by the RL operational components."""

from dataclasses import dataclass
from enum import IntEnum

from model.ocs.stopping_points_stepping import SPSState


class ViolationCode(IntEnum):
    ONGOING = 0
    FAILED_STOP = 1
    SPEED_LOW = 2
    SPEED_HIGH = 3
    STEP_LIMIT = 4


@dataclass(frozen=True, slots=True)
class OperationalState:
    position_m: float
    speed_mps: float
    acceleration_mps2: float
    operation_time_s: float
    redundant_operation_time_s: float
    energy_consumption_kj: float
    slope_permille: float
    min_speed_mps: float
    max_speed_mps: float
    stop_error_m: float
    sps_state: SPSState
    step_count: int

    @property
    def stopping_point_index(self) -> int:
        return self.sps_state.target_stopping_point_index


@dataclass(frozen=True, slots=True)
class OperationalTransition:
    previous_state: OperationalState
    next_state: OperationalState
    acceleration_mps2: float
    distance_m: float
    duration_s: float
    energy_delta_kj: float
    terminated: bool
    truncated: bool
    violation_code: ViolationCode
