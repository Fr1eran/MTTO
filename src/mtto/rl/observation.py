"""Pure conversion from an operational state to the agent observation."""

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from mtto.domain.dynamics import Vehicle
from mtto.domain.line import Line
from mtto.domain.scenario import STOPPED_SPEED_MPS, Task
from mtto.rl.rewards import (
    PUNCTUALITY_POTENTIAL_SIGMA_S,
    SAFETY_RESERVE_HORIZON_STEPS,
    reference_punctuality_slack,
)
from mtto.rl.state import State

POLICY_IO_VERSION: Final[int] = 2

# Scales of the signed-log features.
STOP_SYMLOG_SCALE_M: Final[float] = 0.1
STOP_ERROR_SYMLOG_LIMIT_M: Final[float] = 50.0
SLACK_SYMLOG_SCALE_S: Final[float] = 10.0


def _symlog(value: float, scale: float, limit: float) -> float:
    """Signed log1p(|value| / scale), mapped so that |value| = limit gives 1."""
    magnitude = math.log1p(abs(value) / scale) / math.log1p(limit / scale)
    return math.copysign(min(1.0, magnitude), value)


class ObservationBuilder:
    """Build the 11-dimensional observation.

    0 route progress, 1 signed log distance to the target (negative past it),
    2 speed, 3 previous acceleration, 4 upper speed limit, 5 lower speed limit,
    6 punctuality ratio e / hypot(e, sigma) (the punctuality potential is
    -K * o6^2), 7 acceleration that stops exactly on the target, 8 braking
    reserve, 9 signed log slack time, 10 stop error left by full braking.
    """

    OBSERVATION_DIM: int = 11
    LOW: Final[tuple[float, ...]] = (0, -1, 0, -1, 0, 0, -1, -1, 0, -1, -1)

    def __init__(
        self,
        *,
        vehicle: Vehicle,
        track: Line,
        task: Task,
        whole_distance_m: float,
        initial_min_operation_time_s: float,
    ) -> None:
        self.vehicle: Vehicle = vehicle
        self.task: Task = task
        self.whole_distance_m: float = max(whole_distance_m, 1e-12)
        self.initial_min_operation_time_s = initial_min_operation_time_s
        # Speeds are scaled by the line's highest limit, not the vehicle's.
        self.speed_scale_mps: float = float(np.max(track.speed_limits))
        self._obs_buffer: NDArray[np.float32] = np.empty(
            self.OBSERVATION_DIM, dtype=np.float32
        )

    def build(
        self,
        state: State,
        out: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        uses_internal_buffer = out is None
        target = self._obs_buffer if uses_internal_buffer else out
        assert target is not None
        remaining_m = self.task.target_position_m - state.s_m
        speed = state.v_mps
        scale = self.speed_scale_mps
        target[0] = min(1.0, max(0.0, 1.0 - remaining_m / self.whole_distance_m))
        target[1] = _symlog(remaining_m, STOP_SYMLOG_SCALE_M, self.whole_distance_m)
        target[2] = min(1.0, max(0.0, speed / scale))
        target[3] = self.normalize_acc_to_action(state.commanded_acceleration_mps2)
        target[4] = min(1.0, max(0.0, state.max_speed_mps / scale))
        target[5] = min(1.0, max(0.0, state.lower_limit_mps / scale))
        slack_error_s = state.slack_time_s - reference_punctuality_slack(
            state.s_m,
            state.schedule_time_s,
            self.task,
            initial_min_operation_time_s=self.initial_min_operation_time_s,
        )
        target[6] = slack_error_s / math.hypot(
            slack_error_s, PUNCTUALITY_POTENTIAL_SIGMA_S
        )
        # Full braking once at or past the target while still moving.
        if abs(speed) <= STOPPED_SPEED_MPS:
            stopping_acc = 0.0
        elif remaining_m <= 0.0:
            stopping_acc = self.vehicle.max_dec
        else:
            stopping_acc = -(speed**2) / (2.0 * remaining_m)
        target[7] = self.normalize_acc_to_action(stopping_acc)
        horizon = SAFETY_RESERVE_HORIZON_STEPS
        target[8] = min(horizon, max(0.0, state.braking_reserve_steps)) / horizon
        # Unsaturated slack: tells 60 s late apart from 200 s late.
        target[9] = _symlog(
            state.slack_time_s, SLACK_SYMLOG_SCALE_S, state.schedule_time_s
        )
        # < 0: spare braking distance; > 0: unavoidable overrun (final past it).
        stop_error_m = speed**2 / (2.0 * self.vehicle.max_dec_abs) - remaining_m
        target[10] = _symlog(
            stop_error_m, STOP_SYMLOG_SCALE_M, STOP_ERROR_SYMLOG_LIMIT_M
        )
        # The internal buffer is scratch storage only.  Returning it would
        # expose a mutable array that the next build() call overwrites.
        return target.copy() if uses_internal_buffer else target

    def normalize_acc_to_action(self, acc: float) -> float:
        value = (
            2.0
            * (float(acc) - self.vehicle.max_dec)
            / (self.vehicle.max_acc - self.vehicle.max_dec)
            - 1.0
        )
        return max(-1.0, min(1.0, value))

    def denormalize_action(self, action: float) -> float:
        value = (self.vehicle.max_acc + self.vehicle.max_dec) / 2.0 + float(action) * (
            self.vehicle.max_acc - self.vehicle.max_dec
        ) / 2.0
        return float(value)
