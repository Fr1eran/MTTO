from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from mtto.domain.dynamics import Vehicle
from mtto.domain.energy import EnergyParams
from mtto.domain.line import Line
from mtto.domain.safeguard import Safeguard

# A train at or below this speed counts as stopped.
STOPPED_SPEED_MPS: float = 0.01


class StopState(Enum):
    STOPPED_IN_ZONE = "STOPPED_IN_ZONE"
    STOPPED_SHORT = "STOPPED_SHORT"
    OVERRAN = "OVERRAN"
    MOVING = "MOVING"


@dataclass(frozen=True, slots=True, eq=False)
class Scenario:
    """Operating scenario aggregating line, vehicle, energy, and safeguard models."""

    line: Line
    vehicle: Vehicle
    energy: EnergyParams
    safeguard: Safeguard
    scenario_hash: str


@dataclass(frozen=True, slots=True)
class ScheduleChange:
    """Schedule change event definition."""

    trigger_position_m: float
    new_schedule_time_s: float

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.trigger_position_m)
            and math.isfinite(self.new_schedule_time_s)
        ):
            raise ValueError("ScheduleChange values must be finite")
        if self.new_schedule_time_s <= 0.0:
            raise ValueError(
                f"new_schedule_time_s must be positive, got {self.new_schedule_time_s}"
            )


@dataclass(frozen=True, slots=True)
class Task:
    """Operating task representing a single station-to-station run."""

    start_position_m: float
    target_position_m: float
    schedule_time_s: float | None
    max_acc_change: float
    max_stop_error_m: float
    max_arr_time_error_s: float
    schedule_change: ScheduleChange | None = None

    def __post_init__(self) -> None:
        numeric_values = [
            self.start_position_m,
            self.target_position_m,
            self.max_acc_change,
            self.max_stop_error_m,
            self.max_arr_time_error_s,
        ]
        if self.schedule_time_s is not None:
            numeric_values.append(self.schedule_time_s)
        if not all(math.isfinite(v) for v in numeric_values):
            raise ValueError("All Task numeric values must be finite")
        if self.target_position_m <= self.start_position_m:
            raise ValueError(
                f"target_position_m ({self.target_position_m}) must be greater than "
                f"start_position_m ({self.start_position_m}): 反向运行暂不支持"
            )
        if self.schedule_time_s is not None and self.schedule_time_s <= 0.0:
            raise ValueError(
                f"schedule_time_s must be positive or None, got {self.schedule_time_s}"
            )
        if self.max_acc_change <= 0.0:
            raise ValueError(
                f"max_acc_change must be positive, got {self.max_acc_change}"
            )
        if self.max_stop_error_m <= 0.0:
            raise ValueError(
                f"max_stop_error_m must be positive, got {self.max_stop_error_m}"
            )
        if self.max_arr_time_error_s <= 0.0:
            raise ValueError(
                "max_arr_time_error_s must be positive, "
                f"got {self.max_arr_time_error_s}"
            )
        if self.schedule_change is not None:
            if self.schedule_time_s is None:
                raise ValueError(
                    "schedule_time_s must not be None when schedule_change is provided"
                )
            trig = self.schedule_change.trigger_position_m
            if not (self.start_position_m <= trig < self.target_position_m):
                raise ValueError(
                    f"schedule_change trigger_position_m ({trig}) "
                    f"must be in [{self.start_position_m}, {self.target_position_m})"
                )

    def stop_state(self, position_m: float, speed_mps: float) -> StopState:
        """Evaluate stop state according to stopping thresholds.

        A train may run past the target; the stop is judged where it halts.
        Overrunning the target by more than the stopping zone is final.
        """
        stop_zone_m = 30 * self.max_stop_error_m
        stopped = abs(speed_mps) <= STOPPED_SPEED_MPS
        if stopped and abs(self.target_position_m - position_m) <= stop_zone_m:
            return StopState.STOPPED_IN_ZONE
        if position_m - self.target_position_m > stop_zone_m:
            return StopState.OVERRAN
        if stopped:
            return StopState.STOPPED_SHORT
        return StopState.MOVING

    def final_schedule_time(self, position_m: NDArray[np.float64]) -> float | None:
        """Determine final scheduled time after evaluating schedule change trigger."""
        if self.schedule_time_s is None:
            return None
        if self.schedule_change is None:
            return self.schedule_time_s
        if np.any(position_m[:-1] >= self.schedule_change.trigger_position_m):
            return self.schedule_change.new_schedule_time_s
        return self.schedule_time_s
