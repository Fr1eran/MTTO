"""Golden case definitions.

The controllers below only generate the frozen action sequences in
``actions/``. Once a sequence exists, golden runs replay it and never call
the controller again, so the controllers may break when interfaces change.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from mtto.domain.srtsp import lookup_upper_speed

SCHEDULE_TIME_S = 465.0
STEP_DISTANCE_M = 30.0
GAMMA = 0.998
REWARD_PRESETS = (
    "basic",
    "basic_safety",
    "basic_punctuality",
    "basic_safety_punctuality",
    "li2023_scaled",
)
ACTION_GENERATION_PRESET = "basic_safety_punctuality"
MAX_GENERATED_STEPS = 3000

Controller = Callable[[object, object], float]


def _stopper(stop_offset_m: float, *, speed_ratio: float = 0.9) -> Controller:
    """Track a fraction of the SRTSP look-ahead limit, then brake to a point."""

    def controller(env, state) -> float:
        remaining = env.task.target_position_m - state.s_m
        braking_distance = remaining - stop_offset_m
        speed = state.v_mps
        if braking_distance <= 0.0:
            return -1.0
        if speed * speed / (2.0 * braking_distance) >= 0.6:
            return -speed * speed / (2.0 * braking_distance)
        target = min(
            speed_ratio * lookup_upper_speed(env.srtsp_lookup, state.s_m + 30.0),
            0.98 * math.sqrt(1.2 * braking_distance),
        )
        return (target**2 - speed**2) / (2.0 * min(30.0, remaining))

    return controller


def _look_ahead(speed_ratio: float) -> Controller:
    """Track a fraction of the SRTSP look-ahead limit without a final stop."""

    def controller(env, state) -> float:
        remaining = env.task.target_position_m - state.s_m
        target = min(
            speed_ratio * lookup_upper_speed(env.srtsp_lookup, state.s_m + 30.0),
            math.sqrt(1.6 * max(remaining - 2.0, 0.0)),
        )
        return (target**2 - state.v_mps**2) / (2.0 * min(30.0, remaining))

    return controller


def _hold(speed_mps: float) -> Controller:
    """Cruise slowly so that no stopping-point step is ever requested."""

    def controller(env, state) -> float:
        target = min(
            speed_mps, 0.9 * lookup_upper_speed(env.srtsp_lookup, state.s_m + 30.0)
        )
        return (target**2 - state.v_mps**2) / 60.0

    return controller


def _brake_after(position_m: float) -> Controller:
    cruise = _stopper(0.1)

    def controller(env, state) -> float:
        if state.s_m < position_m:
            return cruise(env, state)
        return -1.0

    return controller


def _with_coasting_window(first_step: int) -> Controller:
    """Insert commands on both sides of the 1e-6 coasting threshold."""
    cruise = _stopper(0.1)
    window = (0.0, 5e-7, -5e-7, 0.0, 2e-6, -2e-6)

    def controller(env, state) -> float:
        offset = state.step_count - first_step
        if 0 <= offset < len(window):
            return window[offset]
        return cruise(env, state)

    return controller


@dataclass(frozen=True, slots=True)
class RLCase:
    name: str
    expected_outcome: str
    controller: Controller


@dataclass(frozen=True, slots=True)
class ScheduleChangeCase:
    name: str
    actions_from: str
    trigger_position_m: float
    delta_time_s: float
    expected_change: str


@dataclass(frozen=True, slots=True)
class DPCase:
    name: str
    target_position_m: float
    schedule_time_s: float
    delta_speed_mps: float
    uniform_step_size_m: float


RL_CASES = (
    RLCase("stop_in_zone", "STOPPED_IN_ZONE", _stopper(0.1)),
    RLCase("stopped_short", "STOPPED_SHORT", _stopper(40.0)),
    RLCase("overran", "OVERRAN", _look_ahead(0.9)),
    RLCase("overran_and_over_limit", "OVERRAN", _look_ahead(0.95)),
    RLCase("under_lower_limit", "UNDER_LOWER_LIMIT", _brake_after(8000.0)),
    RLCase("over_upper_limit", "OVER_UPPER_LIMIT", _hold(8.0)),
    RLCase("over_srtsp", "OVER_SRTSP", lambda env, state: 1.0),
    RLCase("zero_displacement_start", "STOPPED_SHORT", lambda env, state: -1.0),
    RLCase("coasting_threshold", "STOPPED_IN_ZONE", _with_coasting_window(40)),
)

# Terminal trigger: inside the final transition of ``stop_in_zone``
# (29265.0 m -> 29269.946 m), so the episode ends before the change applies.
SCHEDULE_CHANGE_CASES = (
    ScheduleChangeCase(
        "schedule_change_at_start", "stop_in_zone", 135.0, 30.0, "reset"
    ),
    ScheduleChangeCase(
        "schedule_change_en_route", "stop_in_zone", 15000.5, 30.0, "en_route"
    ),
    ScheduleChangeCase(
        "schedule_change_en_route_earlier",
        "stop_in_zone",
        15000.5,
        -30.0,
        "en_route",
    ),
    ScheduleChangeCase(
        "schedule_change_terminal_step", "stop_in_zone", 29267.0, 30.0, "never"
    ),
)

# Reduced DP configuration: the target lies
# inside the auxiliary stopping area [11130, 11590] m, where stopping is allowed.
DP_CASES = (
    DPCase("dp_feasible", 11360.0, 260.0, 0.5, 50.0),
    DPCase("dp_time_tolerance_missed", 11360.0, 220.0, 0.5, 50.0),
)
