from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

from mtto.domain._numerics import get_interval_index_scalar_numba

if TYPE_CHECKING:
    from mtto.domain.safeguard.curves import Safeguard

__all__ = [
    "dynamic_limits",
    "min_speed",
    "max_speed",
    "current_stopping_point",
    "latest_intervention_points",
    "ViolationKind",
    "SafeguardViolation",
    "dynamic_limit_violation",
    "SPSState",
    "SPS",
]


@njit(cache=True)
def _interp_scalar_numba(
    x: float,
    xp_row: NDArray[np.float64],
    fp_row: NDArray[np.float64],
    n: int,
) -> float:
    if n <= 0:
        return 0.0
    if x <= xp_row[0]:
        return fp_row[0]
    last = n - 1
    if x >= xp_row[last]:
        return fp_row[last]

    lo = 0
    hi = last
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if x < xp_row[mid]:
            hi = mid
        else:
            lo = mid

    x0 = xp_row[lo]
    x1 = xp_row[hi]
    y0 = fp_row[lo]
    y1 = fp_row[hi]
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))


@njit(cache=True)
def _get_min_speed_numba(
    current_pos: float,
    current_sp: int,
    min_pos_packed: NDArray[np.float64],
    min_speed_packed: NDArray[np.float64],
    min_lengths: NDArray[np.int32],
) -> float:
    if current_sp == -1:
        return 0.0

    curve_len = int(min_lengths[current_sp])
    if curve_len <= 0:
        return 0.0

    if current_pos > min_pos_packed[current_sp, curve_len - 1]:
        return 0.0

    return _interp_scalar_numba(
        current_pos,
        min_pos_packed[current_sp],
        min_speed_packed[current_sp],
        curve_len,
    )


@njit(cache=True)
def _get_max_speed_numba(
    current_pos: float,
    current_sp: int,
    max_pos_packed: NDArray[np.float64],
    max_speed_packed: NDArray[np.float64],
    max_lengths: NDArray[np.int32],
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    gamma: float,
) -> float:
    max_curve_idx = current_sp + 1
    max_curve_len = int(max_lengths[max_curve_idx])
    if current_pos > max_pos_packed[max_curve_idx, 0]:
        current_max_speed = _interp_scalar_numba(
            current_pos,
            max_pos_packed[max_curve_idx],
            max_speed_packed[max_curve_idx],
            max_curve_len,
        )
        if current_max_speed < 0.0:
            current_max_speed = 0.0
    else:
        idx = get_interval_index_scalar_numba(current_pos, speed_limit_intervals)
        if idx < 0:
            idx = 0
        elif idx >= speed_limits.size:
            idx = speed_limits.size - 1
        current_max_speed = speed_limits[idx] * gamma
    return current_max_speed


@njit(cache=True)
def _get_min_and_max_speed_numba(
    current_pos: float,
    current_sp: int,
    min_pos_packed: NDArray[np.float64],
    min_speed_packed: NDArray[np.float64],
    min_lengths: NDArray[np.int32],
    max_pos_packed: NDArray[np.float64],
    max_speed_packed: NDArray[np.float64],
    max_lengths: NDArray[np.int32],
    speed_limits: NDArray[np.float64],
    speed_limit_intervals: NDArray[np.float64],
    gamma: float,
) -> tuple[float, float]:
    current_min_speed = _get_min_speed_numba(
        current_pos,
        current_sp,
        min_pos_packed,
        min_speed_packed,
        min_lengths,
    )

    current_max_speed = _get_max_speed_numba(
        current_pos,
        current_sp,
        max_pos_packed,
        max_speed_packed,
        max_lengths,
        speed_limits,
        speed_limit_intervals,
        gamma,
    )

    return current_min_speed, current_max_speed


@njit(cache=True)
def _get_current_stopping_point_numba(
    current_pos: float,
    current_speed: float,
    min_pos_packed: NDArray[np.float64],
    min_speed_packed: NDArray[np.float64],
    min_lengths: NDArray[np.int32],
) -> int:
    current_sp = -1
    n_curves = min_lengths.size
    for i in range(n_curves):
        curve_len = int(min_lengths[i])
        if curve_len <= 0:
            continue
        right_end_pos = min_pos_packed[i, curve_len - 1]
        if current_pos <= right_end_pos:
            min_speed = _interp_scalar_numba(
                current_pos,
                min_pos_packed[i],
                min_speed_packed[i],
                curve_len,
            )
            if current_speed <= min_speed:
                break
        current_sp += 1
    return current_sp


def min_speed(safeguard: Safeguard, position_m: float, stopping_point: int) -> float:
    return float(
        _get_min_speed_numba(
            float(position_m),
            int(stopping_point),
            safeguard.min_pos_packed,
            safeguard.min_speed_packed,
            safeguard.min_lengths,
        )
    )


def max_speed(safeguard: Safeguard, position_m: float, stopping_point: int) -> float:
    return float(
        _get_max_speed_numba(
            float(position_m),
            int(stopping_point),
            safeguard.max_pos_packed,
            safeguard.max_speed_packed,
            safeguard.max_lengths,
            safeguard.speed_limits,
            safeguard.speed_limit_intervals,
            float(safeguard.params.factor),
        )
    )


def dynamic_limits(
    safeguard: Safeguard, position_m: float, stopping_point: int
) -> tuple[float, float]:
    min_s, max_s = _get_min_and_max_speed_numba(
        float(position_m),
        int(stopping_point),
        safeguard.min_pos_packed,
        safeguard.min_speed_packed,
        safeguard.min_lengths,
        safeguard.max_pos_packed,
        safeguard.max_speed_packed,
        safeguard.max_lengths,
        safeguard.speed_limits,
        safeguard.speed_limit_intervals,
        float(safeguard.params.factor),
    )
    return float(min_s), float(max_s)


def current_stopping_point(
    safeguard: Safeguard, position_m: float, speed_mps: float
) -> int:
    return int(
        _get_current_stopping_point_numba(
            float(position_m),
            float(speed_mps),
            safeguard.min_pos_packed,
            safeguard.min_speed_packed,
            safeguard.min_lengths,
        )
    )


def _get_monotone_curve_position_by_speed(
    curve_pos: NDArray[np.floating],
    curve_speed: NDArray[np.floating],
    current_speed: float,
) -> float:
    """根据单调递减曲线的速度值反查位置。"""
    curve_pos = np.asarray(curve_pos, dtype=np.float64)
    curve_speed = np.asarray(curve_speed, dtype=np.float64)

    if curve_pos.shape != curve_speed.shape:
        raise ValueError("curve_pos and curve_speed must have the same shape")
    if curve_pos.size == 0:
        raise ValueError("curve must contain at least one point")
    if curve_pos.size == 1:
        return float(curve_pos[0])

    if np.any(np.diff(curve_pos) <= 0.0):
        raise ValueError("curve_pos must be strictly increasing")

    target_speed = float(current_speed)
    speed_scale = max(1.0, float(np.max(np.abs(curve_speed))))
    speed_tol = np.finfo(np.float64).eps * speed_scale * 16.0
    if np.any(np.diff(curve_speed) > speed_tol):
        raise ValueError("curve_speed must be monotone decreasing")

    ascending_pos = curve_pos[::-1]
    ascending_speed = curve_speed[::-1]
    unique_speed, unique_indices = np.unique(ascending_speed, return_index=True)
    unique_pos = ascending_pos[unique_indices]

    if unique_speed.size == 1:
        return float(unique_pos[0])

    if target_speed <= unique_speed[0]:
        speed0 = unique_speed[0]
        speed1 = unique_speed[1]
        pos0 = unique_pos[0]
        pos1 = unique_pos[1]
    elif target_speed >= unique_speed[-1]:
        speed0 = unique_speed[-2]
        speed1 = unique_speed[-1]
        pos0 = unique_pos[-2]
        pos1 = unique_pos[-1]
    else:
        return float(np.interp(target_speed, unique_speed, unique_pos))

    return float(pos0 + (target_speed - speed0) * (pos1 - pos0) / (speed1 - speed0))


def latest_intervention_points(
    safeguard: Safeguard, speed_mps: float, stopping_point: int
) -> tuple[float, float]:
    current_speed_value = float(speed_mps)
    sp = int(stopping_point)

    if sp == -1:
        current_min_pos = 0.0
    else:
        current_min_pos = _get_monotone_curve_position_by_speed(
            curve_pos=safeguard.min_curves[sp][0, :],
            curve_speed=safeguard.min_curves[sp][1, :],
            current_speed=current_speed_value,
        )

    current_max_pos = _get_monotone_curve_position_by_speed(
        curve_pos=safeguard.max_curves[sp + 1][0, :],
        curve_speed=safeguard.max_curves[sp + 1][1, :],
        current_speed=current_speed_value,
    )

    return float(current_min_pos), float(current_max_pos)


class ViolationKind(Enum):
    UNDER_LOWER_LIMIT = "UNDER_LOWER_LIMIT"
    OVER_UPPER_LIMIT = "OVER_UPPER_LIMIT"


@dataclass(frozen=True, slots=True)
class SafeguardViolation:
    kind: ViolationKind
    position_m: float
    margin_mps: float  # 越界量：下限 − 速度 或 速度 − 上限（均 > 0）


def dynamic_limit_violation(
    position_m: float,
    speed_mps: float,
    lower_mps: float,
    upper_mps: float,
) -> SafeguardViolation | None:
    pos = float(position_m)
    spd = float(speed_mps)
    low = float(lower_mps)
    up = float(upper_mps)

    if spd < low:
        return SafeguardViolation(
            kind=ViolationKind.UNDER_LOWER_LIMIT,
            position_m=pos,
            margin_mps=low - spd,
        )
    if spd > up:
        return SafeguardViolation(
            kind=ViolationKind.OVER_UPPER_LIMIT,
            position_m=pos,
            margin_mps=spd - up,
        )
    return None


@dataclass(frozen=True, slots=True)
class SPSState:
    """Per-episode stopping-point target and an optional pending request time."""

    target_stopping_point_index: int = -1
    request_started_at_s: float | None = None

    @property
    def request_pending(self) -> bool:
        return self.request_started_at_s is not None


class SPS:
    """Apply the discrete stopping-point stepping constraint.

    A request is made after the train satisfies the next stopping point's
    minimum-speed trigger. It can complete only after ``step_delay_s`` and
    before the current target's maximum-speed boundary is exceeded.
    """

    def __init__(
        self,
        *,
        safeguard: Safeguard,
        accessible_positions_m: Sequence[float],
        danger_positions_m: Sequence[float],
        step_delay_s: float,
    ) -> None:
        accessible = tuple(float(value) for value in accessible_positions_m)
        danger = tuple(float(value) for value in danger_positions_m)
        delay = float(step_delay_s)
        if not accessible:
            raise ValueError("at least one auxiliary stopping point is required")
        if len(accessible) != len(danger):
            raise ValueError("accessible and danger stopping-point counts must match")
        n_max = len(safeguard.max_curves)
        n_min = len(safeguard.min_curves)
        n_acc = len(accessible)
        if not (n_max == n_min + 1 == n_acc + 1):
            raise ValueError(
                "SPS requires len(safeguard.max_curves) == "
                "len(safeguard.min_curves) + 1 == "
                f"len(accessible_positions_m) + 1, got len(max_curves)={n_max}, "
                f"len(min_curves)={n_min}, len(accessible_positions_m)={n_acc}"
            )
        if not math.isfinite(delay) or delay <= 0.0:
            raise ValueError("step_delay_s must be finite and positive")
        if not all(math.isfinite(value) for value in (*accessible, *danger)):
            raise ValueError("stopping-point positions must be finite")
        if any(
            right <= left
            for left, right in zip(accessible[:-1], accessible[1:], strict=True)
        ):
            raise ValueError("accessible stopping-point positions must increase")
        if any(
            right <= left for left, right in zip(danger[:-1], danger[1:], strict=True)
        ):
            raise ValueError("danger stopping-point positions must increase")
        if any(ap > dp for ap, dp in zip(accessible, danger, strict=True)):
            raise ValueError(
                "each accessible position must not exceed its danger position"
            )

        self.safeguard = safeguard
        self.accessible_positions_m = accessible
        self.danger_positions_m = danger
        self.step_delay_s = delay

    def initial_state(self) -> SPSState:
        return SPSState()

    def advance(
        self,
        state: SPSState,
        *,
        position_m: float,
        speed_mps: float,
        time_s: float,
    ) -> SPSState:
        """Return the next SPS state at one control-step endpoint."""
        position = float(position_m)
        speed = float(speed_mps)
        time = float(time_s)
        if not all(math.isfinite(value) for value in (position, speed, time)):
            raise ValueError("position_m, speed_mps, and time_s must be finite")

        current_index = state.target_stopping_point_index
        next_index = current_index + 1
        if state.request_pending:
            current_max_speed = max_speed(
                self.safeguard,
                position_m=position,
                stopping_point=current_index,
            )
            if speed > current_max_speed:
                return state
            assert state.request_started_at_s is not None
            if time >= state.request_started_at_s + self.step_delay_s:
                return SPSState(target_stopping_point_index=next_index)
            return state

        if next_index >= len(self.accessible_positions_m):
            return state
        next_min_speed = min_speed(
            self.safeguard,
            position_m=position,
            stopping_point=next_index,
        )
        if speed > next_min_speed:
            return SPSState(
                target_stopping_point_index=current_index,
                request_started_at_s=time,
            )
        return state

    def target_position_m(self, index: int) -> float:
        if not 0 <= index < len(self.accessible_positions_m):
            raise IndexError(
                f"stopping-point index {index} is outside "
                + f"[0, {len(self.accessible_positions_m) - 1}]"
            )
        return (self.accessible_positions_m[index] + self.danger_positions_m[index]) / 2
