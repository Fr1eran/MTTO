import math
from collections.abc import Sequence
from dataclasses import dataclass

from model.ocs.safe_guard_utility import SafeGuardUtility


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
        safeguard_utility: SafeGuardUtility,
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

        self.safeguard_utility = safeguard_utility
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
            current_max_speed = self.safeguard_utility.get_max_speed(
                current_pos=position,
                current_sp=current_index,
            )
            # Keep the old target when its protection window has been missed.
            # The caller then checks that old bound and truncates through its
            # existing SPEED_HIGH path.
            if speed > current_max_speed:
                return state
            assert state.request_started_at_s is not None
            if time >= state.request_started_at_s + self.step_delay_s:
                return SPSState(target_stopping_point_index=next_index)
            return state

        if next_index >= len(self.accessible_positions_m):
            return state
        next_min_speed = self.safeguard_utility.get_min_speed(
            current_pos=position,
            current_sp=next_index,
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
