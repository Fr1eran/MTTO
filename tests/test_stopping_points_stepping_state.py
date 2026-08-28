from typing import cast

import pytest

from model.ocs import SPS, SafeGuardUtility


class _SafeGuard:
    def get_min_speed(self, *, current_pos: float, current_sp: int) -> float:
        del current_pos, current_sp
        return 1.0

    def get_max_speed(self, *, current_pos: float, current_sp: int) -> float:
        del current_pos, current_sp
        return 3.0


def test_sps_advances_explicit_state_after_delay() -> None:
    sps = SPS(
        safeguard_utility=cast(SafeGuardUtility, cast(object, _SafeGuard())),
        accessible_positions_m=[100.0],
        danger_positions_m=[110.0],
        step_delay_s=2.0,
    )
    state = sps.initial_state()
    requested = sps.advance(state, position_m=0.0, speed_mps=2.0, time_s=5.0)
    assert requested.target_stopping_point_index == -1
    assert requested.request_pending is True
    assert requested.request_started_at_s == 5.0

    completed = sps.advance(
        requested, position_m=0.0, speed_mps=2.0, time_s=7.0
    )
    assert completed.target_stopping_point_index == 0
    assert completed.request_pending is False


def test_sps_keeps_current_target_when_step_window_is_missed() -> None:
    sps = SPS(
        safeguard_utility=cast(SafeGuardUtility, cast(object, _SafeGuard())),
        accessible_positions_m=[100.0],
        danger_positions_m=[110.0],
        step_delay_s=2.0,
    )
    requested = sps.advance(
        sps.initial_state(), position_m=0.0, speed_mps=2.0, time_s=0.0
    )

    missed = sps.advance(requested, position_m=1.0, speed_mps=3.1, time_s=3.0)

    assert missed == requested


def test_sps_validates_stopping_points_and_returns_target_midpoint() -> None:
    guard = cast(SafeGuardUtility, cast(object, _SafeGuard()))
    sps = SPS(
        safeguard_utility=guard,
        accessible_positions_m=[100.0, 200.0],
        danger_positions_m=[110.0, 220.0],
        step_delay_s=2.0,
    )

    assert sps.target_position_m(1) == 210.0
    with pytest.raises(ValueError, match="counts must match"):
        _ = SPS(
            safeguard_utility=guard,
            accessible_positions_m=[100.0],
            danger_positions_m=[110.0, 220.0],
            step_delay_s=2.0,
        )
