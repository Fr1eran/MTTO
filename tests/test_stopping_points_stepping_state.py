from typing import cast

from model.ocs import SPS, SafeGuardUtility


class _SafeGuard:
    def get_min_speed(self, *, current_pos: float, current_sp: int) -> float:
        del current_pos, current_sp
        return 1.0


def test_sps_advances_explicit_state_after_delay() -> None:
    sps = SPS(
        cast(SafeGuardUtility, cast(object, _SafeGuard())),
        [100.0],
        [110.0],
        T_s=2.0,
    )
    state = sps.initial_state()
    requested = sps.advance(state, current_pos=0.0, current_speed=2.0, current_time=5.0)
    assert requested.target_stopping_point_index == -1
    assert requested.request_pending is True
    assert requested.request_timestamp_s == 5.0

    waiting = sps.advance(
        requested, current_pos=0.0, current_speed=2.0, current_time=7.0
    )
    assert waiting == requested
    completed = sps.advance(
        requested, current_pos=0.0, current_speed=2.0, current_time=7.1
    )
    assert completed.target_stopping_point_index == 0
    assert completed.request_pending is False
