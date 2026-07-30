import math
from types import SimpleNamespace

import pytest

from dp.core import _calculate_transition_with_context
from model.common import (
    calc_transition_from_acc_scalar_numba,
    calc_transition_to_speed_scalar_numba,
)


@pytest.mark.parametrize(
    ("begin_speed", "acceleration", "distance", "expected"),
    [
        (10.0, 0.0, 30.0, (10.0, 30.0, 3.0)),
        (0.0, 0.0, 30.0, (0.0, 0.0, 0.0)),
        (10.0, 1.0, 30.0, (math.sqrt(160.0), 30.0, math.sqrt(160.0) - 10.0)),
        (10.0, -2.0, 30.0, (0.0, 25.0, 5.0)),
    ],
)
def test_calc_transition_from_acc_scalar_numba(
    begin_speed: float,
    acceleration: float,
    distance: float,
    expected: tuple[float, float, float],
) -> None:
    actual = calc_transition_from_acc_scalar_numba(
        begin_speed,
        acceleration,
        distance,
    )

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("begin_speed", "end_speed", "distance", "expected"),
    [
        (10.0, 10.0, 30.0, (0.0, 3.0)),
        (10.0, 12.0, 30.0, (44.0 / 60.0, 30.0 / 11.0)),
        (12.0, 10.0, 30.0, (-44.0 / 60.0, 30.0 / 11.0)),
    ],
)
def test_calc_transition_to_speed_scalar_numba(
    begin_speed: float,
    end_speed: float,
    distance: float,
    expected: tuple[float, float],
) -> None:
    actual = calc_transition_to_speed_scalar_numba(
        begin_speed,
        end_speed,
        distance,
    )

    assert actual == pytest.approx(expected)


def test_dp_transition_uses_shared_end_speed_transition() -> None:
    class FakeSafeguardUtility:
        @staticmethod
        def detect_any_danger(*, pos, speed) -> bool:
            assert pos.size == speed.size
            return False

    class FakeECC:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def calc_energy(self, **kwargs) -> tuple[float, float]:
            self.kwargs = kwargs
            return 2.0, 3.0

    ecc = FakeECC()
    is_valid, energy, duration = _calculate_transition_with_context(
        pos_k=0.0,
        speed_k=10.0,
        displacement=30.0,
        speed_k_1=12.0,
        vehicle=SimpleNamespace(max_acc=1.0, max_dec=-1.0),
        safeguard_utility=FakeSafeguardUtility(),
        ecc=ecc,
        track=object(),
    )

    assert is_valid is True
    assert energy == pytest.approx(5.0)
    assert duration == pytest.approx(30.0 / 11.0)
    assert ecc.kwargs["acc"] == pytest.approx(44.0 / 60.0)
    assert ecc.kwargs["operation_time"] == pytest.approx(30.0 / 11.0)
