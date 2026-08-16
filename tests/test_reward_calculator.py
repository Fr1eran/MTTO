import pytest

from model.ocs import SPSState, TrainService
from rl.operational_state import OperationalState, OperationalTransition, ViolationCode
from rl.reward_calculator import RewardCalculator, RewardConfig


def _state(
    *,
    position: float = 0.0,
    speed: float = 0.0,
    min_speed: float = 0.0,
    max_speed: float = 100.0,
    time: float = 0.0,
    energy: float = 0.0,
    acc: float = 0.0,
) -> OperationalState:
    return OperationalState(
        position_m=position,
        speed_mps=speed,
        acceleration_mps2=acc,
        operation_time_s=time,
        redundant_operation_time_s=0.0,
        energy_consumption_kj=energy,
        slope_permille=0.0,
        min_speed_mps=min_speed,
        max_speed_mps=max_speed,
        stop_error_m=abs(100.0 - position),
        sps_state=SPSState(),
        step_count=1,
    )


@pytest.fixture
def calculator() -> RewardCalculator:
    service = TrainService(0.0, 0.0, 100.0, 20.0, 1.0, 1.0, 2.0)
    return RewardCalculator(
        service,
        max_episode_steps=10,
        whole_distance_m=100.0,
        max_energy_consumption_kj=100.0,
        gamma=0.995,
    )


def test_dense_reward_includes_energy_comfort_and_survival(
    calculator: RewardCalculator,
) -> None:
    previous = _state(acc=0.0)
    current = _state(position=10.0, energy=5.0, acc=1.0)
    transition = OperationalTransition(
        previous, current, 1.0, 10.0, 1.0, 5.0, False, False, ViolationCode.ONGOING
    )
    reward = calculator.calculate(transition)
    assert reward.energy == pytest.approx(-0.75)
    assert reward.comfort == pytest.approx(-2.0)
    assert reward.survival == pytest.approx(10.0)
    assert reward.safety == pytest.approx(0.0)
    assert reward.terminal_stopping == 0.0
    assert reward.terminal_punctuality == 0.0


def test_terminal_punctuality_is_rewarded_only_on_termination(
    calculator: RewardCalculator,
) -> None:
    previous = _state()
    on_time = _state(position=100.0, time=20.0)
    late = _state(position=100.0, time=80.0)
    on_time_reward = calculator.calculate(
        OperationalTransition(
            previous, on_time, 0.0, 0.0, 0.0, 0.0, True, False, ViolationCode.ONGOING
        )
    )
    late_reward = calculator.calculate(
        OperationalTransition(
            previous, late, 0.0, 0.0, 0.0, 0.0, True, False, ViolationCode.ONGOING
        )
    )
    assert on_time_reward.terminal_punctuality > late_reward.terminal_punctuality


def test_truncation_excludes_all_other_reward_components(
    calculator: RewardCalculator,
) -> None:
    previous = _state()
    current = _state(position=20.0, energy=99.0, acc=1.0)
    reward = calculator.calculate(
        OperationalTransition(
            previous,
            current,
            1.0,
            20.0,
            1.0,
            99.0,
            False,
            True,
            ViolationCode.SPEED_HIGH,
        )
    )
    assert reward.truncation == pytest.approx(-1.64)
    assert reward.total == reward.truncation
    assert reward.energy == reward.comfort == reward.survival == 0.0


def test_safety_potential_uses_v3_pbrs_difference(calculator: RewardCalculator) -> None:
    previous = _state(speed=80.0, min_speed=10.0, max_speed=100.0)
    current = _state(position=1.0, speed=95.0, min_speed=10.0, max_speed=100.0)
    transition = OperationalTransition(
        previous,
        current,
        0.0,
        1.0,
        1.0,
        0.0,
        False,
        False,
        ViolationCode.ONGOING,
    )
    reward = calculator.calculate(transition)
    expected = calculator.gamma * calculator._potential_safety(
        speed_mps=current.speed_mps,
        min_speed_mps=current.min_speed_mps,
        max_speed_mps=current.max_speed_mps,
    ) - calculator._potential_safety(
        speed_mps=previous.speed_mps,
        min_speed_mps=previous.min_speed_mps,
        max_speed_mps=previous.max_speed_mps,
    )
    assert reward.safety == pytest.approx(expected)


def test_terminal_stopping_is_rewarded_only_on_termination(
    calculator: RewardCalculator,
) -> None:
    previous = _state(position=90.0)
    current = _state(position=100.0)
    terminal_reward = calculator.calculate(
        OperationalTransition(
            previous,
            current,
            0.0,
            10.0,
            1.0,
            0.0,
            True,
            False,
            ViolationCode.ONGOING,
        )
    )
    dense_reward = calculator.calculate(
        OperationalTransition(
            previous,
            current,
            0.0,
            10.0,
            1.0,
            0.0,
            False,
            False,
            ViolationCode.ONGOING,
        )
    )

    assert terminal_reward.terminal_stopping > 0.0
    assert dense_reward.terminal_stopping == 0.0


def test_safety_potential_can_be_disabled() -> None:
    service = TrainService(0.0, 0.0, 100.0, 20.0, 1.0, 1.0, 2.0)
    calculator = RewardCalculator(
        service,
        max_episode_steps=10,
        whole_distance_m=100.0,
        max_energy_consumption_kj=100.0,
        gamma=0.995,
        reward_config=RewardConfig(enable_potential_safety=False),
    )
    reward = calculator.calculate(
        OperationalTransition(
            _state(position=-1000.0, speed=20.0),
            _state(position=-50.0, speed=5.0),
            0.0,
            950.0,
            1.0,
            0.0,
            False,
            False,
            ViolationCode.ONGOING,
        )
    )
    assert reward.safety == 0.0
