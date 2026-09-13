from dataclasses import replace

import numpy as np
import pytest

from model.ocs import SPSState, TrainService
from rl.operational_state import OperationalState, OperationalTransition, ViolationCode
from rl.reward_calculator import (
    DEFAULT_COMFORT_REWARD_SCALE,
    DEFAULT_ENERGY_REWARD_SCALE,
    DEFAULT_SURVIVAL_REWARD_SCALE,
    PUNCTUALITY_POTENTIAL_SCALE,
    PUNCTUALITY_POTENTIAL_SIGMA_S,
    RewardCalculator,
    RewardConfig,
    punctuality_potential_from_error,
)


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
    service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=20.0,
        max_acc_change=1.0,
        max_stop_error=2.0,
    )
    return RewardCalculator(
        service,
        max_episode_steps=10,
        whole_distance_m=100.0,
        max_energy_consumption_kj=100.0,
        gamma=0.995,
    )


@pytest.fixture
def punctuality_calculator(calculator: RewardCalculator) -> RewardCalculator:
    return RewardCalculator(
        calculator.train_service,
        max_episode_steps=10,
        whole_distance_m=100,
        max_energy_consumption_kj=100,
        gamma=0.9,
        reward_config=RewardConfig(
            enable_potential_safety=False,
            enable_potential_punctuality=True,
        ),
        initial_min_operation_time_s=10,
    )


def test_global_linear_slack_reference(punctuality_calculator):
    calc = punctuality_calculator
    assert [calc.reference_punctuality_slack(x) for x in (-5, 0, 50, 100, 105)] == [
        10,
        10,
        5,
        0,
        0,
    ]
    state = replace(_state(position=50), redundant_operation_time_s=5)
    assert calc.potential_punctuality(state) == 0
    calc.train_service.start_position = 100
    calc.train_service.target_position = 0
    assert [calc.reference_punctuality_slack(x) for x in (100, 50, 0, -5)] == [
        10,
        5,
        0,
        0,
    ]
    calc.train_service.schedule_time = 5
    assert calc.reference_punctuality_slack(50) == -2.5


@pytest.mark.parametrize("code", list(ViolationCode))
@pytest.mark.parametrize("length", [1, 3, 7])
def test_punctuality_shaping_keeps_terminal_next_potential(
    punctuality_calculator, code, length
):
    calc = punctuality_calculator
    state = replace(_state(position=30), redundant_operation_time_s=60)
    initial_phi = calc.potential_punctuality(state)
    total = 0.0
    for index in range(length):
        next_state = replace(
            state,
            position_m=state.position_m + 5,
            redundant_operation_time_s=state.redundant_operation_time_s - 3,
        )
        last = index == length - 1
        transition = OperationalTransition(
            state,
            next_state,
            0,
            5,
            1,
            0,
            last and code == ViolationCode.ONGOING,
            last and code != ViolationCode.ONGOING,
            code if last else ViolationCode.ONGOING,
        )
        breakdown = calc.calculate(transition)
        total += calc.gamma**index * breakdown.punctuality_shaping
        if transition.truncated:
            assert breakdown.total == pytest.approx(
                breakdown.truncation + breakdown.punctuality_shaping
            )
        state = next_state
    assert total == pytest.approx(
        -initial_phi + calc.gamma**length * calc.potential_punctuality(state)
    )


def test_external_sampling_cut_keeps_next_potential(punctuality_calculator):
    calc = punctuality_calculator
    previous = _state(position=40)
    current = _state(position=50)
    transition = OperationalTransition(
        previous, current, 0, 10, 1, 0, False, True, ViolationCode.ONGOING
    )
    assert calc.reward_punctuality_potential(transition) == pytest.approx(
        calc.gamma * calc.potential_punctuality(current)
        - calc.potential_punctuality(previous)
    )


def test_punctuality_potential_is_bounded_and_can_be_disabled(punctuality_calculator):
    calc = punctuality_calculator
    state = replace(_state(position=100), redundant_operation_time_s=1e200)
    assert calc.potential_punctuality(state) == pytest.approx(
        -PUNCTUALITY_POTENTIAL_SCALE
    )
    assert PUNCTUALITY_POTENTIAL_SIGMA_S == 20.0
    calc.reward_config = replace(calc.reward_config, enable_potential_punctuality=False)
    assert calc.potential_punctuality(state) == 0


def test_canonical_punctuality_error_potential_is_symmetric_and_bounded() -> None:
    errors = np.asarray([-1e200, -20.0, 0.0, 20.0, 1e200])
    potential = punctuality_potential_from_error(errors)

    assert isinstance(potential, np.ndarray)
    np.testing.assert_allclose(potential, potential[::-1])
    assert potential[2] == pytest.approx(0.0)
    assert np.all(potential <= 0.0)
    assert np.all(potential >= -PUNCTUALITY_POTENTIAL_SCALE)
    assert punctuality_potential_from_error(20.0) == pytest.approx(
        -PUNCTUALITY_POTENTIAL_SCALE / 2.0
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
    assert reward.survival == pytest.approx(5.0)
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


def test_truncation_keeps_potential_shaping_but_excludes_dense_objectives(
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
    assert reward.truncation == pytest.approx(-8.20)
    assert reward.total == pytest.approx(reward.truncation + reward.safety)
    assert reward.energy == reward.comfort == reward.survival == 0.0


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [(False, False), (True, False), (False, True)],
)
def test_safety_potential_uses_discounted_potential_difference(
    calculator: RewardCalculator, terminated: bool, truncated: bool
) -> None:
    previous = _state(speed=80.0, min_speed=10.0, max_speed=100.0)
    current = _state(position=1.0, speed=95.0, min_speed=10.0, max_speed=100.0)
    transition = OperationalTransition(
        previous,
        current,
        0.0,
        1.0,
        1.0,
        0.0,
        terminated,
        truncated,
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
    service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=20.0,
        max_acc_change=1.0,
        max_stop_error=2.0,
    )
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


def test_survival_reward_scale_is_configurable() -> None:
    service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=20.0,
        max_acc_change=1.0,
        max_stop_error=2.0,
    )
    calculator = RewardCalculator(
        service,
        max_episode_steps=10,
        whole_distance_m=100.0,
        max_energy_consumption_kj=100.0,
        gamma=0.995,
        reward_config=RewardConfig(survival_reward_scale=50.0),
    )
    reward = calculator.calculate(
        OperationalTransition(
            _state(),
            _state(position=10.0),
            1.0,
            10.0,
            1.0,
            0.0,
            False,
            False,
            ViolationCode.ONGOING,
        )
    )
    assert reward.survival == pytest.approx(5.0)


def test_dense_reward_scales_are_configurable() -> None:
    service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=20.0,
        max_acc_change=1.0,
        max_stop_error=2.0,
    )
    calculator = RewardCalculator(
        service,
        max_episode_steps=10,
        whole_distance_m=100.0,
        max_energy_consumption_kj=100.0,
        gamma=0.995,
        reward_config=RewardConfig(
            energy_reward_scale=30.0,
            comfort_reward_scale=10.0,
            survival_reward_scale=0.0,
        ),
    )
    reward = calculator.calculate(
        OperationalTransition(
            _state(acc=0.0),
            _state(position=10.0, energy=5.0, acc=1.0),
            1.0,
            10.0,
            1.0,
            5.0,
            False,
            False,
            ViolationCode.ONGOING,
        )
    )

    assert reward.energy == pytest.approx(-1.5)
    assert reward.comfort == pytest.approx(-1.0)
    assert reward.survival == 0.0


def test_zero_dense_reward_scales_disable_components() -> None:
    config = RewardConfig(energy_reward_scale=0.0, comfort_reward_scale=0.0)
    service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=20.0,
        max_acc_change=1.0,
        max_stop_error=2.0,
    )
    calculator = RewardCalculator(
        service,
        max_episode_steps=10,
        whole_distance_m=100.0,
        max_energy_consumption_kj=100.0,
        gamma=0.995,
        reward_config=config,
    )
    reward = calculator.calculate(
        OperationalTransition(
            _state(acc=0.0),
            _state(position=10.0, energy=5.0, acc=1.0),
            1.0,
            10.0,
            1.0,
            5.0,
            False,
            False,
            ViolationCode.ONGOING,
        )
    )

    assert reward.energy == 0.0
    assert reward.comfort == 0.0


@pytest.mark.parametrize(
    "invalid_value",
    (-1.0, float("nan"), float("inf"), float("-inf"), "invalid", None),
)
def test_reward_config_invalid_scales_fall_back_to_defaults(
    invalid_value: object,
) -> None:
    config = RewardConfig(
        energy_reward_scale=invalid_value,  # type: ignore[arg-type]
        comfort_reward_scale=invalid_value,  # type: ignore[arg-type]
        survival_reward_scale=invalid_value,  # type: ignore[arg-type]
    )

    assert config.energy_reward_scale == DEFAULT_ENERGY_REWARD_SCALE
    assert config.comfort_reward_scale == DEFAULT_COMFORT_REWARD_SCALE
    assert config.survival_reward_scale == DEFAULT_SURVIVAL_REWARD_SCALE


def test_survival_reward_zero_disables_survival() -> None:
    service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=20.0,
        max_acc_change=1.0,
        max_stop_error=2.0,
    )
    calculator = RewardCalculator(
        service,
        max_episode_steps=10,
        whole_distance_m=100.0,
        max_energy_consumption_kj=100.0,
        gamma=0.995,
        reward_config=RewardConfig(survival_reward_scale=0.0),
    )
    reward = calculator.calculate(
        OperationalTransition(
            _state(),
            _state(position=10.0),
            1.0,
            10.0,
            1.0,
            0.0,
            False,
            False,
            ViolationCode.ONGOING,
        )
    )
    assert reward.survival == 0.0
    assert RewardConfig().survival_reward_scale == 50.0
