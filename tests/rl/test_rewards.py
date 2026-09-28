import math
from dataclasses import replace

import numpy as np
import pytest

from mtto.domain.kinematics import Motion
from mtto.domain.safeguard import SPSState
from mtto.domain.scenario import Task
from mtto.rl.rewards import (
    LI_GOAL_REWARD_SCALE,
    PUNCTUALITY_POTENTIAL_SCALE,
    PUNCTUALITY_POTENTIAL_SIGMA_S,
    RewardCalculator,
    RewardConfig,
    RewardNormalization,
    build_reward_config,
    punctuality_potential_from_error,
    resolve_reward_preset,
    reward_config_parameters,
    reward_preset_names,
)
from mtto.rl.state import (
    State,
    StepResult,
    TerminationReason,
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
    schedule_time_s: float = 20.0,
    schedule_changed: bool = False,
) -> State:
    return State(
        s_m=position,
        v_mps=speed,
        commanded_acceleration_mps2=acc,
        t_s=time,
        slack_time_s=0.0,
        propulsion_energy_kj=energy,
        levitation_energy_kj=0.0,
        slope_permille=0.0,
        lower_limit_mps=min_speed,
        upper_limit_mps=max_speed,
        srtsp_limit_mps=max_speed,
        stop_error_m=abs(100.0 - position),
        sps=SPSState(),
        step=1,
        schedule_time_s=schedule_time_s,
        schedule_changed=schedule_changed,
    )


TASK = Task(
    start_position_m=0.0,
    target_position_m=100.0,
    schedule_time_s=20.0,
    max_acc_change=1.0,
    max_stop_error_m=2.0,
    max_arr_time_error_s=10.0,
)


def _transition(previous, current, acceleration, distance, duration, energy, reason):
    result = StepResult(
        step_end_state=current,
        next_state=current,
        commanded_acceleration_mps2=acceleration,
        motion=Motion(current.v_mps, distance, duration),
        propulsion_delta_kj=energy,
        levitation_delta_kj=0.0,
        termination_reason=reason,
        violation=None,
    )
    return previous, result, TASK


@pytest.fixture
def calculator() -> RewardCalculator:
    return RewardCalculator(RewardNormalization(100.0, 10.0, 10), gamma=0.995)


@pytest.fixture
def punctuality_calculator() -> RewardCalculator:
    return RewardCalculator(
        RewardNormalization(100.0, 10.0, 10),
        gamma=0.9,
        reward_config=RewardConfig(
            enable_potential_safety=False,
            enable_potential_punctuality=True,
        ),
    )


def test_global_linear_slack_reference(punctuality_calculator):
    calc = punctuality_calculator
    assert [
        calc.reference_punctuality_slack(x, schedule_time_s=20.0, task=TASK)
        for x in (-5, 0, 50, 100, 105)
    ] == [
        10,
        10,
        5,
        0,
        0,
    ]
    state = replace(
        _state(position=50),
        schedule_time_s=20.0,
        slack_time_s=5,
    )
    assert calc.potential_punctuality(state, TASK) == 0
    assert calc.reference_punctuality_slack(50, schedule_time_s=5.0, task=TASK) == -2.5


@pytest.mark.parametrize("reason_code", [None, *list(TerminationReason)])
@pytest.mark.parametrize("length", [1, 3, 7])
def test_punctuality_shaping_keeps_terminal_next_potential(
    punctuality_calculator, reason_code, length
):
    calc = punctuality_calculator
    state = replace(_state(position=30), slack_time_s=60)
    initial_phi = calc.potential_punctuality(state, TASK)
    total = 0.0
    for index in range(length):
        next_state = replace(
            state,
            s_m=state.s_m + 5,
            slack_time_s=state.slack_time_s - 3,
        )
        last = index == length - 1
        transition = _transition(
            state,
            next_state,
            0,
            5,
            1,
            0,
            reason_code if last else None,
        )
        breakdown = calc.calculate(*transition)
        total += calc.gamma**index * breakdown.punctuality_shaping
        if (
            transition[1].termination_reason is not None
            and transition[1].termination_reason
            is not TerminationReason.STOPPED_IN_ZONE
        ):
            assert breakdown.total == pytest.approx(
                breakdown.truncation + breakdown.punctuality_shaping
            )
        state = next_state
    assert total == pytest.approx(
        -initial_phi + calc.gamma**length * calc.potential_punctuality(state, TASK)
    )


def test_external_sampling_cut_keeps_next_potential(punctuality_calculator):
    calc = punctuality_calculator
    previous = _state(position=40)
    current = _state(position=50)
    transition = _transition(previous, current, 0, 10, 1, 0, None)
    assert calc.reward_punctuality_potential(*transition) == pytest.approx(
        calc.gamma * calc.potential_punctuality(current, TASK)
        - calc.potential_punctuality(previous, TASK)
    )


def test_punctuality_potential_is_bounded_and_can_be_disabled(punctuality_calculator):
    calc = punctuality_calculator
    state = replace(_state(position=100), slack_time_s=1e200)
    assert calc.potential_punctuality(state, TASK) == pytest.approx(
        -PUNCTUALITY_POTENTIAL_SCALE
    )
    assert PUNCTUALITY_POTENTIAL_SIGMA_S == 20.0
    calc.reward_config = replace(calc.reward_config, enable_potential_punctuality=False)
    assert calc.potential_punctuality(state, TASK) == 0


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
    transition = _transition(previous, current, 1.0, 10.0, 1.0, 5.0, None)
    reward = calculator.calculate(*transition)
    assert reward.energy == pytest.approx(-0.75)
    assert reward.comfort == pytest.approx(-2.0)
    assert reward.survival == pytest.approx(5.0)
    distant_upper_potential = -1.0 / (1.0 + math.exp(8.0))
    assert reward.safety == pytest.approx(
        (calculator.gamma - 1.0) * distant_upper_potential
    )
    assert reward.terminal_stopping == 0.0
    assert reward.terminal_punctuality == 0.0


def test_terminal_punctuality_is_rewarded_only_on_termination(
    calculator: RewardCalculator,
) -> None:
    previous = _state()
    on_time = _state(position=100.0, time=20.0)
    late = _state(position=100.0, time=80.0)
    on_time_reward = calculator.calculate(
        *_transition(
            previous,
            on_time,
            0.0,
            0.0,
            0.0,
            0.0,
            TerminationReason.STOPPED_IN_ZONE,
        )
    )
    late_reward = calculator.calculate(
        *_transition(
            previous,
            late,
            0.0,
            0.0,
            0.0,
            0.0,
            TerminationReason.STOPPED_IN_ZONE,
        )
    )
    assert on_time_reward.terminal_punctuality > late_reward.terminal_punctuality


def test_truncation_keeps_potential_shaping_but_excludes_dense_objectives(
    calculator: RewardCalculator,
) -> None:
    previous = _state()
    current = _state(position=20.0, energy=99.0, acc=1.0)
    reward = calculator.calculate(
        *_transition(
            previous,
            current,
            1.0,
            20.0,
            1.0,
            99.0,
            TerminationReason.OVER_UPPER_LIMIT,
        )
    )
    assert reward.truncation == pytest.approx(-8.20)
    assert reward.total == pytest.approx(reward.truncation + reward.safety)
    assert reward.energy == reward.comfort == reward.survival == 0.0


@pytest.mark.parametrize(
    "termination_reason",
    [None, TerminationReason.STOPPED_IN_ZONE, TerminationReason.OVER_UPPER_LIMIT],
)
def test_safety_potential_uses_discounted_potential_difference(
    calculator: RewardCalculator, termination_reason: TerminationReason | None
) -> None:
    previous = _state(speed=80.0, min_speed=10.0, max_speed=100.0)
    current = _state(position=1.0, speed=95.0, min_speed=10.0, max_speed=100.0)
    transition = _transition(
        previous,
        current,
        0.0,
        1.0,
        1.0,
        0.0,
        termination_reason,
    )
    reward = calculator.calculate(*transition)
    phi_previous = -0.5 * (
        2.0 / (1.0 + math.exp(8.0 * 20.0 / 90.0))
        + 2.0 / (1.0 + math.exp(8.0 * 70.0 / 90.0))
    )
    phi_current = -0.5 * (
        2.0 / (1.0 + math.exp(8.0 * 5.0 / 90.0))
        + 2.0 / (1.0 + math.exp(8.0 * 85.0 / 90.0))
    )
    assert reward.safety == pytest.approx(calculator.gamma * phi_current - phi_previous)

    potential = calculator._potential_safety
    assert potential(
        speed_mps=15.0, min_speed_mps=10.0, max_speed_mps=20.0
    ) == pytest.approx(
        potential(speed_mps=60.0, min_speed_mps=10.0, max_speed_mps=110.0)
    )
    assert potential(
        speed_mps=10.0, min_speed_mps=10.0, max_speed_mps=20.0
    ) == pytest.approx(-0.5 * (1.0 + 2.0 / (1.0 + math.exp(8.0))))
    assert potential(
        speed_mps=10.0, min_speed_mps=0.0, max_speed_mps=10.0
    ) == pytest.approx(-0.5)
    assert potential(
        speed_mps=0.0, min_speed_mps=0.0, max_speed_mps=0.0
    ) == pytest.approx(-0.5)
    assert potential(
        speed_mps=10.25, min_speed_mps=10.0, max_speed_mps=10.5
    ) == pytest.approx(-2.0 / (1.0 + math.exp(2.0)))
    assert potential(
        speed_mps=-1e6, min_speed_mps=10.0, max_speed_mps=20.0
    ) == pytest.approx(-1.0)


def test_terminal_stopping_is_rewarded_only_on_termination(
    calculator: RewardCalculator,
) -> None:
    previous = _state(position=90.0)
    current = _state(position=100.0)
    terminal_reward = calculator.calculate(
        *_transition(
            previous,
            current,
            0.0,
            10.0,
            1.0,
            0.0,
            TerminationReason.STOPPED_IN_ZONE,
        )
    )
    dense_reward = calculator.calculate(
        *_transition(
            previous,
            current,
            0.0,
            10.0,
            1.0,
            0.0,
            None,
        )
    )

    assert terminal_reward.terminal_stopping > 0.0
    assert dense_reward.terminal_stopping == 0.0


def test_safety_potential_can_be_disabled() -> None:
    calculator = RewardCalculator(
        RewardNormalization(100.0, 10.0, 10),
        gamma=0.995,
        reward_config=RewardConfig(enable_potential_safety=False),
    )
    reward = calculator.calculate(
        *_transition(
            _state(position=-1000.0, speed=20.0),
            _state(position=-50.0, speed=5.0),
            0.0,
            950.0,
            1.0,
            0.0,
            None,
        )
    )
    assert reward.safety == 0.0


@pytest.mark.parametrize(
    ("arrival_time_s", "stop_position_m", "punctuality", "stopping"),
    ((25.0, 100.0, 50.0, 0.0), (35.0, 97.0, -0.4 * 15.0, -3.0)),
)
def test_li_goal_reward_scales_published_values(
    calculator: RewardCalculator,
    arrival_time_s: float,
    stop_position_m: float,
    punctuality: float,
    stopping: float,
) -> None:
    li = RewardCalculator(
        RewardNormalization(100.0, 10.0, 10),
        gamma=0.995,
        reward_config=RewardConfig(reward_scheme="li2023_scaled"),
        train_mass_kg=1000.0,
    )
    reward = li.calculate(
        *_transition(
            _state(position=90.0, speed=1.0),
            _state(position=stop_position_m, time=arrival_time_s, energy=5.0),
            0.0,
            10.0,
            1.0,
            5.0,
            TerminationReason.STOPPED_IN_ZONE,
        )
    )

    # Published goal-state values of Li et al. (2023) multiplied by 5.
    assert LI_GOAL_REWARD_SCALE == pytest.approx(5.0)
    assert reward.survival == pytest.approx(2.5 + 5.0 * 350.0)
    assert reward.terminal_punctuality == pytest.approx(5.0 * punctuality)
    assert reward.terminal_stopping == pytest.approx(5.0 * stopping)
    # r_E * E with E = 5 kJ * 1000 / (1000 kg * 100 m).
    assert reward.energy == pytest.approx(5.0 * -0.6 * 0.05)


def test_reward_preset_names_include_pirs_component_profiles() -> None:
    assert reward_preset_names() == (
        "basic",
        "basic_safety",
        "basic_punctuality",
        "basic_safety_punctuality",
        "li2023_scaled",
    )


def test_reward_presets_keep_energy_comfort_and_toggle_potential_shaping() -> None:
    expected_flags = {
        "basic": False,
        "basic_safety": True,
    }
    for profile_name, expected in expected_flags.items():
        reward_config = build_reward_config(profile_name)
        assert reward_config.enable_potential_safety is expected


def test_reward_preset_owns_the_runtime_config() -> None:
    preset = resolve_reward_preset("basic_safety")

    assert preset.config is build_reward_config("basic_safety")
    assert preset.enabled_shaping_components() == ("safety",)


def test_reward_metadata_records_fixed_reward_magnitudes() -> None:
    pirs = reward_config_parameters(build_reward_config("basic_safety_punctuality"))
    assert pirs["energy_reward_scale"] == 15.0
    assert pirs["comfort_reward_scale"] == 20.0
    assert pirs["survival_reward_scale"] == 50.0
    assert pirs["safety_potential_scale"] == 0.5
    assert pirs["safety_potential_steepness"] == 8.0
    assert pirs["goal_reward_scale"] == 1.0
    baseline = reward_config_parameters(build_reward_config("li2023_scaled"))
    assert baseline["reward_scheme"] == "li2023_scaled"
    assert baseline["goal_reward_scale"] == pytest.approx(5.0)
