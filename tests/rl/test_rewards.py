import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from mtto.domain.kinematics import Motion
from mtto.domain.safeguard import SPSState
from mtto.domain.scenario import Task
from mtto.rl.rewards import (
    COMFORT_REWARD_WEIGHT,
    PROGRESS_REWARD_SCALE,
    PUNCTUALITY_POTENTIAL_SCALE,
    PUNCTUALITY_POTENTIAL_SIGMA_S,
    SAFETY_RESERVE_HORIZON_STEPS,
    SAFETY_RESERVE_LOWER_SCALE,
    SAFETY_RESERVE_UPPER_SCALE,
    TERMINAL_REWARD_SCALE,
    TRUNCATION_PENALTY,
    RewardCalculator,
    RewardConfig,
    RewardNormalization,
    braking_reserve_steps,
    build_reward_config,
    punctuality_potential_from_error,
    resolve_reward_preset,
    reward_config_parameters,
    reward_preset_names,
    safety_potential,
    traction_reserve_steps,
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
    braking_reserve: float = math.inf,
    traction_reserve: float = math.inf,
) -> State:
    return State(
        s_m=position,
        v_mps=speed,
        commanded_acceleration_mps2=acc,
        t_s=time,
        slack_time_s=0.0,
        propulsion_energy_kj=energy,
        levitation_energy_kj=0.0,
        slope_pct=0.0,
        lower_limit_mps=min_speed,
        upper_limit_mps=max_speed,
        srtsp_limit_mps=max_speed,
        stop_error_m=abs(100.0 - position),
        sps=SPSState(),
        step=1,
        schedule_time_s=schedule_time_s,
        schedule_changed=schedule_changed,
        braking_reserve_steps=braking_reserve,
        traction_reserve_steps=traction_reserve,
    )


TASK = Task(
    start_position_m=0.0,
    target_position_m=100.0,
    schedule_time_s=20.0,
    max_jerk_mps3=1.0,
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
    return RewardCalculator(
        RewardNormalization(100.0, 10.0), gamma=0.995, step_time_s=1.0
    )


@pytest.fixture
def punctuality_calculator() -> RewardCalculator:
    return RewardCalculator(
        RewardNormalization(100.0, 10.0),
        gamma=0.9,
        step_time_s=1.0,
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
        if transition[1].termination_reason not in (
            None,
            TerminationReason.STOPPED_IN_ZONE,
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


def test_punctuality_potential_can_be_disabled(punctuality_calculator):
    calc = punctuality_calculator
    state = replace(_state(position=100), slack_time_s=500.0)
    assert calc.potential_punctuality(state, TASK) < 0.0
    calc.reward_config = replace(calc.reward_config, enable_potential_punctuality=False)
    assert calc.potential_punctuality(state, TASK) == 0


def test_punctuality_potential_is_quadratic_near_zero_and_linear_far_away() -> None:
    k, sigma = PUNCTUALITY_POTENTIAL_SCALE, PUNCTUALITY_POTENTIAL_SIGMA_S
    errors = np.asarray([-400.0, -20.0, 0.0, 20.0, 400.0])
    potential = punctuality_potential_from_error(errors)

    assert isinstance(potential, np.ndarray)
    np.testing.assert_allclose(potential, potential[::-1])
    assert potential[2] == pytest.approx(0.0)
    assert np.all(np.diff(potential[:3]) > 0.0)
    # Near zero it matches -K e^2 / sigma^2.
    assert punctuality_potential_from_error(0.5) == pytest.approx(
        -k * 0.25 / sigma**2, rel=1e-3
    )
    # Far away every further second costs 2K / sigma: no saturation.
    slope = punctuality_potential_from_error(401.0) - punctuality_potential_from_error(
        400.0
    )
    assert slope == pytest.approx(-2.0 * k / sigma, rel=1e-2)


def test_dense_reward_includes_energy_comfort_and_progress(
    calculator: RewardCalculator,
) -> None:
    previous = _state(acc=0.0)
    current = _state(position=10.0, energy=5.0, acc=1.0)
    transition = _transition(previous, current, 1.0, 10.0, 1.0, 5.0, None)
    reward = calculator.calculate(*transition)
    # 0.4 * 50 * (5 kJ / 100 kJ/m) / 100 m.
    assert reward.energy == pytest.approx(-0.01)
    # Jerk at the threshold (ratio 1) doubles the 0.2 per m/s^2 change.
    assert reward.comfort == pytest.approx(-0.4)
    # 10 m of the 100 m route.
    assert reward.progress == pytest.approx(5.0)
    # Ample braking reserve: the hinge potential adds no interior bias.
    assert reward.safety == 0.0
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
    assert reward.truncation == TRUNCATION_PENALTY == -80.0
    assert reward.total == pytest.approx(reward.truncation + reward.safety)
    assert reward.energy == reward.comfort == reward.progress == 0.0


@pytest.mark.parametrize(
    "termination_reason",
    [None, TerminationReason.STOPPED_IN_ZONE, TerminationReason.OVER_UPPER_LIMIT],
)
def test_safety_potential_uses_discounted_potential_difference(
    calculator: RewardCalculator, termination_reason: TerminationReason | None
) -> None:
    previous = _state(braking_reserve=2.0)
    current = _state(position=1.0, braking_reserve=0.5)
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
    horizon = SAFETY_RESERVE_HORIZON_STEPS
    phi_previous = -SAFETY_RESERVE_UPPER_SCALE * (1.0 - 2.0 / horizon) ** 2
    phi_current = -SAFETY_RESERVE_UPPER_SCALE * (1.0 - 0.5 / horizon) ** 2
    assert reward.safety == pytest.approx(calculator.gamma * phi_current - phi_previous)
    assert reward.safety < 0.0


@pytest.mark.parametrize(
    ("braking_reserve", "traction_reserve", "expected"),
    [
        # Ample reserve on both sides: no shaping away from the envelope.
        (math.inf, math.inf, 0.0),
        (SAFETY_RESERVE_HORIZON_STEPS, 10.0, 0.0),
        # Exhausted or overdrawn reserve saturates at the side's full weight.
        (0.0, math.inf, -SAFETY_RESERVE_UPPER_SCALE),
        (-5.0, math.inf, -SAFETY_RESERVE_UPPER_SCALE),
        (math.inf, -1.0, -SAFETY_RESERVE_LOWER_SCALE),
        (
            SAFETY_RESERVE_HORIZON_STEPS / 2,
            SAFETY_RESERVE_HORIZON_STEPS / 2,
            -0.25 * (SAFETY_RESERVE_UPPER_SCALE + SAFETY_RESERVE_LOWER_SCALE),
        ),
    ],
)
def test_safety_potential_is_a_bounded_hinge(
    braking_reserve: float, traction_reserve: float, expected: float
) -> None:
    assert safety_potential(braking_reserve, traction_reserve) == pytest.approx(
        expected
    )


# With full effort 1 m/s^2, one control period moves v^2 by
# 2 * (v * dt + dt^2 / 2) (twice the effective distance ds_eff).
@pytest.mark.parametrize(
    ("speed", "now", "ahead", "step_time_s", "expected"),
    [
        # Stopped: nothing to brake.
        (0.0, 0.0, 0.0, 1.0, math.inf),
        # The one-period-ahead limit binds: (15^2 - 14^2) / 29 + 1.
        (14.0, 20.0, 15.0, 1.0, 29.0 / 29.0 + 1.0),
        # A longer period drains more v^2 each: (15^2 - 14^2) / 60 + 1.
        (14.0, 20.0, 15.0, 2.0, 29.0 / 60.0 + 1.0),
        # The current limit binds once already exceeded: 2 * (10.5 + 0.125).
        (21.0, 20.0, 30.0, 0.5, (400.0 - 441.0) / 21.25),
    ],
)
def test_braking_reserve_counts_current_and_one_period_ahead_limit(
    speed: float, now: float, ahead: float, step_time_s: float, expected: float
) -> None:
    assert braking_reserve_steps(speed, now, ahead, 1.0, step_time_s) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("speed", "now", "ahead", "step_time_s", "expected"),
    [
        # No positive lower limit: no traction constraint.
        (5.0, 0.0, 0.0, 1.0, math.inf),
        # Only the one-period-ahead limit is positive: (11^2 - 12^2) / 23 + 1.
        (11.0, 0.0, 12.0, 1.0, -23.0 / 23.0 + 1.0),
        (11.0, 10.0, 0.0, 2.0, 21.0 / 48.0),
        # A stopped train still has a finite, low-speed-safe denominator.
        (0.0, 10.0, 0.0, 1.0, -100.0),
    ],
)
def test_traction_reserve_only_counts_positive_lower_limits(
    speed: float, now: float, ahead: float, step_time_s: float, expected: float
) -> None:
    assert traction_reserve_steps(speed, now, ahead, 1.0, step_time_s) == pytest.approx(
        expected
    )


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
        RewardNormalization(100.0, 10.0),
        gamma=0.995,
        step_time_s=1.0,
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


def test_reward_preset_names_include_pirs_component_profiles() -> None:
    assert reward_preset_names() == (
        "basic",
        "basic_safety",
        "basic_punctuality",
        "basic_safety_punctuality",
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
    assert pirs["progress_reward_scale"] == 50.0
    assert pirs["energy_reward_weight"] == 0.4
    assert pirs["comfort_reward_weight"] == 0.2
    assert pirs["terminal_reward_scale"] == 100.0
    assert pirs["truncation_penalty"] == -80.0
    assert pirs["safety_reserve_horizon_steps"] == 3.0
    assert pirs["safety_reserve_upper_scale"] == 3.0
    assert pirs["safety_reserve_lower_scale"] == 1.0
    assert pirs["stopping_score_power"] == 4.0
    assert set(pirs) == {
        "progress_reward_scale",
        "energy_reward_weight",
        "comfort_reward_weight",
        "comfort_penalty",
        "terminal_reward_scale",
        "truncation_penalty",
        "enable_potential_safety",
        "safety_reserve_horizon_steps",
        "safety_reserve_upper_scale",
        "safety_reserve_lower_scale",
        "enable_potential_punctuality",
        "stopping_score_power",
        "punctuality_potential_scale",
        "punctuality_potential_sigma_s",
        "potential_transition_formula",
        "terminal_next_potential",
    }


def test_progress_sums_to_scale_over_the_route(calculator: RewardCalculator) -> None:
    positions = (0.0, 30.0, 45.0, 80.0, 99.5, 100.0)
    total = sum(
        calculator.calculate(
            *_transition(
                _state(position=begin),
                _state(position=end),
                0.0,
                end - begin,
                1.0,
                0.0,
                None,
            )
        ).progress
        for begin, end in itertools.pairwise(positions)
    )
    assert total == pytest.approx(PROGRESS_REWARD_SCALE)


@pytest.mark.parametrize(
    ("begin_m", "end_m", "reason", "expected"),
    [
        # Crossing the target counts only the net approach.
        (90.0, 105.0, None, 0.05 * PROGRESS_REWARD_SCALE),
        (95.0, 105.0, None, 0.0),
        # Running on past the target is penalised.
        (105.0, 110.0, None, -0.05 * PROGRESS_REWARD_SCALE),
        (
            100.0,
            101.0,
            TerminationReason.STOPPED_IN_ZONE,
            -0.01 * PROGRESS_REWARD_SCALE,
        ),
        # A stop in the zone still earns its progress.
        (98.0, 100.5, TerminationReason.STOPPED_IN_ZONE, 0.015 * PROGRESS_REWARD_SCALE),
        # Safety and failed-stop terminations take the truncation branch.
        (40.0, 50.0, TerminationReason.OVER_UPPER_LIMIT, 0.0),
        (40.0, 50.0, TerminationReason.STOPPED_SHORT, 0.0),
    ],
)
def test_progress_is_signed_distance_to_the_target(
    calculator: RewardCalculator,
    begin_m: float,
    end_m: float,
    reason: TerminationReason | None,
    expected: float,
) -> None:
    reward = calculator.calculate(
        *_transition(
            _state(position=begin_m),
            _state(position=end_m),
            0.0,
            end_m - begin_m,
            1.0,
            0.0,
            reason,
        )
    )
    assert reward.progress == pytest.approx(expected)


@pytest.mark.parametrize(
    ("jerk_ratio", "multiplier"),
    [(0.0, 1.0), (0.5, 1.25), (1.0, 2.0), (2.0, 5.0)],
)
def test_comfort_is_total_variation_scaled_by_the_jerk_ratio(
    calculator: RewardCalculator, jerk_ratio: float, multiplier: float
) -> None:
    """Cost per m/s^2 of change doubles at the jerk limit and keeps growing."""
    delta_acc = jerk_ratio * TASK.max_jerk_mps3
    reward = calculator.calculate(
        *_transition(
            _state(acc=0.0),
            _state(position=10.0, acc=delta_acc),
            delta_acc,
            10.0,
            1.0,
            0.0,
            None,
        )
    )
    assert reward.comfort == pytest.approx(
        -COMFORT_REWARD_WEIGHT * delta_acc * multiplier, abs=1e-12
    )


def test_comfort_does_not_depend_on_how_fast_the_acceleration_changes(
    calculator: RewardCalculator,
) -> None:
    """One 0.6 m/s^2 change costs less than the same change in two halves' sum
    only through the jerk multiplier: below the limit the totals nearly agree."""

    def cost(previous_acc: float, acc: float) -> float:
        return calculator.calculate(
            *_transition(
                _state(acc=previous_acc),
                _state(position=10.0, acc=acc),
                acc,
                10.0,
                1.0,
                0.0,
                None,
            )
        ).comfort

    slow = cost(0.0, 0.1) * 6
    fast = cost(0.0, 0.6)
    assert slow == pytest.approx(-COMFORT_REWARD_WEIGHT * 0.6, rel=0.02)
    assert fast < slow


@pytest.mark.parametrize(
    ("position_m", "time_s"),
    [(100.0, 20.0), (101.0, 20.0), (102.5, 20.0), (100.0, 40.0), (103.0, 80.0)],
)
def test_terminal_reward_splits_stopping_and_punctuality(
    calculator: RewardCalculator, position_m: float, time_s: float
) -> None:
    current = _state(position=position_m, time=time_s)
    reward = calculator.calculate(
        *_transition(
            _state(position=90.0),
            current,
            0.0,
            position_m - 90.0,
            1.0,
            0.0,
            TerminationReason.STOPPED_IN_ZONE,
        )
    )
    stopping = calculator.stopping_score(current.stop_error_m, TASK)
    punctuality = calculator.punctuality_score(time_s, TASK.schedule_time_s)
    assert reward.terminal_stopping == pytest.approx(
        TERMINAL_REWARD_SCALE * 0.3 * stopping
    )
    assert reward.terminal_stopping + reward.terminal_punctuality == pytest.approx(
        TERMINAL_REWARD_SCALE * stopping * (0.3 + 0.7 * punctuality)
    )
    if stopping == punctuality == 1.0:
        assert reward.terminal_stopping + reward.terminal_punctuality == 100.0
