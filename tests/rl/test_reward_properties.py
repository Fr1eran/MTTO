"""Reward properties on the paper scenario, driven through the real env."""

import itertools

import pytest

from mtto.domain.safeguard import SPSState, current_stopping_point
from mtto.rl.env import MTTOEnv
from mtto.rl.rewards import (
    COMFORT_REWARD_WEIGHT,
    ENERGY_REWARD_WEIGHT,
    PROGRESS_REWARD_SCALE,
    TRUNCATION_PENALTY,
    RewardConfig,
    TerminationReason,
)
from mtto.workflows.train import build_env_references
from paper.figures import load_paper_scenario, load_paper_task

SCENARIO = load_paper_scenario()
TASK = load_paper_task()
LOOKUP, NORMALIZATION = build_env_references(SCENARIO, TASK)
# Start, and points in different slope segments.
POSITIONS = (135.0, 5000.0, 17100.0, 21100.0)
SPEEDS = (0.5, 2.0, 10.0, 60.0, 130.0)
ACCELERATIONS = (-1.0, -0.3, 0.0, 0.5, 1.0)


def _env(step_time_s: float, reward_config: RewardConfig | None = None) -> MTTOEnv:
    return MTTOEnv(
        SCENARIO,
        TASK,
        gamma=0.998,
        step_time_s=step_time_s,
        srtsp_lookup=LOOKUP,
        normalization=NORMALIZATION,
        reward_config=reward_config,
    )


def _state(env: MTTOEnv, position_m: float, speed_mps: float, previous_acc: float):
    return env._build_state(
        s_m=position_m,
        v_mps=speed_mps,
        commanded_acceleration_mps2=previous_acc,
        t_s=0.0,
        propulsion_energy_kj=0.0,
        levitation_energy_kj=0.0,
        step=1,
        sps=SPSState(
            target_stopping_point_index=current_stopping_point(
                env.safeguard, position_m, speed_mps
            )
        ),
        schedule_time_s=TASK.schedule_time_s,
        schedule_changed=False,
    )


def test_peak_propulsion_energy_per_metre_matches_the_line() -> None:
    assert 400.0 <= NORMALIZATION.peak_propulsion_kj_per_m <= 460.0


@pytest.mark.parametrize("step_time_s", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("position_m", POSITIONS)
def test_every_step_towards_the_target_is_dominated_by_progress(
    step_time_s: float, position_m: float
) -> None:
    env = _env(step_time_s)
    checked = 0
    for speed, acc, previous_acc in itertools.product(
        SPEEDS, ACCELERATIONS, (-1.0, 1.0)
    ):
        state = _state(env, position_m, speed, previous_acc)
        result = env.transition(state, acc)
        if result.termination_reason is not None:
            continue
        reward = env.reward_calculator.calculate(state, result, TASK)
        checked += 1
        assert reward.progress > 0.0
        distance_share = result.motion.distance_m / (
            TASK.target_position_m - TASK.start_position_m
        )
        assert reward.energy >= (
            -ENERGY_REWARD_WEIGHT * PROGRESS_REWARD_SCALE * distance_share - 1e-12
        )
        assert (
            reward.progress + reward.energy
            >= (1.0 - ENERGY_REWARD_WEIGHT) * reward.progress - 1e-12
        )
        # Comfort is charged on the acceleration change alone: never positive
        # and at most doubled while the jerk stays within the limit.
        delta_acc = abs(acc - previous_acc)
        assert reward.comfort <= 0.0
        if delta_acc <= TASK.max_jerk_mps3 * step_time_s:
            assert reward.comfort >= -2.0 * COMFORT_REWARD_WEIGHT * delta_acc - 1e-12
    assert checked > 0


@pytest.mark.parametrize("step_time_s", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("position_m", POSITIONS)
def test_step_propulsion_energy_stays_below_the_per_metre_peak(
    step_time_s: float, position_m: float
) -> None:
    env = _env(step_time_s)
    for speed, acc in itertools.product((0.0, *SPEEDS), ACCELERATIONS):
        result = env.transition(_state(env, position_m, speed, 0.0), acc)
        assert (
            result.propulsion_delta_kj
            <= NORMALIZATION.peak_propulsion_kj_per_m * result.motion.distance_m + 1e-9
        )


def test_failed_terminations_cost_the_truncation_penalty() -> None:
    env = _env(1.0, RewardConfig(enable_potential_safety=True))
    start = env.initial_state()
    braking = env.transition(start, -1.0)
    assert braking.termination_reason is TerminationReason.STOPPED_SHORT
    reward = env.reward_calculator.calculate(start, braking, TASK)
    assert reward.safety == pytest.approx(0.0)
    assert reward.total == pytest.approx(TRUNCATION_PENALTY)

    near_target = _state(env, TASK.target_position_m - 3.0, 20.0, 0.0)
    overspeed = env.transition(near_target, 0.0)
    assert overspeed.termination_reason not in (
        None,
        TerminationReason.STOPPED_IN_ZONE,
    )
    reward = env.reward_calculator.calculate(near_target, overspeed, TASK)
    assert reward.truncation == TRUNCATION_PENALTY
    assert reward.total == pytest.approx(
        TRUNCATION_PENALTY + reward.safety + reward.punctuality_shaping
    )
