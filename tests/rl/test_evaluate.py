from itertools import chain, repeat

import numpy as np
import pytest

from mtto.domain.energy import segment_energy
from mtto.rl.env import make_env
from mtto.rl.evaluate import run_policy
from mtto.rl.rewards import build_reward_config
from mtto.rl.state import TerminationReason
from mtto.workflows.train import build_env_references


@pytest.fixture(scope="module")
def env(paper_scenario, paper_task):
    lookup, normalization = build_env_references(paper_scenario, paper_task, 30.0)
    return make_env(
        scenario=paper_scenario,
        task=paper_task,
        gamma=0.998,
        srtsp_lookup=lookup,
        normalization=normalization,
        step_distance=30.0,
        reward_config=build_reward_config("basic_safety_punctuality"),
    )


@pytest.mark.parametrize(
    ("actions", "kind"),
    [
        ([np.float32(0.5), np.float32(5e-7)], "tiny"),
        ([np.float32(-1.0)], "zero_start"),
        ([np.float32(0.5), np.float32(-1.0)], "early_stop"),
    ],
)
def test_acceleration_boundaries(env, actions, kind):
    scripted = iter(chain(actions, repeat(np.float32(-1.0))))
    before = env.state
    run = run_policy(lambda _: next(scripted), env)
    assert env.state is before

    if kind == "zero_start":
        assert run.steps == 1
        assert run.termination_reason == TerminationReason.STOPPED_SHORT
        np.testing.assert_array_equal(run.profile.position_m, [before.s_m, before.s_m])
        np.testing.assert_array_equal(run.profile.time_s, [0.0, 0.0])
        np.testing.assert_array_equal(run.profile.segment_acceleration_mps2, [0.0])
        return

    first = env.transition(
        env.initial_state(), env.observation_builder.denormalize_action(0.5)
    )
    if kind == "tiny":
        command = float(np.float32(5e-7))
        result = env.transition(first.next_state, command)
        assert 0 < abs(result.commanded_acceleration_mps2) < 1e-6
        assert result.commanded_acceleration_mps2 == command
        assert run.profile.segment_acceleration_mps2[1] == 0.0
        propulsion, levitation = segment_energy(
            env.scenario.energy,
            env.vehicle,
            env.track,
            begin_pos=first.next_state.s_m,
            begin_speed=first.next_state.v_mps,
            acc=command,
            distance=result.motion.distance_m,
            direction=1,
            operation_time=result.motion.duration_s,
        )
        assert run.profile.propulsion_energy_kj[2] - run.profile.propulsion_energy_kj[
            1
        ] == pytest.approx(propulsion)
        assert run.profile.levitation_energy_kj[2] - run.profile.levitation_energy_kj[
            1
        ] == pytest.approx(levitation)
    else:
        result = env.transition(first.next_state, -1.0)
        assert result.motion.distance_m < env.step_distance
        assert run.profile.speed_mps[-1] == 0.0
        np.testing.assert_allclose(
            run.profile.segment_acceleration_mps2[-1],
            result.commanded_acceleration_mps2,
            rtol=1e-9,
            atol=1e-9,
        )
