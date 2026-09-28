import json

import numpy as np
import pytest
from numpy.typing import NDArray

from mtto.rl.evaluate import RLRun, run_policy

from .cases import REWARD_PRESETS, RL_CASES, SCHEDULE_CHANGE_CASES
from .drive import (
    OUTPUT_DIR,
    build_env,
    load_actions,
    manifest_context,
    require_snapshot,
)

pytestmark = pytest.mark.golden


def _assert_matches(
    run: RLRun, name: str, preset: str, actions: NDArray[np.float32]
) -> None:
    require_snapshot(name)
    context = manifest_context()
    flags = json.loads((OUTPUT_DIR / f"{name}.json").read_text("utf-8"))[preset]
    assert run.steps == flags["steps"] == len(actions), context
    assert run.termination_reason.name == flags["termination_reason"][-1], context

    with np.load(OUTPUT_DIR / f"{name}.npz") as golden:
        for field in ("position_m", "speed_mps", "time_s"):
            np.testing.assert_allclose(
                getattr(run.profile, field),
                golden[f"{preset}.{field}"],
                rtol=1e-9,
                atol=1e-12,
            )
        for field in ("propulsion_energy_kj", "levitation_energy_kj"):
            np.testing.assert_allclose(
                getattr(run.profile, field),
                golden[f"{preset}.{field}"],
                rtol=1e-7,
                atol=1e-10,
            )
        np.testing.assert_allclose(
            run.total_reward, golden[f"{preset}.reward"].sum(), rtol=1e-7, atol=1e-10
        )
        commanded = golden[f"{preset}.step_acceleration_mps2"]
        distance = golden[f"{preset}.step_distance_m"]
        duration = golden[f"{preset}.step_duration_s"]
        moving = (distance > 0) & (np.abs(commanded) >= 1e-6) & (duration > 0)
        np.testing.assert_allclose(
            run.profile.segment_acceleration_mps2[moving],
            commanded[moving],
            rtol=1e-9,
            atol=1e-9,
        )
        zero = (np.abs(commanded) < 1e-6) | (duration == 0)
        np.testing.assert_array_equal(run.profile.segment_acceleration_mps2[zero], 0.0)


@pytest.mark.parametrize("case", RL_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("preset", REWARD_PRESETS)
def test_rl_speed_profile_golden(case, preset):
    actions = load_actions(case.name)
    scripted = iter(actions)
    run = run_policy(lambda _: next(scripted), build_env(preset))
    _assert_matches(run, case.name, preset, actions)


@pytest.mark.parametrize("case", SCHEDULE_CHANGE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("preset", REWARD_PRESETS)
def test_schedule_change_speed_profile_golden(case, preset):
    actions = load_actions(case.actions_from)
    scripted = iter(actions)
    run = run_policy(lambda _: next(scripted), build_env(preset, schedule_change=case))
    _assert_matches(run, case.name, preset, actions)
