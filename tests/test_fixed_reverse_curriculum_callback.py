from types import SimpleNamespace

import numpy as np
import pytest

from rl.callbacks import FixedReverseCurriculumCallback
from rl.experiment_utils import resolve_curriculum_profile


class _RecordingVecEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray, int]] = []

    def env_method(self, method_name, weights, *, version):
        self.calls.append((method_name, np.asarray(weights), version))


def _build_callback() -> FixedReverseCurriculumCallback:
    return FixedReverseCurriculumCallback(
        remaining_distances_m=np.asarray([6000.0, 3000.0, 1000.0, 300.0]),
        whole_distance_m=6000.0,
        total_timesteps=1000,
        profile=resolve_curriculum_profile("fixed_reverse"),
    )


def test_fixed_reverse_schedule_has_expected_phases() -> None:
    callback = _build_callback()

    np.testing.assert_allclose(
        callback.initial_weights(), [0.15, 0.85 / 3, 0.85 / 3, 0.85 / 3]
    )
    np.testing.assert_allclose(
        callback._weights_for_progress(0.10), [0.15, 0.85 / 3, 0.85 / 3, 0.85 / 3]
    )
    mixed = callback._weights_for_progress(0.50)
    assert mixed[0] == pytest.approx(0.68125)
    assert mixed.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(
        callback._weights_for_progress(0.55), [1.0, 0.0, 0.0, 0.0]
    )


def test_first_rollout_keeps_preinjected_distribution_then_broadcasts() -> None:
    callback = _build_callback()
    env = _RecordingVecEnv()
    callback.model = SimpleNamespace(get_env=lambda: env)

    callback.num_timesteps = 200
    callback._on_rollout_start()
    assert env.calls == []

    callback.num_timesteps = 450
    callback._on_rollout_start()
    assert len(env.calls) == 1
    method_name, weights, version = env.calls[0]
    assert method_name == "set_reference_initial_state_distribution"
    assert version == 1
    np.testing.assert_allclose(weights, callback._weights_for_progress(0.45))


def test_fixed_reverse_fails_when_initial_range_has_no_node() -> None:
    with pytest.raises(ValueError, match="contains no eligible"):
        FixedReverseCurriculumCallback(
            remaining_distances_m=np.asarray([10.0, 0.0]),
            whole_distance_m=10.0,
            total_timesteps=100,
            profile=resolve_curriculum_profile("fixed_reverse"),
        )
