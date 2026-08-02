from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from rl.callbacks import SPDLReferenceCurriculumCallback, _SPDLContextSample
from rl.experiment_utils import resolve_curriculum_profile


class _Policy:
    def obs_to_tensor(self, observations):
        return th.as_tensor(observations, dtype=th.float32), True

    def predict_values(self, observation_tensor):
        return observation_tensor[:, :1]


class _VecEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray, int]] = []

    def normalize_obs(self, observations):
        return np.asarray(observations, dtype=np.float32)

    def env_method(self, method_name, weights, *, version):
        self.calls.append((method_name, np.asarray(weights), version))


def _build_callback() -> tuple[SPDLReferenceCurriculumCallback, _VecEnv]:
    callback = SPDLReferenceCurriculumCallback(
        remaining_distances_m=np.asarray([6000.0, 3000.0, 1000.0, 300.0]),
        reference_observations=np.asarray(
            [[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        gamma=0.9,
        profile=resolve_curriculum_profile("spdl"),
    )
    env = _VecEnv()
    callback.model = SimpleNamespace(policy=_Policy(), get_env=lambda: env)
    return callback, env


def _record_context_steps(
    callback: SPDLReferenceCurriculumCallback,
    *,
    count: int = 32,
    index: int = 0,
    version: int = 0,
) -> None:
    for sample_id in range(count):
        callback.locals = {
            "infos": [
                {
                    "reference_context_sample_id": sample_id,
                    "reference_context_index": index,
                    "reference_context_distribution_version": version,
                }
            ],
            "rewards": np.asarray([1.0]),
            "dones": np.asarray([True]),
        }
        assert callback._on_step() is True


def test_spdl_initial_and_target_distributions_have_required_support() -> None:
    callback, _ = _build_callback()

    initial = callback.initial_weights()
    target = callback.target_weights
    assert np.all(initial > 0.0)
    assert initial.sum() == pytest.approx(1.0)
    assert initial[0] == pytest.approx(0.15 + 0.01 / 4)
    assert target.sum() == pytest.approx(1.0)
    assert target[0] == pytest.approx(1.0 - 1e-3 + 1e-3 / 4)
    assert np.all(target[1:] == pytest.approx(1e-3 / 4))


def test_spdl_importance_coefficients_use_sampling_distribution_version() -> None:
    callback, _ = _build_callback()
    callback._context_samples = [
        _SPDLContextSample(reference_index=0, distribution_version=0),
        _SPDLContextSample(reference_index=0, distribution_version=0),
        _SPDLContextSample(reference_index=3, distribution_version=0),
    ]

    coefficients = callback._importance_weighted_coefficients()
    initial = callback.initial_weights()
    assert coefficients[0] == pytest.approx(2.0 * 4.0 / (3.0 * initial[0]))
    assert coefficients[3] == pytest.approx(1.0 / (3.0 * initial[3]))
    assert coefficients[1] == 0.0


def test_spdl_warmup_broadcasts_kl_bounded_distribution_after_first_rollout() -> None:
    callback, env = _build_callback()
    _record_context_steps(callback)

    callback._on_rollout_start()
    assert env.calls == []
    callback._on_rollout_start()

    assert len(env.calls) == 1
    method, weights, version = env.calls[0]
    assert method == "set_reference_initial_state_distribution"
    assert version == 1
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    assert (
        callback._kl_divergence(weights, callback.initial_weights())
        <= 0.05 + 1e-8
    )


def test_spdl_post_warmup_skips_update_for_nonpositive_returns() -> None:
    callback, env = _build_callback()
    callback._context_update_count = 10
    _record_context_steps(callback)
    callback._completed_returns = [-1.0] * 8
    callback._on_rollout_start()
    callback._on_rollout_start()

    assert env.calls == []
    assert callback._context_samples == []


def test_spdl_closed_form_solver_returns_feasible_distribution() -> None:
    callback, _ = _build_callback()
    candidate = callback._solve_distribution(
        np.asarray([8.0, 1.0, 0.0, -1.0]), alpha=0.5
    )

    assert candidate.sum() == pytest.approx(1.0)
    assert np.all(candidate > 0.0)
    assert (
        callback._kl_divergence(candidate, callback.initial_weights())
        <= 0.05 + 1e-8
    )
