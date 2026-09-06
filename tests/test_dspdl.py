from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm

from rl.context_pool import Context, ContextPool
from rl.dspdl import (
    DSPDL_ALPHA_WARMUP_UPDATES,
    DSPDLCallback,
    DSPDLStatisticsHub,
    DSPDL_UPDATE_INTERVAL_ROLLOUTS,
    DSPDL_ZETA,
    dspdl_protocol_parameters,
)
from rl.operational_state import OperationalState


def _context_pool() -> ContextPool:
    distances = [6000.0, 3000.0, 1000.0, 300.0]
    return ContextPool(
        tuple(
            Context(
                index,
                distance,
                cast(
                    OperationalState,
                    cast(object, SimpleNamespace(position_m=distance)),
                ),
            )
            for index, distance in enumerate(distances)
        )
    )


class _Policy:
    def __init__(self) -> None:
        self.tensor_conversion_count = 0
        self.value_scale = 1.0
        self.value_batch_sizes: list[int] = []

    def obs_to_tensor(self, observations: np.ndarray) -> tuple[th.Tensor, bool]:
        self.tensor_conversion_count += 1
        return th.as_tensor(observations, dtype=th.float32), True

    def predict_values(self, observations: th.Tensor) -> th.Tensor:
        self.value_batch_sizes.append(int(observations.shape[0]))
        return observations[:, :1] * self.value_scale


class _VecEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs


def _callback(
    *,
    num_envs: int = 1,
    potentials: np.ndarray | None = None,
) -> tuple[DSPDLCallback, DSPDLStatisticsHub, _Policy]:
    hub = DSPDLStatisticsHub(context_count=4, num_envs=num_envs, gamma=0.9)
    callback = DSPDLCallback(
        context_pool=_context_pool(),
        context_observations=np.asarray(
            [[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        statistics_hub=hub,
        context_punctuality_potentials=potentials,
    )
    policy = _Policy()
    env = _VecEnv(num_envs)
    callback.model = cast(
        BaseAlgorithm,
        cast(object, SimpleNamespace(policy=policy, get_env=lambda: env)),
    )
    callback._on_training_start()
    return callback, hub, policy


def test_statistics_hub_accumulates_raw_discounted_returns() -> None:
    hub = DSPDLStatisticsHub(context_count=3, num_envs=1, gamma=0.9)
    hub.begin_episode(env_rank=0, context_index=1, distribution_version=0)
    hub.record_transition(0, 1.0, done=False)
    hub.record_transition(0, 2.0, done=True)
    hub.finish_rollout(version=0)

    snapshot = hub.snapshot(version=0)
    np.testing.assert_array_equal(snapshot.context_counts, [0, 1, 0])
    np.testing.assert_array_equal(snapshot.completed_context_indices, [1])
    np.testing.assert_allclose(snapshot.completed_returns, [2.8])
    np.testing.assert_allclose(snapshot.rollout_returns, [2.8])


def test_curriculum_compensates_initial_potential_before_weighting():
    callback, hub, _ = _callback(
        potentials=np.asarray([0, -0.5, -1, -0.25]),
    )
    callback._record_scalar = lambda *args: None
    hub.begin_episode(env_rank=0, context_index=1, distribution_version=0)
    distribution = np.full(4, 0.25)
    coefficients, indices, predictions = callback._estimate_context_coefficients(
        snapshot=hub.snapshot(version=0), current_distribution=distribution
    )
    np.testing.assert_allclose(coefficients, [0, 10, 0, 0])
    np.testing.assert_allclose(predictions, [2.5])


def test_statistics_hub_seals_incomplete_episodes_at_rollout_boundaries() -> None:
    hub = DSPDLStatisticsHub(context_count=2, num_envs=2, gamma=0.5)
    for rank, reward in enumerate((2.0, 4.0)):
        hub.begin_episode(
            env_rank=rank,
            context_index=rank,
            distribution_version=0,
        )
        hub.record_transition(rank, reward, done=False)
        hub.record_transition(rank, reward, done=False)
    hub.finish_rollout(version=0)

    snapshot = hub.snapshot(version=0)
    assert snapshot.completed_returns.size == 0
    np.testing.assert_allclose(snapshot.rollout_returns, [3.0, 6.0])


def test_statistics_version_switch_invalidates_active_old_episode() -> None:
    hub = DSPDLStatisticsHub(context_count=2, num_envs=1, gamma=0.9)
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.validate_version_update(1)
    hub.commit_version(1)
    hub.record_transition(0, 4.0, done=True)

    snapshot = hub.snapshot(version=1)
    assert snapshot.completed_returns.size == 0
    np.testing.assert_array_equal(snapshot.context_counts, [0, 0])


def test_callback_caches_initial_observation_transform() -> None:
    callback, _, policy = _callback()
    first = callback._evaluate_context_values()
    policy.value_scale = 2.0
    second = callback._evaluate_context_values()

    assert policy.tensor_conversion_count == 1
    np.testing.assert_allclose(first, [4.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(second, [8.0, 6.0, 4.0, 2.0])


def test_importance_weighted_coefficients_match_eq5_and_only_evaluate_samples() -> None:
    callback, hub, policy = _callback()
    recorded: dict[str, float] = {}
    callback._record_scalar = lambda key, value: recorded.__setitem__(key, value)
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.begin_episode(env_rank=0, context_index=1, distribution_version=0)

    coefficients, indices, predictions = callback._estimate_context_coefficients(
        snapshot=hub.snapshot(version=0),
        current_distribution=callback.initial_context_distribution(),
    )

    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(predictions, [4.0, 3.0])
    np.testing.assert_allclose(coefficients, [32.0 / 3.0, 4.0, 0.0, 0.0])
    assert policy.value_batch_sizes == [2]
    assert recorded["dspdl/importance_weight_ess"] == pytest.approx(1.8)
    assert recorded["dspdl/importance_weight_ess_ratio"] == pytest.approx(0.9)
    assert recorded["dspdl/importance_weight_max_to_mean"] == pytest.approx(4 / 3)


def test_importance_weighted_coefficients_reject_nonpositive_sample_probability() -> (
    None
):
    callback, hub, _ = _callback()
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)

    with pytest.raises(RuntimeError, match="finite positive sampling probability"):
        _ = callback._estimate_context_coefficients(
            snapshot=hub.snapshot(version=0),
            current_distribution=np.asarray([0.0, 0.5, 0.25, 0.25]),
        )


def test_initial_distribution_is_uniform_and_target_remains_start_peaked() -> None:
    callback, _, _ = _callback()

    np.testing.assert_allclose(
        callback.initial_context_distribution(), np.full(4, 0.25)
    )
    np.testing.assert_allclose(
        callback.target_context_distribution,
        [0.95 + 0.05 / 4, 0.05 / 4, 0.05 / 4, 0.05 / 4],
    )


def test_post_warmup_alpha_uses_mean_raw_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, hub, _ = _callback()
    callback._context_update_count = DSPDL_ALPHA_WARMUP_UPDATES
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.record_transition(0, 2.0, done=False)
    hub.finish_rollout(version=0)
    captured: list[float] = []

    def solve(**kwargs: object) -> np.ndarray:
        captured.append(float(kwargs["alpha"]))
        return callback.initial_context_distribution()

    monkeypatch.setattr(callback._solver, "solve", solve)
    target_kl = callback._solver.kl_divergence(
        callback.initial_context_distribution(),
        callback.target_context_distribution,
    )
    callback._maybe_update_curriculum()

    assert captured == pytest.approx([DSPDL_ZETA * 2.0 / target_kl])


def test_alpha_uses_rollout_returns_without_waiting_for_episode_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, hub, _ = _callback()
    callback._context_update_count = DSPDL_ALPHA_WARMUP_UPDATES
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.record_transition(0, 3.0, done=False)
    hub.finish_rollout(version=0)
    captured: list[float] = []

    def solve(**kwargs: object) -> np.ndarray:
        captured.append(float(kwargs["alpha"]))
        return callback.initial_context_distribution()

    monkeypatch.setattr(callback._solver, "solve", solve)
    callback._maybe_update_curriculum()

    assert captured[0] > 0.0
    assert hub.snapshot(version=0).completed_returns.size == 0


def test_alpha_clips_negative_rollout_return_mean_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, hub, _ = _callback()
    callback._context_update_count = DSPDL_ALPHA_WARMUP_UPDATES
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.record_transition(0, -2.0, done=False)
    hub.finish_rollout(version=0)
    captured: list[float] = []

    def solve(**kwargs: object) -> np.ndarray:
        captured.append(float(kwargs["alpha"]))
        return callback.initial_context_distribution()

    monkeypatch.setattr(callback._solver, "solve", solve)
    callback._maybe_update_curriculum()

    assert captured == [0.0]


def test_callback_seals_each_rollout_before_a_curriculum_update() -> None:
    callback, hub, _ = _callback()
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.record_transition(0, 2.0, done=False)

    callback._on_rollout_end()

    np.testing.assert_allclose(hub.snapshot(version=0).rollout_returns, [2.0])


def test_fixed_interval_attempts_course_update_after_four_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, _, _ = _callback()
    attempts: list[None] = []
    monkeypatch.setattr(
        callback,
        "_maybe_update_curriculum",
        lambda: attempts.append(None),
    )

    callback._on_rollout_start()
    assert attempts == []
    for _ in range(DSPDL_UPDATE_INTERVAL_ROLLOUTS):
        callback._on_rollout_end()
        callback._on_rollout_start()

    assert attempts == [None]


def test_protocol_parameters_are_fixed_and_returned_as_a_copy() -> None:
    first = dspdl_protocol_parameters()
    first["zeta"] = 99
    second = dspdl_protocol_parameters()
    assert second["zeta"] == DSPDL_ZETA
    assert second["update_interval_rollouts"] == DSPDL_UPDATE_INTERVAL_ROLLOUTS


def test_distribution_update_is_atomic_and_convergence_disables_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, hub, _ = _callback()
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.record_transition(0, 1.0, done=False)
    hub.finish_rollout(version=0)
    monkeypatch.setattr(
        callback._solver,
        "solve",
        lambda **_: callback.target_context_distribution,
    )
    callback._maybe_update_curriculum()

    assert callback.distribution_state.version == 1
    assert hub.accepted_version == 1
    assert hub.enabled is False


def test_active_episode_survives_unchanged_distribution_clear_and_calibrates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, hub, _ = _callback(num_envs=1)
    # 1. Active episode begins on context 0 and receives partial rollout
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.record_transition(0, 1.0, done=False)
    hub.finish_rollout(version=0)

    # 2. First curriculum update without changing distribution: clear_consumed is called
    monkeypatch.setattr(
        callback._solver,
        "solve",
        lambda **_: callback.initial_context_distribution(),
    )
    callback._maybe_update_curriculum()

    # Context 0 count should be rebuilt from the active episode
    snapshot_after_clear = hub.snapshot(version=0)
    assert snapshot_after_clear.context_counts[0] == 1
    assert snapshot_after_clear.completed_returns.size == 0

    # 3. Active episode completes on context 0
    hub.record_transition(0, 2.0, done=True)

    # 4. New episode begins on a different context (context 1)
    hub.begin_episode(env_rank=0, context_index=1, distribution_version=0)
    hub.record_transition(0, 1.0, done=False)
    hub.finish_rollout(version=0)

    # 5. Next update: must calibrate without
    # RuntimeError: completed context is missing a value estimate
    callback._maybe_update_curriculum()


def test_clear_consumed_rebuilds_active_episode_counts_for_multiple_workers() -> None:
    hub = DSPDLStatisticsHub(context_count=4, num_envs=3, gamma=0.9)
    # Two workers on context 2, one worker on context 1
    hub.begin_episode(env_rank=0, context_index=2, distribution_version=0)
    hub.begin_episode(env_rank=1, context_index=2, distribution_version=0)
    hub.begin_episode(env_rank=2, context_index=1, distribution_version=0)
    hub.record_transition(0, 1.0, done=False)
    hub.record_transition(1, 2.0, done=False)
    hub.record_transition(2, 3.0, done=False)
    hub.finish_rollout(version=0)

    hub.clear_consumed(version=0)

    snapshot = hub.snapshot(version=0)
    assert snapshot.completed_returns.size == 0
    assert snapshot.rollout_returns.size == 0
    # Duplicate-safe accumulation: worker 0 and worker 1 both increment context 2
    np.testing.assert_array_equal(snapshot.context_counts, [0, 1, 2, 0])

    # Accumulated returns are preserved for calibration upon eventual completion
    hub.record_transition(0, 1.0, done=True)
    snapshot_completed = hub.snapshot(version=0)
    np.testing.assert_array_equal(snapshot_completed.completed_context_indices, [2])
    np.testing.assert_allclose(snapshot_completed.completed_returns, [1.0 + 0.9 * 1.0])


def test_committed_version_change_excludes_old_version_active_episodes() -> None:
    hub = DSPDLStatisticsHub(context_count=3, num_envs=2, gamma=0.9)
    hub.begin_episode(env_rank=0, context_index=1, distribution_version=0)
    hub.begin_episode(env_rank=1, context_index=2, distribution_version=0)
    hub.record_transition(0, 1.0, done=False)
    hub.record_transition(1, 2.0, done=False)
    hub.finish_rollout(version=0)

    hub.validate_version_update(1)
    hub.commit_version(1)

    # Counts must be cleared and no old active episodes rebuilt
    snapshot = hub.snapshot(version=1)
    np.testing.assert_array_equal(snapshot.context_counts, [0, 0, 0])

    # Old episodes finishing under old version are invalidated and excluded
    hub.record_transition(0, 1.0, done=True)
    hub.record_transition(1, 2.0, done=True)
    snapshot_after = hub.snapshot(version=1)
    assert snapshot_after.completed_returns.size == 0
    assert snapshot_after.completed_context_indices.size == 0
    np.testing.assert_array_equal(snapshot_after.context_counts, [0, 0, 0])
