from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm

from rl.context_pool import Context, ContextPool
from rl.dspdl import (
    DSPDLCallback,
    DSPDLConfig,
    DSPDLDistributionSolver,
    DSPDLStatisticsHub,
)
from rl.operational_state import OperationalState


def _state(position_m: float) -> OperationalState:
    return cast(OperationalState, cast(object, SimpleNamespace(position_m=position_m)))


def _context_pool() -> ContextPool:
    remaining = [6000.0, 3000.0, 1000.0, 300.0]
    return ContextPool(
        tuple(
            Context(i, distance, _state(distance))
            for i, distance in enumerate(remaining)
        )
    )


class _Policy:
    def __init__(self) -> None:
        self.tensor_conversion_count = 0
        self.value_scale = 1.0

    def obs_to_tensor(self, observations: np.ndarray) -> tuple[th.Tensor, bool]:
        self.tensor_conversion_count += 1
        return th.as_tensor(observations, dtype=th.float32), True

    def predict_values(self, observation_tensor: th.Tensor) -> th.Tensor:
        return observation_tensor[:, :1] * self.value_scale


class _VecEnv:
    def __init__(self, *, num_envs: int = 1) -> None:
        self.num_envs = num_envs

    def env_method(self, *_: object, **__: object) -> list[object]:
        raise AssertionError("traditional DSPDL must not dispatch environment methods")


def _build_callback(
    *, config: DSPDLConfig | None = None, num_envs: int = 1
) -> tuple[DSPDLCallback, _VecEnv, _Policy, DSPDLStatisticsHub]:
    hub = DSPDLStatisticsHub(context_count=4, num_envs=num_envs, gamma=0.9)
    callback = DSPDLCallback(
        context_pool=_context_pool(),
        context_observations=np.asarray(
            [[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        config=config or DSPDLConfig(),
        statistics_hub=hub,
    )
    env = _VecEnv(num_envs=num_envs)
    policy = _Policy()
    callback.model = cast(
        BaseAlgorithm,
        cast(object, SimpleNamespace(policy=policy, get_env=lambda: env)),
    )
    callback._on_training_start()
    return callback, env, policy, hub


def _complete_episodes(
    hub: DSPDLStatisticsHub,
    count: int,
    *,
    reward: float = 1.0,
    context_index: int = 0,
) -> None:
    for episode_index in range(count):
        env_rank = episode_index % hub.num_envs
        hub.begin_episode(
            env_rank=env_rank,
            context_index=context_index,
            distribution_version=hub.accepted_version,
        )
        hub.record_transition(env_rank, reward, done=True)


def test_statistics_hub_preserves_active_return_when_window_is_cleared() -> None:
    hub = DSPDLStatisticsHub(context_count=3, num_envs=2, gamma=0.9)
    hub.begin_episode(env_rank=0, context_index=1, distribution_version=0)
    hub.record_transition(0, 1.0, done=False)
    hub.begin_episode(env_rank=1, context_index=2, distribution_version=0)
    hub.record_transition(1, 4.0, done=True)

    first = hub.snapshot(version=0)
    np.testing.assert_array_equal(first.context_counts, [0, 1, 1])
    np.testing.assert_array_equal(first.completed_context_indices, [2])
    np.testing.assert_allclose(first.completed_returns, [4.0])
    with pytest.raises(ValueError, match="read-only"):
        first.context_counts[0] = 1

    hub.clear_consumed(version=0)
    hub.record_transition(0, 2.0, done=True)
    second = hub.snapshot(version=0)
    np.testing.assert_array_equal(second.context_counts, [0, 0, 0])
    np.testing.assert_array_equal(second.completed_context_indices, [1])
    np.testing.assert_allclose(second.completed_returns, [2.8])


def test_statistics_hub_invalidates_active_episodes_on_version_change() -> None:
    hub = DSPDLStatisticsHub(context_count=2, num_envs=1, gamma=0.9)
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    hub.validate_version_update(1)
    hub.commit_version(1)
    hub.record_transition(0, 4.0, done=True)

    snapshot = hub.snapshot(version=1)
    np.testing.assert_array_equal(snapshot.context_counts, [0, 0])
    np.testing.assert_array_equal(snapshot.completed_returns, [])
    hub.disable()
    assert hub.enabled is False
    assert hub._context_counts.size == 0
    assert hub._active_context_indices.size == 0


def test_statistics_hub_validates_indices_versions_and_rewards() -> None:
    hub = DSPDLStatisticsHub(context_count=2, num_envs=1, gamma=0.9)
    with pytest.raises(IndexError, match="env_rank"):
        hub.begin_episode(env_rank=1, context_index=0, distribution_version=0)
    with pytest.raises(IndexError, match="context_index"):
        hub.begin_episode(env_rank=0, context_index=2, distribution_version=0)
    hub.begin_episode(env_rank=0, context_index=0, distribution_version=0)
    with pytest.raises(ValueError, match="finite"):
        hub.record_transition(0, np.nan, done=False)
    with pytest.raises(ValueError, match="does not match"):
        hub.snapshot(version=1)
    hub.validate_version_update(1)
    with pytest.raises(RuntimeError, match="already pending"):
        hub.validate_version_update(2)
    hub.cancel_version_update(1)


def test_initial_and_smoothed_start_target_distributions() -> None:
    callback, _, _, _ = _build_callback()
    initial = callback.initial_context_distribution()
    target = callback.target_context_distribution

    assert initial.sum() == pytest.approx(1.0)
    assert np.all(initial > 0.0)
    assert int(np.argmax(initial)) == 1
    assert target == pytest.approx([0.9625, 0.0125, 0.0125, 0.0125])


def test_callback_rejects_invalid_initial_gaussian_std() -> None:
    with pytest.raises(ValueError, match="Gaussian std"):
        _ = DSPDLCallback(
            context_pool=_context_pool(),
            context_observations=np.ones((4, 1), dtype=np.float32),
            config=replace(DSPDLConfig(), initial_gaussian_std_m=0.0),
            statistics_hub=DSPDLStatisticsHub(
                context_count=4, num_envs=1, gamma=0.9
            ),
        )


def test_callback_rejects_mismatched_environment_count() -> None:
    callback = DSPDLCallback(
        context_pool=_context_pool(),
        context_observations=np.ones((4, 1), dtype=np.float32),
        config=DSPDLConfig(),
        statistics_hub=DSPDLStatisticsHub(
            context_count=4, num_envs=2, gamma=0.9
        ),
    )
    env = _VecEnv(num_envs=1)
    callback.model = cast(
        BaseAlgorithm,
        cast(object, SimpleNamespace(policy=_Policy(), get_env=lambda: env)),
    )
    with pytest.raises(ValueError, match="environment count"):
        callback._on_training_start()


def test_callback_caches_context_observation_tensor() -> None:
    callback, _, policy, _ = _build_callback()
    first = callback._evaluate_context_values()
    policy.value_scale = 2.0
    second = callback._evaluate_context_values()

    assert policy.tensor_conversion_count == 1
    assert first == pytest.approx([4.0, 3.0, 2.0, 1.0])
    assert second == pytest.approx([8.0, 6.0, 4.0, 2.0])


def test_callback_updates_only_at_configured_rollout_interval() -> None:
    callback, _, _, _ = _build_callback()
    callback._on_rollout_start()
    callback._on_rollout_end()
    callback._on_rollout_start()
    assert callback._context_update_count == 0

    callback._on_rollout_end()
    callback._on_rollout_start()
    assert callback._context_update_count == 1


def test_post_warmup_retains_statistics_until_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, _, _, hub = _build_callback(num_envs=3)
    callback._context_update_count = callback._config.alpha_warmup_updates
    _complete_episodes(hub, 7)
    solver_calls = 0

    def solve(**_: object) -> np.ndarray:
        nonlocal solver_calls
        solver_calls += 1
        return callback.initial_context_distribution()

    monkeypatch.setattr(callback._solver, "solve", solve)
    callback._maybe_update_curriculum()
    assert solver_calls == 0
    assert hub.snapshot(version=0).completed_returns.size == 7

    _complete_episodes(hub, 1)
    callback._maybe_update_curriculum()
    assert solver_calls == 1
    assert hub.snapshot(version=0).completed_returns.size == 0


def test_post_warmup_alpha_depends_on_current_performance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, _, _, hub = _build_callback()
    callback._context_update_count = callback._config.alpha_warmup_updates
    _complete_episodes(hub, 8, reward=2.0)
    captured_alpha: list[float] = []

    def solve(**kwargs: object) -> np.ndarray:
        captured_alpha.append(float(kwargs["alpha"]))
        return callback.initial_context_distribution()

    monkeypatch.setattr(callback._solver, "solve", solve)
    expected = (
        callback._config.zeta
        * 2.0
        / callback._solver.kl_divergence(
            callback.initial_context_distribution(),
            callback.target_context_distribution,
        )
    )
    callback._maybe_update_curriculum()
    assert captured_alpha == pytest.approx([expected])


def test_target_convergence_updates_shared_version_and_disables_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, _, _, hub = _build_callback()
    monkeypatch.setattr(
        callback._solver,
        "solve",
        lambda **_: callback.target_context_distribution,
    )
    callback._maybe_update_curriculum()

    assert callback._converged is True
    assert callback.distribution_state.version == 1
    assert hub.accepted_version == 1
    assert hub.enabled is False
    callback._maybe_update_curriculum()
    assert callback._context_update_count == 1


def test_distribution_update_failure_keeps_statistics_and_versions_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, _, _, hub = _build_callback()
    _complete_episodes(hub, 1)
    monkeypatch.setattr(
        callback._solver,
        "solve",
        lambda **_: callback.target_context_distribution,
    )
    monkeypatch.setattr(
        callback.distribution_state,
        "update",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    with pytest.raises(RuntimeError, match="failed"):
        callback._maybe_update_curriculum()

    assert callback.distribution_state.version == 0
    assert hub.accepted_version == 0
    assert hub.snapshot(version=0).completed_returns.size == 1
    hub.validate_version_update(1)
    hub.cancel_version_update(1)


def test_distribution_solver_is_feasible_and_stops_on_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = DSPDLDistributionSolver(relative_entropy_bound=0.02)
    current = np.asarray([0.3, 0.3, 0.2, 0.2], dtype=np.float64)
    target = np.asarray([0.95, 0.02, 0.02, 0.01], dtype=np.float64)
    original = solver._distribution_at_dual
    calls = 0

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_distribution_at_dual", counted)
    candidate = solver.solve(
        context_values=np.asarray([8.0, 1.0, 0.0, -1.0]),
        current_distribution=current,
        target_distribution=target,
        alpha=0.5,
    )
    assert candidate.sum() == pytest.approx(1.0)
    assert solver.kl_divergence(candidate, current) <= 0.02 + solver.tolerance
    assert calls < solver.max_iterations


def test_equal_warmup_values_skip_dual_search(monkeypatch: pytest.MonkeyPatch) -> None:
    solver = DSPDLDistributionSolver(relative_entropy_bound=0.02)
    current = np.asarray([0.6, 0.4], dtype=np.float64)
    monkeypatch.setattr(
        solver,
        "_distribution_at_dual",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dual search should be skipped")
        ),
    )
    candidate = solver.solve(
        context_values=np.ones(2),
        current_distribution=current,
        target_distribution=np.asarray([0.9, 0.1]),
        alpha=0.0,
    )
    assert candidate == pytest.approx(current)


def test_training_end_releases_static_observation_cache_and_hub() -> None:
    callback, _, _, hub = _build_callback()
    callback._on_training_end()
    assert callback._context_observation_tensor is None
    assert callback._context_observations.shape == (0, 0)
    assert hub.enabled is False
