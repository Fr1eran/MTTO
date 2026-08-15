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
    DSPDLEpisodeAccumulator,
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


def _payload(
    *,
    version: int = 0,
    counts: list[int] | None = None,
    indices: list[int] | None = None,
    returns: list[float] | None = None,
) -> dict[str, object]:
    return {
        "version": version,
        "context_counts": np.asarray(counts or [0, 0, 0, 0], dtype=np.int64),
        "completed_context_indices": np.asarray(indices or [], dtype=np.int64),
        "completed_returns": np.asarray(returns or [], dtype=np.float64),
    }


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
        self.payload_batches: list[list[dict[str, object]]] = []
        self.distribution_updates: list[tuple[np.ndarray, int]] = []
        self.disable_count = 0
        self.drain_count = 0

    def env_method(
        self, method_name: str, *args: object, **kwargs: object
    ) -> list[object]:
        if method_name == "drain_dspdl_statistics":
            self.drain_count += 1
            if self.payload_batches:
                return list(self.payload_batches.pop(0))
            version = int(kwargs["version"])
            return [_payload(version=version) for _ in range(self.num_envs)]
        if method_name == "set_dspdl_distribution":
            self.distribution_updates.append(
                (np.asarray(args[0], dtype=np.float64), int(kwargs["version"]))
            )
            return [None for _ in range(self.num_envs)]
        if method_name == "disable_dspdl_accumulator":
            self.disable_count += 1
            return [None for _ in range(self.num_envs)]
        raise AssertionError(f"unexpected environment method: {method_name}")


def _build_callback(
    *, config: DSPDLConfig | None = None, num_envs: int = 1
) -> tuple[DSPDLCallback, _VecEnv, _Policy]:
    callback = DSPDLCallback(
        context_pool=_context_pool(),
        context_observations=np.asarray(
            [[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        config=config or DSPDLConfig(),
    )
    env = _VecEnv(num_envs=num_envs)
    policy = _Policy()
    callback.model = cast(
        BaseAlgorithm,
        cast(object, SimpleNamespace(policy=policy, get_env=lambda: env)),
    )
    callback._on_training_start()
    return callback, env, policy


def test_episode_accumulator_preserves_active_return_across_drains() -> None:
    accumulator = DSPDLEpisodeAccumulator(context_count=3, gamma=0.9)
    accumulator.begin_episode(context_index=1, distribution_version=0)
    accumulator.record_transition(1.0, done=False)

    first = accumulator.drain(version=0)
    np.testing.assert_array_equal(first["context_counts"], [0, 1, 0])
    np.testing.assert_array_equal(first["completed_returns"], [])

    accumulator.record_transition(2.0, done=True)
    second = accumulator.drain(version=0)
    np.testing.assert_array_equal(second["context_counts"], [0, 0, 0])
    np.testing.assert_array_equal(second["completed_context_indices"], [1])
    np.testing.assert_allclose(second["completed_returns"], [2.8])


def test_episode_accumulator_rejects_old_version_and_releases_buffers() -> None:
    accumulator = DSPDLEpisodeAccumulator(context_count=2, gamma=0.9)
    accumulator.begin_episode(context_index=0, distribution_version=0)
    accumulator.switch_version(1)
    accumulator.record_transition(4.0, done=True)

    payload = accumulator.drain(version=1)
    np.testing.assert_array_equal(payload["context_counts"], [0, 0])
    np.testing.assert_array_equal(payload["completed_returns"], [])
    accumulator.disable()
    assert accumulator.enabled is False
    assert accumulator._context_counts.size == 0


def test_initial_and_smoothed_start_target_distributions() -> None:
    callback, _, _ = _build_callback()
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
        )


def test_callback_caches_context_observation_tensor() -> None:
    callback, _, policy = _build_callback()
    first = callback._evaluate_context_values()
    policy.value_scale = 2.0
    second = callback._evaluate_context_values()

    assert policy.tensor_conversion_count == 1
    assert first == pytest.approx([4.0, 3.0, 2.0, 1.0])
    assert second == pytest.approx([8.0, 6.0, 4.0, 2.0])


def test_callback_drains_workers_only_at_configured_rollout_interval() -> None:
    callback, env, _ = _build_callback()
    callback._on_rollout_start()
    callback._on_rollout_end()
    callback._on_rollout_start()
    assert env.drain_count == 0

    callback._on_rollout_end()
    callback._on_rollout_start()
    assert env.drain_count == 1
    assert callback._context_update_count == 1


def test_post_warmup_accumulates_statistics_until_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, env, _ = _build_callback(num_envs=3)
    callback._context_update_count = callback._config.alpha_warmup_updates
    env.payload_batches = [
        [_payload(counts=[7, 0, 0, 0], indices=[0] * 7, returns=[1.0] * 7)],
        [_payload(counts=[1, 0, 0, 0], indices=[0], returns=[1.0])],
    ]
    solver_calls = 0

    def solve(**_: object) -> np.ndarray:
        nonlocal solver_calls
        solver_calls += 1
        return callback.initial_context_distribution()

    monkeypatch.setattr(callback._solver, "solve", solve)
    callback._maybe_update_curriculum()
    assert solver_calls == 0
    assert len(callback._completed_returns) == 7

    callback._maybe_update_curriculum()
    assert solver_calls == 1
    assert callback._completed_returns == []


def test_post_warmup_alpha_depends_on_current_performance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, env, _ = _build_callback()
    callback._context_update_count = callback._config.alpha_warmup_updates
    env.payload_batches = [
        [_payload(counts=[8, 0, 0, 0], indices=[0] * 8, returns=[2.0] * 8)]
    ]
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


def test_target_convergence_disables_worker_accumulators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, env, _ = _build_callback()
    monkeypatch.setattr(
        callback._solver,
        "solve",
        lambda **_: callback.target_context_distribution,
    )
    callback._maybe_update_curriculum()

    assert callback._converged is True
    assert env.distribution_updates[0][1] == 1
    assert env.disable_count == 1
    callback._maybe_update_curriculum()
    assert env.drain_count == 1


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


def test_training_end_releases_static_observation_cache() -> None:
    callback, _, _ = _build_callback()
    callback._on_training_end()
    assert callback._context_observation_tensor is None
    assert callback._context_observations.shape == (0, 0)
