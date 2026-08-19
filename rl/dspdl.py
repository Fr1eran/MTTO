"""Discrete SPDL curriculum control over a finite context pool."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import cast, override

import numpy as np
import torch as th
from numpy.typing import NDArray
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

from rl.context_pool import ContextPool
from rl.context_sampler import CurriculumDistributionState

__all__ = [
    "DSPDLCallback",
    "DSPDLConfig",
    "DSPDLDistributionSolver",
    "DSPDLStatisticsHub",
    "DSPDLStatisticsSnapshot",
]


@dataclass(frozen=True)
class DSPDLConfig:
    initial_gaussian_std_m: float = 800.0
    initial_peak_remaining_distance_m: float = 3000.0
    initial_uniform_mass: float = 0.01
    target_uniform_mass: float = 0.05
    target_kl_stop: float = 0.1
    alpha_warmup_updates: int = 5
    relative_entropy_bound: float = 0.02
    zeta: float = 0.01
    update_interval_rollouts: int = 2
    min_completed_episodes: int = 8
    min_completed_episodes_per_env: int = 2


@dataclass(frozen=True, slots=True)
class DSPDLStatisticsSnapshot:
    """Immutable snapshot of centralized DSPDL curriculum statistics."""

    version: int
    context_counts: NDArray[np.int64]
    completed_context_indices: NDArray[np.int64]
    completed_returns: NDArray[np.float64]


class DSPDLStatisticsHub:
    """Collect all DummyVecEnv DSPDL statistics in one shared state."""

    def __init__(self, *, context_count: int, num_envs: int, gamma: float) -> None:
        if context_count <= 0:
            raise ValueError("context_count must be positive")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        self._context_count = int(context_count)
        self._num_envs = int(num_envs)
        self._gamma = float(gamma)
        self._enabled = True
        self._accepted_version = 0
        self._pending_version: int | None = None
        self._active_context_indices = np.full(self._num_envs, -1, dtype=np.int64)
        self._active_versions = np.full(self._num_envs, -1, dtype=np.int64)
        self._active_returns = np.zeros(self._num_envs, dtype=np.float64)
        self._active_discounts = np.ones(self._num_envs, dtype=np.float64)
        self._active_valid = np.zeros(self._num_envs, dtype=np.bool_)
        self._context_counts = np.zeros(self._context_count, dtype=np.int64)
        self._completed_context_indices: list[int] = []
        self._completed_returns: list[float] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def accepted_version(self) -> int:
        return self._accepted_version

    @property
    def context_count(self) -> int:
        return self._context_count

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def begin_episode(
        self,
        *,
        env_rank: int,
        context_index: int,
        distribution_version: int,
    ) -> None:
        if not self._enabled:
            return
        rank = self._validate_env_rank(env_rank)
        if not isinstance(context_index, (int, np.integer)):
            raise TypeError("context_index must be an integer")
        index = int(context_index)
        if not 0 <= index < self._context_count:
            raise IndexError("context_index is outside the context pool")
        if not isinstance(distribution_version, (int, np.integer)):
            raise TypeError("distribution_version must be an integer")
        version = int(distribution_version)
        self._active_context_indices[rank] = index
        self._active_versions[rank] = version
        self._active_returns[rank] = 0.0
        self._active_discounts[rank] = 1.0
        self._active_valid[rank] = version == self._accepted_version
        if self._active_valid[rank]:
            self._context_counts[index] += 1

    def record_transition(self, env_rank: int, reward: float, *, done: bool) -> None:
        if not self._enabled:
            return
        rank = self._validate_env_rank(env_rank)
        if self._active_context_indices[rank] < 0:
            return
        value = float(reward)
        if not np.isfinite(value):
            raise ValueError("DSPDL transition reward must be finite")
        if self._active_valid[rank]:
            self._active_returns[rank] += self._active_discounts[rank] * value
            self._active_discounts[rank] *= self._gamma
        if done:
            if self._active_valid[rank]:
                self._completed_context_indices.append(
                    int(self._active_context_indices[rank])
                )
                self._completed_returns.append(float(self._active_returns[rank]))
            self._clear_active_episode(rank)

    def snapshot(self, *, version: int) -> DSPDLStatisticsSnapshot:
        self._validate_requested_version(version)
        if self._enabled:
            counts = self._context_counts.copy()
            indices = np.asarray(self._completed_context_indices, dtype=np.int64)
            returns = np.asarray(self._completed_returns, dtype=np.float64)
        else:
            counts = np.zeros(self._context_count, dtype=np.int64)
            indices = np.empty(0, dtype=np.int64)
            returns = np.empty(0, dtype=np.float64)
        for values in (counts, indices, returns):
            values.flags.writeable = False
        return DSPDLStatisticsSnapshot(
            version=self._accepted_version,
            context_counts=counts,
            completed_context_indices=indices,
            completed_returns=returns,
        )

    def clear_consumed(self, *, version: int) -> None:
        self._validate_requested_version(version)
        if self._enabled:
            self._clear_completed_statistics()

    def validate_version_update(self, version: int) -> int:
        if not self._enabled:
            raise RuntimeError("DSPDL statistics hub is disabled")
        if not isinstance(version, (int, np.integer)):
            raise TypeError("statistics version must be an integer")
        new_version = int(version)
        if new_version <= self._accepted_version:
            raise ValueError("statistics version must increase")
        if self._pending_version is not None:
            raise RuntimeError("a statistics version update is already pending")
        self._pending_version = new_version
        return new_version

    def cancel_version_update(self, version: int) -> None:
        if self._pending_version != int(version):
            raise ValueError("statistics version was not pending")
        self._pending_version = None

    def commit_version(self, version: int) -> None:
        new_version = int(version)
        if self._pending_version != new_version:
            raise ValueError("statistics version was not validated before commit")
        self._accepted_version = new_version
        self._pending_version = None
        self._clear_completed_statistics()
        self._active_valid &= self._active_versions == new_version

    def disable(self) -> None:
        if not self._enabled:
            return
        self._enabled = False
        self._pending_version = None
        self._active_context_indices = np.empty(0, dtype=np.int64)
        self._active_versions = np.empty(0, dtype=np.int64)
        self._active_returns = np.empty(0, dtype=np.float64)
        self._active_discounts = np.empty(0, dtype=np.float64)
        self._active_valid = np.empty(0, dtype=np.bool_)
        self._context_counts = np.empty(0, dtype=np.int64)
        self._completed_context_indices.clear()
        self._completed_returns.clear()

    def _validate_env_rank(self, env_rank: int) -> int:
        if not isinstance(env_rank, (int, np.integer)):
            raise TypeError("env_rank must be an integer")
        rank = int(env_rank)
        if not 0 <= rank < self._num_envs:
            raise IndexError("env_rank is outside the statistics hub")
        return rank

    def _validate_requested_version(self, version: int) -> None:
        if not isinstance(version, (int, np.integer)):
            raise TypeError("statistics version must be an integer")
        if int(version) != self._accepted_version:
            raise ValueError("requested statistics version does not match the hub")

    def _clear_completed_statistics(self) -> None:
        if self._context_counts.size:
            self._context_counts.fill(0)
        self._completed_context_indices.clear()
        self._completed_returns.clear()

    def _clear_active_episode(self, rank: int) -> None:
        self._active_context_indices[rank] = -1
        self._active_versions[rank] = -1
        self._active_returns[rank] = 0.0
        self._active_discounts[rank] = 1.0
        self._active_valid[rank] = False


class DSPDLDistributionSolver:
    """Solve the finite DSPDL KL-constrained distribution update."""

    def __init__(
        self,
        *,
        relative_entropy_bound: float,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
    ) -> None:
        if relative_entropy_bound <= 0.0:
            raise ValueError("relative_entropy_bound must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.relative_entropy_bound = float(relative_entropy_bound)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

    def solve(
        self,
        *,
        context_values: NDArray[np.float64],
        current_distribution: NDArray[np.float64],
        target_distribution: NDArray[np.float64],
        alpha: float,
    ) -> NDArray[np.float64]:
        values = np.asarray(context_values, dtype=np.float64)
        current = np.asarray(current_distribution, dtype=np.float64)
        target = np.asarray(target_distribution, dtype=np.float64)
        if values.shape != current.shape or target.shape != current.shape:
            raise ValueError("DSPDL solver inputs must have matching shapes")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if alpha == 0.0 and np.allclose(
            values, values[0], rtol=0.0, atol=self.tolerance
        ):
            return current.copy()

        log_target = np.log(target)
        log_current = np.log(current)
        if alpha > 0.0:
            unconstrained = self._distribution_at_dual(
                values, alpha, 0.0, log_target, log_current
            )
            if (
                self.kl_divergence(unconstrained, current)
                <= self.relative_entropy_bound + self.tolerance
            ):
                return unconstrained

        lower = 0.0
        upper = 1.0
        while (
            self.kl_divergence(
                self._distribution_at_dual(
                    values, alpha, upper, log_target, log_current
                ),
                current,
            )
            > self.relative_entropy_bound + self.tolerance
        ):
            upper *= 2.0
            if upper > 1e12:
                raise RuntimeError("could not satisfy the DSPDL relative-entropy bound")

        feasible = self._distribution_at_dual(
            values, alpha, upper, log_target, log_current
        )
        for _ in range(self.max_iterations):
            middle = (lower + upper) / 2.0
            candidate = self._distribution_at_dual(
                values, alpha, middle, log_target, log_current
            )
            candidate_kl = self.kl_divergence(candidate, current)
            if abs(candidate_kl - self.relative_entropy_bound) <= self.tolerance:
                return candidate
            if candidate_kl > self.relative_entropy_bound:
                lower = middle
            else:
                upper = middle
                feasible = candidate
            if upper - lower <= np.finfo(np.float64).eps * max(1.0, upper):
                break
        return feasible

    @staticmethod
    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        smallest = np.finfo(np.float64).tiny
        safe_left = np.maximum(left, smallest)
        safe_right = np.maximum(right, smallest)
        return float(np.sum(safe_left * (np.log(safe_left) - np.log(safe_right))))

    @staticmethod
    def _distribution_at_dual(
        values: np.ndarray,
        alpha: float,
        dual: float,
        log_target: np.ndarray,
        log_current: np.ndarray,
    ) -> NDArray[np.float64]:
        denominator = alpha + dual
        if denominator <= 0.0:
            raise ValueError("DSPDL dual denominator must be positive")
        logits = (
            values / denominator
            + alpha / denominator * log_target
            + dual / denominator * log_current
        )
        logits -= float(np.max(logits))
        distribution = np.maximum(np.exp(logits), np.finfo(np.float64).tiny).astype(
            np.float64
        )
        return distribution / float(np.sum(distribution))


class DSPDLCallback(BaseCallback):
    """Coordinate DSPDL updates using centralized episode statistics."""

    def __init__(
        self,
        *,
        context_pool: ContextPool,
        context_observations: NDArray[np.float32],
        config: DSPDLConfig,
        statistics_hub: DSPDLStatisticsHub,
        solver: DSPDLDistributionSolver | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        observations = np.asarray(context_observations, dtype=np.float32)
        if (
            observations.ndim != 2
            or observations.shape[0] != context_pool.context_count
        ):
            raise ValueError("context_observations must have one row per context")
        if not np.all(np.isfinite(observations)):
            raise ValueError("context_observations must be finite")
        if statistics_hub.context_count != context_pool.context_count:
            raise ValueError(
                "statistics hub context count must match the context pool"
            )
        self._context_pool = context_pool
        self._context_observations = observations
        self._context_observation_tensor: th.Tensor | None = None
        self._config = config
        self._statistics_hub = statistics_hub
        self._validate_config()
        self._solver = solver or DSPDLDistributionSolver(
            relative_entropy_bound=config.relative_entropy_bound
        )
        self._start_index = int(np.argmax(context_pool.remaining_distances_m))
        self._target_distribution = self._build_target_distribution()
        self._distribution_state = CurriculumDistributionState(
            context_count=context_pool.context_count,
            initial_distribution=self._build_initial_distribution(),
        )
        self._rollouts_since_update_attempt = 0
        self._context_update_count = 0
        self._converged = False

    def initial_context_distribution(self) -> NDArray[np.float64]:
        return self._distribution_state.distribution

    @property
    def distribution_state(self) -> CurriculumDistributionState:
        return self._distribution_state

    @property
    def target_context_distribution(self) -> NDArray[np.float64]:
        return self._target_distribution.copy()

    @property
    def statistics_hub(self) -> DSPDLStatisticsHub:
        return self._statistics_hub

    @override
    def _on_training_start(self) -> None:
        policy = cast(ActorCriticPolicy, self.model.policy)
        tensor, _ = policy.obs_to_tensor(self._context_observations)
        self._context_observation_tensor = tensor
        if self._statistics_hub.num_envs != int(self.training_env.num_envs):
            raise ValueError(
                "statistics hub environment count must match the training environment"
            )
        if self._statistics_hub.accepted_version != self._distribution_state.version:
            raise ValueError(
                "statistics hub version must match the curriculum distribution"
            )
        self._record_scalar("dspdl/converged", 0.0)
        snapshot = self._statistics_hub.snapshot(
            version=self._distribution_state.version
        )
        self._record_curriculum_metrics(
            snapshot=snapshot, empirical_distribution=None
        )

    @override
    def _on_rollout_start(self) -> None:
        if (
            not self._converged
            and self._rollouts_since_update_attempt
            >= self._config.update_interval_rollouts
        ):
            self._rollouts_since_update_attempt = 0
            self._maybe_update_curriculum()

    @override
    def _on_rollout_end(self) -> None:
        if not self._converged:
            self._rollouts_since_update_attempt += 1

    @override
    def _on_step(self) -> bool:
        return True

    @override
    def _on_training_end(self) -> None:
        self._context_observations = np.empty((0, 0), dtype=np.float32)
        self._context_observation_tensor = None
        self._statistics_hub.disable()

    def _maybe_update_curriculum(self) -> None:
        if self._converged:
            return
        update_started = perf_counter()
        current_version = self._distribution_state.version
        if self._statistics_hub.accepted_version != current_version:
            raise ValueError(
                "statistics hub version must match the curriculum distribution"
            )
        snapshot = self._statistics_hub.snapshot(version=current_version)
        empirical = self._empirical_context_distribution(snapshot)
        self._record_curriculum_metrics(
            snapshot=snapshot, empirical_distribution=empirical
        )
        current_distribution = self._distribution_state.distribution

        target_kl = self._solver.kl_divergence(
            current_distribution, self._target_distribution
        )
        if target_kl <= self._config.target_kl_stop:
            self._mark_converged()
            return

        alpha = 0.0
        if self._context_update_count >= self._config.alpha_warmup_updates:
            minimum = max(
                self._config.min_completed_episodes,
                self._config.min_completed_episodes_per_env
                * self._statistics_hub.num_envs,
            )
            if snapshot.completed_returns.size < minimum:
                return
            mean_return = float(np.mean(snapshot.completed_returns))
            alpha = self._config.zeta * max(0.0, mean_return) / target_kl

        critic_started = perf_counter()
        context_values = self._evaluate_context_values()
        self._record_scalar(
            "dspdl/critic_values_duration_s", perf_counter() - critic_started
        )
        solve_started = perf_counter()
        candidate = self._solver.solve(
            context_values=context_values,
            current_distribution=current_distribution,
            target_distribution=self._target_distribution,
            alpha=alpha,
        )
        self._record_scalar(
            "dspdl/distribution_solve_duration_s", perf_counter() - solve_started
        )
        self._record_context_value_calibration(context_values, snapshot)
        self._context_update_count += 1
        self._record_scalar("dspdl/alpha", alpha)
        self._record_scalar(
            "dspdl/update_kl",
            self._solver.kl_divergence(candidate, current_distribution),
        )
        reaches_target = (
            self._solver.kl_divergence(candidate, self._target_distribution)
            <= self._config.target_kl_stop
        )
        if not np.allclose(
            candidate, current_distribution, rtol=1e-10, atol=1e-12
        ):
            next_version = current_version + 1
            dispatch_started = perf_counter()
            self._statistics_hub.validate_version_update(next_version)
            try:
                self._distribution_state.update(candidate, version=next_version)
            except Exception:
                self._statistics_hub.cancel_version_update(next_version)
                raise
            self._statistics_hub.commit_version(next_version)
            self._record_scalar(
                "dspdl/worker_distribution_duration_s",
                perf_counter() - dispatch_started,
            )
        else:
            self._statistics_hub.clear_consumed(version=current_version)
        self._record_scalar("dspdl/update_duration_s", perf_counter() - update_started)
        if reaches_target:
            self._mark_converged()

    @staticmethod
    def _empirical_context_distribution(
        snapshot: DSPDLStatisticsSnapshot,
    ) -> NDArray[np.float64] | None:
        count = int(np.sum(snapshot.context_counts))
        if count <= 0:
            return None
        return snapshot.context_counts.astype(np.float64) / count

    def _evaluate_context_values(self) -> NDArray[np.float64]:
        tensor = self._context_observation_tensor
        if tensor is None:
            raise RuntimeError("DSPDL context observation tensor is not initialized")
        policy = cast(ActorCriticPolicy, self.model.policy)
        with th.inference_mode():
            values = policy.predict_values(tensor)
        return values.detach().cpu().numpy().reshape(-1).astype(np.float64)

    def _record_curriculum_metrics(
        self,
        *,
        snapshot: DSPDLStatisticsSnapshot,
        empirical_distribution: NDArray[np.float64] | None,
    ) -> None:
        self._record_scalar(
            "dspdl/current_to_target_kl",
            self._solver.kl_divergence(
                self._distribution_state.distribution, self._target_distribution
            ),
        )
        count = float(np.sum(snapshot.context_counts))
        self._record_scalar("dspdl/empirical_context_count", count)
        if empirical_distribution is not None:
            self._record_scalar(
                "dspdl/empirical_to_target_kl",
                self._solver.kl_divergence(
                    empirical_distribution, self._target_distribution
                ),
            )

    def _record_context_value_calibration(
        self, values: np.ndarray, snapshot: DSPDLStatisticsSnapshot
    ) -> None:
        if snapshot.completed_returns.size == 0:
            return
        indices = snapshot.completed_context_indices
        returns = snapshot.completed_returns
        predictions = values[indices]
        self._record_scalar(
            "dspdl/critic_return_mae",
            float(np.mean(np.abs(predictions - returns))),
        )
        if (
            returns.size >= 2
            and np.std(predictions) > 1e-12
            and np.std(returns) > 1e-12
        ):
            correlation = float(np.corrcoef(predictions, returns)[0, 1])
        else:
            correlation = 0.0
        self._record_scalar("dspdl/critic_return_pearson", correlation)

    def _record_scalar(self, key: str, value: float) -> None:
        logger = getattr(self.model, "logger", None)
        record = getattr(logger, "record", None)
        if callable(record):
            record(key, value)

    def _build_initial_distribution(self) -> NDArray[np.float64]:
        remaining = self._context_pool.remaining_distances_m
        gaussian = np.exp(
            -0.5
            * np.square(
                (remaining - self._config.initial_peak_remaining_distance_m)
                / self._config.initial_gaussian_std_m
            )
        )
        gaussian /= float(np.sum(gaussian))
        uniform = np.full_like(gaussian, 1.0 / gaussian.size)
        result = (
            1.0 - self._config.initial_uniform_mass
        ) * gaussian + self._config.initial_uniform_mass * uniform
        return result / float(np.sum(result))

    def _build_target_distribution(self) -> NDArray[np.float64]:
        start = np.zeros(self._context_pool.context_count, dtype=np.float64)
        start[self._start_index] = 1.0
        uniform = np.full_like(start, 1.0 / start.size)
        result = (
            1.0 - self._config.target_uniform_mass
        ) * start + self._config.target_uniform_mass * uniform
        return result / float(np.sum(result))

    def _mark_converged(self) -> None:
        if self._converged:
            return
        self._converged = True
        self._statistics_hub.disable()
        self._record_scalar("dspdl/converged", 1.0)

    def _validate_config(self) -> None:
        cfg = self._config
        if (
            not np.isfinite(cfg.initial_gaussian_std_m)
            or cfg.initial_gaussian_std_m <= 0
        ):
            raise ValueError("DSPDL initial Gaussian std must be positive")
        maximum = float(np.max(self._context_pool.remaining_distances_m))
        if not 0.0 <= cfg.initial_peak_remaining_distance_m <= maximum:
            raise ValueError("DSPDL initial peak must lie within the context range")
        if not 0.0 < cfg.initial_uniform_mass < 1.0:
            raise ValueError("DSPDL initial uniform mass must be within (0, 1)")
        if not 0.0 < cfg.target_uniform_mass < 1.0:
            raise ValueError("DSPDL target uniform mass must be within (0, 1)")
        if cfg.target_kl_stop <= 0.0 or cfg.relative_entropy_bound <= 0.0:
            raise ValueError("DSPDL KL bounds must be positive")
        if cfg.alpha_warmup_updates < 0 or cfg.zeta < 0.0:
            raise ValueError("DSPDL warm-up and zeta must be non-negative")
        if (
            cfg.update_interval_rollouts <= 0
            or cfg.min_completed_episodes <= 0
            or cfg.min_completed_episodes_per_env <= 0
        ):
            raise ValueError("DSPDL update and episode thresholds must be positive")
