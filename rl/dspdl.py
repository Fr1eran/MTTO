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

__all__ = [
    "DSPDLCallback",
    "DSPDLConfig",
    "DSPDLDistributionSolver",
    "DSPDLEpisodeAccumulator",
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


class DSPDLEpisodeAccumulator:
    """Collect one worker's DSPDL statistics without per-step IPC."""

    def __init__(self, *, context_count: int, gamma: float) -> None:
        if context_count <= 0:
            raise ValueError("context_count must be positive")
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        self._context_count = int(context_count)
        self._gamma = float(gamma)
        self._enabled = True
        self._accepted_version = 0
        self._active_context_index: int | None = None
        self._active_version: int | None = None
        self._active_return = 0.0
        self._active_discount = 1.0
        self._active_valid = False
        self._context_counts = np.zeros(self._context_count, dtype=np.int64)
        self._completed_context_indices: list[int] = []
        self._completed_returns: list[float] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def accepted_version(self) -> int:
        return self._accepted_version

    def begin_episode(self, *, context_index: int, distribution_version: int) -> None:
        if not self._enabled:
            return
        if not 0 <= int(context_index) < self._context_count:
            raise IndexError("context_index is outside the context pool")
        version = int(distribution_version)
        self._active_context_index = int(context_index)
        self._active_version = version
        self._active_return = 0.0
        self._active_discount = 1.0
        self._active_valid = version == self._accepted_version
        if self._active_valid:
            self._context_counts[int(context_index)] += 1

    def record_transition(self, reward: float, *, done: bool) -> None:
        if not self._enabled or self._active_context_index is None:
            return
        if self._active_valid:
            self._active_return += self._active_discount * float(reward)
            self._active_discount *= self._gamma
        if done:
            if self._active_valid:
                self._completed_context_indices.append(self._active_context_index)
                self._completed_returns.append(self._active_return)
            self._clear_active_episode()

    def switch_version(self, version: int) -> None:
        new_version = self.validate_version_update(version)
        self._commit_version_update(new_version)

    def validate_version_update(self, version: int) -> int:
        """Validate a version update without changing collected statistics."""
        if not isinstance(version, (int, np.integer)):
            raise TypeError("version must be an integer")
        new_version = int(version)
        if new_version <= self._accepted_version:
            raise ValueError("accumulator version must increase")
        return new_version

    def _commit_version_update(self, version: int) -> None:
        new_version = int(version)
        self._accepted_version = new_version
        self._context_counts.fill(0)
        self._completed_context_indices.clear()
        self._completed_returns.clear()
        if self._active_version != new_version:
            self._active_valid = False

    def drain(self, *, version: int) -> dict[str, object]:
        if int(version) != self._accepted_version:
            raise ValueError("requested statistics version does not match accumulator")
        if not self._enabled:
            return self._empty_payload(version)
        payload: dict[str, object] = {
            "version": self._accepted_version,
            "context_counts": self._context_counts.copy(),
            "completed_context_indices": np.asarray(
                self._completed_context_indices, dtype=np.int64
            ),
            "completed_returns": np.asarray(self._completed_returns, dtype=np.float64),
        }
        self._context_counts.fill(0)
        self._completed_context_indices.clear()
        self._completed_returns.clear()
        return payload

    def disable(self) -> None:
        self._enabled = False
        self._context_counts = np.empty(0, dtype=np.int64)
        self._completed_context_indices.clear()
        self._completed_returns.clear()
        self._clear_active_episode()

    def _clear_active_episode(self) -> None:
        self._active_context_index = None
        self._active_version = None
        self._active_return = 0.0
        self._active_discount = 1.0
        self._active_valid = False

    def _empty_payload(self, version: int) -> dict[str, object]:
        return {
            "version": int(version),
            "context_counts": np.zeros(self._context_count, dtype=np.int64),
            "completed_context_indices": np.empty(0, dtype=np.int64),
            "completed_returns": np.empty(0, dtype=np.float64),
        }


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
    """Coordinate DSPDL updates using worker-local episode statistics."""

    def __init__(
        self,
        *,
        context_pool: ContextPool,
        context_observations: NDArray[np.float32],
        config: DSPDLConfig,
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
        self._context_pool = context_pool
        self._context_observations = observations
        self._context_observation_tensor: th.Tensor | None = None
        self._config = config
        self._validate_config()
        self._solver = solver or DSPDLDistributionSolver(
            relative_entropy_bound=config.relative_entropy_bound
        )
        self._start_index = int(np.argmax(context_pool.remaining_distances_m))
        self._target_distribution = self._build_target_distribution()
        self._current_distribution = self._build_initial_distribution()
        self._version = 0
        self._rollouts_since_update_attempt = 0
        self._context_update_count = 0
        self._context_counts = np.zeros(context_pool.context_count, dtype=np.int64)
        self._completed_context_indices: list[int] = []
        self._completed_returns: list[float] = []
        self._num_envs = 1
        self._converged = False

    def initial_context_distribution(self) -> NDArray[np.float64]:
        return self._current_distribution.copy()

    @property
    def target_context_distribution(self) -> NDArray[np.float64]:
        return self._target_distribution.copy()

    @override
    def _on_training_start(self) -> None:
        policy = cast(ActorCriticPolicy, self.model.policy)
        tensor, _ = policy.obs_to_tensor(self._context_observations)
        self._context_observation_tensor = tensor
        self._num_envs = int(self.training_env.num_envs)
        self._record_scalar("dspdl/converged", 0.0)
        self._record_curriculum_metrics(empirical_distribution=None)

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
        self._clear_parent_statistics()

    def _maybe_update_curriculum(self) -> None:
        if self._converged:
            return
        update_started = perf_counter()
        self._drain_worker_statistics()
        empirical = self._empirical_context_distribution()
        self._record_curriculum_metrics(empirical_distribution=empirical)

        target_kl = self._solver.kl_divergence(
            self._current_distribution, self._target_distribution
        )
        if target_kl <= self._config.target_kl_stop:
            self._mark_converged()
            return

        alpha = 0.0
        if self._context_update_count >= self._config.alpha_warmup_updates:
            minimum = max(
                self._config.min_completed_episodes,
                self._config.min_completed_episodes_per_env * self._num_envs,
            )
            if len(self._completed_returns) < minimum:
                return
            mean_return = float(np.mean(self._completed_returns))
            alpha = self._config.zeta * max(0.0, mean_return) / target_kl

        critic_started = perf_counter()
        context_values = self._evaluate_context_values()
        self._record_scalar(
            "dspdl/critic_values_duration_s", perf_counter() - critic_started
        )
        solve_started = perf_counter()
        candidate = self._solver.solve(
            context_values=context_values,
            current_distribution=self._current_distribution,
            target_distribution=self._target_distribution,
            alpha=alpha,
        )
        self._record_scalar(
            "dspdl/distribution_solve_duration_s", perf_counter() - solve_started
        )
        self._record_context_value_calibration(context_values)
        self._context_update_count += 1
        self._clear_parent_statistics()
        self._record_scalar("dspdl/alpha", alpha)
        self._record_scalar(
            "dspdl/update_kl",
            self._solver.kl_divergence(candidate, self._current_distribution),
        )
        reaches_target = (
            self._solver.kl_divergence(candidate, self._target_distribution)
            <= self._config.target_kl_stop
        )
        if not np.allclose(
            candidate, self._current_distribution, rtol=1e-10, atol=1e-12
        ):
            self._version += 1
            dispatch_started = perf_counter()
            self.training_env.env_method(
                "set_dspdl_distribution", candidate, version=self._version
            )
            self._current_distribution = candidate
            self._record_scalar(
                "dspdl/worker_distribution_duration_s",
                perf_counter() - dispatch_started,
            )
        self._record_scalar("dspdl/update_duration_s", perf_counter() - update_started)
        if reaches_target:
            self._mark_converged()

    def _drain_worker_statistics(self) -> None:
        payloads = self.training_env.env_method(
            "drain_dspdl_statistics", version=self._version
        )
        for raw_payload in payloads:
            if not isinstance(raw_payload, dict):
                raise TypeError("DSPDL statistics payload must be a dictionary")
            if int(raw_payload["version"]) != self._version:
                raise ValueError("DSPDL worker statistics version mismatch")
            counts = np.asarray(raw_payload["context_counts"], dtype=np.int64)
            indices = np.asarray(
                raw_payload["completed_context_indices"], dtype=np.int64
            )
            returns = np.asarray(raw_payload["completed_returns"], dtype=np.float64)
            if counts.shape != self._context_counts.shape:
                raise ValueError("DSPDL context count payload has an invalid shape")
            if indices.shape != returns.shape:
                raise ValueError("DSPDL completed episode payload shapes differ")
            self._context_counts += counts
            self._completed_context_indices.extend(indices.tolist())
            self._completed_returns.extend(returns.tolist())

    def _empirical_context_distribution(self) -> NDArray[np.float64] | None:
        count = int(np.sum(self._context_counts))
        if count <= 0:
            return None
        return self._context_counts.astype(np.float64) / count

    def _evaluate_context_values(self) -> NDArray[np.float64]:
        tensor = self._context_observation_tensor
        if tensor is None:
            raise RuntimeError("DSPDL context observation tensor is not initialized")
        policy = cast(ActorCriticPolicy, self.model.policy)
        with th.inference_mode():
            values = policy.predict_values(tensor)
        return values.detach().cpu().numpy().reshape(-1).astype(np.float64)

    def _record_curriculum_metrics(
        self, *, empirical_distribution: NDArray[np.float64] | None
    ) -> None:
        self._record_scalar(
            "dspdl/current_to_target_kl",
            self._solver.kl_divergence(
                self._current_distribution, self._target_distribution
            ),
        )
        count = float(np.sum(self._context_counts))
        self._record_scalar("dspdl/empirical_context_count", count)
        if empirical_distribution is not None:
            self._record_scalar(
                "dspdl/empirical_to_target_kl",
                self._solver.kl_divergence(
                    empirical_distribution, self._target_distribution
                ),
            )

    def _record_context_value_calibration(self, values: np.ndarray) -> None:
        if not self._completed_returns:
            return
        indices = np.asarray(self._completed_context_indices, dtype=np.int64)
        returns = np.asarray(self._completed_returns, dtype=np.float64)
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

    def _clear_parent_statistics(self) -> None:
        self._context_counts.fill(0)
        self._completed_context_indices.clear()
        self._completed_returns.clear()

    def _mark_converged(self) -> None:
        if self._converged:
            return
        self._converged = True
        self._clear_parent_statistics()
        self.training_env.env_method("disable_dspdl_accumulator")
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
