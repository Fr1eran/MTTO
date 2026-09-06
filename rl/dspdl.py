"""DSPDL curriculum control over a finite context pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, override

import numpy as np
import torch as th
from numpy.typing import NDArray
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

from rl.context_pool import ContextPool
from rl.context_sampler import CurriculumDistributionState
from rl.dspdl_distribution import DSPDLDistributionSolver

__all__ = [
    "DSPDLCallback",
    "DSPDLStatisticsHub",
    "DSPDLStatisticsSnapshot",
    "DSPDL_ALPHA_WARMUP_UPDATES",
    "DSPDL_RELATIVE_ENTROPY_BOUND",
    "DSPDL_TARGET_KL_STOP",
    "DSPDL_TARGET_UNIFORM_MASS",
    "DSPDL_UPDATE_INTERVAL_ROLLOUTS",
    "DSPDL_ZETA",
    "dspdl_protocol_parameters",
]

DSPDL_TARGET_UNIFORM_MASS = 0.05
DSPDL_TARGET_KL_STOP = 0.1
DSPDL_ALPHA_WARMUP_UPDATES = 4
DSPDL_RELATIVE_ENTROPY_BOUND = 0.01
DSPDL_ZETA = 4.0
DSPDL_UPDATE_INTERVAL_ROLLOUTS = 4


def dspdl_protocol_parameters() -> dict[str, float | int | str]:
    """Return the immutable DSPDL protocol parameters for metadata auditing."""
    return {
        "target_uniform_mass": DSPDL_TARGET_UNIFORM_MASS,
        "target_kl_stop": DSPDL_TARGET_KL_STOP,
        "alpha_warmup_updates": DSPDL_ALPHA_WARMUP_UPDATES,
        "relative_entropy_bound": DSPDL_RELATIVE_ENTROPY_BOUND,
        "zeta": DSPDL_ZETA,
        "update_interval_rollouts": DSPDL_UPDATE_INTERVAL_ROLLOUTS,
        "context_value_estimator": "importance_weighted_samples",
    }


@dataclass(frozen=True, slots=True)
class DSPDLStatisticsSnapshot:
    """Immutable snapshot of centralized on-policy return statistics."""

    version: int
    context_counts: NDArray[np.int64]
    completed_context_indices: NDArray[np.int64]
    completed_returns: NDArray[np.float64]
    rollout_returns: NDArray[np.float64]


class DSPDLStatisticsHub:
    """Collect discounted rollout and episode returns for all environment workers."""

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
        self._rollout_returns_by_env = np.zeros(self._num_envs, dtype=np.float64)
        self._rollout_discounts_by_env = np.ones(self._num_envs, dtype=np.float64)
        self._rollout_has_samples = np.zeros(self._num_envs, dtype=np.bool_)
        self._context_counts = np.zeros(self._context_count, dtype=np.int64)
        self._completed_context_indices: list[int] = []
        self._completed_returns: list[float] = []
        self._rollout_returns: list[float] = []

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
            self._rollout_returns_by_env[rank] += (
                self._rollout_discounts_by_env[rank] * value
            )
            self._rollout_discounts_by_env[rank] *= self._gamma
            self._rollout_has_samples[rank] = True
        if done:
            if self._active_valid[rank]:
                self._completed_context_indices.append(
                    int(self._active_context_indices[rank])
                )
                self._completed_returns.append(float(self._active_returns[rank]))
            self._clear_active_episode(rank)

    def finish_rollout(self, *, version: int) -> None:
        """Seal one PPO rollout into per-worker discounted return samples."""
        self._validate_requested_version(version)
        if not self._enabled:
            return
        sampled_ranks = np.flatnonzero(self._rollout_has_samples)
        self._rollout_returns.extend(
            float(self._rollout_returns_by_env[rank]) for rank in sampled_ranks
        )
        self._rollout_returns_by_env.fill(0.0)
        self._rollout_discounts_by_env.fill(1.0)
        self._rollout_has_samples.fill(False)

    def snapshot(self, *, version: int) -> DSPDLStatisticsSnapshot:
        self._validate_requested_version(version)
        if self._enabled:
            counts = self._context_counts.copy()
            indices = np.asarray(self._completed_context_indices, dtype=np.int64)
            returns = np.asarray(self._completed_returns, dtype=np.float64)
            rollout_returns = np.asarray(self._rollout_returns, dtype=np.float64)
        else:
            counts = np.zeros(self._context_count, dtype=np.int64)
            indices = np.empty(0, dtype=np.int64)
            returns = np.empty(0, dtype=np.float64)
            rollout_returns = np.empty(0, dtype=np.float64)
        for values in (counts, indices, returns, rollout_returns):
            values.flags.writeable = False
        return DSPDLStatisticsSnapshot(
            version=self._accepted_version,
            context_counts=counts,
            completed_context_indices=indices,
            completed_returns=returns,
            rollout_returns=rollout_returns,
        )

    def clear_consumed(self, *, version: int) -> None:
        self._validate_requested_version(version)
        if self._enabled:
            self._clear_completed_statistics()
            self._rebuild_active_context_counts()

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
        self._rollout_returns_by_env = np.empty(0, dtype=np.float64)
        self._rollout_discounts_by_env = np.empty(0, dtype=np.float64)
        self._rollout_has_samples = np.empty(0, dtype=np.bool_)
        self._context_counts = np.empty(0, dtype=np.int64)
        self._completed_context_indices.clear()
        self._completed_returns.clear()
        self._rollout_returns.clear()

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
        self._rollout_returns.clear()
        self._rollout_returns_by_env.fill(0.0)
        self._rollout_discounts_by_env.fill(1.0)
        self._rollout_has_samples.fill(False)

    def _rebuild_active_context_counts(self) -> None:
        if not self._context_counts.size:
            return
        valid_mask = (
            self._active_valid
            & (self._active_versions == self._accepted_version)
            & (self._active_context_indices >= 0)
        )
        active_contexts = self._active_context_indices[valid_mask]
        if active_contexts.size > 0:
            np.add.at(self._context_counts, active_contexts, 1)

    def _clear_active_episode(self, rank: int) -> None:
        self._active_context_indices[rank] = -1
        self._active_versions[rank] = -1
        self._active_returns[rank] = 0.0
        self._active_discounts[rank] = 1.0
        self._active_valid[rank] = False


class DSPDLCallback(BaseCallback):
    """Update the curriculum from PPO value estimates at sampled context starts."""

    def __init__(
        self,
        *,
        context_pool: ContextPool,
        context_observations: NDArray[np.float32],
        statistics_hub: DSPDLStatisticsHub,
        context_punctuality_potentials: NDArray[np.float64] | None = None,
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
            raise ValueError("statistics hub context count must match the context pool")
        self._context_pool = context_pool
        self._context_punctuality_potentials = (
            np.zeros(context_pool.context_count, dtype=np.float64)
            if context_punctuality_potentials is None
            else np.array(context_punctuality_potentials, dtype=np.float64, copy=True)
        )
        if self._context_punctuality_potentials.shape != (
            context_pool.context_count,
        ) or not np.all(np.isfinite(self._context_punctuality_potentials)):
            raise ValueError(
                "context punctuality potentials must be finite, one per context"
            )
        self._context_observations = observations
        self._context_observation_tensor: th.Tensor | None = None
        self._statistics_hub = statistics_hub
        self._solver = DSPDLDistributionSolver(
            relative_entropy_bound=DSPDL_RELATIVE_ENTROPY_BOUND
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
            raise ValueError("statistics hub environment count must match training env")
        if self._statistics_hub.accepted_version != self._distribution_state.version:
            raise ValueError(
                "statistics hub version must match curriculum distribution"
            )
        self._record_scalar("dspdl/converged", 0.0)
        snapshot = self._statistics_hub.snapshot(version=0)
        self._record_curriculum_metrics(snapshot, empirical_distribution=None)

    @override
    def _on_rollout_start(self) -> None:
        if (
            not self._converged
            and self._rollouts_since_update_attempt
            >= DSPDL_UPDATE_INTERVAL_ROLLOUTS
        ):
            self._rollouts_since_update_attempt = 0
            self._maybe_update_curriculum()

    @override
    def _on_rollout_end(self) -> None:
        if not self._converged:
            self._statistics_hub.finish_rollout(
                version=self._distribution_state.version
            )
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
        version = self._distribution_state.version
        if self._statistics_hub.accepted_version != version:
            raise ValueError(
                "statistics hub version must match curriculum distribution"
            )
        snapshot = self._statistics_hub.snapshot(version=version)
        empirical = self._empirical_context_distribution(snapshot)
        self._record_curriculum_metrics(snapshot, empirical_distribution=empirical)
        current = self._distribution_state.distribution
        target_kl = self._solver.kl_divergence(current, self._target_distribution)
        if target_kl <= DSPDL_TARGET_KL_STOP:
            self._mark_converged()
            return

        rollout_returns = snapshot.rollout_returns
        sample_count = int(rollout_returns.size)
        self._record_scalar("dspdl/alpha_return_sample_count", float(sample_count))
        context_sample_count = int(np.sum(snapshot.context_counts))
        sampled_indices = np.flatnonzero(snapshot.context_counts)
        self._record_scalar("dspdl/context_sample_count", float(context_sample_count))
        self._record_scalar(
            "dspdl/context_unique_sample_count", float(sampled_indices.size)
        )
        if sample_count == 0 or context_sample_count == 0:
            return
        mean_return = float(np.mean(rollout_returns))
        self._record_scalar("dspdl/alpha_mean_discounted_return", mean_return)

        alpha = 0.0
        if self._context_update_count >= DSPDL_ALPHA_WARMUP_UPDATES:
            alpha = DSPDL_ZETA * max(0.0, mean_return) / target_kl

        context_values, value_indices, value_predictions = (
            self._estimate_context_coefficients(
                snapshot=snapshot,
                current_distribution=current,
            )
        )
        candidate = self._solver.solve(
            context_values=context_values,
            current_distribution=current,
            target_distribution=self._target_distribution,
            alpha=alpha,
        )
        self._record_context_value_calibration(
            value_indices,
            value_predictions,
            snapshot,
        )
        self._context_update_count += 1
        self._record_scalar("dspdl/alpha", alpha)
        self._record_scalar(
            "dspdl/update_kl", self._solver.kl_divergence(candidate, current)
        )
        reaches_target = (
            self._solver.kl_divergence(candidate, self._target_distribution)
            <= DSPDL_TARGET_KL_STOP
        )
        if not np.allclose(candidate, current, rtol=1e-10, atol=1e-12):
            next_version = version + 1
            self._statistics_hub.validate_version_update(next_version)
            try:
                self._distribution_state.update(candidate, version=next_version)
            except Exception:
                self._statistics_hub.cancel_version_update(next_version)
                raise
            self._statistics_hub.commit_version(next_version)
        else:
            self._statistics_hub.clear_consumed(version=version)
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

    def _estimate_context_coefficients(
        self,
        *,
        snapshot: DSPDLStatisticsSnapshot,
        current_distribution: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        """Build importance-weighted coefficients for sampled contexts."""
        context_count = self._context_pool.context_count
        sampled_indices = np.flatnonzero(snapshot.context_counts).astype(np.int64)
        if sampled_indices.size == 0:
            raise RuntimeError("importance-weighted context estimate requires samples")

        sample_count = int(np.sum(snapshot.context_counts))
        if sample_count <= 0:
            raise RuntimeError("importance-weighted context estimate requires samples")
        probabilities = np.asarray(
            current_distribution[sampled_indices], dtype=np.float64
        )
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
            raise RuntimeError(
                "sampled contexts must have finite positive sampling probability"
            )
        values = self._evaluate_context_values(sampled_indices)
        importance_weights = snapshot.context_counts[sampled_indices].astype(
            np.float64
        ) / (float(sample_count) * probabilities)
        self._record_scalar(
            "dspdl/importance_weight_mean", float(np.mean(importance_weights))
        )
        self._record_scalar(
            "dspdl/importance_weight_max", float(np.max(importance_weights))
        )
        weight_sum = float(np.sum(importance_weights))
        weight_square_sum = float(np.sum(np.square(importance_weights)))
        ess = (
            weight_sum * weight_sum / weight_square_sum
            if weight_square_sum > 0.0
            else 0.0
        )
        self._record_scalar("dspdl/importance_weight_ess", ess)
        self._record_scalar(
            "dspdl/importance_weight_ess_ratio",
            ess / float(max(1, importance_weights.size)),
        )
        mean_weight = float(np.mean(importance_weights))
        self._record_scalar(
            "dspdl/importance_weight_max_to_mean",
            float(np.max(importance_weights)) / max(mean_weight, 1e-12),
        )
        coefficients = np.zeros(context_count, dtype=np.float64)
        coefficients[sampled_indices] = values * importance_weights
        return coefficients, sampled_indices, values

    def _evaluate_context_values(
        self, indices: NDArray[np.int64] | None = None
    ) -> NDArray[np.float64]:
        tensor = self._context_observation_tensor
        if tensor is None:
            raise RuntimeError("DSPDL context tensor is not initialized")
        if indices is None:
            indices = np.arange(self._context_pool.context_count, dtype=np.int64)
        selected_indices = np.asarray(indices, dtype=np.int64)
        if selected_indices.ndim != 1:
            raise ValueError("context value indices must be one-dimensional")
        if np.any(selected_indices < 0) or np.any(
            selected_indices >= self._context_pool.context_count
        ):
            raise IndexError("context value index is outside the context pool")
        if selected_indices.size == 0:
            return np.empty(0, dtype=np.float64)
        policy = cast(ActorCriticPolicy, self.model.policy)
        index_tensor = th.as_tensor(
            selected_indices,
            dtype=th.long,
            device=tensor.device,
        )
        with th.inference_mode():
            values = policy.predict_values(tensor.index_select(0, index_tensor))
        return (
            values.detach().cpu().numpy().reshape(-1).astype(np.float64)
            + self._context_punctuality_potentials[selected_indices]
        )

    def _record_curriculum_metrics(
        self,
        snapshot: DSPDLStatisticsSnapshot,
        *,
        empirical_distribution: NDArray[np.float64] | None,
    ) -> None:
        self._record_scalar(
            "dspdl/current_to_target_kl",
            self._solver.kl_divergence(
                self._distribution_state.distribution, self._target_distribution
            ),
        )
        self._record_scalar(
            "dspdl/empirical_context_count", float(np.sum(snapshot.context_counts))
        )
        if empirical_distribution is not None:
            self._record_scalar(
                "dspdl/empirical_to_target_kl",
                self._solver.kl_divergence(
                    empirical_distribution, self._target_distribution
                ),
            )

    def _record_context_value_calibration(
        self,
        value_indices: NDArray[np.int64],
        values: NDArray[np.float64],
        snapshot: DSPDLStatisticsSnapshot,
    ) -> None:
        if snapshot.completed_returns.size == 0:
            return
        positions = np.full(self._context_pool.context_count, -1, dtype=np.int64)
        positions[value_indices] = np.arange(value_indices.size, dtype=np.int64)
        completed_positions = positions[snapshot.completed_context_indices]
        if np.any(completed_positions < 0):
            raise RuntimeError("completed context is missing a value estimate")
        predictions = values[completed_positions]
        returns = snapshot.completed_returns
        self._record_scalar(
            "dspdl/value_return_mae", float(np.mean(np.abs(predictions - returns)))
        )
        correlation = 0.0
        if (
            returns.size >= 2
            and np.std(predictions) > 1e-12
            and np.std(returns) > 1e-12
        ):
            correlation = float(np.corrcoef(predictions, returns)[0, 1])
        self._record_scalar("dspdl/value_return_pearson", correlation)

    def _record_scalar(self, key: str, value: float) -> None:
        logger = getattr(self.model, "logger", None)
        record = getattr(logger, "record", None)
        if callable(record):
            record(key, value)

    def _build_initial_distribution(self) -> NDArray[np.float64]:
        context_count = self._context_pool.context_count
        return np.full(context_count, 1.0 / context_count, dtype=np.float64)

    def _build_target_distribution(self) -> NDArray[np.float64]:
        start = np.zeros(self._context_pool.context_count, dtype=np.float64)
        start[self._start_index] = 1.0
        uniform = np.full_like(start, 1.0 / start.size)
        result = (
            1.0 - DSPDL_TARGET_UNIFORM_MASS
        ) * start + DSPDL_TARGET_UNIFORM_MASS * uniform
        return result / float(np.sum(result))

    def _mark_converged(self) -> None:
        if self._converged:
            return
        self._converged = True
        self._statistics_hub.disable()
        self._record_scalar("dspdl/converged", 1.0)
