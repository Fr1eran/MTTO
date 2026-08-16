"""Worker-local reward diagnostics and the versioned artifact contract."""

from __future__ import annotations

from typing import Final, TypedDict

import numpy as np
from numpy.typing import NDArray

from rl.reward_calculator import RewardBreakdown

REWARD_DIAGNOSTICS_SCHEMA_VERSION: Final[int] = 2
REWARD_NAMES: Final[tuple[str, ...]] = (
    "safety",
    "energy",
    "comfort",
    "terminal_stopping",
    "terminal_punctuality",
    "survival",
    "truncation",
    "total",
)
REWARD_SIGNAL_COUNT: Final[int] = len(REWARD_NAMES)
TOTAL_REWARD_INDEX: Final[int] = REWARD_NAMES.index("total")


class RewardDiagnosticsBatch(TypedDict):
    """Compact statistics drained from one worker at a rollout boundary."""

    transition_count: NDArray[np.int64]
    reward_sum: NDArray[np.float64]
    reward_abs_sum: NDArray[np.float64]
    reward_nonzero_count: NDArray[np.int64]
    reward_cross_product: NDArray[np.float64]
    episode_end_worker_step: NDArray[np.int64]
    episode_worker_rank: NDArray[np.int16]
    episode_index: NDArray[np.int64]
    episode_length: NDArray[np.int32]
    episode_terminated: NDArray[np.bool_]
    episode_truncated: NDArray[np.bool_]
    episode_complete: NDArray[np.bool_]
    episode_reward_sums: NDArray[np.float64]


class RewardDiagnosticsAccumulator:
    """Accumulate transition moments and raw episode returns inside one worker."""

    def __init__(self, *, worker_rank: int, rollout_capacity: int) -> None:
        self.worker_rank = int(worker_rank)
        if not np.iinfo(np.int16).min <= self.worker_rank <= np.iinfo(np.int16).max:
            raise ValueError("worker_rank does not fit the artifact's int16 schema")
        capacity = int(rollout_capacity)
        if capacity <= 0:
            raise ValueError("rollout_capacity must be positive")
        self._transition_rewards = np.empty(
            (capacity, REWARD_SIGNAL_COUNT), dtype=np.float32
        )
        self._transition_count = 0
        self._worker_transition_step = 0
        self._episode_index = 0
        self._episode_length = 0
        self._episode_reward_sum = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.float64)
        self._episodes: list[
            tuple[int, int, int, int, bool, bool, bool, NDArray[np.float64]]
        ] = []

    def _ensure_capacity(self) -> None:
        if self._transition_count < self._transition_rewards.shape[0]:
            return
        grown = np.empty(
            (self._transition_rewards.shape[0] * 2, REWARD_SIGNAL_COUNT),
            dtype=np.float32,
        )
        grown[: self._transition_count] = self._transition_rewards
        self._transition_rewards = grown

    def record(
        self,
        reward: RewardBreakdown,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._ensure_capacity()
        vector = self._transition_rewards[self._transition_count]
        vector[0] = reward.safety
        vector[1] = reward.energy
        vector[2] = reward.comfort
        vector[3] = reward.terminal_stopping
        vector[4] = reward.terminal_punctuality
        vector[5] = reward.survival
        vector[6] = reward.truncation
        vector[7] = reward.total
        self._transition_count += 1
        self._worker_transition_step += 1
        self._episode_length += 1
        self._episode_reward_sum += vector
        if terminated or truncated:
            self._finish_episode(
                terminated=bool(terminated),
                truncated=bool(truncated),
                complete=True,
            )

    def _finish_episode(
        self, *, terminated: bool, truncated: bool, complete: bool
    ) -> None:
        if self._episode_length <= 0:
            return
        self._episodes.append(
            (
                self._worker_transition_step,
                self.worker_rank,
                self._episode_index,
                self._episode_length,
                terminated,
                truncated,
                complete,
                self._episode_reward_sum.copy(),
            )
        )
        self._episode_index += 1
        self._episode_length = 0
        self._episode_reward_sum.fill(0.0)

    def drain(self, *, finalize: bool = False) -> RewardDiagnosticsBatch:
        if finalize:
            self._finish_episode(
                terminated=False,
                truncated=False,
                complete=False,
            )

        count = self._transition_count
        if count:
            matrix = self._transition_rewards[:count].astype(np.float64)
            reward_sum = matrix.sum(axis=0)
            reward_abs_sum = np.abs(matrix).sum(axis=0)
            reward_nonzero_count = np.count_nonzero(matrix, axis=0).astype(np.int64)
            reward_cross_product = matrix.T @ matrix
        else:
            reward_sum = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.float64)
            reward_abs_sum = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.float64)
            reward_nonzero_count = np.zeros(REWARD_SIGNAL_COUNT, dtype=np.int64)
            reward_cross_product = np.zeros(
                (REWARD_SIGNAL_COUNT, REWARD_SIGNAL_COUNT), dtype=np.float64
            )
        self._transition_count = 0

        episode_count = len(self._episodes)
        if episode_count:
            end_steps = np.fromiter(
                (item[0] for item in self._episodes), dtype=np.int64
            )
            worker_ranks = np.fromiter(
                (item[1] for item in self._episodes), dtype=np.int16
            )
            episode_indices = np.fromiter(
                (item[2] for item in self._episodes), dtype=np.int64
            )
            lengths = np.fromiter((item[3] for item in self._episodes), dtype=np.int32)
        else:
            end_steps = np.empty(0, dtype=np.int64)
            worker_ranks = np.empty(0, dtype=np.int16)
            episode_indices = np.empty(0, dtype=np.int64)
            lengths = np.empty(0, dtype=np.int32)

        # Build the remaining episode arrays explicitly; the episode count is
        # small, while the transition hot path remains allocation-free.
        if episode_count:
            terminated_values = np.asarray(
                [item[4] for item in self._episodes], dtype=np.bool_
            )
            truncated_values = np.asarray(
                [item[5] for item in self._episodes], dtype=np.bool_
            )
            complete_values = np.asarray(
                [item[6] for item in self._episodes], dtype=np.bool_
            )
            episode_rewards = np.stack([item[7] for item in self._episodes]).astype(
                np.float64, copy=False
            )
        else:
            terminated_values = np.empty(0, dtype=np.bool_)
            truncated_values = np.empty(0, dtype=np.bool_)
            complete_values = np.empty(0, dtype=np.bool_)
            episode_rewards = np.empty((0, REWARD_SIGNAL_COUNT), dtype=np.float64)
        self._episodes.clear()

        return {
            "transition_count": np.asarray([count], dtype=np.int64),
            "reward_sum": reward_sum,
            "reward_abs_sum": reward_abs_sum,
            "reward_nonzero_count": reward_nonzero_count,
            "reward_cross_product": reward_cross_product,
            "episode_end_worker_step": end_steps,
            "episode_worker_rank": worker_ranks,
            "episode_index": episode_indices,
            "episode_length": lengths,
            "episode_terminated": terminated_values,
            "episode_truncated": truncated_values,
            "episode_complete": complete_values,
            "episode_reward_sums": episode_rewards,
        }
