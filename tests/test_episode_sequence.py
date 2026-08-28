import numpy as np
import pytest

from rl.reward_diagnostics import REWARD_NAMES
from rl.training_analysis.collect import (
    RewardDiagnosticsArtifact,
    extract_complete_episode_sequence,
)
from rl.training_analysis.process import trailing_moving_average


def _artifact_with_interleaved_workers() -> RewardDiagnosticsArtifact:
    reward_sums = np.zeros((3, len(REWARD_NAMES)), dtype=np.float64)
    reward_sums[:, -1] = [30.0, 20.0, 10.0]
    return RewardDiagnosticsArtifact(
        reward_names=tuple(REWARD_NAMES),
        rollout_end_step=np.asarray([20], dtype=np.int64),
        rollout_transition_count=np.asarray([60], dtype=np.int64),
        rollout_reward_sum=reward_sums.sum(axis=0, keepdims=True),
        rollout_reward_abs_sum=np.abs(reward_sums).sum(axis=0, keepdims=True),
        rollout_reward_nonzero_count=np.count_nonzero(
            reward_sums, axis=0, keepdims=True
        ),
        rollout_reward_cross_product=np.asarray([reward_sums.T @ reward_sums]),
        episode_end_step=np.asarray([20, 10, 10], dtype=np.int64),
        episode_worker_rank=np.asarray([0, 1, 0], dtype=np.int16),
        episode_index=np.asarray([1, 2, 3], dtype=np.int64),
        episode_length=np.asarray([30, 20, 10], dtype=np.int32),
        episode_terminated=np.asarray([True, True, True]),
        episode_truncated=np.asarray([False, False, False]),
        episode_complete=np.asarray([True, True, True]),
        episode_violation_code=np.asarray([0, 0, 0], dtype=np.int8),
        episode_reward_sums=reward_sums,
    )


def test_complete_episode_sequence_merges_workers_without_collapsing_episodes() -> None:
    sequence = extract_complete_episode_sequence(_artifact_with_interleaved_workers())

    np.testing.assert_array_equal(sequence.episode_number, [1, 2, 3])
    np.testing.assert_allclose(sequence.total_reward, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(sequence.length, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(sequence.violation_code, [0, 0, 0])


def test_trailing_moving_average_matches_sb3_window_alignment() -> None:
    np.testing.assert_allclose(
        trailing_moving_average(np.asarray([1.0, 2.0, 3.0, 4.0]), 3),
        [2.0, 3.0],
    )
    np.testing.assert_allclose(
        trailing_moving_average(np.asarray([1.0, 2.0]), 1), [1.0, 2.0]
    )
    assert trailing_moving_average(np.asarray([1.0, 2.0]), 3).size == 0
    with pytest.raises(ValueError, match="window"):
        _ = trailing_moving_average(np.asarray([1.0]), 0)
