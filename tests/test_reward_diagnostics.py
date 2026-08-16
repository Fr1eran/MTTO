import numpy as np

from rl.reward_calculator import RewardBreakdown
from rl.reward_diagnostics import (
    REWARD_SIGNAL_COUNT,
    TOTAL_REWARD_INDEX,
    RewardDiagnosticsAccumulator,
)


def _reward(safety: float, energy: float) -> RewardBreakdown:
    return RewardBreakdown(
        safety=safety,
        energy=energy,
        total=safety + energy,
    )


def test_accumulator_emits_rollout_moments_and_complete_episode() -> None:
    accumulator = RewardDiagnosticsAccumulator(worker_rank=2, rollout_capacity=2)
    accumulator.record(_reward(1.0, -0.25), terminated=False, truncated=False)
    accumulator.record(_reward(2.0, -0.5), terminated=True, truncated=False)

    batch = accumulator.drain()

    np.testing.assert_array_equal(batch["transition_count"], [2])
    np.testing.assert_allclose(
        batch["reward_sum"][[0, 1, TOTAL_REWARD_INDEX]], [3.0, -0.75, 2.25]
    )
    np.testing.assert_array_equal(batch["episode_worker_rank"], [2])
    np.testing.assert_array_equal(batch["episode_length"], [2])
    np.testing.assert_array_equal(batch["episode_complete"], [True])
    assert batch["reward_cross_product"].shape == (
        REWARD_SIGNAL_COUNT,
        REWARD_SIGNAL_COUNT,
    )


def test_finalize_emits_partial_episode_without_recounting_transitions() -> None:
    accumulator = RewardDiagnosticsAccumulator(worker_rank=0, rollout_capacity=1)
    accumulator.record(_reward(1.0, 0.0), terminated=False, truncated=False)
    first = accumulator.drain()
    final = accumulator.drain(finalize=True)

    np.testing.assert_array_equal(first["transition_count"], [1])
    np.testing.assert_array_equal(final["transition_count"], [0])
    np.testing.assert_array_equal(final["episode_complete"], [False])
    np.testing.assert_array_equal(final["episode_length"], [1])
