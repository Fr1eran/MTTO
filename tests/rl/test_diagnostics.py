"""Reward diagnostics, safety-truncation histograms and episode-sequence extraction."""

import numpy as np
import pytest

from mtto.rl.diagnostics import (
    REWARD_DIAGNOSTICS_SCHEMA_VERSION,
    REWARD_NAMES,
    REWARD_SIGNAL_COUNT,
    TOTAL_REWARD_INDEX,
    RewardDiagnostics,
    RewardDiagnosticsAccumulator,
    SafetyTruncationBuffer,
    SafetyTruncationHistogram,
)
from mtto.rl.rewards import RewardBreakdown
from mtto.rl.state import TerminationReason
from mtto.rl.training_analysis.collect import extract_complete_episode_sequence
from mtto.rl.training_analysis.process import trailing_moving_average


def _reward(safety: float, energy: float) -> RewardBreakdown:
    return RewardBreakdown(
        safety=safety,
        energy=energy,
        total=safety + energy,
    )


def test_accumulator_emits_rollout_moments_and_complete_episode() -> None:
    accumulator = RewardDiagnosticsAccumulator(worker_rank=2, rollout_capacity=2)
    accumulator.record(_reward(1.0, -0.25), termination_reason=None)
    accumulator.record(
        _reward(2.0, -0.5),
        termination_reason=TerminationReason.OVER_UPPER_LIMIT,
    )

    batch = accumulator.drain()

    np.testing.assert_array_equal(batch["transition_count"], [2])
    np.testing.assert_allclose(
        batch["reward_sum"][[0, 1, TOTAL_REWARD_INDEX]], [3.0, -0.75, 2.25]
    )
    np.testing.assert_array_equal(batch["episode_worker_rank"], [2])
    np.testing.assert_array_equal(batch["episode_length"], [2])
    np.testing.assert_array_equal(batch["episode_complete"], [True])
    np.testing.assert_array_equal(batch["episode_termination_reason"], [5])
    assert batch["reward_cross_product"].shape == (
        REWARD_SIGNAL_COUNT,
        REWARD_SIGNAL_COUNT,
    )


def test_finalize_emits_partial_episode_without_recounting_transitions() -> None:
    accumulator = RewardDiagnosticsAccumulator(worker_rank=0, rollout_capacity=1)
    accumulator.record(_reward(1.0, 0.0), termination_reason=None)
    first = accumulator.drain()
    final = accumulator.drain(finalize=True)

    np.testing.assert_array_equal(first["transition_count"], [1])
    np.testing.assert_array_equal(final["transition_count"], [0])
    np.testing.assert_array_equal(final["episode_complete"], [False])
    np.testing.assert_array_equal(final["episode_termination_reason"], [0])
    np.testing.assert_array_equal(final["episode_length"], [1])


def test_reward_diagnostics_to_arrays_keys_and_dtypes() -> None:
    diagnostics = RewardDiagnostics(
        schema_version=np.array([REWARD_DIAGNOSTICS_SCHEMA_VERSION], dtype=np.int16),
        reward_names=np.array(REWARD_NAMES, dtype=np.str_),
        rollout_end_step=np.zeros(1, dtype=np.int64),
        rollout_transition_count=np.zeros(1, dtype=np.int64),
        rollout_reward_sum=np.zeros((1, REWARD_SIGNAL_COUNT), dtype=np.float64),
        rollout_reward_abs_sum=np.zeros((1, REWARD_SIGNAL_COUNT), dtype=np.float64),
        rollout_reward_nonzero_count=np.zeros((1, REWARD_SIGNAL_COUNT), dtype=np.int64),
        rollout_reward_cross_product=np.zeros(
            (1, REWARD_SIGNAL_COUNT, REWARD_SIGNAL_COUNT), dtype=np.float64
        ),
        episode_end_step=np.zeros(1, dtype=np.int64),
        episode_worker_rank=np.zeros(1, dtype=np.int16),
        episode_index=np.zeros(1, dtype=np.int64),
        episode_length=np.zeros(1, dtype=np.int32),
        episode_termination_reason=np.zeros(1, dtype=np.int8),
        episode_complete=np.zeros(1, dtype=np.bool_),
        episode_reward_sums=np.zeros((1, REWARD_SIGNAL_COUNT), dtype=np.float64),
    )
    arrays = diagnostics.to_arrays()
    expected_keys = {
        "schema_version",
        "reward_names",
        "rollout_end_step",
        "rollout_transition_count",
        "rollout_reward_sum",
        "rollout_reward_abs_sum",
        "rollout_reward_nonzero_count",
        "rollout_reward_cross_product",
        "episode_end_step",
        "episode_worker_rank",
        "episode_index",
        "episode_length",
        "episode_termination_reason",
        "episode_complete",
        "episode_reward_sums",
    }
    assert set(arrays.keys()) == expected_keys
    assert arrays["schema_version"].dtype == np.int16
    assert arrays["rollout_end_step"].dtype == np.int64
    assert arrays["rollout_transition_count"].dtype == np.int64
    assert arrays["rollout_reward_sum"].dtype == np.float64
    assert arrays["rollout_reward_abs_sum"].dtype == np.float64
    assert arrays["rollout_reward_nonzero_count"].dtype == np.int64
    assert arrays["rollout_reward_cross_product"].dtype == np.float64
    assert arrays["episode_end_step"].dtype == np.int64
    assert arrays["episode_worker_rank"].dtype == np.int16
    assert arrays["episode_index"].dtype == np.int64
    assert arrays["episode_length"].dtype == np.int32
    assert arrays["episode_termination_reason"].dtype == np.int8
    assert arrays["episode_complete"].dtype == np.bool_
    assert arrays["episode_reward_sums"].dtype == np.float64


def test_safety_truncation_buffer_records_only_speed_bound_truncations() -> None:
    buffer = SafetyTruncationBuffer()

    buffer.record(
        position_m=100.0,
        termination_reason=TerminationReason.UNDER_LOWER_LIMIT,
    )
    buffer.record(
        position_m=200.0,
        termination_reason=TerminationReason.OVER_UPPER_LIMIT,
    )
    buffer.record(
        position_m=250.0,
        termination_reason=TerminationReason.OVER_SRTSP,
    )
    buffer.record(
        position_m=300.0,
        termination_reason=None,
    )
    buffer.record(
        position_m=400.0,
        termination_reason=TerminationReason.STOPPED_SHORT,
    )

    batch = buffer.drain()

    assert batch["position_m"].dtype == np.float32
    assert batch["termination_reason"].dtype == np.int8
    np.testing.assert_allclose(batch["position_m"], [100.0, 200.0, 250.0])
    np.testing.assert_array_equal(batch["termination_reason"], [4, 5, 6])
    assert buffer.drain()["position_m"].size == 0


def test_safety_truncation_buffer_rejects_non_finite_recorded_position() -> None:
    buffer = SafetyTruncationBuffer()

    with pytest.raises(ValueError, match="finite"):
        buffer.record(
            position_m=np.nan,
            termination_reason=TerminationReason.UNDER_LOWER_LIMIT,
        )


def test_safety_truncation_histogram_to_arrays_keys_and_dtypes() -> None:
    histogram = SafetyTruncationHistogram(
        bin_start_m=np.zeros(2, dtype=np.float64),
        bin_end_m=np.zeros(2, dtype=np.float64),
        safety_truncation_count=np.zeros(2, dtype=np.int64),
        low_safety_truncation_count=np.zeros(2, dtype=np.int64),
        high_safety_truncation_count=np.zeros(2, dtype=np.int64),
        global_safety_truncation_share=np.zeros(2, dtype=np.float64),
        position_bin_size_m=np.zeros(1, dtype=np.float64),
    )
    arrays = histogram.to_arrays()
    expected_keys = {
        "bin_start_m",
        "bin_end_m",
        "safety_truncation_count",
        "low_safety_truncation_count",
        "high_safety_truncation_count",
        "global_safety_truncation_share",
        "position_bin_size_m",
    }
    assert set(arrays.keys()) == expected_keys
    assert arrays["bin_start_m"].dtype == np.float64
    assert arrays["bin_end_m"].dtype == np.float64
    assert arrays["safety_truncation_count"].dtype == np.int64
    assert arrays["low_safety_truncation_count"].dtype == np.int64
    assert arrays["high_safety_truncation_count"].dtype == np.int64
    assert arrays["global_safety_truncation_share"].dtype == np.float64
    assert arrays["position_bin_size_m"].dtype == np.float64


def _diagnostics_with_interleaved_workers() -> RewardDiagnostics:
    reward_sums = np.zeros((3, len(REWARD_NAMES)), dtype=np.float64)
    reward_sums[:, -1] = [30.0, 20.0, 10.0]
    return RewardDiagnostics(
        schema_version=np.asarray([REWARD_DIAGNOSTICS_SCHEMA_VERSION], dtype=np.int16),
        reward_names=np.asarray(REWARD_NAMES),
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
        episode_termination_reason=np.asarray([1, 1, 1], dtype=np.int8),
        episode_complete=np.asarray([True, True, True]),
        episode_reward_sums=reward_sums,
    )


def test_complete_episode_sequence_merges_workers_without_collapsing_episodes() -> None:
    sequence = extract_complete_episode_sequence(
        _diagnostics_with_interleaved_workers()
    )

    np.testing.assert_array_equal(sequence.episode_number, [1, 2, 3])
    np.testing.assert_allclose(sequence.total_reward, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(sequence.length, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(sequence.termination_reason, [1, 1, 1])


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
