import numpy as np
import pytest

from scripts.evaluate_rl import (
    build_initial_rollout_series,
    build_arg_parser,
    compute_punctuality_error_series,
    compute_punctuality_dense_reward_series,
)


def test_evaluate_rl_cli_accepts_dry_run_and_shared_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "--dry-run",
        "--schedule-time-s",
        "430.0",
        "--reward-profile",
        "basic_safety",
        "--reward-discount",
        "0.95",
        "--device",
        "cuda",
        "--plot-punctuality-dense-reward",
    ])

    assert args.dry_run is True
    assert args.schedule_time_s == 430.0
    assert args.reward_profile == "basic_safety"
    assert args.reward_discount == 0.95
    assert args.device == "cuda"
    assert args.plot_punctuality_dense_reward is True


def test_compute_punctuality_dense_reward_series_uses_adjacent_real_samples() -> None:
    rewards = compute_punctuality_dense_reward_series(
        position_seq=[100.0, 250.0],
        redundant_operation_time_seq=[3.0, -2.0],
        gamma=0.9,
        potential_fn=lambda pos, redundant_time: pos / 10.0
        + 2.0 * redundant_time,
    )

    np.testing.assert_allclose(rewards, np.asarray([2.9], dtype=np.float32))


def test_compute_punctuality_dense_reward_series_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        compute_punctuality_dense_reward_series(
            position_seq=[1.0],
            redundant_operation_time_seq=[],
            gamma=0.9,
            potential_fn=lambda pos, redundant_time: pos + redundant_time,
        )


def test_compute_punctuality_error_series_matches_v34_expected_redundancy() -> None:
    errors = compute_punctuality_error_series(
        position_seq=[0.0, 50.0, 120.0],
        redundant_operation_time_seq=[10.0, 4.0, -1.0],
        target_position_m=100.0,
        whole_distance_m=100.0,
        max_redundant_operation_time_s=20.0,
    )

    np.testing.assert_allclose(errors, np.asarray([-10.0, -6.0, -1.0]))


def test_compute_punctuality_error_series_rejects_invalid_distance() -> None:
    with pytest.raises(ValueError, match="whole_distance_m must be positive"):
        compute_punctuality_error_series(
            position_seq=[0.0],
            redundant_operation_time_seq=[0.0],
            target_position_m=0.0,
            whole_distance_m=0.0,
            max_redundant_operation_time_s=20.0,
        )


def test_build_initial_rollout_series_reads_reset_environment_state() -> None:
    class FakeVecEnv:
        values = {
            "current_position": [123.0],
            "current_speed": [4.5],
            "current_operation_time": [0.0],
            "current_redundant_operation_time": [26.0],
        }

        def get_attr(self, attr_name: str):
            return self.values[attr_name]

    (
        position_seq,
        speed_seq,
        operation_time_seq,
        redundant_operation_time_seq,
    ) = build_initial_rollout_series(FakeVecEnv())

    assert position_seq == [123.0]
    assert speed_seq == [4.5]
    assert operation_time_seq == [0.0]
    assert redundant_operation_time_seq == [26.0]
