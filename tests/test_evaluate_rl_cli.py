import numpy as np
import pytest

from scripts.evaluate_rl import (
    build_arg_parser,
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

    np.testing.assert_allclose(rewards, np.asarray([5.0], dtype=np.float32))


def test_compute_punctuality_dense_reward_series_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        compute_punctuality_dense_reward_series(
            position_seq=[1.0],
            redundant_operation_time_seq=[],
            gamma=0.9,
            potential_fn=lambda pos, redundant_time: pos + redundant_time,
        )
