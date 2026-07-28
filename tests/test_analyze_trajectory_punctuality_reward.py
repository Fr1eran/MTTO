from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analyze_trajectory_punctuality_reward import (  # noqa: E402
    _build_cli_parser,
    compute_punctuality_error_v34,
    compute_dense_reward_from_potential,
    load_or_reconstruct_redundant_operation_time,
    load_or_recover_operation_time,
    plot_punctuality_reward_analysis,
    punctuality_potential_v34,
    reconstruct_redundant_operation_time,
)


def test_v34_error_matches_training_environment_progress_reference() -> None:
    error = compute_punctuality_error_v34(
        pos_arr=np.asarray([0.0, 50.0, 120.0]),
        redundant_operation_time_arr=np.asarray([10.0, 4.0, -1.0]),
        start_position_m=0.0,
        target_position_m=100.0,
        max_redundant_time_s=20.0,
    )

    np.testing.assert_allclose(error, np.asarray([-10.0, -6.0, -1.0]))


def test_v34_potential_matches_training_environment_formula() -> None:
    phi = punctuality_potential_v34(
        pos_arr=np.asarray([0.0, 50.0, 120.0]),
        redundant_operation_time_arr=np.asarray([10.0, 4.0, -1.0]),
        start_position_m=0.0,
        target_position_m=100.0,
        max_redundant_time_s=20.0,
    )

    np.testing.assert_allclose(phi, np.asarray([-1.0, -0.6, -0.1]))


def test_compute_dense_reward_from_potential_uses_gamma_difference() -> None:
    reward = compute_dense_reward_from_potential(
        np.asarray([1.0, 3.0, 2.0]),
        gamma=0.5,
    )

    np.testing.assert_allclose(reward, np.asarray([0.5, -2.0]))


def test_load_or_recover_operation_time_recovers_when_npz_has_no_time_axis(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "trajectory.npz"
    np.savez_compressed(
        npz_path,
        pos_m=np.asarray([0.0, 10.0, 20.0]),
        speed_mps=np.asarray([10.0, 10.0, 10.0]),
    )

    series = load_or_recover_operation_time(
        npz_path=npz_path,
        pos_arr=np.asarray([0.0, 10.0, 20.0]),
        speed_arr=np.asarray([10.0, 10.0, 10.0]),
    )

    assert series.source == "recovered_from_position_speed"
    assert series.key is None
    np.testing.assert_allclose(series.values, np.asarray([0.0, 1.0, 2.0]))


def test_load_or_recover_operation_time_prefers_npz_cum_time(tmp_path: Path) -> None:
    npz_path = tmp_path / "trajectory.npz"
    np.savez_compressed(
        npz_path,
        pos_m=np.asarray([0.0, 10.0]),
        speed_mps=np.asarray([10.0, 10.0]),
        cum_time_s=np.asarray([0.0, 4.0]),
    )

    series = load_or_recover_operation_time(
        npz_path=npz_path,
        pos_arr=np.asarray([0.0, 10.0]),
        speed_arr=np.asarray([10.0, 10.0]),
    )

    assert series.source == "npz:cum_time_s"
    assert series.key == "cum_time_s"
    np.testing.assert_allclose(series.values, np.asarray([0.0, 4.0]))


def test_reconstruct_redundant_operation_time_uses_schedule_elapsed_and_remaining() -> None:
    def fake_min_remaining(
        begin_pos: float,
        begin_speed: float,
        end_pos: float,
        end_speed: float,
    ) -> float:
        assert end_pos == pytest.approx(20.0)
        assert end_speed == pytest.approx(0.0)
        return 0.2 * begin_pos + 0.5 * begin_speed

    redundant = reconstruct_redundant_operation_time(
        pos_arr=np.asarray([0.0, 10.0, 20.0]),
        speed_arr=np.asarray([0.0, 2.0, 0.0]),
        operation_time_arr=np.asarray([0.0, 2.0, 5.0]),
        schedule_time_s=10.0,
        target_position_m=20.0,
        target_speed_mps=0.0,
        min_remaining_time_fn=fake_min_remaining,
    )

    np.testing.assert_allclose(redundant, np.asarray([10.0, 5.0, 1.0]))


def test_load_or_reconstruct_redundant_operation_time_falls_back_when_missing(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "trajectory.npz"
    np.savez_compressed(
        npz_path,
        pos_m=np.asarray([0.0, 10.0]),
        speed_mps=np.asarray([0.0, 0.0]),
    )

    series = load_or_reconstruct_redundant_operation_time(
        npz_path=npz_path,
        pos_arr=np.asarray([0.0, 10.0]),
        speed_arr=np.asarray([0.0, 0.0]),
        operation_time_arr=np.asarray([0.0, 6.0]),
        schedule_time_s=10.0,
        target_position_m=10.0,
        target_speed_mps=0.0,
        redundant_key=None,
        min_remaining_time_fn=lambda *_args: 1.0,
    )

    assert series.source == "reconstructed_from_operation_time"
    assert series.key is None
    np.testing.assert_allclose(series.values, np.asarray([9.0, 3.0]))


def test_plot_punctuality_reward_analysis_returns_five_shared_x_axes() -> None:
    fig = plot_punctuality_reward_analysis(
        pos_arr=np.asarray([0.0, 10.0, 20.0]),
        operation_time_arr=np.asarray([0.0, 2.0, 5.0]),
        redundant_operation_time_arr=np.asarray([10.0, 6.0, 1.0]),
        punctuality_error_arr=np.asarray([-10.0, -6.0, -1.0]),
        phi_arr=np.asarray([1.0, 0.5, 0.0]),
        reward_arr=np.asarray([-0.5, -0.5]),
    )

    assert len(fig.axes) == 5
    shared = fig.axes[0].get_shared_x_axes()
    assert shared.joined(fig.axes[0], fig.axes[1])
    assert shared.joined(fig.axes[0], fig.axes[2])
    assert shared.joined(fig.axes[0], fig.axes[3])
    assert shared.joined(fig.axes[0], fig.axes[4])

    plt.close(fig)


def test_cli_defaults_and_accepts_trajectory_kind_and_gamma() -> None:
    parser = _build_cli_parser()
    default_args = parser.parse_args([])

    assert default_args.trajectory_kind == "dp"
    assert default_args.gamma == pytest.approx(0.998)
    assert default_args.potential_version == "v34"

    rl_args = parser.parse_args(
        [
            "--trajectory-kind",
            "rl",
            "--trajectory-source",
            "final",
            "--gamma",
            "1.0",
        ]
    )

    assert rl_args.trajectory_kind == "rl"
    assert rl_args.trajectory_source == "final"
    assert rl_args.gamma == pytest.approx(1.0)
