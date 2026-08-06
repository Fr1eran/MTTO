from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analyze_dp_redundancy_error import (  # noqa: E402
    compute_expected_redundant_operation_time,
    load_or_reconstruct_redundant_operation_time,
    load_redundant_operation_time_from_npz,
    plot_redundancy_error_series,
    reconstruct_redundant_operation_time,
    summarize_error_statistics,
)


def test_load_redundant_operation_time_from_npz_prefers_known_key(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "curve.npz"
    np.savez_compressed(
        npz_path,
        pos_m=np.asarray([0.0, 1.0]),
        redundant_operation_time_s=np.asarray([8.0, 2.0]),
    )

    series = load_redundant_operation_time_from_npz(
        npz_path,
        expected_size=2,
    )

    assert series is not None
    assert series.source == "npz:redundant_operation_time_s"
    assert series.key == "redundant_operation_time_s"
    np.testing.assert_allclose(series.values, np.asarray([8.0, 2.0]))


def test_load_redundant_operation_time_from_npz_respects_explicit_key(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "curve.npz"
    np.savez_compressed(
        npz_path,
        redundant_operation_time_s=np.asarray([99.0, 99.0]),
        custom_redundant=np.asarray([7.0, 3.0]),
    )

    series = load_redundant_operation_time_from_npz(
        npz_path,
        expected_size=2,
        redundant_key="custom_redundant",
    )

    assert series is not None
    assert series.source == "npz:custom_redundant"
    assert series.key == "custom_redundant"
    np.testing.assert_allclose(series.values, np.asarray([7.0, 3.0]))


def test_load_or_reconstruct_redundant_operation_time_falls_back_when_missing(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "curve.npz"
    np.savez_compressed(
        npz_path,
        pos_m=np.asarray([0.0, 10.0, 20.0]),
        speed_mps=np.asarray([0.0, 2.0, 0.0]),
        cum_time_s=np.asarray([0.0, 2.0, 5.0]),
    )

    def fake_min_remaining(
        begin_pos: float,
        begin_speed: float,
        end_pos: float,
        end_speed: float,
    ) -> float:
        assert end_pos == pytest.approx(20.0)
        assert end_speed == pytest.approx(0.0)
        return 0.1 * begin_pos + 0.5 * begin_speed

    series = load_or_reconstruct_redundant_operation_time(
        npz_path=npz_path,
        pos_arr=np.asarray([0.0, 10.0, 20.0]),
        speed_arr=np.asarray([0.0, 2.0, 0.0]),
        cum_time_arr=np.asarray([0.0, 2.0, 5.0]),
        schedule_time_s=10.0,
        target_position=20.0,
        target_speed=0.0,
        redundant_key=None,
        min_remaining_time_fn=fake_min_remaining,
    )

    assert series.source == "reconstructed_from_cum_time"
    assert series.key is None
    np.testing.assert_allclose(series.values, np.asarray([10.0, 6.0, 3.0]))


def test_reconstruct_redundant_operation_time_validates_lengths() -> None:
    def _zero_remaining_time(*_args: float) -> float:
        return 0.0

    with pytest.raises(ValueError, match="same length"):
        _ = reconstruct_redundant_operation_time(
            pos_arr=np.asarray([0.0, 1.0]),
            speed_arr=np.asarray([0.0]),
            cum_time_arr=np.asarray([0.0, 1.0]),
            schedule_time_s=10.0,
            target_position=1.0,
            target_speed=0.0,
            min_remaining_time_fn=_zero_remaining_time,
        )


def test_compute_expected_redundant_operation_time_is_linear_over_position() -> None:
    expected = compute_expected_redundant_operation_time(
        pos_arr=np.asarray([0.0, 50.0, 100.0, 120.0]),
        start_position=0.0,
        target_position=100.0,
        initial_redundant_s=20.0,
    )

    np.testing.assert_allclose(expected, np.asarray([20.0, 10.0, 0.0, 0.0]))


def test_summarize_error_statistics_reports_positive_negative_and_max_abs() -> None:
    summary = summarize_error_statistics(
        pos_arr=np.asarray([0.0, 10.0, 20.0, 30.0]),
        cum_time_arr=np.asarray([0.0, 1.0, 2.0, 3.0]),
        error_arr=np.asarray([-2.0, 0.0, 0.5, 4.0]),
        zero_eps=0.1,
    )

    assert summary["overall"]["sample_count"] == 4
    assert summary["overall"]["max_abs_s"] == pytest.approx(4.0)
    assert summary["overall"]["max_abs_position_m"] == pytest.approx(30.0)
    assert summary["positive"]["sample_count"] == 2
    assert summary["positive"]["max_s"] == pytest.approx(4.0)
    assert summary["positive"]["max_position_m"] == pytest.approx(30.0)
    assert summary["negative"]["sample_count"] == 1
    assert summary["negative"]["min_s"] == pytest.approx(-2.0)
    assert summary["negative"]["max_abs_s"] == pytest.approx(2.0)
    assert summary["near_zero"]["sample_count"] == 1


def test_plot_redundancy_error_series_places_actual_and_expected_on_same_axes() -> None:
    fig = plot_redundancy_error_series(
        pos_arr=np.asarray([0.0, 1.0, 2.0]),
        actual_redundant_arr=np.asarray([10.0, 5.0, 1.0]),
        expected_redundant_arr=np.asarray([10.0, 4.0, 0.0]),
        error_arr=np.asarray([0.0, 1.0, 1.0]),
    )

    assert len(fig.axes) == 2
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "Actual redundant time" in labels
    assert "Expected redundant time" in labels

    plt.close(fig)
