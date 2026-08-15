from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.compare_speed_profiles import (
    DEFAULT_REAL_CURVE_PATH,
    ProfileMetrics,
    SpeedProfile,
    _build_cli_parser,
    _resolve_target_schedule_time,
    _validate_common_target_position,
    format_comparison_table,
    load_real_operation_profile,
)


def test_compare_speed_profiles_cli_defaults() -> None:
    args = _build_cli_parser().parse_args([])

    assert args.real_curve == DEFAULT_REAL_CURVE_PATH
    assert args.trajectory_source == "best"
    assert args.no_safeguard is False


def test_load_real_operation_profile_reads_required_aligned_arrays(
    tmp_path: Path,
) -> None:
    curve_path = tmp_path / "real_curve.npz"
    np.savez_compressed(
        curve_path,
        position_m=np.asarray([100.0, 110.0]),
        speed_mps=np.asarray([5.0, 0.0]),
        time_s=np.asarray([0.0, 4.0]),
        target_position_m=np.asarray(110.0),
    )

    profile = load_real_operation_profile(curve_path)

    assert profile.label == "Actual operation"
    assert profile.target_position_m == pytest.approx(110.0)
    np.testing.assert_allclose(profile.position_m, [100.0, 110.0])


def test_load_real_operation_profile_rejects_missing_required_arrays(
    tmp_path: Path,
) -> None:
    curve_path = tmp_path / "real_curve.npz"
    np.savez_compressed(
        curve_path,
        position_m=np.asarray([100.0, 110.0]),
        speed_mps=np.asarray([5.0, 0.0]),
    )

    with pytest.raises(ValueError, match="missing required arrays"):
        _ = load_real_operation_profile(curve_path)


def test_target_schedule_time_requires_matching_dp_and_rl_tasks() -> None:
    with pytest.raises(ValueError, match="target_time_s differ"):
        _ = _resolve_target_schedule_time(
            dp_metrics={"target_time_s": 430.0},
            rl_metrics={"target_time_s": 431.0},
        )


def test_common_target_position_rejects_unaligned_profiles() -> None:
    profiles = [
        SpeedProfile(
            "DP",
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            1.0,
        ),
        SpeedProfile(
            "Actual",
            np.asarray([0.0, 2.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            2.0,
        ),
    ]

    with pytest.raises(ValueError, match="target positions differ"):
        _ = _validate_common_target_position(profiles)


def test_format_comparison_table_contains_only_requested_metrics() -> None:
    table = format_comparison_table(
        [
            (
                "DP optimization",
                ProfileMetrics(1.25, 0.0, 123.456, 0.123456),
            ),
            (
                "Proposed RL",
                ProfileMetrics(2.5, 0.25, 120.0, 0.1),
            ),
            (
                "Actual operation",
                ProfileMetrics(3.0, 0.5, 130.0, 0.2),
            ),
        ]
    )

    assert "Time error (s)" in table
    assert "Stop error (m)" in table
    assert "Total energy (kJ)" in table
    assert "comfort_tav (m/s^2)" in table
    assert "123.456" in table
    assert "0.123456" in table
