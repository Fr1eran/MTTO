import os
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_rl_dp import (
    _build_cli_parser,
    _compute_segment_midpoints,
    _resolve_curve_artifacts,
    _resolve_target_schedule_time,
)
from rl.experiment_utils import DEFAULT_SCHEDULE_TIME_S


def _write_dp_artifact(run_dir: Path) -> tuple[Path, Path]:
    curve_path = run_dir / "optimized_speed_curve.npz"
    metrics_path = run_dir / "optimized_speed_curve_metrics.json"
    curve_path.write_bytes(b"dp")
    metrics_path.write_text("{}", encoding="utf-8")
    return curve_path, metrics_path


def _write_rl_artifact(run_dir: Path, *, file_name: str) -> tuple[Path, Path]:
    curve_path = run_dir / file_name
    metrics_path = run_dir / f"{curve_path.stem}_metrics.json"
    curve_path.write_bytes(b"rl")
    metrics_path.write_text("{}", encoding="utf-8")
    return curve_path, metrics_path


def test_compare_rl_dp_cli_defaults() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([])

    assert args.dp_curve_dir == "output/optimal/dp"
    assert args.rl_curve_dir == "output/optimal/rl"
    assert args.trajectory_source == "best"
    assert args.no_safeguard is False
    assert args.factor == pytest.approx(0.99)


def test_compare_rl_dp_cli_accepts_explicit_args() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([
        "--dp-curve-dir",
        "output/custom/dp",
        "--rl-curve-dir",
        "output/custom/rl",
        "--trajectory-source",
        "final",
        "--no-safeguard",
        "--factor",
        "0.95",
    ])

    assert args.dp_curve_dir == "output/custom/dp"
    assert args.rl_curve_dir == "output/custom/rl"
    assert args.trajectory_source == "final"
    assert args.no_safeguard is True
    assert args.factor == pytest.approx(0.95)


def test_resolve_curve_artifacts_loads_latest_dp_and_latest_rl_best(
    tmp_path: Path,
) -> None:
    dp_root = tmp_path / "dp_runs"
    rl_root = tmp_path / "rl_runs"

    old_dp_dir = dp_root / "old"
    new_dp_dir = dp_root / "new"
    old_dp_dir.mkdir(parents=True)
    new_dp_dir.mkdir(parents=True)
    old_dp_curve, _ = _write_dp_artifact(old_dp_dir)
    new_dp_curve, new_dp_metrics = _write_dp_artifact(new_dp_dir)

    old_rl_dir = rl_root / "430p0_100p0__basic_safety_stopping" / "best_steps"
    new_rl_dir = rl_root / "430p0_100p0__basic" / "best_episodes"
    old_rl_dir.mkdir(parents=True)
    new_rl_dir.mkdir(parents=True)
    old_rl_curve, _ = _write_rl_artifact(old_rl_dir, file_name="best_trajectory.npz")
    new_rl_curve, new_rl_metrics = _write_rl_artifact(
        new_rl_dir,
        file_name="best_trajectory.npz",
    )

    os.utime(old_dp_curve, (1, 1))
    os.utime(new_dp_curve, (2, 2))
    os.utime(old_rl_curve, (1, 1))
    os.utime(new_rl_curve, (2, 2))

    dp_artifact, rl_artifact = _resolve_curve_artifacts(
        dp_curve_dir=str(dp_root),
        rl_curve_dir=str(rl_root),
        trajectory_source="best",
    )

    assert dp_artifact.npz_path == str(new_dp_curve)
    assert dp_artifact.metrics_path == str(new_dp_metrics)
    assert rl_artifact.npz_path == str(new_rl_curve)
    assert rl_artifact.metrics_path == str(new_rl_metrics)


def test_resolve_curve_artifacts_raises_when_dp_missing(tmp_path: Path) -> None:
    dp_root = tmp_path / "dp_runs"
    rl_root = tmp_path / "rl_runs"
    dp_root.mkdir(parents=True)
    rl_dir = rl_root / "430p0_100p0__basic" / "best_steps"
    rl_dir.mkdir(parents=True)
    _write_rl_artifact(rl_dir, file_name="best_trajectory.npz")

    with pytest.raises(FileNotFoundError, match="optimized_speed_curve.npz"):
        _resolve_curve_artifacts(
            dp_curve_dir=str(dp_root),
            rl_curve_dir=str(rl_root),
            trajectory_source="best",
        )


def test_resolve_target_schedule_time_prefers_rl_then_dp_then_default() -> None:
    assert _resolve_target_schedule_time(
        dp_metrics={"target_time_s": 420.0},
        rl_metrics={"target_time_s": 430.0},
    ) == pytest.approx(430.0)

    assert _resolve_target_schedule_time(
        dp_metrics={"target_time_s": 420.0},
        rl_metrics={},
    ) == pytest.approx(420.0)

    assert _resolve_target_schedule_time(
        dp_metrics={},
        rl_metrics={},
    ) == pytest.approx(DEFAULT_SCHEDULE_TIME_S)


def test_compute_segment_midpoints_returns_expected_values() -> None:
    midpoints = _compute_segment_midpoints(np.asarray([0.0, 10.0, 20.0, 35.0]))
    np.testing.assert_allclose(midpoints, np.asarray([5.0, 15.0, 27.5]))


def test_compute_segment_midpoints_returns_empty_when_insufficient_points() -> None:
    midpoints = _compute_segment_midpoints(np.asarray([12.0]))
    assert midpoints.size == 0


def test_compute_segment_midpoints_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        _compute_segment_midpoints(np.asarray([[0.0, 1.0], [2.0, 3.0]]))
