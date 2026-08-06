import os
from pathlib import Path

import numpy as np
import pytest

from scripts.show_dp_result import (
    _build_cli_parser,
    _calc_redundant_operation_time_arr,
    _resolve_curve_and_metrics_paths,
)


def test_show_dp_result_cli_defaults() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([])

    assert args.curve_dir == "output/optimal/dp"


def test_show_dp_result_cli_accepts_explicit_curve_dir() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(
        [
            "--curve-dir",
            "output/custom/dp_curves",
        ]
    )

    assert args.curve_dir == "output/custom/dp_curves"


def test_resolve_curve_and_metrics_paths_uses_latest_curve_and_same_dir_metrics(
    tmp_path: Path,
) -> None:
    curve_root = tmp_path / "runs"

    run_old_dir = curve_root / "440p0_0p1"
    run_new_dir = curve_root / "445p0_0p1"

    run_old_dir.mkdir(parents=True)
    run_new_dir.mkdir(parents=True)

    old_curve = run_old_dir / "optimized_speed_curve.npz"
    new_curve = run_new_dir / "optimized_speed_curve.npz"
    old_metrics = run_old_dir / "optimized_speed_curve_metrics.json"
    new_metrics = run_new_dir / "optimized_speed_curve_metrics.json"

    _ = old_curve.write_bytes(b"old")
    _ = new_curve.write_bytes(b"new")
    _ = old_metrics.write_text("{}", encoding="utf-8")
    _ = new_metrics.write_text("{}", encoding="utf-8")

    os.utime(old_curve, (1, 1))
    os.utime(new_curve, (2, 2))
    os.utime(old_metrics, (999, 999))
    os.utime(new_metrics, (1, 1))

    curve_path, metrics_path = _resolve_curve_and_metrics_paths(
        curve_dir=str(curve_root),
    )

    assert curve_path == str(new_curve)
    assert metrics_path == str(new_metrics)


def test_resolve_curve_and_metrics_paths_raises_when_curve_missing(
    tmp_path: Path,
) -> None:
    curve_root = tmp_path / "curves"
    curve_root.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="optimized_speed_curve.npz"):
        _ = _resolve_curve_and_metrics_paths(
            curve_dir=str(curve_root),
        )


def test_resolve_curve_and_metrics_paths_raises_when_same_dir_metrics_missing(
    tmp_path: Path,
) -> None:
    curve_root = tmp_path / "curves"
    run_dir = curve_root / "440p0_0p1"
    run_dir.mkdir(parents=True)

    _ = (run_dir / "optimized_speed_curve.npz").write_bytes(b"curve")

    with pytest.raises(FileNotFoundError, match="optimized_speed_curve_metrics.json"):
        _ = _resolve_curve_and_metrics_paths(
            curve_dir=str(curve_root),
        )


def test_calc_redundant_operation_time_arr_uses_schedule_and_remaining() -> None:
    def fake_min_remaining(
        begin_pos: float,
        begin_speed: float,
        end_pos: float,
        end_speed: float,
    ) -> float:
        assert end_pos == pytest.approx(20.0)
        assert end_speed == pytest.approx(0.0)
        return 0.1 * begin_pos + 0.5 * begin_speed

    redundant = _calc_redundant_operation_time_arr(
        pos_arr=np.asarray([0.0, 10.0, 20.0]),
        speed_arr=np.asarray([0.0, 2.0, 0.0]),
        cum_time_arr=np.asarray([0.0, 2.0, 5.0]),
        schedule_time_s=10.0,
        target_position=20.0,
        target_speed=0.0,
        min_remaining_time_fn=fake_min_remaining,
    )

    np.testing.assert_allclose(redundant, np.asarray([10.0, 6.0, 3.0]))
