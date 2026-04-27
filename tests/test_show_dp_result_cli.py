import os
from pathlib import Path

import pytest

from scripts.show_dp_result import _build_cli_parser, _resolve_curve_and_metrics_paths


def test_show_dp_result_cli_defaults() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([])

    assert args.curve_dir == "output/optimal/dp"


def test_show_dp_result_cli_accepts_explicit_curve_dir() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([
        "--curve-dir",
        "output/custom/dp_curves",
    ])

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

    old_curve.write_bytes(b"old")
    new_curve.write_bytes(b"new")
    old_metrics.write_text("{}", encoding="utf-8")
    new_metrics.write_text("{}", encoding="utf-8")

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
        _resolve_curve_and_metrics_paths(
            curve_dir=str(curve_root),
        )


def test_resolve_curve_and_metrics_paths_raises_when_same_dir_metrics_missing(
    tmp_path: Path,
) -> None:
    curve_root = tmp_path / "curves"
    run_dir = curve_root / "440p0_0p1"
    run_dir.mkdir(parents=True)

    (run_dir / "optimized_speed_curve.npz").write_bytes(b"curve")

    with pytest.raises(FileNotFoundError, match="optimized_speed_curve_metrics.json"):
        _resolve_curve_and_metrics_paths(
            curve_dir=str(curve_root),
        )
