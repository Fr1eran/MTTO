import os
from pathlib import Path

from scripts.show_rl_result import (
    _build_cli_parser,
    resolve_rl_curve_artifact,
)


def _write_artifact(run_dir: Path, file_name: str) -> tuple[Path, Path]:
    curve_path = run_dir / file_name
    metrics_path = run_dir / f"{curve_path.stem}_metrics.json"
    curve_path.write_bytes(b"curve")
    metrics_path.write_text("{}", encoding="utf-8")
    return curve_path, metrics_path


def test_show_rl_result_cli_defaults() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([])

    assert args.curve_dir == "output/optimal/rl"
    assert args.trajectory_source == "best"


def test_show_rl_result_cli_accepts_trajectory_source() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([
        "--trajectory-source",
        "final",
        "--curve-dir",
        "output/custom/rl",
    ])

    assert args.trajectory_source == "final"
    assert args.curve_dir == "output/custom/rl"


def test_show_rl_result_cli_accepts_dry_run() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(["--dry-run", "--curve-dir", "output/custom/rl"])

    assert args.dry_run is True
    assert args.curve_dir == "output/custom/rl"


def test_resolve_rl_curve_artifact_prefers_latest_best_across_trigger_modes(
    tmp_path: Path,
) -> None:
    curve_root = tmp_path / "runs"
    best_steps_dir = curve_root / "430p0_100p0__basic_safety_stopping" / "best_steps"
    best_episodes_dir = curve_root / "430p0_100p0__basic" / "best_episodes"
    best_steps_dir.mkdir(parents=True)
    best_episodes_dir.mkdir(parents=True)

    old_curve, _ = _write_artifact(best_steps_dir, "best_trajectory.npz")
    new_curve, new_metrics = _write_artifact(best_episodes_dir, "best_trajectory.npz")

    os.utime(old_curve, (1, 1))
    os.utime(new_curve, (2, 2))

    artifact = resolve_rl_curve_artifact(
        curve_dir=str(curve_root), trajectory_source="best"
    )

    assert artifact.npz_path == str(new_curve)
    assert artifact.metrics_path == str(new_metrics)
    assert artifact.npz_path.endswith(".npz")
