from pathlib import Path

from rl.experiment_utils import resolve_rl_curve_artifact
from scripts.show_rl_result import (
    _build_cli_parser,
)


def _write_artifact(run_dir: Path) -> tuple[Path, Path]:
    curve_path = run_dir / "trajectory.npz"
    metrics_path = run_dir / "metrics.json"
    _ = curve_path.write_bytes(b"curve")
    _ = metrics_path.write_text("{}", encoding="utf-8")
    return curve_path, metrics_path


def test_show_rl_result_cli_requires_explicit_model_dir() -> None:
    parser = _build_cli_parser()
    try:
        parser.parse_args([])
    except SystemExit:
        pass
    else:
        raise AssertionError("--model-dir must be required")


def test_show_rl_result_cli_accepts_model_dir() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "output/custom/rl",
        ]
    )

    assert args.model_dir == "output/custom/rl"


def test_show_rl_result_cli_accepts_dry_run() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(["--dry-run", "--model-dir", "output/custom/rl"])

    assert args.dry_run is True
    assert args.model_dir == "output/custom/rl"


def test_resolve_rl_curve_artifact_uses_only_explicit_model_directory(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    curve, metrics = _write_artifact(model_dir)
    artifact = resolve_rl_curve_artifact(curve_dir=str(model_dir))

    assert artifact.npz_path == str(curve)
    assert artifact.metrics_path == str(metrics)
    assert artifact.npz_path.endswith(".npz")
