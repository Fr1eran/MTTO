import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from model.ocs import SafeGuardUtility
from rl.experiment_utils import DEFAULT_SCHEDULE_TIME_S
from scripts.analyze_sps_compliance import (
    _build_cli_parser,
    _parse_output_mode,
    _resolve_curve_artifacts,
    _resolve_single_curve_artifact,
    _resolve_target_schedule_time,
    _validate_cli_args,
    replay_sps_compliance,
)


class _StubSafeGuard:
    def __init__(
        self,
        *,
        trigger_min_speed: float = 1.0,
        max_speed_pre_step: float = 5.0,
        max_speed_post_step: float = 5.0,
    ) -> None:
        self.trigger_min_speed: float = trigger_min_speed
        self.max_speed_pre_step: float = max_speed_pre_step
        self.max_speed_post_step: float = max_speed_post_step

    def get_min_and_max_speed(
        self,
        current_pos: float,
        current_sp: int,
    ) -> tuple[float, float]:
        del current_pos
        if current_sp < 0:
            return 0.0, self.max_speed_pre_step
        return self.trigger_min_speed, self.max_speed_post_step

    def get_min_speed(
        self,
        current_pos: float,
        current_sp: int,
    ) -> float:
        return self.get_min_and_max_speed(current_pos, current_sp)[0]

    def get_max_speed(
        self,
        current_pos: float,
        current_sp: int,
    ) -> float:
        return self.get_min_and_max_speed(current_pos, current_sp)[1]


def _write_dp_artifact(run_dir: Path) -> tuple[Path, Path]:
    curve_path = run_dir / "optimized_speed_curve.npz"
    metrics_path = run_dir / "optimized_speed_curve_metrics.json"
    np.savez_compressed(
        curve_path,
        pos_m=np.asarray([0.0, 1.0], dtype=np.float32),
        speed_mps=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    _ = metrics_path.write_text("{}", encoding="utf-8")
    return curve_path, metrics_path


def _write_rl_artifact(run_dir: Path, *, file_name: str) -> tuple[Path, Path]:
    curve_path = run_dir / file_name
    metrics_path = run_dir / f"{curve_path.stem}_metrics.json"
    np.savez_compressed(
        curve_path,
        pos_m=np.asarray([0.0, 1.0], dtype=np.float32),
        speed_mps=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    _ = metrics_path.write_text("{}", encoding="utf-8")
    return curve_path, metrics_path


def test_analyze_sps_compliance_cli_defaults() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([])

    assert args.dp_curve_dir == "output/optimal/dp"
    assert args.rl_curve_dir == "output/optimal/rl"
    assert args.trajectory_source == "best"
    assert args.analysis_mode == "compare"
    assert args.trajectory_kind is None
    assert args.output_mode == "text+plot"
    assert args.event_annotation == "auto"
    assert args.step_delay_s == pytest.approx(2.0)


def test_analyze_sps_compliance_cli_accepts_marker_only_and_json() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(
        [
            "--output-mode",
            "json",
            "--event-annotation",
            "marker-only",
            "--step-delay-s",
            "1.5",
        ]
    )

    assert args.output_mode == "json"
    assert args.event_annotation == "marker-only"
    assert args.step_delay_s == pytest.approx(1.5)


def test_analyze_sps_compliance_cli_accepts_single_mode_and_kind() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(
        [
            "--analysis-mode",
            "single",
            "--trajectory-kind",
            "rl",
        ]
    )
    _validate_cli_args(parser, args)
    assert args.analysis_mode == "single"
    assert args.trajectory_kind == "rl"


def test_validate_cli_args_requires_trajectory_kind_for_single_mode() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(["--analysis-mode", "single"])
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_validate_cli_args_rejects_trajectory_kind_for_compare_mode() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(["--trajectory-kind", "dp"])
    with pytest.raises(SystemExit):
        _validate_cli_args(parser, args)


def test_parse_output_mode_supports_text_plot_and_csv_like_list() -> None:
    assert _parse_output_mode("text+plot") == {"text", "plot"}
    assert _parse_output_mode("text,json") == {"text", "json"}


def test_parse_output_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown output mode"):
        _ = _parse_output_mode("text,invalid")


def test_resolve_target_schedule_precedence() -> None:
    assert _resolve_target_schedule_time(
        dp_metrics={"target_time_s": 420.0},
        rl_metrics={"target_time_s": 430.0},
        schedule_time_s_override=440.0,
    ) == pytest.approx(440.0)

    assert _resolve_target_schedule_time(
        dp_metrics={"target_time_s": 420.0},
        rl_metrics={"target_time_s": 430.0},
        schedule_time_s_override=None,
    ) == pytest.approx(430.0)

    assert _resolve_target_schedule_time(
        dp_metrics={"target_time_s": 420.0},
        rl_metrics={},
        schedule_time_s_override=None,
    ) == pytest.approx(420.0)

    assert _resolve_target_schedule_time(
        dp_metrics={},
        rl_metrics={},
        schedule_time_s_override=None,
    ) == pytest.approx(DEFAULT_SCHEDULE_TIME_S)


def test_resolve_target_schedule_time_single_metrics_takes_effect() -> None:
    assert _resolve_target_schedule_time(
        single_metrics={"target_time_s": 450.0},
        schedule_time_s_override=None,
    ) == pytest.approx(450.0)


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

    old_rl_dir = (
        rl_root / "430p0_100p0__basic_safety" / "best_rollouts"
    )
    new_rl_dir = rl_root / "430p0_100p0__basic" / "best_rollouts"
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


def test_resolve_single_curve_artifact_by_kind(tmp_path: Path) -> None:
    dp_root = tmp_path / "dp_runs"
    rl_root = tmp_path / "rl_runs"
    dp_dir = dp_root / "run"
    rl_dir = rl_root / "430p0_100p0__basic_safety" / "best_rollouts"
    dp_dir.mkdir(parents=True)
    rl_dir.mkdir(parents=True)

    dp_curve, dp_metrics = _write_dp_artifact(dp_dir)
    rl_curve, rl_metrics = _write_rl_artifact(rl_dir, file_name="best_trajectory.npz")

    dp_artifact = _resolve_single_curve_artifact(
        trajectory_kind="dp",
        dp_curve_dir=str(dp_root),
        rl_curve_dir=str(rl_root),
        trajectory_source="best_rollouts",
    )
    rl_artifact = _resolve_single_curve_artifact(
        trajectory_kind="rl",
        dp_curve_dir=str(dp_root),
        rl_curve_dir=str(rl_root),
        trajectory_source="best_rollouts",
    )

    assert dp_artifact.npz_path == str(dp_curve)
    assert dp_artifact.metrics_path == str(dp_metrics)
    assert rl_artifact.npz_path == str(rl_curve)
    assert rl_artifact.metrics_path == str(rl_metrics)


@pytest.mark.parametrize(
    "name,speed_arr,time_arr,guard,expected_trigger,expected_delay_pass,expected_unfinished",
    [
        (
            "case_a_triggered_no_delay_violation",
            np.asarray([0.0, 2.0, 2.0], dtype=np.float64),
            np.asarray([0.0, 0.1, 1.2], dtype=np.float64),
            _StubSafeGuard(),
            True,
            True,
            0,
        ),
        (
            "case_b_not_triggered",
            np.asarray([0.0, 0.5, 0.5], dtype=np.float64),
            np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
            _StubSafeGuard(trigger_min_speed=1.0),
            False,
            True,
            0,
        ),
        (
            "case_c_timeout_no_violation",
            np.asarray([0.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64),
            np.asarray([0.0, 0.1, 0.2, 0.3, 1.4], dtype=np.float64),
            _StubSafeGuard(max_speed_pre_step=5.0),
            True,
            True,
            0,
        ),
        (
            "case_d_timeout_with_delay_related_violation",
            np.asarray([0.0, 2.0, 2.0, 2.0, 3.0], dtype=np.float64),
            np.asarray([0.0, 0.1, 0.2, 0.3, 1.4], dtype=np.float64),
            _StubSafeGuard(max_speed_pre_step=2.0),
            True,
            False,
            0,
        ),
        (
            "case_e_request_unfinished",
            np.asarray([0.0, 2.0, 2.0], dtype=np.float64),
            np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
            _StubSafeGuard(),
            True,
            True,
            1,
        ),
    ],
)
def test_replay_sps_compliance_main_cases(
    name: str,
    speed_arr: np.ndarray,
    time_arr: np.ndarray,
    guard: _StubSafeGuard,
    expected_trigger: bool,
    expected_delay_pass: bool,
    expected_unfinished: int,
) -> None:
    pos_arr = np.arange(speed_arr.size, dtype=np.float64)

    result = replay_sps_compliance(
        label=name,
        pos_arr=pos_arr,
        speed_arr=speed_arr,
        sgu=cast(SafeGuardUtility, cast(object, guard)),
        asa_ap_list=[1000.0],
        asa_dp_list=[1010.0],
        step_delay_s=1.0,
        boundary_eps=0.0,
        time_arr=time_arr,
    )

    assert result.triggered_pass is expected_trigger
    assert result.delay_related_boundary_violation_pass is expected_delay_pass
    assert result.unfinished_count == expected_unfinished

    if name == "case_a_triggered_no_delay_violation":
        assert result.request_count == 1
        assert result.complete_count == 1
        assert result.delay_related_min_violation_count == 0
        assert result.delay_related_max_violation_count == 0

    if name == "case_b_not_triggered":
        assert result.request_count == 0
        assert result.first_failure_reason == "no_step_request_triggered"

    if name == "case_d_timeout_with_delay_related_violation":
        assert result.delay_related_max_violation_count >= 1
        assert result.first_failure_reason == "delay_related_boundary_violation"
