"""Tests for migrated paper result figures and CLI workflows."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mtto.domain.safeguard import ViolationKind  # noqa: E402
from mtto.domain.scenario import Task  # noqa: E402
from mtto.domain.speed_profile import SpeedProfile  # noqa: E402
from mtto.dp.solver import VariableSpacingDPOptimizer  # noqa: E402
from mtto.evaluation.quality import SpsEventKind, assess  # noqa: E402
from mtto.io.artifacts import (  # noqa: E402
    BEST_DIR,
    POLICY_ZIP,
    RLResult,
    RunKind,
    RunPayload,
    RunRecord,
    TrainingOutcome,
    file_sha256,
    task_to_json,
    write_run,
)
from mtto.rl.evaluate import run_policy  # noqa: E402
from mtto.rl.state import TerminationReason  # noqa: E402
from paper.figures import (  # noqa: E402
    dp_redundancy_error,
    dp_result,
    load_paper_scenario,
    load_paper_task,
    rl_result,
    speed_profile_comparison,
    sps_compliance,
)
from tests.golden.drive import build_env, load_actions  # noqa: E402

# Small, fast DP task (same parameters as tests/test_workflows.py's
# test_solve_dp_workflow): only used so these figure/CLI tests have a DP run
# whose target differs from the RL run's target (see
# test_comparison_task_mismatch_raises). None of the assertions below depend
# on this being a globally DP-optimal trajectory for the full route.
_SMALL_DP_TASK = Task(
    start_position_m=135.0,
    target_position_m=335.0,
    schedule_time_s=40.0,
    max_jerk_mps3=0.75,
    max_stop_error_m=0.3,
    max_arr_time_error_s=10.0,
)


def _solve_small_dp_profile(scenario) -> SpeedProfile:
    optimizer = VariableSpacingDPOptimizer(
        scenario=scenario,
        task=_SMALL_DP_TASK,
        cache_dir=None,
        delta_speed=1.0,
        uniform_step_size=50.0,
        precompute_mode="serial",
        show_precompute_progress=False,
    )
    profile = optimizer.optimize(
        _SMALL_DP_TASK.start_position_m,
        0.0,
        _SMALL_DP_TASK.target_position_m,
        0.0,
        _SMALL_DP_TASK.schedule_time_s,
    )
    assert profile is not None
    return profile


def _replay_rl_profile(
    case_name: str, preset: str = "basic_safety_punctuality"
) -> SpeedProfile:
    """Replay a frozen golden action sequence (tests/golden/actions/) live."""
    actions = load_actions(case_name)
    scripted = iter(actions)
    run = run_policy(lambda _: next(scripted), build_env(preset))
    return run.profile


# ============================================================================
# 1. Migrated Domain & CLI Parsing Assertions
# ============================================================================


@pytest.mark.parametrize(
    ("parser_factory", "required_args"),
    [
        (rl_result._build_cli_parser, ["--rl-run", "runs/rl"]),
        (dp_result._build_cli_parser, ["--dp-run", "runs/dp"]),
        (
            speed_profile_comparison._build_cli_parser,
            ["--dp-run", "runs/dp", "--rl-run", "runs/rl"],
        ),
        (dp_redundancy_error._build_cli_parser, ["--dp-run", "runs/dp"]),
        (
            sps_compliance._build_cli_parser,
            [
                "--analysis-mode",
                "single",
                "--trajectory-kind",
                "dp",
                "--dp-run",
                "runs/dp",
            ],
        ),
    ],
)
def test_figure_clis_accept_only_output_directory(
    parser_factory: Callable[[], argparse.ArgumentParser],
    required_args: list[str],
) -> None:
    parser = parser_factory()
    out_dir = Path("output/figures")
    args = parser.parse_args([*required_args, "--output-dir", str(out_dir)])
    assert args.output_dir == out_dir

    with pytest.raises(SystemExit):
        parser.parse_args([*required_args, "--output-file", "custom.pdf"])


def test_fixed_generic_figure_filenames_are_pdf() -> None:
    assert rl_result.FIGURE_FILENAME == "rl_result.pdf"
    assert dp_result.FIGURE_FILENAME == "dp_result.pdf"
    assert speed_profile_comparison.FIGURE_FILENAME == "dp_rl_actual_comparison.pdf"
    assert dp_redundancy_error.FIGURE_FILENAME == "dp_redundancy_error.pdf"
    assert sps_compliance.COMPARE_FIGURE_FILENAME == "dp_rl_sps_compliance.pdf"
    assert set(sps_compliance.SINGLE_FIGURE_FILENAMES.values()) == {
        "dp_sps_compliance.pdf",
        "rl_sps_compliance.pdf",
    }


def test_speed_profile_comparison_cli_accepts_repeated_baselines() -> None:
    parser = speed_profile_comparison._build_cli_parser()
    args = parser.parse_args(
        [
            "--dp-run",
            "runs/dp",
            "--rl-run",
            "runs/rl",
            "--baseline-rl",
            "PPO-BR=output/ppo_br/best",
            "--baseline-rl",
            "PPO=output/ppo/best",
        ]
    )
    parsed = [
        speed_profile_comparison._parse_baseline_spec(b) for b in args.baseline_rl
    ]
    assert parsed == [
        ("PPO-BR", "output/ppo_br/best"),
        ("PPO", "output/ppo/best"),
    ]


@pytest.mark.parametrize(
    "raw",
    ("output/ppo_br/best", "=output/ppo_br", "PPO-BR="),
)
def test_speed_profile_comparison_baseline_spec_validation(raw: str) -> None:
    with pytest.raises(ValueError, match="LABEL=DIR"):
        speed_profile_comparison._parse_baseline_spec(raw)


def test_sps_compliance_cli_validations() -> None:
    parser = sps_compliance._build_cli_parser()

    # Single mode requires --trajectory-kind
    args_single_no_kind = parser.parse_args(
        ["--analysis-mode", "single", "--dp-run", "runs/dp"]
    )
    with pytest.raises(SystemExit):
        sps_compliance._validate_cli_args(parser, args_single_no_kind)

    # Compare mode rejects --trajectory-kind
    args_compare_with_kind = parser.parse_args(
        [
            "--analysis-mode",
            "compare",
            "--dp-run",
            "runs/dp",
            "--rl-run",
            "runs/rl",
            "--trajectory-kind",
            "dp",
        ]
    )
    with pytest.raises(SystemExit):
        sps_compliance._validate_cli_args(parser, args_compare_with_kind)


def test_sps_compliance_parse_output_mode() -> None:
    assert sps_compliance._parse_output_mode("text+plot") == {"text", "plot"}
    assert sps_compliance._parse_output_mode("text,json") == {"text", "json"}
    with pytest.raises(ValueError, match="Unknown output mode"):
        sps_compliance._parse_output_mode("text,invalid")


def test_min_limit_margin_uses_moving_samples_against_step_limit() -> None:
    safeguard = SimpleNamespace(
        speed_limits=np.asarray([20.0, 10.0]),
        speed_limit_intervals=np.asarray([0.0, 100.0]),
        params=SimpleNamespace(factor=0.5),
    )
    profile = SpeedProfile.from_arrays(
        position_m=[0.0, 100.0, 200.0],
        speed_mps=[0.0, 6.0, 0.0],
        time_s=[0.0, 10.0, 20.0],
        propulsion_energy_kj=[0.0, 10.0, 20.0],
        levitation_energy_kj=[0.0, 5.0, 10.0],
    )
    margin = speed_profile_comparison.compute_min_limit_margin_kmh(profile, safeguard)
    # After 100m, limit is 10.0 * 0.5 = 5.0 m/s; speed is 6.0 m/s:
    # margin = -1.0 m/s = -3.6 km/h
    assert margin == pytest.approx((5.0 - 6.0) * 3.6, abs=0.1)


def test_comparison_table_formatting_flags_outside_tolerance() -> None:
    metrics_by_label = [
        (
            "DP",
            speed_profile_comparison.ProfileMetrics(
                time_error_s=-8.3,
                stop_error_m=0.0,
                total_energy_kwh=100.0,
                comfort_tav=1.0,
                within_tolerance=True,
                min_limit_margin_kmh=0.0,
            ),
        ),
        (
            "PPO-CR",
            speed_profile_comparison.ProfileMetrics(
                time_error_s=132.0,
                stop_error_m=0.1,
                total_energy_kwh=90.0,
                comfort_tav=0.5,
                within_tolerance=False,
                min_limit_margin_kmh=2.7,
            ),
        ),
        (
            speed_profile_comparison.ACTUAL_LABEL,
            speed_profile_comparison.ProfileMetrics(
                time_error_s=4.6,
                stop_error_m=0.0,
                total_energy_kwh=200.0,
                comfort_tav=None,
                within_tolerance=True,
                min_limit_margin_kmh=12.4,
            ),
        ),
    ]
    table = speed_profile_comparison.format_comparison_table(metrics_by_label)

    assert "-8.300" in table and "+132.000" in table
    assert "Within stop/time tolerance" in table
    assert "55.00^a" in table  # (200 - 90) / 200 outside tolerance
    assert "-10.00^a" in table  # (90 - 100) / 100 outside tolerance
    assert "50.00 " in table  # DP vs actual, within tolerance
    assert "Min. margin to line speed limit (km/h)" in table


def test_calc_redundant_operation_time_arr_formula() -> None:
    task = SimpleNamespace(
        schedule_time_s=10.0,
        target_position_m=20.0,
    )

    def fake_min_time(
        begin_pos: float, begin_spd: float, end_pos: float, end_spd: float
    ) -> float:
        assert end_pos == 20.0
        assert end_spd == 0.0
        return 0.1 * begin_pos + 0.5 * begin_spd

    redundant = dp_result._calc_redundant_operation_time_arr(
        pos_arr=np.asarray([0.0, 10.0, 20.0]),
        speed_arr=np.asarray([0.0, 2.0, 0.0]),
        cum_time_arr=np.asarray([0.0, 2.0, 5.0]),
        task=task,  # type: ignore[arg-type]
        min_remaining_time_fn=fake_min_time,
    )
    # schedule_time - cum_time - min_remaining:
    # node 0: 10 - 0 - (0 + 0) = 10.0
    # node 1: 10 - 2 - (1.0 + 1.0) = 6.0
    # node 2: 10 - 5 - (2.0 + 0) = 3.0
    np.testing.assert_allclose(redundant, np.asarray([10.0, 6.0, 3.0]))


def test_compute_expected_redundant_operation_time_linear() -> None:
    expected = dp_redundancy_error.compute_expected_redundant_operation_time(
        pos_arr=np.asarray([0.0, 50.0, 100.0, 120.0]),
        start_position=0.0,
        target_position=100.0,
        initial_redundant_s=20.0,
    )
    np.testing.assert_allclose(expected, np.asarray([20.0, 10.0, 0.0, 0.0]))


def test_summarize_error_statistics() -> None:
    summary = dp_redundancy_error.summarize_error_statistics(
        pos_arr=np.asarray([0.0, 10.0, 20.0, 30.0]),
        cum_time_arr=np.asarray([0.0, 1.0, 2.0, 3.0]),
        error_arr=np.asarray([-2.0, 0.0, 0.5, 4.0]),
        zero_eps=0.1,
    )
    assert summary["overall"]["sample_count"] == 4
    assert summary["overall"]["max_abs_s"] == pytest.approx(4.0)
    assert summary["positive"]["sample_count"] == 2
    assert summary["negative"]["sample_count"] == 1
    assert summary["negative"]["min_s"] == pytest.approx(-2.0)
    assert summary["near_zero"]["sample_count"] == 1


def test_dp_redundancy_error_validates_same_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        dp_redundancy_error._validate_same_length(
            [
                ("pos", np.asarray([0.0, 1.0])),
                ("speed", np.asarray([0.0])),
            ]
        )


# ============================================================================
# 2. Synthetic Test Runs (using mtto.io.artifacts.write_run)
# ============================================================================


@pytest.fixture(scope="module")
def paper_env() -> tuple[Any, Any, Any]:
    scenario = load_paper_scenario()
    paper_task = load_paper_task()
    dp_task = _SMALL_DP_TASK
    return scenario, paper_task, dp_task


@pytest.fixture(scope="module")
def dp_run_dir(
    tmp_path_factory: pytest.TempPathFactory, paper_env: tuple[Any, Any, Any]
) -> Path:
    scenario, _, dp_task = paper_env
    run_dir = tmp_path_factory.mktemp("dp_run")

    profile = _solve_small_dp_profile(scenario)
    quality = assess(profile, scenario, dp_task)
    record = RunRecord(
        run_id="test_dp_solve",
        kind=RunKind.DP_SOLVE,
        config={},
        scenario_hash=scenario.scenario_hash,
        task=task_to_json(dp_task),
        policy_io_version=None,
        mtto_version="0.1.0",
        created_at="2026-09-28T00:00:00+00:00",
    )
    write_run(run_dir, record, RunPayload(profile=profile, quality=quality))
    return run_dir


@pytest.fixture(scope="module")
def rl_eval_run_dir(
    tmp_path_factory: pytest.TempPathFactory, paper_env: tuple[Any, Any, Any]
) -> Path:
    scenario, paper_task, _ = paper_env
    run_dir = tmp_path_factory.mktemp("rl_eval_run")

    profile = _replay_rl_profile("stop_in_zone")
    quality = assess(profile, scenario, paper_task)
    record = RunRecord(
        run_id="test_rl_eval",
        kind=RunKind.EVALUATION,
        config={},
        scenario_hash=scenario.scenario_hash,
        task=task_to_json(paper_task),
        policy_io_version=None,
        mtto_version="0.1.0",
        created_at="2026-09-28T00:00:00+00:00",
    )
    write_run(run_dir, record, RunPayload(profile=profile, quality=quality))
    return run_dir


@pytest.fixture(scope="module")
def dp_comp_run_dir(
    tmp_path_factory: pytest.TempPathFactory, paper_env: tuple[Any, Any, Any]
) -> Path:
    """DP run with paper_task for speed_profile_comparison."""
    scenario, paper_task, _ = paper_env
    run_dir = tmp_path_factory.mktemp("dp_comp_run")

    profile = _replay_rl_profile("stop_in_zone")
    quality = assess(profile, scenario, paper_task)
    record = RunRecord(
        run_id="test_dp_comp",
        kind=RunKind.DP_SOLVE,
        config={},
        scenario_hash=scenario.scenario_hash,
        task=task_to_json(paper_task),
        policy_io_version=None,
        mtto_version="0.1.0",
        created_at="2026-09-28T00:00:00+00:00",
    )
    write_run(run_dir, record, RunPayload(profile=profile, quality=quality))
    return run_dir


@pytest.fixture(scope="module")
def rl_train_run_dir(
    tmp_path_factory: pytest.TempPathFactory, paper_env: tuple[Any, Any, Any]
) -> Path:
    """RL training run with best/ artifacts."""
    scenario, paper_task, _ = paper_env
    run_dir = tmp_path_factory.mktemp("rl_train_run")
    best_dir = run_dir / BEST_DIR
    best_dir.mkdir(parents=True)

    dummy_policy_path = run_dir / POLICY_ZIP
    with zipfile.ZipFile(dummy_policy_path, "w") as zf:
        zf.writestr("model.txt", "dummy model")
    best_policy_path = best_dir / POLICY_ZIP
    with zipfile.ZipFile(best_policy_path, "w") as zf:
        zf.writestr("model.txt", "best dummy model")

    root_sha = file_sha256(dummy_policy_path)
    best_sha = file_sha256(best_policy_path)

    profile = _replay_rl_profile("stop_in_zone")
    quality = assess(profile, scenario, paper_task)

    run_id = "test_rl_train"
    record = RunRecord(
        run_id=run_id,
        kind=RunKind.RL_TRAIN,
        config={"seed": 42},
        scenario_hash=scenario.scenario_hash,
        task=task_to_json(paper_task),
        policy_io_version=1,
        mtto_version="0.1.0",
        created_at="2026-09-28T00:00:00+00:00",
    )
    result = RLResult(
        termination_reason=TerminationReason.STOPPED_IN_ZONE,
        truncated=False,
        total_reward=100.0,
        steps=len(profile.position_m),
        final_position_m=float(profile.position_m[-1]),
        final_speed_mps=float(profile.speed_mps[-1]),
        final_time_s=float(profile.time_s[-1]),
        deterministic=True,
        policy_run_id=run_id,
        policy_sha256=root_sha,
        training=TrainingOutcome(
            actual_training_timesteps=1000,
            actual_training_rollouts=10,
            actual_completed_episodes=5,
            target_reached=True,
            stop_reason="success",
        ),
    )
    best_result = RLResult(
        termination_reason=TerminationReason.STOPPED_IN_ZONE,
        truncated=False,
        total_reward=105.0,
        steps=len(profile.position_m),
        final_position_m=float(profile.position_m[-1]),
        final_speed_mps=float(profile.speed_mps[-1]),
        final_time_s=float(profile.time_s[-1]),
        deterministic=True,
        policy_run_id=run_id,
        policy_sha256=best_sha,
        training=None,
    )
    best_payload = RunPayload(profile=profile, quality=quality, result=best_result)
    payload = RunPayload(
        profile=profile,
        quality=quality,
        result=result,
        best=best_payload,
    )
    write_run(run_dir, record, payload)
    return run_dir


# ============================================================================
# 3. Noninteractive Agg Backend Figure Execution Tests
# ============================================================================


def test_rl_result_execution(
    rl_eval_run_dir: Path, rl_train_run_dir: Path, tmp_path: Path
) -> None:
    # 1. Normal evaluation run
    out_eval = tmp_path / "rl_eval"
    rl_result.main(
        ["--rl-run", str(rl_eval_run_dir), "--output-dir", str(out_eval), "--no-show"]
    )
    fig_eval = out_eval / rl_result.FIGURE_FILENAME
    assert fig_eval.is_file()
    assert fig_eval.stat().st_size > 0

    # 2. Training run with --rl-best
    out_best = tmp_path / "rl_best"
    rl_result.main(
        [
            "--rl-run",
            str(rl_train_run_dir),
            "--rl-best",
            "--output-dir",
            str(out_best),
            "--no-show",
        ]
    )
    fig_best = out_best / rl_result.FIGURE_FILENAME
    assert fig_best.is_file()
    assert fig_best.stat().st_size > 0

    # 3. Dry-run mode: prints metrics, does not generate figure
    out_dry = tmp_path / "rl_dry"
    rl_result.main(
        ["--rl-run", str(rl_eval_run_dir), "--dry-run", "--output-dir", str(out_dry)]
    )
    assert not (out_dry / rl_result.FIGURE_FILENAME).exists()


def test_dp_result_execution(dp_run_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "dp_result"
    dp_result.main(
        ["--dp-run", str(dp_run_dir), "--output-dir", str(out_dir), "--no-show"]
    )
    fig_file = out_dir / dp_result.FIGURE_FILENAME
    assert fig_file.is_file()
    assert fig_file.stat().st_size > 0


def test_speed_profile_comparison_execution(
    dp_comp_run_dir: Path,
    rl_eval_run_dir: Path,
    rl_train_run_dir: Path,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "comparison"
    speed_profile_comparison.main(
        [
            "--dp-run",
            str(dp_comp_run_dir),
            "--rl-run",
            str(rl_eval_run_dir),
            "--baseline-rl",
            f"PPO-Best={rl_train_run_dir}",
            "--output-dir",
            str(out_dir),
            "--no-show",
        ]
    )

    fig_file = out_dir / speed_profile_comparison.FIGURE_FILENAME
    assert fig_file.is_file()
    assert fig_file.stat().st_size > 0

    table_file = out_dir / "dp_rl_actual_comparison_table.md"
    assert table_file.is_file()
    content = table_file.read_text(encoding="utf-8")
    assert speed_profile_comparison.PROPOSED_LABEL in content
    assert "DP" in content
    assert "PPO-Best" in content
    assert "| Actual " in content
    assert "Total energy (kWh)" in content


def test_dp_redundancy_error_execution(dp_run_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "redundancy_error"
    json_path = out_dir / "error_summary.json"
    csv_path = out_dir / "error_series.csv"

    dp_redundancy_error.main(
        [
            "--dp-run",
            str(dp_run_dir),
            "--output-dir",
            str(out_dir),
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--no-show",
        ]
    )

    fig_file = out_dir / dp_redundancy_error.FIGURE_FILENAME
    assert fig_file.is_file()
    assert fig_file.stat().st_size > 0

    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "statistics" in data
    summary = data["statistics"]
    assert "overall" in summary
    assert "positive" in summary
    assert "negative" in summary
    assert "near_zero" in summary
    assert summary["overall"]["sample_count"] > 0

    assert csv_path.is_file()
    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) > 1
    assert (
        "position_m,speed_mps,cum_time_s,actual_redundant_time_s,expected_redundant_time_s,error_s"
        in csv_lines[0]
    )


def test_sps_compliance_single_execution_matches_audit_counters(
    dp_run_dir: Path, tmp_path: Path, paper_env: tuple[Any, Any, Any]
) -> None:
    scenario, _, dp_task = paper_env
    out_dir = tmp_path / "sps_single"
    json_path = out_dir / "sps_report.json"

    sps_compliance.main(
        [
            "--analysis-mode",
            "single",
            "--trajectory-kind",
            "dp",
            "--dp-run",
            str(dp_run_dir),
            "--output-mode",
            "text+plot,json",
            "--output-dir",
            str(out_dir),
            "--json-output-path",
            str(json_path),
            "--no-show",
        ]
    )

    fig_file = out_dir / sps_compliance.SINGLE_FIGURE_FILENAMES["dp"]
    assert fig_file.is_file()
    assert fig_file.stat().st_size > 0

    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    res = payload["result"]
    counters = res["counters"]

    # Retrieve quality report from the run to assert counters match audit directly
    profile = _solve_small_dp_profile(scenario)
    quality = assess(profile, scenario, dp_task)
    audit = quality.audit

    expected_requests = len(
        [e for e in audit.events if e.kind == SpsEventKind.REQUEST_START]
    )
    expected_completes = len(
        [e for e in audit.events if e.kind == SpsEventKind.STEP_COMPLETE]
    )
    expected_unfinishes = len(
        [e for e in audit.events if e.kind == SpsEventKind.REQUEST_UNFINISHED]
    )
    expected_max_viols = len(
        [v for v in audit.violations if v.kind == ViolationKind.OVER_UPPER_LIMIT]
    )
    expected_min_viols = len(
        [v for v in audit.violations if v.kind == ViolationKind.UNDER_LOWER_LIMIT]
    )

    assert res["triggered"]["request_count"] == expected_requests
    assert counters["complete_count"] == expected_completes
    assert counters["unfinished_count"] == expected_unfinishes
    assert counters["max_violation_count"] == expected_max_viols
    assert counters["min_violation_count"] == expected_min_viols


def test_sps_compliance_compare_execution(
    dp_comp_run_dir: Path, rl_eval_run_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "sps_compare"
    json_path = out_dir / "sps_compare_report.json"

    sps_compliance.main(
        [
            "--analysis-mode",
            "compare",
            "--dp-run",
            str(dp_comp_run_dir),
            "--rl-run",
            str(rl_eval_run_dir),
            "--output-mode",
            "text+plot,json",
            "--output-dir",
            str(out_dir),
            "--json-output-path",
            str(json_path),
            "--no-show",
        ]
    )

    fig_file = out_dir / sps_compliance.COMPARE_FIGURE_FILENAME
    assert fig_file.is_file()
    assert fig_file.stat().st_size > 0

    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "results" in payload
    assert "dp" in payload["results"]
    assert "rl" in payload["results"]


# ============================================================================
# 4. Error Paths
# ============================================================================


def test_scenario_hash_mismatch_raises(
    dp_run_dir: Path, rl_eval_run_dir: Path, tmp_path: Path
) -> None:
    # Corrupt scenario hash in run.json
    corrupt_dir = tmp_path / "corrupt_hash_run"
    corrupt_dir.mkdir(parents=True)
    for f in dp_run_dir.iterdir():
        if f.name == "run.json":
            data = json.loads(f.read_text(encoding="utf-8"))
            data["scenario_hash"] = "invalid_hash_12345"
            (corrupt_dir / f.name).write_text(json.dumps(data), encoding="utf-8")
        else:
            (corrupt_dir / f.name).write_bytes(f.read_bytes())

    # dp_result must fail with scenario hash mismatch
    with pytest.raises(SystemExit):
        dp_result.main(["--dp-run", str(corrupt_dir), "--no-show"])

    # rl_result must fail with scenario hash mismatch
    with pytest.raises(SystemExit):
        rl_result.main(["--rl-run", str(corrupt_dir), "--no-show"])

    # speed_profile_comparison must fail with scenario hash mismatch
    with pytest.raises(SystemExit):
        speed_profile_comparison.main(
            [
                "--dp-run",
                str(corrupt_dir),
                "--rl-run",
                str(rl_eval_run_dir),
                "--no-show",
            ]
        )

    # dp_redundancy_error must fail with scenario hash mismatch
    with pytest.raises(SystemExit):
        dp_redundancy_error.main(["--dp-run", str(corrupt_dir), "--no-show"])

    # sps_compliance must fail with scenario hash mismatch
    with pytest.raises(SystemExit):
        sps_compliance.main(
            [
                "--analysis-mode",
                "single",
                "--trajectory-kind",
                "dp",
                "--dp-run",
                str(corrupt_dir),
                "--no-show",
            ]
        )


def test_run_kind_mismatch_raises(dp_run_dir: Path, rl_eval_run_dir: Path) -> None:
    # Passing DP run to --rl-run
    with pytest.raises(SystemExit):
        rl_result.main(["--rl-run", str(dp_run_dir), "--no-show"])

    # Passing RL run to --dp-run
    with pytest.raises(SystemExit):
        dp_result.main(["--dp-run", str(rl_eval_run_dir), "--no-show"])

    # Passing wrong types in comparison
    with pytest.raises(SystemExit):
        speed_profile_comparison.main(
            ["--dp-run", str(rl_eval_run_dir), "--rl-run", str(dp_run_dir), "--no-show"]
        )


def test_rl_best_without_best_artifacts_raises(rl_eval_run_dir: Path) -> None:
    # rl_eval_run has kind=EVALUATION and no best/ artifacts
    with pytest.raises(SystemExit):
        rl_result.main(["--rl-run", str(rl_eval_run_dir), "--rl-best", "--no-show"])


def test_comparison_task_mismatch_raises(
    dp_run_dir: Path, rl_eval_run_dir: Path
) -> None:
    # dp_run_dir has target 335.0, rl_eval_run_dir has target 29270.046
    with pytest.raises(SystemExit):
        speed_profile_comparison.main(
            [
                "--dp-run",
                str(dp_run_dir),
                "--rl-run",
                str(rl_eval_run_dir),
                "--no-show",
            ]
        )


def test_comparison_schedule_time_none_raises(
    tmp_path: Path, paper_env: tuple[Any, Any, Any], rl_train_run_dir: Path
) -> None:
    scenario, paper_task, _ = paper_env
    no_sched_task = replace(paper_task, schedule_time_s=None)
    run_dir = tmp_path / "dp_no_sched"
    profile = _replay_rl_profile("stop_in_zone")
    quality = assess(profile, scenario, no_sched_task)
    record = RunRecord(
        run_id="test_dp_no_sched",
        kind=RunKind.DP_SOLVE,
        config={},
        scenario_hash=scenario.scenario_hash,
        task=task_to_json(no_sched_task),
        policy_io_version=None,
        mtto_version="0.1.0",
        created_at="2026-09-28T00:00:00+00:00",
    )
    write_run(run_dir, record, RunPayload(profile=profile, quality=quality))

    with pytest.raises(SystemExit):
        speed_profile_comparison.main(
            [
                "--dp-run",
                str(run_dir),
                "--rl-run",
                str(rl_train_run_dir),
                "--no-show",
            ]
        )


def test_comparison_rl_best_applies_to_all_runs(
    dp_comp_run_dir: Path,
    rl_train_run_dir: Path,
    rl_eval_run_dir: Path,
    tmp_path: Path,
) -> None:
    # 1. Main RL run lacks best/ while --rl-best is requested -> error
    with pytest.raises(SystemExit):
        speed_profile_comparison.main(
            [
                "--dp-run",
                str(dp_comp_run_dir),
                "--rl-run",
                str(rl_eval_run_dir),
                "--rl-best",
                "--no-show",
            ]
        )

    # 2. Baseline lacks best/ while --rl-best is requested -> error
    with pytest.raises(SystemExit):
        speed_profile_comparison.main(
            [
                "--dp-run",
                str(dp_comp_run_dir),
                "--rl-run",
                str(rl_train_run_dir),
                "--baseline-rl",
                f"EvalBaseline={rl_eval_run_dir}",
                "--rl-best",
                "--no-show",
            ]
        )

    # 3. Main RL run and baseline both have best/ -> success
    out_dir = tmp_path / "comp_rl_best"
    speed_profile_comparison.main(
        [
            "--dp-run",
            str(dp_comp_run_dir),
            "--rl-run",
            str(rl_train_run_dir),
            "--baseline-rl",
            f"TrainBest={rl_train_run_dir}",
            "--rl-best",
            "--output-dir",
            str(out_dir),
            "--no-show",
        ]
    )
    assert (out_dir / speed_profile_comparison.FIGURE_FILENAME).is_file()
