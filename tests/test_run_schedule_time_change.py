import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

import scripts.run_method_ablation as method_ablation
import scripts.run_schedule_time_change as schedule_change
from rl.experiment_utils import (
    build_default_training_args,
    load_run_metadata,
    resolve_training_run_spec,
)
from scripts.run_schedule_time_change import (
    DEFAULT_CHANGE_DISTANCE_M,
    DEFAULT_DELTA_TIMES_S,
    DEFAULT_OUTPUT_DIR,
    SUMMARY_FILENAME,
    CandidateEvaluation,
    ScheduleChangeCandidate,
    ScheduleChangeRunResult,
    _add_schedule_change_legend,
    _as_batch_observation,
    _make_experiment_dir,
    _reward_config_from_metadata,
    build_arg_parser,
    build_candidate_rank_key,
    build_schedule_change_case,
    load_schedule_change_candidates,
    load_schedule_change_summary,
    rank_candidate_evaluations,
    resolve_schedule_change_experiment_dir,
    run_evaluate,
    should_trigger_schedule_change,
)


def test_as_batch_observation_preserves_environment_normalized_values() -> None:
    observation = _as_batch_observation([0.25, -0.5, 1.0])

    assert observation.dtype == np.float32
    assert observation.shape == (1, 3)
    np.testing.assert_allclose(observation, [[0.25, -0.5, 1.0]])


def test_cli_requires_explicit_method_and_result_directories() -> None:
    parser = build_arg_parser()

    evaluate_args = parser.parse_args(
        [
            "evaluate",
            "--method-ablation-dir",
            "output/method",
            "--output-dir",
            "output/schedule",
        ]
    )
    show_args = parser.parse_args(["show", "--load-dir", "output/schedule"])

    assert evaluate_args.mode == "evaluate"
    assert evaluate_args.method_ablation_dir == Path("output/method")
    assert evaluate_args.output_dir == Path("output/schedule")
    assert evaluate_args.delta_times_s == DEFAULT_DELTA_TIMES_S
    assert evaluate_args.change_distance_m == DEFAULT_CHANGE_DISTANCE_M == 8_000.0
    assert not hasattr(evaluate_args, "model_dir")

    assert show_args.mode == "show"
    assert show_args.load_dir == Path("output/schedule")
    assert not hasattr(show_args, "output_dir")
    assert DEFAULT_OUTPUT_DIR == "output/paper_experiment/04_schedule_time_change"


@pytest.mark.parametrize(
    "arguments",
    (
        ["evaluate", "--model-dir", "output/model"],
        ["evaluate", "--reward-discount", "0.99"],
        ["evaluate", "--schedule-time-s", "465"],
        ["evaluate", "--step-distance", "30"],
        ["evaluate", "--reward-preset", "basic_safety_punctuality"],
    ),
)
def test_cli_rejects_retired_single_model_and_metadata_overrides(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            [
                "evaluate",
                "--method-ablation-dir",
                "output/method",
                "--output-dir",
                "output/schedule",
                *arguments[1:],
            ]
        )


def test_cli_parses_custom_delta_times() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "evaluate",
            "--method-ablation-dir",
            "output/method",
            "--output-dir",
            "output/schedule",
            "--delta-times-s",
            "0,-5, 7.5",
        ]
    )

    assert args.delta_times_s == (0.0, -5.0, 7.5)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate",
                "--method-ablation-dir",
                "output/method",
                "--output-dir",
                "output/schedule",
                "--delta-times-s",
                "0,30,30",
            ]
        )


def test_show_cli_rejects_figure_filename_override() -> None:
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["show", "--figure-name", "custom-name.pdf"])


def test_metadata_reward_snapshot_reconstructs_fixed_configuration() -> None:
    args = build_default_training_args()
    args.training_episodes = 8
    args.reference_curve_dir = "."
    spec = resolve_training_run_spec(args)
    config = _reward_config_from_metadata(spec.run_metadata.reward_config)

    assert config.enable_potential_punctuality
    assert not hasattr(config, "punctuality_potential_scale")


def test_metadata_reward_snapshot_rejects_retired_parameters() -> None:
    args = build_default_training_args()
    args.training_episodes = 8
    args.reference_curve_dir = "."
    snapshot = replace(
        resolve_training_run_spec(args).run_metadata.reward_config,
        punctuality_potential_scale=3.25,
    )
    with pytest.raises(ValueError, match="fixed DSPL protocol"):
        _reward_config_from_metadata(snapshot)


def _write_completed_method_ablation(
    output_root: Path,
) -> list[method_ablation.AblationRun]:
    args = method_ablation.build_arg_parser().parse_args(
        ["train", "--reference-curve-dir", ".", "--output-root", str(output_root)]
    )
    runs = method_ablation.resolve_run_matrix(args)
    statuses: dict[str, object] = {}
    artifact_names = (
        "policy_final",
        "policy_best",
        "metadata",
        "metadata_best",
        "episodes",
        "evaluations",
        "trajectory_final",
        "trajectory_best",
        "metrics_final",
        "metrics_best",
        "safety_diagnostics",
    )
    for run in runs:
        budget = run.training_spec.run_metadata.training_budget
        assert budget is not None
        completed_budget = replace(
            budget,
            actual_training_timesteps=budget.derived_total_timesteps,
            actual_training_rollouts=budget.training_rollouts,
            target_reached=True,
            stop_reason="target_reached",
        )
        metadata = replace(
            run.training_spec.run_metadata,
            training_budget=completed_budget,
        )
        for name in artifact_names:
            path = getattr(run.artifacts, name)
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            if name in {"metadata", "metadata_best"}:
                path.write_text(
                    json.dumps(metadata.to_mapping()),
                    encoding="utf-8",
                )
            else:
                path.touch()
        statuses[run.run_id] = {
            "status": "completed",
            "training_budget": completed_budget.to_mapping(),
        }
    manifest = method_ablation.build_manifest(args, runs, statuses)
    method_ablation._manifest_store(output_root).save_atomic(manifest)
    return runs


def test_candidate_discovery_loads_best_and_final_for_all_full_method_seeds(
    tmp_path: Path,
) -> None:
    _write_completed_method_ablation(tmp_path)

    candidates = load_schedule_change_candidates(tmp_path)

    assert len(candidates) == 10
    assert {candidate.seed for candidate in candidates} == set(
        method_ablation.DEFAULT_SEEDS
    )
    assert {candidate.source for candidate in candidates} == {"best", "final"}
    assert all(candidate.model_dir.is_dir() for candidate in candidates)


def test_candidate_discovery_rejects_inconsistent_metadata(tmp_path: Path) -> None:
    runs = _write_completed_method_ablation(tmp_path)
    full_method_run = next(run for run in runs if run.variant.id == "ppo_pprs_dspl")
    metadata_path = full_method_run.artifacts.metadata_best
    assert metadata_path is not None
    metadata = load_run_metadata(metadata_path.parent)
    metadata_path.write_text(
        json.dumps(replace(metadata, schedule_time_s=466.0).to_mapping()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inconsistent training metadata"):
        load_schedule_change_candidates(tmp_path)


def test_batch_dry_run_lists_ten_candidates_without_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    method_dir = tmp_path / "method"
    output_dir = tmp_path / "schedule"
    _write_completed_method_ablation(method_dir)
    args = build_arg_parser().parse_args(
        [
            "evaluate",
            "--method-ablation-dir",
            str(method_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    run_evaluate(args)

    assert "candidate_count:     10" in capsys.readouterr().out
    assert not output_dir.exists()


def test_batch_evaluation_writes_ranked_root_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_dir = tmp_path / "method"
    output_dir = tmp_path / "schedule"
    _write_completed_method_ablation(method_dir)
    calls: list[tuple[str, float]] = []

    monkeypatch.setattr(schedule_change.PPO, "load", lambda *_args, **_kwargs: object())

    def fake_run_one_case(**kwargs: object) -> ScheduleChangeRunResult:
        case = kwargs["case"]
        candidate_dir = Path(kwargs["experiment_dir"])
        load_dir = str(kwargs["load_dir"])
        assert isinstance(case, schedule_change.ScheduleChangeCase)
        case_dir = candidate_dir / case.token
        case_dir.mkdir()
        (case_dir / "trajectory.npz").touch()
        (case_dir / "metrics.json").touch()
        calls.append((load_dir, case.delta_time_s))
        energy_j = 100_000.0 if Path(load_dir).name == "best" else 200_000.0
        return _result(
            case=case,
            total_energy_j=energy_j,
            total_energy_kj=energy_j / 1000.0,
            schedule_change_triggered=case.delta_time_s != 0.0,
            trajectory_npz=f"{case.token}/trajectory.npz",
            trajectory_metrics_json=f"{case.token}/metrics.json",
        )

    monkeypatch.setattr(schedule_change, "_run_one_case", fake_run_one_case)
    args = build_arg_parser().parse_args(
        [
            "evaluate",
            "--method-ablation-dir",
            str(method_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    run_evaluate(args)

    assert len(calls) == 30
    summary = json.loads((output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["artifact_type"] == "schedule_time_change_selection"
    assert summary["candidate_count"] == 10
    assert summary["selected"]["source"] == "best"
    assert len(summary["cases"]) == 3
    assert all(
        (output_dir / case["trajectory_npz"]).is_file() for case in summary["cases"]
    )
    assert load_schedule_change_summary(output_dir)["selected"] == summary["selected"]


def test_default_schedule_change_matrix_is_original_plus30_minus30() -> None:
    assert DEFAULT_DELTA_TIMES_S == (0.0, 30.0, -30.0)


def test_schedule_change_case_labels_and_tokens() -> None:
    assert build_schedule_change_case(0.0).label == "Original"
    assert build_schedule_change_case(30.0).label == "Plus 30s"
    assert build_schedule_change_case(-30.0).label == "Minus 30s"
    assert build_schedule_change_case(30.0).token == "plus_30p0s"
    assert build_schedule_change_case(-30.0).token == "minus_30p0s"


def test_shared_legend_contains_only_cases_and_trigger_marker() -> None:
    figure, axis = plt.subplots()
    try:
        _ = axis.plot([0.0, 1.0], [0.0, 1.0], label="Track speed limit")
        original_handle = axis.plot([0.0, 1.0], [1.0, 1.0])[0]
        plus_handle = axis.plot([0.0, 1.0], [1.5, 1.5])[0]
        minus_handle = axis.plot([0.0, 1.0], [2.0, 2.0])[0]
        trigger_handle = axis.scatter([0.5], [1.0], marker="*", color="#C44E52")

        _add_schedule_change_legend(
            figure,
            case_handles=[original_handle, plus_handle, minus_handle],
            case_labels=["Original", "Plus 30s", "Minus 30s"],
            trigger_handle=trigger_handle,
        )

        assert axis.get_legend() is None
        assert len(figure.legends) == 1
        legend = figure.legends[0]
        assert legend.get_frame_on() is False
        assert legend._ncols == 4
        assert [text.get_text() for text in legend.get_texts()] == [
            "Original",
            "Plus 30s",
            "Minus 30s",
            "Schedule change",
        ]
    finally:
        plt.close(figure)


def test_should_trigger_schedule_change_once_when_crossing_forward() -> None:
    assert (
        should_trigger_schedule_change(
            previous_position_m=700.0,
            current_position_m=850.0,
            change_distance_m=800.0,
            direction=1,
            already_triggered=False,
            delta_time_s=10.0,
        )
        is True
    )
    assert (
        should_trigger_schedule_change(
            previous_position_m=700.0,
            current_position_m=850.0,
            change_distance_m=800.0,
            direction=1,
            already_triggered=True,
            delta_time_s=10.0,
        )
        is False
    )


def test_should_not_trigger_original_case() -> None:
    assert (
        should_trigger_schedule_change(
            previous_position_m=700.0,
            current_position_m=850.0,
            change_distance_m=800.0,
            direction=1,
            already_triggered=False,
            delta_time_s=0.0,
        )
        is False
    )


def test_should_trigger_schedule_change_when_crossing_backward() -> None:
    assert (
        should_trigger_schedule_change(
            previous_position_m=850.0,
            current_position_m=700.0,
            change_distance_m=800.0,
            direction=-1,
            already_triggered=False,
            delta_time_s=-10.0,
        )
        is True
    )


def _write_summary(path: Path) -> None:
    path.mkdir(parents=True)
    _ = (path / SUMMARY_FILENAME).write_text(
        json.dumps({"cases": []}),
        encoding="utf-8",
    )


def test_resolve_experiment_dir_accepts_direct_experiment(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "20260101_01"
    _write_summary(experiment_dir)

    assert resolve_schedule_change_experiment_dir(experiment_dir) == experiment_dir


def test_resolve_experiment_dir_rejects_parent_directory(tmp_path: Path) -> None:
    _write_summary(tmp_path / "20260101_01")

    with pytest.raises(FileNotFoundError, match="directly"):
        resolve_schedule_change_experiment_dir(tmp_path)


def test_resolve_experiment_dir_errors_without_summary(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=SUMMARY_FILENAME):
        _ = resolve_schedule_change_experiment_dir(tmp_path)


def test_make_experiment_dir_uses_exact_path_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "04_schedule_time_change" / "20260101_01"
    assert _make_experiment_dir(result_dir) == result_dir
    assert result_dir.is_dir()
    with pytest.raises(FileExistsError):
        _make_experiment_dir(result_dir)


def _result(**overrides: object) -> ScheduleChangeRunResult:
    values: dict[str, object] = {
        "case": build_schedule_change_case(0.0),
        "success": True,
        "precise_arrival": True,
        "punctual_arrival": True,
        "total_reward": 1.0,
        "initial_schedule_time_s": 465.0,
        "final_schedule_time_s": 465.0,
        "total_time_s": 465.0,
        "time_error_s": 0.0,
        "abs_time_error_s": 0.0,
        "stop_error_m": 0.1,
        "total_energy_kj": 100.0,
        "total_energy_j": 100_000.0,
        "final_position_m": 30_000.0,
        "final_speed_mps": 0.0,
        "episode_steps": 100,
        "min_safety_margin_mps": 0.0,
        "mean_safety_margin_mps": 1.0,
        "safety_violation_count": 0,
        "safe": True,
        "feasible": True,
        "schedule_change_triggered": False,
        "schedule_change_step": None,
        "schedule_change_position_m": None,
        "schedule_change_speed_mps": None,
        "trajectory_npz": "original/trajectory.npz",
        "trajectory_metrics_json": "original/metrics.json",
    }
    values.update(overrides)
    return ScheduleChangeRunResult(**values)  # type: ignore[arg-type]


def test_candidate_rank_prioritizes_robust_constraints_before_energy() -> None:
    feasible = (_result(total_energy_j=200_000.0),)
    lower_energy_failure = (
        _result(
            feasible=False,
            punctual_arrival=False,
            abs_time_error_s=20.0,
            total_energy_j=50_000.0,
        ),
    )

    assert build_candidate_rank_key(feasible) > build_candidate_rank_key(
        lower_energy_failure
    )


def test_candidate_rank_uses_worst_error_before_mean_energy() -> None:
    low_worst_error = (
        _result(stop_error_m=0.1, total_energy_j=200_000.0),
        _result(stop_error_m=0.2, total_energy_j=200_000.0),
    )
    high_worst_error = (
        _result(stop_error_m=0.1, total_energy_j=50_000.0),
        _result(stop_error_m=0.3, total_energy_j=50_000.0),
    )

    assert build_candidate_rank_key(low_worst_error) > build_candidate_rank_key(
        high_worst_error
    )


def test_exact_rank_tie_prefers_best_then_run_id() -> None:
    args = build_default_training_args()
    args.training_episodes = 8
    args.reference_curve_dir = "."
    metadata = resolve_training_run_spec(args).run_metadata
    results = (_result(),)

    def evaluation(run_id: str, source: str) -> CandidateEvaluation:
        candidate = ScheduleChangeCandidate(
            candidate_id=f"{run_id}__{source}",
            run_id=run_id,
            seed=11,
            source=source,  # type: ignore[arg-type]
            model_dir=Path("output/model"),
            metadata=metadata,
        )
        return CandidateEvaluation(
            candidate=candidate,
            candidate_dir=Path(candidate.candidate_id),
            results=results,
            rank_key=build_candidate_rank_key(results),
        )

    ranked = rank_candidate_evaluations(
        [
            evaluation("run_b", "best"),
            evaluation("run_a", "final"),
            evaluation("run_a", "best"),
        ]
    )

    assert [item.candidate.candidate_id for item in ranked] == [
        "run_a__best",
        "run_b__best",
        "run_a__final",
    ]
