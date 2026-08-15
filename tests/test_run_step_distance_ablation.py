import json
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from matplotlib.figure import Figure

import scripts.run_step_distance_ablation as step_distance_ablation
from rl.reward_diagnostics import REWARD_NAMES
from scripts.run_step_distance_ablation import (
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_STEP_DISTANCES,
    FIXED_REWARD_PRESET,
    STEP_DISTANCE_MANIFEST_FILENAME,
    StepDistanceCurveAggregate,
    build_arg_parser,
    build_curve_aggregates,
    build_metric_aggregates,
    build_step_distance_manifest,
    load_step_distance_manifest,
    plot_curve_aggregates,
    resolve_metric_source,
    resolve_step_distance_run_matrix,
    save_compact_figure,
    train_step_distance_run,
)


class _RunSpec(Protocol):
    final_output_dir: str


def _write_reward_diagnostics_npz(
    final_dir: Path,
    *,
    rewards: list[float],
    lengths: list[float],
) -> Path:
    final_dir.mkdir(parents=True, exist_ok=True)
    reward_diagnostics_path = final_dir / "reward_diagnostics.npz"
    episode_sums = np.zeros((len(rewards), len(REWARD_NAMES)), dtype=np.float64)
    episode_sums[:, 0] = rewards
    episode_sums[:, -1] = rewards
    end_steps = np.arange(1, len(rewards) + 1, dtype=np.int64)
    np.savez(
        reward_diagnostics_path,
        schema_version=np.asarray([1], dtype=np.int16),
        reward_names=np.asarray(REWARD_NAMES),
        rollout_end_step=np.asarray([end_steps[-1]]),
        rollout_transition_count=np.asarray([int(sum(lengths))]),
        rollout_reward_sum=episode_sums.sum(axis=0, keepdims=True),
        rollout_reward_abs_sum=np.abs(episode_sums).sum(axis=0, keepdims=True),
        rollout_reward_nonzero_count=np.count_nonzero(
            episode_sums, axis=0, keepdims=True
        ),
        rollout_reward_cross_product=np.asarray([episode_sums.T @ episode_sums]),
        episode_end_step=end_steps,
        episode_worker_rank=np.zeros(len(rewards), dtype=np.int16),
        episode_index=np.arange(len(rewards), dtype=np.int64),
        episode_length=np.asarray(lengths, dtype=np.int32),
        episode_terminated=np.ones(len(rewards), dtype=np.bool_),
        episode_truncated=np.zeros(len(rewards), dtype=np.bool_),
        episode_complete=np.ones(len(rewards), dtype=np.bool_),
        episode_reward_sums=episode_sums,
    )
    return reward_diagnostics_path


def _write_trajectory_metrics_json(
    output_dir: Path,
    *,
    stop_error_m: float,
    time_error_s: float,
    total_energy_kj: float,
    comfort_tav: float = 0.0,
    comfort_rms: float = 0.0,
    comfort_er_pct: float = 0.0,
    file_name: str = "best_trajectory_metrics.json",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / file_name
    _ = metrics_path.write_text(
        json.dumps(
            {
                "stop_error_m": stop_error_m,
                "time_error_s": time_error_s,
                "total_energy_kj": total_energy_kj,
                "comfort_tav": comfort_tav,
                "comfort_rms": comfort_rms,
                "comfort_er_pct": comfort_er_pct,
            }
        ),
        encoding="utf-8",
    )
    return metrics_path


def _write_run_metadata(
    final_dir: Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    final_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = final_dir / "run_metadata.json"
    payload: dict[str, object] = {"rollout_record_trigger_mode": "steps"}
    if metadata:
        payload.update(metadata)
    _ = metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_path


def test_train_cli_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["train", "--reference-curve-dir", "."])

    assert args.output_root == DEFAULT_OUTPUT_ROOT
    assert not hasattr(args, "step_distances")
    assert not hasattr(args, "seed_list")
    assert not hasattr(args, "ablation_tag")
    assert not hasattr(args, "n_steps_per_env")
    assert not hasattr(args, "log_interval")
    assert (
        args.evaluation_interval_rollouts
        == DEFAULT_EVALUATION_INTERVAL_ROLLOUTS
    )
    assert not hasattr(args, "max_train_episodes")
    assert args.dry_run is False


def test_show_cli_accepts_compact_figure_options(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "show",
            "--output-file",
            str(tmp_path / "curves"),
            "--dpi",
            "180",
            "--pad-inches",
            "0.05",
            "--no-show",
        ]
    )

    assert args.output_file == tmp_path / "curves"
    assert args.dpi == 180.0
    assert args.pad_inches == 0.05
    assert args.no_show is True


@pytest.mark.parametrize(
    ("command", "removed_option"),
    (
        ("train", "--ablation-tag"),
        ("train", "--step-distances"),
        ("train", "--seed-list"),
        ("train", "--n-steps-per-env"),
        ("train", "--log-interval"),
        ("show", "--metric-source"),
        ("show", "--step-distances"),
    ),
)
def test_step_distance_ablation_rejects_removed_configuration_options(
    command: str,
    removed_option: str,
) -> None:
    argv = [command, removed_option, "value"]
    if command == "train":
        argv.extend(["--reference-curve-dir", "."])
    with pytest.raises(SystemExit):
        _ = build_arg_parser().parse_args(argv)


def test_resolve_metric_source_prefers_existing_best_artifact(tmp_path: Path) -> None:
    best_dir = tmp_path / "best_rollouts"
    _ = _write_trajectory_metrics_json(
        best_dir,
        stop_error_m=1.0,
        time_error_s=2.0,
        total_energy_kj=3.0,
    )
    manifest = {
        "runs": [
            {
                "status": "completed",
                "best_eval_output_dir": str(best_dir),
            }
        ]
    }
    assert resolve_metric_source(manifest) == "best"
    assert resolve_metric_source({"runs": []}) == "final"


def test_resolve_run_matrix_expands_distances_and_seeds() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["train", "--reference-curve-dir", "."])

    run_entries = resolve_step_distance_run_matrix(args)

    assert len(run_entries) == len(DEFAULT_STEP_DISTANCES) * len(DEFAULT_SEEDS)
    first_distance_entries = run_entries[: len(DEFAULT_SEEDS)]
    assert [entry.step_distance for entry in first_distance_entries] == [
        DEFAULT_STEP_DISTANCES[0]
    ] * len(DEFAULT_SEEDS)
    assert [entry.seed for entry in first_distance_entries] == list(DEFAULT_SEEDS)
    assert run_entries[0].experiment_tag == "ds10p0__r01"
    assert not hasattr(run_entries[0].train_args, "max_train_episodes")


def test_run_matrix_keeps_basic_reward_and_fixed_hyperparameters() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "train",
            "--reference-curve-dir",
            ".",
            "--num-envs",
            "2",
            "--rollout-steps-per-update",
            "4096",
            "--evaluation-interval-rollouts",
            "7",
        ]
    )

    run_entries = resolve_step_distance_run_matrix(args)
    first = run_entries[0]
    other_distance = run_entries[len(DEFAULT_SEEDS)]

    assert first.training_run_spec.reward_preset.name == FIXED_REWARD_PRESET
    assert first.training_run_spec.enable_monitor is True
    assert first.training_run_spec.enable_auto_analysis is False
    assert first.training_run_spec.curriculum_profile == "dspdl"
    assert first.training_run_spec.reference_curve_dir == "."
    assert first.training_run_spec.enable_best_evaluation_artifacts is False
    assert first.training_run_spec.evaluation_interval_rollouts == 7
    assert first.training_run_spec.run_metadata["evaluation_interval_rollouts"] == 7
    assert first.training_run_spec.evaluation_deterministic is True

    assert (
        first.training_run_spec.reward_discount
        == other_distance.training_run_spec.reward_discount
    )
    assert (
        first.training_run_spec.schedule_time_s
        == other_distance.training_run_spec.schedule_time_s
    )
    assert first.training_run_spec.num_envs == other_distance.training_run_spec.num_envs
    assert (
        first.training_run_spec.rollout_steps_per_update
        == other_distance.training_run_spec.rollout_steps_per_update
    )
    assert (
        first.training_run_spec.step_distance
        != other_distance.training_run_spec.step_distance
    )
    assert first.seed == other_distance.seed


def test_build_and_load_manifest_records_step_distance_runs(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "train",
            "--reference-curve-dir",
            ".",
            "--output-root",
            str(tmp_path),
        ]
    )
    run_entries = resolve_step_distance_run_matrix(args)
    statuses = {(DEFAULT_STEP_DISTANCES[0], 0): {"status": "completed"}}

    manifest = build_step_distance_manifest(args, run_entries, statuses=statuses)
    manifest_path = tmp_path / STEP_DISTANCE_MANIFEST_FILENAME
    _ = manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_step_distance_manifest(str(tmp_path))

    assert loaded["reward_preset"] == FIXED_REWARD_PRESET
    assert loaded["step_distances"] == list(DEFAULT_STEP_DISTANCES)
    assert loaded["seed_list"] == list(DEFAULT_SEEDS)
    assert loaded["runs"][0]["status"] == "completed"
    assert loaded["runs"][1]["status"] == "pending"
    assert loaded["runs"][0]["reward_diagnostics_path"].endswith(
        "reward_diagnostics.npz"
    )
    assert loaded["runs"][0]["best_eval_output_dir"] is None
    assert loaded["runs"][0]["final_metrics_path"].endswith(
        "final_trajectory_metrics.json"
    )
    assert loaded["training"]["enable_best_evaluation_artifacts"] is False


def test_manifest_uses_total_timesteps_training_budget(
    tmp_path: Path,
) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "train",
            "--reference-curve-dir",
            ".",
            "--output-root",
            str(tmp_path),
        ]
    )
    run_entries = resolve_step_distance_run_matrix(args)

    manifest = build_step_distance_manifest(args, run_entries)

    assert manifest["training"]["total_timesteps"] == int(args.total_timesteps)
    assert "stop_mode" not in manifest
    assert "max_train_episodes" not in manifest


def test_train_step_distance_run_delegates_to_train_single_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "train",
            "--reference-curve-dir",
            ".",
            "--output-root",
            str(tmp_path),
        ]
    )
    entry = resolve_step_distance_run_matrix(args)[0]

    calls: list[tuple[object, object]] = []

    def _fake_train_single_experiment(train_args: object, *, spec: _RunSpec) -> object:
        calls.append((train_args, spec))
        final_output_dir = Path(spec.final_output_dir)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        _write_reward_diagnostics_npz(
            final_output_dir,
            rewards=[1.5, 2.5],
            lengths=[10.0, 20.0],
        )
        return spec

    monkeypatch.setattr(
        step_distance_ablation,
        "train_single_experiment",
        _fake_train_single_experiment,
    )

    _ = train_step_distance_run(entry)

    assert len(calls) == 1
    assert calls[0][0] is entry.train_args
    assert calls[0][1] is entry.training_run_spec
    reward_diagnostics_path = Path(entry.reward_diagnostics_path)
    assert reward_diagnostics_path.is_file()


def test_train_command_runs_final_policy_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(step_distance_ablation, "DEFAULT_STEP_DISTANCES", (50.0,))
    monkeypatch.setattr(step_distance_ablation, "DEFAULT_SEEDS", (42,))
    calls: list[object] = []

    def _fake_train_single_experiment(train_args: object, *, spec: _RunSpec) -> object:
        calls.append((train_args, spec))
        final_output_dir = Path(spec.final_output_dir)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        _write_reward_diagnostics_npz(
            final_output_dir,
            rewards=[1.5],
            lengths=[10.0],
        )
        return spec

    monkeypatch.setattr(
        step_distance_ablation,
        "train_single_experiment",
        _fake_train_single_experiment,
    )
    evaluated: list[object] = []

    def _fake_evaluate_final_training_run(spec: object) -> tuple[str, str]:
        evaluated.append(spec)
        final_output_dir = Path(cast(_RunSpec, spec).final_output_dir)
        metrics_path = final_output_dir / "final_trajectory_metrics.json"
        _ = metrics_path.write_text("{}", encoding="utf-8")
        return str(final_output_dir / "final_trajectory.npz"), str(metrics_path)

    monkeypatch.setattr(
        step_distance_ablation,
        "evaluate_final_training_run",
        _fake_evaluate_final_training_run,
    )

    exit_code = step_distance_ablation.main(
        [
            "train",
            "--reference-curve-dir",
            ".",
            "--output-root",
            str(tmp_path),
            "--total-timesteps",
            "1",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert len(evaluated) == 1
    manifest = load_step_distance_manifest(str(tmp_path))
    assert manifest["runs"][0]["status"] == "completed"
    final_output_dir = Path(str(manifest["runs"][0]["final_output_dir"]))
    assert (final_output_dir / "final_trajectory_metrics.json").exists()


def test_build_curve_aggregates_groups_monitor_data_by_step_distance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "step_distance"
    run_50_r1 = root / "run_50_r1" / "final"
    run_50_r2 = root / "run_50_r2" / "final"
    run_100_r1 = root / "run_100_r1" / "final"

    metrics_50_r1 = _write_reward_diagnostics_npz(
        run_50_r1,
        rewards=[1.0, 3.0, 5.0],
        lengths=[10.0, 9.0, 8.0],
    )
    metrics_50_r2 = _write_reward_diagnostics_npz(
        run_50_r2,
        rewards=[2.0, 4.0],
        lengths=[12.0, 10.0],
    )
    metrics_100_r1 = _write_reward_diagnostics_npz(
        run_100_r1,
        rewards=[10.0, 12.0],
        lengths=[5.0, 4.0],
    )
    _ = _write_run_metadata(run_50_r1)
    _ = _write_run_metadata(run_50_r2)
    _ = _write_run_metadata(run_100_r1)

    manifest = {
        "step_distances": [50.0, 100.0],
        "runs": [
            {
                "step_distance": 50.0,
                "repeat_index": 0,
                "seed": 42,
                "final_output_dir": str(run_50_r1),
                "reward_diagnostics_path": str(metrics_50_r1),
                "status": "completed",
            },
            {
                "step_distance": 50.0,
                "repeat_index": 1,
                "seed": 43,
                "final_output_dir": str(run_50_r2),
                "reward_diagnostics_path": str(metrics_50_r2),
                "status": "completed",
            },
            {
                "step_distance": 100.0,
                "repeat_index": 0,
                "seed": 42,
                "final_output_dir": str(run_100_r1),
                "reward_diagnostics_path": str(metrics_100_r1),
                "status": "completed",
            },
        ],
    }

    aggregates, warnings = build_curve_aggregates(manifest)

    assert warnings == []
    assert [aggregate.step_distance for aggregate in aggregates] == [50.0, 100.0]
    aggregate_50 = aggregates[0]
    assert aggregate_50.valid_run_count == 2
    np.testing.assert_allclose(aggregate_50.reference_steps, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        aggregate_50.mean_reward,
        [1.5, 3.5, 5.0],
    )
    np.testing.assert_allclose(
        aggregate_50.mean_length,
        [11.0, 9.5, 8.0],
    )


def test_build_metric_aggregates_groups_best_trajectory_metrics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "step_distance"
    run_50_r1_best = root / "run_50_r1" / "best_rollouts"
    run_50_r2_best = root / "run_50_r2" / "best_rollouts"
    run_100_output = root / "run_100_r1"
    run_100_best = run_100_output / "best_rollouts"
    run_100_metadata = _write_run_metadata(
        run_100_output,
        {"best_eval_output_dir": str(run_100_best)},
    )
    _ = _write_trajectory_metrics_json(
        run_50_r1_best,
        stop_error_m=1.0,
        time_error_s=3.0,
        total_energy_kj=10.0,
        comfort_tav=0.10,
        comfort_rms=0.20,
        comfort_er_pct=0.30,
    )
    _ = _write_trajectory_metrics_json(
        run_50_r2_best,
        stop_error_m=3.0,
        time_error_s=7.0,
        total_energy_kj=14.0,
        comfort_tav=0.30,
        comfort_rms=0.60,
        comfort_er_pct=0.70,
    )
    _ = _write_trajectory_metrics_json(
        run_100_best,
        stop_error_m=5.0,
        time_error_s=11.0,
        total_energy_kj=20.0,
    )

    manifest = {
        "step_distances": [50.0, 100.0],
        "runs": [
            {
                "step_distance": 50.0,
                "repeat_index": 0,
                "best_eval_output_dir": str(run_50_r1_best),
                "status": "completed",
            },
            {
                "step_distance": 50.0,
                "repeat_index": 1,
                "best_eval_output_dir": str(run_50_r2_best),
                "status": "completed",
            },
            {
                "step_distance": 100.0,
                "repeat_index": 0,
                "run_metadata_path": str(run_100_metadata),
                "status": "completed",
            },
        ],
    }

    aggregates, warnings = build_metric_aggregates(manifest, metric_source="best")

    assert warnings == []
    assert [aggregate.step_distance for aggregate in aggregates] == [50.0, 100.0]
    aggregate_50 = aggregates[0]
    assert aggregate_50.valid_run_count == 2
    assert aggregate_50.metric_means["stop_error_m"] == pytest.approx(2.0)
    assert aggregate_50.metric_vars["stop_error_m"] == pytest.approx(1.0)
    assert aggregate_50.metric_means["time_error_s"] == pytest.approx(5.0)
    assert aggregate_50.metric_vars["time_error_s"] == pytest.approx(4.0)
    assert aggregate_50.metric_means["total_energy_kj"] == pytest.approx(12.0)
    assert aggregate_50.metric_vars["total_energy_kj"] == pytest.approx(4.0)
    aggregate_100 = aggregates[1]
    assert aggregate_100.valid_run_count == 1
    assert aggregate_100.metric_means["stop_error_m"] == pytest.approx(5.0)


def test_best_metric_aggregates_do_not_fall_back_to_final_metrics(
    tmp_path: Path,
) -> None:
    final_dir = tmp_path / "run_50_r1" / "final"
    _ = _write_trajectory_metrics_json(
        final_dir,
        stop_error_m=1.0,
        time_error_s=3.0,
        total_energy_kj=10.0,
        file_name="final_trajectory_metrics.json",
    )

    manifest = {
        "step_distances": [50.0],
        "runs": [
            {
                "step_distance": 50.0,
                "repeat_index": 0,
                "final_output_dir": str(final_dir),
                "status": "completed",
            },
        ],
    }

    aggregates, warnings = build_metric_aggregates(manifest, metric_source="best")

    assert aggregates == []
    assert any("missing best_eval_output_dir" in warning for warning in warnings)
    assert any("No valid best trajectory metrics" in warning for warning in warnings)


def test_print_metric_table_shows_mean_plus_std_cells(
    capsys: pytest.CaptureFixture[str],
) -> None:
    aggregate = step_distance_ablation.StepDistanceMetricAggregate(
        step_distance=50.0,
        valid_run_count=2,
        metric_means={
            "stop_error_m": 2.0,
            "time_error_s": 5.0,
            "total_energy_kj": 12.0,
            "comfort_tav": 0.2,
            "comfort_rms": 0.4,
            "comfort_er_pct": 0.5,
        },
        metric_vars={
            "stop_error_m": 1.0,
            "time_error_s": 4.0,
            "total_energy_kj": 4.0,
            "comfort_tav": 0.01,
            "comfort_rms": 0.04,
            "comfort_er_pct": 0.04,
        },
    )

    step_distance_ablation._print_metric_table([aggregate], metric_source="final")

    output = capsys.readouterr().out
    assert "Final trajectory evaluation summary (mean±std):" in output
    assert "stop_error_m" in output
    assert "time_error_s" in output
    assert "stop_error_m_mean" not in output
    assert "stop_error_m_var" not in output
    assert "2.000000±1.000000" in output
    assert "5.000000±2.000000" in output


def test_save_compact_figure_appends_png_and_uses_tight_bbox(tmp_path: Path) -> None:
    class FakeFigure:
        def __init__(self) -> None:
            self.savefig_calls: list[tuple[Path, dict[str, object]]] = []

        def savefig(self, output_path: Path, **kwargs: object) -> None:
            self.savefig_calls.append((output_path, kwargs))
            _ = output_path.write_bytes(b"fake image")

    figure = FakeFigure()

    output_path = save_compact_figure(
        cast(Figure, cast(object, figure)),
        tmp_path / "nested" / "step_distance",
        dpi=180.0,
        pad_inches=0.05,
    )

    assert output_path == tmp_path / "nested" / "step_distance.png"
    assert output_path.is_file()
    assert figure.savefig_calls == [
        (
            output_path,
            {
                "dpi": 180.0,
                "bbox_inches": "tight",
                "pad_inches": 0.05,
            },
        )
    ]


def test_show_command_saves_compact_figure_without_display(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = StepDistanceCurveAggregate(
        step_distance=50.0,
        reference_steps=np.asarray([10.0], dtype=np.float64),
        mean_reward=np.asarray([1.0], dtype=np.float64),
        std_reward=np.asarray([0.1], dtype=np.float64),
        mean_length=np.asarray([12.0], dtype=np.float64),
        std_length=np.asarray([0.3], dtype=np.float64),
        valid_run_count=1,
        reward_diagnostics_paths=("metrics.npz",),
    )
    fake_figure = object()
    save_calls: list[tuple[object, Path, float, float]] = []
    show_calls: list[bool] = []

    def _fake_load_step_distance_manifest(output_root: object) -> dict[str, object]:
        del output_root
        return {"step_distances": [50.0], "runs": []}

    def _fake_build_curve_aggregates(
        manifest: object, step_distances: object = None
    ) -> tuple[list[StepDistanceCurveAggregate], list[str]]:
        del manifest, step_distances
        return ([aggregate], [])

    def _fake_build_metric_aggregates(
        manifest: object, step_distances: object = None, metric_source: object = None
    ) -> tuple[list[object], list[str]]:
        del manifest, step_distances, metric_source
        return ([], [])

    def _fake_plot_curve_aggregates(aggregates: object, *, show: bool = True) -> object:
        del aggregates, show
        return fake_figure

    monkeypatch.setattr(
        step_distance_ablation,
        "load_step_distance_manifest",
        _fake_load_step_distance_manifest,
    )
    monkeypatch.setattr(
        step_distance_ablation,
        "build_curve_aggregates",
        _fake_build_curve_aggregates,
    )
    monkeypatch.setattr(
        step_distance_ablation,
        "build_metric_aggregates",
        _fake_build_metric_aggregates,
    )
    monkeypatch.setattr(
        step_distance_ablation,
        "plot_curve_aggregates",
        _fake_plot_curve_aggregates,
    )

    def _fake_save_compact_figure(
        fig: Figure,
        output_file: Path,
        *,
        dpi: float,
        pad_inches: float,
    ) -> Path:
        save_calls.append((fig, output_file, dpi, pad_inches))
        return output_file.with_suffix(".png")

    monkeypatch.setattr(
        step_distance_ablation,
        "save_compact_figure",
        _fake_save_compact_figure,
    )
    monkeypatch.setattr(
        step_distance_ablation.plt, "show", lambda: show_calls.append(True)
    )

    exit_code = step_distance_ablation.main(
        [
            "show",
            "--output-root",
            str(tmp_path),
            "--output-file",
            str(tmp_path / "compact"),
            "--dpi",
            "180",
            "--pad-inches",
            "0.05",
            "--no-show",
        ]
    )

    assert exit_code == 0
    assert save_calls == [(fake_figure, tmp_path / "compact", 180.0, 0.05)]
    assert show_calls == []


def test_plot_curve_aggregates_uses_step_axis_and_deduped_legend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        step_distance_ablation, "apply_rl_curve_plot_style", lambda: None
    )
    monkeypatch.setattr(step_distance_ablation.plt, "show", lambda: None)

    aggregates = [
        StepDistanceCurveAggregate(
            step_distance=50.0,
            reference_steps=np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
            mean_reward=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            std_reward=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
            mean_length=np.asarray([12.0, 11.0, 10.0], dtype=np.float64),
            std_length=np.asarray([0.3, 0.2, 0.1], dtype=np.float64),
            valid_run_count=2,
            reward_diagnostics_paths=("a", "b"),
        ),
        StepDistanceCurveAggregate(
            step_distance=100.0,
            reference_steps=np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
            mean_reward=np.asarray([0.5, 1.0, 1.5], dtype=np.float64),
            std_reward=np.asarray([0.1, 0.1, 0.1], dtype=np.float64),
            mean_length=np.asarray([14.0, 13.0, 12.0], dtype=np.float64),
            std_length=np.asarray([0.2, 0.2, 0.2], dtype=np.float64),
            valid_run_count=2,
            reward_diagnostics_paths=("c", "d"),
        ),
    ]

    returned_fig = plot_curve_aggregates(aggregates)

    fig = step_distance_ablation.plt.gcf()
    assert returned_fig is fig
    assert len(fig.axes) == 2
    np.testing.assert_allclose(fig.get_size_inches(), [9.2, 3.9])
    assert fig.axes[0].get_xlabel() == "Training steps"
    assert fig.axes[1].get_xlabel() == "Training steps"
    assert fig.axes[0].get_box_aspect() == pytest.approx(3 / 4)
    assert fig.axes[1].get_box_aspect() == pytest.approx(3 / 4)

    assert fig.legends, "Expected a figure-level legend."
    legend_labels = [text.get_text() for text in fig.legends[0].texts]
    assert legend_labels == ["50 m", "100 m"]
    subplot_labels = [
        text.get_text()
        for ax in fig.axes
        for text in ax.texts
        if text.get_text() in {"(a)", "(b)"}
    ]
    assert subplot_labels == ["(a)", "(b)"]
    subplot_label_y_positions = [
        text.get_position()[1]
        for ax in fig.axes
        for text in ax.texts
        if text.get_text() in {"(a)", "(b)"}
    ]
    assert subplot_label_y_positions == pytest.approx([-0.18, -0.18])
    step_distance_ablation.plt.close(fig)
