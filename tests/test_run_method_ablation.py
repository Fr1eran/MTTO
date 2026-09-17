import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import scripts.run_method_ablation as method_ablation
from contracts.ablation import AblationManifest
from contracts.evaluation import EvaluationHistory, EvaluationMetrics
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION, REWARD_NAMES
from utils.ablation import ManifestStore, artifact_paths


def _write_episodes(path: Path, violation_codes: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    codes = np.asarray(violation_codes or [0, 0], dtype=np.int8)
    episode_count = codes.size
    rewards = np.arange(1, episode_count + 1, dtype=np.float64)
    episode_sums = np.zeros((episode_count, len(REWARD_NAMES)), dtype=np.float64)
    episode_sums[:, 0] = rewards
    episode_sums[:, -1] = rewards
    np.savez(
        path,
        schema_version=np.asarray([REWARD_DIAGNOSTICS_SCHEMA_VERSION], dtype=np.int16),
        reward_names=np.asarray(REWARD_NAMES),
        rollout_end_step=np.asarray([episode_count * 10]),
        rollout_transition_count=np.asarray([episode_count * 10]),
        rollout_reward_sum=episode_sums.sum(axis=0, keepdims=True),
        rollout_reward_abs_sum=np.abs(episode_sums).sum(axis=0, keepdims=True),
        rollout_reward_nonzero_count=np.count_nonzero(
            episode_sums, axis=0, keepdims=True
        ),
        rollout_reward_cross_product=np.asarray([episode_sums.T @ episode_sums]),
        episode_end_step=np.arange(1, episode_count + 1, dtype=np.int64) * 10,
        episode_worker_rank=np.zeros(episode_count, dtype=np.int16),
        episode_index=np.arange(episode_count, dtype=np.int64),
        episode_length=np.full(episode_count, 10, dtype=np.int32),
        episode_terminated=codes == 0,
        episode_truncated=codes != 0,
        episode_complete=np.ones(episode_count, dtype=np.bool_),
        episode_violation_code=codes,
        episode_reward_sums=episode_sums,
    )


def _write_evaluations(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history = EvaluationHistory(
        training_steps=np.asarray([100, 200], dtype=np.int64),
        rollout_indices=np.asarray([1, 2], dtype=np.int64),
        total_reward=np.asarray([1.0, 2.0], dtype=np.float64),
        episode_steps=np.asarray([10, 9], dtype=np.int64),
        success=np.asarray([False, True], dtype=np.bool_),
        safe=np.asarray([False, True], dtype=np.bool_),
        feasible=np.asarray([False, True], dtype=np.bool_),
        stop_error_m=np.asarray([4.0, 1.0], dtype=np.float64),
        time_error_s=np.asarray([-12.0, -2.0], dtype=np.float64),
        total_energy_j=np.asarray([10_000.0, 9_000.0], dtype=np.float64),
        comfort_tav=np.asarray([0.4, 0.2], dtype=np.float64),
        completed_training_episodes=np.asarray([1, 2], dtype=np.int64),
        scheduled_completed_training_episodes=np.asarray([1, 2], dtype=np.int64),
        route_completion_ratio=np.asarray([0.5, 1.0], dtype=np.float64),
        safety_violation_positions_m=np.asarray([], dtype=np.float64),
        safety_violation_position_offsets=np.asarray([0, 0, 0], dtype=np.int64),
    )
    np.savez(path, **history.to_npz_mapping())


def _write_fixed_evaluations(
    path: Path,
    *,
    failed_index: int | None = None,
    evaluation_interval: int = 12,
) -> None:
    steps = (
        np.arange(
            evaluation_interval,
            method_ablation.METHOD_TRAINING_ROLLOUTS,
            evaluation_interval,
            dtype=np.int64,
        )
        * 8192
    )
    success = np.ones(steps.size, dtype=np.bool_)
    if failed_index is not None:
        success[failed_index] = False
    history = EvaluationHistory(
        training_steps=steps,
        rollout_indices=np.arange(
            evaluation_interval,
            method_ablation.METHOD_TRAINING_ROLLOUTS,
            evaluation_interval,
            dtype=np.int64,
        ),
        total_reward=np.arange(steps.size, dtype=np.float64),
        episode_steps=np.full(steps.size, 10, dtype=np.int64),
        success=success,
        safe=success.copy(),
        feasible=success.copy(),
        stop_error_m=np.arange(1, steps.size + 1, dtype=np.float64),
        time_error_s=-np.arange(1, steps.size + 1, dtype=np.float64),
        total_energy_j=np.full(steps.size, 9_000.0),
        comfort_tav=np.full(steps.size, 0.2),
        completed_training_episodes=np.arange(steps.size, dtype=np.int64),
        scheduled_completed_training_episodes=np.arange(steps.size, dtype=np.int64),
        route_completion_ratio=np.linspace(0.0, 1.0, steps.size),
        safety_violation_positions_m=np.asarray([], dtype=np.float64),
        safety_violation_position_offsets=np.zeros(steps.size + 1, dtype=np.int64),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **history.to_npz_mapping())


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = EvaluationMetrics(
        success=True,
        precise_arrival=True,
        punctual_arrival=True,
        total_reward=2.0,
        total_time_s=438.0,
        target_time_s=440.0,
        time_error_s=-2.0,
        start_position_m=0.0,
        target_position_m=100.0,
        final_position_m=99.9,
        final_speed_mps=0.0,
        stop_error_m=0.1,
        total_energy_j=9_000.0,
        comfort_tav=0.2,
        comfort_er_pct=0.0,
        comfort_rms=0.1,
        terminated=True,
        truncated=False,
        episode_steps=10,
        min_safety_margin_mps=0.0,
        mean_safety_margin_mps=0.0,
        safety_violation_count=0,
        safe=True,
        feasible=True,
        strict_stop_error_limit_m=0.3,
        strict_time_error_limit_s=10.0,
        selection_comparison_key=(1.0, -9_000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    path.write_text(json.dumps(metrics.to_mapping()), encoding="utf-8")


def _entry(
    tmp_path: Path, method: str, seed: int = 11, repeat_index: int = 0
) -> dict[str, object]:
    final_dir = tmp_path / f"{method}_{seed}_r{repeat_index + 1:02d}" / "final"
    episodes = final_dir / "episodes.npz"
    evaluations = final_dir / "evaluations.npz"
    metrics = final_dir / "metrics.json"
    best_dir = final_dir.parent / "best"
    best_metrics = best_dir / "metrics.json"
    _write_episodes(episodes)
    _write_evaluations(evaluations)
    _write_metrics(metrics)
    _write_metrics(best_metrics)
    (best_dir / "policy.zip").write_bytes(b"policy")
    (best_dir / "trajectory.npz").write_bytes(b"trajectory")
    (final_dir / "policy.zip").write_bytes(b"policy")
    (final_dir / "trajectory.npz").write_bytes(b"trajectory")
    budget = {
        "mode": "environment_steps",
        "target_reached": True,
        "actual_training_timesteps": 3_276_800,
        "derived_total_timesteps": 3_276_800,
        "actual_training_rollouts": 400,
        "training_rollouts": 400,
    }
    metadata = {"training_budget": budget}
    (best_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (final_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return {
        "run_id": f"method__{method}__seed{seed:04d}__r{repeat_index + 1:02d}",
        "variant_id": method,
        "variant": {"name": method},
        "repeat_index": repeat_index,
        "seed": seed,
        "experiment_tag": f"{method}__r{repeat_index + 1:02d}",
        "artifacts": {
            "policy_final": str(final_dir / "policy.zip"),
            "policy_best": str(best_dir / "policy.zip"),
            "metadata": str(final_dir / "metadata.json"),
            "metadata_best": str(best_dir / "metadata.json"),
            "episodes": str(episodes),
            "evaluations": str(evaluations),
            "metrics_final": str(metrics),
            "metrics_best": str(best_metrics),
            "trajectory_final": str(final_dir / "trajectory.npz"),
            "trajectory_best": str(best_dir / "trajectory.npz"),
            "safety_diagnostics": str(final_dir / "safety_diagnostics.npz"),
        },
        "status": "completed",
        "training_budget": budget,
    }


def _manifest(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": method_ablation.MANIFEST_VERSION,
        "matrix_id": "method",
        "matrix_config": {
            "variants": [item.__dict__ for item in method_ablation.METHODS],
            "seeds": list(method_ablation.DEFAULT_SEEDS),
        },
        "training_signature": {},
        "runs": entries,
    }


def test_matrix_maps_methods_to_expected_training_modes() -> None:
    args = method_ablation.build_arg_parser().parse_args(["train"])
    runs = method_ablation.resolve_run_matrix(args)

    assert len(runs) == len(method_ablation.METHODS) * len(
        method_ablation.DEFAULT_SEEDS
    )
    assert args.num_envs == 8
    assert args.output_root == "output/paper_experiment/02_method_ablation"
    assert not hasattr(args, "vec_env_type")
    assert not hasattr(args, "rollout_steps_per_update")
    assert not hasattr(args, "training_episodes")
    assert args.device == "cpu"
    assert all(run.spec.evaluation_interval_rollouts == 12 for run in runs)
    assert all(run.spec.budget_mode == "environment_steps" for run in runs)
    assert all(run.spec.training_rollouts == 400 for run in runs)
    assert all(run.spec.training_episodes is None for run in runs)
    assert all(run.spec.total_timesteps == 3_276_800 for run in runs)
    assert all(run.spec.rollout_steps_per_update == 8192 for run in runs)
    assert all(run.spec.num_envs == 8 for run in runs)
    first_by_method = {run.method.name: run for run in runs if run.repeat_index == 0}
    assert first_by_method["ppo"].train_args.reward_preset == "basic"
    assert first_by_method["ppo"].method.label == "PPO"
    assert first_by_method["ppo_safety"].train_args.reward_preset == "basic_safety"
    assert first_by_method["ppo_safety"].method.label == "PPO+Safety"
    assert (
        first_by_method["ppo_punctuality"].train_args.reward_preset
        == "basic_punctuality"
    )
    assert first_by_method["ppo_punctuality"].method.label == "PPO+Punctuality"
    assert (
        first_by_method["ppo_pirs"].train_args.reward_preset
        == "basic_safety_punctuality"
    )
    assert first_by_method["ppo_pirs"].method.label == "PPO+PIRS"


@pytest.mark.parametrize("vec_env_type", ("dummy", "subproc"))
def test_train_cli_rejects_vec_env_type(vec_env_type: str) -> None:
    parser = method_ablation.build_arg_parser()
    with pytest.raises(SystemExit):
        _ = parser.parse_args(
            [
                "train",
                "--vec-env-type",
                vec_env_type,
            ]
        )


def test_train_cli_rejects_removed_step_evaluation_interval() -> None:
    with pytest.raises(SystemExit):
        _ = method_ablation.build_arg_parser().parse_args(
            [
                "train",
                "--eval-interval-steps",
                "100000",
            ]
        )


@pytest.mark.parametrize(
    "option",
    [
        "--dpi",
        "--output-file",
        "--safety-output-file",
        "--success-output-file",
        "--selection-output-file",
    ],
)
def test_show_cli_rejects_removed_figure_options(option: str) -> None:
    with pytest.raises(SystemExit):
        _ = method_ablation.build_arg_parser().parse_args(["show", option, "value"])


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--training-episodes", "5000"),
        ("--rollout-steps-per-update", "4096"),
    ),
)
def test_train_cli_rejects_fixed_protocol_overrides(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        _ = method_ablation.build_arg_parser().parse_args(["train", flag, value])


def test_train_cli_accepts_parallelism_and_evaluation_interval() -> None:
    args = method_ablation.build_arg_parser().parse_args(
        [
            "train",
            "--num-envs",
            "4",
            "--evaluation-interval-rollouts",
            "10",
        ]
    )
    runs = method_ablation.resolve_run_matrix(args)
    assert all(run.spec.num_envs == 4 for run in runs)
    assert all(run.spec.evaluation_interval_rollouts == 10 for run in runs)

    for invalid_interval in (0, 400):
        invalid_args = method_ablation.build_arg_parser().parse_args(
            ["train", "--evaluation-interval-rollouts", str(invalid_interval)]
        )
        with pytest.raises(SystemExit, match="must be in"):
            method_ablation.run_train(invalid_args)


def test_manifest_round_trip_and_compatibility(tmp_path: Path) -> None:
    args = method_ablation.build_arg_parser().parse_args(
        ["train", "--output-root", str(tmp_path)]
    )
    runs = method_ablation.resolve_run_matrix(args)
    payload = method_ablation.build_manifest(
        args,
        runs,
        {runs[0].run_id: {"status": "completed"}},
    )

    method_ablation._manifest_store(str(tmp_path)).save_atomic(payload)
    loaded = method_ablation.load_manifest(str(tmp_path))
    method_ablation._validate_manifest_compatibility(loaded, args)

    assert loaded.output_root == str(tmp_path)
    assert payload.output_root == "."
    assert loaded["schema_version"] == 2
    assert loaded["matrix_id"] == "method"
    assert loaded["matrix_config"]["protocol_version"] == 11
    variants = loaded["matrix_config"]["variants"]
    assert variants[0]["label"] == "PPO"
    assert variants[1]["label"] == "PPO+Safety"
    assert variants[2]["label"] == "PPO+Punctuality"
    assert variants[3]["label"] == "PPO+PIRS"
    assert loaded["training_signature"]["budget_mode"] == "environment_steps"
    assert loaded["training_signature"]["training_rollouts"] == 400
    assert loaded["training_signature"]["training_steps"] == 3_276_800
    assert loaded["runs"][0]["status"] == "completed"  # type: ignore[index]


def test_curve_and_final_aggregates_use_canonical_artifacts(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, method.name) for method in method_ablation.METHODS]
    curves, curve_warnings = method_ablation.build_curve_aggregates(_manifest(entries))
    finals, final_warnings = method_ablation.build_final_aggregates(_manifest(entries))

    assert curve_warnings == []
    assert final_warnings == []
    assert [aggregate.variant_id for aggregate in curves] == [
        method.name for method in method_ablation.METHODS
    ]
    assert curves[0].x[-1] == 200
    assert curves[0].means["ep_reward"][0] == pytest.approx(1.0)
    assert [aggregate.variant_id for aggregate in finals] == [
        method.name for method in method_ablation.METHODS
    ]
    assert finals[0].means["stop_error_m"] == 0.1
    assert finals[0].means["abs_time_error_s"] == 2.0


def test_curve_aggregation_uses_periodic_evaluations_and_warmup(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, method_ablation.METHODS[0].name)]

    curves, warnings = method_ablation.build_curve_aggregates(_manifest(entries))

    assert warnings == []
    np.testing.assert_array_equal(curves[0].x, [100.0, 200.0])
    np.testing.assert_allclose(curves[0].means["ep_reward"], [1.0, 1.5])
    np.testing.assert_allclose(curves[0].means["success_rate"], [0.0, 0.5])


def test_periodic_evaluation_aggregation_keeps_failures_and_aligns_axes(
    tmp_path: Path,
) -> None:
    entries = []
    for method in method_ablation.METHODS:
        for index, seed in enumerate(method_ablation.DEFAULT_SEEDS):
            entry = _entry(tmp_path, method.name, seed)
            _write_fixed_evaluations(
                Path(entry["artifacts"]["evaluations"]),  # type: ignore[index]
                failed_index=(0 if index == 0 else None),
            )
            entries.append(entry)

    aggregates, warnings = method_ablation.build_curve_aggregates(_manifest(entries))

    assert warnings == []
    assert len(aggregates) == 4
    first = aggregates[0]
    assert first.x.size == 33
    assert first.means["success_rate"][0] == pytest.approx(0.8)
    assert first.means["stop_error_m"][0] == pytest.approx(1.0)
    assert np.all(first.metrics["success_rate"].count == 5)
    figure = method_ablation._plot_learning_curves(aggregates)
    assert figure is not None
    assert all(
        axis.get_xlim() == pytest.approx((0.0, 3_276_800.0)) for axis in figure.axes[:4]
    )
    assert [line.get_linestyle() for line in figure.axes[0].lines[:4]] == [
        "-",
        "--",
        ":",
        "-.",
    ]
    assert [line.get_marker() for line in figure.axes[0].lines[:4]] == [
        "o",
        "s",
        "^",
        "D",
    ]
    assert [line.get_color() for line in figure.axes[0].lines[:4]] == [
        "#7F8C8D",
        "#0072B2",
        "#CC79A7",
        "#ED7D31",
    ]
    assert len(figure.axes[0].lines) == 5
    assert len(figure.axes[1].lines) == 5
    figure.clear()


@pytest.mark.parametrize("evaluation_interval", [12, 10])
def test_run_show_does_not_generate_selection_file(
    tmp_path: Path,
    evaluation_interval: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = method_ablation.build_arg_parser().parse_args(
        [
            "train",
            "--output-root",
            str(tmp_path),
            "--evaluation-interval-rollouts",
            str(evaluation_interval),
        ]
    )
    runs = method_ablation.resolve_run_matrix(args)
    base_manifest = method_ablation.build_manifest(args, runs)

    entries = []
    for run in runs:
        budget = replace(
            run.training_spec.run_metadata.training_budget,  # type: ignore[arg-type]
            target_reached=True,
            actual_training_timesteps=run.training_spec.run_metadata.training_budget.derived_total_timesteps,  # type: ignore[union-attr]
            actual_training_rollouts=run.training_spec.run_metadata.training_budget.training_rollouts,  # type: ignore[union-attr]
        )
        metadata = run.training_spec.run_metadata.with_updates(training_budget=budget)
        metadata_json = json.dumps(metadata.to_mapping())

        for art_path in (
            run.artifacts.policy_final,
            run.artifacts.policy_best,
            run.artifacts.trajectory_final,
            run.artifacts.trajectory_best,
        ):
            if art_path:
                art_path.parent.mkdir(parents=True, exist_ok=True)
                art_path.write_bytes(b"dummy")

        run.artifacts.metadata.parent.mkdir(parents=True, exist_ok=True)
        run.artifacts.metadata.write_text(metadata_json, encoding="utf-8")
        if run.artifacts.metadata_best:
            run.artifacts.metadata_best.parent.mkdir(parents=True, exist_ok=True)
            run.artifacts.metadata_best.write_text(metadata_json, encoding="utf-8")

        _write_episodes(run.artifacts.episodes)
        _write_fixed_evaluations(
            run.artifacts.evaluations, evaluation_interval=evaluation_interval
        )
        _write_metrics(run.artifacts.metrics_final)
        if run.artifacts.metrics_best:
            _write_metrics(run.artifacts.metrics_best)

        entries.append(
            {
                "run_id": run.run_id,
                "variant_id": run.variant.id,
                "variant": dict(run.variant.manifest),
                "repeat_index": run.repeat_index,
                "seed": run.seed,
                "experiment_tag": run.experiment_tag,
                "artifacts": artifact_paths(run.artifacts, relative_to=tmp_path),
                "status": "completed",
                "training_budget": budget,
            }
        )

    manifest_payload = base_manifest.to_mapping()
    manifest_payload["runs"] = entries
    manifest = AblationManifest.from_mapping(
        {"artifact_type": AblationManifest.ARTIFACT_TYPE, **manifest_payload}
    )
    store = ManifestStore(
        tmp_path,
        matrix_id="method",
        filename=method_ablation.METHOD_ABLATION_MANIFEST_FILENAME,
    )
    store.save_atomic(manifest)

    fig_dir = tmp_path / "dry_run_figures"
    table_dir = tmp_path / "dry_run_tables"
    summary_file = tmp_path / "dry_run_summary" / "summary.json"

    dry_run_result = method_ablation.main(
        [
            "show",
            "--output-root",
            str(tmp_path),
            "--figure-output-dir",
            str(fig_dir),
            "--table-output-dir",
            str(table_dir),
            "--summary-output-file",
            str(summary_file),
            "--dry-run",
            "--no-show",
        ]
    )
    assert dry_run_result == 0
    expected_points = len(range(evaluation_interval, 400, evaluation_interval))
    assert f"{expected_points} evaluation points" in capsys.readouterr().out
    assert not fig_dir.exists()
    assert not table_dir.exists()
    assert not summary_file.parent.exists()

    result = method_ablation.main(
        [
            "show",
            "--output-root",
            str(tmp_path),
            "--table-output-dir",
            str(table_dir),
            "--summary-output-file",
            str(summary_file),
            "--no-show",
        ]
    )
    assert result == 0
    assert not (tmp_path / "selected_policy.json").exists()
    assert list(tmp_path.glob("**/selected_policy*.json")) == []
    assert table_dir.is_dir()
    perf_table = (table_dir / "method_performance_table.md").read_text(encoding="utf-8")
    assert "严格可行率" in perf_table
    assert "TAV (m/s²)" in perf_table
    assert "累计加速度变化量" in perf_table
    assert summary_file.is_file()
    summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "feasible_summary" in summary_data


def test_plot_method_training_curves_includes_late_stage_inset() -> None:
    steps = np.arange(12, 400, 12) * method_ablation.DEFAULT_ROLLOUT_STEPS_PER_UPDATE
    dummy_data = {
        m.id: {
            "speed_violation_rate_means": np.zeros(len(steps)),
            "speed_violation_rate_stds": np.zeros(len(steps)),
            "arrival_ratio_means": np.ones(len(steps)),
            "arrival_ratio_stds": np.zeros(len(steps)),
        }
        for m in method_ablation.METHODS
    }
    fig = method_ablation._plot_method_training_curves(dummy_data, steps)
    assert fig is not None
    assert len(fig.axes[0].child_axes) == 1
    inset = fig.axes[0].child_axes[0]
    assert inset.get_xlim() == pytest.approx(
        (
            200 * method_ablation.DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
            396 * method_ablation.DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        )
    )
    assert inset.get_ylim() == pytest.approx((0.0, 5.0))
    method_ablation.plt.close(fig)


def test_analysis_rejects_old_method_protocol(tmp_path: Path) -> None:
    args = method_ablation.build_arg_parser().parse_args(
        ["train", "--output-root", str(tmp_path)]
    )
    manifest = method_ablation.build_manifest(
        args, method_ablation.resolve_run_matrix(args)
    )
    legacy = replace(
        manifest,
        matrix_config={**manifest.matrix_config, "protocol_version": 1},
    )
    with pytest.raises(ValueError, match="obsolete protocol"):
        method_ablation.validate_method_ablation_manifest(legacy)
