import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import scripts.run_step_distance_ablation as step_distance_ablation
from contracts.ablation import AblationManifest
from contracts.evaluation import EvaluationHistory, EvaluationMetrics
from rl.experiment_utils import train_single_experiment
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION, REWARD_NAMES


def _write_episodes(path: Path, rewards: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_values = np.asarray(rewards, dtype=np.float64)
    episode_count = reward_values.size
    episode_sums = np.zeros((episode_count, len(REWARD_NAMES)), dtype=np.float64)
    episode_sums[:, 0] = reward_values
    episode_sums[:, -1] = reward_values
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
        episode_terminated=np.ones(episode_count, dtype=np.bool_),
        episode_truncated=np.zeros(episode_count, dtype=np.bool_),
        episode_complete=np.ones(episode_count, dtype=np.bool_),
        episode_violation_code=np.zeros(episode_count, dtype=np.int8),
        episode_reward_sums=episode_sums,
    )


def _write_metrics(path: Path, *, value: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    precise_arrival = value <= 0.3
    punctual_arrival = precise_arrival and abs(2.0 * value) < 10.0
    metrics = EvaluationMetrics(
        success=True,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
        total_reward=2.0,
        total_time_s=440.0 - 2.0 * value,
        target_time_s=440.0,
        time_error_s=-2.0 * value,
        start_position_m=0.0,
        target_position_m=100.0,
        final_position_m=100.0 - value,
        final_speed_mps=0.0,
        stop_error_m=value,
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
        feasible=punctual_arrival,
        strict_stop_error_limit_m=0.3,
        strict_time_error_limit_s=10.0,
        selection_comparison_key=(
            (1.0, -9_000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            if punctual_arrival
            else (0.0, 1.0, 0.0, -value, 0.0, -2.0 * value, -9_000.0)
        ),
    )
    path.write_text(json.dumps(metrics.to_mapping()), encoding="utf-8")


def _write_evaluations(path: Path, returns: list[float]) -> None:
    values = np.asarray(returns, dtype=np.float64)
    count = values.size
    scheduled = np.arange(1, count + 1, dtype=np.int64) * 100
    history = EvaluationHistory(
        training_steps=scheduled * 10,
        rollout_indices=np.arange(1, count + 1, dtype=np.int64),
        total_reward=values,
        episode_steps=np.full(count, 10, dtype=np.int64),
        success=values >= 4.0,
        safe=np.ones(count, dtype=np.bool_),
        feasible=values >= 4.0,
        stop_error_m=np.zeros(count, dtype=np.float64),
        time_error_s=np.zeros(count, dtype=np.float64),
        total_energy_j=np.full(count, 9_000.0),
        comfort_tav=np.full(count, 0.2),
        completed_training_episodes=scheduled + 3,
        scheduled_completed_training_episodes=scheduled,
        route_completion_ratio=np.clip(values / 10.0, 0.0, 1.0),
        safety_violation_positions_m=np.empty(0, dtype=np.float64),
        safety_violation_position_offsets=np.zeros(count + 1, dtype=np.int64),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **history.to_npz_mapping())


def _entry(
    tmp_path: Path,
    distance: float,
    repeat_index: int,
    rewards: list[float],
) -> dict[str, object]:
    final_dir = tmp_path / f"run_{distance:g}_{repeat_index}" / "final"
    episodes = final_dir / "episodes.npz"
    metrics = final_dir / "metrics.json"
    best_dir = final_dir.parent / "best"
    best_metrics = best_dir / "metrics.json"
    _write_episodes(episodes, rewards)
    _write_evaluations(final_dir / "evaluations.npz", rewards)
    _write_metrics(metrics, value=float(repeat_index + 1))
    _write_metrics(best_metrics, value=float(repeat_index + 1))
    (best_dir / "policy.zip").write_bytes(b"policy")
    (best_dir / "trajectory.npz").write_bytes(b"trajectory")
    token = step_distance_ablation.format_float_token(distance)
    return {
        "run_id": (
            f"step_distance__ds{token}__seed{repeat_index + 1:04d}__"
            f"r{repeat_index + 1:02d}"
        ),
        "variant_id": token,
        "variant": {"step_distance": distance},
        "step_distance": distance,
        "repeat_index": repeat_index,
        "seed": repeat_index + 1,
        "experiment_tag": f"ds{token}__r{repeat_index + 1:02d}",
        "artifacts": {
            "policy_final": str(final_dir / "policy.zip"),
            "policy_best": str(best_dir / "policy.zip"),
            "metadata": str(final_dir / "metadata.json"),
            "metadata_best": str(best_dir / "metadata.json"),
            "episodes": str(episodes),
            "evaluations": str(final_dir / "evaluations.npz"),
            "metrics_final": str(metrics),
            "metrics_best": str(best_metrics),
            "trajectory_best": str(best_dir / "trajectory.npz"),
            "safety_diagnostics": str(final_dir / "safety_diagnostics.npz"),
        },
        "status": "completed",
    }


def _manifest(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": step_distance_ablation.MANIFEST_VERSION,
        "matrix_id": "step_distance",
        "matrix_config": {
            "step_distances": [50.0, 100.0],
            "seeds": [1, 2],
            "reward_preset": step_distance_ablation.FIXED_REWARD_PRESET,
        },
        "training_signature": {},
        "runs": entries,
    }


def test_run_matrix_expands_default_distances_and_seeds() -> None:
    args = step_distance_ablation.build_arg_parser().parse_args(["train"])
    runs = step_distance_ablation.resolve_step_distance_run_matrix(args)

    assert args.output_root == "output/paper_experiment/01_step_distance"

    run_entries = step_distance_ablation.resolve_step_distance_run_matrix(args)
    first = run_entries[0]
    other_distance = run_entries[len(step_distance_ablation.DEFAULT_SEEDS)]

    assert (
        first.training_run_spec.reward_preset.name
        == step_distance_ablation.FIXED_REWARD_PRESET
    )
    assert first.training_run_spec.enable_monitor is True
    assert first.training_run_spec.enable_auto_analysis is False
    assert first.training_run_spec.enable_best_evaluation_artifacts is True
    assert first.training_run_spec.evaluation_interval_rollouts == 12
    assert first.training_run_spec.evaluation_interval_episodes is None
    assert first.training_run_spec.run_metadata["evaluation_interval_rollouts"] == 12
    assert "evaluation_interval_episodes" not in first.training_run_spec.run_metadata
    assert first.training_run_spec.evaluation_deterministic is True
    assert first.training_run_spec.budget_mode == "environment_steps"
    assert first.training_run_spec.training_rollouts == 400
    assert first.training_run_spec.total_timesteps == 3_276_800

    assert (
        first.training_run_spec.reward_discount
        == other_distance.training_run_spec.reward_discount
    )
    assert (
        first.training_run_spec.schedule_time_s
        == other_distance.training_run_spec.schedule_time_s
    )
    assert runs[0].run_id == "step_distance__ds10p0__seed0011__r01"
    assert runs[0].evaluation_history_path.endswith("evaluations.npz")
    assert runs[0].training_run_spec.evaluation_history_path.endswith("evaluations.npz")


@pytest.mark.parametrize(
    "option",
    [
        "--training-episodes",
        "--evaluation-interval-episodes",
        "--rollout-steps-per-update",
        "--evaluation-interval-rollouts",
    ],
)
def test_train_cli_rejects_fixed_protocol_overrides(option: str) -> None:
    parser = step_distance_ablation.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                option,
                "100",
            ]
        )


@pytest.mark.parametrize("option", ["--dpi", "--pad-inches", "--output-file"])
def test_show_cli_rejects_removed_export_options(option: str) -> None:
    with pytest.raises(SystemExit):
        _ = step_distance_ablation.build_arg_parser().parse_args(
            ["show", option, "300"]
        )


def test_manifest_round_trip_uses_one_new_schema_file(tmp_path: Path) -> None:
    args = step_distance_ablation.build_arg_parser().parse_args(
        [
            "train",
            "--output-root",
            str(tmp_path),
        ]
    )
    runs = step_distance_ablation.resolve_step_distance_run_matrix(args)
    payload = step_distance_ablation.build_step_distance_manifest(args, runs)
    store = step_distance_ablation._manifest_store(str(tmp_path))
    store.save_atomic(payload)

    assert str(store.path) == str(tmp_path / "manifest.json")
    loaded = step_distance_ablation.load_step_distance_manifest(str(tmp_path))
    step_distance_ablation._validate_manifest_compatibility(loaded, args)
    assert loaded["schema_version"] == 2
    assert loaded["matrix_id"] == "step_distance"
    assert loaded["matrix_config"]["protocol_version"] == 12
    assert loaded["training_signature"]["budget_mode"] == "environment_steps"
    assert loaded["training_signature"]["training_rollouts"] == 400
    assert loaded["training_signature"]["training_steps"] == 3_276_800
    assert loaded["training_signature"]["rollout_steps_per_update"] == 8192
    assert loaded["training_signature"]["evaluation_interval_rollouts"] == 12
    assert "training_episodes" not in loaded["training_signature"]
    assert (
        loaded["matrix_config"]["reward_config"]["punctuality_potential_scale"] == 5.0
    )
    assert loaded["runs"][0]["artifacts"]["policy_final"].endswith(  # type: ignore[index]
        "policy.zip"
    )


def test_curve_aggregation_aligns_by_completed_episode_number(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 50.0, 0, [1.0, 3.0, 5.0]),
        _entry(tmp_path, 50.0, 1, [2.0, 4.0]),
        _entry(tmp_path, 100.0, 0, [10.0, 12.0]),
    ]

    aggregates, warnings = step_distance_ablation.build_curve_aggregates(
        _manifest(entries), episode_smoothing_window=1
    )

    assert warnings == []
    assert [aggregate.variant_id for aggregate in aggregates] == ["50p0", "100p0"]
    aggregate = aggregates[0]
    np.testing.assert_allclose(aggregate.x, [1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(
        aggregate.means["route_completion_ratio"], [0.15, 0.35, 0.50]
    )
    np.testing.assert_allclose(aggregate.means["feasible_rate"], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(aggregate.metrics["feasible_rate"].count, [2, 2, 1])


def test_curve_aggregation_keeps_warmup_episodes(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, 50.0, 0, [1.0, 3.0, 8.0])]

    aggregates, warnings = step_distance_ablation.build_curve_aggregates(
        _manifest(entries), episode_smoothing_window=100
    )

    assert warnings == []
    np.testing.assert_allclose(aggregates[0].x, [1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(
        aggregates[0].means["route_completion_ratio"], [0.1, 0.2, 0.4]
    )
    figure = step_distance_ablation.plot_curve_aggregates(aggregates, show=False)
    assert figure is not None
    assert all(
        axis.get_xlim() == pytest.approx((0.0, 3_276_800.0)) for axis in figure.axes
    )
    assert figure.axes[0].get_ylim() == pytest.approx((0.0, 1.0))
    assert figure.axes[1].get_ylim() == pytest.approx((-0.03, 1.03))
    assert len(figure.axes[0].collections) == 1
    assert len(figure.axes[1].collections) == 1
    assert figure.axes[0].lines[0].get_color() == "#0072B2"
    assert figure.axes[0].lines[0].get_linestyle() == ":"
    assert figure.axes[0].lines[0].get_marker() == "^"
    figure.clear()


def test_metric_aggregation_uses_sample_std_and_explicit_best_artifacts(
    tmp_path: Path,
) -> None:
    first = _entry(tmp_path, 50.0, 0, [1.0])
    second = _entry(tmp_path, 50.0, 1, [2.0])
    best_first = tmp_path / "best_first.json"
    best_second = tmp_path / "best_second.json"
    _write_metrics(best_first, value=0.2)
    _write_metrics(best_second, value=1.8)
    first["artifacts"]["metrics_best"] = str(best_first)  # type: ignore[index]
    second["artifacts"]["metrics_best"] = str(best_second)  # type: ignore[index]

    manifest = _manifest([first, second])
    manifest["matrix_config"]["step_distances"] = [50.0]  # type: ignore[index]
    assert step_distance_ablation.resolve_metric_source(manifest) == "best"
    aggregates, warnings = step_distance_ablation.build_metric_aggregates(
        manifest, metric_source="best"
    )

    assert warnings == []
    assert aggregates[0].means["stop_error_m"] == pytest.approx(1.0)
    assert aggregates[0].stds["stop_error_m"] == pytest.approx(np.sqrt(1.28))

    manifest_obj = AblationManifest.from_mapping(
        {"artifact_type": AblationManifest.ARTIFACT_TYPE, **manifest}
    )
    table_md, summary, rec, status = (
        step_distance_ablation.build_step_distance_summary_and_table(manifest_obj)
    )
    assert "严格可行率" in table_md
    assert "严格可行数/5" not in table_md
    assert "50.0% (1/2)" in table_md
    assert "TAV (m/s²)" in table_md
    assert "舒适度 (m/s³)" not in table_md
    assert "累计加速度变化量" in table_md
    assert summary["variants"]["50p0"]["feasible_rate"] == pytest.approx(0.5)
    assert summary["variants"]["50p0"]["feasible_count"] == 1
    assert summary["variants"]["50p0"]["metrics"]["stop_error_m"][
        "mean"
    ] == pytest.approx(1.0)
    assert summary["variants"]["50p0"]["metrics"]["stop_error_m"][
        "std"
    ] == pytest.approx(np.sqrt(1.28))
    assert rec == 50.0
    assert "推荐步长: 50 m" in status

    best_infeasible = tmp_path / "best_infeasible.json"
    _write_metrics(best_infeasible, value=1.2)
    first["artifacts"]["metrics_best"] = str(best_infeasible)  # type: ignore[index]
    manifest_obj = AblationManifest.from_mapping(
        {"artifact_type": AblationManifest.ARTIFACT_TYPE, **manifest}
    )
    _, infeasible_summary, infeasible_rec, _ = (
        step_distance_ablation.build_step_distance_summary_and_table(manifest_obj)
    )
    variant = infeasible_summary["variants"]["50p0"]
    assert variant["mean_feasible_energy_kwh"] is None
    assert variant["mean_feasible_comfort"] is None
    assert infeasible_rec is None
    encoded = json.dumps(infeasible_summary, allow_nan=False)
    assert json.loads(encoded)["variants"]["50p0"]["mean_feasible_energy_kwh"] is None


def test_train_command_records_failure_and_stops_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(step_distance_ablation, "DEFAULT_STEP_DISTANCES", (50.0,))
    monkeypatch.setattr(step_distance_ablation, "DEFAULT_SEEDS", (11, 131))
    calls: list[int] = []
    fail_seed11 = True
    failed_spec: object = None

    def fake_train(_args: object, *, spec: object) -> object:
        nonlocal fail_seed11, failed_spec
        calls.append(int(spec.seed))  # type: ignore[attr-defined]
        if fail_seed11 and spec.seed == 11:  # type: ignore[attr-defined]
            failed_spec = spec
            metadata_path = Path(spec.run_metadata_path)  # type: ignore[attr-defined]
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(spec.run_metadata.to_mapping()),
                encoding="utf-8",  # type: ignore[attr-defined]
            )
            fail_seed11 = False
            raise RuntimeError("synthetic failure")

        # Successful training
        final_dir = Path(spec.final_output_dir)  # type: ignore[attr-defined]
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "policy.zip").write_bytes(b"policy")
        (Path(spec.reward_diagnostics_path)).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        (Path(spec.reward_diagnostics_path)).write_bytes(b"episodes")  # type: ignore[attr-defined]
        if spec.evaluation_history_path:  # type: ignore[attr-defined]
            (Path(spec.evaluation_history_path)).parent.mkdir(  # type: ignore[attr-defined]
                parents=True, exist_ok=True
            )
            (Path(spec.evaluation_history_path)).write_bytes(b"evaluations")  # type: ignore[attr-defined]

        budget = spec.run_metadata.training_budget  # type: ignore[attr-defined]
        if budget is not None:
            updated_budget = replace(
                budget,
                target_reached=True,
                actual_training_timesteps=budget.derived_total_timesteps,
                actual_training_rollouts=budget.training_rollouts,
            )
            spec = replace(  # type: ignore[attr-defined]
                spec,
                run_metadata=spec.run_metadata.with_updates(  # type: ignore[attr-defined]
                    training_budget=updated_budget
                ),
            )
        if getattr(spec, "best_eval_output_dir", None):
            best_dir = Path(spec.best_eval_output_dir)  # type: ignore[attr-defined]
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "policy.zip").write_bytes(b"policy")
            (best_dir / "metadata.json").write_text(
                json.dumps(spec.run_metadata.to_mapping()),  # type: ignore[attr-defined]
                encoding="utf-8",
            )
        Path(spec.run_metadata_path).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        Path(spec.run_metadata_path).write_text(  # type: ignore[attr-defined]
            json.dumps(spec.run_metadata.to_mapping()),
            encoding="utf-8",  # type: ignore[attr-defined]
        )
        return spec

    def fake_eval(spec: object) -> tuple[str, str]:
        final_dir = Path(spec.final_output_dir)  # type: ignore[attr-defined]
        final_dir.mkdir(parents=True, exist_ok=True)
        traj_path = final_dir / "trajectory.npz"
        metrics_path = final_dir / "metrics.json"
        traj_path.write_bytes(b"trajectory")
        _write_metrics(metrics_path)
        if getattr(spec, "best_eval_output_dir", None):
            best_dir = Path(spec.best_eval_output_dir)  # type: ignore[attr-defined]
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "trajectory.npz").write_bytes(b"trajectory")
            _write_metrics(best_dir / "metrics.json")
        return (str(traj_path), str(metrics_path))

    monkeypatch.setattr(step_distance_ablation, "train_single_experiment", fake_train)
    monkeypatch.setattr(
        step_distance_ablation, "evaluate_final_training_run", fake_eval
    )

    # 1. First run fails on seed 11 after writing metadata.json
    assert (
        step_distance_ablation.main(
            [
                "train",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert calls == [11]
    manifest = step_distance_ablation.load_step_distance_manifest(str(tmp_path))
    assert manifest["runs"][0]["status"] == "failed"  # type: ignore[index]
    assert manifest["runs"][1]["status"] == "pending"  # type: ignore[index]
    failed_meta_path = Path(failed_spec.run_metadata_path)  # type: ignore[attr-defined]
    assert failed_meta_path.is_file()

    # 2. Direct training train_single_experiment still refuses to overwrite
    with pytest.raises(
        FileExistsError, match="Target directory already contains training artifacts"
    ):
        train_single_experiment(None, spec=failed_spec)  # type: ignore[arg-type]

    # 3. Running with --resume:
    # Seed 11's directory is cleaned before re-running, then both runs succeed.
    assert (
        step_distance_ablation.main(
            [
                "train",
                "--output-root",
                str(tmp_path),
                "--resume",
            ]
        )
        == 0
    )
    assert calls == [11, 11, 131]
    manifest = step_distance_ablation.load_step_distance_manifest(str(tmp_path))
    assert manifest["runs"][0]["status"] == "completed"  # type: ignore[index]
    assert manifest["runs"][1]["status"] == "completed"  # type: ignore[index]

    # 4. Running with --resume again: all completed runs are skipped!
    assert (
        step_distance_ablation.main(
            [
                "train",
                "--output-root",
                str(tmp_path),
                "--resume",
            ]
        )
        == 0
    )
    assert calls == [11, 11, 131]

    # 5. Running with --force-new: archives manifest, cleans directories,
    # and reruns all runs.
    assert (
        step_distance_ablation.main(
            [
                "train",
                "--output-root",
                str(tmp_path),
                "--force-new",
            ]
        )
        == 0
    )
    assert calls == [11, 11, 131, 11, 131]
    archives = list(tmp_path.glob("manifest.json.bak.*"))
    assert len(archives) == 1
    manifest = step_distance_ablation.load_step_distance_manifest(str(tmp_path))
    assert manifest["runs"][0]["status"] == "completed"  # type: ignore[index]
    assert manifest["runs"][1]["status"] == "completed"  # type: ignore[index]
