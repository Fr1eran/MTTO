import argparse
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_reward_ablation as reward_ablation
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION, REWARD_NAMES


def _train_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return reward_ablation.build_arg_parser().parse_args(
        ["train", "--output-root", str(tmp_path), *extra]
    )


def _write_reward_diagnostics(path: Path, reward_offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    episode_sums = np.zeros((2, len(REWARD_NAMES)), dtype=np.float64)
    episode_sums[:, 0] = np.asarray([1.0, 2.0]) + reward_offset
    episode_sums[:, -1] = np.asarray([1.0, 2.0]) + reward_offset
    np.savez(
        path,
        schema_version=np.asarray([REWARD_DIAGNOSTICS_SCHEMA_VERSION], dtype=np.int16),
        reward_names=np.asarray(REWARD_NAMES),
        rollout_end_step=np.asarray([200]),
        rollout_transition_count=np.asarray([19]),
        rollout_reward_sum=episode_sums.sum(axis=0, keepdims=True),
        rollout_reward_abs_sum=np.abs(episode_sums).sum(axis=0, keepdims=True),
        rollout_reward_nonzero_count=np.count_nonzero(
            episode_sums, axis=0, keepdims=True
        ),
        rollout_reward_cross_product=np.asarray([episode_sums.T @ episode_sums]),
        episode_end_step=np.asarray([100, 200]),
        episode_worker_rank=np.asarray([0, 0]),
        episode_index=np.asarray([0, 1]),
        episode_length=np.asarray([10, 9]),
        episode_terminated=np.asarray([True, True]),
        episode_truncated=np.asarray([False, False]),
        episode_complete=np.asarray([True, True]),
        episode_reward_sums=episode_sums,
    )


def _write_periodic(path: Path, success: tuple[bool, bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        training_steps=np.asarray([100, 200]),
        total_reward=np.asarray([1.0, 2.0]),
        episode_steps=np.asarray([10, 9]),
        success=np.asarray(success),
        stop_error_m=np.asarray([4.0, 1.0]),
        abs_time_error_s=np.asarray([12.0, 2.0]),
        total_energy_kj=np.asarray([10.0, 9.0]),
        comfort_tav=np.asarray([0.4, 0.2]),
        safety_violation_positions_m=np.asarray([500.0, 5_100.0]),
        safety_violation_position_offsets=np.asarray([0, 1, 2]),
    )


def _write_final(path: Path, *, success: bool, time_error_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(
            {
                "success": success,
                "stop_error_m": 1.0,
                "time_error_s": time_error_s,
                "total_energy_kj": 9.0,
                "comfort_tav": 0.2,
            }
        ),
        encoding="utf-8",
    )


def test_default_matrix_uses_all_profiles_and_configured_seeds(tmp_path: Path) -> None:
    args = _train_args(tmp_path)
    runs = reward_ablation.resolve_run_matrix(args)

    assert args.evaluation_interval_rollouts == 12
    assert args.num_envs == 8
    assert args.vec_env_type == "dummy"
    assert args.rollout_steps_per_update == 8192
    assert args.device == "cpu"
    assert all(run.spec.evaluation_interval_rollouts == 12 for run in runs)
    assert len(runs) == len(reward_ablation.REWARD_ABLATIONS) * len(
        reward_ablation.DEFAULT_SEEDS
    )
    assert {run.ablation.preset for run in runs} == {
        "basic",
        "basic_safety",
    }
    assert {run.seed for run in runs} == set(reward_ablation.DEFAULT_SEEDS)
    assert all(run.train_args.curriculum_profile == "none" for run in runs)
    assert all(run.train_args.reference_curve_dir is None for run in runs)
    assert all(run.spec.enable_monitor for run in runs)
    assert all(run.spec.evaluation_history_path for run in runs)
    assert all(run.spec.reward_config is run.spec.reward_preset.config for run in runs)
    assert runs[0].train_args.experiment_tag == "r01"
    assert Path(runs[0].spec.output_dir).name == "465p0_30p0__basic__r01"


def test_train_cli_rejects_removed_step_evaluation_interval(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _ = _train_args(tmp_path, "--eval-interval-steps", "100000")


def test_profile_subset_preserves_order_and_removes_duplicates(tmp_path: Path) -> None:
    args = _train_args(
        tmp_path,
        "--reward-presets",
        "basic_safety",
        "basic",
        "basic_safety",
    )

    selected = reward_ablation.resolve_reward_ablation_specs(args.reward_presets)
    runs = reward_ablation.resolve_run_matrix(args)

    assert [item.preset for item in selected] == ["basic_safety", "basic"]
    assert [
        run.ablation.preset
        for run in runs[:: len(reward_ablation.DEFAULT_SEEDS)]
    ] == [
        "basic_safety",
        "basic",
    ]


def test_manifest_round_trip_and_training_compatibility(tmp_path: Path) -> None:
    args = _train_args(tmp_path)
    runs = reward_ablation.resolve_run_matrix(args)
    statuses = {runs[0].key: {"status": "completed"}}
    payload = reward_ablation.build_manifest(args, runs, statuses)

    reward_ablation._write_manifest(str(tmp_path), payload)
    loaded = reward_ablation.load_manifest(str(tmp_path))
    reward_ablation._validate_manifest_compatibility(loaded, args)

    assert loaded == payload
    assert not (tmp_path / "reward_ablation_manifest.json.tmp").exists()
    changed_args = _train_args(tmp_path, "--total-timesteps", "123")
    try:
        reward_ablation._validate_manifest_compatibility(loaded, changed_args)
    except ValueError as exc:
        assert "different training settings" in str(exc)
    else:
        raise AssertionError("incompatible manifest was accepted")


def test_legacy_vector_settings_require_explicit_cli_values(tmp_path: Path) -> None:
    legacy_args = _train_args(
        tmp_path,
        "--num-envs",
        "1",
        "--vec-env-type",
        "subproc",
    )
    runs = reward_ablation.resolve_run_matrix(legacy_args)
    payload = reward_ablation.build_manifest(legacy_args, runs)

    reward_ablation._validate_manifest_compatibility(payload, legacy_args)
    with pytest.raises(ValueError, match="different training settings"):
        reward_ablation._validate_manifest_compatibility(
            payload, _train_args(tmp_path)
        )


def test_completed_run_is_skipped_only_when_all_artifacts_exist(
    tmp_path: Path, monkeypatch
) -> None:
    args = _train_args(tmp_path, "--reward-presets", "basic")
    selected_runs = reward_ablation.resolve_run_matrix(args)
    all_runs = reward_ablation.resolve_run_matrix(
        args,
        requested_presets=[item.preset for item in reward_ablation.REWARD_ABLATIONS],
    )
    first = selected_runs[0]
    _write_reward_diagnostics(Path(first.reward_diagnostics_path))
    _write_periodic(Path(first.evaluation_history_path), (False, True))
    _write_final(Path(first.final_metrics_path), success=True, time_error_s=-2.0)
    reward_ablation._write_manifest(
        str(tmp_path),
        reward_ablation.build_manifest(
            args,
            all_runs,
            {first.key: {"status": "completed"}},
        ),
    )
    trained: list[int] = []

    def fake_train(_args, *, spec):
        trained.append(int(spec.seed))
        return spec

    monkeypatch.setattr(reward_ablation, "train_single_experiment", fake_train)
    monkeypatch.setattr(
        reward_ablation,
        "evaluate_final_training_run",
        lambda spec: (
            None,
            str(Path(spec.final_output_dir) / "final_trajectory_metrics.json"),
        ),
    )

    assert reward_ablation.run_train(args) == 0
    assert trained == list(reward_ablation.DEFAULT_SEEDS[1:])


def test_failed_run_is_recorded_and_remaining_runs_continue(
    tmp_path: Path, monkeypatch
) -> None:
    args = _train_args(tmp_path, "--reward-presets", "basic")
    calls: list[int] = []

    def fake_train(_args, *, spec):
        calls.append(int(spec.seed))
        if len(calls) == 1:
            raise RuntimeError("synthetic failure")
        return spec

    monkeypatch.setattr(reward_ablation, "train_single_experiment", fake_train)
    monkeypatch.setattr(
        reward_ablation,
        "evaluate_final_training_run",
        lambda spec: (
            None,
            str(Path(spec.final_output_dir) / "final_trajectory_metrics.json"),
        ),
    )

    assert reward_ablation.run_train(args) == 1
    assert calls == list(reward_ablation.DEFAULT_SEEDS)
    manifest = reward_ablation.load_manifest(str(tmp_path))
    basic_runs = [
        entry for entry in manifest["runs"] if entry["reward_preset"] == "basic"
    ]
    assert basic_runs[0]["status"] == "failed"
    assert basic_runs[0]["error_message"] == "synthetic failure"
    assert all(entry["status"] == "completed" for entry in basic_runs[1:])


def test_curve_final_and_safety_aggregates_use_completed_runs(
    tmp_path: Path,
) -> None:
    entries: list[dict[str, object]] = []
    for index, seed in enumerate((11, 131)):
        final_dir = tmp_path / f"run_{seed}" / "final"
        reward_path = final_dir / "reward_diagnostics.npz"
        periodic_path = final_dir / reward_ablation.EVALUATION_HISTORY_FILENAME
        final_path = final_dir / reward_ablation.FINAL_TRAJECTORY_METRICS_FILENAME
        _write_reward_diagnostics(reward_path, reward_offset=float(index))
        _write_periodic(periodic_path, (False, index == 1))
        _write_final(
            final_path,
            success=index == 1,
            time_error_s=-2.0 - index,
        )
        entries.append(
            {
                "reward_preset": "basic",
                "seed": seed,
                "status": "completed",
                "reward_diagnostics_path": str(reward_path),
                "evaluation_history_path": str(periodic_path),
                "final_metrics_path": str(final_path),
            }
        )
    manifest = {"runs": entries}
    selected = reward_ablation.resolve_reward_ablation_specs(["basic"])

    curves, curve_warnings = reward_ablation.build_curve_aggregates(manifest, selected)
    finals, final_warnings = reward_ablation.build_final_aggregates(manifest, selected)

    assert curve_warnings == []
    assert final_warnings == []
    assert curves[0].valid_run_count == 2
    np.testing.assert_allclose(curves[0].means["ep_reward"], [1.5, 2.5])
    np.testing.assert_allclose(curves[0].means["success"], [0.0, 0.5])
    assert finals[0].success_rate == 0.5
    assert finals[0].means["abs_time_error_s"] == 2.5

    learning_figure = reward_ablation._plot_learning_curves(curves)
    safety_figure = reward_ablation._plot_safety_boxplot(manifest, selected)
    assert learning_figure is not None
    assert len(learning_figure.axes) == 6
    assert safety_figure is not None
    reward_ablation.plt.close(learning_figure)
    reward_ablation.plt.close(safety_figure)


def test_train_dry_run_does_not_create_manifest(tmp_path: Path) -> None:
    args = _train_args(
        tmp_path,
        "--reward-presets",
        "basic_safety",
        "--dry-run",
    )

    assert reward_ablation.run_train(args) == 0
    assert not (tmp_path / reward_ablation.REWARD_ABLATION_MANIFEST_FILENAME).exists()


def test_corrupt_run_is_warned_and_valid_run_still_aggregates(tmp_path: Path) -> None:
    valid_dir = tmp_path / "valid" / "final"
    valid_reward = valid_dir / "reward_diagnostics.npz"
    valid_periodic = valid_dir / reward_ablation.EVALUATION_HISTORY_FILENAME
    valid_final = valid_dir / reward_ablation.FINAL_TRAJECTORY_METRICS_FILENAME
    _write_reward_diagnostics(valid_reward)
    _write_periodic(valid_periodic, (False, True))
    _write_final(valid_final, success=True, time_error_s=2.0)
    manifest = {
        "runs": [
            {
                "reward_preset": "basic",
                "status": "completed",
                "reward_diagnostics_path": str(valid_reward),
                "evaluation_history_path": str(valid_periodic),
                "final_metrics_path": str(valid_final),
            },
            {
                "reward_preset": "basic",
                "status": "completed",
                "reward_diagnostics_path": str(tmp_path / "missing_reward.npz"),
                "evaluation_history_path": str(tmp_path / "missing_eval.npz"),
                "final_metrics_path": str(tmp_path / "missing_final.json"),
            },
        ]
    }
    selected = reward_ablation.resolve_reward_ablation_specs(["basic"])

    curves, curve_warnings = reward_ablation.build_curve_aggregates(manifest, selected)
    finals, final_warnings = reward_ablation.build_final_aggregates(manifest, selected)

    assert len(curves) == len(finals) == 1
    assert curves[0].valid_run_count == finals[0].valid_run_count == 1
    assert len(curve_warnings) == len(final_warnings) == 1


def test_show_dry_run_does_not_build_figures(tmp_path: Path, monkeypatch) -> None:
    manifest = {
        "manifest_version": reward_ablation.MANIFEST_VERSION,
        "runs": [],
    }
    reward_ablation._write_manifest(str(tmp_path), manifest)
    args = reward_ablation.build_arg_parser().parse_args(
        ["show", "--output-root", str(tmp_path), "--dry-run"]
    )
    monkeypatch.setattr(
        reward_ablation,
        "_plot_learning_curves",
        lambda _aggregates: (_ for _ in ()).throw(AssertionError("unexpected plot")),
    )

    assert reward_ablation.run_show(args) == 0
