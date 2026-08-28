import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_method_ablation as method_ablation
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION, REWARD_NAMES


def _write_reward_diagnostics(
    path: Path,
    *,
    violation_codes: list[int] | None = None,
    schema_version: int = REWARD_DIAGNOSTICS_SCHEMA_VERSION,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    codes = np.asarray(violation_codes or [0, 0], dtype=np.int8)
    episode_count = codes.size
    episode_sums = np.zeros((episode_count, len(REWARD_NAMES)), dtype=np.float64)
    episode_sums[:, 0] = np.arange(1, episode_count + 1, dtype=np.float64)
    episode_sums[:, -1] = episode_sums[:, 0]
    fields: dict[str, np.ndarray] = {
        "schema_version": np.asarray([schema_version], dtype=np.int16),
        "reward_names": np.asarray(REWARD_NAMES),
        "rollout_end_step": np.asarray([episode_count * 10]),
        "rollout_transition_count": np.asarray([episode_count * 10]),
        "rollout_reward_sum": episode_sums.sum(axis=0, keepdims=True),
        "rollout_reward_abs_sum": np.abs(episode_sums).sum(axis=0, keepdims=True),
        "rollout_reward_nonzero_count": np.count_nonzero(
            episode_sums, axis=0, keepdims=True
        ),
        "rollout_reward_cross_product": np.asarray([episode_sums.T @ episode_sums]),
        "episode_end_step": np.arange(1, episode_count + 1, dtype=np.int64) * 10,
        "episode_worker_rank": np.zeros(episode_count, dtype=np.int16),
        "episode_index": np.arange(episode_count, dtype=np.int64),
        "episode_length": np.full(episode_count, 10, dtype=np.int32),
        "episode_terminated": codes == 0,
        "episode_truncated": codes != 0,
        "episode_complete": np.ones(episode_count, dtype=np.bool_),
        "episode_reward_sums": episode_sums,
    }
    if schema_version == REWARD_DIAGNOSTICS_SCHEMA_VERSION:
        fields["episode_violation_code"] = codes
    np.savez(path, **fields)


def _write_periodic(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        training_steps=np.asarray([100, 200]),
        total_reward=np.asarray([1.0, 2.0]),
        episode_steps=np.asarray([10, 9]),
        success=np.asarray([False, True]),
        stop_error_m=np.asarray([4.0, 1.0]),
        abs_time_error_s=np.asarray([12.0, 2.0]),
        total_energy_kj=np.asarray([10.0, 9.0]),
        comfort_tav=np.asarray([0.4, 0.2]),
        safety_violation_positions_m=np.asarray([500.0, 5_100.0]),
        safety_violation_position_offsets=np.asarray([0, 1, 2]),
        completed_training_episodes=np.asarray([1, 2]),
    )


def test_run_matrix_maps_the_four_methods_to_expected_configurations() -> None:
    args = method_ablation.build_arg_parser().parse_args(
        ["train", "--reference-curve-dir", "."]
    )
    runs = method_ablation.resolve_run_matrix(args)

    assert len(runs) == len(method_ablation.METHODS) * len(
        method_ablation.DEFAULT_SEEDS
    )
    assert args.evaluation_interval_rollouts == 12
    assert args.training_episodes == 7_000
    assert args.num_envs == 8
    assert not hasattr(args, "vec_env_type")
    assert args.rollout_steps_per_update == 8192
    assert args.device == "cpu"
    assert all(run.spec.evaluation_interval_rollouts == 12 for run in runs)
    assert all(run.spec.training_episodes == 7_000 for run in runs)
    first_by_method = {run.method.name: run for run in runs if run.repeat_index == 0}
    assert first_by_method["ppo"].train_args.reward_preset == "basic"
    assert first_by_method["ppo"].train_args.curriculum_profile == "none"
    assert (
        first_by_method["ppo_pbrs"].train_args.reward_preset == "basic_safety"
    )
    assert (
        first_by_method["ppo_dspdl"].train_args.curriculum_profile
        == "dspdl_completion"
    )
    assert first_by_method["ppo_dspdl"].train_args.reference_curve_dir == "."
    assert (
        first_by_method["ppo_pbrs_dspdl"].train_args.reward_preset
        == "basic_safety"
    )
    assert (
        first_by_method["ppo_pbrs_dspdl"].train_args.curriculum_profile
        == "dspdl_completion"
    )


@pytest.mark.parametrize(
    "removed_option",
    ("--total-timesteps", "--target-completed-episodes"),
)
def test_train_cli_rejects_removed_timestep_budget_options(removed_option: str) -> None:
    with pytest.raises(SystemExit):
        _ = method_ablation.build_arg_parser().parse_args(
            ["train", "--reference-curve-dir", ".", removed_option, "1"]
        )


@pytest.mark.parametrize("vec_env_type", ("dummy", "subproc"))
def test_train_cli_rejects_vec_env_type(vec_env_type: str) -> None:
    parser = method_ablation.build_arg_parser()
    with pytest.raises(SystemExit):
        _ = parser.parse_args(
            [
                "train",
                "--reference-curve-dir",
                ".",
                "--vec-env-type",
                vec_env_type,
            ]
        )


def test_train_cli_rejects_removed_step_evaluation_interval() -> None:
    with pytest.raises(SystemExit):
        _ = method_ablation.build_arg_parser().parse_args(
            [
                "train",
                "--reference-curve-dir",
                ".",
                "--eval-interval-steps",
                "100000",
            ]
        )


def test_train_rejects_incompatible_existing_manifest(tmp_path: Path) -> None:
    parser = method_ablation.build_arg_parser()
    args = parser.parse_args(
        [
            "train",
            "--reference-curve-dir",
            ".",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    runs = method_ablation.resolve_run_matrix(args)
    method_ablation._write_manifest(
        str(tmp_path), method_ablation.build_manifest(args, runs)
    )
    changed_args = parser.parse_args(
        [
            "train",
            "--reference-curve-dir",
            ".",
            "--output-root",
            str(tmp_path),
            "--num-envs",
            "1",
            "--dry-run",
        ]
    )

    with pytest.raises(ValueError, match="different training settings"):
        _ = method_ablation.run_train(changed_args)


def test_resume_keeps_only_completed_runs_with_final_artifacts(tmp_path: Path) -> None:
    parser = method_ablation.build_arg_parser()
    args = parser.parse_args(
        [
            "train",
            "--reference-curve-dir",
            ".",
            "--output-root",
            str(tmp_path),
            "--resume",
        ]
    )
    runs = method_ablation.resolve_run_matrix(args)
    completed = runs[0]
    Path(completed.spec.final_model_save_path).parent.mkdir(
        parents=True, exist_ok=True
    )
    Path(completed.spec.final_model_save_path).touch()
    Path(completed.final_metrics_path).write_text("{}", encoding="utf-8")
    method_ablation._write_manifest(
        str(tmp_path),
        method_ablation.build_manifest(
            args,
            runs,
            {(completed.method.name, completed.repeat_index): {"status": "completed"}},
        ),
    )

    statuses = method_ablation._completed_statuses_for_resume(args, runs)

    assert statuses == {
        (completed.method.name, completed.repeat_index): {
            "status": "completed",
            "final_metrics_path": completed.final_metrics_path,
        }
    }


def test_curve_and_final_aggregates_read_fixed_start_artifacts(tmp_path: Path) -> None:
    runs: list[dict[str, object]] = []
    for method in method_ablation.METHODS:
        final_dir = tmp_path / method.name / "final"
        reward_path = final_dir / "reward_diagnostics.npz"
        periodic_path = final_dir / "evaluation_history.npz"
        final_path = final_dir / "final_trajectory_metrics.json"
        _write_reward_diagnostics(reward_path)
        _write_periodic(periodic_path)
        _ = final_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "stop_error_m": 1.0,
                    "time_error_s": -2.0,
                    "total_energy_kj": 9.0,
                    "comfort_tav": 0.2,
                }
            ),
            encoding="utf-8",
        )
        runs.append(
            {
                "method": method.name,
                "status": "completed",
                "reward_diagnostics_path": str(reward_path),
                "evaluation_history_path": str(periodic_path),
                "final_metrics_path": str(final_path),
            }
        )

    manifest = {"runs": runs}
    curves, curve_warnings = method_ablation.build_curve_aggregates(
        manifest, episode_smoothing_window=1
    )
    finals, final_warnings = method_ablation.build_final_aggregates(manifest)

    assert curve_warnings == []
    assert final_warnings == []
    assert [aggregate.method.name for aggregate in curves] == [
        method.name for method in method_ablation.METHODS
    ]
    assert np.isnan(curves[0].means["stop_error_m"][0])
    assert curves[0].means["stop_error_m"][-1] == 1.0
    assert np.isnan(curves[0].means["abs_time_error_s"][0])
    np.testing.assert_allclose(curves[0].episode_numbers, [1.0, 2.0])
    np.testing.assert_allclose(curves[0].evaluation_steps, [100.0, 200.0])
    assert [aggregate.method.name for aggregate in finals] == [
        method.name for method in method_ablation.METHODS
    ]
    assert finals[0].success_rate == 1.0
    assert finals[0].means["time_error_s"] == -2.0


def test_safety_learning_process_uses_completed_episode_axis(tmp_path: Path) -> None:
    reward_path = tmp_path / "final" / "reward_diagnostics.npz"
    _write_reward_diagnostics(reward_path, violation_codes=[3, 0])
    manifest = {
        "runs": [
            {
                "method": "ppo",
                "status": "completed",
                "reward_diagnostics_path": str(reward_path),
            }
        ]
    }
    aggregates, warnings = method_ablation.build_safety_learning_aggregates(manifest)
    figure = method_ablation._plot_safety_learning_process(aggregates)
    assert warnings == []
    assert figure is not None
    assert len(figure.axes) == 1
    assert not figure._suptitle
    assert figure.axes[0].get_ylim() == pytest.approx((-0.03, 1.03))
    assert figure.axes[0].get_ylabel() == "Training safety violation rate"
    method_ablation.plt.close(figure)


def test_safety_learning_process_skips_legacy_reward_diagnostics(
    tmp_path: Path,
) -> None:
    final_dir = tmp_path / "final"
    reward_path = final_dir / "reward_diagnostics.npz"
    _write_reward_diagnostics(reward_path, schema_version=2)
    aggregates, warnings = method_ablation.build_safety_learning_aggregates(
        {
            "runs": [
                {
                    "method": "ppo",
                    "status": "completed",
                    "reward_diagnostics_path": str(reward_path),
                }
            ]
        }
    )

    assert aggregates == []
    assert len(warnings) == 1
    assert "rerun training" in warnings[0]


def test_safety_learning_process_bins_training_violations_by_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(method_ablation, "SAFETY_EPISODE_BIN_WIDTH", 2)
    first_path = tmp_path / "first" / "reward_diagnostics.npz"
    second_path = tmp_path / "second" / "reward_diagnostics.npz"
    _write_reward_diagnostics(first_path, violation_codes=[2, 0, 3, 1, 0])
    _write_reward_diagnostics(second_path, violation_codes=[0, 3, 4, 2])
    aggregates, warnings = method_ablation.build_safety_learning_aggregates(
        {
            "runs": [
                {
                    "method": "ppo",
                    "status": "completed",
                    "reward_diagnostics_path": str(first_path),
                },
                {
                    "method": "ppo",
                    "status": "completed",
                    "reward_diagnostics_path": str(second_path),
                },
            ]
        }
    )

    assert warnings == []
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    np.testing.assert_allclose(
        aggregate.episode_bin_edges, [0.0, 2.0, 4.0, 6.0]
    )
    np.testing.assert_allclose(aggregate.episode_bin_centers, [1.0, 3.0, 5.0])
    np.testing.assert_allclose(aggregate.mean_violation_rate, [0.5, 0.5, 0.0])
    np.testing.assert_allclose(
        aggregate.std_violation_rate,
        [0.0, 0.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_array_equal(aggregate.valid_seed_counts, [2, 2, 1])

    figure = method_ablation._plot_safety_learning_process(aggregates)
    assert figure is not None
    assert len(figure.axes) == 1
    assert len(figure.axes[0].collections) == 1
    method_ablation.plt.close(figure)
