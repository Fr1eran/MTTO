import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_method_ablation as method_ablation
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION, REWARD_NAMES


def _write_reward_diagnostics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    episode_sums = np.zeros((2, len(REWARD_NAMES)), dtype=np.float64)
    episode_sums[:, 0] = [1.0, 2.0]
    episode_sums[:, -1] = [1.0, 2.0]
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
    assert args.num_envs == 8
    assert args.vec_env_type == "dummy"
    assert args.rollout_steps_per_update == 8192
    assert args.device == "cpu"
    assert all(run.spec.evaluation_interval_rollouts == 12 for run in runs)
    first_by_method = {run.method.name: run for run in runs if run.repeat_index == 0}
    assert first_by_method["ppo"].train_args.reward_preset == "basic"
    assert first_by_method["ppo"].train_args.curriculum_profile == "none"
    assert (
        first_by_method["ppo_pbrs"].train_args.reward_preset == "basic_safety"
    )
    assert first_by_method["ppo_dspdl"].train_args.curriculum_profile == "dspdl"
    assert first_by_method["ppo_dspdl"].train_args.reference_curve_dir == "."
    assert (
        first_by_method["ppo_pbrs_dspdl"].train_args.reward_preset
        == "basic_safety"
    )
    assert first_by_method["ppo_pbrs_dspdl"].train_args.curriculum_profile == "dspdl"


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
    curves, curve_warnings = method_ablation.build_curve_aggregates(manifest)
    finals, final_warnings = method_ablation.build_final_aggregates(manifest)

    assert curve_warnings == []
    assert final_warnings == []
    assert [aggregate.method.name for aggregate in curves] == [
        method.name for method in method_ablation.METHODS
    ]
    assert curves[0].means["stop_error_m"][-1] == 1.0
    assert [aggregate.method.name for aggregate in finals] == [
        method.name for method in method_ablation.METHODS
    ]
    assert finals[0].success_rate == 1.0
    assert finals[0].means["time_error_s"] == -2.0


def test_safety_boxplot_accepts_periodic_position_schema(tmp_path: Path) -> None:
    periodic_path = tmp_path / "final" / "evaluation_history.npz"
    _write_periodic(periodic_path)
    manifest = {
        "runs": [
            {
                "method": "ppo",
                "status": "completed",
                "evaluation_history_path": str(periodic_path),
            }
        ]
    }
    figure = method_ablation._plot_safety_boxplot(manifest)
    assert figure is not None
    method_ablation.plt.close(figure)
