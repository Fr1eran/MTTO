import json
from pathlib import Path

import numpy as np

from scripts.run_step_distance_ablation import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_STEP_DISTANCES,
    FIXED_REWARD_PROFILE,
    STEP_DISTANCE_MANIFEST_FILENAME,
    build_arg_parser,
    build_curve_aggregates,
    build_step_distance_manifest,
    load_step_distance_manifest,
    resolve_step_distance_run_matrix,
)


def _write_monitor_csv(
    monitor_dir: Path,
    *,
    rewards: list[float],
    lengths: list[float],
) -> Path:
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = monitor_dir / "monitor.csv"
    rows = ["#{\"t_start\": 0.0, \"env_id\": \"None\"}", "r,l,t"]
    for index, (reward, length) in enumerate(zip(rewards, lengths, strict=True)):
        rows.append(f"{reward},{length},{float(index)}")
    monitor_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return monitor_path


def test_train_cli_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["train"])

    assert args.output_root == DEFAULT_OUTPUT_ROOT
    assert tuple(args.max_step_distances) == DEFAULT_STEP_DISTANCES
    assert tuple(args.seed_list) == DEFAULT_SEEDS
    assert args.max_train_episodes == 1000
    assert args.dry_run is False


def test_resolve_run_matrix_expands_distances_and_seeds() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["train"])

    run_entries = resolve_step_distance_run_matrix(args)

    assert len(run_entries) == len(DEFAULT_STEP_DISTANCES) * len(DEFAULT_SEEDS)
    assert [entry.max_step_distance for entry in run_entries[:3]] == [50.0] * 3
    assert [entry.seed for entry in run_entries[:3]] == [42, 43, 44]
    assert run_entries[0].experiment_tag == "ds50p0__r01"


def test_run_matrix_keeps_basic_reward_and_fixed_hyperparameters() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "train",
        "--max-step-distances",
        "50",
        "100",
        "--seed-list",
        "42",
        "43",
        "--num-envs",
        "2",
        "--rollout-steps-per-update",
        "4096",
    ])

    run_entries = resolve_step_distance_run_matrix(args)
    first = run_entries[0]
    other_distance = run_entries[2]

    assert first.training_run_spec.reward_profile.name == FIXED_REWARD_PROFILE
    assert first.training_run_spec.enable_monitor is True
    assert first.training_run_spec.enable_callback is False
    assert first.training_run_spec.enable_env_diagnostics is False
    assert first.training_run_spec.enable_auto_analysis is False
    assert first.training_run_spec.enable_best_eval is False

    assert first.training_run_spec.reward_discount == other_distance.training_run_spec.reward_discount
    assert first.training_run_spec.schedule_time_s == other_distance.training_run_spec.schedule_time_s
    assert first.training_run_spec.num_envs == other_distance.training_run_spec.num_envs
    assert first.training_run_spec.rollout_steps_per_update == other_distance.training_run_spec.rollout_steps_per_update
    assert first.training_run_spec.max_step_distance != other_distance.training_run_spec.max_step_distance
    assert first.seed == other_distance.seed


def test_build_and_load_manifest_records_step_distance_runs(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "train",
        "--output-root",
        str(tmp_path),
        "--max-step-distances",
        "50",
        "--seed-list",
        "42",
        "43",
    ])
    run_entries = resolve_step_distance_run_matrix(args)
    statuses = {(50.0, 0): {"status": "completed"}}

    manifest = build_step_distance_manifest(args, run_entries, statuses=statuses)
    manifest_path = tmp_path / STEP_DISTANCE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_step_distance_manifest(str(tmp_path))

    assert loaded["reward_profile"] == FIXED_REWARD_PROFILE
    assert loaded["max_step_distances"] == [50.0]
    assert loaded["seed_list"] == [42, 43]
    assert loaded["runs"][0]["status"] == "completed"
    assert loaded["runs"][1]["status"] == "pending"
    assert loaded["runs"][0]["monitor_path"].endswith("monitor.csv")


def test_build_curve_aggregates_groups_monitor_data_by_step_distance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "step_distance"
    run_50_r1 = root / "run_50_r1"
    run_50_r2 = root / "run_50_r2"
    run_100_r1 = root / "run_100_r1"

    monitor_50_r1 = _write_monitor_csv(
        run_50_r1 / "monitor",
        rewards=[1.0, 3.0, 5.0],
        lengths=[10.0, 9.0, 8.0],
    )
    monitor_50_r2 = _write_monitor_csv(
        run_50_r2 / "monitor",
        rewards=[2.0, 4.0],
        lengths=[12.0, 10.0],
    )
    monitor_100_r1 = _write_monitor_csv(
        run_100_r1 / "monitor",
        rewards=[10.0, 12.0],
        lengths=[5.0, 4.0],
    )

    manifest = {
        "max_step_distances": [50.0, 100.0],
        "runs": [
            {
                "max_step_distance": 50.0,
                "repeat_index": 0,
                "seed": 42,
                "monitor_path": str(monitor_50_r1),
                "status": "completed",
            },
            {
                "max_step_distance": 50.0,
                "repeat_index": 1,
                "seed": 43,
                "monitor_path": str(monitor_50_r2),
                "status": "completed",
            },
            {
                "max_step_distance": 100.0,
                "repeat_index": 0,
                "seed": 42,
                "monitor_path": str(monitor_100_r1),
                "status": "completed",
            },
        ],
    }

    aggregates, warnings = build_curve_aggregates(manifest)

    assert warnings == []
    assert [aggregate.max_step_distance for aggregate in aggregates] == [50.0, 100.0]
    aggregate_50 = aggregates[0]
    assert aggregate_50.valid_run_count == 2
    np.testing.assert_allclose(aggregate_50.reference_episodes, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(aggregate_50.mean_reward, [1.5, 3.5, 5.0])
    np.testing.assert_allclose(aggregate_50.mean_length, [11.0, 9.5, 8.0])
