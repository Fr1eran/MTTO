import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_step_distance_ablation as step_distance_ablation
from scripts.run_step_distance_ablation import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_STEP_DISTANCES,
    FIXED_REWARD_PROFILE,
    StepDistanceCurveAggregate,
    STEP_DISTANCE_MANIFEST_FILENAME,
    build_arg_parser,
    build_curve_aggregates,
    build_step_distance_manifest,
    load_step_distance_manifest,
    plot_curve_aggregates,
    resolve_step_distance_run_matrix,
    train_step_distance_run,
)


def _write_episode_metrics_npz(
    final_dir: Path,
    *,
    rewards: list[float],
    lengths: list[float],
) -> Path:
    final_dir.mkdir(parents=True, exist_ok=True)
    episode_metrics_path = final_dir / "episode_metrics.npz"
    np.savez(
        episode_metrics_path,
        index=np.asarray([float(i) for i in range(len(rewards))], dtype=np.float64),
        ep_reward=np.asarray(rewards, dtype=np.float64),
        ep_len=np.asarray(lengths, dtype=np.float64),
    )
    return episode_metrics_path


def _write_run_metadata(
    final_dir: Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    final_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = final_dir / "run_metadata.json"
    payload = {"rollout_record_trigger_mode": "steps"}
    if metadata:
        payload.update(metadata)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_path


def test_train_cli_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["train"])

    assert args.output_root == DEFAULT_OUTPUT_ROOT
    assert tuple(args.max_step_distances) == DEFAULT_STEP_DISTANCES
    assert tuple(args.seed_list) == DEFAULT_SEEDS
    assert not hasattr(args, "max_train_episodes")
    assert args.dry_run is False


def test_resolve_run_matrix_expands_distances_and_seeds() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["train"])

    run_entries = resolve_step_distance_run_matrix(args)

    assert len(run_entries) == len(DEFAULT_STEP_DISTANCES) * len(DEFAULT_SEEDS)
    assert [entry.max_step_distance for entry in run_entries[:3]] == [
        DEFAULT_STEP_DISTANCES[0]
    ] * 3
    assert [entry.seed for entry in run_entries[:3]] == list(DEFAULT_SEEDS)
    assert run_entries[0].experiment_tag == "ds10p0__r01"
    assert not hasattr(run_entries[0].train_args, "max_train_episodes")


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
    assert loaded["runs"][0]["episode_metrics_path"].endswith("episode_metrics.npz")


def test_manifest_uses_total_timesteps_training_budget(
    tmp_path: Path,
) -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "train",
        "--output-root",
        str(tmp_path),
        "--max-step-distances",
        "50",
        "--seed-list",
        "42",
    ])
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
    args = parser.parse_args([
        "train",
        "--output-root",
        str(tmp_path),
        "--max-step-distances",
        "50",
        "--seed-list",
        "42",
    ])
    entry = resolve_step_distance_run_matrix(args)[0]

    calls: list[tuple[object, object]] = []

    def _fake_train_single_experiment(train_args, *, spec):
        calls.append((train_args, spec))
        final_output_dir = Path(spec.final_output_dir)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            final_output_dir / "episode_metrics.npz",
            index=np.asarray([0.0, 1.0], dtype=np.float64),
            ep_reward=np.asarray([1.5, 2.5], dtype=np.float64),
            ep_len=np.asarray([10.0, 20.0], dtype=np.float64),
        )
        return spec

    monkeypatch.setattr(
        step_distance_ablation,
        "train_single_experiment",
        _fake_train_single_experiment,
    )

    train_step_distance_run(entry)

    assert len(calls) == 1
    assert calls[0][0] is entry.train_args
    assert calls[0][1] is entry.training_run_spec
    episode_metrics_path = Path(entry.episode_metrics_path)
    assert episode_metrics_path.is_file()


def test_build_curve_aggregates_groups_monitor_data_by_step_distance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "step_distance"
    run_50_r1 = root / "run_50_r1" / "final"
    run_50_r2 = root / "run_50_r2" / "final"
    run_100_r1 = root / "run_100_r1" / "final"

    metrics_50_r1 = _write_episode_metrics_npz(
        run_50_r1,
        rewards=[1.0, 3.0, 5.0],
        lengths=[10.0, 9.0, 8.0],
    )
    metrics_50_r2 = _write_episode_metrics_npz(
        run_50_r2,
        rewards=[2.0, 4.0],
        lengths=[12.0, 10.0],
    )
    metrics_100_r1 = _write_episode_metrics_npz(
        run_100_r1,
        rewards=[10.0, 12.0],
        lengths=[5.0, 4.0],
    )
    _write_run_metadata(run_50_r1)
    _write_run_metadata(run_50_r2)
    _write_run_metadata(run_100_r1)

    manifest = {
        "max_step_distances": [50.0, 100.0],
        "runs": [
            {
                "max_step_distance": 50.0,
                "repeat_index": 0,
                "seed": 42,
                "final_output_dir": str(run_50_r1),
                "episode_metrics_path": str(metrics_50_r1),
                "status": "completed",
            },
            {
                "max_step_distance": 50.0,
                "repeat_index": 1,
                "seed": 43,
                "final_output_dir": str(run_50_r2),
                "episode_metrics_path": str(metrics_50_r2),
                "status": "completed",
            },
            {
                "max_step_distance": 100.0,
                "repeat_index": 0,
                "seed": 42,
                "final_output_dir": str(run_100_r1),
                "episode_metrics_path": str(metrics_100_r1),
                "status": "completed",
            },
        ],
    }

    aggregates, warnings = build_curve_aggregates(manifest)

    assert warnings == []
    assert [aggregate.max_step_distance for aggregate in aggregates] == [50.0, 100.0]
    aggregate_50 = aggregates[0]
    assert aggregate_50.valid_run_count == 2
    np.testing.assert_allclose(aggregate_50.reference_steps, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        aggregate_50.mean_reward,
        [1.5, 3.5, 5.0],
    )
    np.testing.assert_allclose(
        aggregate_50.mean_length,
        [11.0, 9.5, 8.0],
    )


def test_plot_curve_aggregates_uses_step_axis_and_deduped_legend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(step_distance_ablation, "apply_rl_curve_plot_style", lambda: None)
    monkeypatch.setattr(step_distance_ablation.plt, "show", lambda: None)

    aggregates = [
        StepDistanceCurveAggregate(
            max_step_distance=50.0,
            reference_steps=np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
            mean_reward=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            std_reward=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
            mean_length=np.asarray([12.0, 11.0, 10.0], dtype=np.float64),
            std_length=np.asarray([0.3, 0.2, 0.1], dtype=np.float64),
            valid_run_count=2,
            episode_metrics_paths=("a", "b"),
        ),
        StepDistanceCurveAggregate(
            max_step_distance=100.0,
            reference_steps=np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
            mean_reward=np.asarray([0.5, 1.0, 1.5], dtype=np.float64),
            std_reward=np.asarray([0.1, 0.1, 0.1], dtype=np.float64),
            mean_length=np.asarray([14.0, 13.0, 12.0], dtype=np.float64),
            std_length=np.asarray([0.2, 0.2, 0.2], dtype=np.float64),
            valid_run_count=2,
            episode_metrics_paths=("c", "d"),
        ),
    ]

    plot_curve_aggregates(aggregates)

    fig = step_distance_ablation.plt.gcf()
    assert len(fig.axes) == 2
    assert fig.axes[0].get_xlabel() == "Training steps"
    assert fig.axes[1].get_xlabel() == "Training steps"

    assert fig.legends, "Expected a figure-level legend."
    legend_labels = [text.get_text() for text in fig.legends[0].texts]
    assert legend_labels == ["50 m", "100 m"]
    step_distance_ablation.plt.close(fig)
