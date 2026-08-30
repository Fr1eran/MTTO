import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_step_distance_ablation as step_distance_ablation
from contracts.evaluation import EvaluationMetrics
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
    metrics = EvaluationMetrics(
        success=True,
        precise_arrival=True,
        punctual_arrival=True,
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
        strict_stop_error_limit_m=0.3,
        strict_time_error_limit_s=10.0,
        selection_comparison_key=(1.0, 1.0, 0.0, 1.0, 0.0, -9_000.0),
    )
    path.write_text(json.dumps(metrics.to_mapping()), encoding="utf-8")


def _entry(
    tmp_path: Path,
    distance: float,
    repeat_index: int,
    rewards: list[float],
) -> dict[str, object]:
    final_dir = tmp_path / f"run_{distance:g}_{repeat_index}" / "final"
    episodes = final_dir / "episodes.npz"
    metrics = final_dir / "metrics_final.json"
    _write_episodes(episodes, rewards)
    _write_metrics(metrics, value=float(repeat_index + 1))
    return {
        "run_id": (
            f"step_distance__ds{distance:g}__seed{repeat_index + 1:04d}__"
            f"r{repeat_index + 1:02d}"
        ),
        "variant_id": f"{distance:g}",
        "variant": {"step_distance": distance},
        "step_distance": distance,
        "repeat_index": repeat_index,
        "seed": repeat_index + 1,
        "experiment_tag": f"ds{distance:g}__r{repeat_index + 1:02d}",
        "artifacts": {
            "policy_final": str(final_dir / "policy_final.zip"),
            "metadata": str(final_dir.parent / "metadata.json"),
            "episodes": str(episodes),
            "evaluations": str(final_dir / "evaluations.npz"),
            "metrics_final": str(metrics),
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
            "curriculum_profile": step_distance_ablation.FIXED_CURRICULUM_PROFILE,
            "reference_curve_dir": ".",
        },
        "training_signature": {},
        "runs": entries,
    }


def test_run_matrix_expands_default_distances_and_seeds() -> None:
    args = step_distance_ablation.build_arg_parser().parse_args(
        ["train", "--reference-curve-dir", "."]
    )
    runs = step_distance_ablation.resolve_step_distance_run_matrix(args)

    assert len(runs) == len(step_distance_ablation.DEFAULT_STEP_DISTANCES) * len(
        step_distance_ablation.DEFAULT_SEEDS
    )
    assert runs[0].run_id == "step_distance__ds10p0__seed0011__r01"
    assert runs[0].evaluation_history_path.endswith("evaluations.npz")
    assert runs[0].training_run_spec.evaluation_history_path.endswith("evaluations.npz")


def test_manifest_round_trip_uses_one_new_schema_file(tmp_path: Path) -> None:
    args = step_distance_ablation.build_arg_parser().parse_args(
        [
            "train",
            "--reference-curve-dir",
            ".",
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
    assert loaded["schema_version"] == 1
    assert loaded["matrix_id"] == "step_distance"
    assert loaded["runs"][0]["artifacts"]["policy_final"].endswith(  # type: ignore[index]
        "policy_final.zip"
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
    np.testing.assert_allclose(aggregate.x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(aggregate.means["ep_reward"], [1.5, 3.5, 5.0])
    np.testing.assert_allclose(aggregate.means["ep_len"], [10.0, 10.0, 10.0])
    np.testing.assert_array_equal(aggregate.metrics["ep_reward"].count, [2, 2, 1])


def test_metric_aggregation_uses_sample_std_and_best_artifacts(tmp_path: Path) -> None:
    first = _entry(tmp_path, 50.0, 0, [1.0])
    second = _entry(tmp_path, 50.0, 1, [2.0])
    best_first = tmp_path / "best_first.json"
    best_second = tmp_path / "best_second.json"
    _write_metrics(best_first, value=1.0)
    _write_metrics(best_second, value=3.0)
    first["artifacts"]["metrics_best"] = str(best_first)  # type: ignore[index]
    second["artifacts"]["metrics_best"] = str(best_second)  # type: ignore[index]

    manifest = _manifest([first, second])
    manifest["matrix_config"]["step_distances"] = [50.0]  # type: ignore[index]
    assert step_distance_ablation.resolve_metric_source(manifest) == "best"
    aggregates, warnings = step_distance_ablation.build_metric_aggregates(
        manifest, metric_source="best"
    )

    assert warnings == []
    assert aggregates[0].means["stop_error_m"] == pytest.approx(2.0)
    assert aggregates[0].stds["stop_error_m"] == pytest.approx(np.sqrt(2.0))


def test_train_command_records_failure_and_stops_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(step_distance_ablation, "DEFAULT_STEP_DISTANCES", (50.0,))
    monkeypatch.setattr(step_distance_ablation, "DEFAULT_SEEDS", (11, 131))
    calls: list[int] = []

    def fake_train(_args: object, *, spec: object) -> object:
        calls.append(int(spec.seed))  # type: ignore[attr-defined]
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(step_distance_ablation, "train_single_experiment", fake_train)

    assert (
        step_distance_ablation.main(
            [
                "train",
                "--reference-curve-dir",
                ".",
                "--output-root",
                str(tmp_path),
                "--training-episodes",
                "1",
            ]
        )
        == 1
    )
    assert calls == [11]
    manifest = step_distance_ablation.load_step_distance_manifest(str(tmp_path))
    assert manifest["runs"][0]["status"] == "failed"  # type: ignore[index]
    assert manifest["runs"][1]["status"] == "pending"  # type: ignore[index]
