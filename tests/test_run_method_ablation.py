import json
from pathlib import Path

import numpy as np

import scripts.run_method_ablation as method_ablation
from contracts.evaluation import EvaluationHistory, EvaluationMetrics
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION, REWARD_NAMES


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
        stop_error_m=np.asarray([4.0, 1.0], dtype=np.float64),
        time_error_s=np.asarray([-12.0, -2.0], dtype=np.float64),
        total_energy_j=np.asarray([10_000.0, 9_000.0], dtype=np.float64),
        comfort_tav=np.asarray([0.4, 0.2], dtype=np.float64),
        completed_training_episodes=np.asarray([1, 2], dtype=np.int64),
        safety_violation_positions_m=np.asarray([], dtype=np.float64),
        safety_violation_position_offsets=np.asarray([0, 0, 0], dtype=np.int64),
    )
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
        final_position_m=99.0,
        final_speed_mps=0.0,
        stop_error_m=1.0,
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


def _entry(tmp_path: Path, method: str) -> dict[str, object]:
    final_dir = tmp_path / method / "final"
    episodes = final_dir / "episodes.npz"
    evaluations = final_dir / "evaluations.npz"
    metrics = final_dir / "metrics_final.json"
    _write_episodes(episodes)
    _write_evaluations(evaluations)
    _write_metrics(metrics)
    return {
        "run_id": f"method__{method}__seed0011__r01",
        "variant_id": method,
        "variant": {"name": method},
        "repeat_index": 0,
        "seed": 11,
        "experiment_tag": f"{method}__r01",
        "artifacts": {
            "policy_final": str(final_dir / "policy_final.zip"),
            "metadata": str(final_dir.parent / "metadata.json"),
            "episodes": str(episodes),
            "evaluations": str(evaluations),
            "metrics_final": str(metrics),
            "safety_diagnostics": str(final_dir / "safety_diagnostics.npz"),
        },
        "status": "completed",
    }


def _manifest(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": method_ablation.MANIFEST_VERSION,
        "matrix_id": "method",
        "matrix_config": {
            "variants": [item.__dict__ for item in method_ablation.METHODS],
            "seeds": list(method_ablation.DEFAULT_SEEDS),
            "reference_curve_dir": ".",
        },
        "training_signature": {},
        "runs": entries,
    }


def test_matrix_maps_methods_to_expected_training_modes() -> None:
    args = method_ablation.build_arg_parser().parse_args(
        ["train", "--reference-curve-dir", "."]
    )
    runs = method_ablation.resolve_run_matrix(args)

    assert len(runs) == len(method_ablation.METHODS) * len(
        method_ablation.DEFAULT_SEEDS
    )
    assert runs[0].run_id == "method__ppo__seed0011__r01"
    assert runs[0].train_args.curriculum_profile == "none"
    assert runs[-1].train_args.curriculum_profile == "dspdl_completion"
    assert runs[-1].artifacts.evaluations.name == "evaluations.npz"


def test_manifest_round_trip_and_compatibility(tmp_path: Path) -> None:
    args = method_ablation.build_arg_parser().parse_args(
        ["train", "--reference-curve-dir", ".", "--output-root", str(tmp_path)]
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

    assert loaded == payload
    assert loaded["schema_version"] == 1
    assert loaded["matrix_id"] == "method"
    assert loaded["runs"][0]["status"] == "completed"  # type: ignore[index]


def test_curve_and_final_aggregates_use_canonical_artifacts(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, method.name) for method in method_ablation.METHODS]
    curves, curve_warnings = method_ablation.build_curve_aggregates(
        _manifest(entries), episode_smoothing_window=1
    )
    finals, final_warnings = method_ablation.build_final_aggregates(_manifest(entries))

    assert curve_warnings == []
    assert final_warnings == []
    assert [aggregate.variant_id for aggregate in curves] == [
        method.name for method in method_ablation.METHODS
    ]
    np.testing.assert_allclose(curves[0].means["ep_reward"], [1.0, 2.0])
    assert [aggregate.variant_id for aggregate in finals] == [
        method.name for method in method_ablation.METHODS
    ]
    assert finals[0].means["stop_error_m"] == 1.0
    assert finals[0].means["time_error_s"] == -2.0


def test_safety_learning_process_bins_per_episode_violation_codes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(method_ablation, "SAFETY_EPISODE_BIN_WIDTH", 2)
    first = tmp_path / "first" / "episodes.npz"
    second = tmp_path / "second" / "episodes.npz"
    _write_episodes(first, [2, 0, 3, 1, 0])
    _write_episodes(second, [0, 3, 4, 2])
    entries = []
    for index, path in enumerate((first, second)):
        entries.append(
            {
                "run_id": f"method__ppo__seed{index + 1:04d}__r01",
                "variant_id": "ppo",
                "variant": {"name": "ppo"},
                "repeat_index": index,
                "seed": index + 1,
                "experiment_tag": f"r{index + 1}",
                "artifacts": {"episodes": str(path)},
                "status": "completed",
            }
        )

    aggregates, warnings = method_ablation.build_safety_learning_aggregates(
        {"runs": entries}
    )

    assert warnings == []
    assert len(aggregates) == 1
    np.testing.assert_allclose(aggregates[0].mean_violation_rate, [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(aggregates[0].valid_seed_counts, [2, 2, 1])
