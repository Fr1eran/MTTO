import argparse
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_reward_ablation as reward_ablation
from contracts.evaluation import EvaluationHistory, EvaluationMetrics
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION, REWARD_NAMES


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return reward_ablation.build_arg_parser().parse_args(
        ["train", "--output-root", str(tmp_path), *extra]
    )


def _write_episodes(path: Path, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    episode_sums = np.zeros((2, len(REWARD_NAMES)), dtype=np.float64)
    episode_sums[:, 0] = np.asarray([1.0, 2.0]) + offset
    episode_sums[:, -1] = episode_sums[:, 0]
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
        episode_violation_code=np.asarray([0, 0], dtype=np.int8),
        episode_reward_sums=episode_sums,
    )


def _write_evaluations(path: Path, success: tuple[bool, bool] = (False, True)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history = EvaluationHistory(
        training_steps=np.asarray([100, 200], dtype=np.int64),
        rollout_indices=np.asarray([1, 2], dtype=np.int64),
        total_reward=np.asarray([1.0, 2.0], dtype=np.float64),
        episode_steps=np.asarray([10, 9], dtype=np.int64),
        success=np.asarray(success, dtype=np.bool_),
        stop_error_m=np.asarray([4.0, 1.0], dtype=np.float64),
        time_error_s=np.asarray([-12.0, -2.0], dtype=np.float64),
        total_energy_j=np.asarray([10_000.0, 9_000.0], dtype=np.float64),
        comfort_tav=np.asarray([0.4, 0.2], dtype=np.float64),
        completed_training_episodes=np.asarray([1, 2], dtype=np.int64),
        safety_violation_positions_m=np.asarray([500.0, 5_100.0], dtype=np.float64),
        safety_violation_position_offsets=np.asarray([0, 1, 2], dtype=np.int64),
    )
    np.savez(path, **history.to_npz_mapping())


def _write_metrics(path: Path, *, success: bool, time_error_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = EvaluationMetrics(
        success=success,
        precise_arrival=success,
        punctual_arrival=success,
        total_reward=2.0,
        total_time_s=440.0 + time_error_s,
        target_time_s=440.0,
        time_error_s=time_error_s,
        start_position_m=0.0,
        target_position_m=100.0,
        final_position_m=99.0,
        final_speed_mps=0.0,
        stop_error_m=1.0,
        total_energy_j=9_000.0,
        comfort_tav=0.2,
        comfort_er_pct=0.0,
        comfort_rms=0.1,
        terminated=success,
        truncated=not success,
        episode_steps=10,
        min_safety_margin_mps=0.0,
        mean_safety_margin_mps=0.0,
        strict_stop_error_limit_m=0.3,
        strict_time_error_limit_s=10.0,
        selection_comparison_key=(1.0, 1.0, 0.0, 1.0, 0.0, -9_000.0)
        if success
        else (0.0, 2.0),
    )
    path.write_text(
        json.dumps(metrics.to_mapping()),
        encoding="utf-8",
    )


def _write_trajectory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pos_m=np.asarray([0.0, 100.0], dtype=np.float32),
        speed_mps=np.asarray([0.0, 0.0], dtype=np.float32),
        safety_violation_positions_m=np.asarray([], dtype=np.float32),
    )


def _entry(
    tmp_path: Path,
    *,
    preset: str,
    repeat_index: int,
    seed: int,
    offset: float = 0.0,
) -> dict[str, object]:
    final_dir = tmp_path / f"{preset}_{seed}" / "final"
    episodes = final_dir / "episodes.npz"
    evaluations = final_dir / "evaluations.npz"
    metrics = final_dir / "metrics_final.json"
    _write_episodes(episodes, offset)
    _write_evaluations(evaluations)
    _write_metrics(
        metrics,
        success=repeat_index == 1,
        time_error_s=-2.0 - repeat_index,
    )
    return {
        "run_id": f"reward__{preset}__seed{seed:04d}__r{repeat_index + 1:02d}",
        "variant_id": preset,
        "variant": {"preset": preset},
        "repeat_index": repeat_index,
        "seed": seed,
        "experiment_tag": f"r{repeat_index + 1:02d}",
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
        "schema_version": reward_ablation.MANIFEST_VERSION,
        "matrix_id": "reward",
        "matrix_config": {
            "variants": [item.__dict__ for item in reward_ablation.REWARD_ABLATIONS],
            "seeds": list(reward_ablation.DEFAULT_SEEDS),
        },
        "training_signature": {},
        "runs": entries,
    }


def test_default_matrix_has_stable_run_ids_and_canonical_artifacts(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    runs = reward_ablation.resolve_run_matrix(args)

    assert len(runs) == len(reward_ablation.REWARD_ABLATIONS) * len(
        reward_ablation.DEFAULT_SEEDS
    )
    assert runs[0].run_id == "reward__basic__seed0011__r01"
    assert runs[0].artifacts.policy_final.name == "policy_final.zip"
    assert runs[0].artifacts.episodes.name == "episodes.npz"
    assert runs[0].artifacts.evaluations.name == "evaluations.npz"
    assert runs[0].artifacts.metrics_final.name == "metrics_final.json"
    assert all(run.spec.evaluation_history_path for run in runs)


def test_manifest_round_trip_uses_versioned_matrix_schema(tmp_path: Path) -> None:
    args = _args(tmp_path)
    runs = reward_ablation.resolve_run_matrix(args)
    payload = reward_ablation.build_manifest(
        args,
        runs,
        {runs[0].run_id: {"status": "completed"}},
    )

    reward_ablation._manifest_store(str(tmp_path)).save_atomic(payload)
    loaded = reward_ablation.load_manifest(str(tmp_path))
    reward_ablation._validate_manifest_compatibility(loaded, args)

    assert loaded == payload
    assert loaded["schema_version"] == 1
    assert loaded["matrix_id"] == "reward"
    assert loaded["runs"][0]["run_id"] == runs[0].run_id  # type: ignore[index]
    assert "training" not in loaded
    assert not (tmp_path / ".manifest.json.tmp").exists()

    changed = _args(tmp_path, "--training-episodes", "123")
    with pytest.raises(ValueError, match="different training settings"):
        reward_ablation._validate_manifest_compatibility(loaded, changed)


def test_train_fail_fast_records_failed_run_and_leaves_rest_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reward_ablation, "DEFAULT_SEEDS", (11, 131, 239))
    args = _args(tmp_path, "--reward-presets", "basic")
    calls: list[int] = []

    def fake_train(_args: object, *, spec: object) -> object:
        calls.append(int(spec.seed))  # type: ignore[attr-defined]
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(reward_ablation, "train_single_experiment", fake_train)

    assert reward_ablation.run_train(args) == 1
    assert calls == [11]
    manifest = reward_ablation.load_manifest(str(tmp_path))
    basic = [entry for entry in manifest["runs"] if entry["variant_id"] == "basic"]  # type: ignore[index]
    assert basic[0]["status"] == "failed"
    assert basic[0]["error_message"] == "synthetic failure"
    assert all(entry["status"] == "pending" for entry in basic[1:])


def test_train_resume_skips_only_complete_canonical_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reward_ablation, "DEFAULT_SEEDS", (11, 131))
    args = _args(tmp_path, "--reward-presets", "basic", "--resume")
    selected = reward_ablation.resolve_run_matrix(args)
    first = selected[0]
    first.artifacts.policy_final.parent.mkdir(parents=True, exist_ok=True)
    first.artifacts.policy_final.write_bytes(b"policy")
    _write_episodes(first.artifacts.episodes)
    _write_evaluations(first.artifacts.evaluations)
    _write_metrics(first.artifacts.metrics_final, success=True, time_error_s=-2.0)
    _write_trajectory(first.artifacts.trajectory_final)
    all_runs = reward_ablation.resolve_run_matrix(
        args,
        requested_presets=[item.preset for item in reward_ablation.REWARD_ABLATIONS],
    )
    reward_ablation._manifest_store(str(tmp_path)).save_atomic(
        reward_ablation.build_manifest(
            args, all_runs, {first.run_id: {"status": "completed"}}
        )
    )

    trained: list[int] = []

    def fake_train(_args: object, *, spec: object) -> object:
        trained.append(int(spec.seed))  # type: ignore[attr-defined]
        _write_episodes(Path(spec.reward_diagnostics_path))  # type: ignore[attr-defined]
        _write_evaluations(Path(spec.evaluation_history_path))  # type: ignore[attr-defined]
        final_model = Path(spec.final_model_save_path)  # type: ignore[attr-defined]
        final_model.parent.mkdir(parents=True, exist_ok=True)
        final_model.write_bytes(b"policy")
        return spec

    monkeypatch.setattr(reward_ablation, "train_single_experiment", fake_train)

    def fake_evaluate(spec: object) -> tuple[str, str]:
        trajectory_path = Path(spec.final_output_dir) / "final_trajectory.npz"  # type: ignore[attr-defined]
        _write_trajectory(trajectory_path)
        metrics_path = Path(spec.final_output_dir) / "metrics_final.json"  # type: ignore[attr-defined]
        _write_metrics(metrics_path, success=True, time_error_s=-2.0)
        return (
            str(trajectory_path),
            str(metrics_path),
        )

    monkeypatch.setattr(
        reward_ablation,
        "evaluate_final_training_run",
        fake_evaluate,
    )

    assert reward_ablation.run_train(args) == 0
    assert trained == [131]
    assert (
        reward_ablation.load_manifest(str(tmp_path))["runs"][0]["status"] == "completed"
    )  # type: ignore[index]


def test_aggregates_use_complete_runs_and_sample_std(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, preset="basic", repeat_index=0, seed=11),
        _entry(tmp_path, preset="basic", repeat_index=1, seed=131, offset=1.0),
    ]
    manifest = _manifest(entries)
    selected = reward_ablation.resolve_reward_ablation_specs(["basic"])

    curves, curve_warnings = reward_ablation.build_curve_aggregates(manifest, selected)
    finals, final_warnings = reward_ablation.build_final_aggregates(manifest, selected)

    assert curve_warnings == []
    assert final_warnings == []
    assert curves[0].valid_run_count == 2
    np.testing.assert_allclose(curves[0].means["ep_reward"], [1.5, 2.5])
    np.testing.assert_allclose(curves[0].means["success"], [0.0, 1.0])
    assert finals[0].success_rate == 0.5
    assert finals[0].means["abs_time_error_s"] == pytest.approx(2.5)
    assert finals[0].stds["abs_time_error_s"] == pytest.approx(np.sqrt(0.5))


def test_show_dry_run_does_not_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, "--dry-run")
    runs = reward_ablation.resolve_run_matrix(args)
    reward_ablation._manifest_store(str(tmp_path)).save_atomic(
        reward_ablation.build_manifest(args, runs)
    )
    monkeypatch.setattr(
        reward_ablation,
        "_plot_learning_curves",
        lambda _aggregates: (_ for _ in ()).throw(AssertionError("unexpected plot")),
    )
    monkeypatch.setattr(
        reward_ablation,
        "_plot_safety_boxplot",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected plot")),
    )

    assert (
        reward_ablation.run_show(
            reward_ablation.build_arg_parser().parse_args(
                ["show", "--output-root", str(tmp_path), "--dry-run"]
            )
        )
        == 0
    )
