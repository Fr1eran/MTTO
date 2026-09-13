from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from contracts.training import TrainingBudget
from utils.ablation import (
    ArtifactLayout,
    ManifestSchemaError,
    ManifestStore,
    aggregate_matrix,
    align_exact,
    build_manifest_payload,
    canonical_artifacts_complete,
    execute_matrix,
    smooth_episode_curve,
    training_budget_complete,
)


def _payload(run_ids: tuple[str, ...]) -> dict[str, object]:
    return build_manifest_payload(
        matrix_id="test",
        matrix_config={"variants": ["a"], "seeds": [1]},
        training_signature={"episodes": 1},
        runs=[
            {
                "run_id": run_id,
                "variant_id": "a",
                "variant": {"name": "a"},
                "repeat_index": index,
                "seed": index + 1,
                "experiment_tag": f"r{index + 1}",
                "artifacts": {"result": f"{run_id}.dat"},
                "status": "pending",
            }
            for index, run_id in enumerate(run_ids)
        ],
    )


def test_statistics_keep_nan_missing_values_and_sample_std() -> None:
    mean, std, count = aggregate_matrix(
        np.asarray([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan]])
    )

    np.testing.assert_allclose(mean[:2], [2.0, 2.0])
    np.testing.assert_allclose(std[:2], [np.sqrt(2.0), 0.0])
    np.testing.assert_array_equal(count, [2, 1, 0])
    assert np.isnan(mean[2])
    assert np.isnan(std[2])


def test_completed_episode_budget_requires_target_and_actual_count() -> None:
    base = dict(
        mode="completed_episodes",
        training_episodes=7,
        effective_training_episodes=8,
        max_episode_steps=10,
        derived_total_timesteps=80,
        actual_training_timesteps=64,
        stop_reason="completed_episode_target",
    )
    assert training_budget_complete(
        TrainingBudget(**base, actual_completed_episodes=8, target_reached=True)
    )
    assert not training_budget_complete(
        TrainingBudget(**base, actual_completed_episodes=7, target_reached=True)
    )
    assert not training_budget_complete(
        TrainingBudget(**base, actual_completed_episodes=8, target_reached=False)
    )


def test_environment_step_budget_requires_steps_and_rollouts() -> None:
    budget = TrainingBudget(
        mode="environment_steps",
        training_episodes=None,
        effective_training_episodes=None,
        max_episode_steps=972,
        derived_total_timesteps=4_096_000,
        training_rollouts=500,
        actual_completed_episodes=6_000,
        actual_training_timesteps=4_096_000,
        actual_training_rollouts=500,
        target_reached=True,
        stop_reason="environment_step_target",
    )
    assert training_budget_complete(
        budget,
        expected_training_timesteps=4_096_000,
        expected_training_rollouts=500,
    )
    assert not training_budget_complete(replace(budget, actual_training_rollouts=499))


def test_exact_alignment_and_episode_smoothing_preserve_full_axes() -> None:
    aligned = align_exact(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 3.0]),
        np.asarray([10.0, 30.0]),
    )
    np.testing.assert_allclose(aligned[[0, 2]], [10.0, 30.0])
    assert np.isnan(aligned[1])

    episodes, values = smooth_episode_curve(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 3.0, 5.0]),
        window=2,
    )
    np.testing.assert_allclose(episodes, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(values, [1.0, 2.0, 4.0])


def test_episode_smoothing_expands_when_window_exceeds_history() -> None:
    episodes, values = smooth_episode_curve(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 3.0, 8.0]),
        window=100,
    )

    np.testing.assert_allclose(episodes, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(values, [1.0, 2.0, 4.0])


def test_episode_smoothing_handles_identity_empty_and_invalid_windows() -> None:
    episodes, values = smooth_episode_curve(
        np.asarray([1.0, 2.0]), np.asarray([2.0, 4.0]), window=1
    )
    np.testing.assert_allclose(episodes, [1.0, 2.0])
    np.testing.assert_allclose(values, [2.0, 4.0])

    empty_episodes, empty_values = smooth_episode_curve(
        np.asarray([], dtype=np.float64),
        np.asarray([], dtype=np.float64),
        window=100,
    )
    assert empty_episodes.size == 0
    assert empty_values.size == 0
    with pytest.raises(ValueError, match="episode_smoothing_window"):
        smooth_episode_curve(np.asarray([1.0]), np.asarray([1.0]), window=0)


def test_manifest_is_atomic_and_rejects_old_shape(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path, matrix_id="test")
    payload = _payload(("run-1",))
    store.save_atomic(payload)

    loaded = store.load()
    assert loaded.matrix_id == payload.matrix_id
    assert loaded.runs == payload.runs
    assert loaded.output_root == str(tmp_path)
    assert not (tmp_path / ".manifest.json.tmp").exists()
    with pytest.raises(ManifestSchemaError):
        store.save_atomic({"manifest_version": 1, "runs": []})


def test_manifest_archive_preserves_existing_file(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path, matrix_id="test")
    payload = _payload(("run-1",))
    store.save_atomic(payload)
    original = store.path.read_bytes()

    archive_path = store.archive_existing()

    assert not store.path.exists()
    assert archive_path.name.startswith("manifest.json.bak.")
    assert archive_path.read_bytes() == original


def test_relative_manifest_artifacts_follow_a_moved_output_root(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    payload = build_manifest_payload(
        matrix_id="test",
        matrix_config={"variants": ["a"], "seeds": [1]},
        training_signature={"episodes": 1},
        runs=[
            {
                "run_id": "run-1",
                "variant_id": "a",
                "variant": {"name": "a"},
                "repeat_index": 0,
                "seed": 1,
                "artifacts": {"policy_final": "runs/run-1/final/policy.zip"},
                "status": "pending",
            }
        ],
    )
    store = ManifestStore(original, matrix_id="test")
    store.save_atomic(payload)
    moved = tmp_path / "moved"
    original.rename(moved)

    loaded = ManifestStore(moved, matrix_id="test").load()

    assert loaded.output_root == str(moved)
    assert loaded.runs[0].artifacts.path_for("policy_final") == str(
        moved / "runs/run-1/final/policy.zip"
    )
    assert loaded.runs[0].artifacts.to_mapping()["policy_final"] == (
        "runs/run-1/final/policy.zip"
    )


def test_best_enabled_layout_requires_complete_best_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    final_dir = output_dir / "final"
    spec = SimpleNamespace(
        output_dir=str(output_dir),
        final_output_dir=str(final_dir),
        best_eval_output_dir=str(output_dir / "best"),
        enable_best_evaluation_artifacts=True,
        evaluation_history_path=str(final_dir / "evaluations.npz"),
        final_model_save_path=str(final_dir / "policy.zip"),
        run_metadata_path=str(output_dir / "metadata.json"),
        reward_diagnostics_path=str(final_dir / "episodes.npz"),
    )
    layout = ArtifactLayout.from_training_spec(spec)
    required = (
        layout.policy_final,
        layout.metadata,
        layout.episodes,
        layout.evaluations,
        layout.trajectory_final,
        layout.metrics_final,
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    assert not canonical_artifacts_complete(layout)

    for path in (
        layout.policy_best,
        layout.metadata_best,
        layout.trajectory_best,
        layout.metrics_best,
    ):
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"best")
    assert canonical_artifacts_complete(layout)
    assert layout.metrics_final.is_file()


@dataclass(frozen=True)
class _Run:
    run_id: str
    result_path: Path


def test_runner_fails_fast_then_resumes_by_run_id(tmp_path: Path) -> None:
    runs = tuple(
        _Run(run_id, tmp_path / f"{run_id}.dat") for run_id in ("run-1", "run-2")
    )
    store = ManifestStore(tmp_path / "manifest", matrix_id="test")

    def build(statuses: object) -> dict[str, object]:
        payload = _payload(tuple(run.run_id for run in runs))
        for entry in payload["runs"]:  # type: ignore[index]
            status = statuses.get(entry["run_id"], {})  # type: ignore[union-attr]
            entry.update(status)
        return payload

    calls: list[str] = []
    fail_first = True

    def train(run: _Run) -> object:
        nonlocal fail_first
        calls.append(run.run_id)
        if fail_first and run.run_id == "run-1":
            fail_first = False
            raise RuntimeError("synthetic failure")
        run.result_path.write_text("ok", encoding="utf-8")
        return object()

    result = execute_matrix(
        runs=runs,
        store=store,
        build_manifest=build,
        run_id_of=lambda run: run.run_id,
        required_artifacts=lambda run: run.result_path.is_file(),
        train_one=train,
        evaluate_one=lambda _run, _trained: None,
        resume=False,
        dry_run=False,
    )
    assert result == 1
    assert calls == ["run-1"]
    assert store.load()["runs"][0]["status"] == "failed"  # type: ignore[index]

    result = execute_matrix(
        runs=runs,
        store=store,
        build_manifest=build,
        run_id_of=lambda run: run.run_id,
        required_artifacts=lambda run: run.result_path.is_file(),
        train_one=train,
        evaluate_one=lambda _run, _trained: None,
        resume=True,
        dry_run=False,
    )
    assert result == 0
    assert calls == ["run-1", "run-1", "run-2"]

    _ = execute_matrix(
        runs=runs,
        store=store,
        build_manifest=build,
        run_id_of=lambda run: run.run_id,
        required_artifacts=lambda run: run.result_path.is_file(),
        train_one=train,
        evaluate_one=lambda _run, _trained: None,
        resume=True,
        dry_run=False,
    )
    assert calls == ["run-1", "run-1", "run-2"]


def test_runner_refuses_to_overwrite_existing_manifest_without_resume(
    tmp_path: Path,
) -> None:
    runs = (_Run("run-1", tmp_path / "run-1.dat"),)
    store = ManifestStore(tmp_path / "manifest", matrix_id="test")

    def build(statuses: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        payload = _payload(("run-1",))
        payload["runs"][0].update(statuses.get("run-1", {}))  # type: ignore[index]
        return payload

    store.save_atomic(build({}))
    original = store.path.read_bytes()

    with pytest.raises(FileExistsError, match="use --resume or --force-new"):
        execute_matrix(
            runs=runs,
            store=store,
            build_manifest=build,
            run_id_of=lambda run: run.run_id,
            required_artifacts=lambda run: run.result_path.is_file(),
            train_one=lambda run: run.result_path.write_text(
                "unexpected", encoding="utf-8"
            ),
            evaluate_one=lambda _run, _trained: None,
            resume=False,
            dry_run=False,
        )

    assert store.path.read_bytes() == original


def test_runner_force_new_archives_manifest_and_starts_pending_matrix(
    tmp_path: Path,
) -> None:
    runs = (_Run("run-1", tmp_path / "run-1.dat"),)
    store = ManifestStore(tmp_path / "manifest", matrix_id="test")

    def build(statuses: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        payload = _payload(("run-1",))
        payload["runs"][0].update(statuses.get("run-1", {}))  # type: ignore[index]
        return payload

    store.save_atomic(build({}))
    original = store.path.read_bytes()

    result = execute_matrix(
        runs=runs,
        store=store,
        build_manifest=build,
        run_id_of=lambda run: run.run_id,
        required_artifacts=lambda run: run.result_path.is_file(),
        train_one=lambda run: run.result_path.write_text("ok", encoding="utf-8"),
        evaluate_one=lambda _run, _trained: None,
        resume=False,
        force_new=True,
        dry_run=False,
    )

    assert result == 0
    archives = list(store.output_root.glob("manifest.json.bak.*"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == original
    assert store.load()["runs"][0]["status"] == "completed"  # type: ignore[index]


def test_runner_validates_existing_manifest_and_run_ids_before_resume(
    tmp_path: Path,
) -> None:
    runs = tuple(
        _Run(run_id, tmp_path / f"{run_id}.dat") for run_id in ("run-1", "run-2")
    )
    store = ManifestStore(tmp_path / "manifest", matrix_id="test")
    store.save_atomic(_payload(("run-1",)))
    calls: list[str] = []

    with pytest.raises(ValueError, match="missing run ids"):
        execute_matrix(
            runs=runs,
            store=store,
            build_manifest=lambda statuses: _payload(("run-1", "run-2")),
            run_id_of=lambda run: run.run_id,
            required_artifacts=lambda run: False,
            train_one=lambda _run: None,
            evaluate_one=lambda _run, _trained: None,
            validate_existing=lambda _manifest: calls.append("validated"),
            resume=True,
            dry_run=False,
        )

    assert calls == ["validated"]
