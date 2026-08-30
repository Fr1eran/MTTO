from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.ablation import (
    ArtifactLayout,
    ManifestSchemaError,
    ManifestStore,
    aggregate_matrix,
    align_exact,
    build_manifest_payload,
    canonical_artifacts_complete,
    execute_matrix,
    materialize_canonical_artifacts,
    smooth_episode_curve,
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


def test_exact_alignment_and_trailing_smoothing_preserve_axes() -> None:
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
    np.testing.assert_allclose(episodes, [2.0, 3.0])
    np.testing.assert_allclose(values, [2.0, 4.0])


def test_manifest_is_atomic_and_rejects_old_shape(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path, matrix_id="test")
    payload = _payload(("run-1",))
    store.save_atomic(payload)

    assert store.load() == payload
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


def test_canonical_artifact_check_does_not_materialize_legacy_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    final_dir = output_dir / "final"
    spec = SimpleNamespace(
        output_dir=str(output_dir),
        final_output_dir=str(final_dir),
        best_eval_output_dir=str(output_dir / "best_rollouts"),
        enable_best_evaluation_artifacts=False,
        evaluation_history_path=str(final_dir / "evaluation_history.npz"),
        final_model_save_path=str(final_dir / "final_model.zip"),
        run_metadata_path=str(output_dir / "run_metadata.json"),
        reward_diagnostics_path=str(final_dir / "reward_diagnostics.npz"),
    )
    layout = ArtifactLayout.from_training_spec(spec)
    for source in layout.legacy_paths.values():
        if source is not None:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"legacy")

    assert not canonical_artifacts_complete(layout)
    assert not layout.policy_final.exists()

    materialize_canonical_artifacts(layout)
    layout.trajectory_final.parent.mkdir(parents=True, exist_ok=True)
    layout.trajectory_final.write_bytes(b"trajectory")
    assert canonical_artifacts_complete(layout)


def test_artifact_layout_materializes_legacy_training_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    final_dir = output_dir / "final"
    legacy_evaluations = final_dir / "evaluation_history.npz"
    spec = SimpleNamespace(
        output_dir=str(output_dir),
        final_output_dir=str(final_dir),
        best_eval_output_dir=str(output_dir / "best_rollouts"),
        enable_best_evaluation_artifacts=False,
        evaluation_history_path=str(legacy_evaluations),
        final_model_save_path=str(final_dir / "final_model.zip"),
        run_metadata_path=str(output_dir / "run_metadata.json"),
        reward_diagnostics_path=str(final_dir / "reward_diagnostics.npz"),
    )
    layout = ArtifactLayout.from_training_spec(spec)
    for source in layout.legacy_paths.values():
        if source is not None:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"artifact")

    materialize_canonical_artifacts(layout)

    assert layout.policy_final.is_file()
    assert layout.metadata.is_file()
    assert layout.episodes.is_file()
    assert layout.evaluations.is_file()
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
