import numpy as np
import pytest

from contracts import (
    AblationManifest,
    AblationRunRecord,
    ArtifactRefs,
    ContractError,
    CurriculumMetadata,
    EpisodeInfo,
    EvaluationArtifact,
    EvaluationHistory,
    EvaluationMetrics,
    ManifestStatusUpdate,
    RewardConfigSnapshot,
    RunMetadata,
    TrainingBudget,
    TrajectoryData,
)
from utils.io_utils import load_evaluation_artifact, save_evaluation_artifact


def _metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        success=True,
        precise_arrival=True,
        punctual_arrival=True,
        total_reward=12.5,
        total_time_s=438.0,
        target_time_s=440.0,
        time_error_s=-2.0,
        start_position_m=0.0,
        target_position_m=100.0,
        final_position_m=99.8,
        final_speed_mps=0.0,
        stop_error_m=0.2,
        total_energy_j=12_000.0,
        comfort_tav=0.3,
        comfort_er_pct=1.0,
        comfort_rms=0.4,
        terminated=True,
        truncated=False,
        episode_steps=42,
        min_safety_margin_mps=0.5,
        mean_safety_margin_mps=1.2,
        strict_stop_error_limit_m=0.3,
        strict_time_error_limit_s=10.0,
        selection_comparison_key=(1.0, 1.0, 0.0, 1.0, 0.0, -12_000.0),
        selection_rule="arrival_precise_punctual_energy_else_reward",
        extensions={"trajectory_source": "final"},
    )


def test_evaluation_metrics_persist_joules_and_reject_flat_legacy_fields() -> None:
    metrics = _metrics()
    payload = metrics.to_mapping()

    assert list(payload) == [
        "artifact_type",
        "schema_version",
        "success",
        "precise_arrival",
        "punctual_arrival",
        "total_reward",
        "total_time_s",
        "target_time_s",
        "time_error_s",
        "start_position_m",
        "target_position_m",
        "final_position_m",
        "final_speed_mps",
        "stop_error_m",
        "total_energy_j",
        "comfort_tav",
        "comfort_er_pct",
        "comfort_rms",
        "terminated",
        "truncated",
        "episode_steps",
        "min_safety_margin_mps",
        "mean_safety_margin_mps",
        "strict_stop_error_limit_m",
        "strict_time_error_limit_s",
        "selection_comparison_key",
        "extensions",
        "selection_rule",
    ]
    assert payload["total_energy_j"] == 12_000.0
    assert "total_energy_kj" not in payload
    assert EvaluationMetrics.from_mapping(payload) == metrics

    with pytest.raises(ContractError, match="unknown fields"):
        EvaluationMetrics.from_mapping({**payload, "total_energy_kj": 12.0})


def test_episode_info_round_trips_intermediate_stopping_point_sentinel() -> None:
    info = EpisodeInfo(
        position_m=12.0,
        speed_mps=3.0,
        stopping_point_index=-1,
        operation_time_s=4.0,
        redundant_operation_time_s=1.0,
        energy_consumption_j=2_500.0,
        comfort_tav=0.2,
        comfort_er_pct=3.0,
        comfort_rms=0.4,
    )

    payload = info.to_mapping()
    assert list(payload) == [
        "position_m",
        "speed_mps",
        "stopping_point_index",
        "operation_time_s",
        "redundant_operation_time_s",
        "energy_consumption_j",
        "comfort_tav",
        "comfort_er_pct",
        "comfort_rms",
    ]
    assert EpisodeInfo.from_mapping(payload) == info
    with pytest.raises(ContractError, match="unknown fields"):
        EpisodeInfo.from_mapping({**info.to_mapping(), "energy_consumption_kj": 2.5})


def test_run_metadata_and_nested_budget_have_one_strict_parser() -> None:
    metadata = RunMetadata(
        reward_preset_name="basic",
        reward_preset_label="basic",
        reward_preset_description="test",
        potential_shaping_components=("energy", "comfort"),
        reward_config=RewardConfigSnapshot(
            energy_reward_scale=1.0,
            comfort_reward_scale=2.0,
            enable_potential_safety=False,
            survival_reward_scale=0.0,
        ),
        curriculum=CurriculumMetadata(
            profile_name="none",
            enabled=False,
            value_source=None,
            dspdl_config=None,
            reference_curve_dir=None,
            reference_curve_artifact_path=None,
            reference_curve_metrics_path=None,
            rl_step_distance_m=None,
            context_count=None,
            initial_curriculum_version=None,
            completion_critic=None,
        ),
        schedule_time_s=440.0,
        step_distance=30.0,
        reward_discount=0.998,
        experiment_token="440p0_30p0__basic",
        training_budget=TrainingBudget(
            mode="completed_episodes",
            training_episodes=100,
            effective_training_episodes=104,
            max_episode_steps=10,
            derived_total_timesteps=1_040,
        ),
    )
    payload = metadata.to_mapping()

    assert list(payload) == [
        "artifact_type",
        "schema_version",
        "reward_preset_name",
        "reward_preset_label",
        "reward_preset_description",
        "potential_shaping_components",
        "reward_config",
        "curriculum",
        "schedule_time_s",
        "step_distance",
        "reward_discount",
        "experiment_token",
        "extensions",
        "training_budget",
    ]
    assert RunMetadata.from_mapping(payload) == metadata
    with pytest.raises(ContractError, match="unknown fields"):
        RunMetadata.from_mapping(
            {
                **payload,
                "training_budget": {
                    **payload["training_budget"],
                    "unexpected": True,
                },
            }
        )


def test_manifest_round_trip_and_status_update_are_typed() -> None:
    run = AblationRunRecord(
        run_id="run-1",
        variant_id="basic",
        variant={"preset": "basic"},
        repeat_index=0,
        seed=11,
        artifacts=ArtifactRefs(
            policy_final="run/final/policy_final.zip",
            episodes="run/final/episodes.npz",
            evaluations="run/final/evaluations.npz",
            trajectory_final="run/final/final_trajectory.npz",
            metrics_final="run/final/metrics_final.json",
        ),
        status="pending",
    )
    manifest = AblationManifest(
        matrix_id="reward",
        matrix_config={"variants": ["basic"]},
        training_signature={"training_episodes": 1},
        runs=(run,),
    )

    payload = manifest.to_mapping()
    assert list(payload) == [
        "artifact_type",
        "schema_version",
        "matrix_id",
        "matrix_config",
        "training_signature",
        "runs",
        "extensions",
    ]
    assert list(payload["runs"][0]["artifacts"]) == [
        "policy_final",
        "episodes",
        "evaluations",
        "trajectory_final",
        "metrics_final",
    ]
    assert AblationManifest.from_mapping(payload) == manifest
    assert ManifestStatusUpdate(status="completed").to_mapping() == {
        "status": "completed"
    }
    assert ManifestStatusUpdate.from_mapping(
        {"status": "running", "future_status_detail": True}, context="status"
    ) == ManifestStatusUpdate(status="running")
    with pytest.raises(ContractError, match="unknown fields"):
        AblationManifest.from_mapping({**payload, "legacy": True})


def test_evaluation_history_round_trip_rejects_unknown_arrays() -> None:
    history = EvaluationHistory(
        training_steps=np.asarray([100], dtype=np.int64),
        rollout_indices=np.asarray([1], dtype=np.int64),
        total_reward=np.asarray([2.0]),
        episode_steps=np.asarray([10], dtype=np.int64),
        success=np.asarray([True]),
        stop_error_m=np.asarray([0.1]),
        time_error_s=np.asarray([-1.0]),
        total_energy_j=np.asarray([2_000.0]),
        comfort_tav=np.asarray([0.2]),
        completed_training_episodes=np.asarray([8], dtype=np.int64),
        safety_violation_positions_m=np.asarray([50.0]),
        safety_violation_position_offsets=np.asarray([0, 1], dtype=np.int64),
    )
    payload = history.to_npz_mapping()
    restored = EvaluationHistory.from_npz_mapping(payload)

    np.testing.assert_array_equal(restored.training_steps, history.training_steps)
    np.testing.assert_array_equal(restored.total_energy_j, history.total_energy_j)
    np.testing.assert_array_equal(
        restored.safety_violation_position_offsets,
        history.safety_violation_position_offsets,
    )
    with pytest.raises(ContractError, match="unknown arrays"):
        EvaluationHistory.from_npz_mapping({**payload, "legacy": np.asarray([1])})


def test_typed_evaluation_io_has_exact_npz_schema_and_namespaced_extensions(
    tmp_path,
) -> None:
    artifact = EvaluationArtifact(
        metrics=_metrics(),
        trajectory=TrajectoryData(
            position_m=np.asarray([0.0, 100.0]),
            speed_mps=np.asarray([5.0, 0.0]),
            safety_violation_positions_m=np.asarray([50.0]),
        ),
    )
    trajectory_path = tmp_path / "final_trajectory.npz"
    metrics_path = tmp_path / "metrics_final.json"

    save_evaluation_artifact(
        artifact,
        str(trajectory_path),
        metrics_path=str(metrics_path),
        extra_metadata={"source": {"kind": "test"}},
    )

    with np.load(trajectory_path, allow_pickle=False) as payload:
        assert set(payload.files) == {
            "pos_m",
            "speed_mps",
            "safety_violation_positions_m",
        }
    loaded = load_evaluation_artifact(
        str(trajectory_path),
        str(metrics_path),
        use_metrics_cache=False,
    )
    assert loaded.metrics.total_energy_j == artifact.metrics.total_energy_j
    assert loaded.metrics.extension("source") == {"kind": "test"}
    assert loaded.metrics.created_at is not None

    np.savez_compressed(
        trajectory_path,
        pos_m=np.asarray([0.0, 100.0]),
        speed_mps=np.asarray([5.0, 0.0]),
        safety_violation_positions_m=np.asarray([50.0]),
        unexpected=np.asarray([1.0]),
    )
    with pytest.raises(ValueError, match="unknown NPZ arrays"):
        load_evaluation_artifact(
            str(trajectory_path),
            str(metrics_path),
            use_metrics_cache=False,
        )
