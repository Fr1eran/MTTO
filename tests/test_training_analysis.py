import argparse
import warnings
from pathlib import Path

import numpy as np
import pytest

from rl.experiment_utils import (
    DEFAULT_DEVICE,
    DEFAULT_NUM_ENVS,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    resolve_log_interval,
    resolve_run_mode,
    resolve_training_run_spec,
)
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION, REWARD_NAMES
from rl.training_analysis.analyze import (
    compute_best_eval_metrics,
    compute_curriculum_distribution_metrics,
    compute_regular_training_metrics,
    compute_reward_component_analysis,
    compute_safety_truncation_position_metrics,
    compute_trajectory_evaluation_metrics,
)
from rl.training_analysis.collect import (
    RewardDiagnosticsArtifact,
    ScalarSeries,
    compute_sampling_health,
    load_reward_diagnostics_artifact,
    with_legacy_info_tag_aliases,
)
from rl.training_analysis.output import build_analysis_payload, write_analysis_outputs
from rl.training_analysis.pipeline import AnalysisConfig, run_training_analysis
from scripts.analyze_training_data import build_arg_parser as build_analyze_arg_parser
from scripts.train_rl import build_cli_parser as build_train_rl_arg_parser


def _make_series(tag: str, values: list[float], step_start: int = 0) -> ScalarSeries:
    steps = np.arange(step_start, step_start + len(values), dtype=np.int64)
    vals = np.asarray(values, dtype=np.float64)
    wall_times = np.asarray(steps, dtype=np.float64)
    return ScalarSeries(tag=tag, steps=steps, values=vals, wall_times=wall_times)


def _make_series_with_steps(
    tag: str, steps: list[int], values: list[float]
) -> ScalarSeries:
    arr_steps = np.asarray(steps, dtype=np.int64)
    arr_values = np.asarray(values, dtype=np.float64)
    wall_times = arr_steps.astype(np.float64)
    return ScalarSeries(
        tag=tag, steps=arr_steps, values=arr_values, wall_times=wall_times
    )


def test_regular_training_metrics_basic():
    series_map = {
        "rollout/ep_rew_mean": _make_series(
            "rollout/ep_rew_mean", [1.0, 2.0, 3.0, 4.0]
        ),
        "train/entropy_loss": _make_series(
            "train/entropy_loss", [-1.0, -0.9, -0.8, -0.7]
        ),
        "train/explained_variance": _make_series(
            "train/explained_variance",
            [0.2, 0.5, 0.7, 0.8],
        ),
        "train/approx_kl": _make_series("train/approx_kl", [0.01, 0.02, 0.06, 0.03]),
    }

    metrics = compute_regular_training_metrics(
        series_map, ema_alpha=0.2, kl_threshold=0.03
    )

    assert metrics["convergence_speed_quality"]["available"] is True
    assert metrics["convergence_speed_quality"]["final_ep_rew_mean"] == 4.0
    assert metrics["convergence_speed_quality"]["rise_slope_per_step"] > 0.0

    assert metrics["policy_vitality"]["available"] is True
    assert metrics["critic_foresight"]["available"] is True
    assert metrics["update_safety"]["available"] is True
    assert metrics["update_safety"]["approx_kl_exceed_count"] == 1.0


def _reward_artifact() -> RewardDiagnosticsArtifact:
    transitions = np.asarray(
        [
            [1.0, -0.2, -0.1, 0, 0, 0, 0],
            [1.2, -0.3, -0.2, 0, 0, 0, 0],
            [1.1, -0.4, -0.1, 0, 0, 0, 0],
            [1.3, -0.5, -0.3, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    transitions = np.column_stack((transitions, transitions.sum(axis=1)))
    episode_rewards = np.vstack(
        (transitions[:2].sum(axis=0), transitions[2:].sum(axis=0))
    )
    return RewardDiagnosticsArtifact(
        reward_names=REWARD_NAMES,
        rollout_end_step=np.asarray([2, 4]),
        rollout_transition_count=np.asarray([2, 2]),
        rollout_reward_sum=np.vstack(
            (transitions[:2].sum(axis=0), transitions[2:].sum(axis=0))
        ),
        rollout_reward_abs_sum=np.vstack(
            (np.abs(transitions[:2]).sum(axis=0), np.abs(transitions[2:]).sum(axis=0))
        ),
        rollout_reward_nonzero_count=np.vstack(
            (
                np.count_nonzero(transitions[:2], axis=0),
                np.count_nonzero(transitions[2:], axis=0),
            )
        ),
        rollout_reward_cross_product=np.stack(
            (transitions[:2].T @ transitions[:2], transitions[2:].T @ transitions[2:])
        ),
        episode_end_step=np.asarray([2, 4]),
        episode_worker_rank=np.asarray([0, 0]),
        episode_index=np.asarray([0, 1]),
        episode_length=np.asarray([2, 2]),
        episode_terminated=np.asarray([True, False]),
        episode_truncated=np.asarray([False, True]),
        episode_complete=np.asarray([True, True]),
        episode_violation_code=np.asarray([0, 3], dtype=np.int8),
        episode_reward_sums=episode_rewards,
    )


def _write_reward_artifact(
    path: Path,
    *,
    schema_version: int = REWARD_DIAGNOSTICS_SCHEMA_VERSION,
    rollout_total_offset: float = 0.0,
    episode_total_offset: float = 0.0,
) -> None:
    artifact = _reward_artifact()
    rollout_rewards = artifact.rollout_reward_sum.copy()
    episode_rewards = artifact.episode_reward_sums.copy()
    rollout_rewards[:, -1] += rollout_total_offset
    episode_rewards[:, -1] += episode_total_offset
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema_version=np.asarray([schema_version], dtype=np.int16),
        reward_names=np.asarray(artifact.reward_names),
        **{
            field: (
                rollout_rewards
                if field == "rollout_reward_sum"
                else (
                    episode_rewards
                    if field == "episode_reward_sums"
                    else getattr(artifact, field)
                )
            )
            for field in artifact.__dataclass_fields__
            if field != "reward_names"
        },
    )


def test_reward_diagnostics_rejects_removed_schema(tmp_path: Path) -> None:
    artifact_path = tmp_path / "reward_diagnostics.npz"
    _write_reward_artifact(artifact_path, schema_version=1)

    with pytest.raises(ValueError, match="Unsupported reward diagnostics schema"):
        _ = load_reward_diagnostics_artifact(artifact_path)


def test_reward_diagnostics_accepts_small_reward_total_rounding_error(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "reward_diagnostics.npz"
    _write_reward_artifact(
        artifact_path,
        rollout_total_offset=5e-4,
        episode_total_offset=5e-4,
    )

    _ = load_reward_diagnostics_artifact(artifact_path)


def test_reward_diagnostics_rejects_material_reward_total_error(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "reward_diagnostics.npz"
    _write_reward_artifact(
        artifact_path,
        rollout_total_offset=1e-2,
        episode_total_offset=1e-2,
    )

    with pytest.raises(ValueError, match="total does not equal component sum"):
        _ = load_reward_diagnostics_artifact(artifact_path)


def test_reward_component_analysis_uses_episode_and_transition_data():
    analysis = compute_reward_component_analysis(_reward_artifact())

    assert analysis["available"] is True
    assert analysis["transition_count"] == 4
    assert analysis["complete_episode_count"] == 2
    assert analysis["components"]["safety"]["nonzero_frequency"] == 1.0
    assert analysis["components"]["terminal_stopping"]["nonzero_frequency"] == 0.0
    correlation = analysis["transition_signal_correlation"]
    assert "terminal_stopping" in correlation["excluded_constant_components"]


def test_training_analysis_supports_reward_artifact_without_tensorboard(
    tmp_path: Path,
) -> None:
    final_dir = tmp_path / "run_a" / "final"
    _write_reward_artifact(final_dir / "reward_diagnostics.npz")

    result = run_training_analysis(
        log_root=None,
        config=AnalysisConfig(
            final_output_dir=str(final_dir), output_root=str(tmp_path / "reports")
        ),
    )

    assert result["meta"]["run_name"] == "run_a"
    assert result["reward_component_analysis"]["transition_count"] == 4
    assert result["data_quality"]["sampling_gate"]["available"] is False


def test_best_eval_metrics_basic():
    series_map = {
        "best_eval/best_total_reward": _make_series(
            "best_eval/best_total_reward", [-10.0, -5.0, -3.0, -2.0]
        ),
        "best_eval/best_success": _make_series(
            "best_eval/best_success", [0.0, 1.0, 1.0, 1.0]
        ),
        "best_eval/best_precise_arrival": _make_series(
            "best_eval/best_precise_arrival", [0.0, 0.0, 1.0, 1.0]
        ),
        "best_eval/best_punctual_arrival": _make_series(
            "best_eval/best_punctual_arrival", [0.0, 0.0, 0.0, 1.0]
        ),
        "best_eval/last_total_reward": _make_series(
            "best_eval/last_total_reward", [-12.0, -8.0, -4.0, -3.0]
        ),
        "best_eval/last_success": _make_series(
            "best_eval/last_success", [0.0, 0.0, 1.0, 1.0]
        ),
        "best_eval/last_precise_arrival": _make_series(
            "best_eval/last_precise_arrival", [0.0, 0.0, 0.0, 1.0]
        ),
        "best_eval/last_punctual_arrival": _make_series(
            "best_eval/last_punctual_arrival", [0.0, 0.0, 0.0, 0.0]
        ),
    }

    metrics = compute_best_eval_metrics(series_map)

    assert metrics["available"] is True
    assert metrics["best_total_reward"]["final"] == -2.0
    assert metrics["best_total_reward"]["max"] == -2.0
    assert metrics["best_success"]["final"] == 1.0
    assert metrics["best_precise_arrival"]["final"] == 1.0
    assert metrics["best_punctual_arrival"]["mean"] == 0.25
    assert metrics["last_total_reward"]["final"] == -3.0
    assert metrics["last_success"]["final"] == 1.0
    assert metrics["last_precise_arrival"]["max"] == 1.0
    assert metrics["last_punctual_arrival"]["final"] == 0.0


def test_best_eval_metrics_empty():
    metrics = compute_best_eval_metrics({})
    assert metrics["available"] is False


def test_trajectory_evaluation_metrics_records_required_trends():
    series_map = {
        "best_eval/last_stop_error_m": _make_series(
            "best_eval/last_stop_error_m", [4.0, 2.0, 0.5]
        ),
        "best_eval/last_time_error_s": _make_series(
            "best_eval/last_time_error_s", [20.0, 10.0, 5.0]
        ),
        "best_eval/last_total_energy_j": _make_series(
            "best_eval/last_total_energy_j", [300.0, 250.0, 200.0]
        ),
        "best_eval/last_comfort_rms": _make_series(
            "best_eval/last_comfort_rms", [2.0, 1.5, 1.0]
        ),
    }

    metrics = compute_trajectory_evaluation_metrics(series_map)

    assert metrics["available"] is True
    assert metrics["metrics"]["stop_error_m"]["final"] == 0.5
    assert metrics["metrics"]["time_error_s"]["trend_slope_per_step"] < 0.0
    assert metrics["metrics"]["total_energy_j"]["final"] == 200.0
    assert metrics["metrics"]["comfort_rms"]["final"] == 1.0


def test_curriculum_distribution_metrics_requires_empirical_dspdl_kl():
    unavailable = compute_curriculum_distribution_metrics({})
    assert unavailable["available"] is False

    series_map = {
        "dspdl/empirical_to_target_kl": _make_series(
            "dspdl/empirical_to_target_kl", [1.0, 0.4, 0.2]
        ),
        "dspdl/current_to_target_kl": _make_series(
            "dspdl/current_to_target_kl", [0.8, 0.3, 0.1]
        ),
        "dspdl/alpha": _make_series("dspdl/alpha", [5.0, 5.0, 3.0]),
        "dspdl/converged": _make_series("dspdl/converged", [0.0, 0.0, 1.0]),
        "dspdl/update_kl": _make_series("dspdl/update_kl", [0.05, 0.05, 0.05]),
        "dspdl/critic_return_pearson": _make_series(
            "dspdl/critic_return_pearson", [0.1, 0.3, 0.5]
        ),
    }
    metrics = compute_curriculum_distribution_metrics(series_map)
    assert metrics["available"] is True
    assert metrics["diagnostics"]["converged"]["final"] == 1.0
    assert metrics["empirical_to_target_kl"]["final"] == 0.2
    assert metrics["current_to_target_kl"]["trend_slope_per_step"] < 0.0
    assert metrics["diagnostics"]["alpha"]["final"] == 3.0
    assert metrics["diagnostics"]["update_kl"]["final"] == 0.05
    assert metrics["diagnostics"]["critic_return_pearson"]["final"] == 0.5


def test_safety_position_metrics_identifies_highest_truncation_count_bin(
    tmp_path: Path,
):
    artifact_path = tmp_path / "safety_truncation_position_histogram.npz"
    np.savez(
        artifact_path,
        bin_start_m=np.asarray([0.0, 500.0]),
        bin_end_m=np.asarray([500.0, 1000.0]),
        safety_truncation_count=np.asarray([2, 6], dtype=np.int64),
        low_safety_truncation_count=np.asarray([1, 2], dtype=np.int64),
        high_safety_truncation_count=np.asarray([1, 4], dtype=np.int64),
        global_safety_truncation_share=np.asarray([0.25, 0.75]),
        position_bin_size_m=np.asarray([500.0]),
    )

    metrics = compute_safety_truncation_position_metrics(histogram_path=artifact_path)

    assert metrics["available"] is True
    highest = metrics["highest_safety_truncation_bin"]
    assert highest["bin_start_m"] == 500.0
    assert highest["bin_end_m"] == 1000.0
    assert highest["high_safety_truncation_count"] == 4
    assert highest["global_safety_truncation_share"] == 0.75
    assert metrics["total_safety_truncation_count"] == 8


def test_safety_position_metrics_accepts_empty_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "empty_safety_truncation_position_histogram.npz"
    np.savez(
        artifact_path,
        bin_start_m=np.empty(0, dtype=np.float64),
        bin_end_m=np.empty(0, dtype=np.float64),
        safety_truncation_count=np.empty(0, dtype=np.int64),
        low_safety_truncation_count=np.empty(0, dtype=np.int64),
        high_safety_truncation_count=np.empty(0, dtype=np.int64),
        global_safety_truncation_share=np.empty(0, dtype=np.float64),
        position_bin_size_m=np.asarray([500.0]),
    )

    metrics = compute_safety_truncation_position_metrics(histogram_path=artifact_path)

    assert metrics["available"] is True
    assert metrics["bins"] == []
    assert metrics["highest_safety_truncation_bin"] is None
    assert metrics["total_safety_truncation_count"] == 0


def test_safety_position_metrics_rejects_legacy_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "invalid_safety_truncation_histogram.npz"
    np.savez(
        artifact_path,
        bin_start_m=np.asarray([0.0]),
        bin_end_m=np.asarray([500.0]),
        sample_exposure_count=np.asarray([10.0]),
    )

    metrics = compute_safety_truncation_position_metrics(histogram_path=artifact_path)

    assert metrics["available"] is False
    assert "missing required fields" in metrics["reason"]


def test_markdown_reports_every_safety_position_bin(tmp_path: Path):
    payload = build_analysis_payload(
        run_name="safety_position_report",
        run_directory="dummy",
        available_tags=[],
        regular_metrics={},
        safety_truncation_position_metrics={
            "available": True,
            "total_safety_truncation_count": 8,
            "highest_safety_truncation_bin": {
                "bin_start_m": 500.0,
                "bin_end_m": 1000.0,
                "safety_truncation_count": 6,
                "global_safety_truncation_share": 0.75,
            },
            "bins": [
                {
                    "bin_start_m": 0.0,
                    "bin_end_m": 500.0,
                    "safety_truncation_count": 2,
                    "low_safety_truncation_count": 1,
                    "high_safety_truncation_count": 1,
                    "global_safety_truncation_share": 0.25,
                },
                {
                    "bin_start_m": 500.0,
                    "bin_end_m": 1000.0,
                    "safety_truncation_count": 6,
                    "low_safety_truncation_count": 2,
                    "high_safety_truncation_count": 4,
                    "global_safety_truncation_share": 0.75,
                },
            ],
        },
        step_snapshots=[],
        config={"export_csv": False, "include_snapshots": False},
    )

    output_paths = write_analysis_outputs(
        payload, output_root=tmp_path, run_name="safety_position_report"
    )
    report = Path(output_paths["markdown_report"]).read_text(encoding="utf-8")

    assert "highest_safety_truncation_bin" in report
    assert "total_safety_truncation_count: 8" in report
    assert (
        "[0, 500) m: count=2, low_count=1, high_count=1, global_share=25.00%" in report
    )
    assert (
        "[500, 1000) m: count=6, low_count=2, high_count=4, global_share=75.00%"
        in report
    )


def test_write_outputs_default_no_csv(tmp_path: Path):
    payload = build_analysis_payload(
        run_name="unit_test_run",
        run_directory="dummy",
        available_tags=["rewards/total"],
        regular_metrics={},
        reward_component_analysis={"available": False},
        step_snapshots=[],
        config={"export_csv": False, "include_snapshots": False},
    )

    output_paths = write_analysis_outputs(
        payload,
        output_root=tmp_path,
        run_name="unit_test_run",
    )

    assert "summary_metrics_csv" not in output_paths
    assert "step_snapshots_csv" not in output_paths

    output_dir = tmp_path / "unit_test_run"
    assert (output_dir / "analysis_snapshot.json").exists()
    assert (output_dir / "report.md").exists()
    assert list(output_dir.glob("*.csv")) == []


def test_markdown_best_eval_uses_arrival_layers(tmp_path: Path):
    payload = build_analysis_payload(
        run_name="layered_best_eval",
        run_directory="dummy",
        available_tags=[],
        regular_metrics={},
        best_eval_metrics={
            "available": True,
            "best_success": {"final": 1.0, "max": 1.0, "mean": 0.75},
            "best_precise_arrival": {"final": 1.0, "max": 1.0, "mean": 0.5},
            "best_punctual_arrival": {"final": 0.0, "max": 1.0, "mean": 0.25},
            "best_total_reward": {"final": 12.5, "max": 12.5, "mean": 8.0},
            "best_stop_error_m": {"final": 0.2, "max": 0.4, "mean": 0.3},
            "best_time_error_s": {"final": 8.0, "max": 20.0, "mean": 10.0},
            "best_total_energy_j": {"final": 1000.0, "max": 1200.0, "mean": 1100.0},
            "last_success": {"final": 1.0, "max": 1.0, "mean": 0.5},
            "last_precise_arrival": {"final": 0.0, "max": 1.0, "mean": 0.25},
            "last_punctual_arrival": {"final": 0.0, "max": 0.0, "mean": 0.0},
            "last_total_reward": {"final": 10.0, "max": 11.0, "mean": 7.0},
            "last_stop_error_m": {"final": 0.35, "max": 0.5, "mean": 0.4},
            "last_time_error_s": {"final": 12.0, "max": 30.0, "mean": 15.0},
            "last_total_energy_j": {"final": 1100.0, "max": 1300.0, "mean": 1150.0},
        },
        reward_component_analysis={"available": False},
        step_snapshots=[],
        config={"export_csv": False, "include_snapshots": False},
    )

    output_paths = write_analysis_outputs(
        payload,
        output_root=tmp_path,
        run_name="layered_best_eval",
    )
    report_text = Path(output_paths["markdown_report"]).read_text(encoding="utf-8")

    assert "arrival_success_rate=100.00%" in report_text
    assert "precise_arrival_rate=100.00%" in report_text
    assert "punctual_arrival_rate=0.00%" in report_text
    assert "- best_eval: success_rate=" not in report_text

    best_order = [
        "best_success",
        "best_precise_arrival",
        "best_punctual_arrival",
        "best_stop_error_m",
        "best_time_error_s",
        "best_total_reward",
        "best_total_energy_j",
    ]
    last_order = [
        "last_success",
        "last_precise_arrival",
        "last_punctual_arrival",
        "last_stop_error_m",
        "last_time_error_s",
        "last_total_reward",
        "last_total_energy_j",
    ]
    assert [report_text.index(key) for key in best_order] == sorted(
        report_text.index(key) for key in best_order
    )
    assert [report_text.index(key) for key in last_order] == sorted(
        report_text.index(key) for key in last_order
    )


def test_compute_sampling_health_basic_metrics():
    series_map = {
        "rollout/ep_rew_mean": _make_series_with_steps(
            "rollout/ep_rew_mean", [0, 5000, 10000], [1.0, 2.0, 3.0]
        )
    }
    health = compute_sampling_health(series_map)

    assert health["available"] is True
    tag_metrics = health["tag_metrics"]["rollout/ep_rew_mean"]
    assert tag_metrics["sample_count"] == 3.0
    assert tag_metrics["mean_step_gap"] == 5000.0
    assert tag_metrics["p95_step_gap"] == 5000.0
    assert tag_metrics["samples_per_10k_steps"] == 3.0


def test_analysis_excludes_removed_tags_after_legacy_alias_mapping():
    series_map = {
        "basic/episode_id": _make_series("basic/episode_id", [0.0, 1.0]),
        "basic/position": _make_series("basic/position", [0.0, 100.0]),
        "constraint/is_truncated": _make_series("constraint/is_truncated", [0.0, 1.0]),
        "constraint/violation_code": _make_series(
            "constraint/violation_code", [0.0, 0.0]
        ),
        "event/episode_truncated_count": _make_series(
            "event/episode_truncated_count", [0.0, 1.0]
        ),
        "rewards/terminal_stopping": _make_series(
            "rewards/terminal_stopping", [0.0, 15.0]
        ),
        "rewards/punctuality": _make_series("rewards/punctuality", [0.0, 5.0]),
        "rewards/safety": _make_series("rewards/safety", [0.1, 0.2]),
    }

    resolved = with_legacy_info_tag_aliases(series_map)

    assert "basic/episode_id" not in resolved
    assert "basic/position" not in resolved
    assert "constraint/is_truncated" not in resolved
    assert "constraint/violation_code" not in resolved
    assert "event/episode_truncated_count" not in resolved
    assert "rewards/terminal_stopping" not in resolved
    assert "rewards/punctuality" not in resolved
    assert "outcome/truncated" in resolved
    assert "rewards/safety" in resolved


def _build_sparse_series_map() -> dict[str, ScalarSeries]:
    steps = [0, 10240, 20480]
    return {
        "rollout/ep_rew_mean": _make_series_with_steps(
            "rollout/ep_rew_mean", steps, [-30.0, -29.5, -29.0]
        ),
        "outcome/truncated": _make_series_with_steps(
            "outcome/truncated", steps, [0.0, 0.0, 0.0]
        ),
        "rewards/safety": _make_series_with_steps(
            "rewards/safety", steps, [0.1, 0.2, 0.3]
        ),
    }


def test_sampling_gate_strict_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import rl.training_analysis.pipeline as pipeline_module

    sparse_map = _build_sparse_series_map()

    def _fake_resolve_run_directory(log_root: object, run_name: object = None) -> Path:
        del log_root, run_name
        return Path("fake_run")

    def _fake_load_scalar_series(
        run_dir: object,
    ) -> dict[str, ScalarSeries]:
        del run_dir
        return sparse_map

    monkeypatch.setattr(
        pipeline_module,
        "resolve_run_directory",
        _fake_resolve_run_directory,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_scalar_series_from_run",
        _fake_load_scalar_series,
    )

    config = AnalysisConfig(
        output_root=str(tmp_path),
        sampling_quality_mode="strict_fail",
        min_points_per_10k_steps=1.0,
        rollout_steps_per_update=100,
    )

    with pytest.raises(ValueError, match="rollout_steps_per_update"):
        _ = run_training_analysis(log_root="unused", run_name="unused", config=config)


def test_sampling_gate_warn_mode_outputs_data_quality(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import rl.training_analysis.pipeline as pipeline_module

    sparse_map = _build_sparse_series_map()

    def _fake_resolve_run_directory(log_root: object, run_name: object = None) -> Path:
        del log_root, run_name
        return Path("fake_run")

    def _fake_load_scalar_series(
        run_dir: object,
    ) -> dict[str, ScalarSeries]:
        del run_dir
        return sparse_map

    monkeypatch.setattr(
        pipeline_module,
        "resolve_run_directory",
        _fake_resolve_run_directory,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_scalar_series_from_run",
        _fake_load_scalar_series,
    )

    config = AnalysisConfig(
        output_root=str(tmp_path),
        sampling_quality_mode="warn_only",
        min_points_per_10k_steps=1.0,
        rollout_steps_per_update=100,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = run_training_analysis(
            log_root="unused", run_name="unused", config=config
        )

    assert any(
        "Sampling quality below configured thresholds" in str(item.message)
        for item in captured
    )
    assert "data_quality" in result
    assert result["data_quality"]["sampling_gate"]["is_adequate"] is False
    assert (
        result["data_quality"]["sampling_gate"]["metrics"]["rollout_steps_per_update"]
        == 100.0
    )


def test_sampling_gate_accepts_rollout_sized_mean_gap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import rl.training_analysis.pipeline as pipeline_module

    sparse_map = _build_sparse_series_map()

    def _fake_resolve_run_directory(log_root: object, run_name: object = None) -> Path:
        del log_root, run_name
        return Path("fake_run")

    def _fake_load_scalar_series(
        run_dir: object,
    ) -> dict[str, ScalarSeries]:
        del run_dir
        return sparse_map

    monkeypatch.setattr(
        pipeline_module,
        "resolve_run_directory",
        _fake_resolve_run_directory,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_scalar_series_from_run",
        _fake_load_scalar_series,
    )

    config = AnalysisConfig(
        output_root=str(tmp_path),
        sampling_quality_mode="warn_only",
        min_points_per_10k_steps=1.0,
        rollout_steps_per_update=10240,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = run_training_analysis(
            log_root="unused", run_name="unused", config=config
        )

    assert captured == []
    assert result["data_quality"]["sampling_gate"]["is_adequate"] is True


def test_analyze_cli_sampling_quality_args():
    parser = build_analyze_arg_parser()
    args = parser.parse_args(
        [
            "--min-points-per-10k-steps",
            "6.5",
            "--rollout-steps-per-update",
            "8192",
            "--sampling-quality-mode",
            "strict_fail",
        ]
    )

    assert args.min_points_per_10k_steps == 6.5
    assert args.rollout_steps_per_update == 8192
    assert args.sampling_quality_mode == "strict_fail"


def test_analyze_cli_rejects_removed_max_mean_step_gap() -> None:
    parser = build_analyze_arg_parser()

    with pytest.raises(SystemExit):
        _ = parser.parse_args(["--max-mean-step-gap", "1500"])


def test_analyze_cli_accepts_dry_run() -> None:
    parser = build_analyze_arg_parser()
    args = parser.parse_args(["--dry-run", "--log-root", "tb_logs"])

    assert args.dry_run is True
    assert args.log_root == "tb_logs"


def test_resolve_log_interval_defaults_and_override():
    args = argparse.Namespace(log_interval=None)
    assert resolve_log_interval(args, "tune", True) == 1
    assert resolve_log_interval(args, "reproduce", True) == 5
    assert resolve_log_interval(args, "monitor_best", True) == 1
    assert resolve_log_interval(args, "best_only", True) == 10
    assert resolve_log_interval(args, "reproduce", False) == 1
    assert resolve_log_interval(args, "best_only", False) == 1

    args.log_interval = 3
    assert resolve_log_interval(args, "tune", True) == 3
    assert resolve_log_interval(args, "monitor_best", False) == 3


@pytest.mark.parametrize(
    "run_mode, expected",
    [
        ("tune", (True, True, True, True)),
        ("reproduce", (False, True, False, False)),
        ("monitor_best", (True, True, False, True)),
        ("best_only", (False, False, False, True)),
    ],
)
def test_resolve_run_mode_defaults(run_mode: str, expected: tuple[bool, ...]) -> None:
    args = argparse.Namespace(
        run_mode=run_mode,
        enable_tb=None,
        enable_monitor=None,
        enable_auto_analysis=None,
        enable_best_evaluation_artifacts=None,
    )
    (
        _,
        enable_tb,
        enable_monitor,
        enable_auto_analysis,
        enable_best_evaluation_artifacts,
    ) = resolve_run_mode(args)

    assert (
        enable_tb,
        enable_monitor,
        enable_auto_analysis,
        enable_best_evaluation_artifacts,
    ) == expected


def test_resolve_run_mode_keeps_artifact_callbacks_when_tb_disabled() -> None:
    args = argparse.Namespace(
        run_mode="tune",
        enable_tb=False,
        enable_monitor=None,
        enable_auto_analysis=None,
        enable_best_evaluation_artifacts=None,
    )
    _, enable_tb, *_rest = resolve_run_mode(args)

    assert enable_tb is False


def test_train_rl_cli_rejects_removed_eval_mode() -> None:
    parser = build_train_rl_arg_parser()
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["--run-mode", "eval"])


def test_train_rl_cli_accepts_new_run_modes() -> None:
    parser = build_train_rl_arg_parser()
    for mode in ("tune", "reproduce", "monitor_best", "best_only"):
        args = parser.parse_args(["--run-mode", mode])
        assert args.run_mode == mode


def test_train_rl_cli_accepts_reward_preset_and_experiment_tag() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(
        [
            "--reward-preset",
            "basic_safety",
            "--experiment-tag",
            "trial_a",
        ]
    )

    assert args.reward_preset == "basic_safety"
    assert args.experiment_tag == "trial_a"


def test_train_rl_cli_rejects_removed_fixed_reverse_profile() -> None:
    parser = build_train_rl_arg_parser()
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["--curriculum-profile", "fixed_reverse"])


def test_train_rl_cli_resolves_dspdl_profile(tmp_path: Path) -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(
        [
            "--curriculum-profile",
            "dspdl",
            "--reference-curve-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    spec = resolve_training_run_spec(args)
    assert spec.curriculum_profile == "dspdl"
    assert spec.dspdl_config is not None
    curriculum = spec.run_metadata["curriculum"]
    assert curriculum["profile_name"] == "dspdl"
    assert curriculum["enabled"] is True
    assert curriculum["dspdl_config"]["relative_entropy_bound"] == 0.02
    assert curriculum["dspdl_config"]["target_uniform_mass"] == 0.05
    assert curriculum["dspdl_config"]["target_kl_stop"] == 0.1
    assert curriculum["dspdl_config"]["alpha_warmup_updates"] == 5
    assert curriculum["dspdl_config"]["update_interval_rollouts"] == 2
    assert curriculum["dspdl_config"]["zeta"] == 0.01
    assert "dspdl" in Path(spec.output_dir).name


def test_train_rl_cli_resolves_completion_dspdl_profile(tmp_path: Path) -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(
        [
            "--curriculum-profile",
            "dspdl_completion",
            "--reference-curve-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    spec = resolve_training_run_spec(args)
    assert spec.curriculum_profile == "dspdl_completion"
    assert spec.dspdl_config is not None
    curriculum = spec.run_metadata["curriculum"]
    config = curriculum["dspdl_config"]
    assert curriculum["profile_name"] == "dspdl_completion"
    assert curriculum["enabled"] is True
    assert curriculum["value_source"] == "task_completion"
    assert config["zeta"] == pytest.approx(1.0)
    assert config["completion_floor"] == pytest.approx(0.1)
    assert config["completion_ema_alpha"] == pytest.approx(0.1)
    assert config["alpha_min"] == pytest.approx(0.01)
    assert config["alpha_max"] == pytest.approx(0.05)
    assert "dspdl_completion" in Path(spec.output_dir).name


def test_train_rl_cli_overrides_completion_alpha_max(tmp_path: Path) -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(
        [
            "--curriculum-profile",
            "dspdl_completion",
            "--completion-alpha-max",
            "0.05",
            "--reference-curve-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    spec = resolve_training_run_spec(args)
    assert spec.dspdl_config is not None
    assert spec.run_metadata["curriculum"]["dspdl_config"]["alpha_max"] == (
        pytest.approx(0.05)
    )


def test_completion_alpha_max_rejects_noncompletion_profile() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(["--completion-alpha-max", "0.05", "--dry-run"])

    with pytest.raises(ValueError, match="only valid with dspdl_completion"):
        _ = resolve_training_run_spec(args)


def test_train_rl_curriculum_requires_existing_reference_directory() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(["--curriculum-profile", "dspdl"])
    with pytest.raises(ValueError, match="reference_curve_dir is required"):
        _ = resolve_training_run_spec(args)


def test_train_rl_cli_accepts_dry_run() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--reward-preset",
            "basic",
        ]
    )

    assert args.dry_run is True
    assert args.reward_preset == "basic"


def test_train_rl_cli_uses_shared_vector_environment_defaults() -> None:
    args = build_train_rl_arg_parser().parse_args([])

    assert args.num_envs == DEFAULT_NUM_ENVS
    assert not hasattr(args, "vec_env_type")
    assert args.rollout_steps_per_update == DEFAULT_ROLLOUT_STEPS_PER_UPDATE
    assert args.device == DEFAULT_DEVICE


def test_train_rl_cli_rejects_removed_subproc_start_method_option() -> None:
    parser = build_train_rl_arg_parser()
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["--subproc-start-method", "spawn"])


def test_resolve_training_run_spec_plans_paths_and_switches() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(
        [
            "--run-mode",
            "monitor_best",
            "--reward-preset",
            "basic_safety",
            "--experiment-tag",
            "batch_a",
            "--dry-run",
        ]
    )

    spec = resolve_training_run_spec(args)

    assert spec.run_mode == "monitor_best"
    assert spec.enable_tb is True
    assert spec.enable_best_evaluation_artifacts is True
    assert Path(spec.reward_diagnostics_path).name == "reward_diagnostics.npz"
    assert spec.reward_preset.name == "basic_safety"
    assert Path(spec.output_dir).name == "465p0_30p0__basic_safety__batch_a"
    assert Path(spec.run_metadata_path).name == "run_metadata.json"
    assert spec.run_metadata["reward_preset_name"] == "basic_safety"
    assert spec.num_envs == DEFAULT_NUM_ENVS
    assert "vec_env_type" not in spec.run_metadata
    assert spec.n_steps_per_env == 1024
    assert spec.rollout_steps_per_update == DEFAULT_ROLLOUT_STEPS_PER_UPDATE
    assert "subproc_start_method" not in spec.run_metadata
    assert spec.dry_run is True


def test_training_episode_budget_rounds_up_to_a_whole_vector_batch() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(
        [
            "--training-episodes",
            "7001",
            "--num-envs",
            "8",
            "--dry-run",
        ]
    )

    spec = resolve_training_run_spec(args)

    budget = spec.run_metadata["training_budget"]
    assert spec.training_episodes == 7001
    assert budget["training_episodes"] == 7001
    assert budget["effective_training_episodes"] == 7008
    assert budget["max_episode_steps"] == spec.max_episode_steps
    assert budget["derived_total_timesteps"] == spec.total_timesteps
    assert spec.total_timesteps % spec.rollout_steps_per_update == 0


def test_training_defaults_to_a_completed_episode_budget() -> None:
    spec = resolve_training_run_spec(
        build_train_rl_arg_parser().parse_args(["--dry-run"])
    )

    assert spec.training_episodes == 7000
    assert spec.run_metadata["training_budget"]["effective_training_episodes"] == 7000


def test_tune_mode_enables_safety_truncation_histogram() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(["--run-mode", "tune", "--dry-run"])

    spec = resolve_training_run_spec(args)

    assert spec.enable_safety_truncation_histogram is True


def test_multi_environment_training_omits_backend_metadata() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args(["--num-envs", "2", "--dry-run"])

    spec = resolve_training_run_spec(args)

    assert spec.num_envs == 2
    assert "vec_env_type" not in spec.run_metadata
    assert "subproc_start_method" not in spec.run_metadata


@pytest.mark.parametrize("vec_env_type", ("dummy", "subproc"))
def test_train_rl_cli_rejects_vec_env_type_option(vec_env_type: str) -> None:
    parser = build_train_rl_arg_parser()
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["--vec-env-type", vec_env_type])


def test_train_rl_cli_rejects_removed_monitor_log_dir_option() -> None:
    parser = build_train_rl_arg_parser()
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["--monitor-log-dir", "output/tmp/monitor"])


def test_write_outputs_includes_reward_quality_columns(tmp_path: Path):
    payload = build_analysis_payload(
        run_name="unit_test_run",
        run_directory="dummy",
        available_tags=["rewards/safety", "rewards/energy"],
        regular_metrics={},
        reward_component_analysis={
            "available": True,
            "transition_count": 10,
            "complete_episode_count": 2,
            "partial_episode_count": 0,
            "components": {
                "safety": {
                    "absolute_activity_share": 0.6,
                    "signed_return_ratio": 0.8,
                    "nonzero_frequency": 0.5,
                    "active_mean_absolute_strength": 2.0,
                },
                "energy": {
                    "absolute_activity_share": 0.4,
                    "signed_return_ratio": -0.2,
                    "nonzero_frequency": 1.0,
                    "active_mean_absolute_strength": 0.5,
                },
            },
            "episode_return_correlation": {
                "matrix": {
                    "rewards/safety": {"rewards/safety": 1.0, "rewards/energy": -0.5},
                    "rewards/energy": {"rewards/safety": -0.5, "rewards/energy": 1.0},
                },
                "strong_negative_pairs": [
                    {
                        "left": "rewards/safety",
                        "right": "rewards/energy",
                        "pearson": -0.5,
                    }
                ],
            },
        },
        step_snapshots=[],
        config={"export_csv": False, "include_snapshots": False},
    )

    output_paths = write_analysis_outputs(
        payload,
        output_root=tmp_path,
        run_name="unit_test_run",
    )

    report_path = Path(output_paths["markdown_report"])
    report_text = report_path.read_text(encoding="utf-8")
    assert "absolute_activity_share" in report_text
    assert "safety: [" in report_text
    assert "objective_conflicts(top)" in report_text
