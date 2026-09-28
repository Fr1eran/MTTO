import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from mtto.cli import _load_config_from_toml
from mtto.io.artifacts import (
    ArtifactError,
    read_diagnostics,
    write_analysis_report,
    write_diagnostics,
)
from mtto.io.tensorboard import (
    list_run_directories,
    load_scalar_series_from_run,
    resolve_run_directory,
)
from mtto.rl.diagnostics import (
    REWARD_DIAGNOSTICS_SCHEMA_VERSION,
    REWARD_NAMES,
    RewardDiagnostics,
    SafetyTruncationHistogram,
    TrainingDiagnostics,
)
from mtto.rl.training_analysis.analyze import (
    compute_best_eval_metrics,
    compute_regular_training_metrics,
    compute_reward_component_analysis,
    compute_safety_truncation_position_metrics,
    compute_trajectory_evaluation_metrics,
)
from mtto.rl.training_analysis.collect import ScalarSeries, compute_sampling_health
from mtto.rl.training_analysis.output import (
    build_analysis_payload,
    render_markdown_report,
)
from mtto.rl.training_analysis.pipeline import AnalysisConfig, analyze
from mtto.workflows.analysis import AnalysisResult, analyze_training

# Pre-refactor AnalysisConfig field defaults (gate-3 ruling C), now required
# and carried by paper/specs/analysis.toml instead of the dataclass.
_DEFAULT_ANALYSIS_KWARGS: dict[str, object] = {
    "step_window_size": 5000,
    "ema_alpha": 0.1,
    "kl_threshold": 0.03,
    "include_snapshots": False,
    "export_csv": False,
    "report_bar_width": 24,
    "min_points_per_10k_steps": 5.0,
    "rollout_steps_per_update": 2048,
    "sampling_quality_mode": "warn_only",
}


def _default_analysis_config(**overrides: object) -> AnalysisConfig:
    return AnalysisConfig(**{**_DEFAULT_ANALYSIS_KWARGS, **overrides})


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


def _reward_artifact() -> RewardDiagnostics:
    transitions = np.asarray(
        [
            [1.0, -0.2, -0.1, 0, 0, 0, 0],
            [1.2, -0.3, -0.2, 0, 0, 0, 0],
            [1.1, -0.4, -0.1, 0, 0, 0, 0],
            [1.3, -0.5, -0.3, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    transitions = np.column_stack((transitions, np.zeros(4), transitions.sum(axis=1)))
    episode_rewards = np.vstack(
        (transitions[:2].sum(axis=0), transitions[2:].sum(axis=0))
    )
    return RewardDiagnostics(
        schema_version=np.asarray([REWARD_DIAGNOSTICS_SCHEMA_VERSION], dtype=np.int16),
        reward_names=np.asarray(REWARD_NAMES),
        rollout_end_step=np.asarray([2, 4], dtype=np.int64),
        rollout_transition_count=np.asarray([2, 2], dtype=np.int64),
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
        ).astype(np.int64),
        rollout_reward_cross_product=np.stack(
            (transitions[:2].T @ transitions[:2], transitions[2:].T @ transitions[2:])
        ),
        episode_end_step=np.asarray([2, 4], dtype=np.int64),
        episode_worker_rank=np.asarray([0, 0], dtype=np.int16),
        episode_index=np.asarray([0, 1], dtype=np.int64),
        episode_length=np.asarray([2, 2], dtype=np.int32),
        episode_termination_reason=np.asarray([1, 5], dtype=np.int8),
        episode_complete=np.asarray([True, True], dtype=np.bool_),
        episode_reward_sums=episode_rewards,
    )


def _safety_artifact() -> SafetyTruncationHistogram:
    return SafetyTruncationHistogram(
        bin_start_m=np.asarray([0.0, 500.0]),
        bin_end_m=np.asarray([500.0, 1000.0]),
        safety_truncation_count=np.asarray([2, 6], dtype=np.int64),
        low_safety_truncation_count=np.asarray([1, 2], dtype=np.int64),
        high_safety_truncation_count=np.asarray([1, 4], dtype=np.int64),
        global_safety_truncation_share=np.asarray([0.25, 0.75]),
        position_bin_size_m=np.asarray([500.0]),
    )


def _write_diagnostics_artifact(
    path: Path,
    *,
    schema_version: int = REWARD_DIAGNOSTICS_SCHEMA_VERSION,
    rollout_total_offset: float = 0.0,
    episode_total_offset: float = 0.0,
) -> None:
    reward = _reward_artifact()
    rollout_reward_sum = reward.rollout_reward_sum.copy()
    episode_reward_sums = reward.episode_reward_sums.copy()
    rollout_reward_sum[:, -1] += rollout_total_offset
    episode_reward_sums[:, -1] += episode_total_offset
    reward = dataclasses.replace(
        reward,
        schema_version=np.asarray([schema_version], dtype=np.int16),
        rollout_reward_sum=rollout_reward_sum,
        episode_reward_sums=episode_reward_sums,
    )
    write_diagnostics(
        path, TrainingDiagnostics(reward=reward, safety=_safety_artifact())
    )


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_diagnostics_rejects_unsupported_reward_schema(
    tmp_path: Path, version: int
) -> None:
    # Same validation path as the removed legacy reward-diagnostics reader
    # (_parse_reward_diagnostics_from_mapping), now exercised through the
    # current-format write_diagnostics/read_diagnostics round trip.
    path = tmp_path / "diagnostics.npz"
    _write_diagnostics_artifact(path, schema_version=version)

    with pytest.raises(ArtifactError, match="Unsupported reward diagnostics schema"):
        read_diagnostics(path)


def test_diagnostics_accepts_small_reward_total_rounding_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "diagnostics.npz"
    _write_diagnostics_artifact(
        path,
        rollout_total_offset=5e-4,
        episode_total_offset=5e-4,
    )

    read_diagnostics(path)


def test_diagnostics_rejects_material_reward_total_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "diagnostics.npz"
    _write_diagnostics_artifact(
        path,
        rollout_total_offset=1e-2,
        episode_total_offset=1e-2,
    )

    with pytest.raises(ArtifactError, match="total does not equal component sum"):
        read_diagnostics(path)


def test_reward_component_analysis_uses_episode_and_transition_data():
    analysis = compute_reward_component_analysis(_reward_artifact())

    assert analysis["available"] is True
    assert analysis["transition_count"] == 4
    assert analysis["complete_episode_count"] == 2
    assert analysis["components"]["safety"]["nonzero_frequency"] == 1.0
    assert analysis["components"]["terminal_stopping"]["nonzero_frequency"] == 0.0
    correlation = analysis["transition_signal_correlation"]
    assert "terminal_stopping" in correlation["excluded_constant_components"]


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


def test_safety_position_metrics_identifies_highest_truncation_count_bin():
    hist = SafetyTruncationHistogram(
        bin_start_m=np.asarray([0.0, 500.0]),
        bin_end_m=np.asarray([500.0, 1000.0]),
        safety_truncation_count=np.asarray([2, 6], dtype=np.int64),
        low_safety_truncation_count=np.asarray([1, 2], dtype=np.int64),
        high_safety_truncation_count=np.asarray([1, 4], dtype=np.int64),
        global_safety_truncation_share=np.asarray([0.25, 0.75]),
        position_bin_size_m=np.asarray([500.0]),
    )

    metrics = compute_safety_truncation_position_metrics(hist)

    assert metrics["available"] is True
    highest = metrics["highest_safety_truncation_bin"]
    assert highest["bin_start_m"] == 500.0
    assert highest["bin_end_m"] == 1000.0
    assert highest["high_safety_truncation_count"] == 4
    assert highest["global_safety_truncation_share"] == 0.75
    assert metrics["total_safety_truncation_count"] == 8


def test_safety_position_metrics_accepts_empty_artifact() -> None:
    hist = SafetyTruncationHistogram(
        bin_start_m=np.empty(0, dtype=np.float64),
        bin_end_m=np.empty(0, dtype=np.float64),
        safety_truncation_count=np.empty(0, dtype=np.int64),
        low_safety_truncation_count=np.empty(0, dtype=np.int64),
        high_safety_truncation_count=np.empty(0, dtype=np.int64),
        global_safety_truncation_share=np.empty(0, dtype=np.float64),
        position_bin_size_m=np.asarray([500.0]),
    )

    metrics = compute_safety_truncation_position_metrics(hist)

    assert metrics["available"] is True
    assert metrics["bins"] == []
    assert metrics["highest_safety_truncation_bin"] is None
    assert metrics["total_safety_truncation_count"] == 0


def test_safety_position_metrics_none() -> None:
    metrics = compute_safety_truncation_position_metrics(None)
    assert metrics["available"] is False
    assert "not provided" in metrics["reason"]


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

    markdown = render_markdown_report(payload)
    output_paths = write_analysis_report(
        output_dir=tmp_path / "safety_position_report",
        payload=payload,
        markdown=markdown,
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

    output_paths = write_analysis_report(
        output_dir=tmp_path / "unit_test_run",
        payload=payload,
        markdown="dummy",
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

    markdown = render_markdown_report(payload)
    output_paths = write_analysis_report(
        output_dir=tmp_path / "layered_best_eval",
        payload=payload,
        markdown=markdown,
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


def test_sampling_gate_strict_mode():
    sparse_map = _build_sparse_series_map()
    config = _default_analysis_config(
        sampling_quality_mode="strict_fail",
        min_points_per_10k_steps=1.0,
        rollout_steps_per_update=100,
    )

    with pytest.raises(ValueError, match="rollout_steps_per_update"):
        _ = analyze(series_map=sparse_map, config=config)


def test_sampling_gate_warn_mode_outputs_data_quality():
    sparse_map = _build_sparse_series_map()
    config = _default_analysis_config(
        sampling_quality_mode="warn_only",
        min_points_per_10k_steps=1.0,
        rollout_steps_per_update=100,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = analyze(series_map=sparse_map, config=config)

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


def test_sampling_gate_accepts_rollout_sized_mean_gap():
    sparse_map = _build_sparse_series_map()
    config = _default_analysis_config(
        sampling_quality_mode="warn_only",
        min_points_per_10k_steps=1.0,
        rollout_steps_per_update=10240,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = analyze(series_map=sparse_map, config=config)

    assert captured == []
    assert result["data_quality"]["sampling_gate"]["is_adequate"] is True
    assert result["data_quality"]["sampling_gate"]["is_adequate"] is True


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

    markdown = render_markdown_report(payload)
    output_paths = write_analysis_report(
        output_dir=tmp_path / "unit_test_run",
        payload=payload,
        markdown=markdown,
    )

    report_path = Path(output_paths["markdown_report"])
    report_text = report_path.read_text(encoding="utf-8")
    assert "absolute_activity_share" in report_text
    assert "safety: [" in report_text
    assert "objective_conflicts(top)" in report_text


def test_tensorboard_reading(tmp_path: Path):
    tb_root = tmp_path / "tb_logs"
    run_dir = tb_root / "run_test_1"
    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_scalar("rollout/ep_rew_mean", 10.0, 100)
    writer.add_scalar("rollout/ep_rew_mean", 20.0, 200)
    writer.add_scalar("train/approx_kl", 0.02, 100)
    writer.flush()
    writer.close()

    run_dirs = list_run_directories(tb_root)
    assert len(run_dirs) == 1
    assert run_dirs[0].resolve() == run_dir.resolve()

    resolved = resolve_run_directory(tb_root, "run_test")
    assert resolved.resolve() == run_dir.resolve()

    series_map = load_scalar_series_from_run(run_dir)
    assert "rollout/ep_rew_mean" in series_map
    assert "train/approx_kl" in series_map
    rew_series = series_map["rollout/ep_rew_mean"]
    assert np.array_equal(rew_series.steps, [100, 200])
    assert np.allclose(rew_series.values, [10.0, 20.0])


def test_analyze_training_new_format_mode(tmp_path: Path):
    from mtto.workflows.train import TrainConfig, train
    from paper.figures import load_paper_scenario, load_paper_task

    scenario = load_paper_scenario()
    task = load_paper_task()
    train_dir = tmp_path / "train_run"
    train_config = TrainConfig(
        reward_preset="basic_safety_punctuality",
        step_distance_m=100.0,
        gamma=0.998,
        budget_mode="environment_steps",
        training_episodes=None,
        training_rollouts=1,
        num_envs=1,
        n_steps_per_env=512,
        evaluation_interval_rollouts=None,
        evaluation_interval_episodes=None,
        evaluation_deterministic=True,
        keep_best=False,
        safety_truncation_bin_size_m=5000.0,
        device="cpu",
        seed=1,
    )
    train(scenario, task, train_config, train_dir)

    output_root = tmp_path / "analysis_out"
    res = analyze_training(
        config=_default_analysis_config(),
        output_root=output_root,
        run_name="new_format_test",
        train_run_dir=train_dir,
    )

    assert isinstance(res, AnalysisResult)
    report_dir = output_root / "new_format_test"
    assert (report_dir / "analysis_snapshot.json").exists()
    assert (report_dir / "report.md").exists()
    assert not (report_dir / "run.json").exists()


def test_analysis_config_requires_non_optional_fields():
    # AnalysisConfig no longer carries default values (gate-3 ruling C, D31):
    # every field except the two Optional[None] ones is required.
    with pytest.raises(TypeError, match="missing"):
        AnalysisConfig(sampling_quality_mode="warn_only")


def test_analysis_toml_spec_matches_original_defaults():
    # paper/specs/analysis.toml must carry the exact pre-refactor default
    # values, field-by-field, so analysis output stays numerically unchanged.
    config = _load_config_from_toml(
        Path("paper/specs/analysis.toml"), "analysis", AnalysisConfig, strict=True
    )
    assert config == _default_analysis_config()
