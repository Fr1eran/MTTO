import numpy as np
import pytest
import argparse
from pathlib import Path
from typing import Any, cast
import warnings

from rl.callbacks import TensorboardCallback
from rl.training_analysis.analyze import (
    compute_constraint_diagnostic,
    compute_evolution_metrics,
    compute_regular_training_metrics,
    compute_reward_component_impact,
)
from rl.training_analysis.collect import ScalarSeries, compute_sampling_health
from rl.training_analysis.pipeline import AnalysisConfig, run_training_analysis
from rl.training_analysis.output import build_analysis_payload, write_analysis_outputs
from rl.training_analysis.process import build_episode_snapshots
from scripts.analyze_training_data import build_arg_parser
from scripts.train_rl import resolve_log_interval


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


def test_reward_component_impact_basic():
    series_map = {
        "rewards/safety": _make_series("rewards/safety", [1.0, 1.2, 1.1, 1.3]),
        "rewards/energy": _make_series("rewards/energy", [-0.2, -0.3, -0.25, -0.3]),
        "rewards/comfort": _make_series("rewards/comfort", [-0.1, -0.15, -0.12, -0.18]),
        "rewards/punctuality": _make_series(
            "rewards/punctuality",
            [0.4, 0.5, 0.45, 0.55],
        ),
        "rewards/docking": _make_series("rewards/docking", [0.2, 0.3, 0.2, 0.35]),
    }

    impact = compute_reward_component_impact(series_map)

    assert impact["available"] is True
    dominance_sum = sum(impact["dominance"].values())
    assert np.isclose(dominance_sum, 1.0, atol=1e-6)

    correlation = impact["objective_correlation"]["matrix"]
    assert "rewards/safety" in correlation
    assert "rewards/energy" in correlation["rewards/safety"]


def test_constraint_diagnostic_basic():
    series_map = {
        "state/pos_m": _make_series("state/pos_m", [0.0, 500.0, 1000.0, 1500.0]),
        "constraint/is_truncated": _make_series(
            "constraint/is_truncated",
            [0.0, 1.0, 0.0, 1.0],
        ),
        "state/current_sp": _make_series("state/current_sp", [-1.0, 0.0, 1.0, 1.0]),
        "constraint/speed_limit_segment": _make_series(
            "constraint/speed_limit_segment",
            [0.0, 1.0, 1.0, 2.0],
        ),
        "constraint/margin_to_vmax_mps": _make_series(
            "constraint/margin_to_vmax_mps",
            [2.0, -0.5, 1.0, 0.2],
        ),
        "constraint/margin_to_vmin_mps": _make_series(
            "constraint/margin_to_vmin_mps",
            [1.0, 0.3, 2.0, -0.1],
        ),
    }

    diagnostic = compute_constraint_diagnostic(
        series_map,
        near_miss_threshold_mps=1.0,
        position_bin_size_m=500.0,
    )

    assert diagnostic["available"] is True
    assert diagnostic["geographic_failure_distribution"]["truncated_count"] == 2
    assert diagnostic["safety_band_tolerance"]["available"] is True
    assert diagnostic["safety_band_tolerance"]["near_miss_ratio"] > 0.0
    top_bin = diagnostic["geographic_failure_distribution"]["top_risk_bins"][0]
    assert "near_miss_count" in top_bin
    assert "near_miss_risk" in top_bin
    assert "violation_risk" in top_bin
    assert "failure_risk" in top_bin


def test_build_episode_snapshots_with_state_episode_id():
    series_map = {
        "state/episode_id": _make_series("state/episode_id", [0, 0, 1, 1, 2, 2]),
        "rewards/total": _make_series("rewards/total", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    }

    snapshots = build_episode_snapshots(
        series_map,
        selected_tags=["rewards/total", "state/episode_id"],
        episode_window_size=1,
    )

    assert len(snapshots) == 3
    assert snapshots[0]["episode_start"] == 0
    assert snapshots[0]["episode_end"] == 1
    assert snapshots[0]["metrics"]["rewards/total"]["mean"] == 1.5


def test_reward_component_episode_then_stage_aggregation():
    series_map = {
        "state/episode_id": _make_series("state/episode_id", [0, 0, 1, 1, 2, 2]),
        "rewards/safety": _make_series(
            "rewards/safety", [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        ),
        "rewards/energy": _make_series(
            "rewards/energy",
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ),
    }

    impact = compute_reward_component_impact(series_map, episode_window_size=2)

    assert impact["available"] is True
    assert impact["aggregation_order"] == "episode_then_stage"
    assert impact["episode_count"] == 3
    assert impact["stage_count"] == 2

    stage_profile = impact["stage_component_profile"]
    first_stage_safety_ratio = stage_profile[0]["mean_ratio"]["rewards/safety"]
    second_stage_safety_ratio = stage_profile[1]["mean_ratio"]["rewards/safety"]

    assert np.isclose(first_stage_safety_ratio, (0.5 + 2.0 / 3.0) / 2.0, atol=1e-6)
    assert np.isclose(second_stage_safety_ratio, 0.75, atol=1e-6)


def test_constraint_boundary_adhesion_uses_distance_ratio():
    series_map = {
        "state/pos_m": _make_series("state/pos_m", [0.0, 100.0, 300.0, 600.0]),
        "constraint/is_truncated": _make_series(
            "constraint/is_truncated",
            [0.0, 1.0, 0.0, 1.0],
        ),
        "state/current_sp": _make_series("state/current_sp", [-1.0, 0.0, 1.0, 1.0]),
        "constraint/speed_limit_segment": _make_series(
            "constraint/speed_limit_segment",
            [0.0, 1.0, 1.0, 2.0],
        ),
        "constraint/margin_to_vmax_mps": _make_series(
            "constraint/margin_to_vmax_mps",
            [0.5, 2.0, 0.4, 3.0],
        ),
        "constraint/margin_to_vmin_mps": _make_series(
            "constraint/margin_to_vmin_mps",
            [2.0, 2.0, 2.0, 2.0],
        ),
        "constraint/violation_code": _make_series(
            "constraint/violation_code",
            [0.0, 1.0, 0.0, 2.0],
        ),
        "state/episode_id": _make_series("state/episode_id", [0.0, 0.0, 1.0, 1.0]),
    }

    diagnostic = compute_constraint_diagnostic(
        series_map,
        near_miss_threshold_mps=1.0,
        position_bin_size_m=100.0,
        episode_window_size=1,
    )

    boundary = diagnostic["boundary_adhesion"]
    # Distances are [100, 200, 300, 0], near miss at indices 0 and 2.
    assert np.isclose(boundary["near_miss_distance_ratio"], 400.0 / 600.0, atol=1e-6)


def test_evolution_metrics_transition_matrix():
    series_map = {
        "state/episode_id": _make_series("state/episode_id", [0, 0, 0, 1, 1, 1]),
        "state/pos_m": _make_series(
            "state/pos_m", [0.0, 100.0, 200.0, 0.0, 120.0, 200.0]
        ),
        "constraint/is_truncated": _make_series(
            "constraint/is_truncated",
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ),
        "constraint/violation_code": _make_series(
            "constraint/violation_code",
            [0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
        ),
    }

    evolution = compute_evolution_metrics(series_map, episode_window_size=1)

    assert evolution["available"] is True
    assert evolution["episode_count"] == 2
    assert evolution["stage_count"] == 2

    matrix = np.asarray(evolution["overall_transition_matrix"], dtype=np.int64)
    expected = np.asarray(
        [
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(matrix, expected)


def test_write_outputs_default_no_csv(tmp_path):
    payload = build_analysis_payload(
        run_name="unit_test_run",
        run_directory="dummy",
        available_tags=["rewards/total"],
        regular_metrics={},
        reward_component_impact={"available": False},
        constraint_diagnostic={"available": False},
        evolution_metrics={"available": False},
        step_snapshots=[],
        episode_snapshots=[],
        config={"export_csv": False, "include_snapshots": False},
    )

    output_paths = write_analysis_outputs(
        payload,
        output_root=tmp_path,
        run_name="unit_test_run",
    )

    assert "summary_metrics_csv" not in output_paths
    assert "step_snapshots_csv" not in output_paths
    assert "episode_snapshots_csv" not in output_paths

    output_dir = tmp_path / "unit_test_run"
    assert (output_dir / "analysis_snapshot.json").exists()
    assert (output_dir / "report.md").exists()
    assert list(output_dir.glob("*.csv")) == []


class _FakeEnv:
    def __init__(self):
        self._attrs = {
            "rewards_info": [{"total": 999.0}],
            "state_info": [{"episode_id": 999.0}],
            "constraint_info": [{"is_truncated": 999.0}],
            "event_info": [{"episode_truncated_count": 999.0}],
        }

    def get_attr(self, attr_name: str):
        return self._attrs.get(attr_name, [{}])


class _FakeLogger:
    def __init__(self):
        self.records: list[tuple[str, float]] = []
        self.dumps: list[int] = []

    def record(self, key: str, value: float) -> None:
        self.records.append((key, float(value)))

    def dump(self, step: int) -> None:
        self.dumps.append(int(step))


class _FakeModel:
    def __init__(self, env: _FakeEnv, logger: _FakeLogger):
        self._env = env
        self.logger = logger

    def get_env(self):
        return self._env


def _make_callback_locals(*, truncated: float = 0.0) -> dict[str, Any]:
    return {
        "infos": [
            {
                "tb_diagnostics": {
                    "rewards": {"total": 1.0},
                    "state": {},
                    "constraint": {"is_truncated": truncated},
                    "event": {"episode_truncated_count": truncated},
                }
            }
        ],
        "dones": [False],
    }


def test_callback_sampling_throttle():
    env = _FakeEnv()
    logger = _FakeLogger()
    model = _FakeModel(env, logger)
    callback = TensorboardCallback(min_tb_sample_interval_steps=3)
    callback.init_callback(cast(Any, model))

    for step in range(1, 7):
        callback.num_timesteps = step
        callback.locals = _make_callback_locals()
        assert callback._on_step() is True

    # step=1 and step=4 will be sampled, each sample records four namespaces.
    assert len(logger.records) == 8
    assert all(value != 999.0 for _, value in logger.records)


def test_callback_force_dump_interval():
    env = _FakeEnv()
    logger = _FakeLogger()
    model = _FakeModel(env, logger)
    callback = TensorboardCallback(
        min_tb_sample_interval_steps=100,
        force_dump_interval_steps=5,
    )
    callback.init_callback(cast(Any, model))

    for step in range(1, 12):
        callback.num_timesteps = step
        callback.locals = _make_callback_locals()
        assert callback._on_step() is True

    assert logger.dumps == [5, 10]


def test_callback_reads_terminal_step_diagnostics_from_infos():
    env = _FakeEnv()
    logger = _FakeLogger()
    model = _FakeModel(env, logger)
    callback = TensorboardCallback(min_tb_sample_interval_steps=1)
    callback.init_callback(cast(Any, model))

    callback.num_timesteps = 1
    callback.locals = _make_callback_locals(truncated=1.0)
    assert callback._on_step() is True

    records = dict(logger.records)
    assert records["constraint/is_truncated"] == 1.0
    assert records["event/episode_truncated_count"] == 1.0


def test_callback_tracks_episode_id_from_dones():
    env = _FakeEnv()
    logger = _FakeLogger()
    model = _FakeModel(env, logger)
    callback = TensorboardCallback(min_tb_sample_interval_steps=1)
    callback.init_callback(cast(Any, model))

    callback.num_timesteps = 1
    callback.locals = {
        "infos": [
            {
                "tb_diagnostics": {
                    "rewards": {},
                    "state": {},
                    "constraint": {},
                    "event": {},
                }
            }
        ],
        "dones": [False],
    }
    assert callback._on_step() is True

    callback.num_timesteps = 2
    callback.locals = {
        "infos": [
            {
                "tb_diagnostics": {
                    "rewards": {},
                    "state": {},
                    "constraint": {},
                    "event": {},
                }
            }
        ],
        "dones": [True],
    }
    assert callback._on_step() is True

    callback.num_timesteps = 3
    callback.locals = {
        "infos": [
            {
                "tb_diagnostics": {
                    "rewards": {},
                    "state": {},
                    "constraint": {},
                    "event": {},
                }
            }
        ],
        "dones": [False],
    }
    assert callback._on_step() is True

    episode_id_values = [
        value for key, value in logger.records if key == "state/episode_id"
    ]
    assert episode_id_values == [0.0, 0.0, 1.0]


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
        "state/episode_id": _make_series_with_steps(
            "state/episode_id", steps, [10.0, 120.0, 240.0]
        ),
        "state/pos_m": _make_series_with_steps(
            "state/pos_m", steps, [0.0, 5000.0, 9000.0]
        ),
        "state/current_sp": _make_series_with_steps(
            "state/current_sp", steps, [-1.0, 1.0, 2.0]
        ),
        "constraint/is_truncated": _make_series_with_steps(
            "constraint/is_truncated", steps, [0.0, 0.0, 0.0]
        ),
        "constraint/violation_code": _make_series_with_steps(
            "constraint/violation_code", steps, [0.0, 0.0, 0.0]
        ),
        "constraint/speed_limit_segment": _make_series_with_steps(
            "constraint/speed_limit_segment", steps, [0.0, 1.0, 1.0]
        ),
        "constraint/margin_to_vmax_mps": _make_series_with_steps(
            "constraint/margin_to_vmax_mps", steps, [3.0, 2.0, 1.5]
        ),
        "constraint/margin_to_vmin_mps": _make_series_with_steps(
            "constraint/margin_to_vmin_mps", steps, [2.0, 2.0, 2.0]
        ),
        "rewards/safety": _make_series_with_steps(
            "rewards/safety", steps, [0.1, 0.2, 0.3]
        ),
    }


def test_sampling_gate_strict_mode(monkeypatch, tmp_path):
    import rl.training_analysis.pipeline as pipeline_module

    sparse_map = _build_sparse_series_map()
    monkeypatch.setattr(
        pipeline_module,
        "resolve_run_directory",
        lambda log_root, run_name=None: Path("fake_run"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_scalar_series_from_run",
        lambda run_dir: sparse_map,
    )

    config = AnalysisConfig(
        output_root=str(tmp_path),
        sampling_quality_mode="strict_fail",
        min_points_per_10k_steps=100.0,
        min_unique_episodes=100,
        max_mean_step_gap=100.0,
    )

    with pytest.raises(ValueError):
        run_training_analysis(log_root="unused", run_name="unused", config=config)


def test_sampling_gate_warn_mode_outputs_data_quality(monkeypatch, tmp_path):
    import rl.training_analysis.pipeline as pipeline_module

    sparse_map = _build_sparse_series_map()
    monkeypatch.setattr(
        pipeline_module,
        "resolve_run_directory",
        lambda log_root, run_name=None: Path("fake_run"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_scalar_series_from_run",
        lambda run_dir: sparse_map,
    )

    config = AnalysisConfig(
        output_root=str(tmp_path),
        sampling_quality_mode="warn_only",
        min_points_per_10k_steps=100.0,
        min_unique_episodes=100,
        max_mean_step_gap=100.0,
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


def test_analyze_cli_sampling_quality_args():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--min-points-per-10k-steps",
            "6.5",
            "--min-unique-episodes",
            "80",
            "--max-mean-step-gap",
            "1500",
            "--sampling-quality-mode",
            "strict_fail",
        ]
    )

    assert args.min_points_per_10k_steps == 6.5
    assert args.min_unique_episodes == 80
    assert args.max_mean_step_gap == 1500.0
    assert args.sampling_quality_mode == "strict_fail"


def test_resolve_log_interval_defaults_and_override():
    args = argparse.Namespace(log_interval=None)
    assert resolve_log_interval(args, "tune", True) == 1
    assert resolve_log_interval(args, "reproduce", True) == 5
    assert resolve_log_interval(args, "eval", True) == 10
    assert resolve_log_interval(args, "reproduce", False) == 1
    assert resolve_log_interval(args, "eval", False) == 1

    args.log_interval = 3
    assert resolve_log_interval(args, "tune", True) == 3
    assert resolve_log_interval(args, "eval", False) == 3


def test_write_outputs_includes_extended_risk_columns(tmp_path):
    payload = build_analysis_payload(
        run_name="unit_test_run",
        run_directory="dummy",
        available_tags=["constraint/is_truncated"],
        regular_metrics={},
        reward_component_impact={"available": False},
        constraint_diagnostic={
            "available": True,
            "geographic_failure_distribution": {
                "truncated_count": 1,
                "top_risk_bins": [
                    {
                        "bin_start_m": 0.0,
                        "bin_end_m": 500.0,
                        "exposure_count": 10,
                        "near_miss_count": 4,
                        "violation_count": 2,
                        "failure_count": 1,
                        "near_miss_risk": 0.4,
                        "violation_risk": 0.2,
                        "failure_risk": 0.1,
                    }
                ],
            },
            "safety_band_tolerance": {
                "average_distance_to_vmax_mps": 1.0,
                "average_distance_to_vmin_mps": 1.0,
                "near_miss_ratio": 0.4,
                "violation_ratio": 0.2,
                "sample_count": 10,
            },
            "boundary_adhesion": {
                "near_miss_distance_m": 100.0,
                "total_distance_m": 200.0,
                "near_miss_distance_ratio": 0.5,
            },
            "critical_point_risk": {
                "top_risky_points": [
                    {
                        "type": "slope_transition",
                        "point_m": 1000.0,
                        "exposure_count": 10,
                        "near_miss_count": 4,
                        "violation_count": 2,
                        "failure_count": 1,
                        "near_miss_risk": 0.4,
                        "violation_risk": 0.2,
                        "failure_risk": 0.1,
                    }
                ]
            },
        },
        evolution_metrics={"available": False},
        step_snapshots=[],
        episode_snapshots=[],
        config={"export_csv": False, "include_snapshots": False},
    )

    output_paths = write_analysis_outputs(
        payload,
        output_root=tmp_path,
        run_name="unit_test_run",
    )

    report_path = Path(output_paths["markdown_report"])
    report_text = report_path.read_text(encoding="utf-8")
    assert "near_miss_risk" in report_text
    assert "violation_risk" in report_text
    assert "failure_risk" in report_text
