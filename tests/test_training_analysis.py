import numpy as np
import pytest
import argparse
from pathlib import Path
from typing import Any, cast
import warnings

from rl.callbacks import TensorboardCallback
from rl.experiment_utils import (
    resolve_log_interval,
    resolve_run_mode,
    resolve_training_run_spec,
)
from rl.training_analysis.analyze import (
    compute_best_eval_metrics,
    compute_constraint_diagnostic,
    compute_evolution_metrics,
    compute_regular_training_metrics,
    compute_reward_component_impact,
)
from rl.training_analysis.collect import ScalarSeries, compute_sampling_health
from rl.training_analysis.pipeline import AnalysisConfig, run_training_analysis
from rl.training_analysis.output import build_analysis_payload, write_analysis_outputs
from rl.training_analysis.process import build_episode_snapshots
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


def test_reward_component_impact_basic():
    series_map = {
        "rewards/safety": _make_series("rewards/safety", [1.0, 1.2, 1.1, 1.3]),
        "rewards/energy": _make_series("rewards/energy", [-0.2, -0.3, -0.25, -0.3]),
        "rewards/comfort": _make_series("rewards/comfort", [-0.1, -0.15, -0.12, -0.18]),
        "rewards/punctuality": _make_series(
            "rewards/punctuality",
            [0.4, 0.5, 0.45, 0.55],
        ),
        "rewards/stopping": _make_series("rewards/stopping", [0.2, 0.3, 0.2, 0.35]),
    }

    impact = compute_reward_component_impact(series_map)

    assert impact["available"] is True
    dominance_sum = sum(impact["dominance"].values())
    assert np.isclose(dominance_sum, 1.0, atol=1e-6)

    correlation = impact["objective_correlation"]["matrix"]
    assert "rewards/safety" in correlation
    assert "rewards/energy" in correlation["rewards/safety"]


def test_reward_component_correlation_skips_zero_std_components_without_warning():
    series_map = {
        "rewards/safety": _make_series("rewards/safety", [0.0, 0.0, 0.0, 0.0]),
        "rewards/energy": _make_series("rewards/energy", [-0.1, -0.2, -0.4, -0.8]),
        "rewards/comfort": _make_series("rewards/comfort", [-0.3, -0.2, -0.1, -0.05]),
        "rewards/punctuality": _make_series(
            "rewards/punctuality",
            [0.0, 0.0, 0.0, 0.0],
        ),
    }

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        impact = compute_reward_component_impact(series_map)

    assert captured == []
    correlation = impact["objective_correlation"]["matrix"]
    assert set(correlation) == {"rewards/energy", "rewards/comfort"}
    assert set(correlation["rewards/energy"]) == {"rewards/energy", "rewards/comfort"}


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


def test_constraint_diagnostic_basic():
    series_map = {
        "state/position": _make_series(
            "state/position", [0.0, 500.0, 1000.0, 1500.0]
        ),
        "constraint/is_truncated": _make_series(
            "constraint/is_truncated",
            [0.0, 1.0, 0.0, 1.0],
        ),
        "state/stopping_point_index": _make_series(
            "state/stopping_point_index", [-1.0, 0.0, 1.0, 1.0]
        ),
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


def test_constraint_diagnostic_accepts_new_state_tags():
    series_map = {
        "state/position": _make_series(
            "state/position", [0.0, 500.0, 1000.0, 1500.0]
        ),
        "constraint/is_truncated": _make_series(
            "constraint/is_truncated",
            [0.0, 1.0, 0.0, 1.0],
        ),
        "state/stopping_point_index": _make_series(
            "state/stopping_point_index", [-1.0, 0.0, 1.0, 1.0]
        ),
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
        "state/position": _make_series("state/position", [0.0, 100.0, 300.0, 600.0]),
        "constraint/is_truncated": _make_series(
            "constraint/is_truncated",
            [0.0, 1.0, 0.0, 1.0],
        ),
        "state/stopping_point_index": _make_series(
            "state/stopping_point_index", [-1.0, 0.0, 1.0, 1.0]
        ),
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
        "state/position": _make_series(
            "state/position", [0.0, 100.0, 200.0, 0.0, 120.0, 200.0]
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


def test_evolution_metrics_accepts_new_state_position_tag():
    series_map = {
        "state/episode_id": _make_series("state/episode_id", [0, 0, 0, 1, 1, 1]),
        "state/position": _make_series(
            "state/position", [0.0, 100.0, 200.0, 0.0, 120.0, 200.0]
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


def test_evolution_metrics_maps_step_limit_to_terminal_failure_state():
    series_map = {
        "state/episode_id": _make_series("state/episode_id", [0.0, 0.0, 0.0]),
        "state/position": _make_series("state/position", [0.0, 50.0, 100.0]),
        "constraint/is_truncated": _make_series(
            "constraint/is_truncated",
            [0.0, 0.0, 1.0],
        ),
        "constraint/violation_code": _make_series(
            "constraint/violation_code",
            [0.0, 0.0, 4.0],  # step_limit
        ),
    }

    evolution = compute_evolution_metrics(series_map, episode_window_size=1)

    assert evolution["available"] is True
    assert evolution["state_labels"] == [
        "normal",
        "terminal_failure",
        "speed_violation",
    ]
    terminal_state_ratio = evolution["overall_terminal_state_ratio"]
    assert terminal_state_ratio["terminal_failure"] == pytest.approx(1.0)

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


def test_markdown_best_eval_uses_arrival_layers(tmp_path):
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


class _FakeTensorboardWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_count: int = 0

    def add_scalar(self, key: str, value: float, step: int) -> None:
        self.scalars.append((key, float(value), int(step)))

    def flush(self) -> None:
        self.flush_count += 1


class _FakeTensorboardOutputFormat:
    def __init__(self, writer: _FakeTensorboardWriter):
        self.writer = writer


class _FakeTensorboardLogger:
    def __init__(self, writer: _FakeTensorboardWriter):
        self.output_formats = [_FakeTensorboardOutputFormat(writer)]


class _FakeModel:
    def __init__(self, env: _FakeEnv, logger: Any):
        self._env = env
        self.logger = logger

    def get_env(self):
        return self._env


def _make_callback_locals(
    *,
    truncated: float = 0.0,
    reward_total: float = 1.0,
    position: float | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, dict[str, float]] = {
        "rewards": {"total": reward_total},
        "outcome": {"truncated": truncated},
        "constraint": {},
        "event": {"episode_truncated_count": truncated},
    }
    if position is not None:
        diagnostics["basic"] = {
            "position": position,
            "operation_time": 12.0,
        }

    return {
        "infos": [diagnostics],
        "dones": [False],
    }


def test_callback_sampling_throttle():
    env = _FakeEnv()
    logger = _FakeLogger()
    model = _FakeModel(env, logger)
    callback = TensorboardCallback(tb_sample_interval_steps=3)
    callback.init_callback(cast(Any, model))

    for step in range(1, 7):
        callback.num_timesteps = step
        callback.locals = _make_callback_locals()
        assert callback._on_step() is True

    # step=1 and step=4 will be sampled; empty constraint payloads do not
    # produce a scalar record.
    assert len(logger.records) == 6
    assert all(value != 999.0 for _, value in logger.records)


def test_callback_force_dump_interval():
    env = _FakeEnv()
    logger = _FakeLogger()
    model = _FakeModel(env, logger)
    callback = TensorboardCallback(
        tb_sample_interval_steps=100,
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
    callback = TensorboardCallback(tb_sample_interval_steps=1)
    callback.init_callback(cast(Any, model))

    callback.num_timesteps = 1
    callback.locals = _make_callback_locals(truncated=1.0)
    assert callback._on_step() is True

    records = dict(logger.records)
    assert records["outcome/truncated"] == 1.0
    assert records["event/episode_truncated_count"] == 1.0


def test_callback_reads_runtime_step_diagnostics_from_infos():
    env = _FakeEnv()
    logger = _FakeLogger()
    model = _FakeModel(env, logger)
    callback = TensorboardCallback(tb_sample_interval_steps=1)
    callback.init_callback(cast(Any, model))

    callback.num_timesteps = 1
    callback.locals = _make_callback_locals(position=321.0)
    assert callback._on_step() is True

    records = dict(logger.records)
    assert records["basic/position"] == 321.0
    assert records["basic/operation_time"] == 12.0


def test_callback_event_buffer_preserves_intermediate_steps():
    env = _FakeEnv()
    writer = _FakeTensorboardWriter()
    logger = _FakeTensorboardLogger(writer)
    model = _FakeModel(env, logger)
    callback = TensorboardCallback(
        tb_sample_interval_steps=1,
        batch_dump_records=2,
        force_dump_interval_steps=None,
    )
    callback.init_callback(cast(Any, model))

    callback.num_timesteps = 1
    callback.locals = _make_callback_locals(reward_total=1.0, truncated=0.0)
    assert callback._on_step() is True

    callback.num_timesteps = 2
    callback.locals = _make_callback_locals(reward_total=2.0, truncated=1.0)
    assert callback._on_step() is True

    reward_entries = [entry for entry in writer.scalars if entry[0] == "rewards/total"]
    assert reward_entries == [("rewards/total", 1.0, 1), ("rewards/total", 2.0, 2)]
    assert writer.flush_count == 1


def test_callback_tracks_episode_id_from_dones():
    env = _FakeEnv()
    logger = _FakeLogger()
    model = _FakeModel(env, logger)
    callback = TensorboardCallback(tb_sample_interval_steps=1)
    callback.init_callback(cast(Any, model))

    callback.num_timesteps = 1
    callback.locals = {
        "infos": [
            {
                "rewards": {},
                    "basic": {},
                "constraint": {},
                "event": {},
            }
        ],
        "dones": [False],
    }
    assert callback._on_step() is True

    callback.num_timesteps = 2
    callback.locals = {
        "infos": [
            {
                "rewards": {},
                    "basic": {},
                "constraint": {},
                "event": {},
            }
        ],
        "dones": [True],
    }
    assert callback._on_step() is True

    callback.num_timesteps = 3
    callback.locals = {
        "infos": [
            {
                "rewards": {},
                    "basic": {},
                "constraint": {},
                "event": {},
            }
        ],
        "dones": [False],
    }
    assert callback._on_step() is True

    episode_id_values = [
        value for key, value in logger.records if key == "basic/episode_id"
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
        "basic/episode_id": _make_series_with_steps(
            "basic/episode_id", steps, [10.0, 120.0, 240.0]
        ),
        "basic/position": _make_series_with_steps(
            "basic/position", steps, [0.0, 5000.0, 9000.0]
        ),
        "basic/stopping_point_index": _make_series_with_steps(
            "basic/stopping_point_index", steps, [-1.0, 1.0, 2.0]
        ),
        "outcome/truncated": _make_series_with_steps(
            "outcome/truncated", steps, [0.0, 0.0, 0.0]
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
        min_points_per_10k_steps=1.0,
        min_unique_episodes=3,
        rollout_steps_per_update=100,
    )

    with pytest.raises(ValueError, match="rollout_steps_per_update"):
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
        min_points_per_10k_steps=1.0,
        min_unique_episodes=3,
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


def test_sampling_gate_accepts_rollout_sized_mean_gap(monkeypatch, tmp_path):
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
        min_points_per_10k_steps=1.0,
        min_unique_episodes=3,
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
    args = parser.parse_args([
        "--min-points-per-10k-steps",
        "6.5",
        "--min-unique-episodes",
        "80",
        "--rollout-steps-per-update",
        "8192",
        "--sampling-quality-mode",
        "strict_fail",
    ])

    assert args.min_points_per_10k_steps == 6.5
    assert args.min_unique_episodes == 80
    assert args.rollout_steps_per_update == 8192
    assert args.sampling_quality_mode == "strict_fail"


def test_analyze_cli_rejects_removed_max_mean_step_gap() -> None:
    parser = build_analyze_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--max-mean-step-gap", "1500"])


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
            ("tune", (True, True, True, True, True, True)),
            ("reproduce", (False, False, True, False, False, False)),
            ("monitor_best", (True, False, True, False, False, True)),
            ("best_only", (False, False, False, False, False, True)),
        ],
)
def test_resolve_run_mode_defaults(run_mode: str, expected: tuple[bool, ...]) -> None:
    args = argparse.Namespace(
        run_mode=run_mode,
        enable_tb=None,
        enable_callback=None,
        enable_monitor=None,
        enable_env_diagnostics=None,
        enable_auto_analysis=None,
        enable_best_eval=None,
    )
    (
        _,
        enable_tb,
        enable_callback,
        enable_monitor,
        enable_env_diagnostics,
        enable_auto_analysis,
        enable_best_eval,
    ) = resolve_run_mode(args)

    assert (
        enable_tb,
        enable_callback,
        enable_monitor,
        enable_env_diagnostics,
        enable_auto_analysis,
        enable_best_eval,
    ) == expected


def test_resolve_run_mode_forces_callback_off_when_tb_disabled() -> None:
    args = argparse.Namespace(
        run_mode="tune",
        enable_tb=False,
        enable_callback=True,
        enable_monitor=None,
        enable_env_diagnostics=None,
        enable_auto_analysis=None,
        enable_best_eval=None,
    )
    _, enable_tb, enable_callback, *_rest = resolve_run_mode(args)

    assert enable_tb is False
    assert enable_callback is False


def test_train_rl_cli_rejects_removed_eval_mode() -> None:
    parser = build_train_rl_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-mode", "eval"])


def test_train_rl_cli_accepts_new_run_modes() -> None:
    parser = build_train_rl_arg_parser()
    for mode in ("tune", "reproduce", "monitor_best", "best_only"):
        args = parser.parse_args(["--run-mode", mode])
        assert args.run_mode == mode


def test_train_rl_cli_accepts_reward_profile_and_experiment_tag() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args([
        "--reward-profile",
        "basic_safety_stopping",
        "--experiment-tag",
        "trial_a",
    ])

    assert args.reward_profile == "basic_safety_stopping"
    assert args.experiment_tag == "trial_a"


def test_train_rl_cli_accepts_dry_run() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args([
        "--dry-run",
        "--reward-profile",
        "basic",
    ])

    assert args.dry_run is True
    assert args.reward_profile == "basic"


def test_train_rl_cli_rejects_removed_subproc_start_method_option() -> None:
    parser = build_train_rl_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--subproc-start-method", "spawn"])


def test_resolve_training_run_spec_plans_paths_and_switches() -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args([
        "--run-mode",
        "monitor_best",
        "--reward-profile",
        "basic_safety",
        "--experiment-tag",
        "batch_a",
        "--dry-run",
    ])

    spec = resolve_training_run_spec(args)

    assert spec.run_mode == "monitor_best"
    assert spec.enable_tb is True
    assert spec.enable_callback is False
    assert spec.enable_best_eval is True
    assert spec.reward_profile.name == "basic_safety"
    assert Path(spec.output_dir).name == "430p0_100p0__basic_safety__batch_a"
    assert Path(spec.run_metadata_path).name == "run_metadata.json"
    assert spec.run_metadata["reward_profile_name"] == "basic_safety"
    assert spec.subproc_start_method is None
    assert "subproc_start_method" not in spec.run_metadata
    assert spec.dry_run is True


def test_resolve_training_run_spec_auto_selects_forkserver_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args([
        "--num-envs",
        "2",
        "--vec-env-type",
        "subproc",
        "--dry-run",
    ])
    monkeypatch.setattr(
        "rl.experiment_utils.mp.get_all_start_methods",
        lambda: ["spawn", "forkserver"],
    )

    spec = resolve_training_run_spec(args)

    assert spec.use_subproc is True
    assert spec.subproc_start_method == "forkserver"
    assert spec.run_metadata["subproc_start_method"] == "forkserver"


def test_resolve_training_run_spec_falls_back_to_spawn_when_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = build_train_rl_arg_parser()
    args = parser.parse_args([
        "--num-envs",
        "2",
        "--vec-env-type",
        "subproc",
        "--dry-run",
    ])
    monkeypatch.setattr(
        "rl.experiment_utils.mp.get_all_start_methods",
        lambda: ["spawn"],
    )

    spec = resolve_training_run_spec(args)

    assert spec.use_subproc is True
    assert spec.subproc_start_method == "spawn"
    assert spec.run_metadata["subproc_start_method"] == "spawn"


def test_train_rl_cli_rejects_removed_monitor_log_dir_option() -> None:
    parser = build_train_rl_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--monitor-log-dir", "output/tmp/monitor"])


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
                        "type": "sps_zone_center",
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

