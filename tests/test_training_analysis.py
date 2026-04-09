import numpy as np

from rl.training_analysis.analyze import (
    compute_constraint_diagnostic,
    compute_regular_training_metrics,
    compute_reward_component_impact,
)
from rl.training_analysis.collect import ScalarSeries
from rl.training_analysis.process import build_episode_snapshots


def _make_series(tag: str, values: list[float], step_start: int = 0) -> ScalarSeries:
    steps = np.arange(step_start, step_start + len(values), dtype=np.int64)
    vals = np.asarray(values, dtype=np.float64)
    wall_times = np.asarray(steps, dtype=np.float64)
    return ScalarSeries(tag=tag, steps=steps, values=vals, wall_times=wall_times)


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
