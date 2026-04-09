from __future__ import annotations

from typing import Any

import numpy as np

from .collect import ScalarSeries
from .process import (
    align_tags_to_reference_steps,
    coefficient_of_variation,
    exponential_moving_average,
    linear_slope,
)


DEFAULT_REWARD_COMPONENT_TAGS = [
    "rewards/safety",
    "rewards/energy",
    "rewards/comfort",
    "rewards/punctuality",
    "rewards/docking",
]


def _series_values(
    series_map: dict[str, ScalarSeries], tag: str
) -> tuple[np.ndarray, np.ndarray]:
    series = series_map.get(tag)
    if series is None:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    return series.steps.astype(np.int64), series.values.astype(np.float64)


def _lag1_autocorrelation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return 0.0

    left = values[:-1]
    right = values[1:]
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_regular_training_metrics(
    series_map: dict[str, ScalarSeries],
    ema_alpha: float = 0.1,
    kl_threshold: float = 0.03,
) -> dict[str, Any]:
    regular: dict[str, Any] = {
        "convergence_speed_quality": {"available": False},
        "policy_vitality": {"available": False},
        "critic_foresight": {"available": False},
        "update_safety": {"available": False},
    }

    reward_steps, ep_rew_values = _series_values(series_map, "rollout/ep_rew_mean")
    if ep_rew_values.size > 0:
        ema_values = exponential_moving_average(ep_rew_values, alpha=ema_alpha)
        trend_target = ema_values[0] + 0.9 * (ema_values[-1] - ema_values[0])
        if ema_values[-1] >= ema_values[0]:
            reached = np.where(ema_values >= trend_target)[0]
        else:
            reached = np.where(ema_values <= trend_target)[0]

        regular["convergence_speed_quality"] = {
            "available": True,
            "final_ep_rew_mean": float(ep_rew_values[-1]),
            "ema_final": float(ema_values[-1]),
            "rise_slope_per_step": float(linear_slope(reward_steps, ema_values)),
            "volatility_cv": float(coefficient_of_variation(ep_rew_values)),
            "mean_ep_rew": float(np.mean(ep_rew_values)),
            "std_ep_rew": float(np.std(ep_rew_values)),
            "steps_to_90pct_trend_target": float(
                reward_steps[reached[0]] if reached.size > 0 else reward_steps[-1]
            ),
        }

    entropy_steps, entropy_loss_values = _series_values(
        series_map, "train/entropy_loss"
    )
    if entropy_loss_values.size > 0:
        entropy_proxy = -entropy_loss_values
        low_entropy_threshold = float(np.quantile(entropy_proxy, 0.1))
        early_len = max(1, entropy_proxy.size // 3)
        early_ratio = float(np.mean(entropy_proxy[:early_len] <= low_entropy_threshold))
        full_ratio = float(np.mean(entropy_proxy <= low_entropy_threshold))
        entropy_slope = float(linear_slope(entropy_steps, entropy_proxy))

        regular["policy_vitality"] = {
            "available": True,
            "entropy_proxy_mean": float(np.mean(entropy_proxy)),
            "entropy_proxy_p10": low_entropy_threshold,
            "entropy_trend_slope_per_step": entropy_slope,
            "early_low_entropy_ratio": early_ratio,
            "global_low_entropy_ratio": full_ratio,
            "rigidity_risk_score": float(early_ratio * max(0.0, -entropy_slope)),
        }

    explained_steps, explained_values = _series_values(
        series_map,
        "train/explained_variance",
    )
    if explained_values.size > 0:
        regular["critic_foresight"] = {
            "available": True,
            "explained_variance_mean": float(np.mean(explained_values)),
            "explained_variance_std": float(np.std(explained_values)),
            "explained_variance_min": float(np.min(explained_values)),
            "explained_variance_p10": float(np.quantile(explained_values, 0.1)),
            "low_explained_variance_ratio": float(np.mean(explained_values < 0.3)),
            "negative_explained_variance_ratio": float(np.mean(explained_values < 0.0)),
            "trend_slope_per_step": float(
                linear_slope(explained_steps, explained_values)
            ),
        }

    kl_steps, approx_kl_values = _series_values(series_map, "train/approx_kl")
    if approx_kl_values.size > 0:
        exceed_mask = approx_kl_values > kl_threshold
        regular["update_safety"] = {
            "available": True,
            "approx_kl_mean": float(np.mean(approx_kl_values)),
            "approx_kl_p95": float(np.quantile(approx_kl_values, 0.95)),
            "approx_kl_max": float(np.max(approx_kl_values)),
            "approx_kl_exceed_threshold": float(kl_threshold),
            "approx_kl_exceed_ratio": float(np.mean(exceed_mask)),
            "approx_kl_exceed_count": float(np.sum(exceed_mask)),
            "trend_slope_per_step": float(linear_slope(kl_steps, approx_kl_values)),
        }

    return regular


def compute_reward_component_impact(
    series_map: dict[str, ScalarSeries],
    component_tags: list[str] | None = None,
) -> dict[str, Any]:
    tags = component_tags or DEFAULT_REWARD_COMPONENT_TAGS
    available = [tag for tag in tags if tag in series_map]

    if not available:
        return {
            "available": False,
            "dominance": {},
            "sensitivity": {},
            "objective_correlation": {},
        }

    abs_contribution: dict[str, float] = {}
    for tag in available:
        _, values = _series_values(series_map, tag)
        abs_contribution[tag] = float(np.sum(np.abs(values)))

    abs_total = float(sum(abs_contribution.values()))
    dominance = {
        tag: (value / abs_total if abs_total > 1e-12 else 0.0)
        for tag, value in abs_contribution.items()
    }

    sensitivity: dict[str, dict[str, Any]] = {}
    for tag in available:
        _, values = _series_values(series_map, tag)
        abs_values = np.abs(values)
        trigger_threshold = max(float(np.quantile(abs_values, 0.75)) * 0.5, 1e-6)
        trigger_mask = abs_values >= trigger_threshold
        trigger_frequency = float(np.mean(trigger_mask))
        persistence = _lag1_autocorrelation(values)

        if persistence >= 0.55 and trigger_frequency >= 0.15:
            behavior = "long_term_guidance"
        elif trigger_frequency <= 0.05:
            behavior = "rare_trigger"
        else:
            behavior = "instant_trigger"

        sensitivity[tag] = {
            "trigger_threshold": trigger_threshold,
            "trigger_frequency": trigger_frequency,
            "trigger_strength_mean": float(np.mean(abs_values[trigger_mask]))
            if np.any(trigger_mask)
            else 0.0,
            "persistence_lag1": persistence,
            "behavior": behavior,
        }

    _, aligned = align_tags_to_reference_steps(series_map, available)
    ordered_tags = list(aligned.keys())
    objective_correlation: dict[str, Any] = {"matrix": {}, "strong_negative_pairs": []}

    if len(ordered_tags) >= 2:
        matrix_values = np.column_stack([aligned[tag] for tag in ordered_tags])
        corr_matrix = np.corrcoef(matrix_values, rowvar=False)
        corr_map: dict[str, dict[str, float]] = {}
        strong_negative_pairs: list[dict[str, Any]] = []

        for i, row_tag in enumerate(ordered_tags):
            row_data: dict[str, float] = {}
            for j, col_tag in enumerate(ordered_tags):
                corr = (
                    float(corr_matrix[i, j]) if not np.isnan(corr_matrix[i, j]) else 0.0
                )
                row_data[col_tag] = corr
                if i < j and corr <= -0.4:
                    strong_negative_pairs.append(
                        {"left": row_tag, "right": col_tag, "pearson": corr}
                    )
            corr_map[row_tag] = row_data

        objective_correlation = {
            "matrix": corr_map,
            "strong_negative_pairs": strong_negative_pairs,
        }

    return {
        "available": True,
        "dominance": dominance,
        "absolute_contribution": abs_contribution,
        "sensitivity": sensitivity,
        "objective_correlation": objective_correlation,
    }


def compute_constraint_diagnostic(
    series_map: dict[str, ScalarSeries],
    near_miss_threshold_mps: float = 1.0,
    position_bin_size_m: float = 500.0,
) -> dict[str, Any]:
    required_tags = [
        "state/pos_m",
        "constraint/is_truncated",
        "state/current_sp",
        "constraint/speed_limit_segment",
        "constraint/margin_to_vmax_mps",
        "constraint/margin_to_vmin_mps",
    ]
    if any(tag not in series_map for tag in required_tags):
        return {
            "available": False,
            "geographic_failure_distribution": {},
            "safety_band_tolerance": {},
        }

    _, aligned = align_tags_to_reference_steps(
        series_map,
        required_tags,
        reference_tag="state/pos_m",
    )

    if "state/pos_m" not in aligned:
        return {
            "available": False,
            "geographic_failure_distribution": {},
            "safety_band_tolerance": {},
        }

    positions = aligned["state/pos_m"]
    truncated_flags = aligned["constraint/is_truncated"]
    sp_values = aligned["state/current_sp"]
    speed_segment_values = aligned["constraint/speed_limit_segment"]
    margin_to_vmax = aligned["constraint/margin_to_vmax_mps"]
    margin_to_vmin = aligned["constraint/margin_to_vmin_mps"]

    truncated_mask = truncated_flags >= 0.5
    failure_positions = positions[truncated_mask]
    failure_sp = np.rint(sp_values[truncated_mask]).astype(np.int64)
    failure_segments = np.rint(speed_segment_values[truncated_mask]).astype(np.int64)

    geographic_failure_distribution: dict[str, Any] = {
        "truncated_count": int(np.sum(truncated_mask)),
        "hotspots_by_position_bin": [],
        "hotspots_by_asa_index": {},
        "hotspots_by_speed_limit_segment": {},
    }

    if failure_positions.size > 0:
        bin_size = max(1.0, float(position_bin_size_m))
        pos_min = float(np.floor(np.min(positions) / bin_size) * bin_size)
        pos_max = float(np.ceil(np.max(positions) / bin_size) * bin_size + bin_size)
        bins = np.arange(pos_min, pos_max + bin_size, bin_size)
        histogram, edges = np.histogram(failure_positions, bins=bins)

        hotspots = []
        for idx, count in enumerate(histogram):
            if count <= 0:
                continue
            hotspots.append(
                {
                    "bin_start_m": float(edges[idx]),
                    "bin_end_m": float(edges[idx + 1]),
                    "count": int(count),
                }
            )

        asa_counts = {
            str(int(sp)): int(np.sum(failure_sp == sp))
            for sp in np.unique(failure_sp)
            if sp >= 0
        }
        segment_counts = {
            str(int(segment)): int(np.sum(failure_segments == segment))
            for segment in np.unique(failure_segments)
            if segment >= 0
        }

        geographic_failure_distribution = {
            "truncated_count": int(np.sum(truncated_mask)),
            "hotspots_by_position_bin": hotspots,
            "hotspots_by_asa_index": asa_counts,
            "hotspots_by_speed_limit_segment": segment_counts,
        }

    finite_margin_mask = np.isfinite(margin_to_vmax) & np.isfinite(margin_to_vmin)
    if not np.any(finite_margin_mask):
        safety_band_tolerance = {
            "available": False,
        }
    else:
        vmax_margin = margin_to_vmax[finite_margin_mask]
        vmin_margin = margin_to_vmin[finite_margin_mask]
        near_miss_mask = (vmax_margin <= near_miss_threshold_mps) | (
            vmin_margin <= near_miss_threshold_mps
        )
        violation_mask = (vmax_margin < 0.0) | (vmin_margin < 0.0)

        safety_band_tolerance = {
            "available": True,
            "average_distance_to_vmax_mps": float(np.mean(vmax_margin)),
            "average_distance_to_vmin_mps": float(np.mean(vmin_margin)),
            "p05_distance_to_vmax_mps": float(np.quantile(vmax_margin, 0.05)),
            "p05_distance_to_vmin_mps": float(np.quantile(vmin_margin, 0.05)),
            "min_distance_to_vmax_mps": float(np.min(vmax_margin)),
            "min_distance_to_vmin_mps": float(np.min(vmin_margin)),
            "near_miss_threshold_mps": float(near_miss_threshold_mps),
            "near_miss_ratio": float(np.mean(near_miss_mask)),
            "violation_ratio": float(np.mean(violation_mask)),
            "sample_count": int(vmax_margin.size),
        }

    return {
        "available": True,
        "geographic_failure_distribution": geographic_failure_distribution,
        "safety_band_tolerance": safety_band_tolerance,
    }
