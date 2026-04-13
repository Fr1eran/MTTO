from __future__ import annotations

from typing import Any

import numpy as np

from utils.data_loader import load_auxiliary_stopping_areas_ap_and_dp, load_slopes

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

VIOLATION_STATE_LABELS = ["normal", "low_violation", "high_violation"]


def _series_values(
    series_map: dict[str, ScalarSeries], tag: str
) -> tuple[np.ndarray, np.ndarray]:
    series = series_map.get(tag)
    if series is None:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    return series.steps.astype(np.int64), series.values.astype(np.float64)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


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


def _row_normalized(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return matrix

    row_sums = np.sum(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums > 0,
    )


def _episode_ids_for_reference_steps(
    series_map: dict[str, ScalarSeries],
    reference_steps: np.ndarray,
    episode_tag: str = "state/episode_id",
) -> np.ndarray:
    episode_series = series_map.get(episode_tag)
    if episode_series is None or reference_steps.size == 0:
        return np.asarray([], dtype=np.int64)

    episode_steps = episode_series.steps.astype(np.int64)
    episode_ids = np.rint(episode_series.values).astype(np.int64)
    if episode_steps.size == 0:
        return np.asarray([], dtype=np.int64)

    indices = np.searchsorted(episode_steps, reference_steps, side="right") - 1
    indices = np.clip(indices, 0, episode_steps.size - 1)
    return episode_ids[indices]


def _build_episode_windows(
    unique_episode_ids: np.ndarray,
    window_size: int,
) -> list[tuple[int, int, int, np.ndarray]]:
    if unique_episode_ids.size == 0:
        return []

    window = max(1, int(window_size))
    min_episode = int(np.min(unique_episode_ids))
    max_episode = int(np.max(unique_episode_ids))
    windows: list[tuple[int, int, int, np.ndarray]] = []
    window_index = 0

    for episode_start in range(min_episode, max_episode + 1, window):
        episode_end = episode_start + window
        members = unique_episode_ids[
            (unique_episode_ids >= episode_start) & (unique_episode_ids < episode_end)
        ]
        if members.size == 0:
            continue
        windows.append((window_index, episode_start, episode_end, members))
        window_index += 1

    return windows


def _to_violation_states(violation_codes: np.ndarray) -> np.ndarray:
    rounded = np.rint(violation_codes).astype(np.int64)
    states = np.zeros_like(rounded, dtype=np.int64)
    states[rounded == 1] = 1
    states[rounded >= 2] = 2
    return states


def _compute_component_sensitivity(
    series_values_by_tag: dict[str, np.ndarray],
    ordered_tags: list[str],
) -> dict[str, dict[str, Any]]:
    sensitivity: dict[str, dict[str, Any]] = {}

    for tag in ordered_tags:
        values = np.asarray(series_values_by_tag[tag], dtype=np.float64)
        if values.size == 0:
            sensitivity[tag] = {
                "trigger_threshold": 0.0,
                "trigger_frequency": 0.0,
                "trigger_strength_mean": 0.0,
                "persistence_lag1": 0.0,
                "behavior": "rare_trigger",
            }
            continue

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

    return sensitivity


def _compute_objective_correlation(
    series_values_by_tag: dict[str, np.ndarray],
    ordered_tags: list[str],
) -> dict[str, Any]:
    if len(ordered_tags) < 2:
        return {"matrix": {}, "strong_negative_pairs": []}

    first = series_values_by_tag[ordered_tags[0]]
    if np.asarray(first).size < 2:
        return {"matrix": {}, "strong_negative_pairs": []}

    matrix_values = np.column_stack([series_values_by_tag[tag] for tag in ordered_tags])
    corr_matrix = np.corrcoef(matrix_values, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    corr_map: dict[str, dict[str, float]] = {}
    strong_negative_pairs: list[dict[str, Any]] = []

    for i, row_tag in enumerate(ordered_tags):
        row_data: dict[str, float] = {}
        for j, col_tag in enumerate(ordered_tags):
            corr = float(corr_matrix[i, j])
            row_data[col_tag] = corr
            if i < j and corr <= -0.4:
                strong_negative_pairs.append(
                    {"left": row_tag, "right": col_tag, "pearson": corr}
                )
        corr_map[row_tag] = row_data

    return {"matrix": corr_map, "strong_negative_pairs": strong_negative_pairs}


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


def _compute_reward_component_impact_step_based(
    series_map: dict[str, ScalarSeries],
    component_tags: list[str],
) -> dict[str, Any]:
    abs_contribution: dict[str, float] = {}
    for tag in component_tags:
        _, values = _series_values(series_map, tag)
        abs_contribution[tag] = float(np.sum(np.abs(values)))

    abs_total = float(sum(abs_contribution.values()))
    dominance = {
        tag: _safe_ratio(value, abs_total) for tag, value in abs_contribution.items()
    }

    values_by_tag = {tag: _series_values(series_map, tag)[1] for tag in component_tags}
    sensitivity = _compute_component_sensitivity(values_by_tag, component_tags)

    _, aligned = align_tags_to_reference_steps(series_map, component_tags)
    objective_correlation = _compute_objective_correlation(
        aligned, list(aligned.keys())
    )

    return {
        "available": True,
        "aggregation_order": "step_only_fallback",
        "dominance": dominance,
        "absolute_contribution": abs_contribution,
        "sensitivity": sensitivity,
        "objective_correlation": objective_correlation,
        "episode_count": 0,
        "stage_count": 0,
        "episode_component_summary": {},
        "stage_component_profile": [],
    }


def compute_reward_component_impact(
    series_map: dict[str, ScalarSeries],
    component_tags: list[str] | None = None,
    episode_window_size: int = 20,
    episode_tag: str = "state/episode_id",
) -> dict[str, Any]:
    tags = component_tags or DEFAULT_REWARD_COMPONENT_TAGS
    available = [tag for tag in tags if tag in series_map]

    if not available:
        return {
            "available": False,
            "dominance": {},
            "sensitivity": {},
            "objective_correlation": {},
            "episode_component_summary": {},
            "stage_component_profile": [],
        }

    reference_steps, aligned = align_tags_to_reference_steps(series_map, available)
    if reference_steps.size == 0 or not aligned:
        return _compute_reward_component_impact_step_based(series_map, available)

    episode_ids = _episode_ids_for_reference_steps(
        series_map,
        reference_steps,
        episode_tag=episode_tag,
    )
    valid_mask = episode_ids >= 0
    if episode_ids.size == 0 or not np.any(valid_mask):
        return _compute_reward_component_impact_step_based(series_map, available)

    filtered_episode_ids = episode_ids[valid_mask]
    filtered_values_by_tag = {
        tag: np.asarray(aligned[tag], dtype=np.float64)[valid_mask] for tag in available
    }

    unique_episodes = np.unique(filtered_episode_ids)
    if unique_episodes.size == 0:
        return _compute_reward_component_impact_step_based(series_map, available)

    episode_totals: dict[str, np.ndarray] = {
        tag: np.zeros(unique_episodes.size, dtype=np.float64) for tag in available
    }

    for idx, episode_id in enumerate(unique_episodes):
        episode_mask = filtered_episode_ids == episode_id
        for tag in available:
            episode_totals[tag][idx] = float(
                np.sum(filtered_values_by_tag[tag][episode_mask])
            )

    abs_total_per_episode = np.zeros(unique_episodes.size, dtype=np.float64)
    for tag in available:
        abs_total_per_episode += np.abs(episode_totals[tag])

    episode_ratio_by_tag: dict[str, np.ndarray] = {}
    for tag in available:
        episode_ratio_by_tag[tag] = np.divide(
            np.abs(episode_totals[tag]),
            abs_total_per_episode,
            out=np.zeros_like(abs_total_per_episode),
            where=abs_total_per_episode > 1e-12,
        )

    abs_contribution = {
        tag: float(np.sum(np.abs(episode_totals[tag]))) for tag in available
    }
    abs_total = float(sum(abs_contribution.values()))
    dominance = {
        tag: _safe_ratio(value, abs_total) for tag, value in abs_contribution.items()
    }

    sensitivity = _compute_component_sensitivity(episode_totals, available)
    objective_correlation = _compute_objective_correlation(episode_totals, available)

    episode_component_summary: dict[str, dict[str, float]] = {}
    for tag in available:
        episode_component_summary[tag] = {
            "mean_cumulative": float(np.mean(episode_totals[tag])),
            "std_cumulative": float(np.std(episode_totals[tag])),
            "mean_ratio": float(np.mean(episode_ratio_by_tag[tag])),
            "p95_abs_cumulative": float(np.quantile(np.abs(episode_totals[tag]), 0.95)),
        }

    stage_component_profile: list[dict[str, Any]] = []
    for window_index, episode_start, episode_end, members in _build_episode_windows(
        unique_episodes,
        episode_window_size,
    ):
        member_mask = np.isin(unique_episodes, members)
        mean_cumulative = {
            tag: float(np.mean(episode_totals[tag][member_mask])) for tag in available
        }
        std_cumulative = {
            tag: float(np.std(episode_totals[tag][member_mask])) for tag in available
        }
        mean_ratio = {
            tag: float(np.mean(episode_ratio_by_tag[tag][member_mask]))
            for tag in available
        }

        stage_component_profile.append(
            {
                "window_index": window_index,
                "episode_start": int(episode_start),
                "episode_end": int(episode_end),
                "episode_count": int(np.sum(member_mask)),
                "mean_cumulative": mean_cumulative,
                "std_cumulative": std_cumulative,
                "mean_ratio": mean_ratio,
            }
        )

    return {
        "available": True,
        "aggregation_order": "episode_then_stage",
        "episode_count": int(unique_episodes.size),
        "stage_count": int(len(stage_component_profile)),
        "dominance": dominance,
        "absolute_contribution": abs_contribution,
        "sensitivity": sensitivity,
        "objective_correlation": objective_correlation,
        "episode_component_summary": episode_component_summary,
        "stage_component_profile": stage_component_profile,
    }


def _load_critical_points() -> dict[str, np.ndarray]:
    slope_points = np.asarray([], dtype=np.float64)
    sps_switch_points = np.asarray([], dtype=np.float64)

    try:
        _, slope_intervals = load_slopes()
        slope_intervals = np.asarray(slope_intervals, dtype=np.float64)
        if slope_intervals.size >= 3:
            slope_points = np.unique(slope_intervals[1:-1])

        aps, dps = load_auxiliary_stopping_areas_ap_and_dp()
        if len(aps) == len(dps) and len(aps) > 0:
            sps_switch_points = np.asarray(
                [(float(ap) + float(dp)) / 2.0 for ap, dp in zip(aps, dps)],
                dtype=np.float64,
            )
    except Exception:
        return {
            "slope_transition_points_m": slope_points,
            "sps_switch_points_m": sps_switch_points,
        }

    return {
        "slope_transition_points_m": slope_points,
        "sps_switch_points_m": sps_switch_points,
    }


def _build_critical_point_entries(
    *,
    positions: np.ndarray,
    failure_mask: np.ndarray,
    violation_mask: np.ndarray,
    near_miss_mask: np.ndarray,
    points_m: np.ndarray,
    point_type: str,
    radius_m: float,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if points_m.size == 0:
        return entries

    radius = max(float(radius_m), 1.0)
    for point in points_m:
        near_mask = np.abs(positions - point) <= radius
        exposure_count = int(np.sum(near_mask))
        if exposure_count <= 0:
            continue

        failure_count = int(np.sum(failure_mask & near_mask))
        violation_count = int(np.sum(violation_mask & near_mask))
        near_miss_count = int(np.sum(near_miss_mask & near_mask))
        failure_risk = _safe_ratio(float(failure_count), float(exposure_count))
        violation_risk = _safe_ratio(float(violation_count), float(exposure_count))
        near_miss_risk = _safe_ratio(float(near_miss_count), float(exposure_count))

        entries.append(
            {
                "type": point_type,
                "point_m": float(point),
                "radius_m": radius,
                "exposure_count": exposure_count,
                "near_miss_count": near_miss_count,
                "failure_count": failure_count,
                "violation_count": violation_count,
                "near_miss_risk": near_miss_risk,
                "violation_risk": violation_risk,
                "failure_risk": failure_risk,
                "risk": failure_risk,
            }
        )

    entries.sort(
        key=lambda item: (
            item["failure_risk"],
            item["violation_risk"],
            item["near_miss_risk"],
            item["failure_count"],
            item["violation_count"],
            item["near_miss_count"],
        ),
        reverse=True,
    )
    return entries


def _build_episode_boundary_profiles(
    *,
    episode_ids: np.ndarray,
    distance_weights: np.ndarray,
    near_miss_mask: np.ndarray,
    episode_window_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_mask = episode_ids >= 0
    if not np.any(valid_mask):
        return [], []

    filtered_episode_ids = episode_ids[valid_mask]
    filtered_distance = distance_weights[valid_mask]
    filtered_near_miss = near_miss_mask[valid_mask]
    unique_episodes = np.unique(filtered_episode_ids)

    episode_profile: list[dict[str, Any]] = []
    episode_ratio_map: dict[int, float] = {}

    for episode_id in unique_episodes:
        mask = filtered_episode_ids == episode_id
        total_distance = float(np.sum(filtered_distance[mask]))
        near_miss_distance = float(
            np.sum(filtered_distance[mask] * filtered_near_miss[mask])
        )
        ratio = _safe_ratio(near_miss_distance, total_distance)
        episode_ratio_map[int(episode_id)] = ratio
        episode_profile.append(
            {
                "episode_id": int(episode_id),
                "total_distance_m": total_distance,
                "near_miss_distance_m": near_miss_distance,
                "near_miss_distance_ratio": ratio,
            }
        )

    stage_profile: list[dict[str, Any]] = []
    for window_index, episode_start, episode_end, members in _build_episode_windows(
        unique_episodes,
        episode_window_size,
    ):
        member_ratios = [episode_ratio_map[int(ep)] for ep in members]
        stage_profile.append(
            {
                "window_index": window_index,
                "episode_start": int(episode_start),
                "episode_end": int(episode_end),
                "episode_count": int(len(member_ratios)),
                "mean_near_miss_distance_ratio": float(np.mean(member_ratios))
                if member_ratios
                else 0.0,
                "max_near_miss_distance_ratio": float(np.max(member_ratios))
                if member_ratios
                else 0.0,
            }
        )

    return episode_profile, stage_profile


def compute_constraint_diagnostic(
    series_map: dict[str, ScalarSeries],
    near_miss_threshold_mps: float = 1.0,
    position_bin_size_m: float = 500.0,
    critical_point_radius_m: float = 300.0,
    top_k_spatial_bins: int = 8,
    top_k_critical_points: int = 8,
    episode_window_size: int = 20,
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
            "boundary_adhesion": {},
            "critical_point_risk": {},
        }

    optional_tags = []
    if "constraint/violation_code" in series_map:
        optional_tags.append("constraint/violation_code")
    if "state/episode_id" in series_map:
        optional_tags.append("state/episode_id")

    _, aligned = align_tags_to_reference_steps(
        series_map,
        required_tags + optional_tags,
        reference_tag="state/pos_m",
    )

    if "state/pos_m" not in aligned:
        return {
            "available": False,
            "geographic_failure_distribution": {},
            "safety_band_tolerance": {},
            "boundary_adhesion": {},
            "critical_point_risk": {},
        }

    positions = aligned["state/pos_m"]
    truncated_flags = aligned["constraint/is_truncated"]
    sp_values = aligned["state/current_sp"]
    speed_segment_values = aligned["constraint/speed_limit_segment"]
    margin_to_vmax = aligned["constraint/margin_to_vmax_mps"]
    margin_to_vmin = aligned["constraint/margin_to_vmin_mps"]
    violation_codes = aligned.get("constraint/violation_code", np.zeros_like(positions))
    episode_ids = np.rint(
        aligned.get("state/episode_id", np.full_like(positions, -1.0))
    ).astype(np.int64)

    if positions.size <= 1:
        distance_weights = np.zeros_like(positions, dtype=np.float64)
    else:
        distance_weights = np.abs(np.diff(positions, append=positions[-1]))

    truncated_mask = truncated_flags >= 0.5
    failure_positions = positions[truncated_mask]
    failure_sp = np.rint(sp_values[truncated_mask]).astype(np.int64)
    failure_segments = np.rint(speed_segment_values[truncated_mask]).astype(np.int64)

    margin_violation_mask = (margin_to_vmax < 0.0) | (margin_to_vmin < 0.0)
    violation_mask = (
        np.rint(violation_codes).astype(np.int64) > 0
    ) | margin_violation_mask
    near_miss_event_mask = (
        np.isfinite(margin_to_vmax)
        & np.isfinite(margin_to_vmin)
        & (
            (margin_to_vmax <= near_miss_threshold_mps)
            | (margin_to_vmin <= near_miss_threshold_mps)
        )
    )

    geographic_failure_distribution: dict[str, Any] = {
        "truncated_count": int(np.sum(truncated_mask)),
        "hotspots_by_position_bin": [],
        "hotspots_by_asa_index": {},
        "hotspots_by_speed_limit_segment": {},
        "position_risk_bins": [],
        "top_risk_bins": [],
    }

    if positions.size > 0:
        bin_size = max(1.0, float(position_bin_size_m))
        pos_min = float(np.floor(np.min(positions) / bin_size) * bin_size)
        pos_max = float(np.ceil(np.max(positions) / bin_size) * bin_size + bin_size)
        bins = np.arange(pos_min, pos_max + bin_size, bin_size)

        exposure_hist, edges = np.histogram(positions, bins=bins)
        failure_hist, _ = np.histogram(failure_positions, bins=bins)
        violation_hist, _ = np.histogram(positions[violation_mask], bins=bins)
        near_miss_hist, _ = np.histogram(positions[near_miss_event_mask], bins=bins)

        hotspots = []
        risk_bins: list[dict[str, Any]] = []
        for idx, exposure_count in enumerate(exposure_hist):
            failure_count = int(failure_hist[idx])
            violation_count = int(violation_hist[idx])
            near_miss_count = int(near_miss_hist[idx])
            if failure_count > 0:
                hotspots.append(
                    {
                        "bin_start_m": float(edges[idx]),
                        "bin_end_m": float(edges[idx + 1]),
                        "count": failure_count,
                    }
                )

            if exposure_count <= 0:
                continue

            failure_risk = _safe_ratio(float(failure_count), float(exposure_count))
            violation_risk = _safe_ratio(float(violation_count), float(exposure_count))
            near_miss_risk = _safe_ratio(float(near_miss_count), float(exposure_count))
            risk_bins.append(
                {
                    "bin_start_m": float(edges[idx]),
                    "bin_end_m": float(edges[idx + 1]),
                    "exposure_count": int(exposure_count),
                    "near_miss_count": near_miss_count,
                    "failure_count": failure_count,
                    "violation_count": violation_count,
                    "near_miss_risk": near_miss_risk,
                    "violation_risk": violation_risk,
                    "failure_risk": failure_risk,
                }
            )

        risk_bins.sort(
            key=lambda item: (
                item["failure_risk"],
                item["violation_risk"],
                item["near_miss_risk"],
                item["failure_count"],
                item["violation_count"],
                item["near_miss_count"],
            ),
            reverse=True,
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
            "position_risk_bins": risk_bins,
            "top_risk_bins": risk_bins[: max(1, int(top_k_spatial_bins))],
        }

    finite_margin_mask = np.isfinite(margin_to_vmax) & np.isfinite(margin_to_vmin)
    near_miss_mask = np.zeros_like(margin_to_vmax, dtype=np.float64)

    if not np.any(finite_margin_mask):
        safety_band_tolerance = {
            "available": False,
        }
    else:
        vmax_margin = margin_to_vmax[finite_margin_mask]
        vmin_margin = margin_to_vmin[finite_margin_mask]
        near_miss_local_mask = (vmax_margin <= near_miss_threshold_mps) | (
            vmin_margin <= near_miss_threshold_mps
        )
        near_miss_mask[finite_margin_mask] = near_miss_local_mask.astype(np.float64)
        violation_local_mask = (vmax_margin < 0.0) | (vmin_margin < 0.0)

        total_distance = float(np.sum(distance_weights[finite_margin_mask]))
        near_miss_distance = float(
            np.sum(distance_weights[finite_margin_mask] * near_miss_local_mask)
        )

        safety_band_tolerance = {
            "available": True,
            "average_distance_to_vmax_mps": float(np.mean(vmax_margin)),
            "average_distance_to_vmin_mps": float(np.mean(vmin_margin)),
            "p05_distance_to_vmax_mps": float(np.quantile(vmax_margin, 0.05)),
            "p05_distance_to_vmin_mps": float(np.quantile(vmin_margin, 0.05)),
            "min_distance_to_vmax_mps": float(np.min(vmax_margin)),
            "min_distance_to_vmin_mps": float(np.min(vmin_margin)),
            "near_miss_threshold_mps": float(near_miss_threshold_mps),
            "near_miss_ratio": float(np.mean(near_miss_local_mask)),
            "violation_ratio": float(np.mean(violation_local_mask)),
            "sample_count": int(vmax_margin.size),
            "near_miss_distance_m": near_miss_distance,
            "total_distance_m": total_distance,
            "near_miss_distance_ratio": _safe_ratio(near_miss_distance, total_distance),
        }

    total_distance_all = float(np.sum(distance_weights))
    near_miss_distance_all = float(np.sum(distance_weights * near_miss_mask))
    boundary_ratio_by_distance = _safe_ratio(near_miss_distance_all, total_distance_all)

    episode_boundary_profile, stage_boundary_profile = _build_episode_boundary_profiles(
        episode_ids=episode_ids,
        distance_weights=distance_weights,
        near_miss_mask=near_miss_mask,
        episode_window_size=episode_window_size,
    )

    boundary_adhesion = {
        "near_miss_threshold_mps": float(near_miss_threshold_mps),
        "total_distance_m": total_distance_all,
        "near_miss_distance_m": near_miss_distance_all,
        "near_miss_distance_ratio": boundary_ratio_by_distance,
        "episode_profile": episode_boundary_profile,
        "stage_profile": stage_boundary_profile,
    }

    critical_points = _load_critical_points()
    radius_m = max(1.0, float(critical_point_radius_m))
    slope_entries = _build_critical_point_entries(
        positions=positions,
        failure_mask=truncated_mask,
        violation_mask=violation_mask,
        near_miss_mask=near_miss_event_mask,
        points_m=critical_points["slope_transition_points_m"],
        point_type="slope_transition",
        radius_m=radius_m,
    )
    sps_entries = _build_critical_point_entries(
        positions=positions,
        failure_mask=truncated_mask,
        violation_mask=violation_mask,
        near_miss_mask=near_miss_event_mask,
        points_m=critical_points["sps_switch_points_m"],
        point_type="sps_switch",
        radius_m=radius_m,
    )

    combined_entries = sorted(
        slope_entries + sps_entries,
        key=lambda item: (
            item["failure_risk"],
            item["violation_risk"],
            item["near_miss_risk"],
            item["failure_count"],
            item["violation_count"],
            item["near_miss_count"],
        ),
        reverse=True,
    )
    top_k_points = max(1, int(top_k_critical_points))

    critical_point_risk = {
        "radius_m": radius_m,
        "slope_transition_points": slope_entries[:top_k_points],
        "sps_switch_points": sps_entries[:top_k_points],
        "top_risky_points": combined_entries[:top_k_points],
    }

    return {
        "available": True,
        "geographic_failure_distribution": geographic_failure_distribution,
        "safety_band_tolerance": safety_band_tolerance,
        "boundary_adhesion": boundary_adhesion,
        "critical_point_risk": critical_point_risk,
    }


def compute_evolution_metrics(
    series_map: dict[str, ScalarSeries],
    episode_window_size: int = 20,
) -> dict[str, Any]:
    required_tags = [
        "state/episode_id",
        "state/pos_m",
        "constraint/is_truncated",
        "constraint/violation_code",
    ]
    if any(tag not in series_map for tag in required_tags):
        return {
            "available": False,
            "state_labels": VIOLATION_STATE_LABELS,
            "stage_profiles": [],
        }

    reference_steps, aligned = align_tags_to_reference_steps(
        series_map,
        required_tags,
        reference_tag="state/episode_id",
    )
    if any(tag not in aligned for tag in required_tags):
        return {
            "available": False,
            "state_labels": VIOLATION_STATE_LABELS,
            "stage_profiles": [],
        }

    episode_ids = np.rint(aligned["state/episode_id"]).astype(np.int64)
    positions = aligned["state/pos_m"]
    truncated_flags = aligned["constraint/is_truncated"] >= 0.5
    violation_states = _to_violation_states(aligned["constraint/violation_code"])

    valid_mask = episode_ids >= 0
    if not np.any(valid_mask):
        return {
            "available": False,
            "state_labels": VIOLATION_STATE_LABELS,
            "stage_profiles": [],
        }

    episode_ids = episode_ids[valid_mask]
    positions = positions[valid_mask]
    truncated_flags = truncated_flags[valid_mask]
    violation_states = violation_states[valid_mask]
    reference_steps = reference_steps[valid_mask]

    unique_episodes = np.unique(episode_ids)
    episode_records: list[dict[str, Any]] = []
    terminal_state_counts = np.zeros(3, dtype=np.int64)
    global_start_pos = float(np.min(positions)) if positions.size > 0 else 0.0

    for episode_id in unique_episodes:
        episode_mask = episode_ids == episode_id
        ep_steps = reference_steps[episode_mask]
        ep_positions = positions[episode_mask]
        ep_truncated = truncated_flags[episode_mask]
        ep_states = violation_states[episode_mask]

        if ep_steps.size == 0:
            continue

        trunc_indices = np.where(ep_truncated)[0]
        if trunc_indices.size > 0:
            end_idx = int(trunc_indices[0])
            is_truncated_episode = True
        else:
            end_idx = int(ep_positions.size - 1)
            is_truncated_episode = False

        start_pos = float(ep_positions[0])
        end_pos = float(ep_positions[end_idx])
        if ep_positions.size <= 1:
            # Sparse logger points: fallback to distance from route origin estimate.
            survival_distance = abs(end_pos - global_start_pos)
        else:
            survival_distance = abs(end_pos - start_pos)
        terminal_state = int(ep_states[end_idx])

        transition_matrix = np.zeros((3, 3), dtype=np.int64)
        if ep_states.size >= 2:
            for left, right in zip(ep_states[:-1], ep_states[1:]):
                transition_matrix[int(left), int(right)] += 1

        terminal_state_counts[terminal_state] += 1

        episode_records.append(
            {
                "episode_id": int(episode_id),
                "step_start": int(ep_steps[0]),
                "step_end": int(ep_steps[-1]) + 1,
                "survival_distance_m": float(survival_distance),
                "is_truncated": bool(is_truncated_episode),
                "terminal_state": terminal_state,
                "transition_matrix": transition_matrix,
            }
        )

    if not episode_records:
        return {
            "available": False,
            "state_labels": VIOLATION_STATE_LABELS,
            "stage_profiles": [],
        }

    overall_transition = np.zeros((3, 3), dtype=np.int64)
    if violation_states.size >= 2:
        for left, right in zip(violation_states[:-1], violation_states[1:]):
            overall_transition[int(left), int(right)] += 1

    episode_array = np.asarray(
        [r["episode_id"] for r in episode_records], dtype=np.int64
    )
    survival_array = np.asarray(
        [r["survival_distance_m"] for r in episode_records], dtype=np.float64
    )
    truncated_array = np.asarray(
        [r["is_truncated"] for r in episode_records], dtype=bool
    )
    terminal_state_array = np.asarray(
        [r["terminal_state"] for r in episode_records], dtype=np.int64
    )
    step_start_array = np.asarray(
        [r["step_start"] for r in episode_records], dtype=np.int64
    )
    step_end_array = np.asarray(
        [r["step_end"] for r in episode_records], dtype=np.int64
    )
    transition_stack = np.stack(
        [np.asarray(r["transition_matrix"], dtype=np.int64) for r in episode_records],
        axis=0,
    )

    stage_profiles: list[dict[str, Any]] = []
    previous_avg_survival: float | None = None

    for window_index, episode_start, episode_end, members in _build_episode_windows(
        np.unique(episode_array),
        episode_window_size,
    ):
        member_mask = np.isin(episode_array, members)
        if not np.any(member_mask):
            continue

        stage_survival = survival_array[member_mask]
        stage_truncated = truncated_array[member_mask]
        stage_terminal_states = terminal_state_array[member_mask]
        stage_matrix = np.sum(transition_stack[member_mask], axis=0)
        if int(np.sum(stage_matrix)) == 0:
            stage_indices = np.where(np.isin(episode_ids, members))[0]
            if stage_indices.size >= 2:
                for left_idx, right_idx in zip(stage_indices[:-1], stage_indices[1:]):
                    if right_idx != left_idx + 1:
                        continue
                    left_state = int(violation_states[left_idx])
                    right_state = int(violation_states[right_idx])
                    stage_matrix[left_state, right_state] += 1
        stage_probs = _row_normalized(stage_matrix)

        avg_survival = float(np.mean(stage_survival))
        growth_vs_prev = (
            None
            if previous_avg_survival is None
            else _safe_ratio(
                avg_survival - previous_avg_survival, previous_avg_survival
            )
        )

        state_counts = np.bincount(stage_terminal_states, minlength=3)
        state_total = float(np.sum(state_counts))
        terminal_state_ratio = {
            VIOLATION_STATE_LABELS[idx]: _safe_ratio(
                float(state_counts[idx]), state_total
            )
            for idx in range(3)
        }

        stage_profiles.append(
            {
                "window_index": int(window_index),
                "episode_start": int(episode_start),
                "episode_end": int(episode_end),
                "episode_count": int(np.sum(member_mask)),
                "step_start": int(np.min(step_start_array[member_mask])),
                "step_end": int(np.max(step_end_array[member_mask])),
                "avg_survival_distance_m": avg_survival,
                "survival_growth_rate_vs_prev": growth_vs_prev,
                "truncated_episode_ratio": float(np.mean(stage_truncated)),
                "terminal_state_ratio": terminal_state_ratio,
                "transition_matrix": stage_matrix.astype(np.int64).tolist(),
                "transition_probabilities": stage_probs.tolist(),
                "normal_to_low_transition_rate": float(stage_probs[0, 1]),
                "normal_to_high_transition_rate": float(stage_probs[0, 2]),
                "low_to_high_transition_rate": float(stage_probs[1, 2]),
                "high_to_low_transition_rate": float(stage_probs[2, 1]),
            }
        )

        previous_avg_survival = avg_survival

    avg_survival_values = np.asarray(
        [stage["avg_survival_distance_m"] for stage in stage_profiles], dtype=np.float64
    )
    if avg_survival_values.size >= 2:
        stage_index = np.arange(avg_survival_values.size, dtype=np.float64)
        survival_slope = float(np.polyfit(stage_index, avg_survival_values, 1)[0])
    else:
        survival_slope = 0.0

    overall_probabilities = _row_normalized(overall_transition)
    total_terminal = float(np.sum(terminal_state_counts))
    overall_terminal_state_ratio = {
        VIOLATION_STATE_LABELS[idx]: _safe_ratio(
            float(terminal_state_counts[idx]),
            total_terminal,
        )
        for idx in range(3)
    }

    return {
        "available": True,
        "aggregation_order": "episode_then_stage",
        "stage_basis": "episode_window",
        "episode_window_size": max(1, int(episode_window_size)),
        "state_labels": VIOLATION_STATE_LABELS,
        "episode_count": int(len(episode_records)),
        "stage_count": int(len(stage_profiles)),
        "mean_survival_distance_m": float(np.mean(survival_array)),
        "truncated_episode_ratio": float(np.mean(truncated_array)),
        "survival_distance_slope_per_stage": survival_slope,
        "overall_transition_matrix": overall_transition.astype(np.int64).tolist(),
        "overall_transition_probabilities": overall_probabilities.tolist(),
        "overall_terminal_state_ratio": overall_terminal_state_ratio,
        "stage_profiles": stage_profiles,
    }
