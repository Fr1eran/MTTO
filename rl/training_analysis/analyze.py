from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .collect import RewardDiagnosticsArtifact, ScalarSeries
from .process import (
    coefficient_of_variation,
    exponential_moving_average,
    linear_slope,
)


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


BEST_EVAL_TAGS = [
    "best_eval/last_success",
    "best_eval/last_precise_arrival",
    "best_eval/last_punctual_arrival",
    "best_eval/last_total_reward",
    "best_eval/last_stop_error_m",
    "best_eval/last_time_error_s",
    "best_eval/last_abs_time_error_s",
    "best_eval/last_total_energy_j",
    "best_eval/last_comfort_tav",
    "best_eval/last_comfort_er_pct",
    "best_eval/last_comfort_rms",
    "best_eval/best_success",
    "best_eval/best_precise_arrival",
    "best_eval/best_punctual_arrival",
    "best_eval/best_total_reward",
    "best_eval/best_stop_error_m",
    "best_eval/best_time_error_s",
    "best_eval/best_abs_time_error_s",
    "best_eval/best_total_energy_j",
    "best_eval/best_comfort_tav",
    "best_eval/best_comfort_er_pct",
    "best_eval/best_comfort_rms",
]


def compute_best_eval_metrics(
    series_map: dict[str, ScalarSeries],
    eval_tags: list[str] | None = None,
) -> dict[str, Any]:
    tags = eval_tags or BEST_EVAL_TAGS
    available = [tag for tag in tags if tag in series_map]

    if not available:
        return {"available": False}

    metrics: dict[str, Any] = {"available": True}

    for tag in available:
        _, values = _series_values(series_map, tag)
        if values.size == 0:
            continue

        short_name = tag.split("/", 1)[1] if "/" in tag else tag
        metrics[short_name] = {
            "final": float(values[-1]),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "trend_slope_per_step": float(linear_slope(_, values)),
        }

    return metrics


def _correlation_from_rows(
    values: np.ndarray, names: tuple[str, ...]
) -> dict[str, Any]:
    if values.shape[0] < 2:
        return {
            "matrix": {},
            "strong_negative_pairs": [],
            "excluded_constant_components": list(names),
        }
    variable = [
        index
        for index in range(values.shape[1])
        if float(np.std(values[:, index])) >= 1e-12
    ]
    excluded = [name for index, name in enumerate(names) if index not in variable]
    if len(variable) < 2:
        return {
            "matrix": {},
            "strong_negative_pairs": [],
            "excluded_constant_components": excluded,
        }
    correlation = np.nan_to_num(np.corrcoef(values[:, variable], rowvar=False), nan=0.0)
    selected_names = [names[index] for index in variable]
    return _correlation_payload(correlation, selected_names, excluded)


def _correlation_payload(
    correlation: np.ndarray,
    names: list[str],
    excluded: list[str],
) -> dict[str, Any]:
    matrix: dict[str, dict[str, float]] = {}
    strong_negative: list[dict[str, Any]] = []
    for row_index, row_name in enumerate(names):
        matrix[row_name] = {
            name: float(correlation[row_index, col_index])
            for col_index, name in enumerate(names)
        }
        for col_index in range(row_index + 1, len(names)):
            value = float(correlation[row_index, col_index])
            if value <= -0.4:
                strong_negative.append(
                    {"left": row_name, "right": names[col_index], "pearson": value}
                )
    return {
        "matrix": matrix,
        "strong_negative_pairs": strong_negative,
        "excluded_constant_components": excluded,
    }


def _correlation_from_moments(
    *, count: int, sums: np.ndarray, cross: np.ndarray, names: tuple[str, ...]
) -> dict[str, Any]:
    if count < 2:
        return {
            "matrix": {},
            "strong_negative_pairs": [],
            "excluded_constant_components": list(names),
        }
    covariance_numerator = cross - np.outer(sums, sums) / float(count)
    variance = np.diag(covariance_numerator)
    variable = [index for index, value in enumerate(variance) if value > 1e-12]
    excluded = [name for index, name in enumerate(names) if index not in variable]
    if len(variable) < 2:
        return {
            "matrix": {},
            "strong_negative_pairs": [],
            "excluded_constant_components": excluded,
        }
    selected = covariance_numerator[np.ix_(variable, variable)]
    scale = np.sqrt(np.maximum(np.diag(selected), 0.0))
    correlation = selected / np.outer(scale, scale)
    selected_names = [names[index] for index in variable]
    return _correlation_payload(correlation, selected_names, excluded)


def _contribution_metrics(
    *,
    reward_sum: np.ndarray,
    reward_abs_sum: np.ndarray,
    nonzero_count: np.ndarray,
    transition_count: int,
    names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    component_count = len(names) - 1
    total_sum = float(reward_sum[-1])
    total_abs_sum = float(reward_abs_sum[-1])
    component_abs_total = float(reward_abs_sum[:component_count].sum())
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names[:component_count]):
        result[name] = {
            "signed_sum": float(reward_sum[index]),
            "absolute_sum": float(reward_abs_sum[index]),
            "signed_return_ratio": _safe_ratio(float(reward_sum[index]), total_sum),
            "absolute_activity_share": _safe_ratio(
                float(reward_abs_sum[index]), component_abs_total
            ),
            "relative_total_magnitude": _safe_ratio(
                float(reward_abs_sum[index]), total_abs_sum
            ),
            "nonzero_frequency": _safe_ratio(
                float(nonzero_count[index]), float(transition_count)
            ),
            "active_mean_absolute_strength": _safe_ratio(
                float(reward_abs_sum[index]), float(nonzero_count[index])
            ),
        }
    return result


def compute_reward_component_analysis(
    artifact: RewardDiagnosticsArtifact | None,
) -> dict[str, Any]:
    if artifact is None:
        return {"available": False, "reason": "reward diagnostics artifact unavailable"}
    names = artifact.reward_names
    complete = artifact.episode_complete
    complete_rewards = artifact.episode_reward_sums[complete]
    rollout_sum = artifact.rollout_reward_sum.sum(axis=0)
    rollout_abs_sum = artifact.rollout_reward_abs_sum.sum(axis=0)
    rollout_nonzero = artifact.rollout_reward_nonzero_count.sum(axis=0)
    rollout_cross = artifact.rollout_reward_cross_product.sum(axis=0)
    transition_count = int(artifact.rollout_transition_count.sum())

    episode_groups: dict[str, dict[str, Any]] = {}
    group_masks = {
        "complete": complete,
        "terminated": complete & artifact.episode_terminated,
        "truncated": complete & artifact.episode_truncated,
    }
    for group_name, mask in group_masks.items():
        values = artifact.episode_reward_sums[mask]
        episode_groups[group_name] = {
            "count": int(values.shape[0]),
            "correlation": _correlation_from_rows(values, names),
        }

    phase_metrics: dict[str, dict[str, Any]] = {}
    rollout_indices = np.array_split(
        np.arange(artifact.rollout_transition_count.size), 3
    )
    for phase_name, indices in zip(
        ("early", "middle", "late"), rollout_indices, strict=True
    ):
        if indices.size == 0:
            phase_metrics[phase_name] = {
                "transition_count": 0,
                "components": {},
                "correlation": {},
            }
            continue
        count = int(artifact.rollout_transition_count[indices].sum())
        sums = artifact.rollout_reward_sum[indices].sum(axis=0)
        absolute_sums = artifact.rollout_reward_abs_sum[indices].sum(axis=0)
        nonzero = artifact.rollout_reward_nonzero_count[indices].sum(axis=0)
        cross = artifact.rollout_reward_cross_product[indices].sum(axis=0)
        phase_metrics[phase_name] = {
            "transition_count": count,
            "components": _contribution_metrics(
                reward_sum=sums,
                reward_abs_sum=absolute_sums,
                nonzero_count=nonzero,
                transition_count=count,
                names=names,
            ),
            "correlation": _correlation_from_moments(
                count=count, sums=sums, cross=cross, names=names
            ),
        }

    return {
        "available": True,
        "reward_names": list(names),
        "transition_count": transition_count,
        "complete_episode_count": int(complete_rewards.shape[0]),
        "partial_episode_count": int(np.count_nonzero(~complete)),
        "components": _contribution_metrics(
            reward_sum=rollout_sum,
            reward_abs_sum=rollout_abs_sum,
            nonzero_count=rollout_nonzero,
            transition_count=transition_count,
            names=names,
        ),
        "episode_return_correlation": _correlation_from_rows(complete_rewards, names),
        "episode_groups": episode_groups,
        "transition_signal_correlation": _correlation_from_moments(
            count=transition_count,
            sums=rollout_sum,
            cross=rollout_cross,
            names=names,
        ),
        "transition_phases": phase_metrics,
    }


TRAJECTORY_EVALUATION_TAGS = [
    "best_eval/last_stop_error_m",
    "best_eval/last_time_error_s",
    "best_eval/last_abs_time_error_s",
    "best_eval/last_total_energy_j",
    "best_eval/last_comfort_tav",
    "best_eval/last_comfort_er_pct",
    "best_eval/last_comfort_rms",
]


def compute_trajectory_evaluation_metrics(
    series_map: dict[str, ScalarSeries],
) -> dict[str, Any]:
    """Summarize changes in task-quality measurements across evaluations."""
    metrics: dict[str, Any] = {"available": False, "metrics": {}}
    for tag in TRAJECTORY_EVALUATION_TAGS:
        steps, values = _series_values(series_map, tag)
        if values.size == 0:
            continue
        metrics["metrics"][tag.removeprefix("best_eval/last_")] = {
            "final": float(values[-1]),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "trend_slope_per_step": float(linear_slope(steps, values)),
        }
    metrics["available"] = bool(metrics["metrics"])
    return metrics


def compute_curriculum_distribution_metrics(
    series_map: dict[str, ScalarSeries],
) -> dict[str, Any]:
    """Analyze DSPDL only when it emitted an actual-to-target KL series."""
    steps, values = _series_values(series_map, "dspdl/empirical_to_target_kl")
    if values.size == 0:
        return {
            "available": False,
            "reason": "no DSPDL empirical sampling-distribution KL was logged",
        }
    current_steps, current_values = _series_values(
        series_map, "dspdl/current_to_target_kl"
    )
    result: dict[str, Any] = {
        "available": True,
        "empirical_to_target_kl": {
            "final": float(values[-1]),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "trend_slope_per_step": float(linear_slope(steps, values)),
        },
    }
    if current_values.size:
        result["current_to_target_kl"] = {
            "final": float(current_values[-1]),
            "mean": float(np.mean(current_values)),
            "min": float(np.min(current_values)),
            "trend_slope_per_step": float(linear_slope(current_steps, current_values)),
        }
    diagnostic_tags = {
        "converged": "dspdl/converged",
        "alpha": "dspdl/alpha",
        "update_kl": "dspdl/update_kl",
        "critic_values_duration_s": "dspdl/critic_values_duration_s",
        "distribution_solve_duration_s": "dspdl/distribution_solve_duration_s",
        "critic_return_mae": "dspdl/critic_return_mae",
        "critic_return_pearson": "dspdl/critic_return_pearson",
    }
    diagnostics: dict[str, dict[str, float]] = {}
    for name, tag in diagnostic_tags.items():
        diag_steps, diag_values = _series_values(series_map, tag)
        if diag_values.size == 0:
            continue
        diagnostics[name] = {
            "final": float(diag_values[-1]),
            "mean": float(np.mean(diag_values)),
            "max": float(np.max(diag_values)),
            "trend_slope_per_step": float(linear_slope(diag_steps, diag_values)),
        }
    result["diagnostics"] = diagnostics
    return result


def compute_safety_truncation_position_metrics(
    *,
    histogram_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize worker-recorded safety truncations from the NPZ artifact."""
    unavailable = {
        "available": False,
        "bins": [],
        "highest_safety_truncation_bin": None,
        "total_safety_truncation_count": 0,
    }
    if histogram_path is None:
        return unavailable | {
            "reason": "safety truncation histogram path was not provided"
        }
    path = Path(histogram_path)
    if not path.is_file():
        return unavailable | {"reason": f"safety bins artifact not found: {path}"}
    with np.load(path) as data:
        required = (
            "bin_start_m",
            "bin_end_m",
            "safety_truncation_count",
            "low_safety_truncation_count",
            "high_safety_truncation_count",
            "global_safety_truncation_share",
            "position_bin_size_m",
        )
        if any(key not in data.files for key in required):
            return unavailable | {
                "reason": "safety bins artifact is missing required fields"
            }
        values = {
            "bin_start_m": np.asarray(data["bin_start_m"], dtype=np.float64),
            "bin_end_m": np.asarray(data["bin_end_m"], dtype=np.float64),
            "safety_truncation_count": np.asarray(
                data["safety_truncation_count"], dtype=np.int64
            ),
            "low_safety_truncation_count": np.asarray(
                data["low_safety_truncation_count"], dtype=np.int64
            ),
            "high_safety_truncation_count": np.asarray(
                data["high_safety_truncation_count"], dtype=np.int64
            ),
            "global_safety_truncation_share": np.asarray(
                data["global_safety_truncation_share"], dtype=np.float64
            ),
        }
        bin_size = np.asarray(data["position_bin_size_m"], dtype=np.float64)
    count = values["bin_start_m"].size
    if any(value.ndim != 1 or value.size != count for value in values.values()):
        return unavailable | {"reason": "safety bins artifact has inconsistent arrays"}
    if bin_size.shape != (1,) or not np.isfinite(bin_size[0]) or bin_size[0] <= 0.0:
        return unavailable | {"reason": "safety bins artifact has an invalid bin size"}
    starts = values["bin_start_m"]
    ends = values["bin_end_m"]
    totals = values["safety_truncation_count"]
    low = values["low_safety_truncation_count"]
    high = values["high_safety_truncation_count"]
    shares = values["global_safety_truncation_share"]
    if (
        not np.all(np.isfinite(starts))
        or not np.all(np.isfinite(ends))
        or not np.all(np.isfinite(shares))
        or np.any(ends <= starts)
        or np.any(totals <= 0)
        or np.any(low < 0)
        or np.any(high < 0)
        or np.any(low + high != totals)
        or np.any(shares < 0.0)
        or (count > 0 and not np.isclose(float(shares.sum()), 1.0))
    ):
        return unavailable | {"reason": "safety bins artifact contains invalid values"}
    entries = [
        {
            "bin_start_m": float(starts[index]),
            "bin_end_m": float(ends[index]),
            "safety_truncation_count": int(totals[index]),
            "low_safety_truncation_count": int(low[index]),
            "high_safety_truncation_count": int(high[index]),
            "global_safety_truncation_share": float(shares[index]),
        }
        for index in range(count)
    ]
    highest = max(
        entries,
        key=lambda entry: entry["safety_truncation_count"],
        default=None,
    )
    return {
        "available": True,
        "bins": entries,
        "highest_safety_truncation_bin": highest,
        "total_safety_truncation_count": int(totals.sum()),
        "position_bin_size_m": float(bin_size[0]),
        "artifact_path": str(path),
    }
