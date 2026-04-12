from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .analyze import (
    compute_constraint_diagnostic,
    compute_evolution_metrics,
    compute_regular_training_metrics,
    compute_reward_component_impact,
)
from .collect import (
    compute_sampling_health,
    load_scalar_series_from_run,
    resolve_run_directory,
)
from .output import build_analysis_payload, write_analysis_outputs
from .process import build_episode_snapshots, build_step_snapshots


@dataclass
class AnalysisConfig:
    step_window_size: int = 5000
    episode_window_size: int = 20
    ema_alpha: float = 0.1
    kl_threshold: float = 0.03
    near_miss_threshold_mps: float = 1.0
    position_bin_size_m: float = 500.0
    critical_point_radius_m: float = 300.0
    top_k_spatial_bins: int = 8
    top_k_critical_points: int = 8
    include_snapshots: bool = False
    export_csv: bool = False
    report_bar_width: int = 24
    training_log_interval: int | None = None
    min_points_per_10k_steps: float = 5.0
    min_unique_episodes: int = 100
    max_mean_step_gap: float = 2048.0
    sampling_quality_mode: str = "warn_only"
    output_root: str = "mtto_train_reports"


DEFAULT_SNAPSHOT_TAGS = [
    "rollout/ep_rew_mean",
    "train/entropy_loss",
    "train/explained_variance",
    "train/approx_kl",
    "rewards/total",
    "rewards/safety",
    "rewards/energy",
    "rewards/comfort",
    "rewards/punctuality",
    "rewards/docking",
    "state/episode_id",
    "state/pos_m",
    "constraint/margin_to_vmax_mps",
    "constraint/margin_to_vmin_mps",
    "constraint/is_truncated",
]


def _resolve_sampling_quality_mode(mode: str) -> str:
    return "strict_fail" if mode == "strict_fail" else "warn_only"


def _count_unique_episodes(series_map: dict[str, Any]) -> int:
    episode_series = series_map.get("state/episode_id")
    if episode_series is None or episode_series.values.size == 0:
        return 0
    return int(np.unique(np.rint(episode_series.values).astype(np.int64)).size)


def _evaluate_sampling_quality(
    *,
    series_map: dict[str, Any],
    sampling_health: dict[str, Any],
    config: AnalysisConfig,
) -> dict[str, Any]:
    summary = sampling_health.get("summary", {}) if sampling_health else {}
    points_per_10k = float(summary.get("mean_samples_per_10k_steps", 0.0))
    mean_step_gap = float(summary.get("max_mean_step_gap", float("inf")))
    unique_episode_count = _count_unique_episodes(series_map)

    checks = {
        "points_per_10k_ok": points_per_10k >= float(config.min_points_per_10k_steps),
        "unique_episodes_ok": unique_episode_count >= int(config.min_unique_episodes),
        "mean_step_gap_ok": mean_step_gap <= float(config.max_mean_step_gap),
    }
    is_adequate = all(checks.values())

    reasons: list[str] = []
    if not checks["points_per_10k_ok"]:
        reasons.append(
            "mean_samples_per_10k_steps "
            f"{points_per_10k:.3f} < {float(config.min_points_per_10k_steps):.3f}"
        )
    if not checks["unique_episodes_ok"]:
        reasons.append(
            f"unique_episode_count {unique_episode_count} < {int(config.min_unique_episodes)}"
        )
    if not checks["mean_step_gap_ok"]:
        reasons.append(
            "max_mean_step_gap "
            f"{mean_step_gap:.3f} > {float(config.max_mean_step_gap):.3f}"
        )

    return {
        "is_adequate": is_adequate,
        "checks": checks,
        "reasons": reasons,
        "metrics": {
            "mean_samples_per_10k_steps": points_per_10k,
            "unique_episode_count": float(unique_episode_count),
            "max_mean_step_gap": mean_step_gap,
        },
        "mode": _resolve_sampling_quality_mode(config.sampling_quality_mode),
    }


def _score_snapshot_severity(
    snapshot: dict[str, Any],
    *,
    kl_threshold: float,
    near_miss_threshold_mps: float,
) -> str:
    metrics = snapshot.get("metrics", {})
    if not isinstance(metrics, dict):
        return "normal"

    truncated_stats = metrics.get("constraint/is_truncated", {})
    if isinstance(truncated_stats, dict):
        if float(truncated_stats.get("max", 0.0)) >= 0.5:
            return "critical"

    approx_kl_stats = metrics.get("train/approx_kl", {})
    if isinstance(approx_kl_stats, dict):
        if float(approx_kl_stats.get("p95", 0.0)) > kl_threshold:
            return "warn"

    margin_vmax = metrics.get("constraint/margin_to_vmax_mps", {})
    margin_vmin = metrics.get("constraint/margin_to_vmin_mps", {})
    if isinstance(margin_vmax, dict) and isinstance(margin_vmin, dict):
        if (
            float(margin_vmax.get("p05", near_miss_threshold_mps + 1.0))
            <= near_miss_threshold_mps
            or float(margin_vmin.get("p05", near_miss_threshold_mps + 1.0))
            <= near_miss_threshold_mps
        ):
            return "warn"

    return "normal"


def _annotate_snapshot_severity(
    snapshots: list[dict[str, Any]],
    *,
    kl_threshold: float,
    near_miss_threshold_mps: float,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for snapshot in snapshots:
        snapshot_copy = dict(snapshot)
        snapshot_copy["severity"] = _score_snapshot_severity(
            snapshot,
            kl_threshold=kl_threshold,
            near_miss_threshold_mps=near_miss_threshold_mps,
        )
        enriched.append(snapshot_copy)
    return enriched


def run_training_analysis(
    *,
    log_root: str = "mtto_ppo_tensorboard_logs",
    run_name: str | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AnalysisConfig()
    run_dir = resolve_run_directory(log_root=log_root, run_name=run_name)
    series_map = load_scalar_series_from_run(run_dir)
    sampling_health = compute_sampling_health(series_map)
    sampling_quality = _evaluate_sampling_quality(
        series_map=series_map,
        sampling_health=sampling_health,
        config=cfg,
    )

    if not sampling_quality["is_adequate"]:
        reasons = "; ".join(sampling_quality["reasons"]) or "unknown reason"
        quality_message = (
            "Sampling quality below configured thresholds. "
            f"reasons: {reasons}. "
            "Consider lowering training log_interval or relaxing sampling thresholds."
        )
        if sampling_quality["mode"] == "strict_fail":
            raise ValueError(quality_message)
        warnings.warn(quality_message)

    regular_metrics = compute_regular_training_metrics(
        series_map,
        ema_alpha=cfg.ema_alpha,
        kl_threshold=cfg.kl_threshold,
    )
    reward_component_impact = compute_reward_component_impact(
        series_map,
        episode_window_size=cfg.episode_window_size,
    )
    constraint_diagnostic = compute_constraint_diagnostic(
        series_map,
        near_miss_threshold_mps=cfg.near_miss_threshold_mps,
        position_bin_size_m=cfg.position_bin_size_m,
        critical_point_radius_m=cfg.critical_point_radius_m,
        top_k_spatial_bins=cfg.top_k_spatial_bins,
        top_k_critical_points=cfg.top_k_critical_points,
        episode_window_size=cfg.episode_window_size,
    )
    evolution_metrics = compute_evolution_metrics(
        series_map,
        episode_window_size=cfg.episode_window_size,
    )

    step_snapshots: list[dict[str, Any]] = []
    episode_snapshots: list[dict[str, Any]] = []
    if cfg.include_snapshots:
        step_snapshots = build_step_snapshots(
            series_map,
            selected_tags=DEFAULT_SNAPSHOT_TAGS,
            step_window_size=cfg.step_window_size,
        )
        episode_snapshots = build_episode_snapshots(
            series_map,
            selected_tags=DEFAULT_SNAPSHOT_TAGS,
            episode_window_size=cfg.episode_window_size,
        )

        step_snapshots = _annotate_snapshot_severity(
            step_snapshots,
            kl_threshold=cfg.kl_threshold,
            near_miss_threshold_mps=cfg.near_miss_threshold_mps,
        )
        episode_snapshots = _annotate_snapshot_severity(
            episode_snapshots,
            kl_threshold=cfg.kl_threshold,
            near_miss_threshold_mps=cfg.near_miss_threshold_mps,
        )

    payload = build_analysis_payload(
        run_name=run_dir.name,
        run_directory=str(run_dir),
        available_tags=list(series_map.keys()),
        regular_metrics=regular_metrics,
        reward_component_impact=reward_component_impact,
        constraint_diagnostic=constraint_diagnostic,
        evolution_metrics=evolution_metrics,
        step_snapshots=step_snapshots,
        episode_snapshots=episode_snapshots,
        config=asdict(cfg),
        data_quality={
            "sampling_health": sampling_health,
            "sampling_gate": sampling_quality,
        },
    )

    output_paths = write_analysis_outputs(
        payload,
        output_root=cfg.output_root,
        run_name=run_dir.name,
    )
    payload["output_paths"] = output_paths

    return payload
