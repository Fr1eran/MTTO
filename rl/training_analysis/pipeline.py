from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .analyze import (
    compute_constraint_diagnostic,
    compute_regular_training_metrics,
    compute_reward_component_impact,
)
from .collect import load_scalar_series_from_run, resolve_run_directory
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

    regular_metrics = compute_regular_training_metrics(
        series_map,
        ema_alpha=cfg.ema_alpha,
        kl_threshold=cfg.kl_threshold,
    )
    reward_component_impact = compute_reward_component_impact(series_map)
    constraint_diagnostic = compute_constraint_diagnostic(
        series_map,
        near_miss_threshold_mps=cfg.near_miss_threshold_mps,
        position_bin_size_m=cfg.position_bin_size_m,
    )

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
        step_snapshots=step_snapshots,
        episode_snapshots=episode_snapshots,
        config=asdict(cfg),
    )

    output_paths = write_analysis_outputs(
        payload,
        output_root=cfg.output_root,
        run_name=run_dir.name,
    )
    payload["output_paths"] = output_paths

    return payload
