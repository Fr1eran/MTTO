from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .analyze import (
    compute_best_eval_metrics,
    compute_curriculum_distribution_metrics,
    compute_regular_training_metrics,
    compute_reward_component_analysis,
    compute_safety_truncation_position_metrics,
    compute_trajectory_evaluation_metrics,
)
from .collect import (
    compute_sampling_health,
    load_reward_diagnostics_artifact,
    load_scalar_series_from_run,
    resolve_run_directory,
    with_legacy_info_tag_aliases,
)
from .output import build_analysis_payload, write_analysis_outputs
from .process import build_step_snapshots


@dataclass
class AnalysisConfig:
    step_window_size: int = 5000
    ema_alpha: float = 0.1
    kl_threshold: float = 0.03
    include_snapshots: bool = False
    export_csv: bool = False
    report_bar_width: int = 24
    training_log_interval: int | None = None
    min_points_per_10k_steps: float = 5.0
    rollout_steps_per_update: int = 2048
    sampling_quality_mode: str = "warn_only"
    output_root: str = "mtto_train_reports"
    sampling_health_tags: list[str] | None = None
    final_output_dir: str | None = None

    def __post_init__(self):
        if not 0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if self.step_window_size < 1:
            raise ValueError(
                f"step_window_size must be >= 1, got {self.step_window_size}"
            )
        if self.rollout_steps_per_update < 1:
            raise ValueError(
                "rollout_steps_per_update must be >= 1, "
                + f"got {self.rollout_steps_per_update}"
            )
        if self.sampling_quality_mode not in ("warn_only", "strict_fail"):
            raise ValueError(
                "sampling_quality_mode must be 'warn_only' or 'strict_fail', "
                + f"got {self.sampling_quality_mode}"
            )


DEFAULT_SNAPSHOT_TAGS = [
    "rollout/ep_rew_mean",
    "train/entropy_loss",
    "train/explained_variance",
    "train/approx_kl",
]


def _resolve_sampling_quality_mode(mode: str) -> str:
    return "strict_fail" if mode == "strict_fail" else "warn_only"


def _evaluate_sampling_quality(
    *,
    sampling_health: dict[str, Any],
    config: AnalysisConfig,
) -> dict[str, Any]:
    summary = sampling_health.get("summary", {}) if sampling_health else {}
    points_per_10k = float(summary.get("mean_samples_per_10k_steps", 0.0))
    mean_step_gap = float(summary.get("max_mean_step_gap", float("inf")))
    max_allowed_mean_step_gap = float(config.rollout_steps_per_update)

    checks = {
        "points_per_10k_ok": points_per_10k >= float(config.min_points_per_10k_steps),
        "mean_step_gap_ok": mean_step_gap <= max_allowed_mean_step_gap,
    }
    is_adequate = all(checks.values())

    reasons: list[str] = []
    if not checks["points_per_10k_ok"]:
        reasons.append(
            "mean_samples_per_10k_steps "
            + f"{points_per_10k:.3f} < {float(config.min_points_per_10k_steps):.3f}"
        )
    if not checks["mean_step_gap_ok"]:
        reasons.append(
            "max_mean_step_gap "
            + f"{mean_step_gap:.3f} > rollout_steps_per_update "
            + f"{max_allowed_mean_step_gap:.3f}"
        )

    return {
        "is_adequate": is_adequate,
        "checks": checks,
        "reasons": reasons,
        "metrics": {
            "mean_samples_per_10k_steps": points_per_10k,
            "max_mean_step_gap": mean_step_gap,
            "rollout_steps_per_update": max_allowed_mean_step_gap,
        },
        "mode": _resolve_sampling_quality_mode(config.sampling_quality_mode),
    }


def _score_snapshot_severity(
    snapshot: dict[str, Any],
    *,
    kl_threshold: float,
) -> str:
    p95 = snapshot.get("metrics", {}).get("train/approx_kl", {}).get("p95", 0.0)
    return "warn" if float(p95) > kl_threshold else "normal"


def _annotate_snapshot_severity(
    snapshots: list[dict[str, Any]],
    *,
    kl_threshold: float,
) -> list[dict[str, Any]]:
    return [
        {**s, "severity": _score_snapshot_severity(s, kl_threshold=kl_threshold)}
        for s in snapshots
    ]



def run_training_analysis(
    *,
    log_root: str | None = "mtto_ppo_tensorboard_logs",
    run_name: str | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AnalysisConfig()
    run_dir: Path | None = None
    series_map: dict[str, Any] = {}
    if log_root is not None:
        try:
            run_dir = resolve_run_directory(log_root=log_root, run_name=run_name)
        except FileNotFoundError:
            if cfg.final_output_dir is None:
                raise
        else:
            series_map = with_legacy_info_tag_aliases(
                load_scalar_series_from_run(run_dir)
            )
    sampling_health = compute_sampling_health(
        series_map,
        key_tags=cfg.sampling_health_tags,
    )
    sampling_quality = (
        _evaluate_sampling_quality(sampling_health=sampling_health, config=cfg)
        if sampling_health.get("available", False)
        else {
            "is_adequate": True,
            "checks": {},
            "reasons": ["TensorBoard scalar data unavailable"],
            "metrics": {},
            "mode": _resolve_sampling_quality_mode(cfg.sampling_quality_mode),
            "available": False,
        }
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
        warnings.warn(quality_message, stacklevel=2)

    regular_metrics = compute_regular_training_metrics(
        series_map,
        ema_alpha=cfg.ema_alpha,
        kl_threshold=cfg.kl_threshold,
    )
    best_eval_metrics = compute_best_eval_metrics(series_map)
    trajectory_evaluation_metrics = compute_trajectory_evaluation_metrics(series_map)
    reward_artifact = None
    reward_artifact_path: Path | None = None
    if cfg.final_output_dir is not None:
        reward_artifact_path = Path(cfg.final_output_dir) / "episodes.npz"
        if reward_artifact_path.is_file():
            reward_artifact = load_reward_diagnostics_artifact(reward_artifact_path)
    reward_component_analysis = compute_reward_component_analysis(reward_artifact)
    if reward_artifact_path is not None:
        reward_component_analysis["artifact_path"] = str(reward_artifact_path)
    curriculum_distribution_metrics = compute_curriculum_distribution_metrics(
        series_map
    )
    safety_truncation_position_metrics = compute_safety_truncation_position_metrics(
        histogram_path=(
            None
            if cfg.final_output_dir is None
            else Path(cfg.final_output_dir) / "safety_truncation_position_histogram.npz"
        ),
    )

    step_snapshots: list[dict[str, Any]] = []
    if cfg.include_snapshots:
        step_snapshots = build_step_snapshots(
            series_map,
            selected_tags=DEFAULT_SNAPSHOT_TAGS,
            step_window_size=cfg.step_window_size,
        )

        step_snapshots = _annotate_snapshot_severity(
            step_snapshots,
            kl_threshold=cfg.kl_threshold,
        )

    payload = build_analysis_payload(
        run_name=(
            run_dir.name
            if run_dir is not None
            else run_name
            or Path(cfg.final_output_dir or "training_run").parent.name
            or "training_run"
        ),
        run_directory=str(run_dir) if run_dir is not None else "",
        available_tags=list(series_map.keys()),
        regular_metrics=regular_metrics,
        best_eval_metrics=best_eval_metrics,
        trajectory_evaluation_metrics=trajectory_evaluation_metrics,
        reward_component_analysis=reward_component_analysis,
        curriculum_distribution_metrics=curriculum_distribution_metrics,
        safety_truncation_position_metrics=safety_truncation_position_metrics,
        step_snapshots=step_snapshots,
        config=asdict(cfg),
        data_quality={
            "sampling_health": sampling_health,
            "sampling_gate": sampling_quality,
        },
    )

    output_paths = write_analysis_outputs(
        payload,
        output_root=cfg.output_root,
        run_name=payload["meta"]["run_name"],
    )
    payload["output_paths"] = output_paths

    return payload
