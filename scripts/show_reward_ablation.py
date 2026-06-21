from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from rl.experiment_utils import (
    RL_TRAJECTORY_SOURCE_CHOICES,
    add_panel_label,
    apply_rl_curve_plot_style,
    build_rl_trajectory_comparison_key,
    format_rl_trajectory_terminal_summary,
    load_rl_curve_artifact,
    load_rl_curve_metrics,
    load_run_metadata,
    render_rl_curve_on_axes,
    resolve_reward_profile,
    resolve_rl_curve_artifact,
    reward_profile_names,
)
from utils.scenario import build_safeguard_utility

ABLATION_MANIFEST_FILENAME = "reward_ablation_manifest.json"
EPISODE_METRICS_FILENAME = "episode_metrics.npz"
SAFETY_CURVE_FILENAME = "dangerous_state_rate_curve.npz"
TRAJECTORY_LAYOUT_CHOICES = ("separate",)
DEFAULT_ABLATION_ROOT = "output/optimal/rl/reward_ablation"
PROFILE_COLORS: dict[str, str] = {
    "basic": "#1f77b4",
    "basic_safety": "#2ca02c",
    "basic_safety_stopping": "#ff7f0e",
    "basic_safety_stopping_punctuality": "#d62728",
    "full_shaping": "#9467bd",
}


@dataclass(frozen=True)
class CurveRunArtifact:
    reward_profile_name: str
    repeat_index: int
    seed: int | None
    episode_metrics_path: str
    index: np.ndarray
    ep_reward: np.ndarray
    ep_len: np.ndarray


@dataclass(frozen=True)
class CurveAggregateResult:
    reward_profile_name: str
    reference_steps: np.ndarray
    mean_reward: np.ndarray
    std_reward: np.ndarray
    mean_length: np.ndarray
    std_length: np.ndarray
    valid_repeat_count: int
    episode_metrics_paths: tuple[str, ...]


@dataclass(frozen=True)
class SafetyCurveRunArtifact:
    reward_profile_name: str
    repeat_index: int
    seed: int | None
    safety_curve_path: str
    index: np.ndarray
    dangerous_state_rate: np.ndarray


@dataclass(frozen=True)
class SafetyCurveAggregateResult:
    reward_profile_name: str
    reference_steps: np.ndarray
    mean_dangerous_state_rate: np.ndarray
    std_dangerous_state_rate: np.ndarray
    valid_repeat_count: int
    safety_curve_paths: tuple[str, ...]


@dataclass(frozen=True)
class SelectedTrajectoryCandidate:
    reward_profile_name: str
    repeat_index: int
    seed: int | None
    artifact: Any
    metrics: dict[str, object]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="展示奖励消融实验的学习曲线(固定为overlay)和代表性轨迹"
    )
    parser.add_argument(
        "--ablation-root",
        default=DEFAULT_ABLATION_ROOT,
        help="奖励消融实验输出根目录, 应包含 reward_ablation_manifest.json。",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=RL_TRAJECTORY_SOURCE_CHOICES,
        default="best",
        help="展示最终轨迹或不同 best artifact 来源。",
    )
    parser.add_argument(
        "--trajectory-layout",
        choices=TRAJECTORY_LAYOUT_CHOICES,
        default="separate",
        help="轨迹图布局。首版默认按奖励情形分子图。",
    )
    parser.add_argument(
        "--reward-profiles",
        nargs="+",
        choices=reward_profile_names(),
        default=None,
        help="仅展示指定奖励情形。默认使用 manifest 中的 reward_profiles 顺序。",
    )
    parser.add_argument(
        "--no-safeguard",
        action="store_true",
        help="轨迹图不绘制 safeguard 背景。",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.99,
        help="绘制 safeguard 背景时使用的 factor。",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅解析 manifest、episode_metrics 路径和代表轨迹, 不加载数组或弹图窗。",
    )
    parser.add_argument(
        "--no-safety-curve",
        action="store_true",
        help="不加载或绘制 Dangerous State Rate 曲线。",
    )
    return parser


def load_ablation_manifest(ablation_root: str) -> dict[str, Any]:
    manifest_path = Path(ablation_root) / ABLATION_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Ablation manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def panel_label_for_index(index: int) -> str:
    if index < 0:
        raise ValueError("panel index must be non-negative")
    return f"({chr(ord('a') + index)})"


def _resolve_selected_reward_profiles(
    manifest: dict[str, Any],
    requested_profiles: list[str] | None,
) -> list[str]:
    manifest_profiles = [
        str(profile)
        for profile in manifest.get("reward_profiles", [])
        if isinstance(profile, str)
    ]
    if not manifest_profiles:
        seen: set[str] = set()
        manifest_profiles = []
        for run_entry in manifest.get("runs", []):
            if not isinstance(run_entry, dict):
                continue
            reward_profile_name = run_entry.get("reward_profile_name")
            if not isinstance(reward_profile_name, str) or reward_profile_name in seen:
                continue
            seen.add(reward_profile_name)
            manifest_profiles.append(reward_profile_name)

    if not requested_profiles:
        return manifest_profiles

    manifest_profile_set = set(manifest_profiles)
    return [
        profile for profile in requested_profiles if profile in manifest_profile_set
    ]


def _iter_completed_profile_runs(
    manifest: dict[str, Any],
    reward_profile_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    completed_runs: list[dict[str, Any]] = []
    for run_entry in manifest.get("runs", []):
        if not isinstance(run_entry, dict):
            continue
        if run_entry.get("reward_profile_name") != reward_profile_name:
            continue
        status = str(run_entry.get("status", "pending"))
        if status != "completed":
            warnings.append(
                f"Skipped run for profile={reward_profile_name}, "
                f"repeat={run_entry.get('repeat_index')} due to status={status}."
            )
            continue
        completed_runs.append(run_entry)
    return completed_runs, warnings


def _load_episode_metrics_npz(
    episode_metrics_path: Path,
) -> dict[str, Any]:
    with np.load(episode_metrics_path) as data:
        index = np.asarray(data["index"], dtype=np.float64)
        ep_reward = np.asarray(data["ep_reward"], dtype=np.float64)
        ep_len = np.asarray(data["ep_len"], dtype=np.float64)

    if index.size == 0 or ep_reward.size == 0 or ep_len.size == 0:
        raise ValueError(f"Empty episode metrics arrays in: {episode_metrics_path}")
    if not (index.size == ep_reward.size == ep_len.size):
        raise ValueError(
            f"Mismatched episode metrics array lengths in: {episode_metrics_path}"
        )
    return {
        "index": index,
        "ep_reward": ep_reward,
        "ep_len": ep_len,
    }


def _load_safety_curve_npz(
    safety_curve_path: Path,
) -> dict[str, Any]:
    with np.load(safety_curve_path) as data:
        index_key = "num_timesteps" if "num_timesteps" in data.files else "index"
        index = np.asarray(data[index_key], dtype=np.float64)
        dangerous_state_rate = np.asarray(
            data["dangerous_state_rate"],
            dtype=np.float64,
        )

    if index.size == 0 or dangerous_state_rate.size == 0:
        raise ValueError(f"Empty safety curve arrays in: {safety_curve_path}")
    if index.size != dangerous_state_rate.size:
        raise ValueError(f"Mismatched safety curve arrays in: {safety_curve_path}")

    return {
        "index": index,
        "dangerous_state_rate": dangerous_state_rate,
    }


def build_curve_aggregates(
    manifest: dict[str, Any],
    reward_profiles: list[str] | None = None,
) -> tuple[list[CurveAggregateResult], list[str]]:
    warnings: list[str] = []
    aggregates: list[CurveAggregateResult] = []
    for reward_profile_name in _resolve_selected_reward_profiles(
        manifest, reward_profiles
    ):
        profile_runs, profile_warnings = _iter_completed_profile_runs(
            manifest, reward_profile_name
        )
        warnings.extend(profile_warnings)

        curve_runs: list[CurveRunArtifact] = []
        for run_entry in profile_runs:
            final_output_dir = run_entry.get("final_output_dir")
            if not isinstance(final_output_dir, str) or not final_output_dir:
                warnings.append(
                    f"Skipped curve run for profile={reward_profile_name}, "
                    f"repeat={run_entry.get('repeat_index')} "
                    "because final_output_dir is missing."
                )
                continue

            episode_metrics_path = Path(final_output_dir) / EPISODE_METRICS_FILENAME
            if not episode_metrics_path.is_file():
                warnings.append(
                    f"Skipped curve run for profile={reward_profile_name}, "
                    f"repeat={run_entry.get('repeat_index')} "
                    "because episode_metrics is missing: {episode_metrics_path}"
                )
                continue

            run_metadata = load_run_metadata(final_output_dir)
            run_record_mode = run_metadata.get("rollout_record_trigger_mode")
            repeat_index = int(run_entry.get("repeat_index", 0))
            if not isinstance(run_record_mode, str):
                raise ValueError(
                    "Reward ablation curve loading requires "
                    "run_metadata.rollout_record_trigger_mode='steps', but got "
                    f"missing/invalid value for profile={reward_profile_name}, "
                    f"repeat={repeat_index}, episode_metrics={episode_metrics_path}."
                )
            if run_record_mode != "steps":
                raise ValueError(
                    "Reward ablation no longer supports episodes-based curve artifacts. "
                    f"Expected run_metadata.rollout_record_trigger_mode='steps', got "
                    f"'{run_record_mode}' for profile={reward_profile_name}, "
                    f"repeat={repeat_index}, episode_metrics={episode_metrics_path}."
                )

            try:
                loaded = _load_episode_metrics_npz(episode_metrics_path)
            except KeyError as exc:
                warnings.append(str(exc))
                continue

            index_arr: np.ndarray = loaded["index"]  # type: ignore[assignment]
            ep_reward_arr: np.ndarray = loaded["ep_reward"]  # type: ignore[assignment]
            ep_len_arr: np.ndarray = loaded["ep_len"]  # type: ignore[assignment]

            curve_runs.append(
                CurveRunArtifact(
                    reward_profile_name=reward_profile_name,
                    repeat_index=int(run_entry.get("repeat_index", 0)),
                    seed=(
                        int(run_entry["seed"])
                        if isinstance(run_entry.get("seed"), int)
                        else None
                    ),
                    episode_metrics_path=str(episode_metrics_path),
                    index=index_arr,
                    ep_reward=ep_reward_arr,
                    ep_len=ep_len_arr,
                )
            )

        if not curve_runs:
            warnings.append(
                f"No valid episode_metrics found for reward profile: {reward_profile_name}"  # noqa: E501
            )
            continue

        reference_x = np.unique(np.concatenate([cr.index for cr in curve_runs]))

        aligned_rewards = np.vstack([
            np.interp(
                reference_x,
                cr.index,
                cr.ep_reward,
                left=np.nan,
                right=np.nan,
            )
            for cr in curve_runs
        ])
        aligned_lengths = np.vstack([
            np.interp(
                reference_x,
                cr.index,
                cr.ep_len,
                left=np.nan,
                right=np.nan,
            )
            for cr in curve_runs
        ])

        aggregates.append(
            CurveAggregateResult(
                reward_profile_name=reward_profile_name,
                reference_steps=reference_x,
                mean_reward=np.nanmean(aligned_rewards, axis=0),
                std_reward=np.nanstd(aligned_rewards, axis=0),
                mean_length=np.nanmean(aligned_lengths, axis=0),
                std_length=np.nanstd(aligned_lengths, axis=0),
                valid_repeat_count=len(curve_runs),
                episode_metrics_paths=tuple(
                    curve_run.episode_metrics_path for curve_run in curve_runs
                ),
            )
        )

    return aggregates, warnings


def build_safety_curve_aggregates(
    manifest: dict[str, Any],
    reward_profiles: list[str] | None = None,
) -> tuple[list[SafetyCurveAggregateResult], list[str]]:
    warnings: list[str] = []
    aggregates: list[SafetyCurveAggregateResult] = []
    for reward_profile_name in _resolve_selected_reward_profiles(
        manifest, reward_profiles
    ):
        profile_runs, profile_warnings = _iter_completed_profile_runs(
            manifest, reward_profile_name
        )
        warnings.extend(profile_warnings)

        safety_runs: list[SafetyCurveRunArtifact] = []
        for run_entry in profile_runs:
            best_eval_output_dir = run_entry.get("best_eval_output_dir")
            repeat_index = int(run_entry.get("repeat_index", 0))
            if not isinstance(best_eval_output_dir, str) or not best_eval_output_dir:
                warnings.append(
                    f"Skipped safety curve run for profile={reward_profile_name}, "
                    f"repeat={run_entry.get('repeat_index')} "
                    "because best_eval_output_dir is missing."
                )
                continue

            safety_curve_path = Path(best_eval_output_dir) / SAFETY_CURVE_FILENAME
            if not safety_curve_path.is_file():
                warnings.append(
                    f"Skipped safety curve run for profile={reward_profile_name}, "
                    f"repeat={run_entry.get('repeat_index')} "
                    f"because safety curve is missing: {safety_curve_path}"
                )
                continue

            try:
                loaded = _load_safety_curve_npz(safety_curve_path)
            except (KeyError, ValueError) as exc:
                warnings.append(str(exc))
                continue

            index_arr: np.ndarray = loaded["index"]  # type: ignore[assignment]
            dangerous_state_rate_arr: np.ndarray = loaded[
                "dangerous_state_rate"
            ]  # type: ignore[assignment]

            safety_runs.append(
                SafetyCurveRunArtifact(
                    reward_profile_name=reward_profile_name,
                    repeat_index=repeat_index,
                    seed=(
                        int(run_entry["seed"])
                        if isinstance(run_entry.get("seed"), int)
                        else None
                    ),
                    safety_curve_path=str(safety_curve_path),
                    index=index_arr,
                    dangerous_state_rate=dangerous_state_rate_arr,
                )
            )

        if not safety_runs:
            warnings.append(
                f"No valid safety curves found for reward profile: {reward_profile_name}"
            )
            continue

        reference_x = np.unique(np.concatenate([run.index for run in safety_runs]))
        aligned_danger_rates = np.vstack([
            np.interp(
                reference_x,
                run.index,
                run.dangerous_state_rate,
                left=np.nan,
                right=np.nan,
            )
            for run in safety_runs
        ])

        aggregates.append(
            SafetyCurveAggregateResult(
                reward_profile_name=reward_profile_name,
                reference_steps=reference_x,
                mean_dangerous_state_rate=np.nanmean(
                    aligned_danger_rates,
                    axis=0,
                ),
                std_dangerous_state_rate=np.nanstd(
                    aligned_danger_rates,
                    axis=0,
                ),
                valid_repeat_count=len(safety_runs),
                safety_curve_paths=tuple(
                    safety_run.safety_curve_path for safety_run in safety_runs
                ),
            )
        )

    return aggregates, warnings


def select_representative_trajectory_candidates(
    manifest: dict[str, Any],
    *,
    trajectory_source: str,
    reward_profiles: list[str] | None = None,
) -> tuple[list[SelectedTrajectoryCandidate], list[str]]:
    warnings: list[str] = []
    selected_candidates: list[SelectedTrajectoryCandidate] = []
    for reward_profile_name in _resolve_selected_reward_profiles(
        manifest, reward_profiles
    ):
        profile_runs, profile_warnings = _iter_completed_profile_runs(
            manifest, reward_profile_name
        )
        warnings.extend(profile_warnings)

        candidates: list[SelectedTrajectoryCandidate] = []
        for run_entry in profile_runs:
            output_dir = run_entry.get("output_dir")
            if not isinstance(output_dir, str) or not output_dir:
                warnings.append(
                    f"Skipped trajectory run for profile={reward_profile_name}, "
                    f"repeat={run_entry.get('repeat_index')} "
                    "because output_dir is missing."
                )
                continue

            try:
                artifact = resolve_rl_curve_artifact(
                    curve_dir=output_dir,
                    trajectory_source=trajectory_source,
                )
            except FileNotFoundError as exc:
                warnings.append(str(exc))
                continue

            metrics = load_rl_curve_metrics(artifact)
            candidates.append(
                SelectedTrajectoryCandidate(
                    reward_profile_name=reward_profile_name,
                    repeat_index=int(run_entry.get("repeat_index", 0)),
                    seed=(
                        int(run_entry["seed"])
                        if isinstance(run_entry.get("seed"), int)
                        else None
                    ),
                    artifact=artifact,
                    metrics=metrics,
                )
            )

        if not candidates:
            warnings.append(
                f"No valid trajectory artifacts found for reward profile: {reward_profile_name}"  # noqa: E501
            )
            continue

        selected_candidates.append(
            max(
                candidates,
                key=lambda candidate: build_rl_trajectory_comparison_key(
                    candidate.metrics
                ),
            )
        )

    return selected_candidates, warnings


def _resolve_profile_color(reward_profile_name: str) -> str:
    return PROFILE_COLORS.get(reward_profile_name, "#4c4c4c")


def _build_profile_legend_handles(reward_profiles: list[str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for reward_profile_name in reward_profiles:
        profile = resolve_reward_profile(reward_profile_name)
        handles.append(
            Line2D(
                [0],
                [0],
                color=_resolve_profile_color(reward_profile_name),
                lw=2.0,
                label=profile.label,
            )
        )
    return handles


def _plot_curve_aggregates(
    aggregates: list[CurveAggregateResult],
    safety_aggregates: list[SafetyCurveAggregateResult] | None = None,
) -> None:
    safety_aggregates = safety_aggregates or []
    if not aggregates and not safety_aggregates:
        print("No curve aggregates available; skipped curve figure.")
        return

    apply_rl_curve_plot_style()
    ncols = 3 if safety_aggregates else 2
    fig_width = 16 if safety_aggregates else 12
    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(fig_width, 4.8),
        squeeze=False,
    )
    ax_reward = axes[0][0]
    ax_length = axes[0][1]
    ax_safety = axes[0][2] if safety_aggregates else None
    for aggregate in aggregates:
        color = _resolve_profile_color(aggregate.reward_profile_name)
        profile_label = resolve_reward_profile(aggregate.reward_profile_name).label
        ax_reward.plot(
            aggregate.reference_steps,
            aggregate.mean_reward,
            color=color,
            label=profile_label,
        )
        ax_reward.fill_between(
            aggregate.reference_steps,
            aggregate.mean_reward - aggregate.std_reward,
            aggregate.mean_reward + aggregate.std_reward,
            color=color,
            alpha=0.18,
        )
        ax_length.plot(
            aggregate.reference_steps,
            aggregate.mean_length,
            color=color,
            label=profile_label,
        )
        ax_length.fill_between(
            aggregate.reference_steps,
            aggregate.mean_length - aggregate.std_length,
            aggregate.mean_length + aggregate.std_length,
            color=color,
            alpha=0.18,
        )

    ax_reward.set_xlabel("Training steps")
    ax_reward.set_ylabel("Mean episode reward")
    ax_reward.grid(True, alpha=0.3)
    ax_length.set_xlabel("Training steps")
    ax_length.set_ylabel("Mean episode length")
    ax_length.grid(True, alpha=0.3)
    if ax_safety is not None:
        for aggregate in safety_aggregates:
            color = _resolve_profile_color(aggregate.reward_profile_name)
            profile_label = resolve_reward_profile(aggregate.reward_profile_name).label
            ax_safety.plot(
                aggregate.reference_steps,
                aggregate.mean_dangerous_state_rate,
                color=color,
                label=profile_label,
            )
            ax_safety.fill_between(
                aggregate.reference_steps,
                aggregate.mean_dangerous_state_rate
                - aggregate.std_dangerous_state_rate,
                aggregate.mean_dangerous_state_rate
                + aggregate.std_dangerous_state_rate,
                color=color,
                alpha=0.18,
            )
        ax_safety.set_xlabel("Training steps")
        ax_safety.set_ylabel("Dangerous state rate")
        ax_safety.set_ylim(0.0, 1.0)
        ax_safety.grid(True, alpha=0.3)

    legend_profile_order: list[str] = []
    for reward_profile_name in [
        *(aggregate.reward_profile_name for aggregate in aggregates),
        *(aggregate.reward_profile_name for aggregate in safety_aggregates),
    ]:
        if reward_profile_name not in legend_profile_order:
            legend_profile_order.append(reward_profile_name)

    fig.legend(
        handles=_build_profile_legend_handles(legend_profile_order),
        loc="upper center",
        ncol=min(4, len(legend_profile_order)),
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    plt.show()


def _plot_selected_trajectories(
    selected_candidates: list[SelectedTrajectoryCandidate],
    *,
    no_safeguard: bool,
    factor: float,
) -> None:
    if not selected_candidates:
        print("No trajectory candidates available; skipped trajectory figure.")
        return

    apply_rl_curve_plot_style()
    cols = 2 if len(selected_candidates) > 1 else 1
    rows = int(np.ceil(len(selected_candidates) / cols))
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(12, max(4.6 * rows, 4.8)),
        squeeze=False,
    )
    flat_axes = list(axes.flatten())
    safeguard = None if no_safeguard else build_safeguard_utility(factor)

    for index, candidate in enumerate(selected_candidates):
        ax = flat_axes[index]
        pos_arr, speed_arr, metrics = load_rl_curve_artifact(candidate.artifact)
        profile_color = _resolve_profile_color(candidate.reward_profile_name)
        render_rl_curve_on_axes(
            ax=ax,
            pos_arr=pos_arr,
            speed_arr=speed_arr,
            metrics=metrics,
            no_safeguard=no_safeguard,
            factor=factor,
            curve_color=profile_color,
            curve_label=resolve_reward_profile(candidate.reward_profile_name).label,
            safeguard=safeguard,
        )
        add_panel_label(ax=ax, label=panel_label_for_index(index))

    for ax in flat_axes[len(selected_candidates) :]:
        ax.set_visible(False)

    fig.legend(
        handles=_build_profile_legend_handles([
            candidate.reward_profile_name for candidate in selected_candidates
        ]),
        loc="upper center",
        ncol=min(4, len(selected_candidates)),
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    plt.show()


def _print_warning_block(warnings: list[str]) -> None:
    if not warnings:
        return
    print("Warnings:")
    for warning in warnings:
        print(f"  - {warning}")


def _print_curve_summary(aggregates: list[CurveAggregateResult]) -> None:
    if not aggregates:
        print("Curve summary: no valid reward profiles available.")
        return
    print("Curve summary:")
    for aggregate in aggregates:
        print(
            "  - "
            f"profile={aggregate.reward_profile_name} "
            f"valid_repeats={aggregate.valid_repeat_count} "
            f"steps={aggregate.reference_steps.size}"
        )


def _print_safety_curve_summary(
    safety_aggregates: list[SafetyCurveAggregateResult],
) -> None:
    if not safety_aggregates:
        print("Safety curve summary: no valid reward profiles available.")
        return
    print("Safety curve summary:")
    for aggregate in safety_aggregates:
        final_mean = (
            float(aggregate.mean_dangerous_state_rate[-1])
            if aggregate.mean_dangerous_state_rate.size
            else float("nan")
        )
        print(
            "  - "
            f"profile={aggregate.reward_profile_name} "
            f"valid_repeats={aggregate.valid_repeat_count} "
            f"steps={aggregate.reference_steps.size} "
            f"final_dangerous_state_rate={final_mean:.6g}"
        )


def _print_trajectory_summary(
    selected_candidates: list[SelectedTrajectoryCandidate],
) -> None:
    if not selected_candidates:
        print("Trajectory summary: no valid reward profiles available.")
        return
    print("Panel mapping:")
    for index, candidate in enumerate(selected_candidates):
        print(f"  {panel_label_for_index(index)}={candidate.reward_profile_name}")

    print("Trajectory summary:")
    for index, candidate in enumerate(selected_candidates):
        print(
            "  - "
            + format_rl_trajectory_terminal_summary(
                candidate.metrics,
                panel_label=panel_label_for_index(index),
                reward_profile_name=candidate.reward_profile_name,
                repeat_index=candidate.repeat_index,
                seed=candidate.seed,
                artifact_path=candidate.artifact.npz_path,
            )
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        manifest = load_ablation_manifest(args.ablation_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return

    curve_aggregates, curve_warnings = build_curve_aggregates(
        manifest,
        args.reward_profiles,
    )
    if args.no_safety_curve:
        safety_aggregates: list[SafetyCurveAggregateResult] = []
        safety_warnings: list[str] = []
    else:
        safety_aggregates, safety_warnings = build_safety_curve_aggregates(
            manifest,
            args.reward_profiles,
        )
    selected_candidates, trajectory_warnings = (
        select_representative_trajectory_candidates(
            manifest,
            trajectory_source=args.trajectory_source,
            reward_profiles=args.reward_profiles,
        )
    )

    _print_warning_block(curve_warnings + safety_warnings + trajectory_warnings)
    _print_curve_summary(curve_aggregates)
    if not args.no_safety_curve:
        _print_safety_curve_summary(safety_aggregates)
    _print_trajectory_summary(selected_candidates)

    if args.dry_run:
        print(
            "Dry run completed: reward ablation display plan resolved; skipped loading arrays and plotting."  # noqa: E501
        )
        return

    if not curve_aggregates and not safety_aggregates and not selected_candidates:
        parser.error("No valid ablation artifacts available for plotting.")
        return

    _plot_curve_aggregates(curve_aggregates, safety_aggregates)
    _plot_selected_trajectories(
        selected_candidates,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
    )


if __name__ == "__main__":
    main()
