from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rl.experiment_utils import (
    TrainingRunSpec,
    apply_rl_curve_plot_style,
    build_default_training_args,
    resolve_training_run_spec,
    train_single_experiment,
)
from utils.io_utils import format_float_token

DEFAULT_STEP_DISTANCES: tuple[float, ...] = (50.0, 100.0, 200.0, 500.0)
DEFAULT_SEEDS: tuple[int, ...] = (42,)
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/step_distance_ablation"
STEP_DISTANCE_MANIFEST_FILENAME = "step_distance_ablation_manifest.json"
FIXED_REWARD_PROFILE = "full"
CURVE_SMOOTH_WINDOW = 9


@dataclass(frozen=True)
class StepDistanceRunEntry:
    max_step_distance: float
    repeat_index: int
    seed: int
    experiment_tag: str
    episode_metrics_path: str
    train_args: argparse.Namespace
    training_run_spec: TrainingRunSpec


@dataclass(frozen=True)
class EpisodeMetricsRunArtifact:
    max_step_distance: float
    repeat_index: int
    seed: int
    episode_metrics_path: str
    step_index: np.ndarray
    episode_reward: np.ndarray
    episode_length: np.ndarray


@dataclass(frozen=True)
class StepDistanceCurveAggregate:
    max_step_distance: float
    reference_steps: np.ndarray
    mean_reward: np.ndarray
    std_reward: np.ndarray
    mean_length: np.ndarray
    std_length: np.ndarray
    valid_run_count: int
    episode_metrics_paths: tuple[str, ...]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run max-step-distance ablation with basic reward + PBRS.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run the ablation matrix.")
    train_parser.add_argument(
        "--output-root",
        "--ablation-output-root",
        dest="output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for step-distance ablation outputs.",
    )
    train_parser.add_argument(
        "--ablation-tag",
        default=None,
        help="Optional batch tag appended to each run experiment tag.",
    )
    train_parser.add_argument(
        "--max-step-distances",
        nargs="+",
        type=float,
        default=list(DEFAULT_STEP_DISTANCES),
        help="Max simulation displacement values to compare.",
    )
    train_parser.add_argument(
        "--seed-list",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to run for every max-step-distance value.",
    )
    train_parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Maximum simulation training timesteps.",
    )
    train_parser.add_argument("--schedule-time-s", type=float, default=430.0)
    train_parser.add_argument("--reward-discount", type=float, default=0.99)
    train_parser.add_argument("--num-envs", type=int, default=1)
    train_parser.add_argument(
        "--vec-env-type",
        choices=("dummy", "subproc"),
        default="subproc",
    )
    train_parser.add_argument("--rollout-steps-per-update", type=int, default=2048)
    train_parser.add_argument("--n-steps-per-env", type=int, default=None)
    train_parser.add_argument("--log-interval", type=int, default=None)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve the run matrix without starting training.",
    )

    show_parser = subparsers.add_parser(
        "show", help="Plot episode-metrics learning curves."
    )
    show_parser.add_argument(
        "--output-root",
        "--ablation-root",
        dest="output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory containing the step-distance manifest.",
    )
    show_parser.add_argument(
        "--max-step-distances",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset/order of max-step-distance values to display.",
    )
    show_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve monitor inputs without plotting.",
    )
    return parser


def _dedupe_float_sequence(values: list[float]) -> tuple[float, ...]:
    ordered: list[float] = []
    seen: set[float] = set()
    for value in values:
        normalized = float(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _resolve_seed_list(values: list[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("--seed-list must contain at least one seed.")
    return tuple(int(seed) for seed in values)


def _build_experiment_tag(
    *,
    ablation_tag: str | None,
    max_step_distance: float,
    repeat_index: int,
) -> str:
    distance_token = format_float_token(max_step_distance)
    run_token = f"ds{distance_token}__r{repeat_index + 1:02d}"
    return f"{ablation_tag}__{run_token}" if ablation_tag else run_token


def _build_train_args(
    args: argparse.Namespace,
    *,
    max_step_distance: float,
    repeat_index: int,
    seed: int,
) -> argparse.Namespace:
    train_args = argparse.Namespace(**vars(build_default_training_args()))
    train_args.output_root = args.output_root
    train_args.schedule_time_s = args.schedule_time_s
    train_args.max_step_distance = max_step_distance
    train_args.reward_profile = FIXED_REWARD_PROFILE
    train_args.experiment_tag = _build_experiment_tag(
        ablation_tag=args.ablation_tag,
        max_step_distance=max_step_distance,
        repeat_index=repeat_index,
    )
    train_args.run_mode = "reproduce"
    train_args.enable_tb = False
    train_args.enable_callback = False
    train_args.enable_monitor = True
    train_args.enable_env_diagnostics = False
    train_args.enable_auto_analysis = False
    train_args.enable_best_eval = False
    train_args.reward_discount = args.reward_discount
    train_args.num_envs = args.num_envs
    train_args.vec_env_type = args.vec_env_type
    train_args.rollout_steps_per_update = args.rollout_steps_per_update
    train_args.n_steps_per_env = args.n_steps_per_env
    train_args.total_timesteps = args.total_timesteps
    train_args.tensorboard_log_dir = None
    train_args.tb_log_name = None
    train_args.log_interval = args.log_interval
    train_args.rollout_record_trigger_mode = "episodes"
    train_args.seed = seed
    train_args.device = args.device
    train_args.dry_run = False
    return train_args


def resolve_step_distance_run_matrix(
    args: argparse.Namespace,
) -> list[StepDistanceRunEntry]:
    step_distances = _dedupe_float_sequence(args.max_step_distances)
    seeds = _resolve_seed_list(args.seed_list)

    run_entries: list[StepDistanceRunEntry] = []
    for max_step_distance in step_distances:
        for repeat_index, seed in enumerate(seeds):
            train_args = _build_train_args(
                args,
                max_step_distance=max_step_distance,
                repeat_index=repeat_index,
                seed=seed,
            )
            spec = resolve_training_run_spec(train_args)
            episode_metrics_path = os.path.join(
                spec.final_output_dir, "episode_metrics.npz"
            )
            run_entries.append(
                StepDistanceRunEntry(
                    max_step_distance=max_step_distance,
                    repeat_index=repeat_index,
                    seed=seed,
                    experiment_tag=train_args.experiment_tag,
                    episode_metrics_path=episode_metrics_path,
                    train_args=train_args,
                    training_run_spec=spec,
                )
            )
    return run_entries


def build_step_distance_manifest(
    args: argparse.Namespace,
    run_entries: list[StepDistanceRunEntry],
    *,
    statuses: dict[tuple[float, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    status_map = statuses or {}
    runs: list[dict[str, Any]] = []
    for entry in run_entries:
        status_payload = status_map.get(
            (entry.max_step_distance, entry.repeat_index),
            {"status": "pending"},
        )
        spec = entry.training_run_spec
        runs.append({
            "max_step_distance": float(entry.max_step_distance),
            "repeat_index": int(entry.repeat_index),
            "seed": int(entry.seed),
            "experiment_tag": entry.experiment_tag,
            "output_dir": spec.output_dir,
            "final_output_dir": spec.final_output_dir,
            "episode_metrics_path": entry.episode_metrics_path,
            "run_metadata_path": spec.run_metadata_path,
            **status_payload,
        })

    return {
        "output_root": args.output_root,
        "reward_profile": FIXED_REWARD_PROFILE,
        "max_step_distances": [
            float(value) for value in _dedupe_float_sequence(args.max_step_distances)
        ],
        "seed_list": [int(seed) for seed in _resolve_seed_list(args.seed_list)],
        "training": {
            "schedule_time_s": float(args.schedule_time_s),
            "reward_discount": float(args.reward_discount),
            "num_envs": int(args.num_envs),
            "vec_env_type": str(args.vec_env_type),
            "rollout_steps_per_update": int(args.rollout_steps_per_update),
            "n_steps_per_env": args.n_steps_per_env,
            "total_timesteps": int(args.total_timesteps),
            "device": str(args.device),
            "enable_monitor": False,
            "enable_callback": False,
            "enable_env_diagnostics": False,
            "enable_auto_analysis": False,
            "enable_best_eval": False,
        },
        "runs": runs,
    }


def _write_manifest(output_root: str, manifest: dict[str, Any]) -> str:
    os.makedirs(output_root, exist_ok=True)
    manifest_path = os.path.join(output_root, STEP_DISTANCE_MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=True, indent=2)
    return manifest_path


def load_step_distance_manifest(output_root: str) -> dict[str, Any]:
    manifest_path = Path(output_root) / STEP_DISTANCE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Step-distance manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def train_step_distance_run(
    entry: StepDistanceRunEntry,
) -> TrainingRunSpec:
    return train_single_experiment(
        entry.train_args,
        spec=entry.training_run_spec,
    )


def _load_episode_metrics_run(
    *,
    run_entry: dict[str, Any],
) -> EpisodeMetricsRunArtifact:
    episode_metrics_path = Path(str(run_entry["episode_metrics_path"]))
    if not episode_metrics_path.is_file():
        raise FileNotFoundError(
            f"episode_metrics.npz not found: {episode_metrics_path}"
        )

    with np.load(episode_metrics_path) as data:
        index = np.asarray(data["index"], dtype=np.float64)
        rewards = np.asarray(data["ep_reward"], dtype=np.float64)
        lengths = np.asarray(data["ep_len"], dtype=np.float64)

    if index.size == 0 or rewards.size == 0 or lengths.size == 0:
        raise ValueError(f"Empty episode metrics arrays: {episode_metrics_path}")
    if rewards.size != lengths.size or rewards.size != index.size:
        raise ValueError(f"Mismatched episode metrics arrays: {episode_metrics_path}")

    return EpisodeMetricsRunArtifact(
        max_step_distance=float(run_entry["max_step_distance"]),
        repeat_index=int(run_entry.get("repeat_index", 0)),
        seed=int(run_entry["seed"]),
        episode_metrics_path=str(episode_metrics_path),
        step_index=np.cumsum(lengths, dtype=np.float64),
        episode_reward=rewards,
        episode_length=lengths,
    )


def _print_run_matrix(run_entries: list[StepDistanceRunEntry]) -> None:
    print("Resolved step-distance ablation run matrix:")
    for index, entry in enumerate(run_entries, start=1):
        print(
            f"[{index}] max_step_distance={entry.max_step_distance:g} "
            f"repeat={entry.repeat_index + 1} seed={entry.seed} "
            f"output_dir={entry.training_run_spec.output_dir} "
            f"total_timesteps={entry.training_run_spec.total_timesteps}"
        )


def _resolve_display_distances(
    manifest: dict[str, Any],
    requested_distances: list[float] | None,
) -> list[float]:
    if requested_distances:
        return list(_dedupe_float_sequence(requested_distances))
    return [float(value) for value in manifest.get("max_step_distances", [])]


def _smooth_series(
    values: np.ndarray, window_size: int = CURVE_SMOOTH_WINDOW
) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    if series.size < 2:
        return series.copy()
    window = max(1, int(window_size))
    if window <= 1:
        return series.copy()
    if window > series.size:
        window = series.size

    kernel = np.ones(window, dtype=np.float64) / float(window)
    valid_mask = np.isfinite(series).astype(np.float64)
    filled = np.where(np.isfinite(series), series, 0.0)
    smooth_num = np.convolve(filled, kernel, mode="same")
    smooth_den = np.convolve(valid_mask, kernel, mode="same")
    return np.divide(
        smooth_num,
        smooth_den,
        out=np.full_like(smooth_num, np.nan),
        where=smooth_den > 1e-12,
    )


def build_curve_aggregates(
    manifest: dict[str, Any],
    max_step_distances: list[float] | None = None,
) -> tuple[list[StepDistanceCurveAggregate], list[str]]:
    warnings: list[str] = []
    aggregates: list[StepDistanceCurveAggregate] = []
    selected_distances = _resolve_display_distances(manifest, max_step_distances)

    for max_step_distance in selected_distances:
        metrics_runs: list[EpisodeMetricsRunArtifact] = []
        for run_entry in manifest.get("runs", []):
            if not isinstance(run_entry, dict):
                continue
            if float(run_entry.get("max_step_distance", -1.0)) != float(
                max_step_distance
            ):
                continue
            if str(run_entry.get("status", "pending")) != "completed":
                warnings.append(
                    f"Skipped max_step_distance={max_step_distance:g}, "
                    f"repeat={run_entry.get('repeat_index')} due to "
                    f"status={run_entry.get('status', 'pending')}."
                )
                continue
            try:
                metrics_runs.append(_load_episode_metrics_run(run_entry=run_entry))
            except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
                warnings.append(str(exc))

        if not metrics_runs:
            warnings.append(
                f"No valid episode_metrics runs for max_step_distance={max_step_distance:g}."
            )
            continue

        reference = np.unique(np.concatenate([run.step_index for run in metrics_runs]))
        aligned_rewards = np.vstack([
            np.interp(
                reference,
                run.step_index,
                run.episode_reward,
                left=np.nan,
                right=np.nan,
            )
            for run in metrics_runs
        ])
        aligned_lengths = np.vstack([
            np.interp(
                reference,
                run.step_index,
                run.episode_length,
                left=np.nan,
                right=np.nan,
            )
            for run in metrics_runs
        ])

        aggregates.append(
            StepDistanceCurveAggregate(
                max_step_distance=float(max_step_distance),
                reference_steps=reference,
                mean_reward=np.nanmean(aligned_rewards, axis=0),
                std_reward=np.nanstd(aligned_rewards, axis=0),
                mean_length=np.nanmean(aligned_lengths, axis=0),
                std_length=np.nanstd(aligned_lengths, axis=0),
                valid_run_count=len(metrics_runs),
                episode_metrics_paths=tuple(
                    run.episode_metrics_path for run in metrics_runs
                ),
            )
        )

    return aggregates, warnings


def _color_for_index(index: int) -> Any:
    return plt.get_cmap("tab10")(index % 10)


def plot_curve_aggregates(aggregates: list[StepDistanceCurveAggregate]) -> None:
    if not aggregates:
        print("No curve aggregates available; skipped plotting.")
        return

    apply_rl_curve_plot_style()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8), squeeze=False)
    ax_reward = axes[0][0]
    ax_length = axes[0][1]

    for index, aggregate in enumerate(aggregates):
        color = _color_for_index(index)
        label = f"{aggregate.max_step_distance:g} m"
        smooth_mean_reward = _smooth_series(aggregate.mean_reward)
        smooth_std_reward = _smooth_series(aggregate.std_reward)
        smooth_mean_length = _smooth_series(aggregate.mean_length)
        smooth_std_length = _smooth_series(aggregate.std_length)
        ax_reward.plot(
            aggregate.reference_steps,
            smooth_mean_reward,
            color=color,
            label=label,
        )
        ax_reward.fill_between(
            aggregate.reference_steps,
            smooth_mean_reward - smooth_std_reward,
            smooth_mean_reward + smooth_std_reward,
            color=color,
            alpha=0.18,
        )
        ax_length.plot(
            aggregate.reference_steps,
            smooth_mean_length,
            color=color,
            label=label,
        )
        ax_length.fill_between(
            aggregate.reference_steps,
            smooth_mean_length - smooth_std_length,
            smooth_mean_length + smooth_std_length,
            color=color,
            alpha=0.18,
        )

    ax_reward.set_xlabel("Training steps")
    ax_reward.set_ylabel("Mean episode reward")
    ax_reward.grid(True, alpha=0.3)
    ax_length.set_xlabel("Training steps")
    ax_length.set_ylabel("Mean episode length")
    ax_length.grid(True, alpha=0.3)
    handles, labels = ax_reward.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(aggregates)))
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    plt.show()


def _print_curve_summary(aggregates: list[StepDistanceCurveAggregate]) -> None:
    print("Curve summary:")
    if not aggregates:
        print("  no valid step-distance curves available.")
        return
    for aggregate in aggregates:
        step_end = (
            float(aggregate.reference_steps[-1])
            if aggregate.reference_steps.size > 0
            else 0.0
        )
        print(
            "  - "
            f"max_step_distance={aggregate.max_step_distance:g} "
            f"valid_runs={aggregate.valid_run_count} "
            f"steps_points={aggregate.reference_steps.size} "
            f"step_end={step_end:g}"
        )


def _print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print("Warnings:")
    for warning in warnings:
        print(f"  - {warning}")


def _run_train_command(args: argparse.Namespace) -> int:
    try:
        run_entries = resolve_step_distance_run_matrix(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    _print_run_matrix(run_entries)
    if args.dry_run:
        print("Dry run completed: step-distance run matrix resolved.")
        return 0

    statuses: dict[tuple[float, int], dict[str, Any]] = {}
    manifest_path = _write_manifest(
        args.output_root,
        build_step_distance_manifest(args, run_entries, statuses=statuses),
    )

    for index, entry in enumerate(run_entries, start=1):
        print(
            f"Running step-distance job {index}/{len(run_entries)}: "
            f"max_step_distance={entry.max_step_distance:g}, "
            f"repeat={entry.repeat_index + 1}, seed={entry.seed}, "
            f"total_timesteps={entry.training_run_spec.total_timesteps}"
        )
        try:
            train_step_distance_run(entry)
            statuses[(entry.max_step_distance, entry.repeat_index)] = {
                "status": "completed"
            }
        except Exception as exc:
            statuses[(entry.max_step_distance, entry.repeat_index)] = {
                "status": "failed",
                "error_message": str(exc),
            }
            _write_manifest(
                args.output_root,
                build_step_distance_manifest(args, run_entries, statuses=statuses),
            )
            raise

        _write_manifest(
            args.output_root,
            build_step_distance_manifest(args, run_entries, statuses=statuses),
        )

    print(f"Step-distance ablation training completed. Manifest: {manifest_path}")
    return 0


def _run_show_command(args: argparse.Namespace) -> int:
    try:
        manifest = load_step_distance_manifest(args.output_root)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    aggregates, warnings = build_curve_aggregates(
        manifest,
        max_step_distances=args.max_step_distances,
    )
    _print_warnings(warnings)
    _print_curve_summary(aggregates)

    if args.dry_run:
        print("Dry run completed: episode-metrics curve inputs resolved.")
        return 0
    if not aggregates:
        raise SystemExit("No valid monitor curves available for plotting.")

    plot_curve_aggregates(aggregates)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return _run_train_command(args)
    if args.command == "show":
        return _run_show_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
