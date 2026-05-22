from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.env_factory import make_env
from rl.experiment_utils import (
    TrainingRunSpec,
    apply_rl_curve_plot_style,
    build_default_training_args,
    resolve_training_run_spec,
    save_run_metadata,
)
from utils.io_utils import format_float_token
from utils.scenario import build_scenario

DEFAULT_STEP_DISTANCES: tuple[float, ...] = (50.0, 100.0, 200.0, 500.0)
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44)
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/step_distance_ablation"
STEP_DISTANCE_MANIFEST_FILENAME = "step_distance_ablation_manifest.json"
FIXED_REWARD_PROFILE = "basic"


@dataclass(frozen=True)
class StepDistanceRunEntry:
    max_step_distance: float
    repeat_index: int
    seed: int
    experiment_tag: str
    monitor_path: str
    train_args: argparse.Namespace
    training_run_spec: TrainingRunSpec


@dataclass(frozen=True)
class MonitorRunArtifact:
    max_step_distance: float
    repeat_index: int
    seed: int
    monitor_path: str
    episode_index: np.ndarray
    episode_reward: np.ndarray
    episode_length: np.ndarray


@dataclass(frozen=True)
class StepDistanceCurveAggregate:
    max_step_distance: float
    reference_episodes: np.ndarray
    mean_reward: np.ndarray
    std_reward: np.ndarray
    mean_length: np.ndarray
    std_length: np.ndarray
    valid_run_count: int
    monitor_paths: tuple[str, ...]


class MaxEpisodesStopCallback(BaseCallback):
    """Stop learning once the vectorized env has completed enough episodes."""

    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_episodes = max(1, int(max_episodes))
        self.completed_episodes = 0

    def _on_step(self) -> bool:
        dones_raw = self.locals.get("dones")
        if dones_raw is None:
            return True
        try:
            dones = list(dones_raw)
        except TypeError:
            return True

        self.completed_episodes += sum(1 for done in dones if bool(done))
        return self.completed_episodes < self.max_episodes


def _linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def _resolve_subproc_start_method() -> str:
    available_start_methods = set(mp.get_all_start_methods())
    return "forkserver" if "forkserver" in available_start_methods else "spawn"


def _build_env_initializer(
    *,
    vehicle: VehicleInfo,
    track: TrackInfo,
    safeguard_utility: SafeGuardUtility,
    train_service: TrainService,
    gamma: float,
    max_step_distance: float,
) -> Callable[[], Any]:
    def _init():
        return make_env(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            gamma=gamma,
            max_step_distance=max_step_distance,
            enable_diagnostics=False,
            diagnostics_interval_steps=1,
            reward_config=None,
        )

    return _init


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run max-step-distance ablation with fixed basic reward.",
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
        "--max-train-episodes",
        type=int,
        default=10000,
        help="Training budget measured by completed episodes.",
    )
    train_parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Safety fallback for SB3 learn(); episodes stop training first.",
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

    show_parser = subparsers.add_parser("show", help="Plot monitor learning curves.")
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
    train_args.run_mode = "monitor_best"
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
    if int(args.max_train_episodes) < 1:
        raise ValueError("--max-train-episodes must be >= 1.")

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
            monitor_path = os.path.join(spec.output_dir, "monitor", "monitor.csv")
            run_entries.append(
                StepDistanceRunEntry(
                    max_step_distance=max_step_distance,
                    repeat_index=repeat_index,
                    seed=seed,
                    experiment_tag=train_args.experiment_tag,
                    monitor_path=monitor_path,
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
            "monitor_path": entry.monitor_path,
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
        "max_train_episodes": int(args.max_train_episodes),
        "training": {
            "schedule_time_s": float(args.schedule_time_s),
            "reward_discount": float(args.reward_discount),
            "num_envs": int(args.num_envs),
            "vec_env_type": str(args.vec_env_type),
            "rollout_steps_per_update": int(args.rollout_steps_per_update),
            "n_steps_per_env": args.n_steps_per_env,
            "total_timesteps_fallback": int(args.total_timesteps),
            "device": str(args.device),
            "enable_monitor": True,
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


def _build_vec_env(spec: TrainingRunSpec):
    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=spec.schedule_time_s
    )
    env_initializers = [
        _build_env_initializer(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            gamma=spec.reward_discount,
            max_step_distance=spec.max_step_distance,
        )
        for _ in range(spec.num_envs)
    ]

    if spec.use_subproc:
        return SubprocVecEnv(
            env_initializers,
            start_method=spec.subproc_start_method or _resolve_subproc_start_method(),
        )
    return DummyVecEnv(env_initializers)


def train_step_distance_run(
    entry: StepDistanceRunEntry,
    *,
    max_train_episodes: int,
) -> TrainingRunSpec:
    spec = entry.training_run_spec
    if spec.seed is not None:
        set_random_seed(spec.seed, using_cuda=spec.device == "cuda")

    os.makedirs(spec.output_dir, exist_ok=True)
    os.makedirs(spec.final_output_dir, exist_ok=True)
    save_run_metadata(spec.output_dir, spec.run_metadata)

    venv_train = _build_vec_env(spec)
    venv_train = VecMonitor(venv_train, filename=entry.monitor_path)
    venv_train = VecNormalize(
        venv=venv_train,
        norm_obs=False,
        norm_reward=True,
        gamma=spec.reward_discount,
    )

    model = PPO(
        "MlpPolicy",
        venv_train,
        device=spec.device,
        verbose=0,
        learning_rate=_linear_schedule(3e-4),
        n_steps=spec.n_steps_per_env,
        batch_size=256,
        n_epochs=15,
        gamma=spec.reward_discount,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=None,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    model.learn(
        total_timesteps=spec.total_timesteps,
        callback=MaxEpisodesStopCallback(max_train_episodes),
        log_interval=spec.log_interval,
        progress_bar=True,
    )
    model.save(spec.final_model_save_path)
    venv_train.save(spec.final_vecnormalize_save_path)
    venv_train.close()
    return spec


def _print_run_matrix(run_entries: list[StepDistanceRunEntry]) -> None:
    print("Resolved step-distance ablation run matrix:")
    for index, entry in enumerate(run_entries, start=1):
        print(
            f"[{index}] max_step_distance={entry.max_step_distance:g} "
            f"repeat={entry.repeat_index + 1} seed={entry.seed} "
            f"output_dir={entry.training_run_spec.output_dir}"
        )


def _load_monitor_run(run_entry: dict[str, Any]) -> MonitorRunArtifact:
    monitor_path = Path(str(run_entry["monitor_path"]))
    data_frame = load_results(str(monitor_path.parent))
    rewards = np.asarray(data_frame["r"], dtype=np.float64)
    lengths = np.asarray(data_frame["l"], dtype=np.float64)
    if rewards.size == 0 or lengths.size == 0:
        raise ValueError(f"Empty monitor data: {monitor_path}")

    return MonitorRunArtifact(
        max_step_distance=float(run_entry["max_step_distance"]),
        repeat_index=int(run_entry.get("repeat_index", 0)),
        seed=int(run_entry["seed"]),
        monitor_path=str(monitor_path),
        episode_index=np.arange(rewards.size, dtype=np.float64),
        episode_reward=rewards,
        episode_length=lengths,
    )


def _resolve_display_distances(
    manifest: dict[str, Any],
    requested_distances: list[float] | None,
) -> list[float]:
    if requested_distances:
        return list(_dedupe_float_sequence(requested_distances))
    return [float(value) for value in manifest.get("max_step_distances", [])]


def build_curve_aggregates(
    manifest: dict[str, Any],
    max_step_distances: list[float] | None = None,
) -> tuple[list[StepDistanceCurveAggregate], list[str]]:
    warnings: list[str] = []
    aggregates: list[StepDistanceCurveAggregate] = []
    selected_distances = _resolve_display_distances(manifest, max_step_distances)

    for max_step_distance in selected_distances:
        monitor_runs: list[MonitorRunArtifact] = []
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
                monitor_runs.append(_load_monitor_run(run_entry))
            except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
                warnings.append(str(exc))

        if not monitor_runs:
            warnings.append(
                f"No valid monitor runs for max_step_distance={max_step_distance:g}."
            )
            continue

        max_len = max(len(run.episode_index) for run in monitor_runs)
        reference = np.arange(max_len, dtype=np.float64)
        aligned_rewards = np.vstack([
            np.interp(
                reference,
                run.episode_index,
                run.episode_reward,
                left=np.nan,
                right=np.nan,
            )
            for run in monitor_runs
        ])
        aligned_lengths = np.vstack([
            np.interp(
                reference,
                run.episode_index,
                run.episode_length,
                left=np.nan,
                right=np.nan,
            )
            for run in monitor_runs
        ])

        aggregates.append(
            StepDistanceCurveAggregate(
                max_step_distance=float(max_step_distance),
                reference_episodes=reference,
                mean_reward=np.nanmean(aligned_rewards, axis=0),
                std_reward=np.nanstd(aligned_rewards, axis=0),
                mean_length=np.nanmean(aligned_lengths, axis=0),
                std_length=np.nanstd(aligned_lengths, axis=0),
                valid_run_count=len(monitor_runs),
                monitor_paths=tuple(run.monitor_path for run in monitor_runs),
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
        ax_reward.plot(
            aggregate.reference_episodes,
            aggregate.mean_reward,
            color=color,
            label=label,
        )
        ax_reward.fill_between(
            aggregate.reference_episodes,
            aggregate.mean_reward - aggregate.std_reward,
            aggregate.mean_reward + aggregate.std_reward,
            color=color,
            alpha=0.18,
        )
        ax_length.plot(
            aggregate.reference_episodes,
            aggregate.mean_length,
            color=color,
            label=label,
        )
        ax_length.fill_between(
            aggregate.reference_episodes,
            aggregate.mean_length - aggregate.std_length,
            aggregate.mean_length + aggregate.std_length,
            color=color,
            alpha=0.18,
        )

    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Mean episode reward")
    ax_reward.grid(True, alpha=0.3)
    ax_length.set_xlabel("Episode")
    ax_length.set_ylabel("Mean episode length")
    ax_length.grid(True, alpha=0.3)
    fig.legend(loc="upper center", ncol=min(4, len(aggregates)))
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    plt.show()


def _print_curve_summary(aggregates: list[StepDistanceCurveAggregate]) -> None:
    print("Curve summary:")
    if not aggregates:
        print("  no valid step-distance curves available.")
        return
    for aggregate in aggregates:
        print(
            "  - "
            f"max_step_distance={aggregate.max_step_distance:g} "
            f"valid_runs={aggregate.valid_run_count} "
            f"episodes={aggregate.reference_episodes.size}"
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
            f"repeat={entry.repeat_index + 1}, seed={entry.seed}"
        )
        try:
            train_step_distance_run(
                entry,
                max_train_episodes=int(args.max_train_episodes),
            )
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
        print("Dry run completed: monitor curve inputs resolved.")
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
