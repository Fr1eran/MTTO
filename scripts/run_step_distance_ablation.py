from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsFloat, TypeGuard

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.evaluation import (
    PolicyEvaluationResult,
    build_single_eval_env,
    get_strict_stop_error_limit_m,
    get_strict_time_error_limit_s,
    is_success_within_train_service_limits,
    save_policy_evaluation_curve,
)
from rl.experiment_utils import (
    DEFAULT_MAX_STEP_DISTANCE,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_REWARD_PROFILE_NAME,
    DEFAULT_SCHEDULE_TIME_S,
    RL_FINAL_MODEL_FILENAME,
    RL_FINAL_VECNORMALIZE_FILENAME,
    TrainingRunSpec,
    apply_rl_curve_plot_style,
    build_default_training_args,
    load_run_metadata,
    resolve_reward_profile,
    resolve_training_run_spec,
    train_single_experiment,
)
from utils.io_utils import format_float_token
from utils.scenario import build_scenario

DEFAULT_STEP_DISTANCES: tuple[float, ...] = (
    10.0,
    30.0,
    50.0,
    100.0,
    200.0,
    500.0,
)  # 30.0
# DEFAULT_STEP_DISTANCES: tuple[float, ...] = (30.0, 40.0, 50.0, 60.0, 80.0) # 30.0
# DEFAULT_STEP_DISTANCES: tuple[float, ...] = (30.0, 35.0, 40.0, 45.0) # 30.0
DEFAULT_SEEDS: tuple[int, ...] = (11, 23, 83)
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/step_distance_ablation"
STEP_DISTANCE_MANIFEST_FILENAME = "step_distance_ablation_manifest.json"
FIXED_REWARD_PROFILE = "full_shaping"
FINAL_TRAJECTORY_FILENAME = "final_trajectory.npz"
FINAL_TRAJECTORY_METRICS_FILENAME = "final_trajectory_metrics.json"
FINAL_METRIC_KEYS: tuple[str, ...] = (
    "stop_error_m",
    "time_error_s",
    "total_energy_kj",
    "comfort_tav",
    "comfort_rms",
    "comfort_er_pct",
)


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
    index: np.ndarray
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


@dataclass(frozen=True)
class StepDistanceFinalMetricAggregate:
    max_step_distance: float
    valid_run_count: int
    metric_means: dict[str, float]
    metric_vars: dict[str, float]


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
    train_args.rollout_record_trigger_mode = "steps"
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
    final_output_dir = run_entry.get("final_output_dir")
    if not isinstance(final_output_dir, str) or not final_output_dir:
        raise ValueError(
            "Step-distance curve loading requires a valid final_output_dir, "
            f"but got missing/invalid value for max_step_distance={
                run_entry.get('max_step_distance')
            }, "
            f"repeat={run_entry.get('repeat_index')}."
        )

    episode_metrics_path = Path(str(run_entry["episode_metrics_path"]))
    if not episode_metrics_path.is_file():
        raise FileNotFoundError(
            f"episode_metrics.npz not found: {episode_metrics_path}"
        )

    run_metadata = load_run_metadata(final_output_dir)
    run_record_mode = run_metadata.get("rollout_record_trigger_mode")
    max_step_distance = float(run_entry["max_step_distance"])
    repeat_index = int(run_entry.get("repeat_index", 0))
    if not isinstance(run_record_mode, str):
        raise ValueError(
            "Step-distance curve loading requires "
            "run_metadata.rollout_record_trigger_mode='steps', but got "
            f"missing/invalid value for max_step_distance={max_step_distance:g}, "
            f"repeat={repeat_index}, episode_metrics={episode_metrics_path}."
        )
    if run_record_mode != "steps":
        raise ValueError(
            "Step-distance ablation no longer supports episodes-based curve artifacts. "
            "Expected run_metadata.rollout_record_trigger_mode='steps', got "
            f"'{run_record_mode}' for max_step_distance={max_step_distance:g}, "
            f"repeat={repeat_index}, episode_metrics={episode_metrics_path}."
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
        max_step_distance=max_step_distance,
        repeat_index=repeat_index,
        seed=int(run_entry["seed"]),
        episode_metrics_path=str(episode_metrics_path),
        index=index,
        episode_reward=rewards,
        episode_length=lengths,
    )


def _is_numeric_scalar(value: object) -> TypeGuard[SupportsFloat]:
    return isinstance(value, (int, float, np.integer, np.floating))


def _parse_required_final_metrics(metrics: dict[str, object]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    missing_keys: list[str] = []
    invalid_keys: list[str] = []
    for key in FINAL_METRIC_KEYS:
        if key not in metrics:
            missing_keys.append(key)
            continue
        raw_value = metrics[key]
        if not _is_numeric_scalar(raw_value):
            invalid_keys.append(key)
            continue
        parsed[key] = float(raw_value)

    if missing_keys or invalid_keys:
        parts: list[str] = []
        if missing_keys:
            parts.append(f"missing={missing_keys}")
        if invalid_keys:
            parts.append(f"non_numeric={invalid_keys}")
        details = ", ".join(parts)
        raise ValueError(f"Invalid final trajectory metrics payload: {details}.")
    return parsed


def _load_final_metrics_file(metrics_path: Path) -> dict[str, object]:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Final metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        payload_raw = json.load(file_obj)
    if not isinstance(payload_raw, dict):
        raise ValueError(f"Final metrics payload must be a JSON object: {metrics_path}")
    payload: dict[str, object] = {}
    for key, value in payload_raw.items():
        if isinstance(key, str):
            payload[key] = value
    return payload


def _evaluate_and_save_final_metrics(
    *,
    final_output_dir: str,
    max_step_distance_fallback: float | None,
) -> dict[str, object]:
    run_metadata = load_run_metadata(final_output_dir)
    schedule_time_s = float(
        run_metadata.get("schedule_time_s", DEFAULT_SCHEDULE_TIME_S)
    )
    reward_discount = float(
        run_metadata.get("reward_discount", DEFAULT_REWARD_DISCOUNT)
    )
    fallback_distance = (
        float(max_step_distance_fallback)
        if max_step_distance_fallback is not None
        else DEFAULT_MAX_STEP_DISTANCE
    )
    max_step_distance = float(run_metadata.get("max_step_distance", fallback_distance))
    reward_profile_name = str(
        run_metadata.get("reward_profile_name", DEFAULT_REWARD_PROFILE_NAME)
    )
    reward_profile = resolve_reward_profile(reward_profile_name)
    reward_config = reward_profile.to_reward_config()
    device = str(run_metadata.get("device", "cpu"))

    model_zip_path = Path(final_output_dir) / RL_FINAL_MODEL_FILENAME
    vecnormalize_pkl_path = Path(final_output_dir) / RL_FINAL_VECNORMALIZE_FILENAME
    if not model_zip_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_zip_path}")
    if not vecnormalize_pkl_path.is_file():
        raise FileNotFoundError(
            f"VecNormalize stats file not found: {vecnormalize_pkl_path}"
        )

    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=schedule_time_s
    )

    venv_eval = DummyVecEnv([
        lambda: build_single_eval_env(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            gamma=reward_discount,
            max_step_distance=max_step_distance,
            enable_diagnostics=False,
            enable_trajectory_tracking=True,
            reward_config=reward_config,
        )
    ])
    venv_eval = VecNormalize.load(str(vecnormalize_pkl_path), venv_eval)
    venv_eval.training = False
    venv_eval.norm_reward = False

    try:
        model = PPO.load(str(model_zip_path), device=device)

        total_reward = 0.0
        episode_steps = 0
        trajectory_position_seq: list[float] = [0.0]
        trajectory_speed_seq: list[float] = [0.0]
        obs = venv_eval.reset()
        episode_over = False
        last_info: dict[str, object] = {}

        while not episode_over:
            if not isinstance(obs, np.ndarray):
                raise TypeError(
                    "VecEnv observation must be a numpy.ndarray for MlpPolicy."
                )
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv_eval.step(action)
            total_reward += float(rewards[0])
            episode_steps += 1
            episode_over = bool(dones[0])
            last_info = infos[0]
            if isinstance(last_info, dict):
                basic = last_info.get("basic")
                if isinstance(basic, dict):
                    trajectory_position_seq.append(float(basic.get("position", 0.0)))
                    trajectory_speed_seq.append(float(basic.get("speed", 0.0)))

        target_time_s = float(train_service.schedule_time)
        start_position_m = float(train_service.start_position)
        target_position_m = float(train_service.target_position)
        basic_info = last_info.get("basic") if isinstance(last_info, dict) else None
        basic_snapshot = basic_info if isinstance(basic_info, dict) else {}

        final_position_m = float(basic_snapshot.get("position", 0.0))
        final_speed_mps = float(basic_snapshot.get("speed", 0.0))
        total_time_s = float(basic_snapshot.get("operation_time", 0.0))
        total_energy_kj = float(basic_snapshot.get("energy_consumption", 0.0))
        total_energy_j = total_energy_kj * 1000.0
        stop_error_m = abs(target_position_m - final_position_m)
        time_error_s = total_time_s - target_time_s
        comfort_tav = float(basic_snapshot.get("comfort_tav", 0.0))
        comfort_er_pct = float(basic_snapshot.get("comfort_er_pct", 0.0))
        comfort_rms = float(basic_snapshot.get("comfort_rms", 0.0))

        success = is_success_within_train_service_limits(
            stop_error_m=stop_error_m,
            time_error_s=time_error_s,
            train_service=train_service,
        )

        evaluation_result = PolicyEvaluationResult(
            success=success,
            total_reward=float(total_reward),
            total_time_s=total_time_s,
            target_time_s=target_time_s,
            total_energy_j=total_energy_j,
            total_energy_kj=total_energy_kj,
            start_position_m=start_position_m,
            target_position_m=target_position_m,
            final_position_m=final_position_m,
            final_speed_mps=final_speed_mps,
            stop_error_m=stop_error_m,
            time_error_s=time_error_s,
            strict_stop_error_limit_m=get_strict_stop_error_limit_m(train_service),
            strict_time_error_limit_s=get_strict_time_error_limit_s(train_service),
            comfort_tav=comfort_tav,
            comfort_er_pct=comfort_er_pct,
            comfort_rms=comfort_rms,
            terminated=bool(success),
            truncated=not bool(success),
            episode_steps=episode_steps,
            trajectory_pos_m=np.asarray(trajectory_position_seq, dtype=np.float32),
            trajectory_speed_mps=np.asarray(trajectory_speed_seq, dtype=np.float32),
        )

        trajectory_metadata = dict(run_metadata)
        trajectory_metadata["trajectory_source"] = "final"
        trajectory_metadata["evaluation_load_dir"] = final_output_dir
        trajectory_metadata["deterministic"] = True
        output_path = str(Path(final_output_dir) / FINAL_TRAJECTORY_FILENAME)
        _, metrics_path = save_policy_evaluation_curve(
            evaluation_result,
            output_path,
            extra_metrics=trajectory_metadata,
        )
        return _load_final_metrics_file(Path(metrics_path))
    finally:
        venv_eval.close()


def _ensure_final_metrics_for_run(
    *,
    run_entry: dict[str, Any],
    allow_regenerate: bool,
) -> tuple[dict[str, float] | None, list[str]]:
    warnings: list[str] = []
    final_output_dir = run_entry.get("final_output_dir")
    if not isinstance(final_output_dir, str) or not final_output_dir:
        return None, [
            "Skipped final evaluation metrics due to missing final_output_dir for "
            f"max_step_distance={run_entry.get('max_step_distance')}, "
            f"repeat={run_entry.get('repeat_index')}."
        ]

    metrics_path = Path(final_output_dir) / FINAL_TRAJECTORY_METRICS_FILENAME
    try:
        payload = _load_final_metrics_file(metrics_path)
        return _parse_required_final_metrics(payload), warnings
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        if not allow_regenerate:
            warnings.append(str(exc))
            return None, warnings
        warnings.append(
            "Final evaluation metrics missing/invalid; regenerating for "
            f"max_step_distance={run_entry.get('max_step_distance')}, "
            f"repeat={run_entry.get('repeat_index')}."
        )

    try:
        payload = _evaluate_and_save_final_metrics(
            final_output_dir=final_output_dir,
            max_step_distance_fallback=float(
                run_entry.get("max_step_distance", DEFAULT_MAX_STEP_DISTANCE)
            ),
        )
        return _parse_required_final_metrics(payload), warnings
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        warnings.append(str(exc))
        return None, warnings


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
            except ValueError:
                raise
            except (FileNotFoundError, KeyError, OSError) as exc:
                warnings.append(str(exc))

        if not metrics_runs:
            warnings.append(
                f"No valid episode_metrics runs for max_step_distance={
                    max_step_distance:g}."
            )
            continue

        reference = np.unique(np.concatenate([run.index for run in metrics_runs]))
        aligned_rewards = np.vstack([
            np.interp(
                reference,
                run.index,
                run.episode_reward,
                left=np.nan,
                right=np.nan,
            )
            for run in metrics_runs
        ])
        aligned_lengths = np.vstack([
            np.interp(
                reference,
                run.index,
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


def build_final_metric_aggregates(
    manifest: dict[str, Any],
    *,
    max_step_distances: list[float] | None = None,
    allow_regenerate: bool = True,
) -> tuple[list[StepDistanceFinalMetricAggregate], list[str]]:
    warnings: list[str] = []
    aggregates: list[StepDistanceFinalMetricAggregate] = []
    selected_distances = _resolve_display_distances(manifest, max_step_distances)

    for max_step_distance in selected_distances:
        metric_values: dict[str, list[float]] = {
            metric_key: [] for metric_key in FINAL_METRIC_KEYS
        }
        valid_run_count = 0

        for run_entry in manifest.get("runs", []):
            if not isinstance(run_entry, dict):
                continue
            if float(run_entry.get("max_step_distance", -1.0)) != float(
                max_step_distance
            ):
                continue
            if str(run_entry.get("status", "pending")) != "completed":
                warnings.append(
                    f"Skipped final metrics for max_step_distance={
                        max_step_distance:g}, "
                    f"repeat={run_entry.get('repeat_index')} due to "
                    f"status={run_entry.get('status', 'pending')}."
                )
                continue

            parsed_metrics, run_warnings = _ensure_final_metrics_for_run(
                run_entry=run_entry,
                allow_regenerate=allow_regenerate,
            )
            warnings.extend(run_warnings)
            if parsed_metrics is None:
                continue
            valid_run_count += 1
            for metric_key in FINAL_METRIC_KEYS:
                metric_values[metric_key].append(parsed_metrics[metric_key])

        if valid_run_count == 0:
            warnings.append(
                "No valid final evaluation metrics for "
                f"max_step_distance={max_step_distance:g}."
            )
            continue

        metric_means: dict[str, float] = {}
        metric_vars: dict[str, float] = {}
        for metric_key in FINAL_METRIC_KEYS:
            values = np.asarray(metric_values[metric_key], dtype=np.float64)
            metric_means[metric_key] = float(np.mean(values))
            metric_vars[metric_key] = float(np.var(values, ddof=0))

        aggregates.append(
            StepDistanceFinalMetricAggregate(
                max_step_distance=float(max_step_distance),
                valid_run_count=valid_run_count,
                metric_means=metric_means,
                metric_vars=metric_vars,
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
            aggregate.reference_steps,
            aggregate.mean_reward,
            color=color,
            label=label,
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
            label=label,
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


def _print_final_metric_table(
    aggregates: list[StepDistanceFinalMetricAggregate],
) -> None:
    if not aggregates:
        print("Final model evaluation summary: no valid step-distance metrics.")
        return

    columns: list[str] = ["max_step_distance", "runs"]
    for metric_key in FINAL_METRIC_KEYS:
        columns.append(metric_key)

    rows: list[list[str]] = []
    for aggregate in aggregates:
        row = [f"{aggregate.max_step_distance:g}", str(aggregate.valid_run_count)]
        for metric_key in FINAL_METRIC_KEYS:
            std_value = float(np.sqrt(aggregate.metric_vars[metric_key]))
            row.append(f"{aggregate.metric_means[metric_key]:.6f}±{std_value:.6f}")
        rows.append(row)

    widths = [len(column) for column in columns]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _fmt(row_values: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[idx]) for idx, value in enumerate(row_values)
        )

    print("Final model evaluation summary (mean±std):")
    print(_fmt(columns))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(_fmt(row))


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
            print(
                "Running final policy evaluation: "
                f"max_step_distance={entry.max_step_distance:g}, "
                f"repeat={entry.repeat_index + 1}, seed={entry.seed}"
            )
            _evaluate_and_save_final_metrics(
                final_output_dir=entry.training_run_spec.final_output_dir,
                max_step_distance_fallback=entry.max_step_distance,
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

    curve_aggregates, curve_warnings = build_curve_aggregates(
        manifest,
        max_step_distances=args.max_step_distances,
    )
    final_metric_aggregates, final_metric_warnings = build_final_metric_aggregates(
        manifest,
        max_step_distances=args.max_step_distances,
        allow_regenerate=not args.dry_run,
    )
    _print_warnings(curve_warnings + final_metric_warnings)
    _print_curve_summary(curve_aggregates)
    _print_final_metric_table(final_metric_aggregates)

    if args.dry_run:
        print(
            "Dry run completed: episode-metrics and final-evaluation inputs resolved."
        )
        return 0
    if not curve_aggregates:
        raise SystemExit("No valid monitor curves available for plotting.")

    plot_curve_aggregates(curve_aggregates)
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
