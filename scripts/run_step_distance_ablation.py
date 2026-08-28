from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsFloat, TypeGuard

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from rl.experiment_utils import (
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    DEFAULT_SCHEDULE_TIME_S,
    DEFAULT_TRAINING_EPISODES,
    TrainingRunSpec,
    add_panel_label,
    apply_rl_curve_plot_style,
    build_default_training_args,
    evaluate_final_training_run,
    load_run_metadata,
    resolve_training_run_spec,
    train_single_experiment,
)
from rl.training_analysis.collect import (
    extract_complete_episode_sequence,
    load_reward_diagnostics_artifact,
)
from rl.training_analysis.process import trailing_moving_average
from utils.io_utils import format_float_token
from utils.plot_utils import (
    SCI_EXPORT_PAD_INCHES,
    apply_sci_figure_layout,
    save_sci_figure,
)

DEFAULT_STEP_DISTANCES: tuple[float, ...] = (
    10.0,
    30.0,
    50.0,
    100.0,
)  # 精度上限、黄金平衡点、退化边界
DEFAULT_SEEDS: tuple[int, ...] = (
    11,
    131,
    239,
    359,
    443,
)  # 5个随机数种子
DEFAULT_OUTPUT_ROOT = "output/optimal/rl/step_distance_ablation"
DEFAULT_REFERENCE_CURVE_DIR = "output/optimal/dp/465p0_0p1_uni10p0"
DEFAULT_EPISODE_SMOOTHING_WINDOW = 100
STEP_DISTANCE_MANIFEST_FILENAME = "step_distance_ablation_manifest.json"
MANIFEST_VERSION = 4
FIXED_REWARD_PRESET = "basic_safety"
FIXED_CURRICULUM_PROFILE = "dspdl_completion"
FINAL_TRAJECTORY_METRICS_FILENAME = "final_trajectory_metrics.json"
BEST_TRAJECTORY_METRICS_FILENAME = "best_trajectory_metrics.json"
TRAJECTORY_METRIC_KEYS: tuple[str, ...] = (
    "stop_error_m",
    "time_error_s",
    "total_energy_kj",
    "comfort_tav",
)


@dataclass(frozen=True)
class StepDistanceRunEntry:
    step_distance: float
    repeat_index: int
    seed: int
    experiment_tag: str
    reward_diagnostics_path: str
    final_metrics_path: str
    train_args: argparse.Namespace
    training_run_spec: TrainingRunSpec


@dataclass(frozen=True)
class RewardDiagnosticsRunArtifact:
    step_distance: float
    repeat_index: int
    seed: int
    reward_diagnostics_path: str
    episode_number: np.ndarray
    episode_reward: np.ndarray
    episode_length: np.ndarray


@dataclass(frozen=True)
class StepDistanceCurveAggregate:
    step_distance: float
    episode_numbers: np.ndarray
    mean_reward: np.ndarray
    std_reward: np.ndarray
    mean_length: np.ndarray
    std_length: np.ndarray
    valid_seed_counts: np.ndarray
    valid_run_count: int
    reward_diagnostics_paths: tuple[str, ...]


@dataclass(frozen=True)
class StepDistanceMetricAggregate:
    step_distance: float
    valid_run_count: int
    metric_means: dict[str, float]
    metric_vars: dict[str, float]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed spatial control-step ablation with PBRS + DSPDL.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run the ablation matrix.")
    _ = train_parser.add_argument(
        "--output-root",
        "--ablation-output-root",
        dest="output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for step-distance ablation outputs.",
    )
    _ = train_parser.add_argument(
        "--reference-curve-dir",
        default=DEFAULT_REFERENCE_CURVE_DIR,
        help=(
            "Directory containing the matching DP reference trajectory required "
            "by DSPDL."
        ),
    )
    _ = train_parser.add_argument(
        "--enable-best-evaluation-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optionally retain periodic best-trajectory evaluation; final-policy "
            "evaluation is always enabled."
        ),
    )
    _ = train_parser.add_argument(
        "--training-episodes",
        type=int,
        default=DEFAULT_TRAINING_EPISODES,
        help="Global completed training episodes for every ablation run.",
    )
    _ = train_parser.add_argument(
        "--schedule-time-s", type=float, default=DEFAULT_SCHEDULE_TIME_S
    )
    _ = train_parser.add_argument(
        "--reward-discount", type=float, default=DEFAULT_REWARD_DISCOUNT
    )
    _ = train_parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    _ = train_parser.add_argument(
        "--rollout-steps-per-update", type=int, default=DEFAULT_ROLLOUT_STEPS_PER_UPDATE
    )
    _ = train_parser.add_argument(
        "--evaluation-interval-rollouts",
        type=int,
        default=DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
        help="Completed-rollout interval for best trajectory evaluation.",
    )
    _ = train_parser.add_argument("--device", default=DEFAULT_DEVICE)
    _ = train_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a compatible manifest and skip runs with complete final artifacts."
        ),
    )
    _ = train_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve the run matrix without starting training.",
    )

    show_parser = subparsers.add_parser(
        "show", help="Plot episode-metrics learning curves."
    )
    _ = show_parser.add_argument(
        "--output-root",
        "--ablation-root",
        dest="output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory containing the step-distance manifest.",
    )
    _ = show_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Path for saving a compact paper-ready figure. "
            "If omitted, only display the figure."
        ),
    )
    _ = show_parser.add_argument(
        "--episode-smoothing-window",
        type=int,
        default=DEFAULT_EPISODE_SMOOTHING_WINDOW,
        help=(
            "Trailing moving-average window in completed training episodes "
            f"(default: {DEFAULT_EPISODE_SMOOTHING_WINDOW})."
        ),
    )
    _ = show_parser.add_argument(
        "--dpi",
        type=float,
        default=300.0,
        help="DPI used when saving the figure.",
    )
    _ = show_parser.add_argument(
        "--pad-inches",
        type=float,
        default=SCI_EXPORT_PAD_INCHES,
        help="Padding around the tight saved figure.",
    )
    _ = show_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save without opening the interactive display window.",
    )
    _ = show_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve monitor inputs without plotting.",
    )
    return parser


def _build_experiment_tag(
    *,
    step_distance: float,
    repeat_index: int,
) -> str:
    distance_token = format_float_token(step_distance)
    return f"ds{distance_token}__r{repeat_index + 1:02d}"


def _build_train_args(
    args: argparse.Namespace,
    *,
    step_distance: float,
    repeat_index: int,
    seed: int,
) -> argparse.Namespace:
    train_args = argparse.Namespace(**vars(build_default_training_args()))
    train_args.output_root = args.output_root
    train_args.schedule_time_s = args.schedule_time_s
    train_args.step_distance = step_distance
    train_args.reward_preset = FIXED_REWARD_PRESET
    train_args.curriculum_profile = FIXED_CURRICULUM_PROFILE
    train_args.reference_curve_dir = args.reference_curve_dir
    train_args.experiment_tag = _build_experiment_tag(
        step_distance=step_distance,
        repeat_index=repeat_index,
    )
    train_args.run_mode = "reproduce"
    train_args.enable_tb = False
    train_args.enable_monitor = True
    train_args.enable_auto_analysis = False
    train_args.enable_best_evaluation_artifacts = bool(
        args.enable_best_evaluation_artifacts
    )
    train_args.evaluation_interval_rollouts = max(
        1,
        int(args.evaluation_interval_rollouts),
    )
    train_args.evaluation_deterministic = True
    train_args.reward_discount = args.reward_discount
    train_args.num_envs = args.num_envs
    train_args.rollout_steps_per_update = args.rollout_steps_per_update
    train_args.training_episodes = args.training_episodes
    train_args.tensorboard_log_dir = None
    train_args.tb_log_name = None
    train_args.seed = seed
    train_args.device = args.device
    train_args.dry_run = False
    return train_args


def resolve_step_distance_run_matrix(
    args: argparse.Namespace,
) -> list[StepDistanceRunEntry]:
    step_distances = DEFAULT_STEP_DISTANCES
    seeds = DEFAULT_SEEDS

    run_entries: list[StepDistanceRunEntry] = []
    for step_distance in step_distances:
        for repeat_index, seed in enumerate(seeds):
            train_args = _build_train_args(
                args,
                step_distance=step_distance,
                repeat_index=repeat_index,
                seed=seed,
            )
            spec = resolve_training_run_spec(train_args)
            run_entries.append(
                StepDistanceRunEntry(
                    step_distance=step_distance,
                    repeat_index=repeat_index,
                    seed=seed,
                    experiment_tag=train_args.experiment_tag,
                    reward_diagnostics_path=spec.reward_diagnostics_path,
                    final_metrics_path=os.path.join(
                        spec.final_output_dir,
                        FINAL_TRAJECTORY_METRICS_FILENAME,
                    ),
                    train_args=train_args,
                    training_run_spec=spec,
                )
            )
    return run_entries


def _training_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schedule_time_s": float(args.schedule_time_s),
        "reward_discount": float(args.reward_discount),
        "num_envs": int(args.num_envs),
        "rollout_steps_per_update": int(args.rollout_steps_per_update),
        "n_steps_per_env": None,
        "training_episodes": int(args.training_episodes),
        "device": str(args.device),
        "enable_monitor": True,
        "enable_auto_analysis": False,
        "enable_best_evaluation_artifacts": bool(args.enable_best_evaluation_artifacts),
        "evaluation_interval_rollouts": (
            max(1, int(args.evaluation_interval_rollouts))
            if args.enable_best_evaluation_artifacts
            else None
        ),
        "evaluation_deterministic": (
            True if args.enable_best_evaluation_artifacts else None
        ),
    }


def _manifest_training_budget_summary(spec: TrainingRunSpec) -> dict[str, object]:
    """Return resumable budget fields without duplicating the stop reason."""
    budget = spec.run_metadata["training_budget"]
    return {
        key: budget[key]
        for key in (
            "mode",
            "training_episodes",
            "effective_training_episodes",
            "max_episode_steps",
            "derived_total_timesteps",
            "actual_completed_episodes",
            "actual_training_timesteps",
            "target_reached",
        )
        if key in budget
    }


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
            (entry.step_distance, entry.repeat_index),
            {"status": "pending"},
        )
        spec = entry.training_run_spec
        runs.append(
            {
                "step_distance": float(entry.step_distance),
                "repeat_index": int(entry.repeat_index),
                "seed": int(entry.seed),
                "experiment_tag": entry.experiment_tag,
                "output_dir": spec.output_dir,
                "final_output_dir": spec.final_output_dir,
                "best_eval_output_dir": (
                    spec.best_eval_output_dir
                    if spec.enable_best_evaluation_artifacts
                    else None
                ),
                "reward_diagnostics_path": entry.reward_diagnostics_path,
                "final_metrics_path": entry.final_metrics_path,
                "run_metadata_path": spec.run_metadata_path,
                **status_payload,
            }
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "output_root": args.output_root,
        "reward_preset": FIXED_REWARD_PRESET,
        "curriculum_profile": FIXED_CURRICULUM_PROFILE,
        "reference_curve_dir": args.reference_curve_dir,
        "step_distances": [float(value) for value in DEFAULT_STEP_DISTANCES],
        "seed_list": [int(seed) for seed in DEFAULT_SEEDS],
        "training": _training_signature(args),
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


def _validate_manifest_compatibility(
    manifest: dict[str, Any], args: argparse.Namespace
) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Existing step-distance manifest has an incompatible version")
    if manifest.get("reward_preset") != FIXED_REWARD_PRESET:
        raise ValueError(
            "Existing step-distance manifest uses a different reward preset"
        )
    if manifest.get("curriculum_profile") != FIXED_CURRICULUM_PROFILE:
        raise ValueError(
            "Existing step-distance manifest uses a different curriculum profile"
        )
    if manifest.get("reference_curve_dir") != args.reference_curve_dir:
        raise ValueError(
            "Existing step-distance manifest uses a different reference curve"
        )
    if manifest.get("step_distances") != [
        float(value) for value in DEFAULT_STEP_DISTANCES
    ]:
        raise ValueError(
            "Existing step-distance manifest uses a different distance matrix"
        )
    if manifest.get("seed_list") != [int(seed) for seed in DEFAULT_SEEDS]:
        raise ValueError("Existing step-distance manifest uses different seeds")
    expected_training = _training_signature(args)
    actual_training = manifest.get("training")
    if actual_training != expected_training:
        raise ValueError(
            "Existing step-distance manifest uses different training settings: "
            f"expected={expected_training}, actual={actual_training}"
        )


def _validate_existing_manifest(args: argparse.Namespace) -> None:
    path = Path(args.output_root) / STEP_DISTANCE_MANIFEST_FILENAME
    if path.is_file():
        _validate_manifest_compatibility(
            load_step_distance_manifest(args.output_root), args
        )


def _completed_statuses_for_resume(
    args: argparse.Namespace,
    run_entries: list[StepDistanceRunEntry],
) -> dict[tuple[float, int], dict[str, Any]]:
    """Keep only completed runs whose canonical final artifacts still exist."""
    manifest_path = Path(args.output_root) / STEP_DISTANCE_MANIFEST_FILENAME
    if not args.resume or not manifest_path.is_file():
        return {}

    expected_runs = {
        (entry.step_distance, entry.repeat_index): entry for entry in run_entries
    }
    statuses: dict[tuple[float, int], dict[str, Any]] = {}
    for entry in load_step_distance_manifest(args.output_root).get("runs", []):
        if not isinstance(entry, dict) or entry.get("status") != "completed":
            continue
        try:
            key = (_entry_step_distance(entry), int(entry.get("repeat_index", -1)))
        except (TypeError, ValueError):
            continue
        expected = expected_runs.get(key)
        if expected is None:
            continue
        if (
            Path(expected.training_run_spec.final_model_save_path).is_file()
            and Path(expected.final_metrics_path).is_file()
        ):
            status_payload: dict[str, Any] = {
                "status": "completed",
                "final_metrics_path": expected.final_metrics_path,
            }
            training_budget = entry.get("training_budget")
            if isinstance(training_budget, dict):
                status_payload["training_budget"] = training_budget
            statuses[key] = status_payload
    return statuses


def train_step_distance_run(
    entry: StepDistanceRunEntry,
) -> TrainingRunSpec:
    return train_single_experiment(
        entry.train_args,
        spec=entry.training_run_spec,
    )


def evaluate_final_step_distance_run(entry: StepDistanceRunEntry) -> str:
    """Persist the deterministic real-start evaluation of a completed final model."""
    _, metrics_path = evaluate_final_training_run(entry.training_run_spec)
    return metrics_path


def _entry_step_distance(run_entry: dict[str, Any]) -> float:
    value = run_entry.get("step_distance", run_entry.get("max_step_distance"))
    if not _is_numeric_scalar(value):
        raise ValueError(f"Missing/invalid step distance in run entry: {value!r}")
    return float(value)


def _load_reward_diagnostics_run(
    *,
    run_entry: dict[str, Any],
) -> RewardDiagnosticsRunArtifact:
    final_output_dir = run_entry.get("final_output_dir")
    if not isinstance(final_output_dir, str) or not final_output_dir:
        raise ValueError(
            "Step-distance curve loading requires a valid final_output_dir, "
            + f"but got missing/invalid value for step_distance={
                run_entry.get('step_distance', run_entry.get('max_step_distance'))
            }, "
            + f"repeat={run_entry.get('repeat_index')}."
        )

    reward_diagnostics_path = Path(str(run_entry["reward_diagnostics_path"]))
    if not reward_diagnostics_path.is_file():
        raise FileNotFoundError(
            f"reward_diagnostics.npz not found: {reward_diagnostics_path}"
        )

    step_distance = _entry_step_distance(run_entry)
    repeat_index = int(run_entry.get("repeat_index", 0))
    artifact = load_reward_diagnostics_artifact(reward_diagnostics_path)
    episodes = extract_complete_episode_sequence(artifact)
    episode_number = episodes.episode_number.astype(np.float64)
    rewards = episodes.total_reward
    lengths = episodes.length

    if episode_number.size == 0 or rewards.size == 0 or lengths.size == 0:
        raise ValueError(
            f"No complete episodes in reward diagnostics: {reward_diagnostics_path}"
        )
    if rewards.size != lengths.size or rewards.size != episode_number.size:
        raise ValueError(
            f"Mismatched reward diagnostics arrays: {reward_diagnostics_path}"
        )

    return RewardDiagnosticsRunArtifact(
        step_distance=step_distance,
        repeat_index=repeat_index,
        seed=int(run_entry["seed"]),
        reward_diagnostics_path=str(reward_diagnostics_path),
        episode_number=episode_number,
        episode_reward=rewards,
        episode_length=lengths,
    )


def _is_numeric_scalar(value: object) -> TypeGuard[SupportsFloat]:
    return isinstance(value, (int, float, np.integer, np.floating))


def _parse_required_trajectory_metrics(
    metrics: dict[str, object],
    *,
    payload_name: str,
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    missing_keys: list[str] = []
    invalid_keys: list[str] = []
    for key in TRAJECTORY_METRIC_KEYS:
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
        raise ValueError(f"Invalid {payload_name} metrics payload: {details}.")
    return parsed


def _load_metrics_file(metrics_path: Path, *, payload_name: str) -> dict[str, object]:
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{payload_name} metrics file not found: {metrics_path}"
        )
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        payload_raw = json.load(file_obj)
    if not isinstance(payload_raw, dict):
        raise ValueError(
            f"{payload_name} metrics payload must be a JSON object: {metrics_path}"
        )
    payload: dict[str, object] = {}
    for key, value in payload_raw.items():
        if isinstance(key, str):
            payload[key] = value
    return payload


def _load_run_metadata_from_entry(run_entry: dict[str, Any]) -> dict[str, Any]:
    run_metadata_path = run_entry.get("run_metadata_path")
    if isinstance(run_metadata_path, str) and run_metadata_path:
        metadata_path = Path(run_metadata_path)
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload, dict):
                return payload
            raise ValueError(
                f"Run metadata payload must be a JSON object: {metadata_path}"
            )

    output_dir = run_entry.get("output_dir")
    if isinstance(output_dir, str) and output_dir:
        return load_run_metadata(output_dir)

    return {}


def _resolve_best_eval_output_dir(
    run_entry: dict[str, Any],
) -> tuple[str | None, list[str]]:
    best_eval_output_dir = run_entry.get("best_eval_output_dir")
    if isinstance(best_eval_output_dir, str) and best_eval_output_dir:
        return best_eval_output_dir, []

    try:
        run_metadata = _load_run_metadata_from_entry(run_entry)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [str(exc)]

    metadata_best_eval_output_dir = run_metadata.get("best_eval_output_dir")
    if isinstance(metadata_best_eval_output_dir, str) and metadata_best_eval_output_dir:
        return metadata_best_eval_output_dir, []

    return None, [
        "Skipped best trajectory metrics due to missing best_eval_output_dir for "
        + "step_distance="
        + f"{run_entry.get('step_distance', run_entry.get('max_step_distance'))}, "
        + f"repeat={run_entry.get('repeat_index')}."
    ]


def _ensure_metrics_for_run(
    *,
    run_entry: dict[str, Any],
    metric_source: str,
) -> tuple[dict[str, float] | None, list[str]]:
    warnings: list[str] = []
    if metric_source == "final":
        raw_path = run_entry.get("final_metrics_path")
        if not isinstance(raw_path, str) or not raw_path:
            final_output_dir = run_entry.get("final_output_dir")
            if not isinstance(final_output_dir, str) or not final_output_dir:
                return None, [
                    "Skipped final trajectory metrics due to missing "
                    "final_output_dir for "
                    + f"step_distance={_entry_step_distance(run_entry):g}."
                ]
            raw_path = str(Path(final_output_dir) / FINAL_TRAJECTORY_METRICS_FILENAME)
        metrics_path = Path(raw_path)
        payload_name = "Final trajectory"
    elif metric_source == "best":
        best_eval_output_dir, output_dir_warnings = _resolve_best_eval_output_dir(
            run_entry
        )
        warnings.extend(output_dir_warnings)
        if best_eval_output_dir is None:
            return None, warnings
        metrics_path = Path(best_eval_output_dir) / BEST_TRAJECTORY_METRICS_FILENAME
        payload_name = "Best trajectory"
    else:
        raise ValueError(f"Unsupported metric source: {metric_source}")
    try:
        payload = _load_metrics_file(metrics_path, payload_name=payload_name)
        parsed = _parse_required_trajectory_metrics(
            payload,
            payload_name=payload_name.lower(),
        )
        return parsed, warnings
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        warnings.append(str(exc))
        return None, warnings


def _print_run_matrix(run_entries: list[StepDistanceRunEntry]) -> None:
    print("Resolved step-distance ablation run matrix:")
    for index, entry in enumerate(run_entries, start=1):
        print(
            f"[{index}] step_distance={entry.step_distance:g} "
            + f"repeat={entry.repeat_index + 1} seed={entry.seed} "
            + f"output_dir={entry.training_run_spec.output_dir} "
            + f"training_episodes={entry.training_run_spec.training_episodes} "
            + f"derived_total_timesteps={entry.training_run_spec.total_timesteps}"
        )


def _resolve_display_distances(
    manifest: dict[str, Any],
    requested_distances: list[float] | None = None,
) -> list[float]:
    if requested_distances:
        return [float(value) for value in requested_distances]
    values = manifest.get("step_distances", manifest.get("max_step_distances", []))
    return [float(value) for value in values]


def _smooth_episode_run(
    run: RewardDiagnosticsRunArtifact,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window < 1:
        raise ValueError("episode_smoothing_window must be >= 1")
    rewards = trailing_moving_average(run.episode_reward, window)
    lengths = trailing_moving_average(run.episode_length, window)
    if rewards.size == 0 or lengths.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return run.episode_number[window - 1 :], rewards, lengths


def _align_episode_values(
    reference: np.ndarray,
    episode_numbers: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    aligned = np.full(reference.shape, np.nan, dtype=np.float64)
    if episode_numbers.size == 0:
        return aligned
    indices = episode_numbers.astype(np.int64) - int(reference[0])
    valid = (indices >= 0) & (indices < reference.size)
    aligned[indices[valid]] = values[valid]
    return aligned


def _mean_and_sample_std(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.sum(np.isfinite(values), axis=0)
    means = np.nanmean(values, axis=0)
    stds = np.full(means.shape, np.nan, dtype=np.float64)
    multiple = counts >= 2
    if np.any(multiple):
        stds[multiple] = np.nanstd(values[:, multiple], axis=0, ddof=1)
    return means, stds, counts


def build_curve_aggregates(
    manifest: dict[str, Any],
    step_distances: list[float] | None = None,
    *,
    episode_smoothing_window: int = DEFAULT_EPISODE_SMOOTHING_WINDOW,
) -> tuple[list[StepDistanceCurveAggregate], list[str]]:
    warnings: list[str] = []
    aggregates: list[StepDistanceCurveAggregate] = []
    selected_distances = _resolve_display_distances(manifest, step_distances)

    for step_distance in selected_distances:
        metrics_runs: list[RewardDiagnosticsRunArtifact] = []
        for run_entry in manifest.get("runs", []):
            if not isinstance(run_entry, dict):
                continue
            if _entry_step_distance(run_entry) != float(step_distance):
                continue
            if str(run_entry.get("status", "pending")) != "completed":
                warnings.append(
                    f"Skipped step_distance={step_distance:g}, "
                    + f"repeat={run_entry.get('repeat_index')} due to "
                    + f"status={run_entry.get('status', 'pending')}."
                )
                continue
            try:
                metrics_runs.append(_load_reward_diagnostics_run(run_entry=run_entry))
            except ValueError as e:
                print(f"Error loading {run_entry.get('output_dir')}: {e}")
                raise
                raise
            except (FileNotFoundError, KeyError, OSError) as exc:
                warnings.append(str(exc))

        if not metrics_runs:
            warnings.append(
                "No valid reward diagnostics runs for "
                + f"step_distance={step_distance:g}."
            )
            continue

        smoothed_runs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for run in metrics_runs:
            episode_numbers, rewards, lengths = _smooth_episode_run(
                run, window=episode_smoothing_window
            )
            if episode_numbers.size == 0:
                warnings.append(
                    f"Skipped step_distance={step_distance:g}, "
                    + f"repeat={run.repeat_index} because it has fewer than "
                    + f"{episode_smoothing_window} complete episodes."
                )
                continue
            smoothed_runs.append((episode_numbers, rewards, lengths))
        if not smoothed_runs:
            warnings.append(
                f"No smoothed episode curves for step_distance={step_distance:g}."
            )
            continue

        first_episode = int(min(run[0][0] for run in smoothed_runs))
        last_episode = int(max(run[0][-1] for run in smoothed_runs))
        reference = np.arange(first_episode, last_episode + 1, dtype=np.float64)
        aligned_rewards = np.vstack(
            [
                _align_episode_values(reference, episode_numbers, rewards)
                for episode_numbers, rewards, _ in smoothed_runs
            ]
        )
        aligned_lengths = np.vstack(
            [
                _align_episode_values(reference, episode_numbers, lengths)
                for episode_numbers, _, lengths in smoothed_runs
            ]
        )
        mean_reward, std_reward, reward_counts = _mean_and_sample_std(aligned_rewards)
        mean_length, std_length, length_counts = _mean_and_sample_std(aligned_lengths)

        aggregates.append(
            StepDistanceCurveAggregate(
                step_distance=float(step_distance),
                episode_numbers=reference,
                mean_reward=mean_reward,
                std_reward=std_reward,
                mean_length=mean_length,
                std_length=std_length,
                valid_seed_counts=np.minimum(reward_counts, length_counts),
                valid_run_count=len(smoothed_runs),
                reward_diagnostics_paths=tuple(
                    run.reward_diagnostics_path for run in metrics_runs
                ),
            )
        )

    return aggregates, warnings


def build_metric_aggregates(
    manifest: dict[str, Any],
    *,
    step_distances: list[float] | None = None,
    metric_source: str = "final",
) -> tuple[list[StepDistanceMetricAggregate], list[str]]:
    warnings: list[str] = []
    aggregates: list[StepDistanceMetricAggregate] = []
    selected_distances = _resolve_display_distances(manifest, step_distances)

    for step_distance in selected_distances:
        metric_values: dict[str, list[float]] = {
            metric_key: [] for metric_key in TRAJECTORY_METRIC_KEYS
        }
        valid_run_count = 0

        for run_entry in manifest.get("runs", []):
            if not isinstance(run_entry, dict):
                continue
            if _entry_step_distance(run_entry) != float(step_distance):
                continue
            if str(run_entry.get("status", "pending")) != "completed":
                warnings.append(
                    f"Skipped {metric_source} metrics for step_distance={
                        step_distance:g}, "
                    + f"repeat={run_entry.get('repeat_index')} due to "
                    + f"status={run_entry.get('status', 'pending')}."
                )
                continue

            parsed_metrics, run_warnings = _ensure_metrics_for_run(
                run_entry=run_entry,
                metric_source=metric_source,
            )
            warnings.extend(run_warnings)
            if parsed_metrics is None:
                continue
            valid_run_count += 1
            for metric_key in TRAJECTORY_METRIC_KEYS:
                metric_values[metric_key].append(parsed_metrics[metric_key])

        if valid_run_count == 0:
            warnings.append(
                f"No valid {metric_source} trajectory metrics for "
                + f"step_distance={step_distance:g}."
            )
            continue

        metric_means: dict[str, float] = {}
        metric_vars: dict[str, float] = {}
        for metric_key in TRAJECTORY_METRIC_KEYS:
            values = np.asarray(metric_values[metric_key], dtype=np.float64)
            metric_means[metric_key] = float(np.mean(values))
            metric_vars[metric_key] = float(np.var(values, ddof=0))

        aggregates.append(
            StepDistanceMetricAggregate(
                step_distance=float(step_distance),
                valid_run_count=valid_run_count,
                metric_means=metric_means,
                metric_vars=metric_vars,
            )
        )

    return aggregates, warnings


def resolve_metric_source(manifest: dict[str, Any]) -> str:
    """Prefer best artifacts when the target batch actually contains them."""
    for run_entry in manifest.get("runs", []):
        if not isinstance(run_entry, dict):
            continue
        if str(run_entry.get("status", "pending")) != "completed":
            continue
        best_output_dir = run_entry.get("best_eval_output_dir")
        if not isinstance(best_output_dir, str) or not best_output_dir:
            continue
        if (Path(best_output_dir) / BEST_TRAJECTORY_METRICS_FILENAME).is_file():
            return "best"
    return "final"


def _color_for_index(index: int) -> Any:
    return plt.get_cmap("tab10")(index % 10)


def plot_curve_aggregates(
    aggregates: list[StepDistanceCurveAggregate],
    *,
    show: bool = True,
):
    if not aggregates:
        print("No curve aggregates available; skipped plotting.")
        return None

    apply_rl_curve_plot_style()
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    ax_reward = axes[0][0]
    ax_length = axes[0][1]
    for ax in (ax_reward, ax_length):
        ax.set_box_aspect(3 / 4)

    for index, aggregate in enumerate(aggregates):
        color = _color_for_index(index)
        label = f"{aggregate.step_distance:g} m"
        ax_reward.plot(
            aggregate.episode_numbers,
            aggregate.mean_reward,
            color=color,
            label=label,
        )
        ax_reward.fill_between(
            aggregate.episode_numbers,
            aggregate.mean_reward - aggregate.std_reward,
            aggregate.mean_reward + aggregate.std_reward,
            color=color,
            alpha=0.18,
        )
        ax_length.plot(
            aggregate.episode_numbers,
            aggregate.mean_length,
            color=color,
            label=label,
        )
        ax_length.fill_between(
            aggregate.episode_numbers,
            aggregate.mean_length - aggregate.std_length,
            aggregate.mean_length + aggregate.std_length,
            color=color,
            alpha=0.18,
        )

    ax_reward.set_xlabel("Training episodes")
    ax_reward.set_ylabel("Mean episode reward")
    ax_reward.grid(True, alpha=0.3)
    ax_length.set_xlabel("Training episodes")
    ax_length.set_ylabel("Mean episode length")
    ax_length.grid(True, alpha=0.3)
    _ = add_panel_label(ax=ax_reward, label="(a)")
    _ = add_panel_label(ax=ax_length, label="(b)")
    handles, labels = ax_reward.get_legend_handles_labels()
    _ = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(4, len(aggregates)),
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.2,
        frameon=False,
    )
    apply_sci_figure_layout(
        fig,
        columns=2,
        height_in=3.25,
        left=0.10,
        bottom=0.19,
        top=0.84,
        wspace=0.34,
    )
    if show:
        plt.show()
    return fig


def save_compact_figure(
    fig: Figure,
    output_file: Path,
    dpi: float,
    pad_inches: float,
) -> Path:
    if output_file.suffix == "":
        output_file = output_file.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return save_sci_figure(fig, output_file, dpi=dpi, pad_inches=pad_inches)


def _print_curve_summary(aggregates: list[StepDistanceCurveAggregate]) -> None:
    print("Curve summary:")
    if not aggregates:
        print("  no valid step-distance curves available.")
        return
    for aggregate in aggregates:
        step_end = (
            float(aggregate.episode_numbers[-1])
            if aggregate.episode_numbers.size > 0
            else 0.0
        )
        print(
            "  - "
            + f"step_distance={aggregate.step_distance:g} "
            + f"valid_runs={aggregate.valid_run_count} "
            + f"episode_points={aggregate.episode_numbers.size} "
            + f"episode_end={step_end:g}"
        )


def _print_metric_table(
    aggregates: list[StepDistanceMetricAggregate],
    *,
    metric_source: str,
) -> None:
    if not aggregates:
        print(
            f"{metric_source.title()} trajectory evaluation summary: no valid metrics."
        )
        return

    columns = ["step_distance", *TRAJECTORY_METRIC_KEYS]

    rows: list[list[str]] = []
    for aggregate in aggregates:
        row = [f"{aggregate.step_distance:g}"]
        for metric_key in TRAJECTORY_METRIC_KEYS:
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

    print(f"{metric_source.title()} trajectory evaluation summary (mean±std):")
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

    _validate_existing_manifest(args)
    _print_run_matrix(run_entries)
    if args.dry_run:
        print("Dry run completed: step-distance run matrix resolved.")
        return 0

    statuses = _completed_statuses_for_resume(args, run_entries)
    manifest_path = _write_manifest(
        args.output_root,
        build_step_distance_manifest(args, run_entries, statuses=statuses),
    )

    for index, entry in enumerate(run_entries, start=1):
        key = (entry.step_distance, entry.repeat_index)
        if key in statuses:
            print(
                f"Skipping completed step-distance job {index}/{len(run_entries)}: "
                + f"step_distance={entry.step_distance:g}, "
                + f"repeat={entry.repeat_index + 1}, seed={entry.seed}"
            )
            continue
        print(
            f"Running step-distance job {index}/{len(run_entries)}: "
            + f"step_distance={entry.step_distance:g}, "
            + f"repeat={entry.repeat_index + 1}, seed={entry.seed}, "
            + f"training_episodes={entry.training_run_spec.training_episodes} "
            + f"derived_total_timesteps={entry.training_run_spec.total_timesteps}"
        )
        try:
            completed_spec = train_step_distance_run(entry)
            _, final_metrics_path = evaluate_final_training_run(completed_spec)
            statuses[key] = {
                "status": "completed",
                "final_metrics_path": final_metrics_path,
                "training_budget": _manifest_training_budget_summary(completed_spec),
            }
        except Exception as exc:
            statuses[key] = {
                "status": "failed",
                "error_message": str(exc),
            }
            _ = _write_manifest(
                args.output_root,
                build_step_distance_manifest(args, run_entries, statuses=statuses),
            )

        _ = _write_manifest(
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
    if args.episode_smoothing_window < 1:
        raise SystemExit("--episode-smoothing-window must be >= 1")

    curve_aggregates, curve_warnings = build_curve_aggregates(
        manifest,
        episode_smoothing_window=args.episode_smoothing_window,
    )
    metric_source = resolve_metric_source(manifest)
    metric_aggregates, metric_warnings = build_metric_aggregates(
        manifest,
        metric_source=metric_source,
    )
    _print_warnings(curve_warnings + metric_warnings)
    _print_curve_summary(curve_aggregates)
    print(
        "Episode smoothing: "
        + f"trailing window={args.episode_smoothing_window} completed episodes."
    )
    _print_metric_table(metric_aggregates, metric_source=metric_source)

    if args.dry_run:
        print(
            "Dry run completed: episode-metrics and "
            + f"{metric_source}-trajectory inputs resolved."
        )
        return 0
    if not curve_aggregates:
        raise SystemExit("No valid monitor curves available for plotting.")

    figure = plot_curve_aggregates(curve_aggregates, show=False)
    if figure is None:
        raise SystemExit("No valid monitor curves available for plotting.")
    if args.output_file is not None:
        output_path = save_compact_figure(
            figure,
            args.output_file,
            dpi=args.dpi,
            pad_inches=args.pad_inches,
        )
        print(f"Saved compact figure to {output_path}")
    if not args.no_show:
        plt.show()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return _run_train_command(args)
    if args.command == "show":
        return _run_show_command(args)
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
