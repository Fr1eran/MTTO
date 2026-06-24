from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from model.ocs import SafeGuardUtility
from rl.env_factory import make_env
from rl.evaluation import (
    PolicyEvaluationResult,
    classify_arrival_status,
    get_strict_stop_error_limit_m,
    get_strict_time_error_limit_s,
)
from rl.experiment_utils import (
    DEFAULT_MAX_STEP_DISTANCE,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_REWARD_PROFILE_NAME,
    DEFAULT_SCHEDULE_TIME_S,
    RL_FINAL_MODEL_FILENAME,
    RL_FINAL_VECNORMALIZE_FILENAME,
    apply_rl_curve_plot_style,
    load_run_metadata,
    resolve_reward_profile,
    reward_profile_names,
)
from utils.io_utils import (
    format_float_token,
    load_optimized_curve_and_metrics,
    save_curve_and_metrics,
)
from utils.scenario import build_safeguard_utility, build_scenario

DEFAULT_EVALUATE_LOAD_DIR = "output/optimal/rl/final/"
DEFAULT_OUTPUT_DIR = "output/optimal/rl/schedule_time_change_eval/"
SUMMARY_FILENAME = "schedule_time_change_summary.json"
DEFAULT_FIGURE_FILENAME = "schedule_time_change_comparison.png"
DEFAULT_DELTA_TIMES_S = (0.0, -10.0, 10.0, -20.0, 20.0)


@dataclass(frozen=True)
class ScheduleChangeCase:
    delta_time_s: float
    label: str
    token: str


@dataclass(frozen=True)
class ScheduleChangeRunResult:
    case: ScheduleChangeCase
    success: bool
    precise_arrival: bool
    punctual_arrival: bool
    total_reward: float
    initial_schedule_time_s: float
    final_schedule_time_s: float
    total_time_s: float
    time_error_s: float
    stop_error_m: float
    total_energy_kj: float
    total_energy_j: float
    final_position_m: float
    final_speed_mps: float
    episode_steps: int
    schedule_change_triggered: bool
    schedule_change_step: int | None
    schedule_change_position_m: float | None
    schedule_change_speed_mps: float | None
    trajectory_npz: str
    trajectory_metrics_json: str

    def to_summary_case(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["case"] = asdict(self.case)
        return payload


def parse_delta_times(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",")]
    deltas = tuple(float(part) for part in parts if part)
    if not deltas:
        raise argparse.ArgumentTypeError("delta list must not be empty")
    return deltas


def build_schedule_change_case(delta_time_s: float) -> ScheduleChangeCase:
    delta = float(delta_time_s)
    if delta == 0.0:
        return ScheduleChangeCase(
            delta_time_s=0.0,
            label="Original",
            token="original",
        )

    abs_delta = abs(delta)
    delta_label = f"{abs_delta:g}s"
    delta_token = format_float_token(abs_delta)
    if delta > 0.0:
        return ScheduleChangeCase(
            delta_time_s=delta,
            label=f"Plus {delta_label}",
            token=f"plus_{delta_token}s",
        )

    return ScheduleChangeCase(
        delta_time_s=delta,
        label=f"Minus {delta_label}",
        token=f"minus_{delta_token}s",
    )


def should_trigger_schedule_change(
    *,
    previous_position_m: float,
    current_position_m: float,
    change_distance_m: float,
    direction: int,
    already_triggered: bool,
    delta_time_s: float,
) -> bool:
    if already_triggered or float(delta_time_s) == 0.0:
        return False

    previous_position = float(previous_position_m)
    current_position = float(current_position_m)
    change_distance = float(change_distance_m)
    if direction >= 0:
        return previous_position <= change_distance <= current_position
    return previous_position >= change_distance >= current_position


def resolve_schedule_change_experiment_dir(load_dir: str | os.PathLike[str]) -> Path:
    root = Path(load_dir)
    direct_summary = root / SUMMARY_FILENAME
    if direct_summary.is_file():
        return root

    if not root.is_dir():
        raise FileNotFoundError(f"Schedule-change result directory not found: {root}")

    candidates = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and (path / SUMMARY_FILENAME).is_file()
        ),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find '{SUMMARY_FILENAME}' in '{root}' or its subdirectories"
        )
    return candidates[0]


def load_schedule_change_summary(
    experiment_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    summary_path = Path(experiment_dir) / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or show RL schedule-time change experiments."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run batch evaluation with an in-run schedule-time change.",
    )
    evaluate_parser.add_argument(
        "--load-dir",
        type=str,
        default=DEFAULT_EVALUATE_LOAD_DIR,
        help="PPO model and VecNormalize directory.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for schedule-time-change evaluation outputs.",
    )
    evaluate_parser.add_argument(
        "--reward-discount",
        type=float,
        default=None,
        help="Evaluation environment discount factor.",
    )
    evaluate_parser.add_argument(
        "--schedule-time-s",
        type=float,
        default=None,
        help="Initial schedule time; falls back to run metadata.",
    )
    evaluate_parser.add_argument(
        "--step-distance",
        type=float,
        default=None,
        help="Maximum simulation step distance.",
    )
    evaluate_parser.add_argument(
        "--reward-profile",
        type=str,
        choices=tuple(reward_profile_names()),
        default=None,
        help="Reward profile preset; falls back to run metadata.",
    )
    evaluate_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used to load PPO.",
    )
    evaluate_parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions.",
    )
    evaluate_parser.add_argument(
        "--change-distance-m",
        type=float,
        default=800.0,
        help="Track position at which the schedule time changes.",
    )
    evaluate_parser.add_argument(
        "--delta-times-s",
        type=parse_delta_times,
        default=DEFAULT_DELTA_TIMES_S,
        help="Comma-separated schedule-time deltas, e.g. 0,-10,10,-20,20.",
    )
    evaluate_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve configuration and paths without loading the model.",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Show a saved schedule-time-change evaluation result.",
    )
    show_parser.add_argument(
        "--load-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Saved result directory or root containing timestamped result dirs.",
    )
    show_parser.add_argument(
        "--save-figure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the comparison figure into the experiment directory.",
    )
    show_parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the comparison figure window.",
    )
    show_parser.add_argument(
        "--figure-name",
        type=str,
        default=DEFAULT_FIGURE_FILENAME,
        help="Figure filename used when --save-figure is enabled.",
    )
    show_parser.add_argument(
        "--factor",
        type=float,
        default=0.99,
        help="Safeguard factor used for rendering the safety background.",
    )

    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    return _build_arg_parser()


def _make_experiment_dir(output_dir: str | os.PathLike[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_dir) / timestamp
    suffix = 1
    while experiment_dir.exists():
        experiment_dir = Path(output_dir) / f"{timestamp}_{suffix:02d}"
        suffix += 1
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def _normalize_raw_vec_obs(venv_eval: VecNormalize, raw_obs: Any) -> np.ndarray:
    raw_obs_arr = np.asarray(raw_obs, dtype=np.float32)
    if raw_obs_arr.ndim == 1:
        raw_obs_arr = raw_obs_arr.reshape(1, -1)
    normalized = venv_eval.normalize_obs(raw_obs_arr)
    if not isinstance(normalized, np.ndarray):
        raise TypeError("VecNormalize.normalize_obs must return a numpy.ndarray")
    return normalized


def _run_one_case(
    *,
    model: PPO,
    vecnormalize_pkl_path: str,
    load_dir: str,
    experiment_dir: Path,
    case: ScheduleChangeCase,
    schedule_time_s: float,
    reward_discount: float,
    max_step_distance: float,
    reward_profile_name: str,
    reward_config: Any,
    deterministic: bool,
    change_distance_m: float,
) -> ScheduleChangeRunResult:
    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=schedule_time_s
    )

    venv_eval = DummyVecEnv([
        lambda: make_env(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            gamma=reward_discount,
            max_step_distance=max_step_distance,
            enable_diagnostics=False,
            enable_trajectory_tracking=False,
            reward_config=reward_config,
        )
    ])
    venv_eval = VecNormalize.load(vecnormalize_pkl_path, venv_eval)
    venv_eval.training = False
    venv_eval.norm_reward = False

    total_reward = 0.0
    episode_steps = 0
    start_position_m = float(train_service.start_position)
    target_position_m = float(train_service.target_position)
    trajectory_position_seq: list[float] = [start_position_m]
    trajectory_speed_seq: list[float] = [float(train_service.start_speed)]

    obs = venv_eval.reset()
    episode_over = False
    last_info: dict[str, object] = {}
    previous_position_m = start_position_m
    current_position_m = start_position_m
    current_speed_mps = float(train_service.start_speed)
    change_triggered = False
    change_step: int | None = None
    change_position_m: float | None = None
    change_speed_mps: float | None = None

    try:
        if should_trigger_schedule_change(
            previous_position_m=start_position_m,
            current_position_m=start_position_m,
            change_distance_m=change_distance_m,
            direction=1
            if train_service.target_position >= train_service.start_position
            else -1,
            already_triggered=change_triggered,
            delta_time_s=case.delta_time_s,
        ):
            raw_obs = venv_eval.env_method(
                "change_schedule_time",
                float(schedule_time_s + case.delta_time_s),
            )[0]
            obs = _normalize_raw_vec_obs(venv_eval, raw_obs)
            change_triggered = True
            change_step = 0
            change_position_m = current_position_m
            change_speed_mps = current_speed_mps

        while not episode_over:
            if not isinstance(obs, np.ndarray):
                raise TypeError(
                    "VecEnv observation must be a numpy.ndarray for MlpPolicy."
                )
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = venv_eval.step(action)
            total_reward += float(rewards[0])
            episode_steps += 1
            episode_over = bool(dones[0])
            last_info = infos[0]

            previous_position_m = current_position_m
            if isinstance(last_info, dict):
                basic = last_info.get("basic")
                if isinstance(basic, dict):
                    current_position_m = float(
                        basic.get("position", current_position_m)
                    )
                    current_speed_mps = float(basic.get("speed", current_speed_mps))
                    trajectory_position_seq.append(current_position_m)
                    trajectory_speed_seq.append(current_speed_mps)

            if (not episode_over) and should_trigger_schedule_change(
                previous_position_m=previous_position_m,
                current_position_m=current_position_m,
                change_distance_m=change_distance_m,
                direction=1
                if train_service.target_position >= train_service.start_position
                else -1,
                already_triggered=change_triggered,
                delta_time_s=case.delta_time_s,
            ):
                raw_obs = venv_eval.env_method(
                    "change_schedule_time",
                    float(schedule_time_s + case.delta_time_s),
                )[0]
                obs = _normalize_raw_vec_obs(venv_eval, raw_obs)
                change_triggered = True
                change_step = episode_steps
                change_position_m = current_position_m
                change_speed_mps = current_speed_mps
    finally:
        venv_eval.close()

    target_time_s = float(train_service.schedule_time)
    basic_info = last_info.get("basic") if isinstance(last_info, dict) else None
    basic_snapshot = basic_info if isinstance(basic_info, dict) else {}

    final_position_m = float(basic_snapshot.get("position", current_position_m))
    final_speed_mps = float(basic_snapshot.get("speed", current_speed_mps))
    total_time_s = float(basic_snapshot.get("operation_time", 0.0))
    total_energy_kj = float(basic_snapshot.get("energy_consumption", 0.0))
    total_energy_j = total_energy_kj * 1000.0
    stop_error_m = abs(target_position_m - final_position_m)
    time_error_s = total_time_s - target_time_s
    success, precise_arrival, punctual_arrival = classify_arrival_status(
        stop_error_m=stop_error_m,
        time_error_s=time_error_s,
        final_speed_mps=final_speed_mps,
        train_service=train_service,
    )

    comfort_tav = float(basic_snapshot.get("comfort_tav", 0.0))
    comfort_er_pct = float(basic_snapshot.get("comfort_er_pct", 0.0))
    comfort_rms = float(basic_snapshot.get("comfort_rms", 0.0))

    evaluation_result = PolicyEvaluationResult(
        success=success,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
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
    metrics = evaluation_result.to_metrics()
    metrics.update(
        {
            "trajectory_source": "schedule_time_change",
            "evaluation_load_dir": load_dir,
            "reward_profile_name": reward_profile_name,
            "initial_schedule_time_s": float(schedule_time_s),
            "final_schedule_time_s": target_time_s,
            "delta_time_s": float(case.delta_time_s),
            "schedule_change_case_label": case.label,
            "schedule_change_case_token": case.token,
            "schedule_change_distance_m": float(change_distance_m),
            "schedule_change_triggered": bool(change_triggered),
            "schedule_change_step": change_step,
            "schedule_change_position_m": change_position_m,
            "schedule_change_speed_mps": change_speed_mps,
            "max_step_distance": float(max_step_distance),
            "reward_discount": float(reward_discount),
            "deterministic": bool(deterministic),
        }
    )

    npz_path = experiment_dir / f"trajectory_{case.token}.npz"
    saved_npz_path, saved_json_path = save_curve_and_metrics(
        pos_arr=trajectory_position_seq,
        speed_arr=trajectory_speed_seq,
        output_path=str(npz_path),
        metrics=metrics,
    )

    return ScheduleChangeRunResult(
        case=case,
        success=success,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
        total_reward=float(total_reward),
        initial_schedule_time_s=float(schedule_time_s),
        final_schedule_time_s=target_time_s,
        total_time_s=total_time_s,
        time_error_s=time_error_s,
        stop_error_m=stop_error_m,
        total_energy_kj=total_energy_kj,
        total_energy_j=total_energy_j,
        final_position_m=final_position_m,
        final_speed_mps=final_speed_mps,
        episode_steps=episode_steps,
        schedule_change_triggered=bool(change_triggered),
        schedule_change_step=change_step,
        schedule_change_position_m=change_position_m,
        schedule_change_speed_mps=change_speed_mps,
        trajectory_npz=os.path.basename(saved_npz_path),
        trajectory_metrics_json=os.path.basename(saved_json_path),
    )


def _write_summary(
    *,
    experiment_dir: Path,
    load_dir: str,
    output_root: str,
    schedule_time_s: float,
    reward_profile_name: str,
    reward_discount: float,
    max_step_distance: float,
    deterministic: bool,
    change_distance_m: float,
    results: list[ScheduleChangeRunResult],
) -> Path:
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(experiment_dir),
        "evaluation_load_dir": load_dir,
        "output_root": output_root,
        "initial_schedule_time_s": float(schedule_time_s),
        "reward_profile_name": reward_profile_name,
        "reward_discount": float(reward_discount),
        "max_step_distance": float(max_step_distance),
        "deterministic": bool(deterministic),
        "change_distance_m": float(change_distance_m),
        "cases": [result.to_summary_case() for result in results],
    }
    summary_path = experiment_dir / SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    return summary_path


def run_evaluate(args: argparse.Namespace) -> None:
    load_dir = args.load_dir
    run_metadata = load_run_metadata(load_dir)
    schedule_time_s = float(
        args.schedule_time_s
        if args.schedule_time_s is not None
        else run_metadata.get("schedule_time_s", DEFAULT_SCHEDULE_TIME_S)
    )
    reward_discount = float(
        args.reward_discount
        if args.reward_discount is not None
        else run_metadata.get("reward_discount", DEFAULT_REWARD_DISCOUNT)
    )
    max_step_distance = float(
        args.step_distance
        if args.step_distance is not None
        else run_metadata.get("max_step_distance", DEFAULT_MAX_STEP_DISTANCE)
    )
    reward_profile = resolve_reward_profile(
        args.reward_profile
        or str(run_metadata.get("reward_profile_name", DEFAULT_REWARD_PROFILE_NAME))
    )
    reward_config = reward_profile.to_reward_config()

    model_zip_path = os.path.join(load_dir, RL_FINAL_MODEL_FILENAME)
    vecnormalize_pkl_path = os.path.join(load_dir, RL_FINAL_VECNORMALIZE_FILENAME)
    cases = [build_schedule_change_case(delta) for delta in args.delta_times_s]

    if args.dry_run:
        print("========== Schedule-Time Change Dry Run ==========")
        print("  mode:                evaluate")
        print(f"  load_dir:            {load_dir}")
        print(f"  output_dir:          {args.output_dir}")
        print(f"  reward_profile:      {reward_profile.name}")
        print(f"  schedule_time_s:     {schedule_time_s:.2f}")
        print(f"  reward_discount:     {reward_discount:.4f}")
        print(f"  step_distance:       {max_step_distance:.2f}")
        print(f"  change_distance_m:   {args.change_distance_m:.2f}")
        print(f"  delta_times_s:       {args.delta_times_s}")
        print(f"  deterministic:       {args.deterministic}")
        print(f"  model_zip_path:      {model_zip_path}")
        print(f"  model_exists:        {os.path.exists(model_zip_path)}")
        print(f"  vecnormalize_path:   {vecnormalize_pkl_path}")
        print(f"  vecnormalize_exists: {os.path.exists(vecnormalize_pkl_path)}")
        print("  cases:")
        for case in cases:
            print(f"    - {case.label}: delta={case.delta_time_s:g}s")
        print("==================================================")
        return

    if not os.path.exists(model_zip_path):
        raise FileNotFoundError(f"Model file not found: {model_zip_path}")
    if not os.path.exists(vecnormalize_pkl_path):
        raise FileNotFoundError(
            f"VecNormalize stats file not found: {vecnormalize_pkl_path}"
        )

    experiment_dir = _make_experiment_dir(args.output_dir)
    model = PPO.load(model_zip_path, device=args.device)

    results: list[ScheduleChangeRunResult] = []
    for case in cases:
        result = _run_one_case(
            model=model,
            vecnormalize_pkl_path=vecnormalize_pkl_path,
            load_dir=load_dir,
            experiment_dir=experiment_dir,
            case=case,
            schedule_time_s=schedule_time_s,
            reward_discount=reward_discount,
            max_step_distance=max_step_distance,
            reward_profile_name=reward_profile.name,
            reward_config=reward_config,
            deterministic=bool(args.deterministic),
            change_distance_m=float(args.change_distance_m),
        )
        results.append(result)

    summary_path = _write_summary(
        experiment_dir=experiment_dir,
        load_dir=load_dir,
        output_root=args.output_dir,
        schedule_time_s=schedule_time_s,
        reward_profile_name=reward_profile.name,
        reward_discount=reward_discount,
        max_step_distance=max_step_distance,
        deterministic=bool(args.deterministic),
        change_distance_m=float(args.change_distance_m),
        results=results,
    )

    print("========== Schedule-Time Change Evaluation ==========")
    print(f"  experiment_dir: {experiment_dir}")
    print(f"  summary_json:   {summary_path}")
    for result in results:
        print(
            f"  {result.case.label:<10} "
            f"success={result.success} "
            f"precise={result.precise_arrival} "
            f"punctual={result.punctual_arrival} "
            f"target={result.final_schedule_time_s:.2f}s "
            f"actual={result.total_time_s:.2f}s "
            f"error={result.time_error_s:.2f}s "
            f"energy={result.total_energy_kj:.2f}kJ"
        )
    print("=====================================================")


def _load_case_curves(
    experiment_dir: Path,
    summary: dict[str, Any],
) -> list[tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]]:
    cases_raw = summary.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("Summary must contain a non-empty 'cases' list")

    loaded_cases = []
    for case_payload in cases_raw:
        if not isinstance(case_payload, dict):
            raise ValueError("Each summary case must be a JSON object")
        npz_name = case_payload.get("trajectory_npz")
        metrics_name = case_payload.get("trajectory_metrics_json")
        if not isinstance(npz_name, str) or not isinstance(metrics_name, str):
            raise ValueError("Summary case is missing trajectory paths")

        npz_path = experiment_dir / npz_name
        metrics_path = experiment_dir / metrics_name
        if not npz_path.is_file():
            raise FileNotFoundError(f"Trajectory file not found: {npz_path}")
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Trajectory metrics file not found: {metrics_path}"
            )
        pos_arr, speed_arr, metrics = load_optimized_curve_and_metrics(
            npz_path=str(npz_path),
            metrics_path=str(metrics_path),
            dtype=np.float32,
            use_metrics_cache=False,
        )
        loaded_cases.append((case_payload, pos_arr, speed_arr, metrics))

    return loaded_cases


def _case_sort_key(item: tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]):
    case_payload, _, _, _ = item
    case = case_payload.get("case")
    delta = case.get("delta_time_s", 0.0) if isinstance(case, dict) else 0.0
    order = {0.0: 0, 10.0: 1, 20.0: 2, -10.0: 3, -20.0: 4}
    return (order.get(float(delta), 100), float(delta))


def _style_for_delta(delta_time_s: float) -> dict[str, Any]:
    if delta_time_s == 0.0:
        return {"color": "black", "linestyle": "-", "linewidth": 1.7}
    if delta_time_s == 10.0:
        return {"color": "#9a3f3f", "linestyle": "-", "linewidth": 1.5}
    if delta_time_s == 20.0:
        return {"color": "blue", "linestyle": "--", "linewidth": 1.5}
    if delta_time_s == -10.0:
        return {"color": "green", "linestyle": "-", "linewidth": 1.5}
    if delta_time_s == -20.0:
        return {"color": "orange", "linestyle": "--", "linewidth": 1.5}
    return {"linestyle": "-", "linewidth": 1.5}


def _deduplicate_legend(ax: Any, *, loc: str = "best") -> None:
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles: list[Any] = []
    filtered_labels: list[str] = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels, strict=False):
        if not label or label.startswith("_") or label in seen:
            continue
        seen.add(label)
        filtered_handles.append(handle)
        filtered_labels.append(label)
    if filtered_handles:
        ax.legend(filtered_handles, filtered_labels, loc=loc)


def plot_schedule_change_result(
    *,
    experiment_dir: Path,
    summary: dict[str, Any],
    save_figure: bool,
    show: bool,
    figure_name: str,
    factor: float,
) -> str | None:
    loaded_cases = sorted(
        _load_case_curves(experiment_dir, summary),
        key=_case_sort_key,
    )
    safeguard = build_safeguard_utility(factor=factor)

    apply_rl_curve_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    safeguard.render(ax=ax, layers=SafeGuardUtility.DANGER_VIEW_LAYERS)

    all_pos: list[np.ndarray] = []
    all_speed_kmh: list[np.ndarray] = []
    original_curve: tuple[np.ndarray, np.ndarray] | None = None

    for case_payload, pos_arr, speed_arr, _metrics in loaded_cases:
        case = case_payload.get("case")
        delta = float(case.get("delta_time_s", 0.0)) if isinstance(case, dict) else 0.0
        label = str(case.get("label", f"{delta:g}s")) if isinstance(case, dict) else ""
        speed_kmh = np.asarray(speed_arr, dtype=np.float64) * 3.6
        style = _style_for_delta(delta)
        ax.plot(pos_arr, speed_kmh, label=label, **style)
        all_pos.append(np.asarray(pos_arr, dtype=np.float64))
        all_speed_kmh.append(speed_kmh)
        if delta == 0.0:
            original_curve = (np.asarray(pos_arr, dtype=np.float64), speed_kmh)

    trigger_positions = [
        float(case_payload["schedule_change_position_m"])
        for case_payload, _, _, _ in loaded_cases
        if case_payload.get("schedule_change_position_m") is not None
    ]
    if trigger_positions:
        trigger_pos = trigger_positions[0]
        if original_curve is not None:
            trigger_speed = float(
                np.interp(trigger_pos, original_curve[0], original_curve[1])
            )
        else:
            trigger_speeds = [
                float(case_payload["schedule_change_speed_mps"]) * 3.6
                for case_payload, _, _, _ in loaded_cases
                if case_payload.get("schedule_change_speed_mps") is not None
            ]
            trigger_speed = trigger_speeds[0] if trigger_speeds else 0.0
        ax.scatter(
            [trigger_pos],
            [trigger_speed],
            marker="*",
            s=80,
            color="red",
            label="schedule change",
            zorder=8,
        )

    if all_pos:
        pos_min = min(float(np.nanmin(pos)) for pos in all_pos)
        pos_max = max(float(np.nanmax(pos)) for pos in all_pos)
        margin = max((pos_max - pos_min) * 0.03, 1.0)
        ax.set_xlim(pos_min - margin, pos_max + margin)

    if all_speed_kmh:
        curve_ymax = max(float(np.nanmax(speed)) for speed in all_speed_kmh)
        speed_limit_ymax = float(np.nanmax(safeguard.speed_limits) * 3.6)
        ax.set_ylim(0.0, max(curve_ymax, speed_limit_ymax) * 1.08)

    ax.set_xlabel("Distance(m)")
    ax.set_ylabel("Velocity(km/h)")
    ax.grid(True, alpha=0.35)
    _deduplicate_legend(ax, loc="lower center")
    fig.tight_layout()

    saved_path: str | None = None
    if save_figure:
        figure_path = experiment_dir / figure_name
        fig.savefig(figure_path)
        saved_path = str(figure_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def run_show(args: argparse.Namespace) -> None:
    experiment_dir = resolve_schedule_change_experiment_dir(args.load_dir)
    summary = load_schedule_change_summary(experiment_dir)
    saved_path = plot_schedule_change_result(
        experiment_dir=experiment_dir,
        summary=summary,
        save_figure=bool(args.save_figure),
        show=bool(args.show),
        figure_name=args.figure_name,
        factor=float(args.factor),
    )

    print("========== Schedule-Time Change Result ==========")
    print(f"  experiment_dir: {experiment_dir}")
    print(f"  summary_json:   {experiment_dir / SUMMARY_FILENAME}")
    if saved_path:
        print(f"  figure:         {saved_path}")
    print("=================================================")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "show":
        run_show(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
