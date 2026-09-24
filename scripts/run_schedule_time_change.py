from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from contracts.environment import EpisodeInfo, EpisodeOutcome
from contracts.evaluation import EvaluationArtifact
from contracts.training import RunMetadata
from model.ocs import SafeGuardUtility
from rl.env_factory import make_env
from rl.evaluation import (
    PolicyEvaluationResult,
    classify_arrival_status,
    get_strict_stop_error_limit_m,
    get_strict_time_error_limit_s,
    save_policy_evaluation_curve,
)
from rl.experiment_utils import (
    DEFAULT_DEVICE,
    RL_FINAL_MODEL_FILENAME,
    load_run_metadata,
    reward_config_parameters,
)
from rl.reward_calculator import (
    PUNCTUALITY_POTENTIAL_SCALE,
    PUNCTUALITY_POTENTIAL_SIGMA_S,
    RewardConfig,
)
from scripts.run_method_ablation import (
    DEFAULT_SEEDS as METHOD_ABLATION_SEEDS,
    METHOD_ABLATION_MANIFEST_FILENAME,
    validate_method_ablation_manifest,
)
from utils.ablation import ManifestStore
from utils.io_utils import (
    format_float_token,
    load_evaluation_artifact,
)
from utils.plot_utils import (
    VIS_ACTUAL_PURPLE,
    VIS_DP_BLACK,
    VIS_HARD_LIMIT_RED,
    VIS_PROPOSED_ORANGE,
    apply_sci_curve_style,
    apply_sci_figure_layout,
    apply_sci_grid,
    save_sci_figure,
)
from utils.scenario import build_safeguard_utility, build_scenario

DEFAULT_OUTPUT_DIR = "output/paper_experiment/04_schedule_time_change"
SUMMARY_FILENAME = "schedule_time_change_summary.json"
DEFAULT_FIGURE_FILENAME = "schedule_time_change_comparison.pdf"
DEFAULT_DELTA_TIMES_S = (0.0, 30.0, -30.0)
DEFAULT_CHANGE_DISTANCE_M = 8_000.0
FULL_METHOD_VARIANT_ID = "ppo_pirs"
CANDIDATE_SOURCES: tuple[Literal["best", "final"], ...] = ("best", "final")
SUMMARY_ARTIFACT_TYPE = "schedule_time_change_selection"
SUMMARY_SCHEMA_VERSION = 1
RANK_KEY_FIELDS = (
    "feasible_case_count",
    "safe_case_count",
    "success_case_count",
    "precise_case_count",
    "punctual_case_count",
    "negative_safety_violation_count",
    "minimum_safety_margin_mps",
    "negative_max_stop_error_m",
    "negative_max_abs_time_error_s",
    "negative_mean_stop_error_m",
    "negative_mean_abs_time_error_s",
    "negative_mean_energy_j",
)


def _reward_config_from_metadata(snapshot: object) -> RewardConfig:
    values = asdict(snapshot)  # type: ignore[arg-type]
    scale = float(values.pop("punctuality_potential_scale"))
    sigma_s = float(values.pop("punctuality_potential_sigma_s"))
    potential_formula = values.pop("potential_transition_formula", None)
    terminal_next_potential = values.pop("terminal_next_potential", None)
    if scale != PUNCTUALITY_POTENTIAL_SCALE or sigma_s != PUNCTUALITY_POTENTIAL_SIGMA_S:
        raise ValueError(
            "selected policy uses retired punctuality-potential parameters; "
            "the fixed PIRS protocol requires K=5 and sigma=20 s"
        )
    if potential_formula not in {None, "gamma_phi_next_minus_phi_previous"}:
        raise ValueError(
            "selected policy uses an unsupported potential-shaping transition formula"
        )
    if terminal_next_potential not in {None, "observed_next_state"}:
        raise ValueError(
            "selected policy uses an unsupported terminal potential policy"
        )
    config = RewardConfig(
        enable_potential_safety=bool(values.pop("enable_potential_safety")),
        enable_potential_punctuality=bool(values.pop("enable_potential_punctuality")),
        reward_scheme=str(values.pop("reward_scheme")),
    )
    # Remaining entries are reward magnitudes, now fixed in code; the policy
    # must have been trained with exactly those values.
    fixed = reward_config_parameters(config)
    retired = {
        name: value for name, value in values.items() if fixed.get(name) != value
    }
    if retired:
        raise ValueError(
            f"selected policy was trained with retired reward magnitudes: {retired}"
        )
    return config


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
    abs_time_error_s: float
    stop_error_m: float
    total_energy_kj: float
    total_energy_j: float
    final_position_m: float
    final_speed_mps: float
    episode_steps: int
    min_safety_margin_mps: float
    mean_safety_margin_mps: float
    safety_violation_count: int
    safe: bool
    feasible: bool
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


@dataclass(frozen=True)
class ScheduleChangeCandidate:
    candidate_id: str
    run_id: str
    seed: int
    source: Literal["best", "final"]
    model_dir: Path
    metadata: RunMetadata


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: ScheduleChangeCandidate
    candidate_dir: Path
    results: tuple[ScheduleChangeRunResult, ...]
    rank_key: tuple[float, ...]


def parse_delta_times(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",")]
    deltas = tuple(float(part) for part in parts if part)
    if not deltas:
        raise argparse.ArgumentTypeError("delta list must not be empty")
    if len(deltas) != len(set(deltas)):
        raise argparse.ArgumentTypeError("delta list must not contain duplicates")
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
    if not (root / SUMMARY_FILENAME).is_file():
        raise FileNotFoundError(
            f"Could not find '{SUMMARY_FILENAME}' directly in '{root}'"
        )
    return root


def load_schedule_change_candidates(
    method_ablation_dir: str | os.PathLike[str],
) -> tuple[ScheduleChangeCandidate, ...]:
    root = Path(method_ablation_dir)
    manifest = ManifestStore(
        root,
        matrix_id="method",
        filename=METHOD_ABLATION_MANIFEST_FILENAME,
    ).load()
    validate_method_ablation_manifest(manifest)

    candidates: list[ScheduleChangeCandidate] = []
    for run in manifest.runs:
        if run.variant_id != FULL_METHOD_VARIANT_ID:
            continue
        for source in CANDIDATE_SOURCES:
            policy_path = Path(run.artifacts.path_for(f"policy_{source}"))
            metadata_path = Path(
                run.artifacts.path_for(
                    "metadata_best" if source == "best" else "metadata"
                )
            )
            if not policy_path.is_file():
                raise FileNotFoundError(f"Candidate policy not found: {policy_path}")
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Candidate metadata not found: {metadata_path}"
                )
            model_dir = policy_path.parent
            candidates.append(
                ScheduleChangeCandidate(
                    candidate_id=f"{run.run_id}__{source}",
                    run_id=run.run_id,
                    seed=run.seed,
                    source=source,
                    model_dir=model_dir,
                    metadata=load_run_metadata(model_dir),
                )
            )

    expected_count = len(METHOD_ABLATION_SEEDS) * len(CANDIDATE_SOURCES)
    if len(candidates) != expected_count:
        raise ValueError(
            f"schedule-time evaluation requires {expected_count} complete-method "
            f"best/final candidates; found {len(candidates)}"
        )
    _validate_candidate_metadata(candidates)
    return tuple(candidates)


def _candidate_protocol(metadata: RunMetadata) -> tuple[object, ...]:
    reward_config = _reward_config_from_metadata(metadata.reward_config)
    return (
        metadata.schedule_time_s,
        metadata.step_distance,
        metadata.reward_discount,
        metadata.reward_preset_name,
        reward_config_parameters(reward_config),
    )


def _validate_candidate_metadata(
    candidates: list[ScheduleChangeCandidate],
) -> None:
    expected = _candidate_protocol(candidates[0].metadata)
    if candidates[0].metadata.reward_preset_name != "basic_safety_punctuality":
        raise ValueError("complete-method candidates must use the PIRS reward preset")
    for candidate in candidates[1:]:
        if _candidate_protocol(candidate.metadata) != expected:
            raise ValueError(
                "schedule-time candidates use inconsistent training metadata: "
                f"{candidate.candidate_id}"
            )


def load_schedule_change_summary(
    experiment_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    summary_path = Path(experiment_dir) / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if payload.get("artifact_type") != SUMMARY_ARTIFACT_TYPE:
        raise ValueError("Schedule-change summary has an unsupported artifact type")
    if payload.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("Schedule-change summary has an unsupported schema version")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(
        METHOD_ABLATION_SEEDS
    ) * len(CANDIDATE_SOURCES):
        raise ValueError("Schedule-change summary must contain all ten candidates")
    if any(not isinstance(candidate, dict) for candidate in candidates):
        raise ValueError("Schedule-change summary candidates must be JSON objects")
    if payload.get("candidate_count") != len(candidates):
        raise ValueError("Schedule-change summary candidate count is inconsistent")
    selected = payload.get("selected")
    selected_id = selected.get("candidate_id") if isinstance(selected, dict) else None
    first_candidate = candidates[0]
    assert isinstance(first_candidate, dict)
    if selected_id != first_candidate.get("candidate_id"):
        raise ValueError("Schedule-change summary selected candidate is inconsistent")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Schedule-change summary must contain selected cases")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or show RL schedule-time change experiments."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run batch evaluation with an in-run schedule-time change.",
    )
    _ = evaluate_parser.add_argument(
        "--method-ablation-dir",
        type=Path,
        required=True,
        help="Completed method-ablation result directory containing manifest.json.",
    )
    _ = evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New YYYYMMDD_NN result directory; existing paths are rejected.",
    )
    _ = evaluate_parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device used to load PPO.",
    )
    _ = evaluate_parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions.",
    )
    _ = evaluate_parser.add_argument(
        "--change-distance-m",
        type=float,
        default=DEFAULT_CHANGE_DISTANCE_M,
        help="Track position at which the schedule time changes.",
    )
    _ = evaluate_parser.add_argument(
        "--delta-times-s",
        type=parse_delta_times,
        default=DEFAULT_DELTA_TIMES_S,
        help="Comma-separated schedule-time deltas, e.g. 0,30,-30.",
    )
    _ = evaluate_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve configuration and paths without loading the model.",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Show a saved schedule-time-change evaluation result.",
    )
    _ = show_parser.add_argument(
        "--load-dir",
        type=Path,
        required=True,
        help="Exact YYYYMMDD_NN result directory containing the root summary.",
    )
    _ = show_parser.add_argument(
        "--save-figure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the comparison figure into the experiment directory.",
    )
    _ = show_parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the comparison figure window.",
    )
    _ = show_parser.add_argument(
        "--factor",
        type=float,
        default=0.99,
        help="Safeguard factor used for rendering the safety background.",
    )

    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    return _build_arg_parser()


def _make_experiment_dir(output_dir: str | os.PathLike[str]) -> Path:
    experiment_dir = Path(output_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def _as_batch_observation(raw_obs: Any) -> np.ndarray:
    raw_obs_arr = np.asarray(raw_obs, dtype=np.float32)
    if raw_obs_arr.ndim == 1:
        raw_obs_arr = raw_obs_arr.reshape(1, -1)
    return raw_obs_arr


def _run_one_case(
    *,
    model: PPO,
    load_dir: str,
    experiment_dir: Path,
    case: ScheduleChangeCase,
    schedule_time_s: float,
    reward_discount: float,
    step_distance: float,
    reward_preset_name: str,
    reward_config: Any,
    deterministic: bool,
    change_distance_m: float,
) -> ScheduleChangeRunResult:
    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=schedule_time_s
    )

    venv_eval = DummyVecEnv(
        [
            lambda: make_env(
                vehicle=vehicle,
                track=track,
                safeguard_utility=safeguard_utility,
                train_service=train_service,
                gamma=reward_discount,
                step_distance=step_distance,
                enable_trajectory_tracking=False,
                reward_config=reward_config,
            )
        ]
    )
    total_reward = 0.0
    episode_steps = 0
    start_position_m = float(train_service.start_position)
    target_position_m = float(train_service.target_position)
    trajectory_position_seq: list[float] = [start_position_m]
    trajectory_speed_seq: list[float] = [0.0]
    safety_margins: list[float] = []
    safety_violation_positions_m: list[float] = []

    obs = venv_eval.reset()
    episode_over = False
    last_info: dict[str, object] = {}
    previous_position_m = start_position_m
    current_position_m = start_position_m
    current_speed_mps = 0.0
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
            obs = _as_batch_observation(raw_obs)
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
            episode_payload = last_info.get("episode")
            if not isinstance(episode_payload, dict):
                raise ValueError(
                    "Environment info is missing canonical episode payload"
                )
            episode_info = EpisodeInfo.from_mapping(episode_payload)
            current_position_m = episode_info.position_m
            current_speed_mps = episode_info.speed_mps
            trajectory_position_seq.append(current_position_m)
            trajectory_speed_seq.append(current_speed_mps)
            safety_margin_mps = float(last_info["safety_margin_mps"])
            safety_margins.append(safety_margin_mps)
            if safety_margin_mps < 0.0:
                safety_violation_positions_m.append(current_position_m)

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
                obs = _as_batch_observation(raw_obs)
                change_triggered = True
                change_step = episode_steps
                change_position_m = current_position_m
                change_speed_mps = current_speed_mps
    finally:
        venv_eval.close()

    target_time_s = float(train_service.schedule_time)
    episode_payload = last_info.get("episode")
    if not isinstance(episode_payload, dict):
        raise ValueError("Environment info is missing canonical episode payload")
    episode_info = EpisodeInfo.from_mapping(episode_payload)
    final_position_m = float(episode_info.position_m)
    final_speed_mps = float(episode_info.speed_mps)
    total_time_s = float(episode_info.operation_time_s)
    total_energy_j = float(episode_info.energy_consumption_j)
    total_energy_kj = total_energy_j / 1000.0
    stop_error_m = abs(target_position_m - final_position_m)
    time_error_s = total_time_s - target_time_s
    outcome_payload = last_info.get("outcome")
    if not isinstance(outcome_payload, dict):
        raise ValueError("Environment info is missing canonical outcome payload")
    outcome = EpisodeOutcome.from_mapping(outcome_payload)
    terminated = outcome.terminated
    truncated = outcome.truncated
    success, precise_arrival, punctual_arrival = classify_arrival_status(
        stop_error_m=stop_error_m,
        time_error_s=time_error_s,
        final_speed_mps=final_speed_mps,
        train_service=train_service,
        terminated=terminated,
        truncated=truncated,
    )

    comfort_tav = float(episode_info.comfort_tav)
    comfort_er_pct = float(episode_info.comfort_er_pct)
    comfort_rms = float(episode_info.comfort_rms)
    min_safety_margin_mps = min(safety_margins) if safety_margins else 0.0
    mean_safety_margin_mps = float(np.mean(safety_margins)) if safety_margins else 0.0

    evaluation_result = PolicyEvaluationResult(
        success=success,
        precise_arrival=precise_arrival,
        punctual_arrival=punctual_arrival,
        total_reward=float(total_reward),
        total_time_s=total_time_s,
        target_time_s=target_time_s,
        total_energy_j=total_energy_j,
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
        terminated=terminated,
        truncated=truncated,
        episode_steps=episode_steps,
        trajectory_pos_m=np.asarray(trajectory_position_seq, dtype=np.float32),
        trajectory_speed_mps=np.asarray(trajectory_speed_seq, dtype=np.float32),
        min_safety_margin_mps=min_safety_margin_mps,
        mean_safety_margin_mps=mean_safety_margin_mps,
        safety_violation_positions_m=np.asarray(
            safety_violation_positions_m, dtype=np.float32
        ),
    )
    case_dir = experiment_dir / case.token
    case_dir.mkdir(parents=True, exist_ok=False)
    npz_path = case_dir / "trajectory.npz"
    saved_npz_path, saved_json_path = save_policy_evaluation_curve(
        evaluation_result,
        str(npz_path),
        extra_metrics={
            "trajectory_source": "schedule_time_change",
            "evaluation_model_dir": os.path.relpath(load_dir, case_dir),
            "reward_preset_name": reward_preset_name,
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
            "step_distance": float(step_distance),
            "reward_discount": float(reward_discount),
            "deterministic": bool(deterministic),
        },
        metrics_path=str(case_dir / "metrics.json"),
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
        abs_time_error_s=abs(time_error_s),
        stop_error_m=stop_error_m,
        total_energy_kj=total_energy_kj,
        total_energy_j=total_energy_j,
        final_position_m=final_position_m,
        final_speed_mps=final_speed_mps,
        episode_steps=episode_steps,
        min_safety_margin_mps=evaluation_result.min_safety_margin_mps,
        mean_safety_margin_mps=evaluation_result.mean_safety_margin_mps,
        safety_violation_count=evaluation_result.safety_violation_count,
        safe=evaluation_result.safe,
        feasible=evaluation_result.feasible,
        schedule_change_triggered=bool(change_triggered),
        schedule_change_step=change_step,
        schedule_change_position_m=change_position_m,
        schedule_change_speed_mps=change_speed_mps,
        trajectory_npz=os.path.relpath(saved_npz_path, experiment_dir),
        trajectory_metrics_json=os.path.relpath(saved_json_path, experiment_dir),
    )


def summarize_candidate_results(
    results: tuple[ScheduleChangeRunResult, ...],
) -> dict[str, float | int]:
    if not results:
        raise ValueError("candidate evaluation must contain at least one case")
    return {
        "case_count": len(results),
        "feasible_case_count": sum(result.feasible for result in results),
        "safe_case_count": sum(result.safe for result in results),
        "success_case_count": sum(result.success for result in results),
        "precise_case_count": sum(result.precise_arrival for result in results),
        "punctual_case_count": sum(result.punctual_arrival for result in results),
        "safety_violation_count": sum(
            result.safety_violation_count for result in results
        ),
        "minimum_safety_margin_mps": min(
            result.min_safety_margin_mps for result in results
        ),
        "max_stop_error_m": max(result.stop_error_m for result in results),
        "max_abs_time_error_s": max(result.abs_time_error_s for result in results),
        "mean_stop_error_m": float(
            np.mean([result.stop_error_m for result in results])
        ),
        "mean_abs_time_error_s": float(
            np.mean([result.abs_time_error_s for result in results])
        ),
        "mean_energy_j": float(np.mean([result.total_energy_j for result in results])),
        "mean_energy_kj": float(
            np.mean([result.total_energy_kj for result in results])
        ),
        "mean_total_reward": float(
            np.mean([result.total_reward for result in results])
        ),
    }


def build_candidate_rank_key(
    results: tuple[ScheduleChangeRunResult, ...],
) -> tuple[float, ...]:
    aggregate = summarize_candidate_results(results)
    return (
        float(aggregate["feasible_case_count"]),
        float(aggregate["safe_case_count"]),
        float(aggregate["success_case_count"]),
        float(aggregate["precise_case_count"]),
        float(aggregate["punctual_case_count"]),
        -float(aggregate["safety_violation_count"]),
        float(aggregate["minimum_safety_margin_mps"]),
        -float(aggregate["max_stop_error_m"]),
        -float(aggregate["max_abs_time_error_s"]),
        -float(aggregate["mean_stop_error_m"]),
        -float(aggregate["mean_abs_time_error_s"]),
        -float(aggregate["mean_energy_j"]),
    )


def rank_candidate_evaluations(
    evaluations: list[CandidateEvaluation],
) -> list[CandidateEvaluation]:
    ordered = sorted(
        evaluations,
        key=lambda item: (
            item.candidate.source != "best",
            item.candidate.run_id,
        ),
    )
    ordered.sort(key=lambda item: item.rank_key, reverse=True)
    return ordered


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    return path


def _candidate_summary(
    evaluation: CandidateEvaluation,
    *,
    experiment_dir: Path,
    deterministic: bool,
    change_distance_m: float,
) -> dict[str, Any]:
    candidate = evaluation.candidate
    metadata = candidate.metadata
    return {
        "artifact_type": "schedule_time_change_candidate",
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": candidate.candidate_id,
        "run_id": candidate.run_id,
        "seed": candidate.seed,
        "source": candidate.source,
        "evaluation_model_dir": os.path.relpath(
            candidate.model_dir, evaluation.candidate_dir
        ),
        "experiment_dir": os.path.relpath(evaluation.candidate_dir, experiment_dir),
        "initial_schedule_time_s": float(metadata.schedule_time_s),
        "reward_preset_name": metadata.reward_preset_name,
        "reward_config": reward_config_parameters(
            _reward_config_from_metadata(metadata.reward_config)
        ),
        "reward_discount": float(metadata.reward_discount),
        "step_distance": float(metadata.step_distance),
        "deterministic": deterministic,
        "change_distance_m": change_distance_m,
        "rank_key_fields": list(RANK_KEY_FIELDS),
        "rank_key": list(evaluation.rank_key),
        "aggregate_metrics": summarize_candidate_results(evaluation.results),
        "cases": [result.to_summary_case() for result in evaluation.results],
    }


def _root_case_payload(
    result: ScheduleChangeRunResult,
    *,
    candidate_dir: Path,
    experiment_dir: Path,
) -> dict[str, Any]:
    payload = result.to_summary_case()
    payload["trajectory_npz"] = os.path.relpath(
        candidate_dir / result.trajectory_npz, experiment_dir
    )
    payload["trajectory_metrics_json"] = os.path.relpath(
        candidate_dir / result.trajectory_metrics_json, experiment_dir
    )
    return payload


def _write_root_summary(
    *,
    experiment_dir: Path,
    method_ablation_dir: Path,
    ranked: list[CandidateEvaluation],
    deterministic: bool,
    change_distance_m: float,
) -> Path:
    selected = ranked[0]
    metadata = selected.candidate.metadata
    candidate_payloads = []
    for rank, evaluation in enumerate(ranked, start=1):
        candidate_payloads.append(
            {
                "rank": rank,
                "candidate_id": evaluation.candidate.candidate_id,
                "run_id": evaluation.candidate.run_id,
                "seed": evaluation.candidate.seed,
                "source": evaluation.candidate.source,
                "model_dir": os.path.relpath(
                    evaluation.candidate.model_dir, experiment_dir
                ),
                "summary": os.path.relpath(
                    evaluation.candidate_dir / SUMMARY_FILENAME, experiment_dir
                ),
                "rank_key": list(evaluation.rank_key),
                "aggregate_metrics": summarize_candidate_results(evaluation.results),
            }
        )
    payload = {
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": ".",
        "source_method_ablation_dir": os.path.relpath(
            method_ablation_dir, experiment_dir
        ),
        "candidate_variant_id": FULL_METHOD_VARIANT_ID,
        "candidate_sources": list(CANDIDATE_SOURCES),
        "candidate_count": len(ranked),
        "ranking_rule": (
            "robust constraint counts, safety severity, worst and mean stop/time "
            "errors, then mean energy; best source and run_id break exact ties"
        ),
        "rank_key_fields": list(RANK_KEY_FIELDS),
        "selected": candidate_payloads[0],
        "candidates": candidate_payloads,
        "initial_schedule_time_s": float(metadata.schedule_time_s),
        "reward_preset_name": metadata.reward_preset_name,
        "reward_config": reward_config_parameters(
            _reward_config_from_metadata(metadata.reward_config)
        ),
        "reward_discount": float(metadata.reward_discount),
        "step_distance": float(metadata.step_distance),
        "deterministic": deterministic,
        "change_distance_m": change_distance_m,
        "cases": [
            _root_case_payload(
                result,
                candidate_dir=selected.candidate_dir,
                experiment_dir=experiment_dir,
            )
            for result in selected.results
        ],
    }
    return _write_json(experiment_dir / SUMMARY_FILENAME, payload)


def run_evaluate(args: argparse.Namespace) -> None:
    method_ablation_dir = Path(args.method_ablation_dir)
    candidates = load_schedule_change_candidates(method_ablation_dir)
    cases = tuple(build_schedule_change_case(delta) for delta in args.delta_times_s)
    first_metadata = candidates[0].metadata
    reward_config = _reward_config_from_metadata(first_metadata.reward_config)

    if args.dry_run:
        print("========== Schedule-Time Change Dry Run ==========")
        print(f"  method_ablation_dir: {method_ablation_dir}")
        print(f"  output_dir:          {args.output_dir}")
        print(f"  candidate_count:     {len(candidates)}")
        print(f"  reward_preset:       {first_metadata.reward_preset_name}")
        print(f"  schedule_time_s:     {first_metadata.schedule_time_s:.2f}")
        print(f"  reward_discount:     {first_metadata.reward_discount:.4f}")
        print(f"  step_distance:       {first_metadata.step_distance:.2f}")
        print(f"  change_distance_m:   {args.change_distance_m:.2f}")
        print(f"  delta_times_s:       {args.delta_times_s}")
        print(f"  deterministic:       {args.deterministic}")
        print("  candidates:")
        for candidate in candidates:
            print(f"    - {candidate.candidate_id}: model_dir={candidate.model_dir}")
        return

    experiment_dir = _make_experiment_dir(args.output_dir)
    candidates_root = experiment_dir / "candidates"
    candidates_root.mkdir()
    evaluations: list[CandidateEvaluation] = []
    for candidate in candidates:
        candidate_dir = candidates_root / candidate.candidate_id
        candidate_dir.mkdir()
        model = PPO.load(
            candidate.model_dir / RL_FINAL_MODEL_FILENAME,
            device=args.device,
        )
        results = tuple(
            _run_one_case(
                model=model,
                load_dir=str(candidate.model_dir),
                experiment_dir=candidate_dir,
                case=case,
                schedule_time_s=float(candidate.metadata.schedule_time_s),
                reward_discount=float(candidate.metadata.reward_discount),
                step_distance=float(candidate.metadata.step_distance),
                reward_preset_name=candidate.metadata.reward_preset_name,
                reward_config=reward_config,
                deterministic=bool(args.deterministic),
                change_distance_m=float(args.change_distance_m),
            )
            for case in cases
        )
        evaluation = CandidateEvaluation(
            candidate=candidate,
            candidate_dir=candidate_dir,
            results=results,
            rank_key=build_candidate_rank_key(results),
        )
        evaluations.append(evaluation)
        _write_json(
            candidate_dir / SUMMARY_FILENAME,
            _candidate_summary(
                evaluation,
                experiment_dir=experiment_dir,
                deterministic=bool(args.deterministic),
                change_distance_m=float(args.change_distance_m),
            ),
        )

    ranked = rank_candidate_evaluations(evaluations)
    summary_path = _write_root_summary(
        experiment_dir=experiment_dir,
        method_ablation_dir=method_ablation_dir,
        ranked=ranked,
        deterministic=bool(args.deterministic),
        change_distance_m=float(args.change_distance_m),
    )
    selected = ranked[0]
    print("========== Schedule-Time Change Evaluation ==========")
    print(f"  experiment_dir:      {experiment_dir}")
    print(f"  summary_json:        {summary_path}")
    print(f"  evaluated_candidates:{len(ranked)}")
    print(f"  selected_candidate:  {selected.candidate.candidate_id}")
    print(f"  selected_rank_key:   {selected.rank_key}")
    print("=====================================================")


def _load_case_curves(
    experiment_dir: Path,
    summary: dict[str, Any],
) -> list[tuple[dict[str, Any], EvaluationArtifact]]:
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
        artifact = load_evaluation_artifact(
            npz_path=str(npz_path),
            metrics_path=str(metrics_path),
            dtype=np.float32,
            use_metrics_cache=False,
        )
        loaded_cases.append((case_payload, artifact))

    return loaded_cases


def _case_sort_key(
    item: tuple[dict[str, Any], EvaluationArtifact] | dict[str, Any],
) -> tuple[int, float]:
    case_payload = item[0] if isinstance(item, tuple) else item
    case = case_payload.get("case")
    delta = case.get("delta_time_s", 0.0) if isinstance(case, dict) else 0.0
    delta_value = float(delta)
    if delta_value == 0.0:
        return (0, 0.0)
    if delta_value > 0.0:
        return (1, abs(delta_value))
    return (2, abs(delta_value))


def build_schedule_change_table(
    experiment_dir: Path,
    summary: dict[str, Any],
) -> str:
    cases_raw = summary.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("Summary must contain a non-empty 'cases' list")

    loaded = sorted(cases_raw, key=_case_sort_key)
    header = (
        "| Schedule change | Final time error (s) | Stop error (m) "
        "| Trajectory energy (kWh) | Cumulative acceleration variation (m/s²) |"
    )
    separator = "| --- | --- | --- | --- | --- |"
    lines = [header, separator]

    for case_payload in loaded:
        case = case_payload.get("case", {})
        delta = float(case.get("delta_time_s", 0.0)) if isinstance(case, dict) else 0.0
        if delta == 0.0:
            change_label = "Original"
        elif delta > 0.0:
            change_label = f"+{delta:g} s"
        else:
            change_label = f"-{abs(delta):g} s"

        time_error_s = float(case_payload.get("time_error_s", 0.0))
        stop_error_m = float(case_payload.get("stop_error_m", 0.0))
        total_energy_j = float(case_payload.get("total_energy_j", 0.0))
        total_energy_kwh = total_energy_j / 3_600_000.0

        comfort_str = "—"
        metrics_rel = case_payload.get("trajectory_metrics_json")
        if metrics_rel:
            metrics_path = experiment_dir / metrics_rel
            if metrics_path.is_file():
                metrics_dict = json.loads(metrics_path.read_text(encoding="utf-8"))
                comfort_val = metrics_dict.get("comfort_tav")
                if comfort_val is not None:
                    comfort_str = f"{float(comfort_val):.4f}"

        lines.append(
            f"| {change_label} | {time_error_s:+.4f} | {stop_error_m:.4f} | "
            f"{total_energy_kwh:.4f} | {comfort_str} |"
        )

    lines.append(
        "\n*Note: The cumulative acceleration variation formula is "
        r"$\sum_t |a_t - a_{t-1}|$, with unit $\mathrm{m/s^2}$.*"
    )
    return "\n".join(lines) + "\n"


def _style_for_delta(delta_time_s: float) -> dict[str, Any]:
    if delta_time_s == 0.0:
        return {"color": VIS_DP_BLACK, "linestyle": "-", "linewidth": 1.7}
    if delta_time_s > 0.0:
        return {"color": VIS_PROPOSED_ORANGE, "linestyle": "-", "linewidth": 1.5}
    return {"color": VIS_ACTUAL_PURPLE, "linestyle": "--", "linewidth": 1.5}


def _add_schedule_change_legend(
    fig: plt.Figure,
    *,
    case_handles: list[Any],
    case_labels: list[str],
    trigger_handle: Any | None,
) -> None:
    """Add the shared legend without exposing safeguard-rendering layers."""
    handles = list(case_handles)
    labels = list(case_labels)
    if trigger_handle is not None:
        handles.append(trigger_handle)
        labels.append("Schedule change")
    if not handles:
        return

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.8,
    )


def plot_schedule_change_result(
    *,
    experiment_dir: Path,
    summary: dict[str, Any],
    save_figure: bool,
    show: bool,
    factor: float,
) -> str | None:
    loaded_cases = sorted(
        _load_case_curves(experiment_dir, summary),
        key=_case_sort_key,
    )
    safeguard = build_safeguard_utility(factor=factor)

    apply_sci_curve_style()
    fig, ax = plt.subplots()
    safeguard.render(ax=ax, layers=SafeGuardUtility.DANGER_VIEW_LAYERS)

    all_pos: list[np.ndarray] = []
    all_speed_kmh: list[np.ndarray] = []
    original_curve: tuple[np.ndarray, np.ndarray] | None = None
    case_handles: list[Any] = []
    case_labels: list[str] = []

    for case_payload, artifact in loaded_cases:
        case = case_payload.get("case")
        delta = float(case.get("delta_time_s", 0.0)) if isinstance(case, dict) else 0.0
        if delta == 0.0:
            label = "Original"
        else:
            sign = "+" if delta > 0.0 else "−"
            label = f"{sign}{abs(delta):g} s"
        pos_arr = artifact.trajectory.position_m
        speed_arr = artifact.trajectory.speed_mps
        speed_kmh = np.asarray(speed_arr, dtype=np.float64) * 3.6
        style = _style_for_delta(delta)
        (case_handle,) = ax.plot(pos_arr, speed_kmh, **style)
        case_handles.append(case_handle)
        case_labels.append(label)
        all_pos.append(np.asarray(pos_arr, dtype=np.float64))
        all_speed_kmh.append(speed_kmh)
        if delta == 0.0:
            original_curve = (np.asarray(pos_arr, dtype=np.float64), speed_kmh)

    trigger_positions = [
        float(case_payload["schedule_change_position_m"])
        for case_payload, _ in loaded_cases
        if case_payload.get("schedule_change_position_m") is not None
    ]
    trigger_legend_handle: Line2D | None = None
    if trigger_positions:
        trigger_pos = trigger_positions[0]
        if original_curve is not None:
            trigger_speed = float(
                np.interp(trigger_pos, original_curve[0], original_curve[1])
            )
        else:
            trigger_speeds = [
                float(case_payload["schedule_change_speed_mps"]) * 3.6
                for case_payload, _ in loaded_cases
                if case_payload.get("schedule_change_speed_mps") is not None
            ]
            trigger_speed = trigger_speeds[0] if trigger_speeds else 0.0
        _ = ax.scatter(
            [trigger_pos],
            [trigger_speed],
            marker="*",
            s=80,
            color=VIS_HARD_LIMIT_RED,
            zorder=8,
        )
        trigger_legend_handle = Line2D(
            [],
            [],
            marker="*",
            markersize=9,
            color=VIS_HARD_LIMIT_RED,
            linestyle="None",
        )

    if all_pos:
        pos_min = min(float(np.nanmin(pos)) for pos in all_pos)
        pos_max = max(float(np.nanmax(pos)) for pos in all_pos)
        margin = max((pos_max - pos_min) * 0.03, 1.0)
        _ = ax.set_xlim(pos_min - margin, pos_max + margin)

    if all_speed_kmh:
        curve_ymax = max(float(np.nanmax(speed)) for speed in all_speed_kmh)
        speed_limit_ymax = float(np.nanmax(safeguard.speed_limits) * 3.6)
        _ = ax.set_ylim(0.0, max(curve_ymax, speed_limit_ymax) * 1.08)

    _ = ax.set_xlabel("Position (m)")
    _ = ax.set_ylabel("Speed (km/h)")
    apply_sci_grid(ax)
    _add_schedule_change_legend(
        fig,
        case_handles=case_handles,
        case_labels=case_labels,
        trigger_handle=trigger_legend_handle,
    )
    apply_sci_figure_layout(
        fig,
        columns=2,
        height_in=3.8,
        left=0.09,
        right=0.97,
        bottom=0.15,
        top=0.88,
    )

    saved_path: str | None = None
    if save_figure:
        figure_path = experiment_dir / DEFAULT_FIGURE_FILENAME
        saved_path = str(save_sci_figure(fig, figure_path))
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
        factor=float(args.factor),
    )
    table_content = build_schedule_change_table(experiment_dir, summary)
    table_path = experiment_dir / "schedule_time_change_table.md"
    table_path.write_text(table_content, encoding="utf-8")

    print("========== Schedule-Time Change Result ==========")
    print(f"  experiment_dir: {experiment_dir}")
    print(f"  summary_json:   {experiment_dir / SUMMARY_FILENAME}")
    print(f"  table:          {table_path}")
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
