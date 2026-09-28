"""Run golden cases through the current implementation and collect outputs.

The driver only calls existing code and records what it returns. Environment
construction lives in ``build_env`` / ``build_dp_optimizer`` so that interface
changes only need to touch them.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mtto.domain.kinematics import accel_between
from mtto.domain.safeguard import dynamic_limits
from mtto.domain.scenario import Scenario, ScheduleChange, Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.domain.srtsp import interp_upper_speed
from mtto.dp.solver import VariableSpacingDPOptimizer
from mtto.evaluation.quality import assess
from mtto.rl.env import MTTOEnv, make_env
from mtto.rl.rewards import RewardBreakdown, build_reward_config
from mtto.rl.state import TerminationReason
from mtto.workflows.train import build_env_references
from paper.figures import load_paper_scenario, load_paper_task

from .cases import (
    ACTION_GENERATION_PRESET,
    GAMMA,
    MAX_GENERATED_STEPS,
    REWARD_PRESETS,
    SCHEDULE_TIME_S,
    STEP_DISTANCE_M,
    DPCase,
    RLCase,
    ScheduleChangeCase,
)

# Input action sequences are frozen and tracked in git.
ACTIONS_DIR = Path(__file__).parent / "actions"
# Recorded outputs are regression snapshots of the current implementation;
# they are not tracked in git (see README.md, "Golden 回归快照"). The path is
# resolved from the repository root regardless of the current working directory.
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "golden"

Numerics = dict[str, NDArray[Any]]
Flags = dict[str, Any]

BREAKDOWN_FIELDS = tuple(RewardBreakdown.__dataclass_fields__)
EPISODE_FIELDS = (
    "energy_consumption_j",
    "operation_time_s",
    "redundant_operation_time_s",
    "position_m",
    "speed_mps",
    "comfort_tav",
    "comfort_er_pct",
    "comfort_rms",
)


def build_env(
    preset: str,
    schedule_time_s: float = SCHEDULE_TIME_S,
    schedule_change: ScheduleChangeCase | None = None,
) -> MTTOEnv:
    scenario = load_paper_scenario()
    task = load_paper_task(schedule_time_s=schedule_time_s)
    if schedule_change is not None:
        task = replace(
            task,
            schedule_change=ScheduleChange(
                trigger_position_m=float(schedule_change.trigger_position_m),
                new_schedule_time_s=float(
                    schedule_time_s + schedule_change.delta_time_s
                ),
            ),
        )
    srtsp_lookup, normalization = build_env_references(scenario, task, STEP_DISTANCE_M)
    return make_env(
        scenario=scenario,
        task=task,
        gamma=GAMMA,
        srtsp_lookup=srtsp_lookup,
        normalization=normalization,
        step_distance=STEP_DISTANCE_M,
        reward_config=build_reward_config(preset),
        enable_safety_truncation_tracking=True,
    )


def build_dp_optimizer(case: DPCase) -> VariableSpacingDPOptimizer:
    scenario = load_paper_scenario()
    scenario_task = load_paper_task(schedule_time_s=case.schedule_time_s)
    task = Task(
        start_position_m=scenario_task.start_position_m,
        target_position_m=case.target_position_m,
        schedule_time_s=case.schedule_time_s,
        max_acc_change=scenario_task.max_acc_change,
        max_stop_error_m=scenario_task.max_stop_error_m,
        max_arr_time_error_s=scenario_task.max_arr_time_error_s,
    )
    return VariableSpacingDPOptimizer(
        scenario=scenario,
        task=task,
        delta_speed=case.delta_speed_mps,
        show_precompute_progress=False,
        precompute_mode="serial",
        stage_division="uniform",
        uniform_step_size=case.uniform_step_size_m,
        cache_dir=None,
    )


def generate_actions(case: RLCase) -> NDArray[np.float32]:
    """Run the case controller once; the result is frozen as golden input."""
    env = build_env(ACTION_GENERATION_PRESET)
    env.reset()
    actions: list[float] = []
    for _ in range(MAX_GENERATED_STEPS):
        action = np.float32(np.clip(case.controller(env, env.state), -1.0, 1.0))
        actions.append(float(action))
        _, _, terminated, truncated, _ = env.step(np.asarray([action]))
        if terminated or truncated:
            return np.asarray(actions, dtype=np.float32)
    raise RuntimeError(f"{case.name}: episode did not end")


def load_actions(name: str) -> NDArray[np.float32]:
    return np.load(ACTIONS_DIR / f"{name}.npy")


def _record_calls(env: MTTOEnv) -> tuple[list[Any], list[RewardBreakdown]]:
    transitions: list[Any] = []
    breakdowns: list[RewardBreakdown] = []
    transition = env.transition
    calculate = env.reward_calculator.calculate

    def recording_transition(*args, **kwargs):
        result = transition(*args, **kwargs)
        transitions.append(result)
        return result

    def recording_calculate(*args, **kwargs):
        breakdown = calculate(*args, **kwargs)
        breakdowns.append(breakdown)
        return breakdown

    env.transition = recording_transition
    env.reward_calculator.calculate = recording_calculate
    return transitions, breakdowns


def _run_episode(
    preset: str,
    actions: NDArray[np.float32],
    schedule_change: ScheduleChangeCase | None,
) -> tuple[Numerics, Flags]:
    env = build_env(preset, schedule_change=schedule_change)
    transitions, breakdowns = _record_calls(env)
    observation, _ = env.reset()
    initial_state = env.state
    observations = [observation]
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    safety_margins: list[float] = []
    change_step: int | None = 0 if initial_state.schedule_changed else None
    info: dict[str, Any] = {}

    for action in actions:
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([action], dtype=np.float32)
        )
        done = terminated or truncated
        if change_step is None and env.state.schedule_changed:
            change_step = len(rewards) + 1
        observations.append(observation)
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        safety_margins.append(float(info["safety_margin_mps"]))
        if done:
            break

    if len(rewards) != len(actions) or not done:
        raise RuntimeError("episode end does not match the frozen action sequence")

    safeguard = env.safeguard
    states = [initial_state, *(t.step_end_state for t in transitions)]
    guard_min_max = [
        dynamic_limits(safeguard, state.s_m, state.sps.target_stopping_point_index)
        for state in states
    ]
    srtsp_upper = [state.srtsp_limit_mps for state in states]
    propulsion = np.concatenate(
        ([0.0], np.cumsum([t.propulsion_delta_kj for t in transitions]))
    )
    levitation = np.concatenate(
        ([0.0], np.cumsum([t.levitation_delta_kj for t in transitions]))
    )
    safety = env.drain_safety_truncations()

    numerics: Numerics = {
        "position_m": np.array([s.s_m for s in states]),
        "speed_mps": np.array([s.v_mps for s in states]),
        "commanded_acceleration_mps2": np.array(
            [s.commanded_acceleration_mps2 for s in states]
        ),
        "time_s": np.array([s.t_s for s in states]),
        "redundant_time_s": np.array([s.slack_time_s for s in states]),
        "energy_kj": np.array([s.total_energy_kj for s in states]),
        "propulsion_energy_kj": propulsion,
        "levitation_energy_kj": levitation,
        "slope_permille": np.array([s.slope_permille for s in states]),
        "min_speed_mps": np.array([s.lower_limit_mps for s in states]),
        "max_speed_mps": np.array([s.max_speed_mps for s in states]),
        "guard_min_speed_mps": np.array([float(lo) for lo, _ in guard_min_max]),
        "guard_max_speed_mps": np.array([float(hi) for _, hi in guard_min_max]),
        "srtsp_max_speed_mps": np.array(srtsp_upper),
        "stop_error_m": np.array([s.stop_error_m for s in states]),
        "sps_request_started_at_s": np.array(
            [
                np.nan
                if s.sps.request_started_at_s is None
                else s.sps.request_started_at_s
                for s in states
            ]
        ),
        "action": np.asarray(actions, dtype=np.float32),
        "step_acceleration_mps2": np.array(
            [t.commanded_acceleration_mps2 for t in transitions]
        ),
        "step_distance_m": np.array([t.motion.distance_m for t in transitions]),
        "step_duration_s": np.array([t.motion.duration_s for t in transitions]),
        "step_energy_kj": np.array(
            [(t.propulsion_delta_kj + t.levitation_delta_kj) for t in transitions]
        ),
        "reward": np.array(rewards),
        "safety_margin_mps": np.array(safety_margins),
        "observation": np.asarray(observations, dtype=np.float32),
        "safety_truncation_position_m": safety["position_m"],
    }
    for field in BREAKDOWN_FIELDS:
        numerics[f"reward_{field}"] = np.array([getattr(b, field) for b in breakdowns])
    for field in EPISODE_FIELDS:
        numerics[f"episode_{field}"] = np.array([info["episode"][field]])

    stopped = [abs(t.step_end_state.v_mps) <= 0.01 for t in transitions]
    flags: Flags = {
        "steps": len(transitions),
        "terminated": terminated_flags,
        "truncated": truncated_flags,
        "termination_reason": [
            t.termination_reason.name if t.termination_reason is not None else None
            for t in transitions
        ],
        # Same thresholds as Task.stop_state, recorded so that
        # step 2 can split FAILED_STOP into STOPPED_SHORT / OVERRAN.
        "stopped": stopped,
        "within_stop_tolerance": [
            t.step_end_state.stop_error_m <= env.task.max_stop_error_m * 30
            for t in transitions
        ],
        "reached_target": [t.step_end_state.stop_error_m <= 1e-6 for t in transitions],
        # Which upper limit was exceeded; step 2 splits SPEED_HIGH with these.
        "over_guard_max": [
            t.step_end_state.v_mps > float(hi)
            for t, (_, hi) in zip(transitions, guard_min_max[1:], strict=True)
        ],
        "over_srtsp_max": [
            t.step_end_state.v_mps > limit
            for t, limit in zip(transitions, srtsp_upper[1:], strict=True)
        ],
        "sps_target_index": [s.sps.target_stopping_point_index for s in states],
        "sps_request_pending": [s.sps.request_pending for s in states],
        "episode_stopping_point_index": info["episode"]["stopping_point_index"],
        "safety_termination_reason": [
            TerminationReason(int(c)).name
            for c in safety["termination_reason"].tolist()
        ],
        "schedule_change_step": change_step,
        "final_schedule_time_s": env.state.schedule_time_s,
    }
    return numerics, flags


def drive_rl(
    actions: NDArray[np.float32],
    schedule_change: ScheduleChangeCase | None = None,
) -> tuple[Numerics, Flags]:
    numerics: Numerics = {}
    flags: Flags = {}
    for preset in REWARD_PRESETS:
        preset_numerics, preset_flags = _run_episode(preset, actions, schedule_change)
        numerics.update({f"{preset}.{k}": v for k, v in preset_numerics.items()})
        flags[preset] = preset_flags
    return numerics, flags


def _quality_snapshot(
    profile: SpeedProfile, scenario: Scenario, task: Task
) -> tuple[Numerics, Flags]:
    """Flatten an ``evaluation.quality.assess`` report for the golden snapshot."""
    report = assess(profile, scenario, task)
    metrics = report.metrics
    numerics: Numerics = {
        "quality.propulsion_energy_kj": np.array([metrics.propulsion_energy_kj]),
        "quality.levitation_energy_kj": np.array([metrics.levitation_energy_kj]),
        "quality.run_time_s": np.array([metrics.run_time_s]),
        "quality.stop_error_m": np.array([metrics.stop_error_m]),
        "quality.comfort_tav_mps2": np.array([metrics.comfort_tav_mps2]),
        "quality.comfort_rms_mps2": np.array([metrics.comfort_rms_mps2]),
        "quality.comfort_exceedance_pct": np.array([metrics.comfort_exceedance_pct]),
        "quality.arrival_time_error_s": np.array(
            [
                np.nan
                if metrics.arrival_time_error_s is None
                else metrics.arrival_time_error_s
            ]
        ),
        "quality.audit_min_margin_mps": np.array([report.audit.min_margin_mps]),
    }
    flags: Flags = {
        "completed": report.completed,
        "precise_stop": report.precise_stop,
        "safe": report.safe,
        "punctual": report.punctual,
        "feasible": report.feasible,
        "violation_count": len(report.audit.violations),
        "violation_kinds": sorted({v.kind.name for v in report.audit.violations}),
        "event_count": len(report.audit.events),
        "event_kinds": sorted({e.kind.name for e in report.audit.events}),
    }
    return numerics, flags


def drive_rl_case(actions: NDArray[np.float32]) -> tuple[Numerics, Flags]:
    """RL_CASES only: episode numerics plus a quality-assessment snapshot.

    Schedule-change cases are driven through ``drive_rl`` directly; work
    package 8 only requires quality snapshots for the plain RL and DP cases.
    """
    numerics, flags = drive_rl(actions)
    preset = REWARD_PRESETS[0]
    profile = SpeedProfile.from_arrays(
        numerics[f"{preset}.position_m"],
        numerics[f"{preset}.speed_mps"],
        numerics[f"{preset}.time_s"],
        numerics[f"{preset}.propulsion_energy_kj"],
        numerics[f"{preset}.levitation_energy_kj"],
    )
    quality_numerics, quality_flags = _quality_snapshot(
        profile, load_paper_scenario(), load_paper_task(schedule_time_s=SCHEDULE_TIME_S)
    )
    numerics.update(quality_numerics)
    flags["quality"] = quality_flags
    return numerics, flags


def require_snapshot(name: str) -> None:
    """Fail fast with a clear message when a golden snapshot has not been recorded."""
    if (
        not (OUTPUT_DIR / f"{name}.json").exists()
        or not (OUTPUT_DIR / f"{name}.npz").exists()
    ):
        raise FileNotFoundError(
            f"{name}: no recorded snapshot in {OUTPUT_DIR}; run "
            "`uv run python -m tests.golden.record` first"
        )


def manifest_context() -> str:
    """Summarize output/golden/manifest.json for golden-comparison failure messages."""
    manifest_path = OUTPUT_DIR / "manifest.json"
    if not manifest_path.exists():
        return f"no manifest.json in {OUTPUT_DIR}"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    return (
        f"snapshot recorded at mtto_version={manifest['mtto_version']} "
        f"git_commit={manifest['git_commit']} dirty={manifest['dirty']}"
    )


def drive_dp(case: DPCase) -> tuple[Numerics, Flags]:
    optimizer = build_dp_optimizer(case)
    task = optimizer.task
    profile = optimizer.optimize(
        start_pos=task.start_position_m,
        start_speed=0.0,
        target_pos=task.target_position_m,
        target_speed=0.0,
        schedule_time=case.schedule_time_s,
    )
    if profile is None:
        raise RuntimeError(f"{case.name}: DP returned no trajectory")
    position = profile.position_m
    speed = profile.speed_mps
    time = profile.time_s
    segments = []
    for k in range(position.size - 1):
        displacement = position[k + 1] - position[k]
        acc, duration = accel_between(speed[k], speed[k + 1], displacement)
        segments.append((acc, duration))
    upper = interp_upper_speed(optimizer.srtsp_curve, position)
    time_error = float(profile.time_s[-1]) - case.schedule_time_s
    numerics: Numerics = {
        "position_m": position,
        "speed_mps": speed,
        "time_s": time,
        "segment_acceleration_mps2": np.array([s[0] for s in segments]),
        "segment_duration_s": np.array([s[1] for s in segments]),
        "propulsion_energy_kj": profile.propulsion_energy_kj,
        "levitation_energy_kj": profile.levitation_energy_kj,
        "total_time_s": np.array([float(profile.time_s[-1])]),
        "total_energy_kj": np.array([float(profile.total_energy_kj[-1])]),
        "time_error_s": np.array([time_error]),
        # Interior nodes only: both endpoints sit at zero speed and zero limit.
        "min_upper_margin_mps": np.array([float(np.min((upper - speed)[1:-1]))]),
    }
    flags: Flags = {
        "node_count": int(position.size),
        "within_time_tolerance": abs(time_error) <= task.max_arr_time_error_s,
        "valid_edge_count": int(optimizer._graph_cache["total_valid_edges"]),
    }
    quality_numerics, quality_flags = _quality_snapshot(
        profile, optimizer.scenario, task
    )
    numerics.update(quality_numerics)
    flags["quality"] = quality_flags
    return numerics, flags
