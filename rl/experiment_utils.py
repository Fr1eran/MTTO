from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from contracts.evaluation import EvaluationMetrics
from contracts.training import (
    RewardConfigSnapshot,
    RunMetadata,
    TrainingBudget,
)
from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.callbacks import (
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    BestEvaluationArtifactHandler,
    CompletedEpisodeProgress,
    EpisodeProgressBarCallback,
    EvaluationHistoryArtifactHandler,
    RewardDiagnosticsArtifactCallback,
    SafetyTruncationPositionHistogramCallback,
    ScheduledPolicyEvaluationCallback,
    StopTrainingOnCompletedEpisodes,
)
from rl.env_factory import make_env
from rl.evaluation import build_single_eval_env, evaluate_and_save_final_policy
from rl.operational_stepper import OperationalStepper
from rl.reward_calculator import (
    COMFORT_REWARD_SCALE,
    ENERGY_REWARD_SCALE,
    LI_GOAL_REWARD_SCALE,
    PUNCTUALITY_POTENTIAL_SCALE,
    PUNCTUALITY_POTENTIAL_SIGMA_S,
    SAFETY_POTENTIAL_SCALE,
    SAFETY_POTENTIAL_STEEPNESS,
    SURVIVAL_REWARD_SCALE,
    RewardConfig,
)
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION
from rl.task_terminal_vec_env import TaskTerminalVecEnv
from rl.training_analysis import AnalysisConfig, run_training_analysis
from utils.io_utils import (
    format_float_token,
    load_evaluation_artifact,
    load_evaluation_metrics,
)
from utils.plot_utils import render_trajectory_on_axes
from utils.scenario import build_scenario
from utils.trajectory import OptimizedCurveArtifact
from utils.type_utils import as_float

__all__ = [
    # 常量
    "RL_FINAL_MODEL_FILENAME",
    "RUN_METADATA_FILENAME",
    "REWARD_DIAGNOSTICS_FILENAME",
    "EVALUATION_HISTORY_FILENAME",
    "DEFAULT_SCHEDULE_TIME_S",
    "DEFAULT_REWARD_DISCOUNT",
    "DEFAULT_EVALUATION_INTERVAL_ROLLOUTS",
    "DEFAULT_ROLLOUT_STEPS_PER_UPDATE",
    "DEFAULT_STEP_DISTANCE",
    "DEFAULT_NUM_ENVS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_DEVICE",
    "DEFAULT_REWARD_PRESET_NAME",
    "ENERGY_REWARD_SCALE",
    "COMFORT_REWARD_SCALE",
    "SURVIVAL_REWARD_SCALE",
    # dataclass
    "RewardPreset",
    "RunMetadata",
    "TrainingBudget",
    "TrainingRunSpec",
    # reward preset
    "reward_preset_names",
    "resolve_reward_preset",
    "build_reward_config",
    "reward_config_parameters",
    # 路径 & 元数据
    "resolve_output_dir",
    "resolve_tb_log_name",
    "build_run_metadata",
    "save_run_metadata",
    "load_run_metadata",
    # 训练配置
    "build_default_training_args",
    "resolve_run_mode",
    "resolve_log_interval",
    "resolve_training_run_spec",
    "learning_rate_schedule_parameters",
    # 训练
    "train_single_experiment",
    "evaluate_final_training_run",
    # 轨迹产物
    "resolve_rl_curve_artifact",
    "load_rl_curve_artifact",
    "load_rl_curve_metrics",
    # 轨迹对比
    "build_rl_trajectory_comparison_key",
    # 可视化
    "get_rl_trajectory_status_text",
    "format_rl_trajectory_terminal_summary",
    "render_rl_curve_on_axes",
]

# =============================================================================
# 文件路径常量
# =============================================================================

RL_MODEL_FILENAME = "policy.zip"
RL_FINAL_MODEL_FILENAME = RL_MODEL_FILENAME
RUN_METADATA_FILENAME = "metadata.json"
RL_TRAJECTORY_FILENAME = "trajectory.npz"
RL_METRICS_FILENAME = "metrics.json"
REWARD_DIAGNOSTICS_FILENAME = "episodes.npz"
EVALUATION_HISTORY_FILENAME = "evaluations.npz"
DEFAULT_TRAINING_EPISODES = 5_000
# =============================================================================
# 训练超参数常量
# =============================================================================

DEFAULT_SCHEDULE_TIME_S = 465.0
DEFAULT_REWARD_DISCOUNT = 0.998
DEFAULT_ROLLOUT_STEPS_PER_UPDATE = 8192
DEFAULT_STEP_DISTANCE = 30.0
DEFAULT_NUM_ENVS = 8
DEFAULT_BATCH_SIZE = 512
DEFAULT_N_EPOCHS = 8
DEFAULT_DEVICE = "cpu"
LEARNING_RATE_SCHEDULE_ID = "cosine_completed_episodes_v1"
STEP_LEARNING_RATE_SCHEDULE_ID = "cosine_environment_steps_v1"
INITIAL_LEARNING_RATE = 3e-4
FINAL_LEARNING_RATE = 1e-5
DEFAULT_REWARD_PRESET_NAME = "basic_safety_punctuality"

# =============================================================================
# 数据结构 (dataclass)
# =============================================================================


@dataclass(frozen=True)
class RewardPreset:
    """具名奖励预设，将实验身份与运行时奖励配置组合在一起。"""

    name: str
    label: str
    description: str
    config: RewardConfig

    def enabled_shaping_components(self) -> tuple[str, ...]:
        components: list[str] = []
        if self.config.enable_potential_safety:
            components.append("safety")
        if self.config.enable_potential_punctuality:
            components.append("punctuality")
        return tuple(components)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reward_preset_name": self.name,
            "reward_preset_label": self.label,
            "reward_preset_description": self.description,
            "potential_shaping_components": list(self.enabled_shaping_components()),
            "reward_config": reward_config_parameters(self.config),
        }


@dataclass(frozen=True)
class TrainingRunSpec:
    """单次训练运行的完整配置快照，由 CLI 参数解析得到。"""

    schedule_time_s: float
    step_distance: float
    reward_discount: float
    reward_preset: RewardPreset
    output_root: str
    output_dir: str
    final_output_dir: str
    best_eval_output_dir: str
    reward_diagnostics_path: str
    final_model_save_path: str
    run_metadata_path: str
    run_metadata: RunMetadata
    run_mode: str
    enable_tb: bool
    enable_monitor: bool
    enable_auto_analysis: bool
    enable_best_evaluation_artifacts: bool
    tb_log_name: str
    tensorboard_log_dir: str
    log_interval: int
    num_envs: int
    n_steps_per_env: int
    rollout_steps_per_update: int
    evaluation_interval_rollouts: int | None
    evaluation_interval_episodes: int | None
    evaluation_deterministic: bool
    evaluation_history_path: str | None
    enable_safety_truncation_histogram: bool
    safety_truncation_bin_size_m: float
    budget_mode: Literal["completed_episodes", "environment_steps"]
    training_episodes: int | None
    training_rollouts: int | None
    max_episode_steps: int
    total_timesteps: int
    device: str
    seed: int | None
    dry_run: bool

    @property
    def reward_config(self) -> RewardConfig:
        """Return the runtime config owned by the selected reward preset."""
        return self.reward_preset.config


# =============================================================================
# Reward Preset
# =============================================================================


def goal_reward_scale(reward_config: RewardConfig) -> float:
    """Fixed goal-state multiplier recorded in metadata for the reward scheme."""
    if reward_config.reward_scheme == "li2023_scaled":
        return LI_GOAL_REWARD_SCALE
    return 1.0


def reward_config_parameters(reward_config: RewardConfig) -> dict[str, Any]:
    """Return the reward switches plus the fixed reward magnitudes."""
    return {
        "energy_reward_scale": ENERGY_REWARD_SCALE,
        "comfort_reward_scale": COMFORT_REWARD_SCALE,
        "enable_potential_safety": bool(reward_config.enable_potential_safety),
        "survival_reward_scale": SURVIVAL_REWARD_SCALE,
        "safety_potential_scale": SAFETY_POTENTIAL_SCALE,
        "safety_potential_steepness": SAFETY_POTENTIAL_STEEPNESS,
        "enable_potential_punctuality": reward_config.enable_potential_punctuality,
        "punctuality_potential_scale": PUNCTUALITY_POTENTIAL_SCALE,
        "punctuality_potential_sigma_s": PUNCTUALITY_POTENTIAL_SIGMA_S,
        "potential_transition_formula": "gamma_phi_next_minus_phi_previous",
        "terminal_next_potential": "observed_next_state",
        "reward_scheme": reward_config.reward_scheme,
        "goal_reward_scale": goal_reward_scale(reward_config),
    }


REWARD_PRESETS: dict[str, RewardPreset] = {
    "basic": RewardPreset(
        name="basic",
        label="basic",
        description="Base reward only: energy and comfort are always enabled.",
        config=RewardConfig(enable_potential_safety=False),
    ),
    "basic_safety": RewardPreset(
        name="basic_safety",
        label="basic+safety",
        description="Base reward plus potential-based safety shaping.",
        config=RewardConfig(enable_potential_safety=True),
    ),
    "basic_punctuality": RewardPreset(
        name="basic_punctuality",
        label="basic+punctuality",
        description="Base reward plus linear-slack punctuality potential.",
        config=RewardConfig(
            enable_potential_safety=False,
            enable_potential_punctuality=True,
        ),
    ),
    "basic_safety_punctuality": RewardPreset(
        name="basic_safety_punctuality",
        label="basic+safety+punctuality",
        description=(
            "Base reward plus Physics-Informed Reward Shaping (PIRS), combining "
            "safety and linear-slack punctuality potentials."
        ),
        config=RewardConfig(enable_potential_punctuality=True),
    ),
    "li2023_scaled": RewardPreset(
        name="li2023_scaled",
        label="Li et al. (2023), goal reward rescaled",
        description=(
            "Binary goal-directed reward of Li et al. (2023) with the published "
            "per-step penalties and tuned coefficients; T_lim set to the 10 s "
            "strict punctuality tolerance; every goal-state term multiplied by "
            "(1 - 0.99) / (1 - 0.998) = 5 so that r_g keeps the margin over "
            "r_inf / (1 - gamma) that Li et al. require (their gamma = 0.99)."
        ),
        config=RewardConfig(
            enable_potential_safety=False, reward_scheme="li2023_scaled"
        ),
    ),
}

REWARD_PRESET_ALIASES: dict[str, str] = {
    "default": DEFAULT_REWARD_PRESET_NAME,
    "basic": "basic",
    "basic+safety": "basic_safety",
    "basic+punctuality": "basic_punctuality",
    "basic+safety+punctuality": "basic_safety_punctuality",
}


def reward_preset_names() -> tuple[str, ...]:
    """返回所有已注册奖励情形的名称元组。"""
    return tuple(REWARD_PRESETS.keys())


def _normalize_reward_preset_token(preset_name: str | None) -> str:
    if preset_name is None:
        return DEFAULT_REWARD_PRESET_NAME
    normalized = str(preset_name).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return DEFAULT_REWARD_PRESET_NAME
    return normalized


def resolve_reward_preset(preset_name: str | None = None) -> RewardPreset:
    """将奖励情形名称（含别名）解析为 RewardPreset 实例。

    Args:
        preset_name: 预设名，支持 ``basic``、``basic_safety``、
            ``basic_safety_punctuality`` 和 ``default``。

    Returns:
        对应的 RewardPreset 实例。

    Raises:
        ValueError: 未知的情形名。
    """
    normalized = _normalize_reward_preset_token(preset_name)
    canonical = REWARD_PRESET_ALIASES.get(normalized, normalized)
    preset = REWARD_PRESETS.get(canonical)
    if preset is None:
        available = ", ".join(reward_preset_names())
        raise ValueError(
            f"Unknown reward preset '{preset_name}'. Available presets: {available}"
        )
    return preset


def build_reward_config(preset_name: str | None = None) -> RewardConfig:
    """根据奖励情形名返回 RewardConfig 实例。

    Args:
        preset_name: 预设名，同 resolve_reward_preset。
    Returns:
        用于初始化 MTTOEnv 的 RewardConfig。
    """
    return resolve_reward_preset(preset_name).config


# =============================================================================
# 实验命名 & 路径解析
# =============================================================================


def _sanitize_identifier_token(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("identifier token cannot be empty")
    return normalized


def _build_experiment_token(
    *,
    schedule_time_s: float,
    step_distance: float,
    reward_preset_name: str | None = None,
    experiment_tag: str | None = None,
) -> str:
    schedule_token = format_float_token(schedule_time_s)
    step_token = format_float_token(step_distance)
    preset = resolve_reward_preset(reward_preset_name)

    tokens = [f"{schedule_token}_{step_token}", preset.name]
    if experiment_tag:
        tokens.append(_sanitize_identifier_token(experiment_tag))
    return "__".join(tokens)


def resolve_output_dir(
    *,
    output_root: str,
    schedule_time_s: float,
    step_distance: float,
    reward_preset_name: str | None = None,
    experiment_tag: str | None = None,
) -> str:
    """根据实验参数解析输出目录路径。

    Args:
        output_root: 输出根目录。
        schedule_time_s: 规划运行时间 (s)。
        step_distance: 固定空间控制步长 (m)。
        reward_preset_name: 奖励预设名。
        experiment_tag: 实验标签。

    Returns:
        拼接后的输出目录路径字符串。
    """
    experiment_token = _build_experiment_token(
        schedule_time_s=schedule_time_s,
        step_distance=step_distance,
        reward_preset_name=reward_preset_name,
        experiment_tag=experiment_tag,
    )
    return os.path.join(output_root, experiment_token)


def resolve_tb_log_name(
    *,
    tb_log_name: str | None,
    run_mode: str,
    schedule_time_s: float,
    step_distance: float,
    reward_preset_name: str | None = None,
    experiment_tag: str | None = None,
) -> str:
    """解析 TensorBoard 日志名称。

    Args:
        tb_log_name: 用户指定的日志名（优先使用）。
        run_mode: 运行模式 (tune/reproduce/monitor_best/best_only)。
        schedule_time_s: 规划运行时间 (s)。
        step_distance: 固定空间控制步长 (m)。
        reward_preset_name: 奖励预设名。
        experiment_tag: 实验标签。

    Returns:
        TensorBoard 日志名称字符串。
    """
    if tb_log_name is not None and tb_log_name.strip():
        return tb_log_name.strip()

    experiment_token = _build_experiment_token(
        schedule_time_s=schedule_time_s,
        step_distance=step_distance,
        reward_preset_name=reward_preset_name,
        experiment_tag=experiment_tag,
    )
    return f"train_log__{_sanitize_identifier_token(run_mode)}__{experiment_token}"


# =============================================================================
# 运行元数据管理
# =============================================================================


def build_run_metadata(
    *,
    reward_preset: RewardPreset,
    schedule_time_s: float,
    step_distance: float,
    reward_discount: float,
    run_mode: str | None = None,
    experiment_tag: str | None = None,
    training_episodes: int | None = None,
    training_rollouts: int | None = None,
    budget_mode: Literal["completed_episodes", "environment_steps"] = (
        "completed_episodes"
    ),
    max_episode_steps: int | None = None,
    derived_total_timesteps: int | None = None,
    enable_tb: bool | None = None,
    enable_monitor: bool | None = None,
    enable_auto_analysis: bool | None = None,
    enable_best_evaluation_artifacts: bool | None = None,
    enable_safety_truncation_histogram: bool | None = None,
    safety_truncation_bin_size_m: float | None = None,
    evaluation_interval_rollouts: int | None = None,
    evaluation_interval_episodes: int | None = None,
    evaluation_deterministic: bool | None = None,
    evaluation_history_path: str | None = None,
    num_envs: int | None = None,
    n_steps_per_env: int | None = None,
    rollout_steps_per_update: int | None = None,
    output_dir: str | None = None,
    final_output_dir: str | None = None,
    reward_diagnostics_path: str | None = None,
    best_eval_output_dir: str | None = None,
    tensorboard_log_dir: str | None = None,
    tb_log_name: str | None = None,
    seed: int | None = None,
) -> RunMetadata:
    """构建单次训练运行的类型化元数据。

    Args:
        reward_preset: 具名奖励预设。
        schedule_time_s: 规划运行时间 (s)。
        step_distance: 固定空间控制步长 (m)。
        reward_discount: 奖励折扣因子 γ。
        run_mode: 运行模式。
        experiment_tag: 实验标签。
        training_episodes: 请求的全局完成训练回合数。
        max_episode_steps: 训练任务的理论单回合最大步数。
        derived_total_timesteps: 由回合预算推导、传入 PPO 的环境步上限。
        enable_tb: 是否启用 TensorBoard。
        enable_monitor: 是否启用 VecMonitor。
        enable_auto_analysis: 是否启用训练后自动分析。
        enable_best_evaluation_artifacts: 是否保存最优评估产物。
        num_envs: 训练环境数量。
        n_steps_per_env: 每个环境的 rollout 步数。
        rollout_steps_per_update: 单次 PPO 更新的总 rollout 步数。
        output_dir: 输出目录。
        final_output_dir: 最终产出目录。
        best_eval_output_dir: 最优轨迹评估产出目录。
        tensorboard_log_dir: TensorBoard 日志目录。
        tb_log_name: TensorBoard 日志名称。

    Returns:
        包含实验完整元数据的 ``RunMetadata``。
    """
    experiment_token = _build_experiment_token(
        schedule_time_s=schedule_time_s,
        step_distance=step_distance,
        reward_preset_name=reward_preset.name,
        experiment_tag=experiment_tag,
    )
    effective_training_episodes = (
        None
        if budget_mode != "completed_episodes"
        or training_episodes is None
        or num_envs is None
        else int(math.ceil(training_episodes / max(1, int(num_envs))))
        * max(1, int(num_envs))
    )
    training_budget = (
        TrainingBudget(
            mode=budget_mode,
            training_episodes=(
                int(training_episodes) if training_episodes is not None else None
            ),
            effective_training_episodes=effective_training_episodes,
            max_episode_steps=(
                int(max_episode_steps) if max_episode_steps is not None else None
            ),
            derived_total_timesteps=int(derived_total_timesteps),
            training_rollouts=(
                int(training_rollouts) if training_rollouts is not None else None
            ),
        )
        if derived_total_timesteps is not None
        else None
    )
    reward_config = RewardConfigSnapshot(
        energy_reward_scale=ENERGY_REWARD_SCALE,
        comfort_reward_scale=COMFORT_REWARD_SCALE,
        enable_potential_safety=bool(reward_preset.config.enable_potential_safety),
        survival_reward_scale=SURVIVAL_REWARD_SCALE,
        safety_potential_scale=SAFETY_POTENTIAL_SCALE,
        safety_potential_steepness=SAFETY_POTENTIAL_STEEPNESS,
        enable_potential_punctuality=reward_preset.config.enable_potential_punctuality,
        punctuality_potential_scale=PUNCTUALITY_POTENTIAL_SCALE,
        punctuality_potential_sigma_s=PUNCTUALITY_POTENTIAL_SIGMA_S,
        potential_transition_formula="gamma_phi_next_minus_phi_previous",
        terminal_next_potential="observed_next_state",
        reward_scheme=reward_preset.config.reward_scheme,
        goal_reward_scale=goal_reward_scale(reward_preset.config),
    )
    return RunMetadata(
        reward_preset_name=reward_preset.name,
        reward_preset_label=reward_preset.label,
        reward_preset_description=reward_preset.description,
        potential_shaping_components=reward_preset.enabled_shaping_components(),
        reward_config=reward_config,
        schedule_time_s=float(schedule_time_s),
        step_distance=float(step_distance),
        reward_discount=float(reward_discount),
        experiment_token=experiment_token,
        training_budget=training_budget,
        experiment_tag=experiment_tag,
        run_mode=run_mode,
        enable_tb=enable_tb,
        enable_monitor=enable_monitor,
        enable_auto_analysis=enable_auto_analysis,
        enable_best_evaluation_artifacts=enable_best_evaluation_artifacts,
        enable_safety_truncation_histogram=enable_safety_truncation_histogram,
        safety_truncation_bin_size_m=safety_truncation_bin_size_m,
        evaluation_interval_rollouts=evaluation_interval_rollouts,
        evaluation_interval_episodes=evaluation_interval_episodes,
        evaluation_deterministic=evaluation_deterministic,
        evaluation_history_path=evaluation_history_path,
        num_envs=num_envs,
        n_steps_per_env=n_steps_per_env,
        rollout_steps_per_update=rollout_steps_per_update,
        output_dir=output_dir,
        final_output_dir=final_output_dir,
        reward_diagnostics_path=reward_diagnostics_path,
        best_eval_output_dir=best_eval_output_dir,
        tensorboard_log_dir=tensorboard_log_dir,
        tb_log_name=tb_log_name,
        reward_diagnostics_schema_version=(
            REWARD_DIAGNOSTICS_SCHEMA_VERSION
            if reward_diagnostics_path is not None
            else None
        ),
        seed=seed,
    )


def save_run_metadata(
    output_dir: str | os.PathLike[str],
    metadata: RunMetadata,
) -> str:
    """将类型化实验元数据保存为版本化 JSON 文件。

    Args:
        output_dir: 输出目录路径。
        metadata: 元数据对象。

    Returns:
        写入的 JSON 文件路径。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_path = output_path / RUN_METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata.to_mapping(), file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    return str(metadata_path)


def load_run_metadata(search_dir: str | os.PathLike[str]) -> RunMetadata:
    """从指定目录（或其父目录）加载实验元数据。

    Args:
        search_dir: 搜索目录路径。

    Raises:
        FileNotFoundError: 未找到元数据文件。
        ValueError: 元数据不符合 canonical schema。
    """
    base_path = Path(search_dir)
    candidate_paths = [
        base_path / RUN_METADATA_FILENAME,
        base_path.parent / RUN_METADATA_FILENAME,
    ]

    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            with candidate_path.open("r", encoding="utf-8") as file_obj:
                return RunMetadata.from_mapping(json.load(file_obj))

    raise FileNotFoundError(
        f"Could not find {RUN_METADATA_FILENAME!r} in '{base_path}' or its parent"
    )


# =============================================================================
# 训练配置解析
# =============================================================================


def build_default_training_args() -> argparse.Namespace:
    """构建包含所有训练默认参数的 argparse.Namespace。

    Returns:
        默认训练参数命名空间。
    """
    return argparse.Namespace(
        output_root="output/optimal/rl/",
        schedule_time_s=DEFAULT_SCHEDULE_TIME_S,
        step_distance=DEFAULT_STEP_DISTANCE,
        reward_preset=DEFAULT_REWARD_PRESET_NAME,
        experiment_tag=None,
        run_mode="tune",
        enable_tb=None,
        enable_monitor=None,
        enable_auto_analysis=None,
        enable_best_evaluation_artifacts=None,
        analysis_output_root="mtto_train_reports",
        analysis_min_points_per_10k_steps=5.0,
        analysis_sampling_quality_mode="warn_only",
        reward_discount=DEFAULT_REWARD_DISCOUNT,
        num_envs=DEFAULT_NUM_ENVS,
        rollout_steps_per_update=DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        n_steps_per_env=None,
        training_episodes=None,
        budget_mode="completed_episodes",
        training_rollouts=None,
        tensorboard_log_dir="mtto_ppo_tb_logs",
        tb_log_name=None,
        log_interval=None,
        evaluation_interval_rollouts=DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
        evaluation_interval_episodes=None,
        evaluation_deterministic=True,
        evaluation_history_path=None,
        enable_safety_truncation_histogram=False,
        safety_truncation_bin_size_m=5000.0,
        seed=None,
        device=DEFAULT_DEVICE,
        dry_run=False,
    )


@cache
def _route_distance_m(schedule_time_s: float) -> float:
    """Return the fixed route distance for a training schedule."""
    _, _, _, train_service = build_scenario(schedule_time_s=float(schedule_time_s))
    return abs(train_service.target_position - train_service.start_position)


def _derive_training_budget(
    *,
    training_episodes: int,
    num_envs: int,
    step_distance: float,
    rollout_steps_per_update: int,
    schedule_time_s: float,
) -> tuple[int, int, int]:
    """Resolve the effective episode target and PPO-safe timestep ceiling."""
    if training_episodes <= 0:
        raise ValueError("training_episodes must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if step_distance <= 0.0 or not math.isfinite(step_distance):
        raise ValueError("step_distance must be finite and positive")
    if rollout_steps_per_update <= 0:
        raise ValueError("rollout_steps_per_update must be positive")

    effective_training_episodes = math.ceil(training_episodes / num_envs) * num_envs
    max_episode_steps = math.ceil(_route_distance_m(schedule_time_s) / step_distance)
    raw_total_timesteps = effective_training_episodes * max_episode_steps
    derived_total_timesteps = (
        math.ceil(raw_total_timesteps / rollout_steps_per_update)
        * rollout_steps_per_update
    )
    return (
        int(effective_training_episodes),
        int(max_episode_steps),
        int(derived_total_timesteps),
    )


def resolve_run_mode(
    args: argparse.Namespace,
) -> tuple[str, bool, bool, bool, bool]:
    """根据 CLI 参数和预设模式解析各功能开关。

    Args:
        args: CLI 解析后的命名空间。

    Returns:
        (run_mode, enable_tb, enable_monitor, enable_auto_analysis,
         enable_best_evaluation_artifacts) 五元组。
    """
    run_mode = args.run_mode
    defaults_by_mode = {
        "tune": {
            "tb": True,
            "monitor": True,
            "analysis": True,
            "best_eval": True,
        },
        "reproduce": {
            "tb": False,
            "monitor": True,
            "analysis": False,
            "best_eval": False,
        },
        "monitor_best": {
            "tb": True,
            "monitor": True,
            "analysis": False,
            "best_eval": True,
        },
        "best_only": {
            "tb": False,
            "monitor": False,
            "analysis": False,
            "best_eval": True,
        },
    }
    mode_defaults = defaults_by_mode[run_mode]
    enable_tb = mode_defaults["tb"] if args.enable_tb is None else args.enable_tb
    enable_monitor = (
        mode_defaults["monitor"] if args.enable_monitor is None else args.enable_monitor
    )
    enable_auto_analysis = (
        mode_defaults["analysis"]
        if args.enable_auto_analysis is None
        else args.enable_auto_analysis
    )
    enable_best_evaluation_artifacts = (
        mode_defaults["best_eval"]
        if args.enable_best_evaluation_artifacts is None
        else args.enable_best_evaluation_artifacts
    )

    return (
        run_mode,
        enable_tb,
        enable_monitor,
        enable_auto_analysis,
        enable_best_evaluation_artifacts,
    )


def resolve_log_interval(
    args: argparse.Namespace, run_mode: str, enable_tb: bool
) -> int:
    """根据运行模式解析 TensorBoard 日志记录间隔。

    Args:
        args: CLI 解析后的命名空间。
        run_mode: 运行模式。
        enable_tb: 是否启用 TensorBoard。

    Returns:
        日志记录步数间隔。
    """
    defaults_by_mode = {
        "tune": 1,
        "reproduce": 5,
        "monitor_best": 1,
        "best_only": 10,
    }
    if args.log_interval is not None:
        return max(1, int(args.log_interval))
    if not enable_tb:
        return 1
    return int(defaults_by_mode.get(run_mode, 1))


def _resolve_n_steps_per_env(args: argparse.Namespace, num_envs: int) -> int:
    if args.n_steps_per_env is not None:
        return max(1, int(args.n_steps_per_env))

    target_rollout_steps = max(1, int(args.rollout_steps_per_update))
    return max(1, int(np.ceil(target_rollout_steps / max(1, int(num_envs)))))


def resolve_training_run_spec(args: argparse.Namespace) -> TrainingRunSpec:
    """将 CLI 解析后的参数转换为完整的 TrainingRunSpec。

    Args:
        args: CLI 解析后的命名空间。

    Returns:
        TrainingRunSpec 实例，包含所有解析后的训练配置。
    """
    schedule_time_s = float(args.schedule_time_s)
    ds = float(args.step_distance)
    if not math.isfinite(ds) or ds <= 0.0:
        raise ValueError("step_distance must be finite and positive")
    reward_discount = float(args.reward_discount)
    reward_preset = resolve_reward_preset(args.reward_preset)
    output_root = args.output_root
    output_dir = resolve_output_dir(
        output_root=output_root,
        schedule_time_s=schedule_time_s,
        step_distance=ds,
        reward_preset_name=reward_preset.name,
        experiment_tag=args.experiment_tag,
    )
    final_output_dir = os.path.join(output_dir, "final")
    final_model_save_path = os.path.join(final_output_dir, RL_FINAL_MODEL_FILENAME)
    reward_diagnostics_path = os.path.join(
        final_output_dir, REWARD_DIAGNOSTICS_FILENAME
    )
    best_eval_output_dir = os.path.join(output_dir, "best")
    evaluation_history_path_raw = getattr(args, "evaluation_history_path", None)
    evaluation_history_path = (
        str(evaluation_history_path_raw)
        if isinstance(evaluation_history_path_raw, str)
        and evaluation_history_path_raw.strip()
        else None
    )

    (
        run_mode,
        enable_tb,
        enable_monitor,
        enable_auto_analysis,
        enable_best_evaluation_artifacts,
    ) = resolve_run_mode(args)

    effective_tb_log_name = resolve_tb_log_name(
        tb_log_name=args.tb_log_name,
        run_mode=run_mode,
        schedule_time_s=schedule_time_s,
        step_distance=ds,
        reward_preset_name=reward_preset.name,
        experiment_tag=args.experiment_tag,
    )

    log_interval = resolve_log_interval(args, run_mode, enable_tb)
    enable_safety_truncation_histogram = run_mode == "tune" or bool(
        getattr(args, "enable_safety_truncation_histogram", False)
    )
    safety_truncation_bin_size_m = float(
        getattr(args, "safety_truncation_bin_size_m", 5000.0)
    )
    if (
        not np.isfinite(safety_truncation_bin_size_m)
        or safety_truncation_bin_size_m <= 0.0
    ):
        raise ValueError("safety_truncation_bin_size_m must be finite and positive")
    num_envs = max(1, int(args.num_envs))
    n_steps_per_env = _resolve_n_steps_per_env(args, num_envs)
    rollout_steps_per_update = n_steps_per_env * num_envs
    evaluation_interval_rollouts_raw = getattr(
        args, "evaluation_interval_rollouts", None
    )
    evaluation_interval_episodes_raw = getattr(
        args, "evaluation_interval_episodes", None
    )
    if (
        evaluation_interval_rollouts_raw is not None
        and evaluation_interval_episodes_raw is not None
    ):
        raise ValueError(
            "evaluation_interval_rollouts and evaluation_interval_episodes "
            "are mutually exclusive"
        )
    if (
        evaluation_interval_rollouts_raw is None
        and evaluation_interval_episodes_raw is None
    ):
        evaluation_interval_rollouts_raw = DEFAULT_EVALUATION_INTERVAL_ROLLOUTS
    evaluation_interval_rollouts = (
        None
        if evaluation_interval_rollouts_raw is None
        else int(evaluation_interval_rollouts_raw)
    )
    evaluation_interval_episodes = (
        None
        if evaluation_interval_episodes_raw is None
        else int(evaluation_interval_episodes_raw)
    )
    if evaluation_interval_rollouts is not None and evaluation_interval_rollouts <= 0:
        raise ValueError("evaluation_interval_rollouts must be positive")
    if evaluation_interval_episodes is not None and evaluation_interval_episodes <= 0:
        raise ValueError("evaluation_interval_episodes must be positive")
    budget_mode = str(getattr(args, "budget_mode", "completed_episodes"))
    if budget_mode not in {"completed_episodes", "environment_steps"}:
        raise ValueError(f"Unsupported training budget mode: {budget_mode}")
    max_episode_steps = math.ceil(_route_distance_m(schedule_time_s) / ds)
    training_rollouts_raw = getattr(args, "training_rollouts", None)
    if budget_mode == "environment_steps":
        if getattr(args, "training_episodes", None) is not None:
            raise ValueError(
                "training_episodes cannot be used with environment_steps budgets"
            )
        if training_rollouts_raw is None or int(training_rollouts_raw) <= 0:
            raise ValueError("training_rollouts must be positive for step budgets")
        training_rollouts = int(training_rollouts_raw)
        training_episodes = None
        effective_training_episodes = None
        derived_total_timesteps = training_rollouts * rollout_steps_per_update
    else:
        if training_rollouts_raw is not None:
            raise ValueError(
                "training_rollouts cannot be used with completed_episodes budgets"
            )
        training_rollouts = None
        training_episodes_raw = getattr(args, "training_episodes", None)
        training_episodes = int(
            DEFAULT_TRAINING_EPISODES
            if training_episodes_raw is None
            else training_episodes_raw
        )
        if training_episodes <= 0:
            raise ValueError("training_episodes must be positive")
        (
            effective_training_episodes,
            max_episode_steps,
            derived_total_timesteps,
        ) = _derive_training_budget(
            training_episodes=training_episodes,
            num_envs=num_envs,
            step_distance=ds,
            rollout_steps_per_update=rollout_steps_per_update,
            schedule_time_s=schedule_time_s,
        )

    run_metadata = build_run_metadata(
        reward_preset=reward_preset,
        schedule_time_s=schedule_time_s,
        step_distance=ds,
        reward_discount=reward_discount,
        run_mode=run_mode,
        experiment_tag=args.experiment_tag,
        budget_mode=cast(
            Literal["completed_episodes", "environment_steps"], budget_mode
        ),
        training_episodes=training_episodes,
        training_rollouts=training_rollouts,
        max_episode_steps=max_episode_steps,
        derived_total_timesteps=derived_total_timesteps,
        enable_tb=bool(enable_tb),
        enable_monitor=bool(enable_monitor),
        enable_auto_analysis=bool(enable_auto_analysis),
        enable_best_evaluation_artifacts=bool(enable_best_evaluation_artifacts),
        enable_safety_truncation_histogram=bool(enable_safety_truncation_histogram),
        safety_truncation_bin_size_m=safety_truncation_bin_size_m,
        evaluation_interval_rollouts=evaluation_interval_rollouts,
        evaluation_interval_episodes=evaluation_interval_episodes,
        evaluation_deterministic=bool(args.evaluation_deterministic),
        evaluation_history_path=evaluation_history_path,
        num_envs=int(num_envs),
        n_steps_per_env=int(n_steps_per_env),
        rollout_steps_per_update=int(rollout_steps_per_update),
        output_dir=output_dir,
        final_output_dir=final_output_dir,
        reward_diagnostics_path=reward_diagnostics_path,
        best_eval_output_dir=(
            best_eval_output_dir if enable_best_evaluation_artifacts else None
        ),
        tensorboard_log_dir=args.tensorboard_log_dir if enable_tb else None,
        tb_log_name=effective_tb_log_name if enable_tb else None,
        seed=(None if args.seed is None else int(args.seed)),
    )

    return TrainingRunSpec(
        schedule_time_s=schedule_time_s,
        step_distance=ds,
        reward_discount=reward_discount,
        reward_preset=reward_preset,
        output_root=output_root,
        output_dir=output_dir,
        final_output_dir=final_output_dir,
        best_eval_output_dir=best_eval_output_dir,
        reward_diagnostics_path=reward_diagnostics_path,
        final_model_save_path=final_model_save_path,
        run_metadata_path=os.path.join(final_output_dir, RUN_METADATA_FILENAME),
        run_metadata=run_metadata,
        run_mode=run_mode,
        enable_tb=bool(enable_tb),
        enable_monitor=bool(enable_monitor),
        enable_auto_analysis=bool(enable_auto_analysis),
        enable_best_evaluation_artifacts=bool(enable_best_evaluation_artifacts),
        tb_log_name=effective_tb_log_name,
        tensorboard_log_dir=args.tensorboard_log_dir,
        log_interval=int(log_interval),
        num_envs=int(num_envs),
        n_steps_per_env=int(n_steps_per_env),
        rollout_steps_per_update=int(rollout_steps_per_update),
        evaluation_interval_rollouts=evaluation_interval_rollouts,
        evaluation_interval_episodes=evaluation_interval_episodes,
        evaluation_deterministic=bool(args.evaluation_deterministic),
        evaluation_history_path=evaluation_history_path,
        enable_safety_truncation_histogram=bool(enable_safety_truncation_histogram),
        safety_truncation_bin_size_m=safety_truncation_bin_size_m,
        budget_mode=cast(
            Literal["completed_episodes", "environment_steps"], budget_mode
        ),
        training_episodes=training_episodes,
        training_rollouts=training_rollouts,
        max_episode_steps=max_episode_steps,
        total_timesteps=derived_total_timesteps,
        device=args.device,
        seed=args.seed,
        dry_run=bool(args.dry_run),
    )


# =============================================================================
# 训练执行
# =============================================================================


def learning_rate_schedule_parameters(
    budget_mode: Literal["completed_episodes", "environment_steps"] = (
        "completed_episodes"
    ),
) -> dict[str, float | str]:
    return {
        "id": (
            STEP_LEARNING_RATE_SCHEDULE_ID
            if budget_mode == "environment_steps"
            else LEARNING_RATE_SCHEDULE_ID
        ),
        "progress_unit": (
            "global_environment_transitions"
            if budget_mode == "environment_steps"
            else "global_completed_training_episodes"
        ),
        "initial_value": INITIAL_LEARNING_RATE,
        "final_value": FINAL_LEARNING_RATE,
    }


def _completed_episode_cosine_annealing_schedule(
    progress: CompletedEpisodeProgress,
    initial_value: float = INITIAL_LEARNING_RATE,
    final_value: float = FINAL_LEARNING_RATE,
) -> Callable[[float], float]:
    def func(_sb3_progress_remaining: float) -> float:
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress.fraction))
        lr = final_value + (initial_value - final_value) * cosine_decay

        return lr

    return func


def _environment_step_cosine_annealing_schedule(
    initial_value: float = INITIAL_LEARNING_RATE,
    final_value: float = FINAL_LEARNING_RATE,
) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        progress = min(1.0, max(0.0, 1.0 - float(progress_remaining)))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_value + (initial_value - final_value) * cosine_decay

    return func


def _build_env_initializer(
    *,
    vehicle: VehicleInfo,
    track: TrackInfo,
    safeguard_utility: SafeGuardUtility,
    train_service: TrainService,
    gamma: float,
    step_distance: float,
    worker_rank: int,
    rollout_capacity: int,
    reward_config: RewardConfig | None = None,
    stepper: OperationalStepper | None = None,
    enable_safety_truncation_tracking: bool = False,
) -> Callable[[], Any]:
    def _init():
        return make_env(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            gamma=gamma,
            step_distance=step_distance,
            compact_training_info=True,
            reward_config=reward_config,
            stepper=stepper,
            enable_safety_truncation_tracking=enable_safety_truncation_tracking,
            reward_diagnostics_worker_rank=worker_rank,
            reward_diagnostics_rollout_capacity=rollout_capacity,
        )

    return _init


def train_single_experiment(
    args: argparse.Namespace,
    *,
    spec: TrainingRunSpec | None = None,
) -> TrainingRunSpec:
    """执行单次 PPO 训练实验。

    构建向量化环境、PPO 模型和回调链，完成训练后保存最优/最终轨迹产物，
    并在 enable_auto_analysis 时自动运行训练分析。

    Args:
        args: CLI 解析后的参数命名空间（含分析输出配置）。
        spec: 预构建的 TrainingRunSpec。为 None 时从 args 构建。

    Returns:
        本次训练使用的 TrainingRunSpec。
    """
    resolved_spec = spec if spec is not None else resolve_training_run_spec(args)

    final_dir = Path(resolved_spec.final_output_dir)
    target_artifacts = [
        Path(resolved_spec.run_metadata_path),
        Path(resolved_spec.final_model_save_path),
        Path(resolved_spec.reward_diagnostics_path),
        final_dir / RL_METRICS_FILENAME,
        final_dir / RL_TRAJECTORY_FILENAME,
    ]
    if any(p.exists() for p in target_artifacts):
        raise FileExistsError(
            "Target directory already contains training artifacts: "
            f"{resolved_spec.final_output_dir}"
        )

    if resolved_spec.seed is not None:
        set_random_seed(
            seed=resolved_spec.seed,
            using_cuda=True if resolved_spec.device == "cuda" else False,
        )

    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=resolved_spec.schedule_time_s
    )
    shared_stepper = OperationalStepper(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        step_distance_m=resolved_spec.step_distance,
    )

    resolved_spec = replace(
        resolved_spec,
        run_metadata=resolved_spec.run_metadata.with_updates(
            extensions={
                **resolved_spec.run_metadata.extensions,
                "learning_rate_schedule": learning_rate_schedule_parameters(
                    resolved_spec.budget_mode
                ),
            }
        ),
    )
    os.makedirs(resolved_spec.output_dir, exist_ok=True)
    os.makedirs(resolved_spec.final_output_dir, exist_ok=True)
    _ = save_run_metadata(resolved_spec.final_output_dir, resolved_spec.run_metadata)

    env_initializers: list[Callable[[], Any]] = [
        _build_env_initializer(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            gamma=resolved_spec.reward_discount,
            step_distance=resolved_spec.step_distance,
            worker_rank=env_rank,
            rollout_capacity=resolved_spec.n_steps_per_env,
            reward_config=resolved_spec.reward_config,
            stepper=shared_stepper,
            enable_safety_truncation_tracking=(
                resolved_spec.enable_safety_truncation_histogram
            ),
        )
        for env_rank in range(resolved_spec.num_envs)
    ]
    venv_train = DummyVecEnv(env_initializers)
    if resolved_spec.reward_config.enable_potential_punctuality:
        venv_train = TaskTerminalVecEnv(venv_train)

    if resolved_spec.enable_monitor:
        venv_train = VecMonitor(venv_train)

    if resolved_spec.run_metadata.training_budget is None:
        raise RuntimeError("training metadata is missing its training budget")
    effective_training_episodes = (
        resolved_spec.run_metadata.training_budget.effective_training_episodes
    )
    episode_progress: CompletedEpisodeProgress | None = None
    episode_stop_callback: StopTrainingOnCompletedEpisodes | None = None
    if resolved_spec.budget_mode == "completed_episodes":
        if effective_training_episodes is None:
            raise RuntimeError(
                "training metadata is missing effective training episodes"
            )
        episode_progress = CompletedEpisodeProgress(int(effective_training_episodes))
        episode_stop_callback = StopTrainingOnCompletedEpisodes(episode_progress)
        learning_rate = _completed_episode_cosine_annealing_schedule(episode_progress)
    else:
        learning_rate = _environment_step_cosine_annealing_schedule()

    model = PPO(
        "MlpPolicy",
        venv_train,
        device=resolved_spec.device,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=resolved_spec.n_steps_per_env,
        batch_size=DEFAULT_BATCH_SIZE,
        n_epochs=DEFAULT_N_EPOCHS,
        gamma=resolved_spec.reward_discount,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=(
            resolved_spec.tensorboard_log_dir if resolved_spec.enable_tb else None
        ),
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    )

    callbacks: list[BaseCallback] = []
    if episode_stop_callback is not None:
        callbacks.append(episode_stop_callback)
    reward_diagnostics_callback = RewardDiagnosticsArtifactCallback(
        output_path=resolved_spec.reward_diagnostics_path
    )
    callbacks.append(reward_diagnostics_callback)
    if resolved_spec.enable_safety_truncation_histogram:
        callbacks.append(
            SafetyTruncationPositionHistogramCallback(
                output_path=os.path.join(
                    resolved_spec.final_output_dir,
                    "safety_truncation_position_histogram.npz",
                ),
                position_bin_size_m=resolved_spec.safety_truncation_bin_size_m,
            )
        )

    if (
        resolved_spec.enable_best_evaluation_artifacts
        or resolved_spec.evaluation_history_path
    ):
        evaluation_handlers: list[Any] = []
        if resolved_spec.evaluation_history_path:
            evaluation_handlers.append(
                EvaluationHistoryArtifactHandler(
                    output_path=resolved_spec.evaluation_history_path,
                )
            )
        if resolved_spec.enable_best_evaluation_artifacts:
            evaluation_handlers.append(
                BestEvaluationArtifactHandler(
                    output_dir=resolved_spec.best_eval_output_dir,
                    artifact_metadata=resolved_spec.run_metadata.to_mapping(),
                )
            )
        callbacks.append(
            ScheduledPolicyEvaluationCallback(
                eval_env=build_single_eval_env(
                    vehicle=vehicle,
                    track=track,
                    safeguard_utility=safeguard_utility,
                    train_service=train_service,
                    gamma=resolved_spec.reward_discount,
                    step_distance=resolved_spec.step_distance,
                    enable_trajectory_tracking=(
                        resolved_spec.enable_best_evaluation_artifacts
                    ),
                    reward_config=resolved_spec.reward_config,
                ),
                handlers=evaluation_handlers,
                evaluation_interval_rollouts=(
                    resolved_spec.evaluation_interval_rollouts
                ),
                evaluation_interval_episodes=(
                    resolved_spec.evaluation_interval_episodes
                ),
                deterministic=resolved_spec.evaluation_deterministic,
                evaluate_at_boundaries=bool(
                    getattr(args, "evaluate_at_boundaries", False)
                ),
                get_completed_training_episodes=(
                    (lambda: episode_progress.completed_episodes)
                    if episode_progress is not None
                    else (lambda: reward_diagnostics_callback.completed_episode_count)
                ),
                max_rollouts_exclusive=(
                    resolved_spec.training_rollouts
                    if resolved_spec.budget_mode == "environment_steps"
                    else None
                ),
                max_completed_episodes_exclusive=(
                    int(effective_training_episodes)
                    if resolved_spec.budget_mode == "completed_episodes"
                    and effective_training_episodes is not None
                    else None
                ),
            )
        )

    if effective_training_episodes is not None:
        callbacks.append(
            EpisodeProgressBarCallback(total_episodes=int(effective_training_episodes))
        )

    callback = CallbackList(callbacks) if callbacks else None

    _ = model.learn(
        total_timesteps=resolved_spec.total_timesteps,
        callback=callback,
        log_interval=resolved_spec.log_interval,
        tb_log_name=resolved_spec.tb_log_name,
        progress_bar=(resolved_spec.budget_mode == "environment_steps"),
    )
    actual_completed_episodes = reward_diagnostics_callback.completed_episode_count
    actual_training_timesteps = int(model.num_timesteps)
    actual_training_rollouts = (
        actual_training_timesteps // resolved_spec.rollout_steps_per_update
        if resolved_spec.budget_mode == "environment_steps"
        else None
    )
    if resolved_spec.budget_mode == "completed_episodes":
        assert episode_progress is not None
        assert effective_training_episodes is not None
        target_reached = (
            episode_progress.completed_episodes >= effective_training_episodes
        )
        stop_reason = (
            "completed_episode_target" if target_reached else "derived_timestep_ceiling"
        )
    else:
        target_reached = actual_training_timesteps >= resolved_spec.total_timesteps
        stop_reason = (
            "environment_step_target"
            if target_reached
            else "training_stopped_before_step_target"
        )
    if resolved_spec.run_metadata.training_budget is None:
        raise RuntimeError("training metadata is missing its training budget")
    training_budget = replace(
        resolved_spec.run_metadata.training_budget,
        actual_completed_episodes=actual_completed_episodes,
        actual_training_timesteps=actual_training_timesteps,
        actual_training_rollouts=actual_training_rollouts,
        target_reached=target_reached,
        stop_reason=stop_reason,
    )
    resolved_metadata = resolved_spec.run_metadata.with_updates(
        training_budget=training_budget
    )
    resolved_spec = replace(
        resolved_spec,
        run_metadata=resolved_metadata,
    )
    _ = save_run_metadata(resolved_spec.final_output_dir, resolved_spec.run_metadata)
    if resolved_spec.enable_best_evaluation_artifacts:
        _ = save_run_metadata(
            resolved_spec.best_eval_output_dir, resolved_spec.run_metadata
        )
    model.save(resolved_spec.final_model_save_path)
    venv_train.close()

    print("Training finished.")
    print(f"Final Model saved to: {resolved_spec.final_model_save_path}")
    if resolved_spec.enable_best_evaluation_artifacts:
        print(
            f"Best trajectory artifacts saved under: \
            {resolved_spec.best_eval_output_dir}"
        )
    print("Run python -m scripts.evaluate_rl to evaluate the trained policy.")

    if resolved_spec.enable_auto_analysis:
        try:
            analyze_config = AnalysisConfig(
                output_root=args.analysis_output_root,
                training_log_interval=(
                    resolved_spec.log_interval if resolved_spec.enable_tb else None
                ),
                min_points_per_10k_steps=args.analysis_min_points_per_10k_steps,
                rollout_steps_per_update=resolved_spec.rollout_steps_per_update,
                sampling_quality_mode=args.analysis_sampling_quality_mode,
                final_output_dir=resolved_spec.final_output_dir,
            )
            analysis_result = run_training_analysis(
                log_root=(
                    resolved_spec.tensorboard_log_dir
                    if resolved_spec.enable_tb
                    else None
                ),
                run_name=resolved_spec.tb_log_name if resolved_spec.enable_tb else None,
                config=analyze_config,
            )
            output_paths = analysis_result.get("output_paths", {})
            print("Training analysis completed.")
            print(f"Analysis JSON: {output_paths.get('json_snapshot', 'N/A')}")
            print(f"Analysis report: {output_paths.get('markdown_report', 'N/A')}")
        except Exception as exc:
            print(f"Training analysis skipped due to error: {exc}")

    return resolved_spec


def evaluate_final_training_run(
    spec: TrainingRunSpec,
) -> tuple[str, str]:
    """Evaluate ``final_model`` from a completed run at the real start state."""
    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=spec.schedule_time_s
    )
    env = build_single_eval_env(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        gamma=spec.reward_discount,
        step_distance=spec.step_distance,
        enable_trajectory_tracking=True,
        reward_config=spec.reward_config,
    )
    try:
        model = PPO.load(spec.final_model_save_path, device=spec.device)
        _, npz_path, metrics_path = evaluate_and_save_final_policy(
            model,
            env,
            output_path=os.path.join(spec.final_output_dir, RL_TRAJECTORY_FILENAME),
            metadata=spec.run_metadata.to_mapping(),
            deterministic=True,
            metrics_path=os.path.join(spec.final_output_dir, RL_METRICS_FILENAME),
        )
        return npz_path, metrics_path
    finally:
        env.close()


# =============================================================================
# 轨迹产物解析
# =============================================================================


def _resolve_rl_metrics_path(curve_path: Path) -> Path:
    metrics_path = curve_path.with_name(RL_METRICS_FILENAME)
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Could not find '{metrics_path.name}' in directory: {curve_path.parent}"
        )
    return metrics_path


def resolve_rl_curve_artifact(
    *,
    curve_dir: str,
) -> OptimizedCurveArtifact:
    """Load one canonical trajectory from the explicitly supplied model directory."""
    model_dir = Path(curve_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    curve_path = model_dir / RL_TRAJECTORY_FILENAME
    if not curve_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {curve_path}")
    metrics_path = _resolve_rl_metrics_path(curve_path)

    return OptimizedCurveArtifact(
        npz_path=str(curve_path),
        metrics_path=str(metrics_path),
    )


def load_rl_curve_artifact(
    artifact: OptimizedCurveArtifact,
) -> tuple[np.ndarray, np.ndarray, EvaluationMetrics]:
    """加载 canonical RL 轨迹产物和类型化指标。

    Args:
        artifact: 由 resolve_rl_curve_artifact 返回的产物定位信息。

    Returns:
        (位置数组, 速度数组, ``EvaluationMetrics``) 三元组。
    """
    loaded = load_evaluation_artifact(
        npz_path=artifact.npz_path,
        metrics_path=artifact.metrics_path,
        dtype=np.float32,
        use_metrics_cache=True,
    )
    return (
        loaded.trajectory.position_m,
        loaded.trajectory.speed_mps,
        loaded.metrics,
    )


def load_rl_curve_metrics(artifact: OptimizedCurveArtifact) -> EvaluationMetrics:
    """仅加载 canonical 轨迹产物的类型化指标。

    Args:
        artifact: 由 resolve_rl_curve_artifact 返回的产物定位信息。

    Returns:
        ``EvaluationMetrics``。

    Raises:
        FileNotFoundError: 指标文件不存在。
    """
    return load_evaluation_metrics(
        artifact.metrics_path,
        use_metrics_cache=True,
    )


# =============================================================================
# 轨迹指标 & 对比
# =============================================================================


def build_rl_trajectory_comparison_key(
    metrics: EvaluationMetrics | Mapping[str, object],
) -> tuple[float, ...]:
    """构建严格可行优先的多轨迹排序键。

    Args:
        metrics: 轨迹指标字典。

    Returns:
        可直接用于 max() 的比较键。
    """
    if isinstance(metrics, EvaluationMetrics):
        return metrics.selection_comparison_key

    raw_key = metrics.get("selection_comparison_key")
    if isinstance(raw_key, (list, tuple)) and len(raw_key) > 0:
        converted: list[float] = []
        for value in raw_key:
            numeric_value = as_float(value)
            if numeric_value is None:
                raise ValueError("selection_comparison_key must contain only numbers")
            converted.append(numeric_value)
        return tuple(converted)

    if raw_key is not None:
        raise ValueError("selection_comparison_key must be a non-empty numeric list")

    raise ValueError(
        "selection_comparison_key is required by evaluation metrics schema v2"
    )


# =============================================================================
# 可视化辅助
# =============================================================================


def _metrics_display_mapping(
    metrics: EvaluationMetrics | Mapping[str, object],
) -> Mapping[str, object]:
    if isinstance(metrics, EvaluationMetrics):
        return metrics.to_display_mapping()
    return metrics


def get_rl_trajectory_status_text(
    metrics: EvaluationMetrics | Mapping[str, object],
) -> str | None:
    """根据轨迹指标返回中文状态描述文本。

    Args:
        metrics: 轨迹指标字典（需含 success 和 trajectory_source 键）。

    Returns:
        如 "RL 最优轨迹（完成任务）" 或 None（success 不是 bool 时）。
    """
    display_metrics = _metrics_display_mapping(metrics)
    success_value = display_metrics.get("success")
    trajectory_source = display_metrics.get("trajectory_source")
    if not isinstance(success_value, bool):
        return None
    prefix = "RL 最终轨迹" if trajectory_source == "final" else "RL 最优轨迹"
    return f"{prefix}（完成任务）" if success_value else f"{prefix}（未完成任务）"


def format_rl_trajectory_terminal_summary(
    metrics: EvaluationMetrics | Mapping[str, object],
    *,
    panel_label: str | None = None,
    reward_preset_name: str | None = None,
    repeat_index: int | None = None,
    seed: int | None = None,
    artifact_path: str | None = None,
) -> str:
    """格式化 RL 轨迹的终端摘要字符串，用于打印或日志。

    Args:
        metrics: 轨迹指标字典。
        panel_label: 面板标签。
        reward_preset_name: 奖励预设名。
        repeat_index: 重复实验索引。
        seed: 随机种子。
        artifact_path: 产物文件路径。

    Returns:
        " | " 分隔的摘要字符串。
    """
    display_metrics = _metrics_display_mapping(metrics)
    effective_preset = reward_preset_name or str(
        display_metrics.get("reward_preset_name", "unknown")
    )
    fields = [f"preset={effective_preset}"]
    if panel_label:
        fields.insert(0, f"panel={panel_label}")
    if repeat_index is not None:
        fields.append(f"repeat={repeat_index + 1}")
    if seed is not None:
        fields.append(f"seed={seed}")

    for key in (
        "trajectory_source",
        "success",
        "total_reward",
        "total_energy_j",
        "time_error_s",
        "stop_error_m",
    ):
        if key in display_metrics:
            fields.append(f"{key}={display_metrics[key]}")

    if artifact_path:
        fields.append(f"artifact={artifact_path}")
    return " | ".join(fields)


def _get_rl_trajectory_display_name(
    metrics: EvaluationMetrics | Mapping[str, object],
) -> str:
    trajectory_source = _metrics_display_mapping(metrics).get("trajectory_source")
    if trajectory_source == "final":
        return "RL final trajectory"
    return "RL best trajectory"


def render_rl_curve_on_axes(
    *,
    ax: Any,
    pos_arr: np.ndarray,
    speed_arr: np.ndarray,
    metrics: EvaluationMetrics | Mapping[str, object],
    no_safeguard: bool,
    factor: float,
    curve_color: str = "blue",
    curve_label: str | None = None,
    safeguard: SafeGuardUtility | None = None,
    render_endpoints: bool = True,
) -> None:
    """在给定的 matplotlib Axes 上渲染 RL 速度曲线及安全防护边界。

    Args:
        ax: matplotlib Axes 对象。
        pos_arr: 位置数组 (m)。
        speed_arr: 速度数组 (m/s)。
        metrics: 轨迹指标字典。
        no_safeguard: True 时跳过安全防护边界渲染。
        factor: 安全系数。
        curve_color: 曲线颜色。
        curve_label: 图例标签，None 时自动生成。
        safeguard: 预构建的 SafeGuardUtility，None 时按 factor 构建。
        render_endpoints: 是否绘制起点与终点散点标记。
    """
    render_trajectory_on_axes(
        ax=ax,
        pos_arr=pos_arr,
        speed_arr=speed_arr,
        metrics=metrics,
        no_safeguard=no_safeguard,
        factor=factor,
        curve_color=curve_color,
        curve_label=curve_label or _get_rl_trajectory_display_name(metrics),
        safeguard=safeguard,
        render_endpoints=render_endpoints,
    )
