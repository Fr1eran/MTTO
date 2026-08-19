from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.callbacks import (
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    BestEvaluationArtifactHandler,
    EvaluationHistoryArtifactHandler,
    RewardDiagnosticsArtifactCallback,
    SafetyTruncationPositionHistogramCallback,
    ScheduledPolicyEvaluationCallback,
)
from rl.completion_critic import (
    CompletionDSPDLCallback,
    CompletionDSPDLConfig,
    completion_critic_metadata,
)
from rl.context_pool import ContextPool, ContextPoolBuilder
from rl.context_sampler import CurriculumDistributionState
from rl.dp_trajectory_reader import DPTrajectoryReader
from rl.dspdl import DSPDLCallback, DSPDLConfig, DSPDLStatisticsHub
from rl.env_factory import make_env
from rl.evaluation import build_single_eval_env, evaluate_and_save_final_policy
from rl.observation_builder import ObservationBuilder
from rl.operational_stepper import OperationalStepper
from rl.reward_calculator import (
    DEFAULT_COMFORT_REWARD_SCALE,
    DEFAULT_ENERGY_REWARD_SCALE,
    DEFAULT_SURVIVAL_REWARD_SCALE,
    RewardConfig,
)
from rl.reward_diagnostics import REWARD_DIAGNOSTICS_SCHEMA_VERSION
from rl.training_analysis import AnalysisConfig, run_training_analysis
from utils.io_utils import format_float_token, load_optimized_curve_and_metrics
from utils.plot_utils import set_global_plot_style
from utils.scenario import build_safeguard_utility, build_scenario
from utils.trajectory import OptimizedCurveArtifact

__all__ = [
    # 常量
    "RL_FINAL_MODEL_FILENAME",
    "RUN_METADATA_FILENAME",
    "RL_DEFAULT_SEARCH_DIR",
    "RL_TRAJECTORY_SOURCE_CHOICES",
    "DEFAULT_SCHEDULE_TIME_S",
    "DEFAULT_REWARD_DISCOUNT",
    "DEFAULT_EVALUATION_INTERVAL_ROLLOUTS",
    "DEFAULT_ROLLOUT_STEPS_PER_UPDATE",
    "DEFAULT_STEP_DISTANCE",
    "DEFAULT_NUM_ENVS",
    "DEFAULT_DEVICE",
    "DEFAULT_REWARD_PRESET_NAME",
    "DEFAULT_CURRICULUM_PROFILE_NAME",
    "DEFAULT_ENERGY_REWARD_SCALE",
    "DEFAULT_COMFORT_REWARD_SCALE",
    "DEFAULT_SURVIVAL_REWARD_SCALE",
    # dataclass
    "DSPDLConfig",
    "CompletionDSPDLConfig",
    "RewardPreset",
    "TrainingRunSpec",
    # reward preset
    "reward_preset_names",
    "resolve_reward_preset",
    "build_reward_config",
    "resolve_survival_reward_scale",
    # curriculum profile
    "curriculum_profile_names",
    "resolve_curriculum_profile_name",
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
    "apply_rl_curve_plot_style",
    "get_rl_trajectory_status_text",
    "add_panel_label",
    "format_rl_trajectory_terminal_summary",
    "render_rl_curve_on_axes",
]

# =============================================================================
# 文件路径常量
# =============================================================================

RL_FINAL_MODEL_FILENAME = "final_model.zip"
RUN_METADATA_FILENAME = "run_metadata.json"
RL_DEFAULT_SEARCH_DIR = "output/optimal/rl"

RL_TRAJECTORY_SOURCE_CHOICES: tuple[str, ...] = (
    "best",
    "best_rollouts",
    "final",
)
_RL_BEST_TRAJECTORY_FILENAME = "best_trajectory.npz"
_RL_FINAL_TRAJECTORY_FILENAME = "final_trajectory.npz"
_RL_TRAJECTORY_METRICS_SUFFIX = "_metrics.json"
# =============================================================================
# 训练超参数常量
# =============================================================================

DEFAULT_SCHEDULE_TIME_S = 465.0
DEFAULT_REWARD_DISCOUNT = 0.998
DEFAULT_ROLLOUT_STEPS_PER_UPDATE = 8192
DEFAULT_STEP_DISTANCE = 30.0
DEFAULT_NUM_ENVS = 8
DEFAULT_DEVICE = "cpu"
DEFAULT_REWARD_PRESET_NAME = "basic_safety"
DEFAULT_CURRICULUM_PROFILE_NAME = "none"

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
        return tuple(components)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reward_preset_name": self.name,
            "reward_preset_label": self.label,
            "reward_preset_description": self.description,
            "potential_shaping_components": list(self.enabled_shaping_components()),
            "reward_config": _reward_config_to_dict(self.config),
        }


CurriculumProfileName = Literal["none", "dspdl", "dspdl_completion"]


@dataclass(frozen=True)
class TrainingRunSpec:
    """单次训练运行的完整配置快照，由 CLI 参数解析得到。"""

    schedule_time_s: float
    step_distance: float
    reward_discount: float
    reward_preset: RewardPreset
    curriculum_profile: CurriculumProfileName
    dspdl_config: DSPDLConfig | CompletionDSPDLConfig | None
    reference_curve_dir: str | None
    reference_curve_artifact_path: str | None
    reference_curve_metrics_path: str | None
    output_root: str
    output_dir: str
    final_output_dir: str
    best_eval_output_dir: str
    reward_diagnostics_path: str
    final_model_save_path: str
    run_metadata_path: str
    run_metadata: dict[str, Any]
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
    evaluation_interval_rollouts: int
    evaluation_deterministic: bool
    evaluation_history_path: str | None
    enable_safety_truncation_histogram: bool
    safety_truncation_bin_size_m: float
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


def resolve_survival_reward_scale(raw_scale: float | None) -> float:
    """解析生存奖励尺度；无效值由 RewardConfig 回退为默认值。"""
    value = DEFAULT_SURVIVAL_REWARD_SCALE if raw_scale is None else raw_scale
    return RewardConfig(survival_reward_scale=value).survival_reward_scale


def _reward_config_to_dict(reward_config: RewardConfig) -> dict[str, Any]:
    return {
        "energy_reward_scale": float(reward_config.energy_reward_scale),
        "comfort_reward_scale": float(reward_config.comfort_reward_scale),
        "enable_potential_safety": bool(reward_config.enable_potential_safety),
        "survival_reward_scale": float(reward_config.survival_reward_scale),
    }


REWARD_PRESETS: dict[str, RewardPreset] = {
    "basic": RewardPreset(
        name="basic",
        label="basic",
        description="Base reward only: energy and comfort are always enabled.",
        config=RewardConfig(
            enable_potential_safety=False,
            survival_reward_scale=DEFAULT_SURVIVAL_REWARD_SCALE,
        ),
    ),
    "basic_safety": RewardPreset(
        name="basic_safety",
        label="basic+safety",
        description="Base reward plus safety PBRS shaping.",
        config=RewardConfig(
            enable_potential_safety=True,
            survival_reward_scale=DEFAULT_SURVIVAL_REWARD_SCALE,
        ),
    ),
}

REWARD_PRESET_ALIASES: dict[str, str] = {
    "default": DEFAULT_REWARD_PRESET_NAME,
    "basic": "basic",
    "basic+safety": "basic_safety",
}


def reward_preset_names() -> tuple[str, ...]:
    """返回所有已注册奖励情形的名称元组。"""
    return tuple(REWARD_PRESETS.keys())


def curriculum_profile_names() -> tuple[str, ...]:
    return ("none", "dspdl", "dspdl_completion")


def resolve_curriculum_profile_name(
    profile_name: str | None = None,
) -> CurriculumProfileName:
    normalized = (
        DEFAULT_CURRICULUM_PROFILE_NAME
        if profile_name is None
        else str(profile_name).strip().lower().replace("-", "_").replace(" ", "_")
    )
    if not normalized:
        normalized = DEFAULT_CURRICULUM_PROFILE_NAME
    if normalized not in curriculum_profile_names():
        available = ", ".join(curriculum_profile_names())
        raise ValueError(
            "Unknown curriculum profile "
            + f"'{profile_name}'. Available profiles: {available}"
        )
    return cast(CurriculumProfileName, normalized)


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
        preset_name: 预设名，支持 ``basic``、``basic_safety`` 和 ``default``。

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


def build_reward_config(
    preset_name: str | None = None,
    *,
    survival_reward_scale: float | None = None,
) -> RewardConfig:
    """根据奖励情形名构建 RewardConfig 实例。

    Args:
        preset_name: 预设名，同 resolve_reward_preset。
        survival_reward_scale: 可选覆盖的生存奖励尺度；
            若为负数或非有限数则覆写为默认值。

    Returns:
        用于初始化 MTTOEnv 的 RewardConfig。
    """
    base_config = resolve_reward_preset(preset_name).config
    if survival_reward_scale is None:
        return base_config
    effective_scale = resolve_survival_reward_scale(survival_reward_scale)
    return RewardConfig(
        energy_reward_scale=base_config.energy_reward_scale,
        comfort_reward_scale=base_config.comfort_reward_scale,
        enable_potential_safety=base_config.enable_potential_safety,
        survival_reward_scale=effective_scale,
    )


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
    curriculum_profile_name: str | None = None,
    experiment_tag: str | None = None,
) -> str:
    schedule_token = format_float_token(schedule_time_s)
    step_token = format_float_token(step_distance)
    preset = resolve_reward_preset(reward_preset_name)
    curriculum_profile = resolve_curriculum_profile_name(curriculum_profile_name)

    tokens = [f"{schedule_token}_{step_token}", preset.name]
    if curriculum_profile != "none":
        tokens.append(curriculum_profile)
    if experiment_tag:
        tokens.append(_sanitize_identifier_token(experiment_tag))
    return "__".join(tokens)


def resolve_output_dir(
    *,
    output_root: str,
    schedule_time_s: float,
    step_distance: float,
    reward_preset_name: str | None = None,
    curriculum_profile_name: str | None = None,
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
        curriculum_profile_name=curriculum_profile_name,
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
    curriculum_profile_name: str | None = None,
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
        curriculum_profile_name=curriculum_profile_name,
        experiment_tag=experiment_tag,
    )
    return f"train_log__{_sanitize_identifier_token(run_mode)}__{experiment_token}"


# =============================================================================
# 运行元数据管理
# =============================================================================


def build_run_metadata(
    *,
    reward_preset: RewardPreset,
    curriculum_profile: CurriculumProfileName = DEFAULT_CURRICULUM_PROFILE_NAME,
    dspdl_config: DSPDLConfig | CompletionDSPDLConfig | None = None,
    reference_curve_dir: str | None = None,
    reference_curve_artifact_path: str | None = None,
    reference_curve_metrics_path: str | None = None,
    schedule_time_s: float,
    step_distance: float,
    reward_discount: float,
    run_mode: str | None = None,
    experiment_tag: str | None = None,
    total_timesteps: int | None = None,
    enable_tb: bool | None = None,
    enable_monitor: bool | None = None,
    enable_auto_analysis: bool | None = None,
    enable_best_evaluation_artifacts: bool | None = None,
    enable_safety_truncation_histogram: bool | None = None,
    safety_truncation_bin_size_m: float | None = None,
    evaluation_interval_rollouts: int | None = None,
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
) -> dict[str, Any]:
    """构建单次训练运行的元数据字典，用于持久化记录实验参数。

    Args:
        reward_preset: 具名奖励预设。
        schedule_time_s: 规划运行时间 (s)。
        step_distance: 固定空间控制步长 (m)。
        reward_discount: 奖励折扣因子 γ。
        run_mode: 运行模式。
        experiment_tag: 实验标签。
        total_timesteps: PPO 总训练步数。
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
        包含实验完整元数据的字典。
    """
    metadata = reward_preset.to_metadata()
    resolved_curriculum = resolve_curriculum_profile_name(curriculum_profile)
    curriculum_enabled = resolved_curriculum != "none"
    resolved_dspdl_config = (
        (
            dspdl_config
            if dspdl_config is not None
            else (
                CompletionDSPDLConfig()
                if resolved_curriculum == "dspdl_completion"
                else DSPDLConfig()
            )
        )
        if curriculum_enabled
        else None
    )
    if resolved_curriculum == "dspdl_completion" and not isinstance(
        resolved_dspdl_config, CompletionDSPDLConfig
    ):
        raise TypeError("completion DSPDL metadata requires CompletionDSPDLConfig")
    if resolved_curriculum == "dspdl" and not isinstance(
        resolved_dspdl_config, DSPDLConfig
    ):
        raise TypeError("legacy DSPDL metadata requires DSPDLConfig")
    metadata["curriculum"] = {
        "profile_name": resolved_curriculum,
        "enabled": curriculum_enabled,
        "value_source": (
            "task_completion"
            if resolved_curriculum == "dspdl_completion"
            else ("ppo_critic_return" if curriculum_enabled else None)
        ),
        "dspdl_config": (
            asdict(resolved_dspdl_config) if resolved_dspdl_config is not None else None
        ),
        "reference_curve_dir": reference_curve_dir,
        "reference_curve_artifact_path": reference_curve_artifact_path,
        "reference_curve_metrics_path": reference_curve_metrics_path,
        "rl_step_distance_m": (float(step_distance) if curriculum_enabled else None),
        "context_count": None,
        "initial_curriculum_version": (0 if curriculum_enabled else None),
        "completion_critic": None,
    }

    metadata.update(
        {
            "schedule_time_s": float(schedule_time_s),
            "step_distance": float(step_distance),
            "reward_discount": float(reward_discount),
        }
    )

    metadata["experiment_token"] = _build_experiment_token(
        schedule_time_s=schedule_time_s,
        step_distance=step_distance,
        reward_preset_name=reward_preset.name,
        curriculum_profile_name=resolved_curriculum,
        experiment_tag=experiment_tag,
    )
    if experiment_tag is not None:
        metadata["experiment_tag"] = str(experiment_tag)
    if run_mode is not None:
        metadata["run_mode"] = str(run_mode)
    if total_timesteps is not None:
        metadata["total_timesteps"] = int(total_timesteps)

    if enable_tb is not None:
        metadata["enable_tb"] = bool(enable_tb)
    if enable_monitor is not None:
        metadata["enable_monitor"] = bool(enable_monitor)
    if enable_auto_analysis is not None:
        metadata["enable_auto_analysis"] = bool(enable_auto_analysis)
    if enable_best_evaluation_artifacts is not None:
        metadata["enable_best_evaluation_artifacts"] = bool(
            enable_best_evaluation_artifacts
        )
    if enable_safety_truncation_histogram is not None:
        metadata["enable_safety_truncation_histogram"] = bool(
            enable_safety_truncation_histogram
        )
    if safety_truncation_bin_size_m is not None:
        metadata["safety_truncation_bin_size_m"] = float(safety_truncation_bin_size_m)
    if evaluation_interval_rollouts is not None:
        metadata["evaluation_interval_rollouts"] = int(evaluation_interval_rollouts)
    if evaluation_deterministic is not None:
        metadata["evaluation_deterministic"] = bool(evaluation_deterministic)
    if evaluation_history_path is not None:
        metadata["evaluation_history_path"] = str(evaluation_history_path)

    if num_envs is not None:
        metadata["num_envs"] = int(num_envs)
    if n_steps_per_env is not None:
        metadata["n_steps_per_env"] = int(n_steps_per_env)
    if rollout_steps_per_update is not None:
        metadata["rollout_steps_per_update"] = int(rollout_steps_per_update)

    if output_dir is not None:
        metadata["output_dir"] = str(output_dir)
    if final_output_dir is not None:
        metadata["final_output_dir"] = str(final_output_dir)
    if reward_diagnostics_path is not None:
        metadata["reward_diagnostics_path"] = str(reward_diagnostics_path)
        metadata["reward_diagnostics_schema_version"] = (
            REWARD_DIAGNOSTICS_SCHEMA_VERSION
        )
    if best_eval_output_dir is not None:
        metadata["best_eval_output_dir"] = str(best_eval_output_dir)
    if tensorboard_log_dir is not None:
        metadata["tensorboard_log_dir"] = str(tensorboard_log_dir)
    if tb_log_name is not None:
        metadata["tb_log_name"] = str(tb_log_name)

    return metadata


def save_run_metadata(
    output_dir: str | os.PathLike[str],
    metadata: dict[str, Any],
) -> str:
    """将实验元数据保存为 JSON 文件。

    Args:
        output_dir: 输出目录路径。
        metadata: 元数据字典。

    Returns:
        写入的 JSON 文件路径。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_path = output_path / RUN_METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)
    return str(metadata_path)


def load_run_metadata(search_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """从指定目录（或其父目录）加载实验元数据。

    Args:
        search_dir: 搜索目录路径。

    Returns:
        元数据字典，未找到时返回空字典。
    """
    base_path = Path(search_dir)
    candidate_paths = [
        base_path / RUN_METADATA_FILENAME,
        base_path.parent / RUN_METADATA_FILENAME,
    ]

    for candidate_path in candidate_paths:
        if not candidate_path.exists() or not candidate_path.is_file():
            continue
        with candidate_path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    return {}


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
        curriculum_profile=DEFAULT_CURRICULUM_PROFILE_NAME,
        reference_curve_dir=None,
        completion_alpha_max=None,
        experiment_tag=None,
        run_mode="tune",
        enable_tb=None,
        enable_monitor=None,
        enable_auto_analysis=None,
        enable_best_evaluation_artifacts=None,
        analysis_output_root="mtto_train_reports",
        analysis_min_points_per_10k_steps=5.0,
        analysis_sampling_quality_mode="warn_only",
        survival_reward_scale=None,
        reward_discount=DEFAULT_REWARD_DISCOUNT,
        num_envs=DEFAULT_NUM_ENVS,
        rollout_steps_per_update=DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        n_steps_per_env=None,
        total_timesteps=100_000,
        tensorboard_log_dir="mtto_ppo_tb_logs",
        tb_log_name=None,
        log_interval=None,
        evaluation_interval_rollouts=DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
        evaluation_deterministic=True,
        evaluation_history_path=None,
        enable_safety_truncation_histogram=False,
        safety_truncation_bin_size_m=5000.0,
        seed=None,
        device=DEFAULT_DEVICE,
        dry_run=False,
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


def _normalize_optional_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    return None if int(value) <= 0 else int(value)


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
    raw_survival_scale = getattr(args, "survival_reward_scale", None)
    if raw_survival_scale is not None:
        effective_scale = resolve_survival_reward_scale(raw_survival_scale)
        custom_reward_config = replace(
            reward_preset.config,
            survival_reward_scale=effective_scale,
        )
        reward_preset = replace(
            reward_preset,
            config=custom_reward_config,
        )
    curriculum_profile = resolve_curriculum_profile_name(
        getattr(args, "curriculum_profile", DEFAULT_CURRICULUM_PROFILE_NAME)
    )
    dspdl_config: DSPDLConfig | CompletionDSPDLConfig | None
    if curriculum_profile == "dspdl":
        dspdl_config = DSPDLConfig()
    elif curriculum_profile == "dspdl_completion":
        dspdl_config = CompletionDSPDLConfig()
    else:
        dspdl_config = None
    completion_alpha_max = getattr(args, "completion_alpha_max", None)
    if completion_alpha_max is not None:
        if curriculum_profile != "dspdl_completion" or not isinstance(
            dspdl_config, CompletionDSPDLConfig
        ):
            raise ValueError("completion_alpha_max is only valid with dspdl_completion")
        resolved_alpha_max = float(completion_alpha_max)
        if (
            not math.isfinite(resolved_alpha_max)
            or resolved_alpha_max < dspdl_config.alpha_min
        ):
            raise ValueError(
                "completion_alpha_max must be finite and not smaller than alpha_min"
            )
        dspdl_config = replace(dspdl_config, alpha_max=resolved_alpha_max)
    reference_curve_dir_raw = getattr(args, "reference_curve_dir", None)
    reference_curve_dir: str | None = None
    if curriculum_profile != "none":
        if (
            not isinstance(reference_curve_dir_raw, str)
            or not reference_curve_dir_raw.strip()
        ):
            raise ValueError(
                "reference_curve_dir is required when a curriculum profile is enabled"
            )
        reference_dir_path = Path(reference_curve_dir_raw)
        if not reference_dir_path.is_dir():
            raise FileNotFoundError(
                f"reference trajectory directory does not exist: {reference_dir_path}"
            )
        reference_curve_dir = str(reference_dir_path)

    output_root = args.output_root
    output_dir = resolve_output_dir(
        output_root=output_root,
        schedule_time_s=schedule_time_s,
        step_distance=ds,
        reward_preset_name=reward_preset.name,
        curriculum_profile_name=curriculum_profile,
        experiment_tag=args.experiment_tag,
    )
    final_output_dir = os.path.join(output_dir, "final")
    final_model_save_path = os.path.join(final_output_dir, RL_FINAL_MODEL_FILENAME)
    reward_diagnostics_path = os.path.join(final_output_dir, "reward_diagnostics.npz")
    best_eval_output_dir = os.path.join(output_dir, "best_rollouts")
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
        curriculum_profile_name=curriculum_profile,
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

    run_metadata = build_run_metadata(
        reward_preset=reward_preset,
        curriculum_profile=curriculum_profile,
        dspdl_config=dspdl_config,
        reference_curve_dir=reference_curve_dir,
        schedule_time_s=schedule_time_s,
        step_distance=ds,
        reward_discount=reward_discount,
        run_mode=run_mode,
        experiment_tag=args.experiment_tag,
        total_timesteps=int(args.total_timesteps),
        enable_tb=bool(enable_tb),
        enable_monitor=bool(enable_monitor),
        enable_auto_analysis=bool(enable_auto_analysis),
        enable_best_evaluation_artifacts=bool(enable_best_evaluation_artifacts),
        enable_safety_truncation_histogram=bool(enable_safety_truncation_histogram),
        safety_truncation_bin_size_m=safety_truncation_bin_size_m,
        evaluation_interval_rollouts=max(1, int(args.evaluation_interval_rollouts)),
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
    )

    return TrainingRunSpec(
        schedule_time_s=schedule_time_s,
        step_distance=ds,
        reward_discount=reward_discount,
        reward_preset=reward_preset,
        curriculum_profile=curriculum_profile,
        dspdl_config=dspdl_config,
        reference_curve_dir=reference_curve_dir,
        reference_curve_artifact_path=None,
        reference_curve_metrics_path=None,
        output_root=output_root,
        output_dir=output_dir,
        final_output_dir=final_output_dir,
        best_eval_output_dir=best_eval_output_dir,
        reward_diagnostics_path=reward_diagnostics_path,
        final_model_save_path=final_model_save_path,
        run_metadata_path=os.path.join(output_dir, RUN_METADATA_FILENAME),
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
        evaluation_interval_rollouts=max(1, int(args.evaluation_interval_rollouts)),
        evaluation_deterministic=bool(args.evaluation_deterministic),
        evaluation_history_path=evaluation_history_path,
        enable_safety_truncation_histogram=bool(enable_safety_truncation_histogram),
        safety_truncation_bin_size_m=safety_truncation_bin_size_m,
        total_timesteps=int(args.total_timesteps),
        device=args.device,
        seed=args.seed,
        dry_run=bool(args.dry_run),
    )


# =============================================================================
# 训练执行
# =============================================================================


def _cosine_annealing_schedule(
    initial_value: float, final_value: float = 1e-5
) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = final_value + (initial_value - final_value) * cosine_decay

        return lr

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
    context_pool: ContextPool | None = None,
    curriculum_distribution_state: CurriculumDistributionState | None = None,
    context_sampling_seed: int | None = None,
    dspdl_statistics_hub: DSPDLStatisticsHub | None = None,
    enable_completion_accumulator: bool = False,
    completion_config: CompletionDSPDLConfig | None = None,
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
            context_pool=context_pool,
            curriculum_distribution_state=curriculum_distribution_state,
            context_sampling_seed=context_sampling_seed,
            dspdl_statistics_hub=dspdl_statistics_hub,
            curriculum_env_rank=(
                worker_rank if dspdl_statistics_hub is not None else None
            ),
            enable_completion_accumulator=enable_completion_accumulator,
            completion_config=completion_config,
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

    context_pool: ContextPool | None = None
    curriculum_distribution_state: CurriculumDistributionState | None = None
    dspdl_statistics_hub: DSPDLStatisticsHub | None = None
    curriculum_callback: BaseCallback | None = None
    if resolved_spec.curriculum_profile != "none":
        if resolved_spec.reference_curve_dir is None:
            raise RuntimeError("enabled curriculum is missing reference_curve_dir")
        artifact = DPTrajectoryReader.resolve_matching_artifact(
            curve_dir=resolved_spec.reference_curve_dir,
            train_service=train_service,
        )
        reference_trajectory = DPTrajectoryReader.from_artifact(
            artifact=artifact,
            train_service=train_service,
        )
        context_pool = ContextPoolBuilder(
            reference_trajectory,
            stepper=shared_stepper,
        ).build()
        if resolved_spec.dspdl_config is None:
            raise RuntimeError("DSPDL curriculum is missing its configuration")
        observation_builder = ObservationBuilder(
            vehicle=vehicle,
            track=track,
            train_service=train_service,
            step_distance_m=resolved_spec.step_distance,
            direction=shared_stepper.direction,
            whole_distance_m=shared_stepper.whole_distance_m,
            get_upper_speed_or_zero=shared_stepper.get_upper_speed_or_zero,
        )
        context_observations = np.stack(
            [
                observation_builder.build(context.initial_state)
                for context in context_pool.contexts
            ],
            axis=0,
        )
        if resolved_spec.curriculum_profile == "dspdl_completion":
            if not isinstance(resolved_spec.dspdl_config, CompletionDSPDLConfig):
                raise TypeError(
                    "completion DSPDL curriculum has an invalid configuration"
                )
            curriculum_callback = CompletionDSPDLCallback(
                context_pool=context_pool,
                context_observations=context_observations,
                config=resolved_spec.dspdl_config,
                seed=resolved_spec.seed,
            )
            curriculum_distribution_state = curriculum_callback.distribution_state
        else:
            if not isinstance(resolved_spec.dspdl_config, DSPDLConfig):
                raise TypeError(
                    "traditional DSPDL curriculum has an invalid configuration"
                )
            dspdl_statistics_hub = DSPDLStatisticsHub(
                context_count=context_pool.context_count,
                num_envs=resolved_spec.num_envs,
                gamma=resolved_spec.reward_discount,
            )
            curriculum_callback = DSPDLCallback(
                context_pool=context_pool,
                context_observations=context_observations,
                config=resolved_spec.dspdl_config,
                statistics_hub=dspdl_statistics_hub,
            )
            curriculum_distribution_state = curriculum_callback.distribution_state
        curriculum_metadata = dict(resolved_spec.run_metadata["curriculum"])
        curriculum_metadata.update(
            {
                "reference_curve_artifact_path": artifact.npz_path,
                "reference_curve_metrics_path": artifact.metrics_path,
                "rl_step_distance_m": resolved_spec.step_distance,
                "context_count": context_pool.context_count,
                "initial_curriculum_version": 0,
            }
        )
        resolved_metadata = dict(resolved_spec.run_metadata)
        resolved_metadata["curriculum"] = curriculum_metadata
        resolved_spec = replace(
            resolved_spec,
            reference_curve_artifact_path=artifact.npz_path,
            reference_curve_metrics_path=artifact.metrics_path,
            run_metadata=resolved_metadata,
        )

    os.makedirs(resolved_spec.output_dir, exist_ok=True)
    os.makedirs(resolved_spec.final_output_dir, exist_ok=True)
    _ = save_run_metadata(resolved_spec.output_dir, resolved_spec.run_metadata)

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
            context_pool=context_pool,
            curriculum_distribution_state=curriculum_distribution_state,
            context_sampling_seed=(
                None if resolved_spec.seed is None else resolved_spec.seed + env_rank
            ),
            dspdl_statistics_hub=dspdl_statistics_hub,
            enable_completion_accumulator=(
                resolved_spec.curriculum_profile == "dspdl_completion"
            ),
            completion_config=(
                resolved_spec.dspdl_config
                if isinstance(resolved_spec.dspdl_config, CompletionDSPDLConfig)
                else None
            ),
            enable_safety_truncation_tracking=(
                resolved_spec.enable_safety_truncation_histogram
            ),
        )
        for env_rank in range(resolved_spec.num_envs)
    ]
    venv_train = DummyVecEnv(env_initializers)

    if resolved_spec.enable_monitor:
        venv_train = VecMonitor(venv_train)

    model = PPO(
        "MlpPolicy",
        venv_train,
        device=resolved_spec.device,
        verbose=0,
        learning_rate=_cosine_annealing_schedule(3e-4),
        # learning_rate=3e-4,
        n_steps=resolved_spec.n_steps_per_env,
        batch_size=256,
        n_epochs=10,
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

    if resolved_spec.curriculum_profile == "dspdl_completion":
        curriculum_metadata = dict(resolved_spec.run_metadata["curriculum"])
        curriculum_metadata["completion_critic"] = completion_critic_metadata(model)
        resolved_metadata = dict(resolved_spec.run_metadata)
        resolved_metadata["curriculum"] = curriculum_metadata
        resolved_spec = replace(resolved_spec, run_metadata=resolved_metadata)
        _ = save_run_metadata(resolved_spec.output_dir, resolved_spec.run_metadata)

    callbacks: list[BaseCallback] = []
    if curriculum_callback is not None:
        callbacks.append(curriculum_callback)
    callbacks.append(
        RewardDiagnosticsArtifactCallback(
            output_path=resolved_spec.reward_diagnostics_path
        )
    )
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
                    artifact_metadata=resolved_spec.run_metadata,
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
                deterministic=resolved_spec.evaluation_deterministic,
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    _ = model.learn(
        total_timesteps=resolved_spec.total_timesteps,
        callback=callback,
        log_interval=resolved_spec.log_interval,
        tb_log_name=resolved_spec.tb_log_name,
        progress_bar=True,
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
    """Evaluate ``final_model`` from a completed run at the real start state.

    This deliberately does not attach the curriculum reference-state provider:
    final artifacts always represent the deployed policy on the full task.
    """
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
            output_path=os.path.join(spec.final_output_dir, "final_trajectory.npz"),
            metadata=spec.run_metadata,
            deterministic=True,
        )
        return npz_path, metrics_path
    finally:
        env.close()


# =============================================================================
# 轨迹产物解析
# =============================================================================


def _find_latest_matching_file(
    *,
    search_dir: str,
    glob_pattern: str,
    filter_fn: Callable[[Path], bool] | None = None,
) -> Path:
    search_root = Path(search_dir)
    if not search_root.is_dir():
        raise FileNotFoundError(f"Search directory does not exist: {search_dir}")

    def _matches(path: Path) -> bool:
        if not path.is_file():
            return False
        if filter_fn is None:
            return True
        return bool(filter_fn(path))

    matches = sorted(
        (path for path in search_root.rglob(glob_pattern) if _matches(path)),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find files matching '{glob_pattern}' \
            under directory: {search_dir}"
        )

    if len(matches) > 1:
        print(
            f"Found {len(matches)} '{glob_pattern}' files under '{search_dir}', "
            + f"using latest: {matches[0]}"
        )
    return matches[0]


def _resolve_rl_metrics_path(curve_path: Path) -> Path:
    metrics_path = curve_path.with_name(
        f"{curve_path.stem}{_RL_TRAJECTORY_METRICS_SUFFIX}"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Could not find '{metrics_path.name}' in directory: {curve_path.parent}"
        )
    return metrics_path


def _best_trajectory_filter(path: Path) -> bool:
    return path.name == _RL_BEST_TRAJECTORY_FILENAME and path.parent.name.startswith(
        "best_"
    )


def _best_rollouts_trajectory_filter(path: Path) -> bool:
    return (
        path.name == _RL_BEST_TRAJECTORY_FILENAME
        and path.parent.name == "best_rollouts"
    )


def _final_trajectory_filter(path: Path) -> bool:
    return path.name == _RL_FINAL_TRAJECTORY_FILENAME and path.parent.name == "final"


def resolve_rl_curve_artifact(
    *,
    curve_dir: str,
    trajectory_source: str = "best",
) -> OptimizedCurveArtifact:
    """在训练输出目录中定位最优或最终轨迹产物。

    Args:
        curve_dir: 训练输出根目录。
        trajectory_source: 轨迹来源标识 (best/best_rollouts/final)。

    Returns:
        OptimizedCurveArtifact 实例。

    Raises:
        ValueError: 未知的 trajectory_source。
        FileNotFoundError: 未找到匹配的轨迹文件。
    """
    filter_map: dict[str, tuple[str, Callable[[Path], bool]]] = {
        "best": (_RL_BEST_TRAJECTORY_FILENAME, _best_trajectory_filter),
        "best_rollouts": (
            _RL_BEST_TRAJECTORY_FILENAME,
            _best_rollouts_trajectory_filter,
        ),
        "final": (_RL_FINAL_TRAJECTORY_FILENAME, _final_trajectory_filter),
    }
    try:
        file_name, filter_fn = filter_map[trajectory_source]
    except KeyError as exc:
        choices = ", ".join(RL_TRAJECTORY_SOURCE_CHOICES)
        raise ValueError(
            f"Unknown trajectory source '{trajectory_source}'. Choices: {choices}"
        ) from exc

    curve_path = _find_latest_matching_file(
        search_dir=curve_dir,
        glob_pattern=file_name,
        filter_fn=filter_fn,
    )
    metrics_path = _resolve_rl_metrics_path(curve_path)

    return OptimizedCurveArtifact(
        npz_path=str(curve_path),
        metrics_path=str(metrics_path),
    )


def load_rl_curve_artifact(
    artifact: OptimizedCurveArtifact,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """加载轨迹产物中的位置数组、速度数组和指标字典。

    Args:
        artifact: 由 resolve_rl_curve_artifact 返回的产物定位信息。

    Returns:
        (pos_arr, speed_arr, metrics) 三元组。
    """
    pos_arr, speed_arr, metrics = load_optimized_curve_and_metrics(
        npz_path=artifact.npz_path,
        metrics_path=artifact.metrics_path,
        dtype=np.float32,
        use_metrics_cache=True,
    )

    return pos_arr, speed_arr, metrics


def load_rl_curve_metrics(artifact: OptimizedCurveArtifact) -> dict[str, object]:
    """仅加载轨迹产物的指标字典（不加载位置/速度数组）。

    Args:
        artifact: 由 resolve_rl_curve_artifact 返回的产物定位信息。

    Returns:
        指标字典。

    Raises:
        FileNotFoundError: 指标文件不存在。
    """
    metrics_path = Path(artifact.metrics_path)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {artifact.metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)

    return metrics


# =============================================================================
# 轨迹指标 & 对比
# =============================================================================


def _metric_as_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def build_rl_trajectory_comparison_key(
    metrics: Mapping[str, object],
) -> tuple[float, ...]:
    """构建用于多轨迹排序对比的键。

    排序优先级：success > 高总奖励(仅非success) > 精确到站 > 小停站误差
    > 准点到站 > 小时间误差 > 低能耗。

    Args:
        metrics: 轨迹指标字典。

    Returns:
        可直接用于 max() 的比较键。
    """
    raw_key = metrics.get("selection_comparison_key")
    if isinstance(raw_key, (list, tuple)) and len(raw_key) > 0:
        converted: list[float] = []
        for value in raw_key:
            numeric_value = _metric_as_float(value)
            if numeric_value is None:
                raise ValueError("selection_comparison_key must contain only numbers")
            converted.append(numeric_value)
        return tuple(converted)

    if raw_key is not None:
        raise ValueError("selection_comparison_key must be a non-empty numeric list")

    success = bool(metrics.get("success", False))
    total_reward = _metric_as_float(metrics.get("total_reward"))
    total_energy_j = _metric_as_float(metrics.get("total_energy_j"))
    stop_error_m = _metric_as_float(metrics.get("stop_error_m"))
    time_error_s = _metric_as_float(metrics.get("time_error_s"))
    strict_stop_error_limit_m = _metric_as_float(
        metrics.get("strict_stop_error_limit_m")
    )
    strict_time_error_limit_s = _metric_as_float(
        metrics.get("strict_time_error_limit_s")
    )

    required = {
        "total_reward": total_reward,
        "total_energy_j": total_energy_j,
        "stop_error_m": stop_error_m,
        "time_error_s": time_error_s,
        "strict_stop_error_limit_m": strict_stop_error_limit_m,
        "strict_time_error_limit_s": strict_time_error_limit_s,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "metrics are missing required trajectory selection fields: "
            + ", ".join(missing)
        )

    assert total_reward is not None
    assert total_energy_j is not None
    assert stop_error_m is not None
    assert time_error_s is not None
    assert strict_stop_error_limit_m is not None
    assert strict_time_error_limit_s is not None

    if not success:
        return (0.0, float(total_reward))

    precise_arrival_value = metrics.get("precise_arrival")
    punctual_arrival_value = metrics.get("punctual_arrival")
    precise_arrival = (
        bool(precise_arrival_value)
        if isinstance(precise_arrival_value, bool)
        else float(stop_error_m) <= float(strict_stop_error_limit_m)
    )
    punctual_arrival = (
        bool(punctual_arrival_value)
        if isinstance(punctual_arrival_value, bool)
        else precise_arrival
        and abs(float(time_error_s)) < float(strict_time_error_limit_s)
    )
    return (
        1.0,
        1.0 if precise_arrival else 0.0,
        0.0 if precise_arrival else -float(stop_error_m),
        1.0 if punctual_arrival else 0.0,
        0.0 if punctual_arrival else -abs(float(time_error_s)),
        -float(total_energy_j),
    )


# =============================================================================
# 可视化辅助
# =============================================================================


def apply_rl_curve_plot_style() -> None:
    """设置 RL 轨迹曲线绘图的全局 matplotlib 样式。"""
    _ = set_global_plot_style(
        font_preset="sci",
        preferred_font="Times New Roman",
        title_font_size=8.0,
        axis_label_font_size=8.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=150.0,
        savefig_dpi=300.0,
    )


def get_rl_trajectory_status_text(metrics: dict[str, object]) -> str | None:
    """根据轨迹指标返回中文状态描述文本。

    Args:
        metrics: 轨迹指标字典（需含 success 和 trajectory_source 键）。

    Returns:
        如 "RL 最优轨迹（完成任务）" 或 None（success 不是 bool 时）。
    """
    success_value = metrics.get("success")
    trajectory_source = metrics.get("trajectory_source")
    if not isinstance(success_value, bool):
        return None
    prefix = "RL 最终轨迹" if trajectory_source == "final" else "RL 最优轨迹"
    return f"{prefix}（完成任务）" if success_value else f"{prefix}（未完成任务）"


def add_panel_label(
    *,
    ax: Any,
    label: str,
    x: float = 0.02,
    y: float = 0.98,
    fontsize: float = 10.0,
) -> Any:
    """在 matplotlib Axes 左上角添加加粗面板标签 (如 "a", "b")。

    Args:
        ax: matplotlib Axes 对象。
        label: 标签文本。
        x: 相对 x 坐标。
        y: 相对 y 坐标。
        fontsize: 字体大小。

    Returns:
        matplotlib Text 实例。
    """
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
    )


def format_rl_trajectory_terminal_summary(
    metrics: dict[str, object],
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
    effective_preset = reward_preset_name or str(
        metrics.get("reward_preset_name", "unknown")
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
        if key in metrics:
            fields.append(f"{key}={metrics[key]}")

    if artifact_path:
        fields.append(f"artifact={artifact_path}")
    return " | ".join(fields)


def _get_rl_trajectory_display_name(metrics: dict[str, object]) -> str:
    trajectory_source = metrics.get("trajectory_source")
    if trajectory_source == "final":
        return "RL final trajectory"
    return "RL best trajectory"


def render_rl_curve_on_axes(
    *,
    ax: Any,
    pos_arr: np.ndarray,
    speed_arr: np.ndarray,
    metrics: dict[str, object],
    no_safeguard: bool,
    factor: float,
    curve_color: str = "blue",
    curve_label: str | None = None,
    safeguard: SafeGuardUtility | None = None,
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
    """
    if not no_safeguard:
        resolved_safeguard = (
            safeguard if safeguard is not None else build_safeguard_utility(factor)
        )
        resolved_safeguard.render(ax=ax, layers=SafeGuardUtility.DANGER_VIEW_LAYERS)

    ax.plot(
        pos_arr,
        speed_arr * 3.6,
        color=curve_color,
        alpha=0.85,
        linewidth=1.5,
        label=curve_label or _get_rl_trajectory_display_name(metrics),
    )

    start_position = _metric_as_float(metrics.get("start_position_m"))
    target_position = _metric_as_float(metrics.get("target_position_m"))

    if start_position is not None:
        ax.scatter(
            start_position,
            0.0,
            marker="o",
            color="green",
            s=40,
            alpha=0.85,
            label="start",
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )
    if target_position is not None:
        ax.scatter(
            target_position,
            0.0,
            marker="o",
            color="red",
            s=40,
            alpha=0.85,
            label="end",
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_xlim((0.0, 30000.0))
    ax.set_ylim((0.0, 500.0))
    ax.grid(True, alpha=0.3)
