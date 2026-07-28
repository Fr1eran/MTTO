import math
import os
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.typing import NDArray

from model.common import ECC, ORS
from model.ocs import SPS, SafeGuardUtility, TrainService
from model.track import TrackInfo, get_slope_scalar_numba, get_speed_limit_scalar_numba
from model.vehicle import VehicleInfo, calc_levi_deceleration_scalar_numba
from utils.indexing_utils import get_interval_index
from utils.plot_utils import set_chinese_font
from utils.score_function import SigmoidVariant

DEFAULT_PUNCTUALITY_DP_CURVE_DIR = os.path.join("output", "optimal", "dp")


class RewardInfoForTB(TypedDict, total=False):
    safety: float
    stopping: float
    punctuality: float
    energy: float
    comfort: float
    total: float


class StateInfoForTB(TypedDict, total=False):
    position: float
    speed: float
    stopping_point_index: int


class ConstraintInfoForTB(TypedDict, total=False):
    margin_to_vmax_mps: float
    margin_to_vmin_mps: float
    is_truncated: bool
    violation_code: int
    speed_limit_mps: float
    speed_limit_segment: int
    is_near_miss: bool


class EventInfoForTB(TypedDict, total=False):
    episode_truncated_count: int
    episode_low_violation_count: int
    episode_high_violation_count: int


class BasicInfo(TypedDict, total=False):
    position: float
    speed: float
    stopping_point_index: int
    operation_time: float
    redundant_operation_time: float
    energy_consumption: float
    comfort_tav: float
    comfort_er_pct: float
    comfort_rms: float


class DiagnosticsSnapshotForTB(TypedDict, total=False):
    rewards: RewardInfoForTB
    state: StateInfoForTB
    constraint: ConstraintInfoForTB
    event: EventInfoForTB
    runtime: BasicInfo


class TrainState(TypedDict, total=True):
    pos: float
    speed: float
    acc: float
    min_speed: float
    max_speed: float
    # latest_traction_intervention_point: float
    # latest_braking_intervention_point: float
    operation_time: float
    redundant_operation_time: float
    energy_consumption: float
    stopping_point_index: int


@dataclass
class RewardConfig:
    """稠密奖励分量开关配置，用于消融实验。"""

    enable_energy: bool = True
    enable_comfort: bool = True
    enable_potential_safety: bool = True
    enable_potential_stopping: bool = True
    enable_potential_punctuality: bool = True


class MTTOEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    VIOLATION_CODE_ONGOING = 0
    VIOLATION_CODE_FAILED_STOP = 1
    VIOLATION_CODE_SPEED_LOW = 2
    VIOLATION_CODE_SPEED_HIGH = 3
    VIOLATION_CODE_STEP_LIMIT = 4

    def __init__(
        self,
        vehicle: VehicleInfo,
        track: TrackInfo,
        safeguard_utility: SafeGuardUtility,
        train_service: TrainService,
        gamma: float,
        max_step_distance: float,
        enable_diagnostics: bool = False,
        diagnostics_interval_steps: int = 1,
        enable_trajectory_tracking: bool = False,
        render_mode: str | None = None,
        use_animation: bool = False,
        reward_config: RewardConfig | None = None,
        punctuality_dp_curve_dir: str | os.PathLike[str] | None = (
            DEFAULT_PUNCTUALITY_DP_CURVE_DIR
        ),
        punctuality_reference_match_tolerance: float = 1e-3,
    ):
        super().__init__()

        # 磁浮列车运行优化车辆实例
        self.vehicle = vehicle

        # 磁浮列车运行优化的线路实例
        self.track = track

        # 磁浮列车安全防护实例
        self.safeguard_utility = safeguard_utility

        # 运行任务约束
        self.train_service = train_service

        # 回报折扣因子
        self.gamma = gamma

        # 奖励分量配置
        self.reward_config = (
            reward_config if reward_config is not None else RewardConfig()
        )
        self.punctuality_dp_curve_dir = punctuality_dp_curve_dir
        self.punctuality_reference_match_tolerance = float(
            punctuality_reference_match_tolerance
        )

        # 是否采集诊断信息
        self.enable_diagnostics = enable_diagnostics
        self.diagnostics_interval_steps = max(1, int(diagnostics_interval_steps))
        self._collect_step_diagnostics: bool = False

        # 单步状态转移容许的最大位移量
        self.max_step_distance: float = max_step_distance

        # 最大转移步数
        self.max_episode_steps: int = (
            math.ceil(
                abs(
                    self.train_service.target_position
                    - self.train_service.start_position
                )
                / self.max_step_distance
            )
            + 1
        )

        # 精确停站任务计分函数
        self._stopping_score_func = SigmoidVariant(
            x1=self.train_service.max_stop_error,
            x2=9.0,
            c=10.0,
        )

        # 准点到站任务计分函数
        self._punctuality_score_func = SigmoidVariant(
            x1=10.0,
            x2=60.0,
            c=8.0,
        )

        # 定义常量
        # 包含：
        # - 运行总距离, 单位: m
        # - 运动方向, 1为正向, -1为负向
        # - 目标速度, 单位: m/s
        # - 目标吸引域的半径，单位: m
        self.whole_distance: float = abs(
            self.train_service.target_position - self.train_service.start_position
        )
        self.direction: int = (
            1
            if self.train_service.start_position < self.train_service.target_position
            else -1
        )
        self.goal_speed: float = 0.0
        self.target_attraction_domain_radius = 3000.0

        # 能耗计算类
        self.ecc = ECC(
            R_m=0.2796,
            L_d=0.0002,
            R_k=50.0,
            L_k=0.000142,
            Tau=0.258,
            Psi_fd=3.9629,
            k_c=0.8,
        )

        # 参考运行系统
        self.ors = ORS(
            vehicle=self.vehicle,
            track=self.track,
            factor=self.safeguard_utility.gamma,
        )

        # 计算最短运行时间参考曲线 - 速度上限曲线
        self.upper_speed_profile_pos_arr, self.upper_speed_profile_speed_arr = (
            self.ors.calc_min_operation_time_curve(
                begin_pos=self.train_service.start_position,
                begin_speed=self.train_service.start_speed,
                end_pos=self.train_service.target_position
                + self.train_service.max_stop_error * 20,
                end_speed=0.0,
            )
        )
        # 使用 np.interp 预采样速度上限曲线，训练步进阶段通过数组查表获取上限速度。
        (
            self._upper_speed_lut_pos_min,
            self._upper_speed_lut_step,
            self._upper_speed_lut_speed_arr,
        ) = self._build_upper_speed_lookup_table(
            self.upper_speed_profile_pos_arr,
            self.upper_speed_profile_speed_arr,
        )

        # 计算最大能耗和最短运行时间
        mec, lec, self.min_operation_time = (
            self.ors.calc_max_energy_and_min_operation_time(
                begin_pos=self.train_service.start_position,
                begin_speed=self.train_service.start_speed,
                end_pos=self.train_service.target_position,
                end_speed=0.0,
                distance=self.train_service.target_position
                - self.train_service.start_position,
                energy_con_calc=self.ecc,
            )
        )
        self.max_energy_consumption = mec + lec

        # 初始（最大）冗余时间
        self.max_redundant_operation_time = (
            self.train_service.schedule_time - self.min_operation_time
        )

        # 计算参考曲线上每个位置对应的最短累计耗时
        self.ref_curve_cum_time = self._calc_ref_cum_time()
        self.ref_total_operation_time = self._get_ref_cum_time(
            self.train_service.target_position
        )
        self.ref_redundant_operation_time_pos_arr = np.asarray([], dtype=np.float64)
        self.ref_redundant_operation_time_arr = np.asarray([], dtype=np.float64)
        self.ref_dp_speed_pos_arr = np.asarray([], dtype=np.float64)
        self.ref_dp_speed_arr = np.asarray([], dtype=np.float64)
        if self.reward_config.enable_potential_punctuality:
            self._load_ref_redundant_operation_time_from_dp()

        # 初始化状态
        # 包含:
        # - 当前位置, 单位: m
        # - 当前运行速度大小, 单位: m/s
        # - 当前加速度, 单位: m/s^2
        # - 当前列车运行总时间
        # - 当前列车消耗总能量, 单位: J
        # - 列车质量（对智能体决策似乎不起作用）, 单位: t
        # - 当前位置对应的坡度, 千分位
        # - 当前位置对应的最大运行速度大小, 单位: m/s
        # - 当前位置对应的最小运行速度大小, 单位: m/s
        self.current_position: float = self.train_service.start_position
        self.current_speed: float = self.train_service.start_speed
        self.current_acc: float = 0.0
        self.current_operation_time: float = 0.0
        self.current_redundant_operation_time: float = (
            self._calc_redundant_operation_time()
        )
        self.current_energy_consumption: float = 0.0
        # self.mass = self.vehicle.mass
        self.current_slope: float = get_slope_scalar_numba(
            self.current_position,
            self.track.slopes,
            self.track.slope_intervals,
        )
        self.current_stopping_point_index: int = (
            -1
        )  # 初始时在加速区，尚未步进至第一个辅助停车区
        self.current_min_speed, self.current_max_speed = (
            self.safeguard_utility.get_min_and_max_speed(
                current_pos=self.current_position,
                current_sp=self.current_stopping_point_index,
            )
        )
        self.current_max_speed: float = min(
            self._get_upper_speed(self.current_position),
            self.current_max_speed,
        )
        # (
        #     self.current_latest_traction_intervention_point,
        #     self.current_latest_braking_intervention_point,
        # ) = self.safeguard_utility.get_latest_traction_and_braking_intervention_points(  # noqa: E501
        #     current_speed=self.current_speed, current_sp=self.current_sp
        # )

        self.stop_error: float = abs(
            self.train_service.target_position - self.current_position
        )

        # 定义智能体能够观测的状态信息（扁平化向量，避免每步构造字典对象）
        obs_low = np.array(
            [
                0.0,  # remaining_distance
                0.0,  # current_speed
                -1.0,  # current_acc
                -1.0,  # suggested_dec
                -1.0,  # remaining_schedule_time
                -1.0,  # time_redundancy
                0.0,  # current_max_speed
                0.0,  # current_min_speed
                -1.0,  # current_slope
                -1.0,  # lookahead_avg_slope
                0.0,  # lookahead_avg_upper_speed
                0.0,  # approach_progress
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                1.0,  # remaining_distance
                1.0,  # current_speed
                1.0,  # current_acc
                1.0,  # suggested_dec
                1.0,  # remaining_schedule_time
                1.0,  # time_redundancy
                1.0,  # current_max_speed
                1.0,  # current_min_speed
                1.0,  # current_slope
                1.0,  # lookahead_avg_slope
                1.0,  # lookahead_avg_upper_speed
                1.0,  # approach_progress
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        # 定义智能体的动作空间, 归一化
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # 当前训练步数
        self.current_steps: int = 0

        # 当前步进停车点编号
        self.current_stopping_point_index: int = -1

        # 停车点步进机制模拟
        self.sps = SPS(
            sgu=self.safeguard_utility,
            ASA_ap_list=self.track.ASA_aps,
            ASA_dp_list=self.track.ASA_dps,
            T_s=2.0,
        )

        # Q初始值
        # self.q_init: float = 0.0

        self.basic_info: BasicInfo = {}
        self.rewards_info: RewardInfoForTB = {}
        self.state_info: StateInfoForTB = {}
        self.constraint_info: ConstraintInfoForTB = {}
        self.event_info: EventInfoForTB = {}

        self.episode_truncated_count: int = 0
        self.episode_low_violation_count: int = 0
        self.episode_high_violation_count: int = 0
        self._last_violation_code: int = self.VIOLATION_CODE_ONGOING

        # 状态历史记录
        self._reset_history()

        # 渲染模式
        self.render_mode = render_mode
        # 是否启用动画
        self.use_animation = use_animation
        # 轨迹缓存与渲染解耦，训练和评估都可按需启用轨迹记录。
        self.enable_trajectory_tracking = enable_trajectory_tracking

        # 列车运行轨迹
        self.trajectory_pos: list[float] | None = None
        self.trajectory_speed_mps: list[float] | None = None
        if self.enable_trajectory_tracking:
            self._reset_trajectory()

        # 可视化时需要的绘图实例
        self.fig = None
        self.ax = None
        self.vehicle_dot, self.traj_line = None, None
        self.animation = None
        self.animation_running = False

        # 动画配置
        self.animation_interval = 15  # 动画更新间隔(ms)

        # 舒适度指标累积量
        self._comfort_tav: float = 0.0
        self._comfort_sum_sq_delta_acc: float = 0.0
        self._comfort_exceedance_count: int = 0

    def _reset_infos_diagnostic(self):
        self.rewards_info = {}
        self.state_info = {}
        self.constraint_info = {}
        self.event_info = {}

    def _reset_episode_counters(self) -> None:
        self.episode_truncated_count = 0
        self.episode_low_violation_count = 0
        self.episode_high_violation_count = 0

    def _get_obs(self) -> NDArray[np.float32]:
        """
        将内部状态转换为可观测形式，并进行归一化
        可观测状态全部由np.ndarray组成

        Returns:
            np.ndarray: shape=(12,), dtype=np.float32 的扁平化观测向量
        """

        dist_to_target = self.train_service.target_position - self.current_position
        suggested_dec = self._calc_coasting_acc()
        if abs(dist_to_target) <= self.target_attraction_domain_radius:
            # 末段按匀减速停车估算所需减速度（物理符号为负）并映射到动作归一化区间
            dist_abs = max(abs(dist_to_target), 1e-6)
            suggested_dec = -(self.current_speed**2) / (2.0 * dist_abs)

        suggested_dec_normalized = self._normalize_acc_to_action(suggested_dec)

        obs = np.array(
            [
                dist_to_target / self.whole_distance,
                self.current_speed / self.vehicle.max_speed,
                self._normalize_acc_to_action(self.current_acc),
                suggested_dec_normalized,
                (self.train_service.schedule_time - self.current_operation_time)
                / self.train_service.schedule_time,
                self.current_redundant_operation_time
                / self.train_service.schedule_time,
                self.current_max_speed / self.vehicle.max_speed,
                self.current_min_speed / self.vehicle.max_speed,
                self.current_slope / self.vehicle.max_slope_capacity,
                self._calc_lookahead_avg_slope() / self.vehicle.max_slope_capacity,
                self._calc_lookahead_avg_upper_speed() / self.vehicle.max_speed,
                self._calc_approach_progress(dist_to_target),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _clip_observation_value(self, value: float | np.floating) -> float:
        """将归一化观测值裁剪到 observation_space 声明范围内。"""
        return float(np.clip(float(value), -1.0, 1.0))

    def _calc_approach_progress(self, dist_to_target: float | np.floating) -> float:
        progress = (
            1.0 - abs(float(dist_to_target)) / self.target_attraction_domain_radius
        )
        return float(np.clip(progress, 0.0, 1.0))

    def _calc_lookahead_avg_slope(
        self,
        lookahead_distance: float = 1000.0,
        num_samples: int = 10,
    ) -> float:
        offsets = np.linspace(
            self.max_step_distance,
            lookahead_distance,
            max(1, int(num_samples)),
            dtype=np.float64,
        )
        slope_sum = 0.0
        for offset in offsets:
            pos = self.current_position + self.direction * float(offset)
            slope_sum += get_slope_scalar_numba(
                pos,
                self.track.slopes,
                self.track.slope_intervals,
            )
        return slope_sum / float(offsets.size)

    def _get_upper_speed_or_zero(self, pos: float | np.floating) -> float:
        if self._upper_speed_lut_speed_arr.size == 0:
            return 0.0

        pos_value = float(pos)
        pos_min = self._upper_speed_lut_pos_min
        pos_max = pos_min + self._upper_speed_lut_step * (
            self._upper_speed_lut_speed_arr.size - 1
        )
        if pos_value < pos_min or pos_value > pos_max:
            return 0.0

        return self._get_upper_speed(pos_value)

    def _calc_lookahead_avg_upper_speed(
        self,
        lookahead_distance: float = 1000.0,
        num_samples: int = 10,
    ) -> float:
        offsets = np.linspace(
            self.max_step_distance,
            lookahead_distance,
            max(1, int(num_samples)),
            dtype=np.float64,
        )
        upper_speed_sum = 0.0
        for offset in offsets:
            pos = self.current_position + self.direction * float(offset)
            upper_speed_sum += self._get_upper_speed_or_zero(pos)

        return upper_speed_sum / float(offsets.size)

    def _calc_coasting_acc(self) -> float:
        """计算当前状态下惰行时带物理符号的加速度，单位 m/s^2。"""
        coasting_dec = calc_levi_deceleration_scalar_numba(
            self.current_speed,
            self.current_slope,
            self.vehicle.mass,
            self.vehicle.numoftrainsets,
        )
        return -coasting_dec

    def _normalize_acc_to_action(self, acc: float | np.floating) -> float:
        """将物理加速度映射到动作归一化区间[-1, 1]，并做截断。"""
        normalized = (
            2.0
            * (float(acc) - self.vehicle.max_dec)
            / (self.vehicle.max_acc - self.vehicle.max_dec)
            - 1.0
        )
        return float(np.clip(normalized, -1.0, 1.0))

    def _get_basic_info(self):
        """
        获取反映轨迹性能指标的信息
        """
        runtime_snapshot = dict(self.basic_info)
        if not runtime_snapshot:
            return {}
        return {"basic": runtime_snapshot}

    def _get_speed_limit_segment(self, pos: float) -> int:
        segment_idx = get_interval_index(pos, self.track.speed_limit_intervals)
        return int(np.clip(segment_idx, 0, len(self.track.speed_limits) - 1))

    def _should_collect_step_diagnostics(self) -> bool:
        if not self.enable_diagnostics:
            return False
        return self.current_steps % self.diagnostics_interval_steps == 0

    def _record_basic_info(self) -> None:
        self.basic_info["energy_consumption"] = self.current_energy_consumption
        self.basic_info["operation_time"] = self.current_operation_time
        self.basic_info["redundant_operation_time"] = (
            self.current_redundant_operation_time
        )
        self.basic_info["position"] = self.current_position
        self.basic_info["speed"] = self.current_speed
        self.basic_info["stopping_point_index"] = self.current_stopping_point_index
        self.basic_info["comfort_tav"] = self._comfort_tav
        self.basic_info["comfort_er_pct"] = (
            self._comfort_exceedance_count / self.current_steps * 100.0
        )
        self.basic_info["comfort_rms"] = math.sqrt(
            self._comfort_sum_sq_delta_acc / self.current_steps
        )

    def _record_step_diagnostics(
        self,
        near_miss_margin_mps: float = 1.0,
    ) -> None:
        margin_to_vmax = self.current_max_speed - self.current_speed
        margin_to_vmin = self.current_speed - self.current_min_speed

        if margin_to_vmin < 0.0:
            self.episode_truncated_count += 1
            self.episode_low_violation_count += 1
        elif margin_to_vmax < 0.0:
            self.episode_truncated_count += 1
            self.episode_high_violation_count += 1

        is_truncated = self._last_violation_code in {
            self.VIOLATION_CODE_FAILED_STOP,
            self.VIOLATION_CODE_SPEED_LOW,
            self.VIOLATION_CODE_SPEED_HIGH,
            self.VIOLATION_CODE_STEP_LIMIT,
        }

        is_near_miss = (margin_to_vmax <= near_miss_margin_mps) or (
            margin_to_vmin <= near_miss_margin_mps
        )

        self.state_info["position"] = self.current_position
        self.state_info["speed"] = self.current_speed
        self.state_info["stopping_point_index"] = self.current_stopping_point_index

        self.constraint_info["margin_to_vmax_mps"] = margin_to_vmax
        self.constraint_info["margin_to_vmin_mps"] = margin_to_vmin
        self.constraint_info["is_truncated"] = is_truncated
        self.constraint_info["violation_code"] = self._last_violation_code
        self.constraint_info["speed_limit_mps"] = get_speed_limit_scalar_numba(
            self.current_position,
            self.track.speed_limits,
            self.track.speed_limit_intervals,
        )
        self.constraint_info["speed_limit_segment"] = self._get_speed_limit_segment(
            self.current_position
        )
        self.constraint_info["is_near_miss"] = is_near_miss

        self.event_info["episode_truncated_count"] = self.episode_truncated_count
        self.event_info["episode_low_violation_count"] = (
            self.episode_low_violation_count
        )

        self.event_info["episode_high_violation_count"] = (
            self.episode_high_violation_count
        )

    def _get_action_denormalized(self, action: float | np.floating) -> float:
        """将动作反归一化为列车加速度"""
        return float(
            (self.vehicle.max_acc + self.vehicle.max_dec) / 2
            + action * (self.vehicle.max_acc - self.vehicle.max_dec) / 2
        )

    def _load_ref_redundant_operation_time_from_dp(self) -> None:
        if self.punctuality_dp_curve_dir is None:
            raise FileNotFoundError(
                "punctuality_dp_curve_dir must be provided to load the v18 "
                "DP reference redundant operation-time manifold."
            )

        pos_arr, speed_arr, _cum_time_arr, ref_redundant_operation_time_arr = (
            self.ors.load_or_build_ref_redundant_operation_time_from_dp(
                start_position=self.train_service.start_position,
                start_speed=self.train_service.start_speed,
                target_position=self.train_service.target_position,
                target_speed=0.0,
                schedule_time_s=self.train_service.schedule_time,
                dp_curve_dir=self.punctuality_dp_curve_dir,
                force_recompute=False,
                match_tolerance=self.punctuality_reference_match_tolerance,
            )
        )

        pos = np.asarray(pos_arr, dtype=np.float64)
        speed = np.asarray(speed_arr, dtype=np.float64)
        ref_redundant = np.asarray(ref_redundant_operation_time_arr, dtype=np.float64)
        if pos.ndim != 1 or speed.ndim != 1 or ref_redundant.ndim != 1:
            raise ValueError(
                "DP reference position, speed, and redundant-time arrays must be 1-D"
            )
        if pos.size != speed.size or pos.size != ref_redundant.size:
            raise ValueError(
                "DP reference position, speed, and redundant-time arrays must have "
                "equal length"
            )
        if pos.size == 0:
            raise ValueError("DP reference speed and redundant-time manifold is empty")
        if not (
            np.all(np.isfinite(pos))
            and np.all(np.isfinite(speed))
            and np.all(np.isfinite(ref_redundant))
        ):
            raise ValueError("DP reference arrays must contain only finite values")

        order = np.argsort(pos)
        pos_sorted = pos[order]
        speed_sorted = speed[order]
        ref_sorted = ref_redundant[order]
        keep_mask = np.empty(pos_sorted.size, dtype=bool)
        keep_mask[0] = True
        keep_mask[1:] = np.diff(pos_sorted) != 0.0

        self.ref_dp_speed_pos_arr = pos_sorted[keep_mask]
        self.ref_dp_speed_arr = speed_sorted[keep_mask]
        self.ref_redundant_operation_time_pos_arr = pos_sorted[keep_mask]
        self.ref_redundant_operation_time_arr = ref_sorted[keep_mask]

    def _get_ref_dp_speed(self, pos: float | np.floating) -> float:
        if self.ref_dp_speed_pos_arr.size == 0 or self.ref_dp_speed_arr.size == 0:
            raise RuntimeError("DP reference speed curve is not initialized.")

        return float(
            np.interp(
                float(pos),
                self.ref_dp_speed_pos_arr,
                self.ref_dp_speed_arr,
                left=float(self.ref_dp_speed_arr[0]),
                right=float(self.ref_dp_speed_arr[-1]),
            )
        )

    def _get_ref_redundant_operation_time(self, pos: float | np.floating) -> float:
        if (
            self.ref_redundant_operation_time_pos_arr.size == 0
            or self.ref_redundant_operation_time_arr.size == 0
        ):
            raise RuntimeError(
                "DP reference redundant operation-time manifold is not initialized."
            )

        return float(
            np.interp(
                float(pos),
                self.ref_redundant_operation_time_pos_arr,
                self.ref_redundant_operation_time_arr,
                left=float(self.ref_redundant_operation_time_arr[0]),
                right=float(self.ref_redundant_operation_time_arr[-1]),
            )
        )

    def change_schedule_time(self, new_schedule_time: float):
        if new_schedule_time != self.train_service.schedule_time:
            self.train_service.schedule_time = new_schedule_time
            self._punctuality_score_func = SigmoidVariant(
                x1=self.train_service.schedule_time
                * self.train_service.max_arr_time_error_ratio,
                x2=self.train_service.schedule_time
                * self.train_service.max_arr_time_error_ratio
                * 10.0,
                c=6.0,
            )
            if self.reward_config.enable_potential_punctuality:
                self._load_ref_redundant_operation_time_from_dp()
            self.current_redundant_operation_time = (
                self._calc_redundant_operation_time()
            )

        return self._get_obs()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        开启新回合

        Args:
            seed: 用于可重现回合的随机数种子
            options: 附加配置信息（可能用不上）

        Returns:
            tuple: (observation, info) for the initial state

        """
        # 首先调用此超类方法设置随机数生成器
        super().reset(seed=seed, options=options)

        # 重置运行状态
        self.current_position = self.train_service.start_position
        self.current_speed = self.train_service.start_speed
        self.current_acc = 0.0
        self.current_operation_time = 0.0
        self.current_redundant_operation_time = self._calc_redundant_operation_time()
        self.current_energy_consumption = 0.0
        # self.mass = self.vehicle.mass
        self.current_slope = get_slope_scalar_numba(
            self.current_position,
            self.track.slopes,
            self.track.slope_intervals,
        )

        # 重置停车点步进
        self.current_stopping_point_index = -1
        self.sps.reset()

        self.current_min_speed, self.current_max_speed = (
            self.safeguard_utility.get_min_and_max_speed(
                current_pos=self.current_position,
                current_sp=self.current_stopping_point_index,
            )
        )
        self.current_max_speed = min(
            self._get_upper_speed(self.current_position),
            self.current_max_speed,
        )
        # (
        #     self.current_latest_traction_intervention_point,
        #     self.current_latest_braking_intervention_point,
        # ) = self.safeguard_utility.get_latest_traction_and_braking_intervention_points(  # noqa: E501
        #     current_speed=self.current_speed, current_sp=self.current_sp
        # )

        self.stop_error = abs(
            self.train_service.target_position - self.current_position
        )

        self.basic_info = {}

        if self.enable_diagnostics:
            self._reset_infos_diagnostic()
            self._reset_episode_counters()

            self._collect_step_diagnostics = False

        self._comfort_tav = 0.0
        self._comfort_sum_sq_delta_acc = 0.0
        self._comfort_exceedance_count = 0

        # 重置历史数据
        self._reset_history()

        # 重置轨迹数据
        if self.enable_trajectory_tracking:
            self._reset_trajectory()

        # 重置仿真步数
        self.current_steps = 0
        self._last_violation_code = self.VIOLATION_CODE_ONGOING

        observation = self._get_obs()
        info = self._get_basic_info()

        return observation, info

    def step(self, action):  # type: ignore
        """
        在环境中执行一个时间步

        Args:
            action: 需要执行的动作，即列车运行加速度百分比

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_steps += 1
        self._record_history()
        self.current_acc = self._get_action_denormalized(action[0])

        # 累积舒适度指标
        _delta_acc = abs(self.current_acc - self.last_state["acc"])
        self._comfort_tav += _delta_acc
        self._comfort_sum_sq_delta_acc += _delta_acc**2
        if _delta_acc > self.train_service.max_acc_change:
            self._comfort_exceedance_count += 1

        # 根据当前速度大小、加速度、状态转移最大位移量
        # 计算转移至下一状态的速度大小、位移量和运行时间
        next_speed, distance, operation_time = self._update_motion()

        # 计算当前能耗
        current_mec, current_lec = self.ecc.calc_energy(
            begin_pos=self.last_state["pos"],
            begin_speed=self.last_state["speed"],
            acc=self.current_acc,
            distance=distance,
            direction=self.direction,
            operation_time=operation_time,
            vehicle=self.vehicle,
            track=self.track,
        )
        energy_consumption = current_mec + current_lec

        # 更新运行状态
        self.current_position += distance * self.direction
        self.current_speed = next_speed
        self.current_operation_time += operation_time
        self.current_redundant_operation_time = self._calc_redundant_operation_time()
        self.current_energy_consumption += energy_consumption
        # self.mass = self.vehicle.mass
        self.current_slope = get_slope_scalar_numba(
            self.current_position,
            self.track.slopes,
            self.track.slope_intervals,
        )
        self.current_stopping_point_index = self.sps.step_to_next_stopping_point(
            current_pos=self.current_position,
            current_speed=self.current_speed,
            current_time=self.current_operation_time,
            current_sp=self.current_stopping_point_index,
        )
        self.current_min_speed, self.current_max_speed = (
            self.safeguard_utility.get_min_and_max_speed(
                current_pos=self.current_position,
                current_sp=self.current_stopping_point_index,
            )
        )
        self.current_max_speed = min(
            self._get_upper_speed(self.current_position),
            self.current_max_speed,
        )
        # (
        #     self.current_latest_traction_intervention_point,
        #     self.current_latest_braking_intervention_point,
        # ) = (
        #     self.safeguard_utility
        #     .get_latest_traction_and_braking_intervention_points(
        #         current_speed=self.current_speed, current_sp=self.current_sp
        #     )
        # )

        self.stop_error = abs(
            self.train_service.target_position - self.current_position
        )
        is_stopped = math.isclose(self.current_speed, 0.0, abs_tol=0.01)
        success = is_stopped and self.stop_error <= 9.0
        is_speed_low_violation = self.current_speed < self.current_min_speed
        is_speed_high_violation = self.current_speed > self.current_max_speed
        is_step_limit_reached = self.current_steps > self.max_episode_steps
        is_failed_stop = is_stopped and not success

        # 仅成功任务视为正常终止。
        terminated = success
        # 失败停车与约束/步数超限统一视为截断结束。
        truncated = (
            is_speed_low_violation
            or is_speed_high_violation
            or is_step_limit_reached
            or is_failed_stop
        )
        self._last_violation_code = self._resolve_violation_code(
            terminated=terminated,
            failed_stop=is_failed_stop,
            speed_low_violation=is_speed_low_violation,
            speed_high_violation=is_speed_high_violation,
            step_limit_reached=is_step_limit_reached,
        )

        # 计算奖励
        reward = self._get_reward(terminated=terminated, truncated=truncated)

        if self.enable_trajectory_tracking:
            # 记录轨迹数据
            self._record_trajectory(
                pos=self.current_position,
                speed=self.current_speed,
            )

        # 获取可观测状态和信息
        observation = self._get_obs()

        self._collect_step_diagnostics = self._should_collect_step_diagnostics()
        self._record_basic_info()
        info = self._get_basic_info()
        if self.enable_diagnostics and self._collect_step_diagnostics:
            self._record_step_diagnostics()

            info["rewards"] = dict(self.rewards_info)
            info["state"] = dict(self.state_info)
            info["constraint"] = dict(self.constraint_info)
            info["event"] = dict(self.event_info)

        return (observation, reward, terminated, truncated, info)

    def _resolve_violation_code(
        self,
        *,
        terminated: bool,
        failed_stop: bool,
        speed_low_violation: bool,
        speed_high_violation: bool,
        step_limit_reached: bool,
    ) -> int:
        if terminated:
            return self.VIOLATION_CODE_ONGOING
        if failed_stop:
            return self.VIOLATION_CODE_FAILED_STOP
        if speed_low_violation:
            return self.VIOLATION_CODE_SPEED_LOW
        if speed_high_violation:
            return self.VIOLATION_CODE_SPEED_HIGH
        if step_limit_reached:
            return self.VIOLATION_CODE_STEP_LIMIT
        return self.VIOLATION_CODE_ONGOING

    def _update_motion(self) -> tuple[float, float, float]:
        """
        磁浮列车匀变速直线运动仿真

        Args:
            acc(float): 加速度(m/s^2)

        Returns:
            tuple: (next_speed, distance, operation_time)
        """

        distance = self.max_step_distance

        # 当加速度极小时，认为列车做匀速运动
        if abs(self.current_acc) < 1e-6:
            next_speed = self.current_speed
            if next_speed < 1e-6:
                distance = 0.0  # 无法运动
                operation_time = 0.0
                next_speed = 0.0
            else:
                operation_time = distance / next_speed

        else:
            # 一般情况下，列车做匀变速直线运动
            next_speed_squared = (
                self.current_speed**2 + 2 * self.current_acc * distance
            )  # 速度大小的平方

            if next_speed_squared < 1e-6:
                # 边界情况：减速为0
                next_speed = 0.0
                distance = -(self.current_speed**2) / (2 * self.current_acc)
            else:
                next_speed = np.sqrt(next_speed_squared)

            operation_time = (next_speed - self.current_speed) / self.current_acc

        return next_speed, distance, operation_time

    def _calc_ref_cum_time(self):
        """计算最短运行模式下, 到达每个参考位置的累计运行时间"""

        ds = np.abs(np.diff(self.upper_speed_profile_pos_arr))
        speed_avg = 0.5 * (
            self.upper_speed_profile_speed_arr[1:]
            + self.upper_speed_profile_speed_arr[:-1]
        )

        # 平均速度过小时将该段时间记为0
        dt = np.divide(
            ds,
            speed_avg,
            out=np.zeros_like(ds, dtype=np.float32),
            where=speed_avg > 1e-3,
        )

        ref_cum_time = np.empty_like(self.upper_speed_profile_pos_arr, dtype=np.float32)
        ref_cum_time[0] = 0.0
        ref_cum_time[1:] = np.cumsum(dt, dtype=np.float32)

        return ref_cum_time

    def _build_upper_speed_lookup_table(
        self,
        pos_arr: np.ndarray,
        speed_arr: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        """构建速度上限查找表，供 step 阶段通过数组访问快速获取上限速度。"""
        pos_raw = np.asarray(pos_arr, dtype=np.float64)
        speed_raw = np.asarray(speed_arr, dtype=np.float64)
        if pos_raw.size == 0:
            return 0.0, 1.0, np.zeros(1, dtype=np.float32)

        interp_pos = pos_raw
        interp_speed = speed_raw
        if interp_pos.size == 1:
            return (
                float(interp_pos[0]),
                1.0,
                np.array([max(0.0, float(interp_speed[0]))], dtype=np.float32),
            )

        # 兼容反向里程场景：统一转为升序位置进行预采样。
        if interp_pos[0] > interp_pos[-1]:
            interp_pos = interp_pos[::-1]
            interp_speed = interp_speed[::-1]

        # 预采样分辨率按 10m 固定，兼顾精度与查表成本。
        lut_step = 10.0
        pos_min = float(interp_pos[0])
        pos_max = float(interp_pos[-1])
        lut_size = int(np.ceil((pos_max - pos_min) / lut_step)) + 1
        lut_positions = pos_min + np.arange(lut_size, dtype=np.float64) * lut_step
        lut_speed = np.interp(
            lut_positions,
            interp_pos,
            interp_speed,
            left=float(interp_speed[0]),
            right=float(interp_speed[-1]),
        )

        return pos_min, lut_step, np.maximum(lut_speed, 0.0).astype(np.float32)

    def _get_ref_cum_time(self, pos: float | np.floating) -> float:
        """根据位置在最短运行参考曲线上插值累计运行时间。"""
        if self.upper_speed_profile_pos_arr.size == 0:
            return 0.0

        if self.upper_speed_profile_pos_arr[0] <= self.upper_speed_profile_pos_arr[-1]:
            interp_pos = self.upper_speed_profile_pos_arr
            interp_time = self.ref_curve_cum_time
        else:
            interp_pos = self.upper_speed_profile_pos_arr[::-1]
            interp_time = self.ref_curve_cum_time[::-1]

        return float(
            np.interp(
                float(pos),
                interp_pos,
                interp_time,
                left=float(interp_time[0]),
                right=float(interp_time[-1]),
            )
        )

    def _get_ref_remaining_operation_time(self, pos: float | np.floating) -> float:
        """根据位置估计沿最短运行参考曲线抵达终点的剩余时间。"""
        reference_cum_time = self._get_ref_cum_time(pos)
        return float(
            np.clip(
                self.ref_total_operation_time - reference_cum_time,
                0.0,
                self.ref_total_operation_time,
            )
        )

    def _calc_redundant_operation_time(self) -> float:
        # min_remaining = self._get_reference_remaining_operation_time(
        #     self.current_position
        # )
        min_remaining = self.ors.calc_min_operation_time(
            begin_pos=self.current_position,
            begin_speed=self.current_speed,
            end_pos=self.train_service.target_position,
            end_speed=0.0,
        )
        actual_remaining = (
            self.train_service.schedule_time - self.current_operation_time
        )
        return actual_remaining - min_remaining

    def _get_upper_speed(self, pos: float | np.floating):
        if self._upper_speed_lut_speed_arr.size == 0:
            return 0.0

        idx_float = (
            float(pos) - self._upper_speed_lut_pos_min
        ) / self._upper_speed_lut_step
        if idx_float <= 0.0:
            return float(self._upper_speed_lut_speed_arr[0])

        last_idx = self._upper_speed_lut_speed_arr.size - 1
        if idx_float >= float(last_idx):
            return float(self._upper_speed_lut_speed_arr[last_idx])

        idx0 = int(idx_float)
        frac = idx_float - idx0
        v0 = float(self._upper_speed_lut_speed_arr[idx0])
        v1 = float(self._upper_speed_lut_speed_arr[idx0 + 1])
        return v0 + (v1 - v0) * frac

    def _get_reward(
        self,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if not truncated:
            # 基本生存奖励
            basic_survival_reward = 100.0 / self.max_episode_steps
            if terminated:
                reward_total = self._get_reward_dense() + self._get_reward_goal()
            else:
                reward_total = self._get_reward_dense()

            reward_total += basic_survival_reward
        else:
            progress = (
                abs(self.current_position - self.train_service.target_position)
                / self.whole_distance
            )
            reward_total = -1.0 - 1.0 * (progress**2)

        if self.enable_diagnostics and self._collect_step_diagnostics:
            self.rewards_info["total"] = reward_total

        return reward_total

    def _get_reward_dense(
        self,
    ) -> float:
        # 安全奖励
        reward_safety = (
            self._get_reward_safety_dense()
            if self.reward_config.enable_potential_safety
            else 0.0
        )

        # 能耗奖励
        reward_energy = (
            self._get_reward_energy_dense() if self.reward_config.enable_energy else 0.0
        )

        # 舒适度奖励
        reward_comfort = (
            self._get_reward_comfort_dense()
            if self.reward_config.enable_comfort
            else 0.0
        )

        # 运行时间奖励
        reward_punctuality = (
            self._get_reward_punctuality_dense()
            if self.reward_config.enable_potential_punctuality
            else 0.0
        )

        # 停站奖励
        reward_stopping = (
            self._get_reward_stopping_dense()
            if self.reward_config.enable_potential_stopping
            else 0.0
        )

        if self.enable_diagnostics and self._collect_step_diagnostics:
            self.rewards_info["safety"] = reward_safety
            self.rewards_info["energy"] = reward_energy
            self.rewards_info["comfort"] = reward_comfort
            self.rewards_info["punctuality"] = reward_punctuality
            self.rewards_info["stopping"] = reward_stopping

        return (
            reward_safety
            + reward_energy
            + reward_comfort
            + reward_punctuality
            + reward_stopping
        )

    def _get_reward_safety_dense(self) -> float:
        # 计算当前状态势能

        # phi_curr = self._potential_safety_position(
        #     pos=self.current_pos,
        #     min_pos=self.current_latest_traction_intervention_point,
        #     max_pos=self.current_latest_braking_intervention_point,
        #     target_pos=self.sps.get_auxiliary_stopping_area_target_position(
        #         sp=self.current_sp
        #     )
        #     if self.current_sp >= 0
        #     else self.train_service.target_position,
        # )

        # phi_curr = self._potential_safety_speed_asymmetric_v2(
        #     pos=self.current_position,
        #     speed=self.current_speed,
        #     min_speed=self.current_min_speed,
        #     max_speed=self.current_max_speed,
        #     target_pos=self.sps.get_auxiliary_stopping_area_target_position(
        #         sp=self.current_stopping_point_index
        #     )
        #     if self.current_stopping_point_index >= 0
        #     else self.train_service.target_position,
        # )

        phi_curr = self._potential_safety_speed_asymmetric_v3(
            speed=self.current_speed,
            min_speed=self.current_min_speed,
            max_speed=self.current_max_speed,
        )

        # 计算上个状态势能

        # phi_prev = self._potential_safety_position(
        #     pos=self.last_state["pos"],
        #     min_pos=self.last_state["latest_traction_intervention_point"],
        #     max_pos=self.last_state["latest_braking_intervention_point"],
        #     # target_pos=self.sps.get_auxiliary_stopping_area_target_position(
        #     #     sp=self.last_state["stopping_point_index"]
        #     # )
        #     # if self.last_state["stopping_point_index"] >= 0
        #     # else self.task.target_position,
        #     target_pos=self.sps.get_auxiliary_stopping_area_target_position(
        #         sp=self.current_sp
        #     )
        #     if self.current_sp >= 0
        #     else self.train_service.target_position,
        # )

        # phi_prev = self._potential_safety_speed_asymmetric_v2(
        #     pos=self.last_state["pos"],
        #     speed=self.last_state["speed"],
        #     min_speed=self.last_state["min_speed"],
        #     max_speed=self.last_state["max_speed"],
        #     # target_pos=self.sps.get_auxiliary_stopping_area_target_position(
        #     #     sp=self.last_state["stopping_point_index"]
        #     # )
        #     # if self.last_state["stopping_point_index"] >= 0
        #     # else self.task.target_position,
        #     target_pos=self.sps.get_auxiliary_stopping_area_target_position(
        #         sp=self.current_stopping_point_index
        #     )
        #     if self.current_stopping_point_index >= 0
        #     else self.train_service.target_position,
        # )

        phi_prev = self._potential_safety_speed_asymmetric_v3(
            speed=self.last_state["speed"],
            min_speed=self.last_state["min_speed"],
            max_speed=self.last_state["max_speed"],
        )

        return self.gamma * phi_curr - phi_prev
        # return (
        #     phi_curr - phi_prev
        # )  # 轻微地偏离PBRS的最优性保证，但能换来更好的训练稳定性

    def _potential_safety_speed(
        self,
        pos: float,
        speed: float,
        min_speed: float,
        max_speed: float,
        target_pos: float,
    ) -> float:
        distance_to_target = abs(target_pos - pos)
        center_speed = (max_speed + min_speed) / 2.0
        safe_margin = max((max_speed - min_speed) / 2.0, 0.5)

        # 基础偏离惩罚(二次方项, 引导列车走中间)
        norm_speed_diff = (speed - center_speed) / safe_margin
        phi_base = 2.0 * np.log(1.01 - norm_speed_diff**2)

        # 靠近目标位置时，适当增大惩罚力度
        scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

        return scale * phi_base

    def _potential_safety_speed_adaptive(
        self,
        pos: float,
        speed: float,
        min_speed: float,
        max_speed: float,
        target_pos: float,
    ):
        distance_to_target = abs(target_pos - pos)

        speed_star = (max_speed + min_speed) / 2.0 if min_speed > 0.0 else 0.0
        safe_margin = (
            max((max_speed - min_speed) / 2.0, 0.5)
            if min_speed > 0.0
            else max(max_speed, 0.5)
        )

        norm_speed_diff = (speed - speed_star) / safe_margin
        phi_base = 2.0 * np.log(1.01 - norm_speed_diff**2)

        scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

        return scale * phi_base

    def _potential_safety_speed_asymmetric_v1(
        self,
        pos: float,
        speed: float,
        min_speed: float,
        max_speed: float,
        target_pos: float,
    ):
        distance_to_target = np.abs(target_pos - pos)

        # 设定一个危险缓冲距离 (m/s)，仅当距离边界小于该值时才触发惩罚
        upper_bound = 8.0
        lower_bound = 5.0

        # 1. 上限惩罚 (始终激活)
        phi_max = 0.0
        margin_max = max_speed - speed
        if margin_max < upper_bound:
            risk_max = 1.0 - margin_max / upper_bound
            phi_max = 2.0 * np.log(1.01 - risk_max**2)

        # 2. 下限惩罚 (条件激活：仅当存在实质性的最小速度约束时才惩罚)
        phi_min = 0.0
        margin_min = speed - min_speed

        # 当 min_speed 极小 (例如接近 0) 时，说明当前允许停车，直接关闭下限惩罚
        if min_speed > 0.0 and margin_min < lower_bound:
            risk_min = 1.0 - margin_min / lower_bound
            phi_min = 2.0 * np.log(1.01 - risk_min**2)
            # phi_min = 2.0 * np.log(1.01 - risk_min**2) * fade_factor

        # 距离缩放系数
        scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

        # 最终势能为两侧惩罚之和
        return scale * (phi_max + phi_min)

    def _potential_safety_speed_asymmetric_v2(
        self,
        pos: float,
        speed: float,
        min_speed: float,
        max_speed: float,
        target_pos: float,
    ):
        distance_to_target = np.abs(target_pos - pos)

        # 设定一个危险缓冲距离 (m/s)，仅当距离边界小于该值时才触发惩罚
        upper_bound = 5.0
        lower_bound = 5.0

        # 1. 上限惩罚 (始终激活)
        margin_max = max_speed - speed
        norm_margin_max = max(1.0 - margin_max / upper_bound, 0.0)
        phi_max = -(norm_margin_max**2)

        # 2. 下限惩罚 (条件激活：仅当存在实质性的最小速度约束时才惩罚)
        if min_speed > 0.0:
            margin_min = speed - min_speed
            norm_margin_min = max(1.0 - margin_min / lower_bound, 0.0)
            phi_min = -(norm_margin_min**2)
        else:
            phi_min = 0

        # 距离缩放系数
        scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

        # 最终势能为两侧惩罚之和
        return scale * (phi_max + phi_min)

    def _potential_safety_speed_asymmetric_v3(
        self,
        speed: float,
        min_speed: float,
        max_speed: float,
    ) -> float:
        K_Safety = 1.0
        speed_band = max_speed - min_speed + 0.1
        upper_bound = max(speed_band * 0.1, 2.0)
        lower_bound = speed_band * 0.1
        alpha = 3.0

        margin_upper = max_speed - speed
        x_upper = 1.0 - margin_upper / upper_bound
        z_upper = math.log1p(math.exp(alpha * x_upper)) / alpha
        phi_upper = -(z_upper**2)

        if min_speed > 0.0:
            margin_lower = speed - min_speed
            x_lower = 1.0 - margin_lower / lower_bound
            z_lower = math.log1p(math.exp(alpha * x_lower)) / alpha
            phi_lower = -(z_lower**2)
        else:
            phi_lower = 0.0

        return K_Safety * (phi_upper + phi_lower)

    def _potential_safety_position(
        self, pos: float, min_pos: float, max_pos: float, target_pos: float
    ):
        distance_to_target = abs(target_pos - pos)

        center_pos = (max_pos + min_pos) / 2.0
        safe_margin = (max_pos - min_pos) / 2.0

        safe_margin = max(safe_margin, 1e-3)

        norm_pos_diff = (pos - center_pos) / safe_margin
        phi_base = 2.0 * np.log(1.1 - norm_pos_diff**2)

        scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

        return scale * phi_base

    def _get_reward_energy_dense(self) -> float:

        val = (
            -15.0
            * (self.current_energy_consumption - self.last_state["energy_consumption"])
            / self.max_energy_consumption
        )

        return val

    def _get_reward_comfort_dense(self) -> float:
        delta_acc = abs(self.last_state["acc"] - self.current_acc)
        norm_jerk = delta_acc / (self.train_service.max_acc_change)

        val = -20.0 / self.max_episode_steps * norm_jerk**2

        return val

    def _get_reward_punctuality_dense(self) -> float:

        # phi_curr = self._potential_punctuality_v18(
        #     pos=self.current_position,
        #     redundant_operation_time=self.current_redundant_operation_time,
        # )

        # phi_prev = self._potential_punctuality_v18(
        #     pos=self.last_state["pos"],
        #     redundant_operation_time=self.last_state["redundant_operation_time"],
        # )

        phi_curr = self._potential_punctuality_v39(
            pos=self.current_position,
            speed=self.current_speed,
        )

        phi_prev = self._potential_punctuality_v39(
            pos=self.last_state["pos"],
            speed=self.last_state["speed"],
        )

        # phi_curr = self._potential_punctuality_v35(
        #     pos=self.current_position,
        #     redundant_operation_time=self.current_redundant_operation_time,
        # )

        # phi_prev = self._potential_punctuality_v35(
        #     pos=self.last_state["pos"],
        #     redundant_operation_time=self.last_state["redundant_operation_time"],
        # )

        # phi_curr = self._potential_punctuality_v36(
        #     redundant_operation_time=self.current_redundant_operation_time,
        # )

        # phi_prev = self._potential_punctuality_v36(
        #     redundant_operation_time=self.last_state["redundant_operation_time"],
        # )

        # phi_curr = self._potential_punctuality_v37(
        #     operation_time=self.current_operation_time,
        #     redundant_operation_time=self.current_redundant_operation_time,
        # )

        # phi_prev = self._potential_punctuality_v37(
        #     operation_time=self.last_state["operation_time"],
        #     redundant_operation_time=self.last_state["redundant_operation_time"],
        # )

        # phi_curr = self._potential_punctuality_v38(
        #     redundant_operation_time=self.current_redundant_operation_time
        # )

        # phi_prev = self._potential_punctuality_v38(
        #     redundant_operation_time=self.last_state["redundant_operation_time"]
        # )

        return self.gamma * phi_curr - phi_prev
        # return (
        #     phi_curr - phi_prev
        # )  # 轻微地偏离PBRS的最优性保证，但能换来更好的训练稳定性

    def _potential_punctuality_v1(self, redundant_operation_time: float):
        """势能值随冗余时间的减小而减小，当冗余时间为负值时，减小速率极快。"""
        return -1.0 * np.log1p(np.exp(-1.0 * redundant_operation_time / 100.0))

    def _potential_punctuality_v2(self, redundant_operation_time: float):
        """
        势能值随冗余时间的减小而减小，当冗余时间为正值时，呈线性递减趋势；
        当冗余时间为负值时，呈抛物线性递减趋势。
        """
        K_early = 0.01
        K_late = 0.001
        K_base = 10.0

        if redundant_operation_time > 0.0:
            return K_early * redundant_operation_time + K_base
        else:
            return (
                K_early * redundant_operation_time
                - K_late * redundant_operation_time**2
                + K_base
            )

    def _potential_punctuality_v3(self, redundant_operation_time: float):
        """
        势能值随冗余时间的减小而减小，当冗余时间为正值时，呈线性递减趋势；
        当冗余时间为负值时，呈指数型递减趋势。另外，为防止指数项数值爆炸，
        对冗余时间做了截断处理。
        """
        K_base = 1.0
        K_safe = 1.0
        K_late = 10.0
        alpha = 5.0

        time_redundancy_clipped = float(
            np.clip(
                redundant_operation_time / self.train_service.schedule_time, -1.0, 1.0
            )
        )
        if time_redundancy_clipped >= 0.0:
            return K_base + K_safe * time_redundancy_clipped
        else:
            return (
                K_base
                + K_safe * time_redundancy_clipped
                - K_late
                / alpha
                * (
                    np.exp(-alpha * time_redundancy_clipped)
                    + alpha * time_redundancy_clipped
                    - 1
                )
            )

    def _potential_punctuality_v4(self, redundant_operation_time: float):
        """准点势能：预计准点到达时最大，预计晚点时快速下降。"""

        K_peak = 4.0
        K_early = 4.0
        K_late = 20.0
        alpha_late = 8.0

        time_redundancy_clipped = float(
            np.clip(
                redundant_operation_time / self.train_service.schedule_time, -1.0, 1.0
            )
        )
        if time_redundancy_clipped >= 0.0:
            return K_peak - K_early * time_redundancy_clipped**2

        late_error_ratio = -time_redundancy_clipped
        return K_peak - K_late / alpha_late * (
            np.exp(alpha_late * late_error_ratio) - 1.0
        )

    def _potential_punctuality_v5(
        self, operation_time: float, redundant_operation_time: float
    ):
        """势函数在剩余运行时间和冗余时间为0时取最大值，否则，势函数值呈指数型下降。"""
        K_T = 20.0
        sigma_tau_early = 300.0
        sigma_tau_late = 180.0
        sigma_rho_early = 240.0
        sigma_rho_late = 60.0

        remaining_schedule_time = self.train_service.schedule_time - operation_time

        if remaining_schedule_time > 0.0:
            e_time = np.exp(-((remaining_schedule_time / sigma_tau_early) ** 2))
        else:
            e_time = np.exp(-((remaining_schedule_time / sigma_tau_late) ** 2))

        if redundant_operation_time > 0.0:
            e_redundancy = np.exp(-((redundant_operation_time / sigma_rho_early) ** 2))
        else:
            e_redundancy = np.exp(-((redundant_operation_time / sigma_rho_late) ** 2))

        return K_T * (e_time * e_redundancy)

    def _potential_punctuality_v6(
        self, operation_time: float, redundant_operation_time: float
    ):
        """
        当剩余运行时间和冗余运行时间大于0时，势能值最大；
        一旦两者任何一个小于0，则势能随着时间的减小平滑下降到0
        """
        K_T = 10.0
        sigma_tau = 10.0
        sigma_rho = 100.0

        remaining_schedule_time = self.train_service.schedule_time - operation_time

        if remaining_schedule_time > 0.0:
            e_time = 1.0
        else:
            e_time = np.exp(-((remaining_schedule_time / sigma_tau) ** 2))

        if redundant_operation_time > 0.0:
            e_redundancy = 1.0
        else:
            e_redundancy = np.exp(-((redundant_operation_time / sigma_rho) ** 2))

        return K_T * (e_time * e_redundancy)

    def _potential_punctuality_v7(
        self, operation_time: float, redundant_operation_time: float
    ):
        K_T = 10.0
        early_time_lambda = 0.1
        early_redundancy_lambda = 0.1
        late_time_sigma = 30.0
        late_redundancy_sigma = 20.0
        schedule_time = self.train_service.schedule_time

        e_time = (
            (1 - np.exp(-early_time_lambda * (operation_time / schedule_time)))
            / (1 - np.exp(-early_time_lambda))
            if operation_time < schedule_time
            else np.exp(-(((operation_time - schedule_time) / late_time_sigma) ** 2))
        )

        e_redundancy = (
            (
                1
                - np.exp(
                    -early_redundancy_lambda
                    * (schedule_time - redundant_operation_time)
                    / schedule_time
                )
            )
            / (1 - np.exp(-early_redundancy_lambda))
            if redundant_operation_time > 0.0
            else np.exp(-((redundant_operation_time / late_redundancy_sigma) ** 2))
        )

        return K_T * (e_time * e_redundancy)

    def _potential_punctuality_v8(
        self, operation_time: float, redundant_operation_time: float
    ):
        K_T = 5.0
        gamma_t = 0.1
        gamma_r = 0.1
        sigma_t = 100.0
        sigma_r = 80.0
        schedule_time = self.train_service.schedule_time

        e_time = (
            1.0 + gamma_t * (1.0 - operation_time / schedule_time) ** 2
            if operation_time < schedule_time
            else np.exp(-(((operation_time - schedule_time) / sigma_t) ** 2))
        )

        e_redundancy = (
            1.0 + gamma_r * (redundant_operation_time / schedule_time) ** 2
            if redundant_operation_time > 0.0
            else np.exp(-((redundant_operation_time / sigma_r) ** 2))
        )

        return K_T * e_time * e_redundancy

    def _potential_punctuality_v9(self, redundant_operation_time: float):
        """
        势能值随着冗余时间的减小而减小，当冗余时间为正时，势能值下降速率随冗余时间的减小而减小；
        当冗余时间为负时，势能值下降速率随冗余时间的减小而增大。
        """
        K_T = 10.0
        gamma = 0.1
        omega = 1.0

        ratio = redundant_operation_time / self.max_redundant_operation_time

        if ratio > 0.0:
            e_redundancy = gamma * (ratio**2)
        else:
            e_redundancy = -omega * (ratio**2)

        return K_T * e_redundancy

    def _potential_punctuality_v10(
        self, redundant_operation_time: float, operation_time: float
    ):
        """
        变化趋势与v9类似，但在冗余时间为负侧添加了随运行时间变化的势能缩放因子。
        目的是为了防止势能值下降幅度过小，差分奖励反而为正值的情形。
        随机数种子为19937时性能表现较好
        """
        K_T = 15.0
        gamma = 1.0  # 只加不减 1.0
        omega = 15.0
        alpha = 4.0
        margin = 2.0

        schedule_time = self.train_service.schedule_time
        ratio = (redundant_operation_time - margin) / schedule_time
        progress = operation_time / schedule_time

        e_redundancy = (
            gamma * (ratio**2)
            if ratio > 0.0
            else -omega * (ratio**2) * (1.0 + alpha * (progress**2))
        )

        return K_T * e_redundancy

    def _potential_punctuality_v11(
        self, redundant_operation_time: float, operation_time: float
    ):
        """势能值随剩余运行时间的减小而减小，增益随冗余时间的减小而减小"""
        K_T = 5.0
        lambda_plus = 0.5
        lambda_minus = 0.8
        scale = 0.08
        tau = (
            self.train_service.schedule_time - operation_time
        ) / self.train_service.schedule_time

        rho = redundant_operation_time / self.train_service.schedule_time

        def _sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        sig_norm = _sigmoid(-rho / scale)

        e_time = (
            (1.0 + lambda_plus * sig_norm) * tau**2
            if tau > 0.0
            else -(1.0 + lambda_minus * sig_norm) * tau**2
        )

        return K_T * e_time

    def _potential_punctuality_v12(
        self, redundant_operation_time: float, operation_time: float
    ):
        """
        势能值在冗余时间为0时最大，其他情况下随绝对值的增大而减小，
        靠近目标位置时，势能值归零
        """
        K_T = 8.0
        min_stage_weight = 0.1
        sigma_early = 0.14
        sigma_late = 0.06
        tail_smooth = 1.0e-6

        progress = operation_time / self.train_service.schedule_time
        rho = redundant_operation_time / self.train_service.schedule_time

        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        stage_weight = min_stage_weight + (1.0 - min_stage_weight) * smooth_progress

        sigma = sigma_early if rho >= 0.0 else sigma_late
        normalized_error = rho / sigma
        pseudo_huber_error = np.sqrt(normalized_error**2 + tail_smooth) - np.sqrt(
            tail_smooth
        )
        punctuality_peak = np.exp(-pseudo_huber_error)

        return K_T * stage_weight * punctuality_peak

    def _potential_punctuality_v13(
        self,
        pos: float,
        speed: float,
        operation_time: float,
    ):
        """
        e_ETA > 0 时，说明按照当前速度开下去，到终点会晚点
        e_ETA < 0 时，说明按照当前速度开下去，到终点会早到
        e_ETA = 0 时，说明按照当前速度是完美的“准点巡航速度”
        """
        K_T = 1.0
        epsilon = 0.5
        sigma_early = 20.0
        sigma_late = 10.0
        e_ETA = (
            operation_time
            + (self.train_service.target_position - pos) / (speed + epsilon)
            - self.train_service.schedule_time
        )

        sigma = sigma_early if e_ETA <= 0.0 else sigma_late

        return K_T * np.exp(-((e_ETA / sigma) ** 2))

    def _potential_punctuality_v14(
        self, redundant_operation_time: float, operation_time: float
    ):
        K_T = 8.0
        alpha = 0.1
        gamma_pot = 0.2
        omega = 80.0

        schedule_time = self.train_service.schedule_time

        # 物理可行域校验
        if operation_time + redundant_operation_time > schedule_time + 1e-3:
            # 违反则回归到零势能面
            return 0.0

        # 时空状态变量归一化解耦
        progress = operation_time / schedule_time
        ratio = redundant_operation_time / schedule_time

        # 计算时光流逝带来的基础线性势能衰减基底
        time_consumption_base = 1.0 - alpha * progress

        # 分段冗余协同流形
        e_redundancy = (
            1.0 + gamma_pot * (ratio**2)
            if ratio > 0.0
            else 1.0 / (1.0 + omega * (ratio**2))
        )

        return K_T * time_consumption_base * e_redundancy

    def _potential_punctuality_v15(
        self, redundant_operation_time: float, operation_time: float
    ):
        K_T = 10.0
        omega_pos = 5.0
        omega_neg = 2.0
        T_ref = 100.0
        margin = 2.0

        progress = operation_time / self.train_service.schedule_time
        ratio = (redundant_operation_time - margin) / T_ref

        e_redundancy = (
            omega_pos * (ratio**2) * (1.0 - progress)
            if ratio > 0.0
            else -omega_neg * np.log1p(ratio**2) * (1.0 + progress)
        )

        return K_T * e_redundancy

    def _potential_punctuality_v16(
        self,
        pos: float,
        redundant_operation_time: float,
    ):
        omega_pos = 0.05
        omega_neg = 0.1
        T_scale_pos = 10.0
        T_scale_neg = 5.0
        margin = 0.0

        # 控制Log-Cosh从原点过渡到线性区的速度
        # 越小过渡越平缓
        alpha = 1.0

        # dist_to_target = abs(self.train_service.target_position - position)
        # fade_factor = 1.0 - math.exp(-((dist_to_target / 600.0) ** 2))

        if redundant_operation_time - margin > 0.0:
            ratio = (redundant_operation_time - margin) / T_scale_pos
            phi = omega_pos * (ratio**2)
        else:
            ratio = (redundant_operation_time - margin) / T_scale_neg
            abs_ratio = -alpha * ratio
            phi = -(omega_neg / alpha) * (
                abs_ratio + math.log1p(math.exp(-2.0 * abs_ratio)) - math.log(2.0)
            )
            # phi = -omega_neg * np.log1p(-ratio)

        # return phi * fade_factor
        return phi

    def _potential_punctuality_v17(
        self, redundant_operation_time: float, operation_time: float
    ):
        omega_pos = 15.0
        omega_neg = 4.5
        phi_zero = 10.0
        T_ref = 100.0
        margin = 2.0

        progress = operation_time / self.train_service.schedule_time
        ratio = (redundant_operation_time - margin) / T_ref
        if ratio > 0.0:
            phi = phi_zero + omega_pos * ratio * (1.0 - progress)
        else:
            phi = phi_zero * np.exp(omega_neg * ratio * (1.0 + progress))

        return phi

    def _potential_punctuality_v18(self, pos: float, redundant_operation_time: float):
        K_T = 0.01

        bias = redundant_operation_time - self._get_ref_redundant_operation_time(pos)

        delta = 1.0
        x_dz = max(0.0, abs(bias) - delta)

        phi = 0.0
        if x_dz > 0.0:
            phi = -K_T * x_dz

        # phi = -K_T * abs(bias)

        return phi

    def _potential_punctuality_v19(self, pos: float, redundant_operation_time: float):
        K_T = 1.0
        T_scale = 5.0

        bias = redundant_operation_time - self._get_ref_redundant_operation_time(pos)
        ratio = bias / T_scale

        phi = K_T * math.exp(-(ratio**2))
        progress = (
            1.0 - abs(self.train_service.target_position - pos) / self.whole_distance
        )
        alpha = 0.5

        return phi * (1.0 + alpha * progress)

    def _potential_punctuality_v20(self, pos: float, redundant_operation_time: float):
        K_T = 0.1
        progress = (
            1.0 - abs(self.train_service.target_position - pos) / self.whole_distance
        )

        T_scale = 10.0 + (2.0 - 10.0) * progress

        # T_scale_acc = 6.0
        # T_scale_cruise = 4.0
        # T_scale_brake = 12.0
        # T_scale = (
        #     T_scale_cruise
        #     + (T_scale_acc - T_scale_cruise) * ((1 - progress) ** 4)
        #     + (T_scale_brake - T_scale_cruise) * (progress**4)
        # )

        bias = redundant_operation_time - self._get_ref_redundant_operation_time(pos)
        ratio = bias / T_scale

        # ln(cosh(x))
        abs_ratio = abs(ratio)

        phi = -K_T * (
            abs_ratio + math.log1p(math.exp(-2.0 * abs_ratio)) - math.log(2.0)
        )

        return phi

    def _potential_punctuality_v21(self, pos: float, redundant_operation_time: float):
        K_T = 1.0
        T_scale = 5.0

        bias = redundant_operation_time - self._get_ref_redundant_operation_time(pos)
        ratio = bias / T_scale
        ratio_sq = ratio**2

        phi = -K_T * math.tanh(ratio_sq)

        return phi

    def _smoothstep(self, edge0: float, edge1: float, x: float) -> float:
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    def _potential_punctuality_v22(self, pos: float, redundant_operation_time: float):
        K_T = 0.02

        dist_to_target = abs(self.train_service.target_position - pos)
        fade_factor = 1.0 - math.exp(-((dist_to_target / 300.0) ** 2))

        # d_start = pos - self.train_service.start_position
        # d_target = self.train_service.target_position - pos
        # d_buffer = 5000.0

        # factor_start = self._smoothstep(0.0, d_buffer, d_start)
        # factor_target = self._smoothstep(0.0, d_buffer, d_target)

        # K_T = K_T_max * factor_start * factor_target

        if redundant_operation_time >= 0.0:
            return K_T * redundant_operation_time * fade_factor
        else:
            return (
                K_T
                * (redundant_operation_time - 0.05 * (redundant_operation_time**2))
                * fade_factor
            )

    def _potential_punctuality_v23(
        self, operation_time: float, redundant_operation_time: float
    ) -> float:
        # K_T = 0.2
        # T_scale = 10.0

        # alpha = 0.4
        # beta = 1.0

        # r = redundant_operation_time / T_scale
        # tau = operation_time / self.train_service.schedule_time

        # if r > 0.0:
        #     g_t_r = ((1.0 + tau) ** 2) * (r / (1 + alpha * r))
        # else:
        #     g_t_r = ((1.0 + tau) ** 2) * (r / (1 - beta * r))

        # return K_T * g_t_r

        K_T = 2.0
        T_scale = 10.0
        r = redundant_operation_time / T_scale
        tau = operation_time / self.train_service.schedule_time
        g_t_r = math.tanh(r) * tau

        return K_T * g_t_r

    def _potential_punctuality_v24(
        self, operation_time: float, redundant_operation_time: float
    ) -> float:
        phi_mid = 2.0
        K_T = 1.0
        T_scale = 10.0
        r_max = self.train_service.schedule_time - self.min_operation_time

        tau = operation_time / self.train_service.schedule_time

        r_ref = r_max * (1.0 - tau)

        r_error = redundant_operation_time - r_ref
        phi = phi_mid + K_T * math.tanh(r_error / T_scale)

        return phi

    def _potential_punctuality_v25(
        self, operation_time: float, redundant_operation_time: float
    ) -> float:
        K_T = 1.0
        T_scale = 10.0

        alpha = 0.14
        beta = 0.005

        r = redundant_operation_time / T_scale
        tau = operation_time / self.train_service.schedule_time

        if r > 0.0:
            phi = K_T * math.exp(-alpha * (r**2))
        else:
            time_scaler = (1.0 + tau) ** 2
            penalty = 0.5 * (r**2) + r + 1.0 - math.exp(r)
            phi = K_T - beta * penalty * time_scaler

        return phi

    def _potential_punctuality_v26(
        self, speed: float, redundant_operation_time: float
    ) -> float:
        K_T = 1.0
        T_scale = 10.0
        alpha = 0.14
        lambda_leak = 0.015
        V_base = 10.0

        r = redundant_operation_time / T_scale

        if r > 0.0:
            phi = K_T * math.exp(-alpha * (r**2))
        else:
            v_gate = math.tanh(speed / V_base)
            phi = K_T * (1.0 + math.tanh(r)) + (lambda_leak * r * v_gate)

        return phi

    def _potential_punctuality_v27(
        self, pos: float, redundant_operation_time: float
    ) -> float:
        K_T = 1.0
        T_scale = 10.0

        f_x = pos / self.train_service.target_position

        g_r = math.asinh(redundant_operation_time / T_scale)

        return K_T * f_x * g_r

    def _potential_punctuality_v28(
        self, pos: float, redundant_operation_time: float
    ) -> float:
        K_reward = 0.5
        T_scale = 5.0

        K_penalty_early = 0.1
        K_penalty_late = 0.2

        eps_floor = 0.05

        bias = redundant_operation_time - self._get_ref_redundant_operation_time(pos)
        ratio = bias / T_scale

        phi_pos = K_reward * math.exp(-(ratio**2))

        penalty = math.log(math.cosh(ratio))
        K_penalty = K_penalty_early if ratio > 0.0 else K_penalty_late
        phi_neg = -K_penalty * penalty

        phi_core = phi_pos + phi_neg

        dist_to_target = abs(self.train_service.target_position - pos)

        t_space = max(0.0, min(1.0, dist_to_target / 1000.0))
        smooth_decay = t_space * t_space * (3.0 - 2.0 * t_space)

        K_x = eps_floor + (1.0 - eps_floor) * smooth_decay

        return phi_core * K_x

    def _potential_punctuality_v29(
        self, pos: float, speed: float, redundant_operation_time: float
    ) -> float:
        K_T = 1.0
        T_scale = 10.0

        bias = redundant_operation_time - self._get_ref_redundant_operation_time(
            pos=pos
        )
        ratio = bias / T_scale

        phi = K_T * math.log1p(math.exp(-abs(ratio) * (speed / self.vehicle.max_speed)))

        return phi

    def _potential_punctuality_v30(
        self, pos: float, speed: float, redundant_operation_time: float
    ) -> float:
        K_T = 1.0

        d = abs(self.train_service.target_position - pos) / self.whole_distance

        progress = 1.0 - d

        r = redundant_operation_time / self.max_redundant_operation_time
        v = speed / self.vehicle.max_speed

        if r > 0.0:
            g_r_v_x = r * v
        else:
            g_r_v_x = r * (1.0 - v)

        return K_T * g_r_v_x / (1.2 - progress)

    def _potential_punctuality_v31(
        self, pos: float, speed: float, redundant_operation_time: float
    ) -> float:
        K_track = 0.5
        K_late_cliff = 2.0
        T_scale = 5.0
        V_scale = 20.0

        bias = redundant_operation_time - self._get_ref_redundant_operation_time(pos)
        ratio = bias / T_scale

        if ratio > 0.0:
            phi_core = K_track * (math.exp(-(ratio**2)) - 1.0)
        else:
            phi_core = -K_late_cliff * math.log1p(abs(ratio))

        sigma_speed = math.tanh(speed / V_scale)

        return phi_core * sigma_speed

    def _potential_punctuality_v32(
        self, pos: float, speed: float, operation_time: float
    ) -> float:
        K_T = 0.01

        delta_s = abs(self.train_service.target_position - pos)
        delta_t = max(0.0, self.train_service.schedule_time - operation_time)

        delta_t_reg = delta_t + 1.5

        v_req = delta_s / delta_t_reg
        # v_req = min(self.vehicle.max_speed, delta_s / delta_t_reg)

        v_norm = speed / self.vehicle.max_speed
        v_req_norm = v_req / self.vehicle.max_speed

        # J_star_norm = (4.0 / delta_t_reg) * (
        #     v_norm**2 - 3.0 * v_norm * v_req_norm + 3.0 * (v_req_norm**2)
        # )

        J_star_norm = 1.0 * (
            v_norm**2 - 3.0 * v_norm * v_req_norm + 3.0 * (v_req_norm**2)
        )

        return -K_T * J_star_norm

    def _potential_punctuality_v33(
        self, pos: float, redundant_operation_time: float
    ) -> float:
        omega_pos = 0.1
        omega_neg = -0.001

        dist_to_target = self.train_service.target_position - pos

        progress = min(1.0, 1.0 - dist_to_target / self.whole_distance)

        r_exp = self.max_redundant_operation_time * (1.0 - progress)
        e_r = redundant_operation_time - r_exp

        if e_r > 0.0:
            phi = omega_pos * math.log1p(e_r)
        else:
            phi = omega_neg * (e_r**2)

        return phi

    def _potential_punctuality_v34(
        self, pos: float, redundant_operation_time: float
    ) -> float:
        K_T = 0.10

        dist_to_target = self.train_service.target_position - pos

        progress = min(1.0, 1.0 - dist_to_target / self.whole_distance)

        r_exp = self.max_redundant_operation_time * (1.0 - progress)
        e_r = redundant_operation_time - r_exp

        phi = -K_T * abs(e_r)

        return phi

    def _potential_punctuality_v35(
        self, pos: float, redundant_operation_time: float
    ) -> float:
        omega_pos = 0.05
        omega_neg = -0.01

        dist_to_target = self.train_service.target_position - pos

        progress = min(1.0, 1.0 - dist_to_target / self.whole_distance)

        r_exp = self.max_redundant_operation_time * (1.0 - progress)
        e_r = redundant_operation_time - r_exp

        if e_r > 0.0:
            phi = omega_pos * e_r
        else:
            phi = omega_neg * (abs(e_r) ** 1.5)

        return phi

    def _potential_punctuality_v36(self, redundant_operation_time: float) -> float:
        K_T = 10.0
        alpha = 0.1

        if redundant_operation_time > 0.0:
            phi_base = math.cos(
                (math.pi / 2.0)
                * (redundant_operation_time / self.max_redundant_operation_time)
            )
        else:
            phi_base = 1.0 / (1.0 + alpha * (redundant_operation_time**2))

        return K_T * phi_base

    def _potential_punctuality_v37(
        self, operation_time: float, redundant_operation_time: float
    ) -> float:
        K_T = 1.0
        epsilon = 1.0

        remaining_operation_time = self.train_service.schedule_time - operation_time
        min_operation_time = remaining_operation_time - redundant_operation_time

        theta = (remaining_operation_time + epsilon) / (min_operation_time + epsilon)

        phi = K_T * (theta - 1)

        return phi

    def _potential_punctuality_v38(self, redundant_operation_time: float) -> float:
        K_T = 0.001

        if redundant_operation_time > 0.0:
            phi = 0.0
        else:
            phi = -K_T * (redundant_operation_time**2)

        return phi

    def _potential_punctuality_v39(self, pos: float, speed: float) -> float:
        """基于 DP 速度曲线跟踪误差的准点势能。

        在当前位置插值得到动态规划参考速度，速度越接近参考曲线，
        势能越高（最大值为 0）。
        """
        K_V = 1.0
        sigma_v = 10.0

        reference_speed = self._get_ref_dp_speed(pos)
        speed_error = speed - reference_speed
        return -K_V * (speed_error / sigma_v) ** 2

    def _get_reward_stopping_dense(self):

        phi_curr = self._potential_stopping_v1(
            pos=self.current_position, speed=self.current_speed
        )

        phi_prev = self._potential_stopping_v1(
            pos=self.last_state["pos"], speed=self.last_state["speed"]
        )

        # phi_curr = self._potential_stopping_v2(
        #     pos=self.current_pos, speed=self.current_speed
        # )

        # phi_prev = self._potential_stopping_v2(
        #     pos=self.last_state["pos"], speed=self.last_state["speed"]
        # )

        # phi_curr = self._potential_stopping_v3(
        #     pos=self.current_position, speed=self.current_speed
        # )

        # phi_prev = self._potential_stopping_v3(
        #     pos=self.last_state["pos"], speed=self.last_state["speed"]
        # )

        return self.gamma * phi_curr - phi_prev
        # return (
        #     phi_curr - phi_prev
        # )  # 轻微地偏离PBRS的最优性保证，但能换来更好的训练稳定性

    def _potential_stopping_v1(self, pos: float, speed: float):
        K_Stopping = 10.0
        sigma_d = 0.1 * self.target_attraction_domain_radius
        sigma_v = 0.2 * self.vehicle.max_speed

        dist_error_abs = abs(pos - self.train_service.target_position)

        if dist_error_abs > self.target_attraction_domain_radius:
            return 0.0
        else:
            gaussian_exp = math.exp(-dist_error_abs / sigma_d - speed / sigma_v)

        return K_Stopping * gaussian_exp

    def _potential_stopping_v2(self, pos: float, speed: float):
        dist_error_abs = abs(self.train_service.target_position - pos)

        d_hat = dist_error_abs / self.target_attraction_domain_radius
        v_hat = speed / self.vehicle.max_speed

        phi_pos = math.exp(-(d_hat**2) / (2 * 0.1**2))
        phi_speed = math.exp(-(v_hat**2) / (2 * 0.1**2))

        return 20.0 * phi_pos * phi_speed

    def _potential_stopping_v3(self, pos: float, speed: float) -> float:
        K_Stopping = 10.0
        sigma_d = 300.0
        sigma_v = 0.2 * self.vehicle.max_speed

        dist_error = pos - self.train_service.target_position

        if abs(dist_error) > self.target_attraction_domain_radius:
            return 0.0
        else:
            gaussian_exp = (
                -((dist_error / sigma_d) ** 2) / 2.0 - ((speed / sigma_v) ** 2) / 2.0
            )

            return K_Stopping * math.exp(gaussian_exp)

    def _get_reward_goal(
        self,
    ) -> float:
        _stopping = self._calc_stopping_score()
        _punctuality = self._calc_punctuality_score()
        reward_stopping = _stopping * 15.0
        reward_punctuality = _punctuality * 5.0 + (_stopping**2) * _punctuality * 20.0

        if self.enable_diagnostics and self._collect_step_diagnostics:
            self.rewards_info["stopping"] = reward_stopping
            self.rewards_info["punctuality"] = reward_punctuality

        return reward_stopping + reward_punctuality

    def _calc_stopping_score(self) -> float:
        # return self._stopping_score_func(self.stop_error)
        beta = 0.8
        delta = max(0.0, abs(self.stop_error) - self.train_service.max_stop_error)
        return 1.0 / (1.0 + (delta / beta) ** 2)

    def _calc_punctuality_score(self) -> float:
        # return self._punctuality_score_func(
        #     abs(self.train_service.schedule_time - self.current_operation_time)
        # )
        time_error = abs(self.train_service.schedule_time - self.current_operation_time)

        score = math.exp(-time_error / 30.0)

        return score

    def _gaussian_kernel(self, A: float, B: float, k: float, x: float) -> float:
        return A * np.exp(-k * x) + B

    def _record_history(self) -> None:
        self.last_state["pos"] = self.current_position
        self.last_state["speed"] = self.current_speed
        self.last_state["acc"] = self.current_acc
        self.last_state["min_speed"] = self.current_min_speed
        self.last_state["max_speed"] = self.current_max_speed
        # self.last_state["latest_traction_intervention_point"] = (
        #     self.current_latest_traction_intervention_point
        # )
        # self.last_state["latest_braking_intervention_point"] = (
        #     self.current_latest_braking_intervention_point
        # )
        self.last_state["operation_time"] = self.current_operation_time
        self.last_state["redundant_operation_time"] = (
            self.current_redundant_operation_time
        )
        self.last_state["energy_consumption"] = self.current_energy_consumption
        self.last_state["stopping_point_index"] = self.current_stopping_point_index

    def _reset_history(self) -> None:
        self.last_state: TrainState = {
            "pos": 0.0,
            "speed": 0.0,
            "acc": 0.0,
            "min_speed": 0.0,
            "max_speed": 0.0,
            # "latest_traction_intervention_point": 0.0,
            # "latest_braking_intervention_point": 0.0,
            "operation_time": 0.0,
            "redundant_operation_time": 0.0,
            "energy_consumption": 0.0,
            "stopping_point_index": -1,
        }

    def _record_trajectory(
        self,
        pos: float,
        speed: float,
    ):
        """记录离散轨迹点。"""
        if not self.enable_trajectory_tracking:
            return

        assert self.trajectory_pos is not None
        assert self.trajectory_speed_mps is not None
        self.trajectory_pos.append(float(pos))
        self.trajectory_speed_mps.append(float(speed))

    def _reset_trajectory(self):
        """重置轨迹历史数据并记录初始状态"""
        if not self.enable_trajectory_tracking:
            self.trajectory_pos = None
            self.trajectory_speed_mps = None
            return

        self.trajectory_pos = [self.train_service.start_position]
        self.trajectory_speed_mps = [abs(self.train_service.start_speed)]

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render_mode."
                "You can specify the render_mode at initialization."
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        if self.fig is None:
            self._setup_figure(mode)

        if mode == "human":
            if self.use_animation and not self.animation_running:
                self._start_animation()
            elif not self.use_animation:
                self._update_figure_data()
                assert self.fig is not None
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.01)
        elif mode == "rgb_array":
            self._update_figure_data()
            assert self.fig is not None
            canvas = cast(FigureCanvasAgg, self.fig.canvas)
            canvas.draw()
            w, h = canvas.get_width_height()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            img = buf.reshape((h, w, 4))[:, :, :3]  # 提取 RGB 通道，去掉 Alpha 通道
            return img.copy()

    def _setup_figure(self, mode: str):
        """初始化绘图对象"""
        set_chinese_font()
        # 仅在human模式下启用交互模式
        if mode == "human":
            plt.ion()

        # 创建图形窗口
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle("磁悬浮列车智能体训练过程", fontsize=14)

        # 设置坐标轴范围
        pos_margin = (
            abs(self.train_service.target_position - self.current_position) * 0.1
        )
        self.ax.set_xlim(
            min(self.current_position, self.train_service.target_position) - pos_margin,
            max(self.current_position, self.train_service.target_position) + pos_margin,
        )
        self.ax.set_ylim(0, 600.0)  # 速度范围，单位：km/h

        # 绘制起点和终点
        self.ax.scatter(
            self.train_service.start_position,
            abs(self.train_service.start_speed * 3.6),
            marker="o",
            color="blue",
            s=60,
            alpha=0.8,
            label="起点",
            zorder=5,
        )
        self.ax.scatter(
            x=self.train_service.target_position,
            y=0.0,
            marker="o",
            color="red",
            s=60,
            alpha=0.8,
            label="终点",
            zorder=5,
        )

        # 绘制限速和危险速度域
        self.safeguard_utility.render(ax=self.ax)

        # 初始化动态绘制对象
        (self.vehicle_dot,) = self.ax.plot(
            [], [], "g*", markersize=8, label="列车", zorder=4
        )
        (self.traj_line,) = self.ax.plot([], [], "b-", lw=2, label="轨迹", zorder=2)

        # 设置标签和格式
        self.ax.set_xlabel("位置 (m)")
        self.ax.set_ylabel("速度 (km/h)")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def _start_animation(self):
        """启动FuncAnimation动画"""
        if self.animation is None and self.fig is not None:
            self.animation = FuncAnimation(
                self.fig,
                self._animate,  # type:ignore
                interval=self.animation_interval,
                blit=True,  # 启用blit以提高性能
                cache_frame_data=False,
            )
            self.animation_running = True

    def _stop_animation(self):
        """停止动画"""
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
            self.animation_running = False

    def _animate(self, frame):
        """FuncAnimation的更新函数"""

        self._update_figure_data()

        return self.vehicle_dot, self.traj_line

    def _update_figure_data(self):
        assert self.vehicle_dot is not None
        assert self.traj_line is not None
        # 更新列车位置
        self.vehicle_dot.set_data([self.current_position], [self.current_speed * 3.6])
        # 更新轨迹
        if (
            self.trajectory_pos is not None
            and self.trajectory_speed_mps is not None
            and len(self.trajectory_pos) > 1
        ):
            trajectory_speed_km_h = [speed * 3.6 for speed in self.trajectory_speed_mps]
            self.traj_line.set_data(self.trajectory_pos, trajectory_speed_km_h)

    def close(self):
        """清理资源"""
        self._stop_animation()
        if self.fig is not None:
            if self.render_mode == "human":
                plt.ioff()
            plt.close(self.fig)
            self.fig = None
