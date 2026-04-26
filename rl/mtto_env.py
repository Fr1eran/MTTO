from typing import Any, TypedDict, cast
import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.interpolate import PchipInterpolator

from model.ocs import SafeGuardUtility, TrainService, SPS
from model.vehicle import VehicleInfo
from model.track import TrackInfo, TrackProfile
from model.common import ECC, ORS
from utils.indexing_utils import get_interval_index
from utils.plot_utils import set_chinese_font


class RewardInfoForTB(TypedDict, total=False):
    safety: float
    docking: float
    punctuality: float
    energy: float
    comfort: float
    total: float


class StateInfoForTB(TypedDict, total=False):
    pos_m: float
    speed_mps: float
    current_sp: float


class ConstraintInfoForTB(TypedDict, total=False):
    margin_to_vmax_mps: float
    margin_to_vmin_mps: float
    is_truncated: float
    violation_code: float
    speed_limit_mps: float
    speed_limit_segment: float
    is_near_miss: float


class EventInfoForTB(TypedDict, total=False):
    episode_truncated_count: float
    episode_low_violation_count: float
    episode_high_violation_count: float


class BasicInfo(TypedDict, total=False):
    energy_consumption: float
    operation_time: float
    position: float
    stopping_point_index: float
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
    latest_traction_intervention_point: float
    latest_braking_intervention_point: float
    operation_time: float
    energy_consumption: float
    stopping_point_index: int


class MTTOEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        vehicle: VehicleInfo,
        track: TrackInfo,
        safeguard_utility: SafeGuardUtility,
        train_service: TrainService,
        gamma: float,
        max_step_distance: float,
        enable_diagnostics: bool = True,
        diagnostics_interval_steps: int = 1,
        enable_trajectory_tracking: bool = False,
        render_mode: str | None = None,
        use_animation: bool = False,
    ):
        super().__init__()

        # 磁浮列车运行优化车辆实例
        self.vehicle = vehicle

        # 磁浮列车运行优化的线路实例
        self.track = track
        self.trackprofile = TrackProfile(track=self.track)

        # 磁浮列车安全防护实例
        self.safeguard_utility = safeguard_utility

        # 运行任务约束
        self.train_service = train_service

        # 回报折扣因子
        self.gamma = gamma

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
            + 3
        )

        # 定义常量
        # 包含：
        # - 运行总距离, 单位: m
        # - 运动方向
        # - 目标速度, 单位: m/s
        self.whole_distance: float = abs(
            self.train_service.target_position - self.train_service.start_position
        )
        self.direction: int = (
            1
            if self.train_service.start_position < self.train_service.target_position
            else -1
        )
        self.goal_speed: float = 0.0

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
        # 速度上限曲线采用三次埃尔米特插值
        self.upper_speed_profile_interp_func = PchipInterpolator(
            x=self.upper_speed_profile_pos_arr,
            y=self.upper_speed_profile_speed_arr,
            extrapolate=False,
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

        # 计算参考曲线上每个位置对应的最短累计耗时
        # self.ref_curve_cum_time = self._calc_ref_cum_time()

        # 初始化状态
        # 包含:
        # - 当前位置, 单位: m
        # - 当前运行速度大小, 单位: m/s
        # - 当前加速度, 单位: m/s^2
        # - 当前列车运行总时间
        # - 当前列车消耗总能量, 单位: J
        # - 列车质量（对智能体决策似乎不起作用）, 单位: t
        # - 当前位置对应的坡度, 百分比制
        # - 当前位置对应的最大运行速度大小, 单位: m/s
        # - 当前位置对应的最小运行速度大小, 单位: m/s
        # - 下步状态转移后可能位置对应的坡度, 百分比制
        # - 下步状态转移后可能位置对应的最大运行速度大小, 单位: m/s
        # - 下步状态转移后可能位置对应的最小运行速度大小, 单位: m/s
        # - 当前状态所对应的最迟牵引干预位置, 单位: m/s
        # - 当前状态所对应的最迟制动干预位置, 单位: m/s
        self.current_pos: float = self.train_service.start_position
        self.current_speed: float = self.train_service.start_speed
        self.current_acc: float = 0.0
        self.current_operation_time: float = 0.0
        self.current_energy_consumption: float = 0.0
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.get_slope(pos=self.current_pos)
        self.current_sp: int = -1  # 初始时在加速区，尚未步进至第一个辅助停车区
        self.current_min_speed, self.current_max_speed = (
            self.safeguard_utility.get_min_and_max_speed(
                current_pos=self.current_pos,
                current_sp=self.current_sp,
            )
        )
        self.current_max_speed = min(
            self._get_upper_speed(self.current_pos),
            self.current_max_speed,
        )
        self.next_slope = self.trackprofile.get_slope(
            pos=self.current_pos + self.max_step_distance * self.direction
        )
        self.next_min_speed, self.next_max_speed = (
            self.safeguard_utility.get_min_and_max_speed(
                current_pos=self.current_pos + self.max_step_distance * self.direction,
                current_sp=self.current_sp,
            )
        )
        self.next_max_speed = min(
            self._get_upper_speed(
                self.current_pos + self.max_step_distance * self.direction
            ),
            self.next_max_speed,
        )
        (
            self.current_latest_traction_intervention_point,
            self.current_latest_braking_intervention_point,
        ) = self.safeguard_utility.get_latest_traction_and_braking_intervention_points(
            current_speed=self.current_speed, current_sp=self.current_sp
        )

        # 定义智能体能够观测的状态信息
        self.observation_space = gym.spaces.Dict(
            {
                "remaining_distance": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "current_speed": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "current_acc": gym.spaces.Box(
                    -1.0,
                    1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "remaining_schedule_time": gym.spaces.Box(
                    -1.0,  # 最多超时10分钟
                    1.0,
                    shape=(1,),
                    dtype=np.float32,  # 允许超时或提前
                ),
                "current_slope": gym.spaces.Box(
                    -1.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "current_max_speed": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "current_min_speed": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "next_slope": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "next_max_speed": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "next_min_speed": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "current_latest_traction_intervention_point": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "current_latest_braking_intervention_point": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "is_final_approach": gym.spaces.Box(
                    -1.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "rel_dist_to_target": gym.spaces.Box(
                    -1.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "required_dec": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

        # 定义智能体的动作空间, 归一化
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # 当前训练步数
        self.current_steps: int = 0

        # 当前步进停车点编号
        self.current_sp: int = -1

        # 停车点步进机制模拟
        self.sps = SPS(
            sgu=self.safeguard_utility,
            ASA_ap_list=self.track.ASA_aps,
            ASA_dp_list=self.track.ASA_dps,
            T_s=1.0,
        )

        # Q初始值
        self.q_init: float = 0.0

        self.rewards_info: RewardInfoForTB = {}
        self.state_info: StateInfoForTB = {}
        self.constraint_info: ConstraintInfoForTB = {}
        self.event_info: EventInfoForTB = {}
        self.basic_info: BasicInfo = {}
        self.episode_truncated_count: int = 0
        self.episode_low_violation_count: int = 0
        self.episode_high_violation_count: int = 0

        self.current_speed_limit_mps: float = float(
            self.trackprofile.get_speed_limit(pos=self.current_pos)
        )
        self.current_speed_limit_segment: int = self._get_speed_limit_segment(
            self.current_pos
        )
        self.last_constraint_is_truncated: bool = False
        self.last_constraint_violation_code: int = 0

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

    def _get_obs(self):
        """
        将内部状态转换为可观测形式，并进行归一化
        可观测状态全部由np.ndarray组成

        Returns:
            dict: Observation with agent and target positions
        """

        dist_to_target = self.train_service.target_position - self.current_pos
        is_final_approach = True if abs(dist_to_target) <= 3000.0 else False
        rel_dist_to_target = 0.0
        required_dec_normalized = 0.0
        if is_final_approach:
            rel_dist_to_target = dist_to_target / 3000.0
            # 末段按匀减速停车估算所需减速度（物理符号为负）并映射到动作归一化区间
            dist_abs = max(abs(dist_to_target), 1e-6)
            required_dec = -(self.current_speed**2) / (2.0 * dist_abs)
            required_dec_normalized = self._normalize_acc_to_action(required_dec)

        return {
            "remaining_distance": np.array(
                [(self.whole_distance - self.current_pos) / self.whole_distance],
                dtype=np.float32,
            ),
            "current_speed": np.array(
                [self.current_speed / self.vehicle.max_speed], dtype=np.float32
            ),
            "current_acc": np.array([self.current_acc], dtype=np.float32),
            "remaining_schedule_time": np.array(
                [
                    (self.train_service.schedule_time - self.current_operation_time)
                    / self.train_service.schedule_time
                ],
                dtype=np.float32,
            ),
            "current_slope": np.array(
                [self.current_slope / self.vehicle.max_slope_capacity], dtype=np.float32
            ),
            "current_max_speed": np.array(
                [self.current_max_speed / self.vehicle.max_speed], dtype=np.float32
            ),
            "current_min_speed": np.array(
                [self.current_min_speed / self.vehicle.max_speed], dtype=np.float32
            ),
            "next_slope": np.array(
                [self.next_slope / self.vehicle.max_slope_capacity], dtype=np.float32
            ),
            "next_max_speed": np.array(
                [self.next_max_speed / self.vehicle.max_speed], dtype=np.float32
            ),
            "next_min_speed": np.array(
                [self.next_min_speed / self.vehicle.max_speed], dtype=np.float32
            ),
            "current_latest_traction_intervention_point": np.array(
                [self.current_latest_traction_intervention_point / self.whole_distance],
                dtype=np.float32,
            ),
            "current_latest_braking_intervention_point": np.array(
                [self.current_latest_braking_intervention_point / self.whole_distance],
                dtype=np.float32,
            ),
            "is_final_approach": np.array(
                [1.0 if is_final_approach else 0.0],
                dtype=np.float32,
            ),
            "rel_dist_to_target": np.array(
                [rel_dist_to_target],
                dtype=np.float32,
            ),
            "required_dec": np.array(
                [required_dec_normalized],
                dtype=np.float32,
            ),
        }

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

    def _reset_episode_counters(self) -> None:
        self.episode_truncated_count = 0
        self.episode_low_violation_count = 0
        self.episode_high_violation_count = 0

    def _should_collect_step_diagnostics(self) -> bool:
        if not self.enable_diagnostics:
            return False
        return self.current_steps % self.diagnostics_interval_steps == 0

    def _record_runtime_info(self) -> None:
        self.basic_info["energy_consumption"] = float(self.current_energy_consumption)
        self.basic_info["operation_time"] = float(self.current_operation_time)
        self.basic_info["position"] = float(self.current_pos)
        self.basic_info["stopping_point_index"] = float(self.current_sp)
        self.basic_info["comfort_tav"] = self._comfort_tav
        if self.current_steps > 0:
            self.basic_info["comfort_er_pct"] = (
                self._comfort_exceedance_count / self.current_steps * 100.0
            )
            self.basic_info["comfort_rms"] = math.sqrt(
                self._comfort_sum_sq_delta_acc / self.current_steps
            )

    def _record_step_diagnostics(
        self,
        *,
        truncated: bool,
        violation_code: int,
        margin_to_vmax: float,
        margin_to_vmin: float,
        near_miss_margin_mps: float = 1.0,
    ) -> None:
        self.last_constraint_is_truncated = truncated
        self.last_constraint_violation_code = violation_code

        if truncated:
            self.episode_truncated_count += 1
            if violation_code == 1:
                self.episode_low_violation_count += 1
            elif violation_code == 2:
                self.episode_high_violation_count += 1

        is_near_miss = (margin_to_vmax <= near_miss_margin_mps) or (
            margin_to_vmin <= near_miss_margin_mps
        )

        self.state_info["pos_m"] = float(self.current_pos)
        self.state_info["speed_mps"] = float(self.current_speed)
        self.state_info["current_sp"] = float(self.current_sp)

        self.constraint_info["margin_to_vmax_mps"] = float(margin_to_vmax)
        self.constraint_info["margin_to_vmin_mps"] = float(margin_to_vmin)
        self.constraint_info["is_truncated"] = float(truncated)
        self.constraint_info["violation_code"] = float(violation_code)
        self.constraint_info["speed_limit_mps"] = float(self.current_speed_limit_mps)
        self.constraint_info["speed_limit_segment"] = float(
            self.current_speed_limit_segment
        )
        self.constraint_info["is_near_miss"] = float(is_near_miss)

        self.event_info["episode_truncated_count"] = float(self.episode_truncated_count)
        self.event_info["episode_low_violation_count"] = float(
            self.episode_low_violation_count
        )
        self.event_info["episode_high_violation_count"] = float(
            self.episode_high_violation_count
        )

    def _get_action_denormalized(self, action: float | np.floating) -> float:
        """将动作反归一化为列车加速度"""
        return float(
            (self.vehicle.max_acc + self.vehicle.max_dec) / 2
            + action * (self.vehicle.max_acc - self.vehicle.max_dec) / 2
        )

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
        self._reset_episode_counters()
        self._comfort_tav = 0.0
        self._comfort_sum_sq_delta_acc = 0.0
        self._comfort_exceedance_count = 0

        # 重新初始化运行状态
        self.current_pos = self.train_service.start_position
        self.current_speed = self.train_service.start_speed
        self.current_acc = 0.0
        self.current_operation_time = 0.0
        self.current_energy_consumption = 0.0
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.get_slope(pos=self.current_pos)

        # 重置停车点步进
        self.current_sp = -1
        self.sps.reset()

        self.current_min_speed, self.current_max_speed = (
            self.safeguard_utility.get_min_and_max_speed(
                current_pos=self.current_pos, current_sp=self.current_sp
            )
        )
        self.current_max_speed = min(
            self._get_upper_speed(self.current_pos),
            self.current_max_speed,
        )
        self.next_slope = self.trackprofile.get_slope(
            pos=self.current_pos + self.max_step_distance
        )
        (
            self.next_min_speed,
            self.next_max_speed,
        ) = self.safeguard_utility.get_min_and_max_speed(
            current_pos=self.current_pos + self.max_step_distance,
            current_sp=self.current_sp,
        )
        self.next_max_speed = min(
            self._get_upper_speed(self.current_pos + self.max_step_distance),
            self.next_max_speed,
        )
        (
            self.current_latest_traction_intervention_point,
            self.current_latest_braking_intervention_point,
        ) = self.safeguard_utility.get_latest_traction_and_braking_intervention_points(
            current_speed=self.current_speed, current_sp=self.current_sp
        )
        self.current_speed_limit_mps = float(
            self.trackprofile.get_speed_limit(pos=self.current_pos)
        )
        self.current_speed_limit_segment = self._get_speed_limit_segment(
            self.current_pos
        )
        self.last_constraint_is_truncated = False
        self.last_constraint_violation_code = 0

        # 重置历史数据
        self._reset_history()

        # 重置轨迹数据
        if self.enable_trajectory_tracking:
            self._reset_trajectory()

        # 重置仿真步数
        self.current_steps = 0

        # 重置奖励记录
        self.rewards_info = {}
        self.state_info = {}
        self.constraint_info = {}
        self.event_info = {}
        self.basic_info = {}
        self._collect_step_diagnostics = False

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
            trackprofile=self.trackprofile,
        )
        energy_consumption = current_mec + current_lec

        # 更新运行状态
        self.current_pos += distance * self.direction
        self.current_speed = next_speed
        self.current_operation_time += operation_time
        self.current_energy_consumption += energy_consumption
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.get_slope(pos=self.current_pos)
        self.current_sp = self.sps.step_to_next_stopping_point(
            current_pos=self.current_pos,
            current_speed=self.current_speed,
            current_time=self.current_operation_time,
            current_sp=self.current_sp,
        )
        self.current_min_speed, self.current_max_speed = (
            self.safeguard_utility.get_min_and_max_speed(
                current_pos=self.current_pos,
                current_sp=self.current_sp,
            )
        )
        self.current_max_speed = min(
            self._get_upper_speed(self.current_pos),
            self.current_max_speed,
        )
        self.next_slope = self.trackprofile.get_slope(
            pos=self.current_pos + self.max_step_distance * self.direction
        )
        self.next_min_speed, self.next_max_speed = (
            self.safeguard_utility.get_min_and_max_speed(
                current_pos=self.current_pos + self.max_step_distance * self.direction,
                current_sp=self.current_sp,
            )
        )
        self.next_max_speed = min(
            self._get_upper_speed(
                self.current_pos + self.max_step_distance * self.direction
            ),
            self.next_max_speed,
        )
        (
            self.current_latest_traction_intervention_point,
            self.current_latest_braking_intervention_point,
        ) = self.safeguard_utility.get_latest_traction_and_braking_intervention_points(
            current_speed=self.current_speed, current_sp=self.current_sp
        )
        self.current_speed_limit_mps = float(
            self.trackprofile.get_speed_limit(pos=self.current_pos)
        )
        self.current_speed_limit_segment = self._get_speed_limit_segment(
            self.current_pos
        )

        # 判断智能体是否到达目标区域
        terminated = abs(
            self.train_service.target_position - self.current_pos
        ) <= self.train_service.max_stop_error * 10 and math.isclose(
            self.current_speed, 0.0, abs_tol=0.1
        )

        # 若智能体违反安全约束或者达到最大步数，则截断训练进程
        truncated = (
            (self.current_speed < self.current_min_speed)
            or (self.current_speed > self.current_max_speed)
            or self.current_steps == self.max_episode_steps
        )

        # 计算奖励
        reward = self._get_reward(
            terminated=terminated,
            truncated=truncated,
        )

        if self.enable_trajectory_tracking:
            # 记录轨迹数据
            self._record_trajectory(
                pos=self.last_state["pos"],
                speed=self.last_state["speed"],
                acc=self.current_acc,
                displacement=distance,
                operation_time=operation_time,
            )

        # 获取可观测状态和信息
        observation = self._get_obs()

        self._collect_step_diagnostics = self._should_collect_step_diagnostics()
        self._record_runtime_info()
        info = self._get_basic_info()
        if self.enable_diagnostics and self._collect_step_diagnostics:
            if self.current_speed < self.current_min_speed:
                violation_code = 1
            elif self.current_speed >= self.current_max_speed:
                violation_code = 2
            else:
                violation_code = 0

            margin_to_vmax = self.current_max_speed - self.current_speed
            margin_to_vmin = self.current_speed - self.current_min_speed
            self._record_step_diagnostics(
                truncated=truncated,
                violation_code=violation_code,
                margin_to_vmax=margin_to_vmax,
                margin_to_vmin=margin_to_vmin,
            )

            info["rewards"] = dict(self.rewards_info)
            info["state"] = dict(self.state_info)
            info["constraint"] = dict(self.constraint_info)
            info["event"] = dict(self.event_info)

        return (observation, reward, terminated, truncated, info)

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

        ds = np.diff(self.upper_speed_profile_pos_arr)
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

    def _get_upper_speed(self, pos: float | np.floating):
        return max(0.0, self.upper_speed_profile_interp_func(pos))

    def _get_reward(
        self,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        reward_total = 0.0

        if not truncated:
            if terminated:
                reward_total = self._get_reward_goal() + 50.0
            else:
                reward_total = self._get_reward_dense()
        else:
            progress = (
                abs(self.current_pos - self.train_service.start_position)
                / self.whole_distance
            )
            # reward_total = -150.0 * (1.0 - np.sqrt(progress)) - 50.0
            reward_total = -50.0 * (1.0 - np.sqrt(progress))

        if self.enable_diagnostics and self._collect_step_diagnostics:
            self.rewards_info["total"] = reward_total

        return reward_total

    def _get_reward_dense(
        self,
    ) -> float:
        # 安全奖励
        reward_safety = self._get_reward_safety_dense()

        # 能耗奖励
        reward_energy = self._get_reward_energy_dense()

        # 舒适度奖励
        reward_comfort = self._get_reward_comfort_dense()

        # 运行时间奖励
        reward_punctuality = self._get_reward_punctuality_dense()

        # 停站奖励
        reward_docking = self._get_reward_docking_dense()

        if self.enable_diagnostics and self._collect_step_diagnostics:
            self.rewards_info["safety"] = reward_safety
            self.rewards_info["energy"] = reward_energy
            self.rewards_info["comfort"] = reward_comfort
            self.rewards_info["punctuality"] = reward_punctuality
            self.rewards_info["docking"] = reward_docking

        return (
            reward_safety
            + reward_energy
            + reward_comfort
            + reward_punctuality
            + reward_docking
        )

    def _get_reward_safety_dense(self) -> float:
        # 里程碑式奖励
        small_bonus = 0.0
        if self.current_sp != self.last_state["stopping_point_index"]:
            small_bonus = 10.0 / self.sps.num_of_stopping_points

        # 计算当前状态势能

        phi_curr = self._potential_safety_speed(
            pos=self.current_pos,
            speed=self.current_speed,
            min_speed=self.current_min_speed,
            max_speed=self.current_max_speed,
            target_pos=self.sps.get_auxiliary_stopping_area_target_position(
                sp=self.current_sp
            )
            if self.current_sp >= 0
            else self.train_service.target_position,
        )

        # phi_curr = self._potential_safety_position(
        #     pos=self.current_pos,
        #     min_pos=self.current_latest_traction_intervention_point,
        #     max_pos=self.current_latest_braking_intervention_point,
        #     target_pos=self.sps.get_auxiliary_stopping_area_target_position(
        #         sp=self.current_sp
        #     )
        #     if self.current_sp >= 0
        #     else self.task.target_position,
        # )

        # 计算上个状态势能

        phi_prev = self._potential_safety_speed(
            pos=self.last_state["pos"],
            speed=self.last_state["speed"],
            min_speed=self.last_state["min_speed"],
            max_speed=self.last_state["max_speed"],
            # target_pos=self.sps.get_auxiliary_stopping_area_target_position(
            #     sp=self.last_state["stopping_point_index"]
            # )
            # if self.last_state["stopping_point_index"] >= 0
            # else self.task.target_position,
            target_pos=self.sps.get_auxiliary_stopping_area_target_position(
                sp=self.current_sp
            )
            if self.current_sp >= 0
            else self.train_service.target_position,
        )

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
        #     else self.task.target_position,
        # )

        return self.gamma * phi_curr - phi_prev + small_bonus

    def _potential_safety_speed(
        self,
        pos: float,
        speed: float,
        min_speed: float,
        max_speed: float,
        target_pos: float,
    ) -> float:
        distanceToTarget = abs(target_pos - pos)
        center_speed = (max_speed + min_speed) / 2.0
        safe_margin = max((max_speed - min_speed) / 2.0, 0.5)

        # 基础偏离惩罚(二次方项, 引导列车走中间)
        norm_speed_diff = (speed - center_speed) / safe_margin
        phi_base = 2.0 * np.log(1.01 - norm_speed_diff**2)

        # 靠近目标位置时，适当增大惩罚力度
        scale = 1.0 + 1.0 * np.exp(-0.001 * distanceToTarget)

        return scale * phi_base

    def _potential_safety_position(
        self, pos: float, min_pos: float, max_pos: float, target_pos: float
    ):
        distanceToTarget = np.abs(target_pos - pos)

        center_pos = (max_pos + min_pos) / 2.0
        safe_margin = (max_pos - min_pos) / 2.0

        safe_margin = max(safe_margin, 1e-3)

        norm_pos_diff = (pos - center_pos) / safe_margin
        phi_base = 2.0 * np.log(1.1 - norm_pos_diff**2)

        scale = 1.0 + 1.0 * np.exp(-0.001 * distanceToTarget)

        return scale * phi_base

    def _get_reward_energy_dense(self) -> float:
        # if (
        #     abs(self.current_pos - self.task.start_position) < 3000.0
        #     or abs(self.current_pos - self.task.target_position) < 3000.0
        # ):
        #     return 0.0

        val = (
            -(self.current_energy_consumption - self.last_state["energy_consumption"])
            / self.max_energy_consumption
        ) * 10.0

        return val

    def _get_reward_comfort_dense(self) -> float:
        # if (
        #     abs(self.current_pos - self.task.start_position) < 3000.0
        #     or abs(self.current_pos - self.task.target_position) < 3000.0
        # ):
        #     return 0.0

        delta_acc = abs(self.last_state["acc"] - self.current_acc)
        # 使用指数衰减式的钟形曲线
        norm_jerk = delta_acc / (self.train_service.max_acc_change)

        val = -0.08 * (1 - np.exp(-3.0 * norm_jerk))

        return val

    def _get_reward_punctuality_dense(self) -> float:
        phi_curr = self._potential_punctuality(
            pos=self.current_pos,
            speed=self.current_speed,
            operation_time=self.current_operation_time,
        )

        phi_prev = self._potential_punctuality(
            pos=self.last_state["pos"],
            speed=self.last_state["speed"],
            operation_time=self.last_state["operation_time"],
        )

        return self.gamma * phi_curr - phi_prev

    # 基于连续非线性衰减的冗余时间势能场
    def _potential_punctuality(self, pos: float, speed: float, operation_time: float):
        # 计算理论上跑完剩余路程的最短运行时间
        min_remaining_operation_time = self.ors.calc_min_operation_time(
            begin_pos=pos,
            begin_speed=speed,
            end_pos=self.train_service.target_position,
            end_speed=0.0,
        )

        # 计算实际剩余规划运行时间
        actual_remaining_operation_time = (
            self.train_service.schedule_time - operation_time
        )

        # 计算冗余运行时间
        redundant_operation_time = (
            actual_remaining_operation_time - min_remaining_operation_time
        )

        # 计算时间冗余度
        time_redundancy_norm = (
            redundant_operation_time / self.train_service.schedule_time
        )

        return -10.0 * np.log1p(np.exp(-12.0 * time_redundancy_norm))

    def _get_reward_docking_dense(self):
        phi_curr = self._potential_docking(
            pos=self.current_pos, speed=self.current_speed
        )

        phi_prev = self._potential_docking(
            pos=self.last_state["pos"], speed=self.last_state["speed"]
        )

        return self.gamma * phi_curr - phi_prev

    def _potential_docking(self, pos: float, speed: float):
        # 基础参数
        sigma_x_hat = 0.1
        sigma_v_hat = 0.2
        K_L = 4.0
        K_G = 12.0

        # 正则化
        dist_error = self.train_service.target_position - pos
        x_hat = dist_error / self.whole_distance
        v_hat = speed / self.vehicle.max_speed

        # 势能项
        phi_linear = -K_L * np.sqrt(x_hat**2 + v_hat**2)
        phi_strong = K_G * np.exp(
            -np.abs(x_hat) / sigma_x_hat - np.abs(v_hat) / sigma_v_hat
        )

        return phi_linear + phi_strong

    def _get_reward_goal(
        self,
    ) -> float:
        reward_docking = self._get_reward_docking_goal() * 8.0
        reward_punctuality = self._get_reward_punctuality_goal() * 5.0

        if self.enable_diagnostics and self._collect_step_diagnostics:
            self.rewards_info["docking"] = reward_docking
            self.rewards_info["punctuality"] = reward_punctuality

        return reward_docking + reward_punctuality

    def _get_reward_docking_goal(self) -> float:
        # 停站位置误差不超过0.3m
        docking_pos_error = abs(self.train_service.target_position - self.current_pos)

        return self._gaussian_kernel(A=2, B=-1, k=2.310490602, x=docking_pos_error)

    def _get_reward_punctuality_goal(self) -> float:
        # 列车到站时刻误差不超过2min
        ontime_error = abs(
            self.train_service.schedule_time - self.current_operation_time
        )

        return self._gaussian_kernel(
            A=2,
            B=-1,
            k=69.314718056
            / (
                self.train_service.schedule_time
                * self.train_service.max_arr_time_error_ratio
            ),
            x=ontime_error,
        )

    def _gaussian_kernel(self, A: float, B: float, k: float, x: float) -> float:
        return A * np.exp(-k * x) + B

    def _record_history(self) -> None:
        self.last_state["pos"] = self.current_pos
        self.last_state["speed"] = self.current_speed
        self.last_state["acc"] = self.current_acc
        self.last_state["min_speed"] = self.current_min_speed
        self.last_state["max_speed"] = self.current_max_speed
        self.last_state["latest_traction_intervention_point"] = (
            self.current_latest_traction_intervention_point
        )
        self.last_state["latest_braking_intervention_point"] = (
            self.current_latest_braking_intervention_point
        )
        self.last_state["operation_time"] = self.current_operation_time
        self.last_state["energy_consumption"] = self.current_energy_consumption
        self.last_state["stopping_point_index"] = self.current_sp

    def _reset_history(self) -> None:
        self.last_state: TrainState = {
            "pos": 0.0,
            "speed": 0.0,
            "acc": 0.0,
            "min_speed": 0.0,
            "max_speed": 0.0,
            "latest_traction_intervention_point": 0.0,
            "latest_braking_intervention_point": 0.0,
            "operation_time": 0.0,
            "energy_consumption": 0.0,
            "stopping_point_index": -1,
        }

    def _record_trajectory(
        self,
        pos: float,
        speed: float,
        acc: float = 0.0,
        displacement: float = 0.0,
        operation_time: float = 0.0,
    ):
        """记录完整的匀变速运动轨迹"""
        if not self.enable_trajectory_tracking:
            return

        assert self.trajectory_pos is not None
        assert self.trajectory_speed_mps is not None

        # 如果是初始状态或者没有运动，只记录当前点
        if abs(operation_time) < 1e-6 or abs(displacement) < 1e-6:
            self.trajectory_pos.append(pos)
            self.trajectory_speed_mps.append(abs(speed))
            return

        # 生成中间轨迹点数量（基于运动时间动态调整）
        # 确保轨迹足够平滑，同时避免过多的点
        num_points = max(5, min(100, int(operation_time * 4)))  # 5-100个点之间

        # 生成时间序列
        time_steps = np.linspace(0, operation_time, num_points)

        # 向量化计算所有时间点的位置和速度（匀变速运动公式）
        # 位置：s = s0 + v0*t + 0.5*a*t^2
        pos_array = pos + speed * time_steps + 0.5 * acc * time_steps**2
        # 速度：v = v0 + a*t
        vel_array = speed + acc * time_steps

        # 批量添加到历史记录中
        self.trajectory_pos.extend(pos_array.tolist())
        self.trajectory_speed_mps.extend(np.abs(vel_array).tolist())

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
        pos_margin = abs(self.train_service.target_position - self.current_pos) * 0.1
        self.ax.set_xlim(
            min(self.current_pos, self.train_service.target_position) - pos_margin,
            max(self.current_pos, self.train_service.target_position) + pos_margin,
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
        self.vehicle_dot.set_data([self.current_pos], [self.current_speed * 3.6])
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
