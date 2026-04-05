from typing import Any, TypedDict, cast
import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.interpolate import PchipInterpolator

from model.SafeGuard import SafeGuardUtility
from model.Vehicle import Vehicle
from model.Track import Track, TrackProfile
from model.Task import Task
from model.ORS import ORS
from model.SPS import SPS
from model.ECC import ECC
from utils.misc import SetChineseFont


class RewardsInfoForTB(TypedDict, total=False):
    reward_safety: float
    reward_docking: float
    reward_punctuality: float
    reward_energy: float
    reward_comfort: float
    reward_total: float


class TrainState(TypedDict, total=True):
    pos: float
    speed: float
    acc: float
    min_speed: float
    max_speed: float
    latest_traction_intervation_point: float
    latest_braking_intervation_point: float
    operation_time: float
    energy_consumption: float
    stopping_point_index: int


class MTTOEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        vehicle: Vehicle,
        track: Track,
        safeguardutil: SafeGuardUtility,
        task: Task,
        gamma: float,
        max_step_distance: float,
        render_mode: str | None = None,
        use_animation=False,
    ):
        super().__init__()

        # 磁浮列车运行优化车辆实例
        self.vehicle = vehicle

        # 磁浮列车运行优化的线路实例
        self.track = track
        self.trackprofile = TrackProfile(track=self.track)

        # 磁浮列车安全防护实例
        self.safeguardutil = safeguardutil

        # 运行任务约束
        self.task = task

        # 回报折扣因子
        self.gamma = gamma

        # 单步状态转移容许的最大位移量
        self.max_step_distance: float = max_step_distance

        # 最大转移步数
        self.max_episode_steps: int = (
            math.ceil(
                abs(self.task.target_position - self.task.start_position)
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
            self.task.target_position - self.task.start_position
        )
        self.direction: int = (
            1 if self.task.start_position < self.task.target_position else -1
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
            gamma=self.safeguardutil.gamma,
        )

        # 计算最短运行时间参考曲线
        self.ref_curve_pos, self.ref_curve_speed = self.ors.CalcMinRuntimeCurve(
            begin_pos=self.task.start_position,
            begin_speed=self.task.start_speed,
            end_pos=self.task.target_position + self.task.max_stop_error * 10,
            end_speed=0.0,
        )
        # 最短运行时间曲线采用三次埃尔米特插值
        self.ref_curve_interp_func = PchipInterpolator(
            x=self.ref_curve_pos, y=self.ref_curve_speed, extrapolate=False
        )

        # 计算最大能耗和最短运行时间
        mec, lec, self.min_operation_time = self.ors.CalcRefEnergyAndOperationTime(
            begin_pos=self.task.start_position,
            begin_speed=self.task.start_speed,
            end_pos=self.task.target_position + self.task.max_stop_error * 10,
            end_speed=0.0,
            distance=self.task.target_position
            + self.task.max_stop_error
            - self.task.start_position,
            energy_con_calc=self.ecc,
        )
        self.max_energy_consumption = mec + lec

        # 计算参考曲线上每个位置对应的最短累计耗时
        self.ref_curve_cum_time = self._calc_ref_cum_time()

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
        self.current_pos: float = self.task.start_position
        self.current_speed: float = self.task.start_speed
        self.current_acc: float = 0.0
        self.current_operation_time: float = 0.0
        self.current_energy_consumption: float = 0.0
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.GetSlope(pos=self.current_pos)
        self.current_sp: int = -1  # 初始时在加速区，尚未步进至第一个辅助停车区
        self.current_min_speed, self.current_max_speed = (
            self.safeguardutil.GetMinAndMaxSpeed(
                current_pos=self.current_pos,
                current_sp=self.current_sp,
            )
        )
        self.current_max_speed = min(
            self._get_ref_speed(self.current_pos),
            self.current_max_speed,
        )
        self.next_slope = self.trackprofile.GetSlope(
            pos=self.current_pos + self.max_step_distance * self.direction
        )
        self.next_min_speed, self.next_max_speed = self.safeguardutil.GetMinAndMaxSpeed(
            current_pos=self.current_pos + self.max_step_distance * self.direction,
            current_sp=self.current_sp,
        )
        self.next_max_speed = min(
            self._get_ref_speed(
                self.current_pos + self.max_step_distance * self.direction
            ),
            self.next_max_speed,
        )
        (
            self.current_latest_traction_intervation_point,
            self.current_latest_braking_intervention_point,
        ) = self.safeguardutil.GetLatestTranctionAndBrakingIntervationPoint(
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
                "current_latest_traction_intervation_point": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "current_latest_braking_intervation_point": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
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
            sgu=self.safeguardutil,
            ASA_ap_list=self.track.ASA_aps,
            ASA_dp_list=self.track.ASA_dps,
            T_s=1.0,
        )

        # Q初始值
        self.q_init: float = 0.0

        self.rewards_info: RewardsInfoForTB = {}

        # 状态历史记录
        self._reset_history()

        # 列车运行轨迹
        self.trajectory_pos: list[float] = [self.current_pos]
        self.trajectory_speed_km_h: list[float] = [self.current_speed * 3.6]

        # 渲染模式
        self.render_mode = render_mode
        # 是否启用动画
        self.use_animation = use_animation
        # 可视化时需要的绘图实例
        self.fig = None
        self.ax = None
        self.vehicle_dot, self.traj_line = None, None
        self.animation = None
        self.animation_running = False

        # 动画配置
        self.animation_interval = 15  # 动画更新间隔(ms)

    def _get_obs(self):
        """
        将内部状态转换为可观测形式，并进行归一化
        可观测状态全部由np.ndarray组成

        Returns:
            dict: Observation with agent and target positions
        """

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
                    (self.task.schedule_time - self.current_operation_time)
                    / self.task.schedule_time
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
            "current_latest_traction_intervation_point": np.array(
                [self.current_latest_traction_intervation_point / self.whole_distance],
                dtype=np.float32,
            ),
            "current_latest_braking_intervation_point": np.array(
                [self.current_latest_braking_intervention_point / self.whole_distance],
                dtype=np.float32,
            ),
        }

    def _get_info(self):
        """
        计算用于调试的辅助信息

        Returns:
            dict: 智能体当前消耗的总能量和当前运行时间
        """
        return {
            "current_energy_consumption": self.current_energy_consumption,
            "current_operation_time": self.current_operation_time,
            "docking_position": self.current_pos,
        }

    def _get_action_denormalized(self, action: float | np.floating) -> float:
        """将动作反归一化为列车加速度"""
        return float(
            action * (self.vehicle.max_acc + self.vehicle.max_dec) / 2
            + (self.vehicle.max_acc - self.vehicle.max_dec) / 2
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

        # 重新初始化运行状态
        self.current_pos = self.task.start_position
        self.current_speed = self.task.start_speed
        self.current_acc = 0.0
        self.current_operation_time = 0.0
        self.current_energy_consumption = 0.0
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.GetSlope(pos=self.current_pos)

        # 重置停车点步进
        self.current_sp = -1
        self.sps.Reset()

        self.current_min_speed, self.current_max_speed = (
            self.safeguardutil.GetMinAndMaxSpeed(
                current_pos=self.current_pos, current_sp=self.current_sp
            )
        )
        self.current_max_speed = min(
            self._get_ref_speed(self.current_pos),
            self.current_max_speed,
        )
        self.next_slope = self.trackprofile.GetSlope(
            pos=self.current_pos + self.max_step_distance
        )
        (
            self.next_min_speed,
            self.next_max_speed,
        ) = self.safeguardutil.GetMinAndMaxSpeed(
            current_pos=self.current_pos + self.max_step_distance,
            current_sp=self.current_sp,
        )
        self.next_max_speed = min(
            self._get_ref_speed(self.current_pos + self.max_step_distance),
            self.next_max_speed,
        )
        (
            self.current_latest_traction_intervation_point,
            self.current_latest_braking_intervention_point,
        ) = self.safeguardutil.GetLatestTranctionAndBrakingIntervationPoint(
            current_speed=self.current_speed, current_sp=self.current_sp
        )

        # 重置历史数据
        self._reset_history()

        # 重置轨迹数据
        if self.render_mode is not None:
            self._reset_trajectory()

        # 重置仿真步数
        self.current_steps = 0

        # 重置奖励记录
        self.rewards_info = {}

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):  # type: ignore
        """
        在环境中执行一个时间步

        Args:
            action: 需要执行的动作，即列车运行加速度

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self._record_history()

        self.current_acc = self._get_action_denormalized(action[0])

        # 根据当前速度大小、加速度、状态转移最大位移量
        # 计算转移至下一状态的速度大小、位移量和运行时间
        next_speed, distance, operation_time = self._update_motion()
        self.current_steps += 1

        # 计算当前列车在参考运行模式下的能耗和运行时间
        # ref_mec, ref_lec, ref_operation_time = self.ors.CalRefEnergyAndOperationTime(
        #     begin_pos=prev_pos,
        #     begin_speed=prev_speed,
        #     displacement=displacement,
        # )
        # ref_energy_consumption = ref_mec + ref_lec

        # 计算当前能耗
        current_mec, current_lec = self.ecc.CalcEnergy(
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
        self.current_slope = self.trackprofile.GetSlope(pos=self.current_pos)
        self.current_sp = self.sps.StepToNextSP(
            current_pos=self.current_pos,
            current_speed=self.current_speed,
            current_time=self.current_operation_time,
            current_sp=self.current_sp,
        )
        self.current_min_speed, self.current_max_speed = (
            self.safeguardutil.GetMinAndMaxSpeed(
                current_pos=self.current_pos,
                current_sp=self.current_sp,
            )
        )
        self.current_max_speed = min(
            self._get_ref_speed(self.current_pos),
            self.current_max_speed,
        )
        self.next_slope = self.trackprofile.GetSlope(
            pos=self.current_pos + self.max_step_distance * self.direction
        )
        self.next_min_speed, self.next_max_speed = self.safeguardutil.GetMinAndMaxSpeed(
            current_pos=self.current_pos + self.max_step_distance * self.direction,
            current_sp=self.current_sp,
        )
        self.next_max_speed = min(
            self._get_ref_speed(
                self.current_pos + self.max_step_distance * self.direction
            ),
            self.next_max_speed,
        )
        (
            self.current_latest_traction_intervation_point,
            self.current_latest_braking_intervention_point,
        ) = self.safeguardutil.GetLatestTranctionAndBrakingIntervationPoint(
            current_speed=self.current_speed, current_sp=self.current_sp
        )

        # 判断智能体是否到达目标区域
        terminated = (
            abs(self.task.target_position - self.current_pos)
            <= self.task.max_stop_error * 10
            and math.isclose(self.current_speed, 0.0, abs_tol=0.1)
        ) or self.current_steps == self.max_episode_steps

        # 若智能体违反安全约束，则截断训练进程
        truncated = (
            True
            if self.current_speed < self.current_min_speed
            or self.current_speed >= self.current_max_speed
            else False
        )

        # 计算奖励
        reward = self._get_reward(
            terminated=terminated,
            truncated=truncated,
        )

        if self.render_mode is not None:
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
        info = self._get_info()

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

        ds = np.diff(self.ref_curve_pos)
        speed_avg = 0.5 * (self.ref_curve_speed[1:] + self.ref_curve_speed[:-1])

        # 平均速度过小时将该段时间记为0
        dt = np.divide(
            ds,
            speed_avg,
            out=np.zeros_like(ds, dtype=np.float32),
            where=speed_avg > 1e-3,
        )

        ref_cum_time = np.empty_like(self.ref_curve_pos, dtype=np.float32)
        ref_cum_time[0] = 0.0
        ref_cum_time[1:] = np.cumsum(dt, dtype=np.float32)

        return ref_cum_time

    def _get_ref_speed(self, pos: float | np.floating):
        return max(0.0, self.ref_curve_interp_func(pos))

    def _get_reward(
        self,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        reward_total = 0.0

        if not truncated:
            if terminated:
                reward_total = self._get_reward_goal() + 10.0
            else:
                reward_total = self._get_reward_dense()
        else:
            progress = (
                abs(self.current_pos - self.task.start_position) / self.whole_distance
            )
            reward_total = -15.0 * (1.0 - progress) - max(
                0,
                self.current_min_speed - self.current_speed,
                self.current_speed - self.current_max_speed,
            )

        self.rewards_info["reward_total"] = reward_total

        return reward_total

    def _get_reward_dense(
        self,
    ) -> float:
        # 安全奖励
        reward_safety = self._get_reward_safety_dense()

        # 能耗奖励
        reward_energy = self._get_reward_energy_dense()

        # 舒适度奖励
        reward_comfort = self._get_reward_comfort()

        # 运行时间奖励
        reward_puncuality = self._get_reward_punctuality_dense()

        # 停站奖励
        reward_docking = self._get_reward_docking_dense()

        # 记录
        self.rewards_info["reward_safety"] = reward_safety
        self.rewards_info["reward_energy"] = reward_energy
        self.rewards_info["reward_comfort"] = reward_comfort
        self.rewards_info["reward_punctuality"] = reward_puncuality
        self.rewards_info["reward_docking"] = reward_docking

        return (
            reward_safety
            + reward_energy
            + reward_comfort
            + reward_puncuality
            + reward_docking
        )

    def _get_reward_safety_dense(self) -> float:
        # 计算当前状态势能
        phi_curr = self._potential_safety_speed(
            pos=self.current_pos,
            speed=self.current_speed,
            min_speed=self.current_min_speed,
            max_speed=self.current_max_speed,
            target_pos=self.sps.GetASATargetPointPosition(sp=self.current_sp)
            if self.current_sp >= 0
            else self.task.target_position,
        )

        # phi_curr = self._potential_safety_position(
        #     pos=self.current_pos,
        #     min_pos=self.current_latest_traction_intervation_point,
        #     max_pos=self.current_latest_braking_intervention_point,
        #     target_pos=self.sps.GetASATargetPointPosition(sp=self.current_sp)
        #     if self.current_sp >= 0
        #     else self.task.target_position,
        # )

        # 计算上个状态势能
        phi_prev = self._potential_safety_speed(
            pos=self.last_state["pos"],
            speed=self.last_state["speed"],
            min_speed=self.last_state["min_speed"],
            max_speed=self.last_state["max_speed"],
            target_pos=self.sps.GetASATargetPointPosition(
                sp=self.last_state["stopping_point_index"]
            )
            if self.last_state["stopping_point_index"] >= 0
            else self.task.target_position,
        )

        # phi_prev = self._potential_safety_position(
        #     pos=self.last_state["pos"],
        #     min_pos=self.last_state["latest_traction_intervation_point"],
        #     max_pos=self.last_state["latest_braking_intervation_point"],
        #     target_pos=self.sps.GetASATargetPointPosition(
        #         sp=self.last_state["stopping_point_index"]
        #     )
        #     if self.last_state["stopping_point_index"] >= 0
        #     else self.task.target_position,
        # )

        return self.gamma * phi_curr - phi_prev

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
        safe_margin = (max_speed - min_speed) / 2.0

        safe_margin = max(safe_margin, 1e-3)

        norm_speed = (speed - center_speed) / safe_margin

        # 靠近目标位置时，适当增大惩罚力度
        scale = 1.0 + 1.0 * np.exp(-0.002 * distanceToTarget)

        return -4.0 * scale * norm_speed**2

    def _potential_safety_position(
        self, pos: float, min_pos: float, max_pos: float, target_pos: float
    ):
        distanceToTarget = np.abs(target_pos - pos)

        center_pos = (max_pos + min_pos) / 2.0
        safe_margin = (max_pos - min_pos) / 2.0

        safe_margin = max(safe_margin, 1e-3)

        norm_pos = (pos - center_pos) / safe_margin

        scale = 1.0 + 1.0 * np.exp(-0.002 * distanceToTarget)

        return -10.0 * scale * (norm_pos**2)

    def _get_reward_energy_dense(self) -> float:
        return (
            -(
                (
                    self.current_energy_consumption
                    - self.last_state["energy_consumption"]
                )
                / self.max_energy_consumption
            )
            * 5
        )

    def _get_reward_comfort(self) -> float:
        delta_acc = abs(self.last_state["acc"] - self.current_acc)
        # 使用指数衰减式的钟形曲线
        norm_jerk = delta_acc / (self.task.max_acc_change)
        return -0.02 * (1 - np.exp(-2.0 * norm_jerk))

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
        min_remaining_operation_time = self.ors.CalcRefOperationTime(
            begin_pos=pos,
            begin_speed=speed,
            end_pos=self.task.target_position,
            end_speed=0.0,
        )

        # 计算实际剩余规划运行时间
        actual_remaining_operation_time = self.task.schedule_time - operation_time

        # 计算冗余运行时间
        redundant_operation_time = (
            actual_remaining_operation_time - min_remaining_operation_time
        )

        # 计算时间冗余度
        time_redundancy_norm = redundant_operation_time / self.task.schedule_time

        # 防止梯度爆炸, 对时间冗余度进行下界截断
        time_redundancy_norm_cliped = max(time_redundancy_norm, -0.3)

        return -2.0 * np.exp(-8.0 * time_redundancy_norm_cliped)

    def _get_reward_docking_dense(self):
        phi_curr = self._potential_docking(
            pos=self.current_pos, speed=self.current_speed
        )

        phi_prev = self._potential_docking(
            pos=self.last_state["pos"], speed=self.last_state["speed"]
        )

        return self.gamma * phi_curr - phi_prev

    def _potential_docking(self, pos: float, speed: float):
        sigma_d = 1000.0
        sigma_v = 40.0

        distance_error = self.task.target_position - pos

        distance_term = np.exp(-(distance_error**2) / (2.0 * sigma_d**2))
        speed_term = np.exp(-(speed**2) / (2.0 * sigma_v**2))

        return 2.0 * distance_term * speed_term

    def _get_reward_goal(
        self,
    ) -> float:
        reward_docking = self._get_reward_docking_goal() * 5
        reward_punctuality = self._get_reward_punctuality_goal() * 5

        self.rewards_info["reward_docking"] = reward_docking
        self.rewards_info["reward_punctuality"] = reward_punctuality

        return reward_docking + reward_punctuality

    def _get_reward_docking_goal(self) -> float:
        # 停站位置误差不超过0.3m
        docking_pos_error = abs(self.task.target_position - self.current_pos)

        return self._gaussian_kernel(A=2, B=-1, k=2.310490602, x=docking_pos_error)
        # return -abs(self.current_pos - self.task.target_position)

    def _get_reward_punctuality_goal(self) -> float:
        # 列车到站时刻误差不超过2min
        ontime_error = abs(self.task.schedule_time - self.current_operation_time)

        return self._gaussian_kernel(A=2, B=-1, k=0.005776227, x=ontime_error)
        # return -abs(self.remaining_schedule_time) / self.task.schedule_time

    def _gaussian_kernel(self, A: float, B: float, k: float, x: float) -> float:
        return A * np.exp(-k * x) + B

    def _record_history(self) -> None:
        self.last_state["pos"] = self.current_pos
        self.last_state["speed"] = self.current_speed
        self.last_state["acc"] = self.current_acc
        self.last_state["min_speed"] = self.current_min_speed
        self.last_state["max_speed"] = self.current_max_speed
        self.last_state["latest_traction_intervation_point"] = (
            self.current_latest_traction_intervation_point
        )
        self.last_state["latest_braking_intervation_point"] = (
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
            "latest_traction_intervation_point": 0.0,
            "latest_braking_intervation_point": 0.0,
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
        # 如果是初始状态或者没有运动，只记录当前点
        if abs(operation_time) < 1e-6 or abs(displacement) < 1e-6:
            self.trajectory_pos.append(pos)
            self.trajectory_speed_km_h.append(abs(speed * 3.6))
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
        self.trajectory_speed_km_h.extend(np.abs(vel_array * 3.6).tolist())

    def _reset_trajectory(self):
        """重置轨迹历史数据并记录初始状态"""
        self.trajectory_pos = [self.task.start_position]
        self.trajectory_speed_km_h = [abs(self.task.start_speed * 3.6)]

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
        SetChineseFont()
        # 仅在human模式下启用交互模式
        if mode == "human":
            plt.ion()

        # 创建图形窗口
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle("磁悬浮列车智能体训练过程", fontsize=14)

        # 设置坐标轴范围
        pos_margin = abs(self.task.target_position - self.current_pos) * 0.1
        self.ax.set_xlim(
            min(self.current_pos, self.task.target_position) - pos_margin,
            max(self.current_pos, self.task.target_position) + pos_margin,
        )
        self.ax.set_ylim(0, 600.0)  # 速度范围，单位：km/h

        # 绘制起点和终点
        self.ax.scatter(
            self.task.start_position,
            abs(self.task.start_speed * 3.6),
            marker="o",
            color="blue",
            s=60,
            alpha=0.8,
            label="起点",
            zorder=5,
        )
        self.ax.scatter(
            x=self.task.target_position,
            y=0.0,
            marker="o",
            color="red",
            s=60,
            alpha=0.8,
            label="终点",
            zorder=5,
        )

        # 绘制限速和危险速度域
        self.safeguardutil.Render(ax=self.ax)

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
        if len(self.trajectory_pos) > 1:
            self.traj_line.set_data(self.trajectory_pos, self.trajectory_speed_km_h)

    def close(self):
        """清理资源"""
        self._stop_animation()
        if self.fig is not None:
            if self.render_mode == "human":
                plt.ioff()
            plt.close(self.fig)
            self.fig = None
