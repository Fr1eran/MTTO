from typing import Any
import logging
import math
import typing
import structlog
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy.integrate import trapezoid

from model.SafeGuard import SafeGuardUtility
from model.Vehicle import Vehicle, VehicleDynamic
from model.Track import Track, TrackProfile
from model.Task import Task
from model.ORS import ORS
from utils.CalcEnergy import CalcEnergy
from utils.misc import SetChineseFont


def setup_logger(logfile: str):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile, mode="w", encoding="utf-8"),
        ],
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(
            file=open(logfile, "a", encoding="utf-8")
        ),
    )
    return structlog.getLogger()


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
        # 初始化日志记录器
        self.logger = setup_logger("rl_training.jsonl")

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

        # 参考运行系统
        self.ors = ORS(
            vehicle=self.vehicle,
            track=self.track,
            task=self.task,
            gamma=self.safeguardutil.gamma,
        )

        # 计算最短运行时间参考曲线
        self.ref_curve_pos, self.ref_curve_speed = self.ors.CalMinRuntimeCurve(
            begin_pos=self.task.start_position,
            begin_speed=self.task.start_speed,
        )
        # 计算最大能耗和最短运行时间
        mec, lec, self.min_operation_time = self.ors.CalRefEnergyAndOperationTime(
            begin_pos=self.task.start_position,
            begin_speed=self.task.start_speed,
            displacement=self.task.target_position - self.task.start_position,
        )
        self.max_total_energy = mec + lec

        # 初始化运行状态
        # 包含:
        # - 当前位置, 单位: m
        # - 当前列车运行总时间
        # - 当前列车消耗总能量, 单位: J
        self.current_pos: float = self.task.start_position
        self.current_operation_time: float = 0.0
        self.current_energy_consumption: float = 0.0

        # 初始化智能体可观测状态
        # 包含:
        # - 当前位置到目标位置的总距离, 单位: m
        # - 当前运行速度大小, 单位: m/s
        # - 剩余规划运行时间, 单位: s
        # - 列车质量（对智能体决策似乎不起作用）, 单位: t
        # - 当前位置对应的坡度, 百分比制
        # - 当前位置对应的最大运行速度大小, 单位: m/s
        # - 当前位置对应的最小运行速度大小, 单位: m/s
        # - 下步状态转移后可能位置对应的坡度, 百分比制
        # - 下步状态转移后可能位置对应的最大运行速度大小, 单位: m/s
        # - 下步状态转移后可能位置对应的最小运行速度大小, 单位: m/s
        self.remainning_distance: float = abs(
            self.task.target_position - self.current_pos
        )
        self.current_speed: float = self.task.start_speed
        self.remainning_schedule_time: float = self.task.schedule_time
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.GetSlope(
            pos=self.current_pos, interpolate=False
        )
        self.current_min_speed, self.current_max_speed, self.current_sp = (
            self.safeguardutil.GetMinAndMaxSpeed(
                current_pos=self.current_pos,
                current_speed=self.current_speed,
                current_sp=None,
            )
        )
        self.current_max_speed = min(
            self._get_ref_speed(self.current_pos),
            self.current_max_speed,
        )
        self.next_slope = self.trackprofile.GetSlope(
            pos=self.current_pos + self.max_step_distance * self.direction,
            interpolate=False,
        )
        self.next_min_speed, self.next_max_speed, _ = (
            self.safeguardutil.GetMinAndMaxSpeed(
                current_pos=self.current_pos + self.max_step_distance * self.direction,
                current_speed=self.current_speed,
                current_sp=self.current_sp,
            )
        )
        self.next_max_speed = min(
            self._get_ref_speed(
                self.current_pos + self.max_step_distance * self.direction
            ),
            self.next_max_speed,
        )

        # 定义智能体能够观测的状态信息
        self.observation_space = gym.spaces.Dict(
            {
                "agent_remainning_distance": gym.spaces.Box(
                    0.0,
                    self.whole_distance,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "agent_current_speed": gym.spaces.Box(
                    0.0, 600.0 / 3.6, shape=(1,), dtype=np.float32
                ),
                "agent_remainning_schedule_time": gym.spaces.Box(
                    -600.0,  # 最多超时10分钟
                    self.task.schedule_time,
                    shape=(1,),
                    dtype=np.float32,  # 允许超时或提前
                ),
                "current_slope": gym.spaces.Box(
                    -4.0, 4.0, shape=(1,), dtype=np.float32
                ),
                "current_max_speed": gym.spaces.Box(
                    0.0, 600 / 3.6, shape=(1,), dtype=np.float32
                ),
                "current_min_speed": gym.spaces.Box(
                    0.0, 600 / 3.6, shape=(1,), dtype=np.float32
                ),
                "next_slope": gym.spaces.Box(-4.0, 4.0, shape=(1,), dtype=np.float32),
                "next_max_speed": gym.spaces.Box(
                    0.0, 600 / 3.6, shape=(1,), dtype=np.float32
                ),
                "next_min_speed": gym.spaces.Box(
                    0.0, 600 / 3.6, shape=(1,), dtype=np.float32
                ),
            }
        )

        # 定义智能体的动作空间, 归一化
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # 当前训练步数
        self.current_steps: int = 0

        # 当前步进停车点编号
        self.current_sp: int = -1

        # Q初始值
        self.q_init: float = 0.0

        # 最大转移步数
        self.max_episode_steps: int = math.ceil(
            abs(self.task.target_position - self.current_pos) / self.max_step_distance
        )

        # 训练历史
        self.history_pos: list[float] = [self.current_pos]
        self.history_speed: list[float] = [self.current_speed]
        self.history_acc: list[float] = [0.0]
        self.dense_rewards: list[float] = [0.0]

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
            "agent_remainning_distance": np.array(
                [self.remainning_distance], dtype=np.float32
            ),
            "agent_current_speed": np.array([self.current_speed], dtype=np.float32),
            "agent_remainning_schedule_time": np.array(
                [self.remainning_schedule_time], dtype=np.float32
            ),
            "current_slope": np.array([self.current_slope], dtype=np.float32),
            "current_max_speed": np.array([self.current_max_speed], dtype=np.float32),
            "current_min_speed": np.array([self.current_min_speed], dtype=np.float32),
            "next_slope": np.array([self.next_slope], dtype=np.float32),
            "next_max_speed": np.array([self.next_max_speed], dtype=np.float32),
            "next_min_speed": np.array([self.next_min_speed], dtype=np.float32),
        }

    def _get_info(self):
        """
        计算用于调试的辅助信息

        Returns:
            dict: 智能体当前消耗的总能量和当前运行时间
        """
        return {
            "current_pos": self.current_pos,
            "current_energy_consumption": self.current_energy_consumption,
            "current_operation_time": self.current_operation_time,
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
        self.current_operation_time = 0.0
        self.current_energy_consumption = 0.0

        # 重新初始化智能体可观测状态
        self.remainning_distance = abs(self.task.target_position - self.current_pos)
        self.current_speed = self.task.start_speed
        self.remainning_schedule_time = self.task.schedule_time
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.GetSlope(
            pos=self.current_pos, interpolate=False
        )
        self.current_min_speed, self.current_max_speed, self.current_sp = (
            self.safeguardutil.GetMinAndMaxSpeed(
                current_pos=self.current_pos,
                current_speed=self.current_speed,
                current_sp=None,
            )
        )
        self.current_max_speed = min(
            self._get_ref_speed(self.current_pos),
            self.current_max_speed,
        )
        self.next_slope = self.trackprofile.GetSlope(
            pos=self.current_pos + self.max_step_distance, interpolate=False
        )
        self.next_min_speed, self.next_max_speed, _ = (
            self.safeguardutil.GetMinAndMaxSpeed(
                current_pos=self.current_pos + self.max_step_distance,
                current_speed=self.current_speed,
                current_sp=self.current_sp,
            )
        )
        self.next_max_speed = min(
            self._get_ref_speed(self.current_pos + self.max_step_distance),
            self.next_max_speed,
        )

        # 重置历史数据
        self._reset_history()
        # 重置轨迹数据
        if self.render_mode is not None:
            self._reset_trajectory()

        # 重置仿真步数
        self.current_steps = 0

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
        prev_pos = self.current_pos
        prev_speed = self.current_speed
        current_acc = self._get_action_denormalized(action[0])

        # 根据当前速度大小、加速度、状态转移最大位移量
        # 计算转移至下一状态的速度大小、位移量和运行时间
        next_speed, displacement, operation_time = self._update_motion(current_acc)
        self.current_steps += 1

        # 计算当前列车在参考运行模式下的能耗和运行时间
        # ref_mec, ref_lec, ref_operation_time = self.ors.CalRefEnergyAndOperationTime(
        #     begin_pos=prev_pos,
        #     begin_speed=prev_speed,
        #     displacement=displacement,
        # )
        # ref_energy_consumption = ref_mec + ref_lec

        # 计算当前能耗
        current_mec, current_lec = CalcEnergy(
            begin_pos=prev_pos,
            begin_speed=prev_speed,
            acc=current_acc,
            displacement=displacement,
            operation_time=operation_time,
            vehicle=self.vehicle,
            trackprofile=self.trackprofile,
        )
        energy_consumption = current_mec + current_lec

        # 更新运行状态
        self.current_pos += displacement
        self.current_operation_time += operation_time
        self.current_energy_consumption += energy_consumption

        # 更新智能体可观测状态
        self.remainning_distance -= displacement * self.direction
        self.current_speed = next_speed
        self.remainning_schedule_time -= operation_time
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.GetSlope(
            pos=self.current_pos, interpolate=True
        )
        self.current_min_speed, self.current_max_speed, self.current_sp = (
            self.safeguardutil.GetMinAndMaxSpeed(
                current_pos=self.current_pos,
                current_speed=self.current_speed,
                current_sp=self.current_sp,
            )
        )
        self.current_max_speed = min(
            self._get_ref_speed(self.current_pos),
            self.current_max_speed,
        )
        self.next_slope = self.trackprofile.GetSlope(
            pos=self.current_pos + self.max_step_distance * self.direction,
            interpolate=False,
        )
        self.next_min_speed, self.next_max_speed, _ = (
            self.safeguardutil.GetMinAndMaxSpeed(
                current_pos=self.current_pos + self.max_step_distance * self.direction,
                current_speed=self.current_speed,
                current_sp=self.current_sp,
            )
        )
        self.next_max_speed = min(
            self._get_ref_speed(
                self.current_pos + self.max_step_distance * self.direction
            ),
            self.next_max_speed,
        )

        # 判断智能体是否到达目标区域
        terminated = (
            self.remainning_distance <= self.task.max_stop_error
            or self.current_steps == self.max_episode_steps
        )

        # 若智能体违反安全约束，则截断训练进程
        truncated = (
            True
            if self.current_speed < self.current_min_speed
            or self.current_speed > self.current_max_speed
            else False
        )

        # 计算奖励
        reward = self._get_reward(
            prev_pos=prev_pos,
            prev_speed=prev_speed,
            prev_acc=self.history_acc[-1],
            current_pos=self.current_pos,
            current_speed=self.current_speed,
            current_acc=current_acc,
            energy_consumption=energy_consumption,
            operation_time=operation_time,
            terminated=terminated,
            truncated=truncated,
        )

        self._record_history(prev_pos=prev_pos, prev_speed=prev_speed, acc=current_acc)
        if self.render_mode is not None:
            # 记录轨迹数据
            self._record_trajectory(
                prev_pos=prev_pos,
                prev_speed=prev_speed,
                acc=current_acc,
                displacement=displacement,
                operation_time=operation_time,
            )

        # 获取可观测状态和信息
        observation = self._get_obs()
        info = self._get_info()

        return (observation, reward, terminated, truncated, info)

    def _update_motion(self, acc: float) -> tuple[float, float, float]:
        """
        磁浮列车匀变速直线运动仿真

        Args:
            acc(float): 加速度(m/s^2)

        Returns:
            tuple: (next_speed, displacement, operation_time)
        """

        distance = self.max_step_distance

        # 当加速度极小时，认为列车做匀速运动
        if abs(acc) < 1e-6:
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
                self.current_speed**2 + 2 * acc * distance
            )  # 速度大小的平方

            if next_speed_squared < 1e-6:
                # 边界情况：减速为0
                next_speed = 0.0
                distance = -(self.current_speed**2) / (2 * acc)
            else:
                next_speed = np.sqrt(next_speed_squared)

            operation_time = (next_speed - self.current_speed) / acc

        displacement = distance * self.direction

        return next_speed, displacement, operation_time

    def _cal_energy_consumption(
        self, acc: float, displacement: float, travel_time: float | None
    ):
        """
        计算系统以一个仿真位移步运动消耗的总能量，包括驱动力做的总功和悬浮能耗

        Args:
            acc(float): 加速度(m/s^2)
            displacement(float): 位移(m)
            travel_time(float): 运行时间(s)

        Returns:
            energy_consumption: 总能耗(J)
        """

        # 处理极小位移的情况，避免数值计算问题
        if abs(displacement) < 1e-6:
            # 对于极小位移，使用起始点的力值近似
            force = abs(
                VehicleDynamic.CalcLongitudinalForce(
                    vehicle=self.vehicle,
                    acc=acc,
                    speed=self.current_speed,
                    slope=self.trackprofile.GetSlope(
                        self.current_pos, interpolate=True
                    ),
                )
            )
            MEC = abs(float(force) * displacement)
        else:
            # 磁浮列车纵向力模型存在不光滑的区域
            # 故使用离散采样进行数值积分
            n_samples = max(10, int(abs(displacement) / 1.0))  # 自适应采样密度
            d_sample = np.linspace(0, displacement, n_samples, endpoint=False)
            s_sample = self.current_pos + d_sample

            speed_squared_sample = self.current_speed**2 + 2 * acc * d_sample
            speed_sample = np.sqrt(speed_squared_sample)

            # 计算每个采样点的纵向力
            f_longitudinal = VehicleDynamic.CalcLongitudinalForce(
                vehicle=self.vehicle,
                acc=acc,
                speed=speed_sample,
                slope=self.trackprofile.GetSlope(s_sample, interpolate=True),
            )

            # 使用梯形法则计算机械能耗（力的绝对值乘以距离）
            MEC = trapezoid(y=np.abs(f_longitudinal), x=np.abs(d_sample))

        if travel_time is None:
            v_next_2 = self.current_speed**2 + 2 * acc * displacement
            v_next = np.sqrt(v_next_2) * self.direction
            travel_time = (v_next - self.current_speed) / acc

        assert travel_time is not None
        # 悬浮能耗
        LEC = travel_time * self.vehicle.levi_power_per_mass * self.vehicle.mass

        # 总能耗
        energy_consumption = float(MEC) + LEC
        return energy_consumption

    def _get_reward(
        self,
        *,
        prev_pos: float,
        prev_speed: float,
        prev_acc: float,
        current_pos: float,
        current_speed: float,
        current_acc: float,
        energy_consumption: float,
        operation_time: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        self.logger.info(
            "train",
            current_steps=self.current_steps,
            current_pos=current_pos,
            current_speed=current_speed,
            current_acc=current_acc,
        )
        reward = 0.0
        if not truncated:
            if terminated:
                reward = self._get_goal_reward(
                    current_pos=current_pos, current_speed=current_speed
                )
            else:
                dense_reward = self._get_dense_reward(
                    current_pos=current_pos,
                    current_speed=current_speed,
                    energy_consumption=energy_consumption,
                    operation_time=operation_time,
                    current_acc=current_acc,
                )
                # reward = dense_reward + self._get_PBRS_reward(
                #     prev_pos=prev_pos,
                #     prev_speed=prev_speed,
                #     current_pos=current_pos,
                #     current_speed=current_speed,
                #     prev_dense_reward=self.dense_rewards[-1],
                #     current_dense_reward=dense_reward,
                # )
                self.dense_rewards.append(dense_reward)
                reward = dense_reward

        return reward

    def _get_dense_reward(
        self,
        current_pos: float,
        current_speed: float,
        energy_consumption: float,
        operation_time: float,
        current_acc: float,
    ) -> float:
        # 安全奖励 (只要智能体训练没被截断，则持续获得小的正奖励)
        safety_reward = 10 / self.max_episode_steps

        # 能耗奖励 (范围为[-1, 0])
        # energy_reward = self._get_energy_reward_per_step(energy_consumption)

        # 舒适度奖励 (范围为[-1, 1])
        # comfort_reward = self._get_comfort_reward(
        #     current_acc=current_acc, previous_acc=self.history_acc[-1]
        # )

        # 运行时间奖励 (范围为[-1, 0])
        # operation_time_reward = self._get_operation_time_reward_per_step(
        #     operation_time=operation_time
        # )

        self.logger.info(
            "dense_reward",
            safety_reward=safety_reward,
            # energy_reward=energy_reward,
            # operation_time_reward=operation_time_reward,
            # comfort_reward=comfort_reward * 0.005,
        )

        return (
            safety_reward
            # + energy_reward + operation_time_reward
            # + comfort_reward * 0.005
        )

    def _get_goal_reward(
        self,
        current_pos: float,
        current_speed: float,
    ) -> float:
        docking_position_reward = self._get_docking_position_reward(
            current_pos=current_pos
        )
        punctuality_reward = self._get_punctuality_reward()
        energy_reward = self._get_total_energy_reward()

        self.logger.info(
            "goal_reward",
            docking_position_reward=docking_position_reward,
            punctuality_reward=punctuality_reward * 10,
            energy_reward=energy_reward * 10,
        )

        return docking_position_reward + punctuality_reward * 10 + energy_reward * 10

    def _get_safety_reward(self, current_pos, current_speed) -> float:
        if self.safeguardutil.DetectDanger(
            pos=current_pos, speed=current_speed
        ) or self._detect_ref_speed_exceed(pos=current_pos, speed=current_speed):
            return -1.0  # 减少惩罚幅度
        else:
            return 0

    def _get_ref_speed(self, pos: float | np.floating):
        return max(0.0, np.interp(pos, self.ref_curve_pos, self.ref_curve_speed))

    def _detect_ref_speed_exceed(
        self, pos: float | np.floating, speed: float | np.floating
    ):
        speed_limit = self._get_ref_speed(pos=pos)

        return speed > speed_limit

    def _get_docking_position_reward(self, current_pos) -> float:
        # 停站位置误差不超过0.3m
        # docking_pos_error = abs(self.task.target_position - current_pos)
        # A = 2
        # B = -1
        # k_D = 2.310490602
        # return float(A * np.exp(-k_D * docking_pos_error**2) + B)
        return current_pos - self.task.target_position

    def _get_punctuality_reward(self) -> float:
        # 列车到站时刻误差不超过2min
        # ontime_error = abs(self.remainning_schedule_time)
        # A = 2
        # B = -1
        # k_T = 0.005776227
        # return float(A * np.exp(-k_T * ontime_error) + B)
        return abs(self.remainning_schedule_time) / self.task.schedule_time

    # def _get_energy_reward(self, energy_consumption, ref_energy_consumption) -> float:
    #     """基于能耗给予奖励"""
    #     if ref_energy_consumption > 1e-6:
    #         return (
    #             ref_energy_consumption - energy_consumption
    #         ) / ref_energy_consumption
    #     else:
    #         return 0.0

    def _get_energy_reward_per_step(self, energy_consumption: float) -> float:
        """基于能耗给予奖励"""
        return -(energy_consumption / self.max_total_energy) * 10

    def _get_total_energy_reward(self) -> float:
        return (
            self.max_total_energy - self.current_energy_consumption
        ) / self.max_total_energy

    # def _get_operation_time_reward(
    #     self, operation_time: float, ref_operation_time: float
    # ) -> float:
    #     """基于运行时间给予奖励

    #     设计原则：
    #     1. 运行时间 >= 参考时间：给予正奖励（安全保守策略）
    #     2. 运行时间越长，奖励越小（鼓励效率）
    #     3. 运行时间 < 参考时间：给予负奖励（过于激进，可能不安全）

    #     Returns:
    #         奖励值，范围约为 [-1, 1]
    #     """
    #     if ref_operation_time > 1e-6:
    #         # 计算时间比率
    #         time_ratio = operation_time / ref_operation_time

    #         if time_ratio > 1.0 or math.isclose(time_ratio, 1.0, rel_tol=1e-6):
    #             # 运行时间大于等于参考时间，给予正奖励
    #             # 使用指数衰减函数，运行时间越长奖励越小
    #             # reward = exp(-k * (time_ratio - 1))
    #             # 当 time_ratio = 1.0 时，reward = 1.0
    #             # 当 time_ratio = 1.5 时，reward ≈ 0.37
    #             # 当 time_ratio = 2.0 时，reward ≈ 0.14
    #             k = 1.0  # 衰减系数，控制奖励下降速度
    #             return float(np.exp(-k * (time_ratio - 1.0)))
    #         else:
    #             # 运行时间小于参考时间，给予负奖励（线性惩罚）
    #             # time_ratio 越小（越快），惩罚越大
    #             return time_ratio - 1.0  # 范围: [-1, 0)

    #     else:
    #         return 0.0

    def _get_operation_time_reward_per_step(
        self,
        operation_time: float,
    ) -> float:
        """
        基于运行时间给予奖励

        Returns:
            奖励值，范围约为 [-1, 0]
        """
        return -(operation_time / self.min_operation_time)

    def _get_comfort_reward(
        self, prev_acc: float, current_acc: float, operation_time: float
    ) -> float:
        # 加速度变化不得大于0.75m/s^2
        # delta_acc = abs(current_acc - prev_acc)
        # A = 2
        # B = -1
        # k_C = 0.924196241
        # return float(A * np.exp(-k_C * delta_acc) + B)
        if math.isclose(operation_time, 0.0):
            return 0.0
        else:
            delta_acc = abs(current_acc - prev_acc) / operation_time
            if delta_acc > 0.75:
                return -0.1
            else:
                return 0.1

    def _get_PBRS_reward(
        self,
        prev_pos: float,
        prev_speed: float,
        current_pos: float,
        current_speed: float,
        prev_dense_reward: float,
        current_dense_reward: float,
    ) -> float:
        Phi_c = (
            self._potential(prev_pos, prev_speed)
            # + (1 - self.gamma) * self.q_init
            # - current_dense_reward
        )
        Phi_p = (
            self._potential(current_pos, current_speed)
            # + (1 - self.gamma) * self.q_init
            # - prev_dense_reward
        )
        # Phi_c = np.exp(Phi_c)
        # Phi_p = np.exp(Phi_p)
        self.logger.info("PBRS", Phi_c=Phi_c, Phi_p=Phi_p)
        self.logger.info("PBRS", PBRS_reward=self.gamma * Phi_c - Phi_p)
        return self.gamma * Phi_c - Phi_p

    def _potential(self, pos: float, speed: float) -> float:
        ref_speed = self._get_ref_speed(pos=pos)
        return float(0.1 * (ref_speed - speed))

    def _record_history(self, prev_pos: float, prev_speed: float, acc: float) -> None:
        self.history_pos.append(prev_pos)
        self.history_speed.append(prev_speed)
        self.history_acc.append(acc)

    def _reset_history(self) -> None:
        self.history_pos = [self.task.start_position]
        self.history_speed = [self.task.start_speed]
        self.history_acc = [0.0]
        self.dense_rewards = [0.0]

    def _record_trajectory(
        self,
        prev_pos: float,
        prev_speed: float,
        acc: float = 0.0,
        displacement: float = 0.0,
        operation_time: float = 0.0,
    ):
        """记录完整的匀变速运动轨迹"""
        # 如果是初始状态或者没有运动，只记录当前点
        if abs(operation_time) < 1e-6 or abs(displacement) < 1e-6:
            self.trajectory_pos.append(prev_pos)
            self.trajectory_speed_km_h.append(abs(prev_speed * 3.6))
            return

        # 生成中间轨迹点数量（基于运动时间动态调整）
        # 确保轨迹足够平滑，同时避免过多的点
        num_points = max(5, min(100, int(operation_time * 4)))  # 5-100个点之间

        # 生成时间序列
        time_steps = np.linspace(0, operation_time, num_points)

        # 向量化计算所有时间点的位置和速度（匀变速运动公式）
        # 位置：s = s0 + v0*t + 0.5*a*t^2
        pos_array = prev_pos + prev_speed * time_steps + 0.5 * acc * time_steps**2
        # 速度：v = v0 + a*t
        vel_array = prev_speed + acc * time_steps

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
            canvas = typing.cast(FigureCanvasAgg, self.fig.canvas)
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
