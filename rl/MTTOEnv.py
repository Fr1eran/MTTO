from typing import Any

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.integrate import trapezoid

from model.SafeGuard import SafeGuardUtility
from model.Vehicle import Vehicle, VehicleDynamic
from model.Track import Track, TrackProfile
from model.Task import Task
from model.ORS import ORS
from utils.CalcEnergy import CalcEnergy


class MTTOEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        vehicle: Vehicle,
        track: Track,
        safeguardutil: SafeGuardUtility,
        task: Task,
        ds: float,
        render_mode: str | None = None,
        use_animation=False,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.use_animation = use_animation
        # 可视化时需要的绘图实例
        self.fig = None
        self.ax = None
        self.vehicle_dot, self.traj_line = None, None
        self.animation = None
        self.animation_running = False

        # 动画配置
        self.animation_interval = 50  # 动画更新间隔（毫秒）

        # 磁浮列车运行优化（单）的车辆实例
        self.vehicle = vehicle

        # 磁浮列车运行优化（单）的线路实例
        self.track = track
        self.trackprofile = TrackProfile(track=self.track)

        # 磁浮列车安全防护实例
        self.safeguard = safeguardutil

        # 运行任务约束
        self.task = task

        # 一个时间步仿真步长
        self.ds = ds

        # 参考运行系统
        self.ors = ORS(
            vehicle=self.vehicle,
            track=self.track,
            task=self.task,
            gamma=self.safeguard.gamma,
        )

        self.total_steps: int = 0

        # 最大仿真时间步
        self.max_episode_steps: int = (
            int(abs(self.task.destination - self.task.starting_position) / self.ds) + 5
        )

        # 列车历史状态
        self.history_pos: list[float] = [self.task.starting_position]
        self.history_velocity: list[float] = [self.task.starting_velocity]
        self.history_acc: list[float] = [0.0]

        # 列车运行轨迹
        self.trajectory_pos: list[float] = [self.task.starting_position]
        self.trajectory_speed_km_h: list[float] = [
            abs(self.task.starting_velocity * 3.6)
        ]

        # 初始化智能体
        # 包含:
        # - 当前位置, 单位:m
        # - 剩余位移, 单位: m
        # - 规划运行总距离, 单位: m
        # - 运动方向
        # - 当前车辆速度, 单位: m/s
        # - 目标速度, 单位: m/s
        # - 当前运行时间, 单位: s
        # - 剩余规划运行时间, 单位: s
        # - 列车质量（对智能体决策似乎不起作用）, 单位: t
        # - 当前坡度, 百分位
        # - 下一坡度区间的坡度变化量, 百分位
        # - 与下一坡度变化区间的距离, 单位: m
        # - 当前限速, 单位: m/s
        # - 下一限速区间的速度变化量, 单位: m/s
        # - 与下一限速区间的距离, 单位: m
        # - 当前时刻列车消耗总能量, 单位: J
        self.current_pos: float = self.task.starting_position
        self.remainning_displacement: float = self.task.destination - self.current_pos
        self.whole_distance: float = abs(
            self.task.destination - self.current_pos
        )  # 常量
        self.direction: int = (
            1 if self.current_pos < self.task.destination else -1
        )  # 常量
        self.current_velocity: float = self.task.starting_velocity
        self.goal_velocity: float = 0.0
        self.current_operation_time: float = 0.0
        self.remainning_schedule_time: float = self.task.schedule_time
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.GetSlope(
            pos=self.current_pos, interpolate=True
        )

        self.next_slope, self.displacement_between_next_slope = (
            self.trackprofile.GetNextSlopeAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.current_speed_limit = (
            self.trackprofile.GetSpeedlimit(pos=self.current_pos) * self.direction
        )

        self.next_speed_limit, self.displacement_between_next_speed_limit = (
            self.trackprofile.GetNextSpeedlimitAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.next_speed_limit = self.next_speed_limit * self.direction
        self.current_energy_consumption: float = 0.0

        self.starting_eculi_distance_pow2 = (
            self.task.destination - self.task.starting_position
        ) ** 2 + (self.goal_velocity - self.task.starting_velocity) ** 2

        # 定义智能体能够观测的状态信息
        self.observation_space = gym.spaces.Dict(
            {
                "agent_remainning_displacement": gym.spaces.Box(
                    -self.whole_distance,
                    self.whole_distance,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "agent_current_velocity": gym.spaces.Box(
                    -600.0 / 3.6, 600.0 / 3.6, shape=(1,), dtype=np.float32
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
                "next_slope": gym.spaces.Box(-4.0, 4.0, shape=(1,), dtype=np.float32),
                "displacement_between_next_slope": gym.spaces.Box(
                    -100000.0, 100000.0, shape=(1,), dtype=np.float32
                ),
                "current_speed_limit": gym.spaces.Box(
                    -600.0 / 3.6, 600 / 3.6, shape=(1,), dtype=np.float32
                ),
                "next_speed_limit": gym.spaces.Box(
                    -600.0 / 3.6, 600.0 / 3.6, shape=(1,), dtype=np.float32
                ),
                "displacement_between_next_speed_limit": gym.spaces.Box(
                    -100000.0, 100000.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        # 定义智能体能够观测的状态信息 (归一化后的空间)
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "agent_remainning_displacement": gym.spaces.Box(
        #             -1.0, 1.0, shape=(1,), dtype=np.float32
        #         ),
        #         "agent_current_speed": gym.spaces.Box(
        #             -1.0, 1.0, shape=(1,), dtype=np.float32
        #         ),
        #         "agent_remainning_schedule_time": gym.spaces.Box(
        #             -2.0,
        #             2.0,
        #             shape=(1,),
        #             dtype=np.float32,  # 允许超时或提前
        #         ),
        #         "current_slope": gym.spaces.Box(
        #             -1.0, 1.0, shape=(1,), dtype=np.float32
        #         ),
        #         "next_dslope": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        #         "displacement_between_next_dslope": gym.spaces.Box(
        #             -1.0, 1.0, shape=(1,), dtype=np.float32
        #         ),
        #         "current_vlimit": gym.spaces.Box(
        #             0.0, 1.0, shape=(1,), dtype=np.float32
        #         ),
        #         "next_dvlimit": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        #         "displacement_between_next_dvlimit": gym.spaces.Box(
        #             -1.0, 1.0, shape=(1,), dtype=np.float32
        #         ),
        #     }
        # )

        # 定义智能体的动作空间, 归一化
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        """
        将内部状态转换为可观测形式，并进行归一化
        可观测状态全部由np.ndarray组成

        Returns:
            dict: Observation with agent and target positions
        """
        # 归一化处理
        # remaining_disp_norm = np.clip(
        #     self.remainning_displacement / self.whole_distance, -1.0, 1.0
        # )
        # speed_norm = np.clip(self.current_velocity / (600.0 / 3.6), -1.0, 1.0)
        # time_norm = np.clip(
        #     self.remainning_schedule_time / self.vehicle.schedule_time, -2.0, 2.0
        # )
        # slope_norm = np.clip(self.current_slope / 4.0, -1.0, 1.0)
        # dslope_norm = np.clip(self.next_dslope / 8.0, -1.0, 1.0)
        # disp_dslope_norm = np.clip(
        #     self.displacement_between_next_dslope / self.whole_distance, -1.0, 1.0
        # )
        # vlimit_norm = np.clip(self.current_vlimit / (600.0 / 3.6), 0.0, 1.0)
        # dvlimit_norm = np.clip(self.next_dvlimit / (600.0 / 3.6), -1.0, 1.0)
        # disp_dvlimit_norm = np.clip(
        #     self.displacement_between_next_dvlimit / self.whole_distance, -1.0, 1.0
        # )

        # return {
        #     "agent_remainning_displacement": np.array(
        #         [remaining_disp_norm], dtype=np.float32
        #     ),
        #     "agent_current_speed": np.array([speed_norm], dtype=np.float32),
        #     "agent_remainning_schedule_time": np.array([time_norm], dtype=np.float32),
        #     "current_slope": np.array([slope_norm], dtype=np.float32),
        #     "next_dslope": np.array([dslope_norm], dtype=np.float32),
        #     "displacement_between_next_dslope": np.array(
        #         [disp_dslope_norm], dtype=np.float32
        #     ),
        #     "current_vlimit": np.array([vlimit_norm], dtype=np.float32),
        #     "next_dvlimit": np.array([dvlimit_norm], dtype=np.float32),
        #     "displacement_between_next_dvlimit": np.array(
        #         [disp_dvlimit_norm], dtype=np.float32
        #     ),
        # }

        return {
            "agent_remainning_displacement": np.array(
                [self.remainning_displacement], dtype=np.float32
            ),
            "agent_current_velocity": np.array(
                [self.current_velocity], dtype=np.float32
            ),
            "agent_remainning_schedule_time": np.array(
                [self.remainning_schedule_time], dtype=np.float32
            ),
            "current_slope": np.array([self.current_slope], dtype=np.float32),
            "next_slope": np.array([self.next_slope], dtype=np.float32),
            "displacement_between_next_slope": np.array(
                [self.displacement_between_next_slope], dtype=np.float32
            ),
            "current_speed_limit": np.array(
                [self.current_speed_limit], dtype=np.float32
            ),
            "next_speed_limit": np.array([self.next_speed_limit], dtype=np.float32),
            "displacement_between_next_speed_limit": np.array(
                [self.displacement_between_next_speed_limit], dtype=np.float32
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
        }

    def _get_action_denormalized(self, action: float | np.floating) -> float:
        """将动作反归一化为列车加速度"""
        return float(
            action * (self.vehicle.max_acc + self.vehicle.max_dacc) / 2
            + (self.vehicle.max_acc - self.vehicle.max_dacc) / 2
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

        # 将智能体的状态重设为初始状态
        self.current_pos = self.task.starting_position
        self.current_velocity = self.task.starting_velocity
        self.remainning_displacement = self.task.destination - self.current_pos
        self.current_operation_time = 0.0
        self.remainning_schedule_time = self.task.schedule_time

        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.GetSlope(
            pos=self.current_pos, interpolate=False
        )
        self.next_slope, self.displacement_between_next_slope = (
            self.trackprofile.GetNextSlopeAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.current_speed_limit = (
            self.trackprofile.GetSpeedlimit(self.current_pos) * self.direction
        )
        self.next_speed_limit, self.displacement_between_next_speed_limit = (
            self.trackprofile.GetNextSpeedlimitAndDistance(
                self.current_pos, self.direction
            )
        )
        self.next_speed_limit = self.next_speed_limit * self.direction
        self.current_energy_consumption = 0.0

        # 重置历史数据
        self._reset_history()
        # 重置轨迹数据
        if self.render_mode is not None:
            self._reset_trajectory()

        self.total_steps = 0

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
        prev_velocity = self.current_velocity
        current_acc = self._get_action_denormalized(action[0])

        self.total_steps += 1

        # 根据当前时刻速度、加速度、步长计算下一时刻速度、位移量和消耗时间
        v_next, displacement, operation_time = self._update_motion(current_acc)

        # 计算当前列车在参考运行模式下的能耗和运行时间
        ref_mec, ref_lec, ref_operation_time = self.ors.CalRefEnergyAndOperationTime(
            begin_pos=self.current_pos,
            begin_speed=self.current_velocity,
            displacement=displacement,
        )
        ref_energy_consumption = ref_mec + ref_lec

        # 计算当前能耗
        current_mec, current_lec = CalcEnergy(
            begin_pos=self.current_pos,
            begin_velocity=self.current_velocity,
            acc=action[0],
            displacement=displacement,
            operation_time=operation_time,
            vehicle=self.vehicle,
            trackprofile=self.trackprofile,
        )
        energy_consumption = current_mec + current_lec

        # 更新智能体状态
        self.current_pos = self.current_pos + displacement
        self.current_velocity = v_next
        self.remainning_displacement = self.task.destination - self.current_pos
        self.current_operation_time += operation_time
        self.remainning_schedule_time -= operation_time

        self.current_slope = self.trackprofile.GetSlope(
            pos=self.current_pos, interpolate=True
        )
        self.next_slope, self.displacement_between_next_slope = (
            self.trackprofile.GetNextSlopeAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.current_speed_limit = (
            self.trackprofile.GetSpeedlimit(pos=self.current_pos) * self.direction
        )
        self.next_speed_limit, self.displacement_between_next_speed_limit = (
            self.trackprofile.GetNextSpeedlimitAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.next_speed_limit = self.next_speed_limit * self.direction
        self.current_energy_consumption = (
            self.current_energy_consumption + energy_consumption
        )

        # 判断智能体是否到达目标区域
        terminated = abs(self.remainning_displacement) <= self.task.max_stop_error

        # 判断智能体是否达到仿真步数上限
        truncated = True if self.total_steps >= self.max_episode_steps else False

        # 计算奖励
        reward = self._get_reward(
            energy_consumption=energy_consumption,
            operation_time=operation_time,
            ref_energy_consumption=ref_energy_consumption,
            ref_operation_time=ref_operation_time,
            current_acc=current_acc,
            terminated=terminated | truncated,
        )

        self._record_history(
            prev_pos=prev_pos, prev_velocity=prev_velocity, acc=current_acc
        )
        if self.render_mode is not None:
            # 记录轨迹数据
            self._record_trajectory(
                prev_pos=prev_pos,
                prev_velocity=prev_velocity,
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
            tuple: (v_next, displacement, time_consumption)
        """

        displacement = self.ds * self.direction

        # 当加速度极小时，认为列车做匀速运动
        if abs(acc) < 1e-6:
            v_next = self.current_velocity
            if abs(v_next) < 1e-6:
                displacement = 0.0  # 无法运动
                travel_time = 0.0
                v_next = 0.0
            else:
                travel_time = displacement / v_next
            return v_next, displacement, travel_time

        # 一般情况下，列车做匀变速直线运动
        v_next_2 = self.current_velocity**2 + 2 * acc * displacement  # 速度大小的平方

        # 边界情况：减速为0
        if v_next_2 <= 0.0:
            v_next = 0.0
            displacement = -(self.current_velocity**2) / (2 * acc)
        else:
            v_next = np.sqrt(v_next_2) * self.direction

        travel_time = (v_next - self.current_velocity) / acc
        return v_next, displacement, travel_time

    def cal_energy_consumption(
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
                    speed=self.current_velocity,
                    slope=self.trackprofile.GetSlope(
                        self.current_pos, interpolate=True
                    ),
                )
            )
            MEC = abs(force * displacement)
        else:
            # 磁浮列车纵向力模型存在不光滑的区域
            # 故使用离散采样进行数值积分
            n_samples = max(10, int(abs(displacement) / 1.0))  # 自适应采样密度
            d_sample = np.linspace(0, displacement, n_samples, endpoint=False)
            s_sample = self.current_pos + d_sample

            speed_squared_sample = self.current_velocity**2 + 2 * acc * d_sample
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
            v_next_2 = self.current_velocity**2 + 2 * acc * displacement
            v_next = np.sqrt(v_next_2) * self.direction
            travel_time = (v_next - self.current_velocity) / acc

        assert travel_time is not None
        # 悬浮能耗
        LEC = travel_time * self.vehicle.levi_power_per_mass * self.vehicle.mass

        # 总能耗
        energy_consumption = MEC + LEC
        return energy_consumption

    def _get_reward(
        self,
        *,
        energy_consumption: float,
        operation_time: float,
        ref_energy_consumption: float,
        ref_operation_time: float,
        current_acc: float,
        terminated,
    ) -> float:
        reward = 0.0

        # 安全奖励 (每步，非安全时负奖励为-1.0，安全时无奖励)
        safety_reward = self._get_safety_reward()

        # 能耗奖励 (每步，范围为[0, 1])
        energy_reward = self._get_energy_reward(
            energy_consumption, ref_energy_consumption
        )

        # 舒适度奖励
        comfort_reward = self._get_comfort_reward(
            current_acc=current_acc, previous_acc=self.history_acc[-1]
        )

        if terminated:
            # 终止时的额外奖励
            docking_position_reward = self._get_docking_position_reward()
            docking_speed_reward = self._get_docking_speed_reward()
            punctuality_reward = self._get_punctuality_reward()

            reward = (
                safety_reward  # 安全最重要
                + energy_reward * 0.01
                + docking_position_reward * 0.33  # 停车位置
                + docking_speed_reward * 0.33  # 停车速度
                + punctuality_reward * 0.33  # 准点性
                + comfort_reward * 0.001  # 舒适度
            )
        else:
            # 目标接近奖励 (每步，范围为[0, 1])
            # goal_approach_reward = self._get_goal_approach_reward()
            operation_time_reward = self._get_operation_time_reward(
                operation_time=operation_time, ref_operation_time=ref_operation_time
            )
            reward = (
                safety_reward  # 安全最重要
                + energy_reward * 0.1  # 能耗
                + operation_time_reward * 0.01  # 运行时间
                + comfort_reward * 0.001  # 舒适度
                # + goal_approach_reward
            )

        # 奖励归一化
        return np.clip(reward, -1.0, 1.0)

    def _get_safety_reward(self) -> float:
        if self.safeguard.DetectDanger(
            pos=self.current_pos, speed=abs(self.current_velocity)
        ):
            return -1.0  # 减少惩罚幅度
        else:
            return 0.0

    def _get_docking_position_reward(self) -> float:
        # 停站位置误差不超过0.3m
        docking_pos_error = abs(self.task.destination - self.current_pos)
        A = 2
        B = -1
        k_D = 2.310490602
        return float(A * np.exp(-k_D * docking_pos_error**2) + B)

    def _get_docking_speed_reward(self) -> float:
        # 到站时因减速到10km/h(2.778m/s)内，
        docking_speed_error = abs(self.goal_velocity - self.current_velocity)
        return 2.0 * np.exp(-0.173286795 * docking_speed_error**2) - 1.0

    def _get_punctuality_reward(self) -> float:
        # 列车到站时刻误差不超过2min
        ontime_error = abs(self.remainning_schedule_time)
        A = 2
        B = -1
        k_T = 0.005776227
        return float(A * np.exp(-k_T * ontime_error) + B)

    def _get_energy_reward(self, energy_consumption, ref_energy_consumption) -> float:
        """基于能耗给予奖励"""
        if ref_energy_consumption > 1e-6:
            return (
                ref_energy_consumption - energy_consumption
            ) / ref_energy_consumption
        else:
            return 0.0

    def _get_operation_time_reward(
        self, operation_time: float, ref_operation_time: float
    ) -> float:
        """基于运行时间给予奖励

        设计原则：
        1. 运行时间 >= 参考时间：给予正奖励（安全保守策略）
        2. 运行时间越长，奖励越小（鼓励效率）
        3. 运行时间 < 参考时间：给予负奖励（过于激进，可能不安全）

        Returns:
            奖励值，范围约为 [-1, 1]
        """
        if ref_operation_time > 1e-6:
            # 计算时间比率
            time_ratio = operation_time / ref_operation_time

            if time_ratio < 1.0:
                # 运行时间小于参考时间，给予负奖励（线性惩罚）
                # time_ratio 越小（越快），惩罚越大
                return time_ratio - 1.0  # 范围: [-1, 0)
            else:
                # 运行时间大于等于参考时间，给予正奖励
                # 使用指数衰减函数，运行时间越长奖励越小
                # reward = exp(-k * (time_ratio - 1))
                # 当 time_ratio = 1.0 时，reward = 1.0
                # 当 time_ratio = 1.5 时，reward ≈ 0.37
                # 当 time_ratio = 2.0 时，reward ≈ 0.14
                k = 1.0  # 衰减系数，控制奖励下降速度
                return float(np.exp(-k * (time_ratio - 1.0)))
        else:
            return 0.0

    def _get_comfort_reward(self, current_acc: float, previous_acc: float) -> float:
        # 加速度变化不得大于0.75m/s^2
        delta_acc = abs(current_acc - previous_acc)
        A = 2
        B = -1
        k_C = 0.924196241
        return float(A * np.exp(-k_C * delta_acc) + B)

    def _get_goal_approach_reward(self) -> float:
        """基于当前状态给予目标接近奖励"""
        pos_reward = np.exp(
            -((self.remainning_displacement / self.whole_distance) ** 2)
        )
        target_velocity = self._get_target_velocity_by_displacement()
        vel_reward = np.exp(
            -(((self.current_velocity - target_velocity) / (600.0 / 3.6)) ** 2)
        )

        return 0.01 * pos_reward + 0.99 * vel_reward

    def _get_target_velocity_by_displacement(self):
        """根据剩余位移和时间计算目标速度"""
        if abs(self.remainning_schedule_time) < 1e-6:
            return 0.0
        else:
            return self.remainning_displacement / self.remainning_schedule_time

    def _record_history(
        self, prev_pos: float, prev_velocity: float, acc: float
    ) -> None:
        self.history_pos.append(prev_pos)
        self.history_velocity.append(prev_velocity)
        self.history_acc.append(acc)

    def _reset_history(self) -> None:
        self.history_pos = [self.task.starting_position]
        self.history_velocity = [self.task.starting_velocity]
        self.history_acc = [0.0]

    def _record_trajectory(
        self,
        prev_pos: float,
        prev_velocity: float,
        acc: float = 0.0,
        displacement: float = 0.0,
        operation_time: float = 0.0,
    ):
        """记录完整的匀变速运动轨迹"""
        # 如果是初始状态或者没有运动，只记录当前点
        if abs(operation_time) < 1e-6 or abs(displacement) < 1e-6:
            self.trajectory_pos.append(prev_pos)
            self.trajectory_speed_km_h.append(abs(prev_velocity * 3.6))
            return

        # 生成中间轨迹点数量（基于运动时间动态调整）
        # 确保轨迹足够平滑，同时避免过多的点
        num_points = max(5, min(100, int(operation_time * 4)))  # 5-100个点之间

        # 生成时间序列
        time_steps = np.linspace(0, operation_time, num_points)

        # 向量化计算所有时间点的位置和速度（匀变速运动公式）
        # 位置：s = s0 + v0*t + 0.5*a*t^2
        pos_array = prev_pos + prev_velocity * time_steps + 0.5 * acc * time_steps**2
        # 速度：v = v0 + a*t
        vel_array = prev_velocity + acc * time_steps

        # 批量添加到历史记录中
        self.trajectory_pos.extend(pos_array.tolist())
        self.trajectory_speed_km_h.extend(np.abs(vel_array * 3.6).tolist())

    def _reset_trajectory(self):
        """重置轨迹历史数据并记录初始状态"""
        self.trajectory_pos = [self.task.starting_position]
        self.trajectory_speed_km_h = [abs(self.task.starting_velocity * 3.6)]

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
            self._setup_figure()

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
            self.fig.canvas.draw()
            w, h = self.fig.canvas.get_width_height()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(  # type: ignore
                (h, w, 3)
            )
            return img

    def _setup_figure(self):
        """初始化绘图对象"""
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.ion()

        # 创建图形窗口
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle("磁悬浮列车智能体训练过程", fontsize=14)

        # 设置坐标轴范围
        pos_margin = abs(self.task.destination - self.current_pos) * 0.1
        self.ax.set_xlim(
            min(self.current_pos, self.task.destination) - pos_margin,
            max(self.current_pos, self.task.destination) + pos_margin,
        )
        self.ax.set_ylim(0, 600.0)  # 速度范围，单位：km/h

        # 绘制起点和终点
        self.ax.scatter(
            self.task.starting_position,
            abs(self.task.starting_velocity * 3.6),
            marker="o",
            color="blue",
            s=60,
            alpha=0.8,
            label="起点",
            zorder=5,
        )
        self.ax.scatter(
            x=self.task.destination,
            y=0.0,
            marker="o",
            color="red",
            s=60,
            alpha=0.8,
            label="终点",
            zorder=5,
        )

        # 绘制限速和危险速度域
        self.safeguard.render(ax=self.ax)

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
        self.vehicle_dot.set_data(
            [self.current_pos], [abs(self.current_velocity * 3.6)]
        )
        # 更新轨迹
        if len(self.trajectory_pos) > 1:
            self.traj_line.set_data(self.trajectory_pos, self.trajectory_speed_km_h)

    def close(self):
        """清理资源"""
        self._stop_animation()
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
