from typing import Any

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.integrate import trapezoid

from model.SafeGuard import SafeGuardUtility
from model.Vehicle import Vehicle, VehicleDynamic
from model.Track import TrackProfile
from model.Task import Task


class MTTOEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        vehicle: Vehicle,
        trackprofile: TrackProfile,
        safeguard: SafeGuardUtility,
        task: Task,
        ds,
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

        # 列车历史轨迹
        self.history_pos = []
        self.history_speed = []

        # 动画配置
        self.animation_interval = 50  # 动画更新间隔（毫秒）

        # 检查依赖对象是否有必要的属性和方法
        if not all(
            hasattr(vehicle, attr)
            for attr in [
                "mass",
                "numoftrainsets",
                "length",
                "starting_velocity",
                "starting_position",
                "destination",
                "schedule_time",
                "levi_power",
            ]
        ):
            raise AttributeError("vehicle对象缺少必要的属性")

        # 磁浮列车运行优化（单）的车辆实例
        self.vehicle = vehicle

        # 磁浮列车运行优化（单）的线路实例
        self.trackprofile = trackprofile

        # 磁浮列车安全防护实例
        self.safeguard = safeguard

        # 运行任务约束
        self.task = task

        # 一个时间步仿真步长
        self.ds = ds

        # 初始化智能体
        # 包含：
        # - 当前位置，单位：m
        # - 剩余位移，单位：m
        # - 运行总距离
        # - 运动方向
        # - 当前车辆速度，单位：m/s
        # - 目标速度，单位：m/s
        # - 剩余规划运行时间，单位：s
        # - 列车质量（对智能体决策似乎不起作用），单位：t
        # - 当前坡度，百分位
        # - 下一坡度区间的坡度变化量，百分位
        # - 与下一坡度变化区间的距离，单位：m
        # - 当前限速，单位：m/s
        # - 下一限速区间的速度变化量，单位：m/s
        # - 与下一限速区间的距离，单位：m
        # - 当前时刻列车消耗总能量，单位：J
        self.current_pos = self.task.starting_position
        self.remainning_displacement = self.task.destination - self.current_pos
        self.whole_distance = abs(self.task.destination - self.current_pos)  # 常量
        self.direction = 1 if self.current_pos < self.task.destination else -1  # 常量
        self.current_velocity = self.task.starting_velocity
        self.goal_velocity = 0.0
        self.remainning_schedule_time = self.task.schedule_time
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.GetSlope(
            pos=self.current_pos, interpolate=True
        )

        self.next_slope, self.displacement_between_next_slope = (
            self.trackprofile.GetNextSlopeAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.current_vlimit = (
            self.trackprofile.GetSpeedlimit(pos=self.current_pos) * self.direction
        )

        self.next_vlimit, self.displacement_between_next_vlimit = (
            self.trackprofile.GetNextSpeedlimitAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.next_vlimit = self.next_vlimit * self.direction
        self.total_energy_consumption = 0.0

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
                "current_vlimit": gym.spaces.Box(
                    -600.0 / 3.6, 600 / 3.6, shape=(1,), dtype=np.float32
                ),
                "next_vlimit": gym.spaces.Box(
                    -600.0 / 3.6, 600.0 / 3.6, shape=(1,), dtype=np.float32
                ),
                "displacement_between_next_vlimit": gym.spaces.Box(
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

        # 定义智能体的动作空间
        self.action_space = gym.spaces.Box(
            -self.vehicle.max_dacc, self.vehicle.max_acc, shape=(1,), dtype=np.float32
        )

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
            "current_vlimit": np.array([self.current_vlimit], dtype=np.float32),
            "next_vlimit": np.array([self.next_vlimit], dtype=np.float32),
            "displacement_between_next_vlimit": np.array(
                [self.displacement_between_next_vlimit], dtype=np.float32
            ),
        }

    def _get_info(self):
        """
        计算用于调试的辅助信息

        Returns:
            dict: 智能体当前消耗的总能量
        """
        return {"total_energy_consumption": self.total_energy_consumption}

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
        self.remainning_schedule_time = self.task.schedule_time

        # 重置轨迹历史数据
        self.reset_trajectory_history()
        # self.mass = self.vehicle.mass
        self.current_slope = self.trackprofile.GetSlope(
            pos=self.current_pos, interpolate=False
        )
        self.next_slope, self.displacement_between_next_slope = (
            self.trackprofile.GetNextSlopeAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.current_vlimit = (
            self.trackprofile.GetSpeedlimit(self.current_pos) * self.direction
        )
        self.next_vlimit, self.displacement_between_next_vlimit = (
            self.trackprofile.GetNextSpeedlimitAndDistance(
                self.current_pos, self.direction
            )
        )
        self.next_vlimit = self.next_vlimit * self.direction
        self.total_energy_consumption = 0.0

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

        # 根据当前时刻速度、加速度、步长计算下一时刻速度、位移量和消耗时间
        v_next, displacement, time_consumption = self._update_motion(action[0])

        # 计算最大能耗
        maximum_energy_consumption = self.cal_energy_consumption(
            acc=1.0 * self.direction,
            displacement=displacement,
            travel_time=None,
        )

        # 计算当前能耗
        energy_consumption = self.cal_energy_consumption(
            acc=action[0], displacement=displacement, travel_time=time_consumption
        )

        # 更新智能体状态
        self.current_pos = self.current_pos + displacement
        self.current_velocity = v_next
        self.remainning_displacement = self.task.destination - self.current_pos
        self.remainning_schedule_time = self.remainning_schedule_time - time_consumption

        # 记录历史轨迹数据（传递运动参数以生成完整轨迹）
        self._record_trajectory_data(
            acc=action[0], displacement=displacement, travel_time=time_consumption
        )
        self.current_slope = self.trackprofile.GetSlope(
            pos=self.current_pos, interpolate=True
        )
        self.next_slope, self.displacement_between_next_slope = (
            self.trackprofile.GetNextSlopeAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.current_vlimit = (
            self.trackprofile.GetSpeedlimit(pos=self.current_pos) * self.direction
        )
        self.next_vlimit, self.displacement_between_next_vlimit = (
            self.trackprofile.GetNextSpeedlimitAndDistance(
                pos=self.current_pos, direction=self.direction
            )
        )
        self.next_vlimit = self.next_vlimit * self.direction
        self.total_energy_consumption = (
            self.total_energy_consumption + energy_consumption
        )

        # 判断智能体是否到达目标区域
        terminated = self.remainning_displacement * self.direction <= self.ds * 0.5

        # 判断智能体是否达到仿真步数上限
        truncated = True if self.remainning_schedule_time <= -600.0 else False

        # 计算奖励
        reward = self._get_reward(
            energy_consumption, maximum_energy_consumption, terminated
        )

        # 获取可观测状态和信息
        observation = self._get_obs()
        info = self._get_info()

        return (observation, reward, terminated, truncated, info)

    def _update_motion(self, acc: float):
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
                VehicleDynamic.CalLongitudinalForce(
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
            f_longitudinal = VehicleDynamic.CalLongitudinalForce(
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
        energy_consumption,
        maximum_energy_consumption,
        terminated,
    ) -> float:
        reward = 0.0

        # 安全奖励 (每步，非安全时负奖励为-1.0，安全时无奖励)
        safety_reward = self._get_safety_reward(self.current_velocity)

        # 能耗奖励 (每步，范围为[0, 1])
        # energy_reward = self._get_energy_reward(
        #     energy_consumption, maximum_energy_consumption
        # )

        if terminated:
            # 终止时的额外奖励
            docking_position_reward = self._get_docking_position_reward()
            docking_velocity_reward = self._get_docking_velocity_reward()
            punctuality_reward = self._get_punctuality_reward()

            reward = (
                safety_reward * 100.0  # 安全最重要
                # +energy_reward * 1.0  # 能耗次要
                + docking_position_reward * 30.0  # 停车精度重要
                + docking_velocity_reward * 59.0  # 停车速度重要
                + punctuality_reward * 10.0  # 准点性重要
            )
        else:
            # 目标接近奖励 (每步，范围为[0, 1])
            goal_approach_reward = self._get_goal_approach_reward()
            reward = (
                safety_reward * 100.0  # 安全最重要
                # + energy_reward * 1.0  # 能耗奖励
                + goal_approach_reward * 10.0
            )

        return np.clip(reward, -100.0, 100.0)  # 限制奖励范围
        # return reward

    def _get_safety_reward(self, v) -> float:
        if self.safeguard.DetectDanger(pos=self.current_pos, speed=abs(v * 3.6)):
            return -1.0  # 减少惩罚幅度
        else:
            return 0.0

    def _get_docking_position_reward(self) -> float:
        docking_error = abs(self.task.destination - self.current_pos)
        # if docking_error <= 5.0:  # 5米内为精确停车
        #     return 10.0 * (1.0 - docking_error / 50.0)
        # elif docking_error <= 10.0:  # 10米内为可接受
        #     return -1.0 * (docking_error - 50.0) / 50.0
        # else:  # 超过10米为不可接受
        #     return -5.0
        return 2.0 * np.exp(-0.006931472 * docking_error**2) - 1.0

    def _get_docking_velocity_reward(self) -> float:
        docking_error = abs(self.goal_velocity - self.current_velocity)
        # if docking_error <= 0.5:  # 0.5 m/s内为精确停车
        #     return 5.0 * (1.0 - docking_error / 0.5)
        # elif docking_error <= 2.0:  # 2 m/s内为可接受
        #     return -1.0 * (docking_error - 0.5) / 1.5
        # else:  # 超过2 m/s为不可接受
        #     return -3.0
        return 2.0 * np.exp(-0.173286795 * docking_error**2) - 1.0

    def _get_energy_reward(
        self, energy_consumption, maximum_energy_consumption
    ) -> float:
        if maximum_energy_consumption > 1e-6:
            energy_ratio = energy_consumption / maximum_energy_consumption
            return energy_ratio
        else:
            return 0.0

    def _get_punctuality_reward(self) -> float:
        time_error = abs(self.remainning_schedule_time)
        if time_error <= 30.0:  # 30秒内为准点
            return 5.0 * (1.0 - time_error / 30.0)
        elif time_error <= 120.0:  # 2分钟内为可接受
            return -1.0 * (time_error - 30.0) / 90.0
        else:  # 超过2分钟为不可接受
            return -3.0

    def _get_time_consumption_reward(self, time_consumption) -> float:
        """基于运行时间给予奖励"""

        # 与目标速度越接近，奖励越高
        return np.exp(-1 * time_consumption)

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

    def _record_trajectory_data(
        self, acc: float = 0.0, displacement: float = 0.0, travel_time: float = 0.0
    ):
        """记录完整的匀变速运动轨迹"""
        # 如果是初始状态或者没有运动，只记录当前点
        if abs(travel_time) < 1e-6 or abs(displacement) < 1e-6:
            self.history_pos.append(self.current_pos)
            self.history_speed.append(abs(self.current_velocity * 3.6))
            return

        # 计算上一个时刻的位置和速度
        prev_pos = self.current_pos - displacement
        prev_velocity = self.current_velocity - acc * travel_time

        # 生成中间轨迹点数量（基于运动时间动态调整）
        # 确保轨迹足够平滑，同时避免过多的点
        num_points = max(5, min(100, int(travel_time * 4)))  # 5-100个点之间

        # 生成时间序列
        time_steps = np.linspace(0, travel_time, num_points)

        # 向量化计算所有时间点的位置和速度（匀变速运动公式）
        # 位置：s = s0 + v0*t + 0.5*a*t^2
        pos_array = prev_pos + prev_velocity * time_steps + 0.5 * acc * time_steps**2
        # 速度：v = v0 + a*t
        vel_array = prev_velocity + acc * time_steps

        # 批量添加到历史记录中
        self.history_pos.extend(pos_array.tolist())
        self.history_speed.extend(np.abs(vel_array * 3.6).tolist())

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
        if len(self.history_pos) > 1:
            self.traj_line.set_data(self.history_pos, self.history_speed)

    def close(self):
        """清理资源"""
        self._stop_animation()
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None

    def reset_trajectory_history(self):
        """重置轨迹历史数据并记录初始状态"""
        self.history_pos = []
        self.history_speed = []
        # 记录初始状态点
        self._record_trajectory_data()
