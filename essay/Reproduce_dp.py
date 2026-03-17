import numpy as np
import json
import pickle
import os
import sys
from typing import TypedDict, Sequence
from numpy.typing import NDArray
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.SafeGuard import SafeGuardUtility
from model.Vehicle import Vehicle
from model.Track import Track, TrackProfile
from model.Task import Task
from model.ECC import ECC
from model.ORS import ORS
from utils.misc import SaveCurveAndMetrics


class OptimalSpeedProfile(TypedDict):
    pos: Sequence[float] | NDArray
    speed: Sequence[float] | NDArray
    total_time: float
    total_energy: float


class VariableSpacingDPOptimizer:
    """
    采用动态规划算法计算磁浮列车最优运行速度曲线

    1.内层动态规划
    _slove_dp_inner 接收运行时间的拉格朗日乘子, 执行一次二维变间距动态规划,
    并返回此时的最优解。

    2.外层二分法
    在动态规划算法的外层引入二分搜索循环, 根据内层计算出的实际最优运行时间,
    动态调整运行时间乘子, 直到运行时间逼近设定的规划运行时间。

    参考文献：
    [1] 赖晴鹰, 刘军, 赵若愚, 等. 基于变间距动态规划的中高速磁悬浮列车速度曲线优化[J].
    吉林大学学报（工学版）, 2019, 49(3): 749-756.
    [2] Lai Q, Liu J, Haghani A, et al. Optimal Energy Speed Profile of Medium-Speed
    Maglev Trains Integrating the Power Supply System and Train Control System[J].
    Transportation Research Record, 2020, 2674(Compendex): 729-738.

    """

    def __init__(
        self,
        vehicle: Vehicle,
        track: Track,
        safeguardutil: SafeGuardUtility,
        task: Task,
        time_tolerance: float,
    ) -> None:
        self.vehicle = vehicle
        self.track = track
        self.trackprofile = TrackProfile(track=self.track)
        self.safeguardutil = safeguardutil
        self.task = task
        self.time_tolerance = time_tolerance
        # self.direction = (
        #     1 if self.task.target_position >= self.task.start_position else -1
        # )
        self.ecc = ECC(
            R_m=0.2796,
            L_d=0.0002,
            R_k=50.0,
            L_k=0.000142,
            Tau=0.258,
            Psi_fd=3.9629,
            k_c=0.8,
        )
        self.ors = ORS(
            vehicle=self.vehicle, track=self.track, gamma=self.safeguardutil.gamma
        )
        # 计算最短运行时间参考曲线
        self.ref_curve_pos, self.ref_curve_speed = self.ors.CalcMinRuntimeCurve(
            begin_pos=self.task.start_position,
            begin_speed=self.task.start_speed,
            end_pos=self.task.target_position,
            end_speed=0.0,
        )

    def _get_ref_speed(self, pos: float) -> float:
        return max(0.0, np.interp(pos, self.ref_curve_pos, self.ref_curve_speed))

    def _generate_variable_spacing_stages(self, sub_stage_count: int = 30) -> NDArray:
        """
        基于临界点的变间距阶段划分
        将线路按照临界点划分为大分区, 每个大分区等分为 sub_stage_count 个子阶段
        """
        critical_points_position_arr = np.concatenate(
            (
                np.array([self.task.start_position]),
                self.safeguardutil.GetIDPPosition(),
                np.array([self.task.target_position]),
            )
        )
        stages = []

        for i in range(len(critical_points_position_arr) - 1):
            interval_start_pos = critical_points_position_arr[i]
            interval_end_pos = critical_points_position_arr[i + 1]
            # 对每个大区间进行等分，生成子阶段边界
            partition_stages = np.linspace(
                interval_start_pos, interval_end_pos, sub_stage_count + 1
            )
            if i == 0:
                stages.extend(partition_stages)
            else:
                stages.extend(partition_stages[1:])

        return np.array(stages)

    def _calculate_transition(
        self, pos_k: float, speed_k: float, displacement: float, speed_k_1: float
    ) -> tuple[bool, float, float]:
        distance_sample = np.linspace(
            0, displacement, max(10, int(np.abs(displacement) / 10.0))
        )
        pos_sample = pos_k + distance_sample
        k = (speed_k_1 - speed_k) / displacement
        speed_sample = k * distance_sample + speed_k

        # 检查是否进入危险速度域
        if self.safeguardutil.DetectDanger(pos=pos_sample, speed=speed_sample).all():
            return False, np.inf, np.inf

        # 检查列车是否静止不动
        if np.isclose(speed_k + speed_k_1, 0.0):
            return False, np.inf, np.inf

        acc_sample = k**2 * distance_sample + speed_k * k

        # 检查加速度是否超出预设范围
        if (
            np.max(acc_sample) > self.vehicle.max_acc
            or np.min(acc_sample) < -self.vehicle.max_dec
        ):
            return False, np.inf, np.inf

        # 引入极小正数补偿, 提供数值稳定性
        epsilon = 1e-3
        safe_speed_k = max(speed_k, epsilon)
        safe_speed_k_1 = max(speed_k_1, epsilon)
        safe_k = (safe_speed_k_1 - safe_speed_k) / displacement

        # 使用补偿后的安全速度和安全斜率来计算状态转移时间
        # if np.isclose(safe_speed_k, safe_speed_k_1):
        #     time = displacement / safe_speed_k
        # else:
        #     time = (np.log(safe_speed_k_1) - np.log(safe_speed_k)) / safe_k

        time = (
            displacement / safe_speed_k
            if np.isclose(safe_speed_k, safe_speed_k_1)
            else (np.log(safe_speed_k_1) - np.log(safe_speed_k)) / safe_k
        )

        propulsion_energy, leviation_energy = self.ecc.CalcEnergy(
            begin_pos=pos_k,
            begin_speed=speed_k,
            acc=lambda distance: k**2 * distance + speed_k * k,
            distance=abs(displacement),
            direction=1 if displacement > 0 else -1,
            operation_time=time,
            vehicle=self.vehicle,
            trackprofile=self.trackprofile,
        )

        return True, propulsion_energy + leviation_energy, time

    def _slove_dp_inner(
        self, lambda_time: float, max_speed: float, delta_speed: float
    ) -> OptimalSpeedProfile | None:
        """
        逆推法求解变间距动态规划, 结合目标函数及约束
        使用拉格朗日乘子法简化时间惩罚
        """
        stages = self._generate_variable_spacing_stages()
        total_steps = len(stages) - 1

        # 离散状态空间
        speed_states = np.arange(0, max_speed, delta_speed)
        num_speed_states = len(speed_states)

        # 初始化DP状态值表
        # 存储形式为 dict, 键为[状态索引], 值为[从该状态到终点状态的最小代价]
        # 代价为包含能耗以及转化为惩罚的到站时刻误差
        dp_cost = np.full(
            (total_steps + 1, num_speed_states), np.inf
        )  # 相当于状态值, 以代价的形式
        dp_time_accum = np.full(
            (total_steps + 1, num_speed_states), np.inf
        )  # 记录对应的累计时间
        dp_policy = np.zeros(
            (total_steps + 1, num_speed_states), dtype=int
        )  # 记录状态转移动作

        # 状态初始化
        dp_cost[total_steps, 0] = 0.0
        dp_time_accum[total_steps, 0] = 0.0

        print(f"开始使用逆推法求解，共{total_steps}个阶段")

        # 逆推法
        for k in range(total_steps - 1, -1, -1):
            pos_k = stages[k]
            delta_pos = stages[k + 1] - pos_k

            for i, speed_k in enumerate(speed_states):
                # 检查是否在最短运行时间速度曲线下方
                if speed_k > self._get_ref_speed(pos=pos_k):
                    break

                min_cost = np.inf
                best_next_j = -1
                best_time = np.inf

                # 遍历下一个阶段的所有可能速度状态
                for j, speed_next in enumerate(speed_states):
                    if np.isinf(dp_cost[k + 1, j]):
                        continue

                    # 阶段内的运动过程与能耗计算
                    is_valid, delta_energy, delta_time = self._calculate_transition(
                        pos_k=pos_k,
                        speed_k=speed_k,
                        displacement=delta_pos,
                        speed_k_1=speed_next,
                    )

                    if not is_valid:
                        continue

                    # 计算代价函数
                    transition_cost = delta_energy + lambda_time * delta_time
                    total_cost = transition_cost + dp_cost[k + 1, j]

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_next_j = j
                        best_time = delta_time + dp_time_accum[k + 1, j]

                dp_cost[k, i] = min_cost
                dp_policy[k, i] = best_next_j
                dp_time_accum[k, i] = best_time

        # 检验是否找到可行解
        if np.isinf(dp_cost[0, 0]):
            return None

        # 提取最优速度曲线
        optimal_speed_indices = [0]
        current_speed_idx = 0
        for k in range(total_steps):
            next_speed_idx = dp_policy[k, current_speed_idx]
            optimal_speed_indices.append(next_speed_idx)
            current_speed_idx = next_speed_idx

        optimal_speed_profile = [speed_states[idx] for idx in optimal_speed_indices]
        total_time = dp_time_accum[0, 0]
        total_energy = dp_cost[0, 0] - lambda_time * total_time  # 剥离时间惩罚

        return {
            "pos": stages.tolist(),
            "speed": optimal_speed_profile,
            "total_time": total_time,
            "total_energy": total_energy,
        }

    def optimize(
        self, max_speed: float, delta_speed: float, max_iters: int = 30
    ) -> OptimalSpeedProfile | None:
        """
        二分法调整运行时间乘子, 从而使结果逼近规定的运行时间
        """
        target_time = self.task.schedule_time
        print(
            f"开始双层寻优: 目标时间为{target_time:.2f}s, 时间误差容忍率为 {self.time_tolerance}"
        )

        lambda_min = 0.0
        lambda_max = 1e4
        best_result = None

        for iteration in range(max_iters):
            lambda_mid = (lambda_min + lambda_max) / 2.0
            print(
                f"第{iteration + 1}次迭代: 测试 lambda = {lambda_mid:.2f} ...", end=""
            )

            # 调用内层DP
            result = self._slove_dp_inner(
                lambda_time=lambda_mid, max_speed=max_speed, delta_speed=delta_speed
            )

            if result is None:
                print("未找到可行解, 请检查代码逻辑")
                break

            total_time = result["total_time"]
            total_energy = result["total_energy"]
            print(f"实际耗时为{total_time:.2f}s, 能耗为{total_energy:.2f}J")

            best_result = result

            # 判断时间误差率是否满足约束
            time_error_ratio = abs(total_time - target_time) / target_time
            if time_error_ratio <= self.time_tolerance:
                print(
                    f"成功收敛! 运行时间误差率为{time_error_ratio * 100:.4f}%, 满足精度要求"
                )
                break

            # 二分查找
            if total_time > target_time:
                # 耗时太长, 增大时间惩罚
                lambda_min = lambda_mid
            else:
                # 耗时太短, 减小时间惩罚
                lambda_max = lambda_mid

        if best_result is not None:
            print("最终优化结果")
            print(f"目标运行时间: {target_time:.2f}s")
            print(f"实际规划时间: {best_result['total_time']:.2f}s")
            print(f"最低运行能耗: {best_result['total_energy']:.2f}J")

        return best_result


if __name__ == "__main__":
    # 坡度，百分位
    with open("data/rail/raw/slopes.json", "r", encoding="utf-8") as f:
        slope_data = json.load(f)
        slopes = slope_data["slopes"]
        slope_intervals = slope_data["intervals"]

    # 区间限速
    with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
        speedlimit_data = json.load(f)
        speed_limits = speedlimit_data["speed_limits"]
        speed_limits = np.asarray(speed_limits) / 3.6
        speed_limit_intervals = speedlimit_data["intervals"]

    # 车站
    with open("data/rail/raw/stations.json", "r", encoding="utf-8") as f:
        stations_data = json.load(f)
        ly_zp = stations_data["LY"]["zp"]
        pa_zp = stations_data["PA"]["zp"]

    # 防护曲线
    with open("data/rail/safeguard/min_curves_list.pkl", "rb") as f:
        min_curves_list = pickle.load(f)
    with open("data/rail/safeguard/max_curves_list.pkl", "rb") as f:
        max_curves_list = pickle.load(f)

    factor: float = 0.99
    safeguardutility = SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=factor,
    )

    track = Track(slopes, slope_intervals, speed_limits.tolist(), speed_limit_intervals)

    vehicle = Vehicle(
        mass=317.5,
        numoftrainsets=5,
        length=128.5,
        max_acc=1.0,
        max_dec=1.0,
        levi_power_per_mass=1.7,
    )

    task = Task(
        start_position=ly_zp,
        start_speed=0.0,
        target_position=pa_zp,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error=120.0,
        max_stop_error=0.3,
    )

    VSDP = VariableSpacingDPOptimizer(
        vehicle=vehicle,
        track=track,
        safeguardutil=safeguardutility,
        task=task,
        time_tolerance=0.01,
    )

    result = VSDP.optimize(max_speed=500.0 / 3.6, delta_speed=1.0, max_iters=30)

    if result is not None:
        output_file = "data/operation/optimized_speed_curve.npz"
        saved_npz_path, saved_metrics_path = SaveCurveAndMetrics(
            pos_arr=result["pos"],
            speed_arr=result["speed"],
            output_path=output_file,
            metrics={
                "target_time_s": float(task.schedule_time),
                "total_time_s": float(result["total_time"]),
                "total_energy_j": float(result["total_energy"]),
                "start_position_m": float(task.start_position),
                "target_position_m": float(task.target_position),
            },
        )
        print(f"优化速度曲线已保存到: {saved_npz_path}")
        print(f"性能指标已保存到: {saved_metrics_path}")

        fig, ax = plt.subplots(figsize=(12, 7))

        # 绘制静态元素（区间限速、危险速度域和终点等）
        safeguardutility.Render(ax=ax)

        # 绘制起点
        ax.scatter(
            task.start_position,
            task.start_speed * 3.6,
            marker="o",
            color="green",
            s=100,
            alpha=0.8,
            label="起点",
            zorder=5,
            edgecolors="black",
            linewidths=1.5,
        )

        # 绘制终点
        ax.scatter(
            task.target_position,
            0.0,
            marker="o",
            color="red",
            s=100,
            alpha=0.8,
            label="终点",
            zorder=5,
            edgecolors="black",
            linewidths=1.5,
        )

        ax.plot(
            result["pos"],
            np.asarray(result["speed"]) * 3.6,
            label="最短运行时间曲线",
            color="blue",
            alpha=0.7,
            linewidth=2,
        )

        ax.set_xlim((0.0, 30000.0))
        ax.set_ylim((0.0, 500.0))
        ax.set_xlabel(r"里程$s\left( m \right)$")
        ax.set_ylabel(r"速度$v\left( km/h \right)$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
