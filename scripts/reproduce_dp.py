import numpy as np
import os
from typing import TypedDict, Sequence, Any, Iterable
from numpy.typing import NDArray
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from model.ocs import SafeGuardUtility, TrainService
from model.vehicle import VehicleInfo
from model.track import TrackInfo, TrackProfile
from model.common import ECC, ORS
from utils.io_utils import save_curve_and_metrics
from utils.plot_utils import set_chinese_font
from utils.data_loader import (
    load_slopes,
    load_safeguard_curves,
    load_speed_limits,
    load_stations,
)


class OptimalSpeedProfile(TypedDict):
    pos: Sequence[float] | NDArray
    speed: Sequence[float] | NDArray
    total_time: float
    total_energy: float


class VariableSpacingDPOptimizer:
    """
    采用动态规划算法计算磁浮列车最优运行速度曲线

    1.内层动态规划
    _solve_dp_inner 接收运行时间的拉格朗日乘子, 执行一次二维变间距动态规划,
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
    [3] Fu C, Sun P, Wang Q, et al. Modeling and energy-saving operation optimization
    of high-speed maglev trains[J]. Journal of Cleaner Production, 2025, 519.


    """

    def __init__(
        self,
        vehicle: VehicleInfo,
        track: TrackInfo,
        safeguard_utility: SafeGuardUtility,
        train_service: TrainService,
        time_tolerance: float,
        show_precompute_progress: bool = True,
        precompute_progress_desc: str = "状态转移图预计算",
    ) -> None:
        self.vehicle = vehicle
        self.track = track
        self.trackprofile = TrackProfile(track=self.track)
        self.safeguard_utility = safeguard_utility
        self.task = train_service
        self.time_tolerance = time_tolerance
        self.show_precompute_progress = show_precompute_progress
        self.precompute_progress_desc = precompute_progress_desc
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
            vehicle=self.vehicle, track=self.track, factor=self.safeguard_utility.gamma
        )
        # 计算最短运行时间参考曲线
        self.ref_curve_pos, self.ref_curve_speed = (
            self.ors.calc_min_operation_time_curve(
                begin_pos=self.task.start_position,
                begin_speed=self.task.start_speed,
                end_pos=self.task.target_position,
                end_speed=0.0,
            )
        )
        self._graph_cache_signature: tuple[float, float, int] | None = None
        self._graph_cache: dict[str, Any] | None = None

    def _get_ref_speed(self, pos: float) -> float:
        return max(0.0, np.interp(pos, self.ref_curve_pos, self.ref_curve_speed))

    def _get_stage_speed_upper_indices(
        self, stages: NDArray[np.float64], speed_states: NDArray[np.float64]
    ) -> NDArray[np.int_]:
        """预插值参考速度上界, 避免在DP内层重复插值。"""
        ref_speed_at_stage = np.maximum(
            0.0, np.interp(stages, self.ref_curve_pos, self.ref_curve_speed)
        )
        upper_idx = np.searchsorted(speed_states, ref_speed_at_stage, side="right") - 1
        return np.clip(upper_idx, -1, len(speed_states) - 1).astype(np.int_)

    def _build_transition_graph(
        self, stages: NDArray[np.float64], speed_states: NDArray[np.float64]
    ) -> dict[str, Any]:
        """
        预计算状态转移图（可行性/能耗/时间）, 供外层不同lambda复用。
        """
        total_steps = len(stages) - 1
        num_speed_states = len(speed_states)
        stage_speed_upper_idx = self._get_stage_speed_upper_indices(
            stages, speed_states
        )

        transitions: list[
            list[
                tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]] | None
            ]
        ] = [[None for _ in range(num_speed_states)] for _ in range(total_steps)]
        total_valid_edges = 0

        for k in self._iter_precompute_steps(total_steps):
            k_idx = int(k)
            pos_k = float(stages[k_idx])
            delta_pos = float(stages[k_idx + 1] - stages[k_idx])
            abs_delta_pos = abs(delta_pos)
            current_upper = int(stage_speed_upper_idx[k_idx])
            next_upper = int(stage_speed_upper_idx[k_idx + 1])

            if current_upper < 0 or next_upper < 0:
                continue

            for i in range(current_upper + 1):
                speed_k = float(speed_states[i])

                # 基于加减速度物理边界的下一阶段速度索引剪枝
                v2_min = max(
                    speed_k**2 + 2.0 * self.vehicle.max_dec * abs_delta_pos, 0.0
                )
                v2_max = max(
                    speed_k**2 + 2.0 * self.vehicle.max_acc * abs_delta_pos, 0.0
                )
                v_next_min = np.sqrt(v2_min)
                v_next_max = np.sqrt(v2_max)

                j_min = int(np.searchsorted(speed_states, v_next_min, side="left"))
                j_max = int(np.searchsorted(speed_states, v_next_max, side="right") - 1)
                j_max = min(j_max, next_upper)

                if j_min > j_max:
                    continue

                next_indices: list[int] = []
                delta_energy_list: list[float] = []
                delta_time_list: list[float] = []

                for j in range(j_min, j_max + 1):
                    speed_next = float(speed_states[j])
                    is_valid, delta_energy, delta_time = self._calculate_transition(
                        pos_k=pos_k,
                        speed_k=speed_k,
                        displacement=delta_pos,
                        speed_k_1=speed_next,
                    )
                    if not is_valid:
                        continue

                    next_indices.append(j)
                    delta_energy_list.append(delta_energy)
                    delta_time_list.append(delta_time)

                if not next_indices:
                    continue

                transitions[k_idx][i] = (
                    np.asarray(next_indices, dtype=np.int_),
                    np.asarray(delta_energy_list, dtype=np.float64),
                    np.asarray(delta_time_list, dtype=np.float64),
                )
                total_valid_edges += len(next_indices)

        return {
            "stages": stages,
            "speed_states": speed_states,
            "stage_speed_upper_idx": stage_speed_upper_idx,
            "transitions": transitions,
            "total_valid_edges": total_valid_edges,
        }

    def _iter_precompute_steps(self, total_steps: int) -> Iterable[int]:
        if self.show_precompute_progress and tqdm is not None:
            return tqdm(
                range(total_steps),
                total=total_steps,
                desc=self.precompute_progress_desc,
                dynamic_ncols=True,
                unit="stage",
                mininterval=0.2,
            )
        return range(total_steps)

    def _prepare_transition_graph_cache(
        self, max_speed: float, delta_speed: float
    ) -> dict[str, Any]:
        stages = self._generate_variable_spacing_stages().astype(np.float64)
        speed_states = np.arange(0, max_speed, delta_speed, dtype=np.float64)
        cache_signature = (float(max_speed), float(delta_speed), len(stages))

        if self._graph_cache_signature != cache_signature:
            print("正在预计算状态转移图（仅首次或参数变化时执行）...")
            if self.show_precompute_progress and tqdm is None:
                print("未检测到 tqdm，已回退为普通循环输出。")
            self._graph_cache = self._build_transition_graph(stages, speed_states)
            self._graph_cache_signature = cache_signature
            print(
                f"转移图预计算完成: 可行转移边数量 {self._graph_cache['total_valid_edges']}"
            )

        assert self._graph_cache is not None
        return self._graph_cache

    def _generate_variable_spacing_stages(self, sub_stage_count: int = 30) -> NDArray:
        """
        基于临界点的变间距阶段划分
        将线路按照临界点划分为大分区, 每个大分区等分为 sub_stage_count 个子阶段
        """
        critical_points_position_arr = np.concatenate((
            np.array([self.task.start_position]),
            self.safeguard_utility.get_intersecting_dangerous_point(),
            np.array([self.task.target_position]),
        ))
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
        if np.isclose(displacement, 0.0):
            return False, np.inf, np.inf

        if np.isclose(speed_k + speed_k_1, 0.0):
            return False, np.inf, np.inf

        # 匀变速模型: a = (v1^2 - v0^2) / (2 * ds)
        acc = (speed_k_1**2 - speed_k**2) / (2.0 * displacement)

        acc_tol = 1e-9
        if acc > self.vehicle.max_acc + acc_tol or acc < self.vehicle.max_dec - acc_tol:
            return False, np.inf, np.inf

        sample_count = max(10, int(np.abs(displacement) / 10.0))
        distance_sample = np.linspace(0.0, displacement, sample_count, dtype=np.float64)
        pos_sample = pos_k + distance_sample

        # 由匀变速公式采样速度，保证与端点速度一致
        speed_sq_sample = speed_k**2 + 2.0 * acc * distance_sample
        speed_sample = np.sqrt(np.maximum(speed_sq_sample, 0.0))

        # 检查是否进入危险速度域
        if self.safeguard_utility.detect_danger(
            pos=pos_sample, speed=speed_sample
        ).any():
            return False, np.inf, np.inf

        if np.abs(acc) < 1e-9:
            time = np.abs(displacement) / speed_k
        else:
            time = (speed_k_1 - speed_k) / acc

        propulsion_energy, leviation_energy = self.ecc.calc_energy(
            begin_pos=pos_k,
            begin_speed=speed_k,
            acc=acc,
            distance=abs(displacement),
            direction=1 if displacement > 0 else -1,
            operation_time=time,
            vehicle=self.vehicle,
            trackprofile=self.trackprofile,
        )

        return True, propulsion_energy + leviation_energy, time

    def BuildSmoothedDisplayCurve(
        self,
        pos_arr: Sequence[float] | NDArray,
        speed_arr: Sequence[float] | NDArray,
        samples_per_segment: int = 20,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        基于匀变速直线运动模型对离散优化结果进行分段加密采样。
        仅用于可视化展示，不参与优化与能耗计算。
        """
        pos = np.asarray(pos_arr, dtype=np.float64)
        speed = np.asarray(speed_arr, dtype=np.float64)

        if pos.ndim != 1 or speed.ndim != 1 or pos.size != speed.size:
            raise ValueError(
                "pos_arr and speed_arr must be 1-D arrays with equal length"
            )
        if pos.size < 2:
            return pos.copy(), speed.copy()

        dense_pos: list[float] = [float(pos[0])]
        dense_speed: list[float] = [float(max(speed[0], 0.0))]
        n_local = max(3, int(samples_per_segment))

        for i in range(pos.size - 1):
            p0 = float(pos[i])
            p1 = float(pos[i + 1])
            v0 = float(max(speed[i], 0.0))
            v1 = float(max(speed[i + 1], 0.0))
            ds = p1 - p0

            if np.isclose(ds, 0.0):
                continue

            # 对每个阶段采用匀变速方程重建速度曲线
            acc = (v1**2 - v0**2) / (2.0 * ds)
            s_local = np.linspace(0.0, ds, n_local, endpoint=True, dtype=np.float64)
            v_sq_local = v0**2 + 2.0 * acc * s_local
            v_local = np.sqrt(np.maximum(v_sq_local, 0.0))
            p_local = p0 + s_local

            dense_pos.extend(p_local[1:].tolist())
            dense_speed.extend(v_local[1:].tolist())

        return np.asarray(dense_pos, dtype=np.float64), np.asarray(
            dense_speed, dtype=np.float64
        )

    def _solve_dp_inner(
        self, lambda_time: float, max_speed: float, delta_speed: float
    ) -> OptimalSpeedProfile | None:
        """
        逆推法求解变间距动态规划, 结合目标函数及约束
        使用拉格朗日乘子法简化时间惩罚
        """
        cache = self._prepare_transition_graph_cache(
            max_speed=max_speed, delta_speed=delta_speed
        )
        stages = cache["stages"]
        speed_states = cache["speed_states"]
        stage_speed_upper_idx = cache["stage_speed_upper_idx"]
        transitions = cache["transitions"]

        total_steps = len(stages) - 1
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
        dp_policy = np.full(
            (total_steps + 1, num_speed_states), -1, dtype=int
        )  # 记录状态转移动作

        # 状态初始化
        dp_cost[total_steps, 0] = 0.0
        dp_time_accum[total_steps, 0] = 0.0

        print(f"开始使用逆推法求解，共{total_steps}个阶段")

        # 逆推法
        for k in range(total_steps - 1, -1, -1):
            current_upper = int(stage_speed_upper_idx[k])
            if current_upper < 0:
                continue

            for i in range(current_upper + 1):
                transition = transitions[k][i]
                if transition is None:
                    continue

                next_indices, delta_energy_arr, delta_time_arr = transition
                next_dp_cost = dp_cost[k + 1, next_indices]
                finite_mask = np.isfinite(next_dp_cost)
                if not np.any(finite_mask):
                    continue

                valid_next_indices = next_indices[finite_mask]
                valid_delta_energy = delta_energy_arr[finite_mask]
                valid_delta_time = delta_time_arr[finite_mask]
                valid_next_dp_cost = next_dp_cost[finite_mask]

                total_cost_arr = (
                    valid_delta_energy
                    + lambda_time * valid_delta_time
                    + valid_next_dp_cost
                )
                best_local_idx = int(np.argmin(total_cost_arr))
                best_next_j = int(valid_next_indices[best_local_idx])

                dp_cost[k, i] = float(total_cost_arr[best_local_idx])
                dp_policy[k, i] = best_next_j
                dp_time_accum[k, i] = float(
                    valid_delta_time[best_local_idx] + dp_time_accum[k + 1, best_next_j]
                )

        # 检验是否找到可行解
        if np.isinf(dp_cost[0, 0]):
            return None

        # 提取最优速度曲线
        optimal_speed_indices = [0]
        current_speed_idx = 0
        for k in range(total_steps):
            next_speed_idx = dp_policy[k, current_speed_idx]
            if next_speed_idx < 0:
                return None
            optimal_speed_indices.append(next_speed_idx)
            current_speed_idx = next_speed_idx

        optimal_speed_profile = speed_states[
            np.asarray(optimal_speed_indices, dtype=int)
        ]
        total_time = dp_time_accum[0, 0]
        total_energy = dp_cost[0, 0] - lambda_time * total_time  # 剥离时间惩罚

        return {
            "pos": stages.tolist(),
            "speed": optimal_speed_profile.tolist(),
            "total_time": total_time,
            "total_energy": total_energy,
        }

    def optimize(
        self, max_speed: float, delta_speed: float, max_iters: int = 100
    ) -> OptimalSpeedProfile | None:
        """
        二分法调整运行时间乘子, 从而使结果逼近规定的运行时间
        """
        target_time = self.task.schedule_time
        print(
            f"开始双层寻优: 目标时间为{target_time:.2f}s, 时间误差容忍率为 {self.time_tolerance}"
        )

        lambda_time = 1e5
        lambda_min = 0.0
        lambda_max = 1e8
        best_result = None

        for iteration in range(max_iters):
            print(
                f"第{iteration + 1}次迭代: 测试 lambda = {lambda_time:.2f} ...", end=""
            )

            # 调用内层DP
            result = self._solve_dp_inner(
                lambda_time=lambda_time, max_speed=max_speed, delta_speed=delta_speed
            )

            if result is None:
                print("未找到可行解, 请检查代码逻辑")
                break

            total_time = result["total_time"]
            total_energy = result["total_energy"]
            print(f"实际耗时为{total_time:.2f}s, 能耗为{total_energy:.2f}kJ")

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
                lambda_min = lambda_time
            else:
                # 耗时太短, 减小时间惩罚
                lambda_max = lambda_time

            lambda_time = (lambda_min + lambda_max) / 2.0

        if best_result is not None:
            print("最终优化结果")
            print(f"目标运行时间: {target_time:.2f}s")
            print(f"实际规划时间: {best_result['total_time']:.2f}s")
            print(f"最低运行能耗: {best_result['total_energy']:.2f}kJ")

        return best_result


if __name__ == "__main__":
    dp_result_save_dir = "output/optimal/dp"
    os.makedirs(os.path.dirname(dp_result_save_dir), exist_ok=True)

    # 坡度，百分位
    slopes, slope_intervals = load_slopes()

    # 区间限速
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)

    # 车站
    stations_data = load_stations()
    longyang_start = stations_data["start_station"]["start"]
    longyang_target = stations_data["start_station"]["target"]
    longyang_end = stations_data["start_station"]["end"]
    putong_start = stations_data["end_station"]["start"]
    putong_target = stations_data["end_station"]["target"]
    putong_end = stations_data["end_station"]["end"]

    # 防护曲线
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list",
        "max_curves_list",
    )

    factor: float = 0.99
    safeguard_utility = SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=factor,
    )

    track = TrackInfo(
        slopes, slope_intervals, speed_limits.tolist(), speed_limit_intervals
    )

    vehicle = VehicleInfo(
        mass=317.5,
        numoftrainsets=5,
        length=128.5,
        max_acc=1.0,
        max_dec=-1.0,
        levi_power_per_mass=1.7,
    )

    train_service = TrainService(
        start_position=longyang_target,
        start_speed=0.0,
        target_position=putong_target,
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error=120.0,
        max_stop_error=0.3,
    )

    VSDP = VariableSpacingDPOptimizer(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        time_tolerance=0.01,
        show_precompute_progress=True,
        precompute_progress_desc="状态转移图预计算",
    )

    result = VSDP.optimize(max_speed=500.0 / 3.6, delta_speed=0.1, max_iters=100)

    if result is not None:
        output_file = f"{dp_result_save_dir}/optimized_speed_curve.npz"
        saved_npz_path, saved_metrics_path = save_curve_and_metrics(
            pos_arr=result["pos"],
            speed_arr=result["speed"],
            output_path=output_file,
            metrics={
                "target_time_s": float(train_service.schedule_time),
                "total_time_s": float(result["total_time"]),
                "total_energy_kj": float(result["total_energy"]),
                "start_position_m": float(train_service.start_position),
                "target_position_m": float(train_service.target_position),
            },
        )
        print(f"优化速度曲线已保存到: {saved_npz_path}")
        print(f"性能指标已保存到: {saved_metrics_path}")

        set_chinese_font()

        fig, ax = plt.subplots(figsize=(12, 7))

        # 绘制静态元素（区间限速、危险速度域和终点等）
        safeguard_utility.render(ax=ax)

        smooth_pos, smooth_speed = VSDP.BuildSmoothedDisplayCurve(
            pos_arr=result["pos"],
            speed_arr=result["speed"],
            samples_per_segment=24,
        )

        # 绘制起点
        ax.scatter(
            train_service.start_position,
            train_service.start_speed * 3.6,
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
            train_service.target_position,
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
            smooth_pos,
            smooth_speed * 3.6,
            label="优化速度曲线(平滑展示)",
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

        plt.show()
