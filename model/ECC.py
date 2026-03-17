from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from numpy.typing import NDArray
from model.Vehicle import Vehicle, VehicleDynamic
from model.Track import TrackProfile

AccProfile = Callable[[NDArray[np.floating]], NDArray[np.floating]]


class ECC:
    """
    计算高速磁浮运行过程中牵引系统产生的能量消耗

    P_s = P_v + P_m

    P_s: 牵引变电站输出功率
    P_v: 列车机械功率
    P_m: 电机损耗功率

    P_lev = Φ_1 * v + Φ_2 * (m_v + m_p)

    Φ_1: 0.1049
    Φ_2: 1.006
    v: 列车运行速度
    m_v: 列车空载质量
    m_p: 载客质量

    参考文献：
    [1] 赖晴鹰. 中速磁浮运控电一体化运行策略优化[D]. 北京交通大学, 2022.
    [2] Q. Lai, J. Liu, A. Haghani, L. Meng, and Y. Wang, “Optimal Energy Speed Profile of Medium-Speed Maglev
    Trains Integrating the Power Supply System and Train Control System,” Transportation Research Record,
    vol. 2674, no. Compendex, pp. 729-738, 2020, doi: 10.1177/0361198120938052.
    [3] 柴晓凤．中速磁浮节能运行图优化方法研究[D]. 北京交通大学, 2020.


    Attributes:
        R_m: 直线电机等效阻抗(Ω)
        L_d: 直线电机d轴等效电感(H)
        R_k: 馈电线等效阻抗(Ω)
        L_k: 馈电线等效电感(H)
        Tau: 直线电机极距(m)
        Psi_fd: 动子有效磁链(Wb)
        k_c: 牵引变电站电流分配系数

    Methods:
        CalcEnergyCumul:
    """

    def __init__(
        self,
        R_m: float,
        L_d: float,
        R_k: float,
        L_k: float,
        Tau: float,
        Psi_fd: float,
        k_c: float,
    ) -> None:
        self.R_m = R_m
        self.L_d = L_d
        self.R_k = R_k
        self.L_k = L_k
        self.Tau = Tau
        self.Psi_fd = Psi_fd
        self.k_c = k_c
        self.h = np.pi * Psi_fd / Tau
        self.Phi_1 = 0.1049
        self.Phi_2 = 1.006

    def CalcEnergyCumul(
        self,
        pos_arr: NDArray[np.floating],
        speed_arr: NDArray[np.floating],
        acc_arr: NDArray[np.floating],
        vehicle: Vehicle,
        trackprofile: TrackProfile,
        travel_time_arr: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        计算列车运行一段距离消耗的总能量(kJ),
        包括牵引系统能耗(kJ)和悬浮系统能耗(kJ)

        Args:
            pos: 位置(单位: m)
            speed: 速度大小(单位: m/s)
            vehicle: 车辆类
            trackprofile: 线路特性对象
            travel_time: 总共旅行时间

        Returns:
            tuple(propulsion_energy_consumption, leviation_energy_consumption)
        """

        F_longitudinal = VehicleDynamic.CalcLongitudinalForce(
            vehicle,
            acc_arr,
            speed_arr,
            trackprofile.GetSlope(pos_arr, interpolate=True),
        )

        mechanic_energy_consumption = cumulative_trapezoid(
            y=np.abs(F_longitudinal * speed_arr), x=travel_time_arr, initial=0
        )

        motor_energy_consumption = cumulative_trapezoid(
            y=(2 * F_longitudinal**2 / (3 * self.h**2))
            * (self.R_m + self.k_c**2 * self.R_k + (1 - self.k_c) ** 2 * self.R_k),
            x=travel_time_arr,
            initial=0,
        ) + cumulative_trapezoid(
            y=(np.abs(F_longitudinal) * 2 / (3 * self.h**2))
            * (self.L_d + self.k_c**2 * self.L_k + (1 - self.k_c) ** 2 * self.L_k),
            x=np.abs(F_longitudinal),
            initial=0,
        )

        propulsion_energy_consumption = (
            mechanic_energy_consumption + motor_energy_consumption
        )

        leviation_energy_consumption = np.cumulative_sum(
            (
                self.Phi_1 * np.diff(pos_arr)
                + self.Phi_2 * vehicle.mass * np.diff(travel_time_arr)
            ),
            include_initial=True,
        )

        return (
            propulsion_energy_consumption,
            leviation_energy_consumption,
        )

    def CalcEnergy(
        self,
        begin_pos: float,
        begin_speed: float,
        acc: float | AccProfile,
        distance: float,
        direction: int,
        operation_time: float | None,
        vehicle: Vehicle,
        trackprofile: TrackProfile,
    ) -> tuple[float, float]:
        """
        计算列车从起始位置和速度连续位移一段距离消耗的能量(kJ),
        包括牵引系统能耗(kJ)和悬浮系统能耗(kJ)

        Args:
            begin_pos: 起始位置(m)
            begin_speed: 起始速度(m/s)
            acc: 加速度常数，或以位移距离为输入返回加速度的回调函数
                回调必须支持数组输入（向量化），并返回与输入同维度同形状的数组
            distance: 运动距离(m)
            operation_time: 运行时间(s)
            vehicle: 车辆实例
            trackprofile: 轨道特性实例

        Returns:
            propulsion_energy_consumption: 牵引能耗(kJ)
            leviation_energy_consumption: 悬浮能耗(kJ)
        """

        acc_is_callable = callable(acc)
        if acc_is_callable:
            acc_func = cast(AccProfile, acc)
            acc_value = None
        else:
            acc_func = None
            acc_value = float(acc)

        def _eval_acc_nodes(d_nodes: NDArray[np.floating]) -> NDArray[np.float64]:
            assert acc_func is not None
            try:
                acc_eval = np.asarray(acc_func(d_nodes), dtype=np.float64)
            except Exception as exc:
                raise TypeError(
                    "acc callback must accept numpy array input (vectorized callback required)"
                ) from exc

            if acc_eval.ndim != d_nodes.ndim or acc_eval.shape != d_nodes.shape:
                raise ValueError(
                    "acc callback must return array with same ndim and shape as input distance samples"
                )

            return np.asarray(acc_eval, dtype=np.float64)

        auto_operation_time: float | None = None

        # 计算机械能耗
        if np.abs(distance) < 1e-6:
            # 位移极小，使用起始点的力近似
            if acc_func is not None:
                acc_for_force = float(
                    _eval_acc_nodes(np.asarray([0.0], dtype=np.float64))[0]
                )
            else:
                acc_for_force = acc_value
            assert acc_for_force is not None
            F_longitudinal = VehicleDynamic.CalcLongitudinalForce(
                vehicle=vehicle,
                acc=acc_for_force,
                speed=begin_speed,
                slope=trackprofile.GetSlope(pos=begin_pos, interpolate=True),
            )
            mechanic_energy_consumption = np.abs(F_longitudinal * distance)
            motor_energy_consumption = 0.0
            auto_operation_time = 0.0
        else:
            # 离散采样进行数值积分
            n_samples = max(10, int(np.abs(distance) / 1.0))
            d_nodes = np.linspace(0.0, distance, n_samples + 1)
            delta_d = np.diff(d_nodes)
            p_nodes = begin_pos + d_nodes * direction

            if acc_func is not None:
                acc_nodes = _eval_acc_nodes(d_nodes)
            else:
                assert acc_value is not None
                acc_nodes = np.full_like(d_nodes, acc_value, dtype=np.float64)

            speed_nodes = np.empty_like(d_nodes)
            speed_nodes[0] = begin_speed
            for i in range(n_samples):
                speed_nodes[i + 1] = np.sqrt(
                    np.maximum(
                        speed_nodes[i] ** 2 + 2.0 * acc_nodes[i] * delta_d[i],
                        0.0,
                    )
                )

            t_nodes = np.zeros_like(d_nodes)
            for i in range(n_samples):
                # 用平均速度求平均时间的数值稳定性更强
                avg_speed = np.maximum(
                    (speed_nodes[i] + speed_nodes[i + 1]) / 2.0, 1e-6
                )
                t_nodes[i + 1] = t_nodes[i] + np.abs(delta_d[i]) / avg_speed

            slope_nodes = trackprofile.GetSlope(p_nodes, interpolate=True)

            F_longitudinal = VehicleDynamic.CalcLongitudinalForce(
                vehicle=vehicle,
                acc=acc_nodes,
                speed=speed_nodes,
                slope=slope_nodes,
            )
            # |F| 对距离积分，使用绝对距离增量确保能耗非负
            mechanic_energy_consumption = np.sum(
                0.5
                * (np.abs(F_longitudinal[:-1]) + np.abs(F_longitudinal[1:]))
                * np.abs(delta_d)
            )
            motor_energy_consumption = trapezoid(
                y=(2 * F_longitudinal**2 / (3 * self.h**2))
                * (self.R_m + self.k_c**2 * self.R_k + (1 - self.k_c) ** 2 * self.R_k),
                x=t_nodes,
            ) + trapezoid(
                y=(np.abs(F_longitudinal) * 2 / (3 * self.h**2))
                * (self.L_d + self.k_c**2 * self.L_k + (1 - self.k_c) ** 2 * self.L_k),
                x=np.abs(F_longitudinal),
            )
            auto_operation_time = float(t_nodes[-1])

        # 计算悬浮能耗
        if operation_time is None:
            if acc_func is not None:
                assert auto_operation_time is not None
                time = auto_operation_time
            else:
                assert acc_value is not None
                # 保持恒加速度输入的原有时间计算逻辑
                if np.abs(acc_value) < 1e-9:
                    time = distance / np.maximum(begin_speed, 1e-6)
                else:
                    next_speed_squared = begin_speed**2 + 2 * acc_value * distance
                    next_speed = np.sqrt(np.maximum(next_speed_squared, 0))
                    time = (next_speed - begin_speed) / acc_value
        else:
            time = operation_time

        propulsion_energy_consumption = (
            mechanic_energy_consumption + motor_energy_consumption
        )

        leviation_energy_consumption = (
            self.Phi_1 * distance + self.Phi_2 * vehicle.mass * time
        )

        return float(propulsion_energy_consumption), float(leviation_energy_consumption)
