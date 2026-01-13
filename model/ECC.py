import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from numpy.typing import NDArray
from model.Vehicle import Vehicle, VehicleDynamic
from model.Track import TrackProfile


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
        acc: float,
        distance: float,
        direction: int,
        operation_time: float | None,
        vehicle: Vehicle,
        trackprofile: TrackProfile,
    ) -> tuple[float, float]:
        """
        计算列车从起始位置和速度以某恒定加速度连续位移一段距离消耗的能量(kJ),
        包括牵引系统能耗(kJ)和悬浮系统能耗(kJ)

        Args:
            begin_pos: 起始位置(m)
            begin_speed: 起始速度(m/s)
            acc: 加速度(m/s^2)
            distance: 运动距离(m)
            travel_time: 运行时间(s)
            vehicle: 车辆实例
            trackprofile: 轨道特性实例

        Returns:
            propulsion_energy_consumption: 牵引能耗(kJ)
            leviation_energy_consumption: 悬浮能耗(kJ)
        """

        # 计算机械能耗
        if np.abs(distance) < 1e-6:
            # 位移极小，使用起始点的力近似
            F_longitudinal = VehicleDynamic.CalcLongitudinalForce(
                vehicle=vehicle,
                acc=acc,
                speed=begin_speed,
                slope=trackprofile.GetSlope(pos=begin_pos, interpolate=True),
            )
            mechanic_energy_consumption = np.abs(F_longitudinal * distance)
            motor_energy_consumption = 0.0
        else:
            # 离散采样进行数值积分
            n_samples = max(10, int(np.abs(distance) / 1.0))
            d_samples = np.linspace(0, distance, n_samples, endpoint=False)
            p_samples = begin_pos + d_samples * direction

            # 加速度极小时，认为做匀速直线运动
            if np.abs(acc) < 1e-6:
                t_samples = d_samples / np.maximum(begin_speed, 1e-6)
            else:
                t_samples = (
                    np.sqrt(np.maximum(begin_speed**2 + 2 * acc * d_samples, 0))
                    - begin_speed
                ) / acc

            speed_squared = begin_speed**2 + 2 * acc * d_samples
            speed_samples = np.sqrt(np.maximum(speed_squared, 0))

            F_longitudinal = VehicleDynamic.CalcLongitudinalForce(
                vehicle=vehicle,
                acc=acc,
                speed=speed_samples,
                slope=trackprofile.GetSlope(p_samples, interpolate=True),
            )
            mechanic_energy_consumption = trapezoid(
                y=np.abs(F_longitudinal), x=d_samples
            )
            motor_energy_consumption = trapezoid(
                y=(2 * F_longitudinal**2 / (3 * self.h**2))
                * (self.R_m + self.k_c**2 * self.R_k + (1 - self.k_c) ** 2 * self.R_k),
                x=t_samples,
            ) + trapezoid(
                y=(np.abs(F_longitudinal) * 2 / (3 * self.h**2))
                * (self.L_d + self.k_c**2 * self.L_k + (1 - self.k_c) ** 2 * self.L_k),
                x=np.abs(F_longitudinal),
            )

        # 计算悬浮能耗
        if operation_time is None:
            # 根据运动学计算时间
            if np.abs(acc) < 1e-9:
                # 匀速运动
                time = distance / np.maximum(begin_speed, 1e-6)
            else:
                # 变速运动
                next_speed_squared = begin_speed**2 + 2 * acc * distance
                next_speed = np.sqrt(np.maximum(next_speed_squared, 0))
                time = (next_speed - begin_speed) / acc
        else:
            time = operation_time

        propulsion_energy_consumption = (
            mechanic_energy_consumption + motor_energy_consumption
        )

        leviation_energy_consumption = (
            self.Phi_1 * distance + self.Phi_2 * vehicle.mass * time
        )

        return float(propulsion_energy_consumption), float(leviation_energy_consumption)
