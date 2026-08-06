import numpy as np
from numba import (
    njit,
)
from numpy.typing import NDArray
from scipy.integrate import (
    cumulative_trapezoid,
)

from model.track import TrackInfo, get_slope, get_slope_scalar_numba
from model.vehicle import (
    VehicleInfo,
    calc_longitudinal_force,
    calc_longitudinal_force_scalar_numba,
)


@njit(cache=True)
def _calc_energy_constant_acc_numba(
    begin_pos: float,
    begin_speed: float,
    acc: float,
    distance: float,
    direction: int,
    operation_time_value: float,
    mass: float,
    numoftrainsets: float,
    slopes: NDArray[np.float64],
    slope_intervals: NDArray[np.float64],
    r_m: float,
    l_d: float,
    r_k: float,
    l_k: float,
    k_c: float,
    h: float,
    phi_1: float,
    phi_2: float,
) -> tuple[float, float]:
    mechanic_energy_consumption = 0.0
    motor_energy_consumption = 0.0
    abs_distance = np.abs(distance)

    if abs_distance < 1e-6:
        slope = get_slope_scalar_numba(begin_pos, slopes, slope_intervals)
        f_longitudinal = calc_longitudinal_force_scalar_numba(
            begin_speed, slope, acc, mass, numoftrainsets
        )
        mechanic_energy_consumption = np.abs(f_longitudinal * distance)
    else:
        n_samples = int(abs_distance / 1.0)
        if n_samples < 10:
            n_samples = 10

        delta_d = distance / n_samples
        abs_delta_d = np.abs(delta_d)
        motor_r_coeff = (
            2.0 / (3.0 * h**2) * (r_m + k_c**2 * r_k + (1.0 - k_c) ** 2 * r_k)
        )
        motor_l_coeff = (
            2.0 / (3.0 * h**2) * (l_d + k_c**2 * l_k + (1.0 - k_c) ** 2 * l_k)
        )

        speed = begin_speed
        time_current = 0.0
        slope = get_slope_scalar_numba(begin_pos, slopes, slope_intervals)
        f_current = calc_longitudinal_force_scalar_numba(
            speed, slope, acc, mass, numoftrainsets
        )
        abs_f_current = np.abs(f_current)
        motor_r_current = f_current**2 * motor_r_coeff
        motor_l_current = abs_f_current * motor_l_coeff

        for i in range(n_samples):
            next_speed_squared = speed**2 + 2.0 * acc * delta_d
            if next_speed_squared < 0.0:
                next_speed_squared = 0.0
            speed_next = np.sqrt(next_speed_squared)

            avg_speed = (speed + speed_next) / 2.0
            if avg_speed < 1e-6:
                avg_speed = 1e-6
            time_next = time_current + abs_delta_d / avg_speed

            d_next = delta_d * (i + 1)
            pos_next = begin_pos + d_next * direction
            slope_next = get_slope_scalar_numba(pos_next, slopes, slope_intervals)
            f_next = calc_longitudinal_force_scalar_numba(
                speed_next,
                slope_next,
                acc,
                mass,
                numoftrainsets,
            )
            abs_f_next = np.abs(f_next)
            motor_r_next = f_next**2 * motor_r_coeff
            motor_l_next = abs_f_next * motor_l_coeff

            mechanic_energy_consumption += (
                0.5 * (abs_f_current + abs_f_next) * abs_delta_d
            )
            motor_energy_consumption += (
                0.5 * (motor_r_current + motor_r_next) * (time_next - time_current)
            )
            motor_energy_consumption += (
                0.5 * (motor_l_current + motor_l_next) * (abs_f_next - abs_f_current)
            )

            speed = speed_next
            time_current = time_next
            f_current = f_next
            abs_f_current = abs_f_next
            motor_r_current = motor_r_next
            motor_l_current = motor_l_next

    if np.isnan(operation_time_value):
        if np.abs(acc) < 1e-9:
            speed_denom = begin_speed
            if speed_denom < 1e-6:
                speed_denom = 1e-6
            time = distance / speed_denom
        else:
            next_speed_squared = begin_speed**2 + 2.0 * acc * distance
            if next_speed_squared < 0.0:
                next_speed_squared = 0.0
            next_speed = np.sqrt(next_speed_squared)
            time = (next_speed - begin_speed) / acc
    else:
        time = operation_time_value

    propulsion_energy_consumption = (
        mechanic_energy_consumption + motor_energy_consumption
    )
    leviation_energy_consumption = phi_1 * distance + phi_2 * mass * time

    return propulsion_energy_consumption, leviation_energy_consumption


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
    [2] Q. Lai, J. Liu, A. Haghani, L. Meng, and Y. Wang,
    “Optimal Energy Speed Profile of Medium-Speed Maglev
    Trains Integrating the Power Supply System and Train Control System,”
    Transportation Research Record,
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
        self.R_m: float = R_m
        self.L_d: float = L_d
        self.R_k: float = R_k
        self.L_k: float = L_k
        self.Tau: float = Tau
        self.Psi_fd: float = Psi_fd
        self.k_c: float = k_c
        self.h: float = np.pi * Psi_fd / Tau
        self.Phi_1: float = 0.1049
        self.Phi_2: float = 1.006

    def calc_energy_cumulative(
        self,
        pos_arr: NDArray[np.float64],
        speed_arr: NDArray[np.float64],
        acc_arr: NDArray[np.float64],
        vehicle: VehicleInfo,
        track: TrackInfo,
        travel_time_arr: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        计算列车运行一段距离消耗的总能量(kJ),
        包括牵引系统能耗(kJ)和悬浮系统能耗(kJ)

        Args:
            pos: 位置(单位: m)
            speed: 速度大小(单位: m/s)
            vehicle: 车辆类
            track: 线路数据对象
            travel_time: 总共旅行时间

        Returns:
            tuple(propulsion_energy_consumption, leviation_energy_consumption)
        """

        F_longitudinal = calc_longitudinal_force(
            mass=vehicle.mass,
            numoftrainsets=vehicle.numoftrainsets,
            acc=acc_arr,
            speed=speed_arr,
            slope=get_slope(
                pos_arr,
                track.slopes,
                track.slope_intervals,
                dtype=np.float64,
            ),
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

    def calc_energy(
        self,
        begin_pos: float,
        begin_speed: float,
        acc: float,
        distance: float,
        direction: int,
        operation_time: float | None,
        vehicle: VehicleInfo,
        track: TrackInfo,
    ) -> tuple[float, float]:
        """
        计算列车从起始位置和速度连续位移一段距离消耗的能量(kJ),
        包括牵引系统能耗(kJ)和悬浮系统能耗(kJ)

        Args:
            begin_pos: 起始位置(m)
            begin_speed: 起始速度(m/s)
            acc: 加速度常数
            distance: 运动距离(m)
            operation_time: 运行时间(s)
            vehicle: 车辆实例
            track: 轨道数据实例

        Returns:
            propulsion_energy_consumption: 牵引能耗(kJ)
            leviation_energy_consumption: 悬浮能耗(kJ)
        """

        operation_time_value = np.nan if operation_time is None else operation_time
        propulsion_energy_consumption, leviation_energy_consumption = (
            _calc_energy_constant_acc_numba(
                begin_pos=begin_pos,
                begin_speed=begin_speed,
                acc=acc,
                distance=distance,
                direction=direction,
                operation_time_value=operation_time_value,
                mass=vehicle.mass,
                numoftrainsets=vehicle.numoftrainsets,
                slopes=track.slopes,
                slope_intervals=track.slope_intervals,
                r_m=self.R_m,
                l_d=self.L_d,
                r_k=self.R_k,
                l_k=self.L_k,
                k_c=self.k_c,
                h=self.h,
                phi_1=self.Phi_1,
                phi_2=self.Phi_2,
            )
        )

        return propulsion_energy_consumption, leviation_energy_consumption
