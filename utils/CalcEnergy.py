import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from numpy.typing import NDArray
from model.Vehicle import Vehicle, VehicleDynamic
from model.Track import TrackProfile


def CalcEnergyCumul(
    pos_arr: NDArray[np.floating],
    speed_arr: NDArray[np.floating],
    acc_arr: NDArray[np.floating],
    vehicle: Vehicle,
    trackprofile: TrackProfile,
    travel_time_arr: NDArray[np.floating],
):
    """
    计算列车运行一段距离消耗的总能量，包括纵向力做的总功和悬浮能耗

    Args:
        pos: 位置(单位: m)
        speed: 速度大小(单位: m/s)
        vehicle: 车辆类
        trackprofile: 线路特性对象
        travel_time: 总共旅行时间

    Returns:
        mechanic_energy_consumption
        levi_energy_consumption
    """

    f_longitudinal = VehicleDynamic.CalcLongitudinalForce(
        vehicle,
        acc_arr,
        speed_arr,
        trackprofile.GetSlope(pos_arr, interpolate=True),
    )

    mechanic_energy_consumption = cumulative_trapezoid(
        y=np.abs(f_longitudinal) * speed_arr, x=travel_time_arr, initial=0
    )

    levi_energy_consumption = np.cumulative_sum(
        np.diff(travel_time_arr) * vehicle.levi_power_per_mass * vehicle.mass,
        include_initial=True,
    )

    return mechanic_energy_consumption, levi_energy_consumption


def CalcEnergy(
    begin_pos: float,
    begin_velocity: float,
    acc: float,
    displacement: float,
    operation_time: float | None,
    vehicle: Vehicle,
    trackprofile: TrackProfile,
) -> tuple[float, float]:
    """
    计算列车在当前位置和速度下以某加速度连续位移一段距离消耗的能量,
    包括纵向驱动力做的总功和悬浮能耗。

    Args:
        current_pos: 列车当前位置(m)
        current_velocity: 列车当前速度(m/s)
        acc(float): 加速度(m/s^2)
        displacement(float): 位移(m)
        travel_time(float | None): 运行时间(s)
        vehicle: 车辆实例
        trackprofile: 轨道特性实例

    Returns:
        mechanic_energy_consumption: 总机械能耗(J)
        leviated_energy_consumption: 悬浮能耗(J)
    """

    # 计算机械能耗
    if np.abs(displacement) < 1e-6:
        # 位移极小，使用起始点的力近似
        F_longitudinal = VehicleDynamic.CalcLongitudinalForce(
            vehicle=vehicle,
            acc=acc,
            speed=np.abs(begin_velocity),
            slope=trackprofile.GetSlope(pos=begin_pos, interpolate=True),
        )
        MEC = np.abs(F_longitudinal * displacement)
    else:
        # 离散采样进行数值积分
        n_samples = max(10, int(np.abs(displacement) / 1.0))
        d_samples = np.linspace(0, displacement, n_samples, endpoint=False)
        p_samples = begin_pos + d_samples

        speed_squared = begin_velocity**2 + 2 * acc * d_samples
        speed_samples = np.sqrt(np.maximum(speed_squared, 0))

        F_longitudinal = VehicleDynamic.CalcLongitudinalForce(
            vehicle=vehicle,
            acc=acc,
            speed=speed_samples,
            slope=trackprofile.GetSlope(p_samples, interpolate=True),
        )
        MEC = trapezoid(y=np.abs(F_longitudinal), x=np.abs(d_samples))

    # 计算悬浮能耗
    if operation_time is None:
        # 根据运动学计算时间
        if np.abs(acc) < 1e-9:
            # 匀速运动
            time = np.abs(displacement) / np.maximum(np.abs(begin_velocity), 1e-6)
        else:
            # 变速运动
            v_next_squared = begin_velocity**2 + 2 * acc * displacement
            v_next = np.sqrt(np.maximum(v_next_squared, 0)) * np.sign(displacement)
            time = (v_next - begin_velocity) / acc
    else:
        time = operation_time

    LEC = vehicle.levi_power_per_mass * vehicle.mass * time

    return float(MEC), float(LEC)
