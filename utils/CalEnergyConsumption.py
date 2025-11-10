import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scipy.integrate import cumulative_trapezoid

from model.Vehicle import Vehicle, VehicleDynamic
from model.Track import TrackProfile


def CalEnergyConsumption(
    pos, speed, acc, vehicle: Vehicle, trackprofile: TrackProfile, travel_time
):
    """
    计算列车运行一段距离消耗的总能量，包括纵向力做的总功和悬浮能耗

    Args:
        pos: 位置
        speed: 速度大小(单位: m/s)
        vehicle: 车辆类
        trackprofile: 线路特性对象
        travel_time: 总共旅行时间

    Returns:
        mechanic_energy_consumption
        levi_energy_consumption
    """

    f_longitudinal = VehicleDynamic.CalLongitudinalForce(
        vehicle, acc, speed, trackprofile.GetSlope(pos, interpolate=True)
    )

    mechanic_energy_consumption = cumulative_trapezoid(
        y=np.abs(f_longitudinal) * speed, x=travel_time, initial=0
    )

    levi_energy_consumption = np.cumulative_sum(
        np.diff(travel_time) * vehicle.levi_power_per_mass * vehicle.mass,
        include_initial=True,
    )

    return mechanic_energy_consumption, levi_energy_consumption
