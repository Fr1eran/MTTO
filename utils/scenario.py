from __future__ import annotations

import numpy as np

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)

__all__ = [
    "build_scenario",
    "build_safeguard_utility",
]


def build_scenario(
    *, schedule_time_s: float
) -> tuple[VehicleInfo, TrackInfo, SafeGuardUtility, TrainService]:
    """根据规划运行时间构建磁浮列车运行优化场景的四元组。

    Args:
        schedule_time_s: 规划运行时间 (s)。

    Returns:
        (VehicleInfo, TrackInfo, SafeGuardUtility, TrainService) 四元组。
    """
    slopes, slope_intervals = load_slopes()
    speed_limits, speed_limit_intervals = load_speed_limits(
        to_mps=True, dtype=np.float64
    )
    accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()
    longyang_start_position, putong_end_position = load_stations_goal_positions()
    levi_curves_list, brake_curves_list, min_curves_list, max_curves_list = (
        load_safeguard_curves(
            "levi_curves_list",
            "brake_curves_list",
            "min_curves_list",
            "max_curves_list",
        )
    )

    safeguard_utility = SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        levi_curves_list=levi_curves_list,
        brake_curves_list=brake_curves_list,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=0.99,
    )
    track = TrackInfo(
        slopes=slopes,
        slope_intervals=slope_intervals,
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        ASA_aps=accessible_points,
        ASA_dps=dangerous_points,
    )
    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    train_service = TrainService(
        start_position=longyang_start_position,
        start_speed=0.0,
        target_position=putong_end_position,
        schedule_time=schedule_time_s,
        max_acc_change=0.75,
        max_arr_time_error_ratio=0.01,
        max_stop_error=0.3,
    )

    return vehicle, track, safeguard_utility, train_service


def build_safeguard_utility(factor: float = 0.99) -> SafeGuardUtility:
    """构建用于轨迹可视化时渲染安全防护边界的 SafeGuardUtility 实例。

    Args:
        factor: 安全系数，默认 0.99。

    Returns:
        SafeGuardUtility 实例。
    """
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)
    levi_curves_list, brake_curves_list, min_curves_list, max_curves_list = (
        load_safeguard_curves(
            "levi_curves_list",
            "brake_curves_list",
            "min_curves_list",
            "max_curves_list",
        )
    )

    return SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        levi_curves_list=levi_curves_list,
        brake_curves_list=brake_curves_list,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=factor,
    )
