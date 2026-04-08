import os
import pickle
import sys
import numpy as np

# 将项目根目录添加到模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.SafeGuard import SafeGuardCurves
from model.Vehicle import Vehicle
from model.Track import Track, TrackProfile
from utils.data_loader import (
    load_acceleration_zones,
    load_auxiliary_stopping_areas_ap_and_dp,
    load_slopes,
    load_speed_limits,
    load_stations,
)

# 上海磁浮线数据

slopes, slope_intervals = load_slopes()

# 辅助停车区
# 目标车站被视为特殊的辅助停车区，现已包含
accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()

# 车站
stations_data = load_stations()
longyang_start = stations_data["start_station"]["start"]
longyang_target = stations_data["start_station"]["target"]
longyang_end = stations_data["start_station"]["end"]
putong_start = stations_data["end_station"]["start"]
putong_target = stations_data["end_station"]["target"]
putong_end = stations_data["end_station"]["end"]
stations_cor = np.array([[longyang_start, putong_start], [longyang_end, putong_end]])

# 加速区
acceleration_zone_data = load_acceleration_zones()
acceleration_zone_start = acceleration_zone_data["uplink"]["start"]
acceleration_zone_end = acceleration_zone_data["uplink"]["end"]

# 区间限速
speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)

track = Track(
    slopes,
    slope_intervals,
    speed_limits.tolist(),
    speed_limit_intervals,
    accessible_points,
    dangerous_points,
)
trackprofile = TrackProfile(track=track)
cal_SGC = SafeGuardCurves(trackprofile=trackprofile)
vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)

# 终点站同样需要计算防护曲线
# aps = np.append(aps, pa_begin)
# dps = np.append(dps, pa_end)

# 加速区需要计算安全制动曲线
dangerous_points = np.insert(dangerous_points, 0, acceleration_zone_end)

# 计算安全悬浮曲线、安全制动曲线和最小速度曲线和最大速度曲线
levi_curves_list, min_curves_list = cal_SGC.CalcLeviAndMinCurves(
    apoffsets=np.asarray(accessible_points),
    vehicle=vehicle,
    ds=1.0,
)
brake_curves_list, max_curves_list = cal_SGC.CalcBrakeAndMaxCurves(
    dpoffsets=dangerous_points, vehicle=vehicle, ds=1.0
)

# 保存计算后的防护曲线
with open("data/rail/safeguard/levi_curves_list.pkl", "wb") as f:
    pickle.dump(levi_curves_list, f)

with open("data/rail/safeguard/brake_curves_list.pkl", "wb") as f:
    pickle.dump(brake_curves_list, f)

with open("data/rail/safeguard/min_curves_list.pkl", "wb") as f:
    pickle.dump(min_curves_list, f)

with open("data/rail/safeguard/max_curves_list.pkl", "wb") as f:
    pickle.dump(max_curves_list, f)
