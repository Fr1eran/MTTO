import json
import os
import pickle
import sys
import numpy as np

# 将项目根目录添加到模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.SafeGuard import SafeGuardCurves
from model.Vehicle import Vehicle
from model.Track import Track, TrackProfile

# 上海磁浮线数据

# 坡度，百分位
with open("data/rail/raw/slopes.json", "r", encoding="utf-8") as f:
    slope_data = json.load(f)
    slopes = slope_data["slopes"]
    slope_intervals = slope_data["intervals"]

# 辅助停车区
with open("data/rail/raw/auxiliary_parking_areas.json", "r", encoding="utf-8") as f:
    apa_data = json.load(f)
    aps = apa_data["accessible_points"]
    dps = apa_data["dangerous_points"]

# 车站
with open("data/rail/raw/stations.json", "r", encoding="utf-8") as f:
    stations_data = json.load(f)
    ly_begin = stations_data["LY"]["begin"]
    ly_zp = stations_data["LY"]["zp"]
    ly_end = stations_data["LY"]["end"]
    pa_begin = stations_data["PA"]["begin"]
    pa_zp = stations_data["PA"]["zp"]
    pa_end = stations_data["PA"]["end"]
    stations_cor = np.array([[ly_begin, pa_begin], [ly_end, pa_end]])

# 加速区
with open("data/rail/raw/acceleration_zones.json", "r", encoding="utf-8") as f:
    az_datas = json.load(f)
    az_begin = az_datas["uplink"]["begin"]
    az_end = az_datas["uplink"]["end"]

# 区间限速
with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
    speedlimit_data = json.load(f)
    speed_limits = speedlimit_data["speed_limits"]
    speed_limits = np.asarray(speed_limits) / 3.6
    speed_limit_intervals = speedlimit_data["intervals"]

track = Track(slopes, slope_intervals, speed_limits.tolist(), speed_limit_intervals)
trackprofile = TrackProfile(track=track)
cal_SGC = SafeGuardCurves(trackprofile=trackprofile)
vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)

# 终点站同样需要计算防护曲线
aps = np.append(aps, pa_begin)
dps = np.append(dps, pa_end)

# 加速区需要计算安全制动曲线
dps = np.insert(dps, 0, az_end)

# 计算安全悬浮曲线、安全制动曲线和最小速度曲线和最大速度曲线
levi_curves_list, min_curves_list = cal_SGC.CalcLeviAndMinCurves(
    apoffsets=aps,
    vehicle=vehicle,
    ds=1.0,
)
brake_curves_list, max_curves_list = cal_SGC.CalcBrakeAndMaxCurves(
    dpoffsets=dps, vehicle=vehicle, ds=1.0
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
