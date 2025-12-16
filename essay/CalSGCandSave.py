import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

# 将项目根目录添加到模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.SafeGuard import SafeGuardCurves
from model.Vehicle import Vehicle
from model.SafeGuard import SafeGuardUtility
from utils.curve import ConcatenateCurvesWithNaN
from model.Track import Track, TrackProfile
from utils.misc import SetChineseFont

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

# 将所有的坐标值数组列表合并为一个数组并在相邻段之间插入np.nan
levi_curves_pos_con, levi_curves_speed_con = ConcatenateCurvesWithNaN(levi_curves_list)
brake_curves_pos_con, brake_curves_speed_con = ConcatenateCurvesWithNaN(
    brake_curves_list
)
min_curves_pos_con, min_curves_speed_con = ConcatenateCurvesWithNaN(min_curves_list)
max_curves_pos_con, max_curves_speed_con = ConcatenateCurvesWithNaN(max_curves_list)

SetChineseFont()

# 绘制区间限速、安全悬浮曲线、安全制动曲线、最小速度曲线
# 最大速度曲线、辅助停车区、车站、加速区
fig1 = plt.figure(dpi=100)
ax1 = fig1.add_subplot(111)
ax1.step(
    speed_limit_intervals[:-1],
    speed_limits * 3.6,
    where="post",
    color="red",
    linestyle="dashdot",
    label="区间限速",
)
ax1.hlines(
    y=np.zeros_like(aps[:-1]),
    xmin=aps[:-1],
    xmax=dps[1:-1],
    colors="green",
    linestyles="solid",
    linewidth=8,
    label="辅助停车区",
    alpha=0.7,
)
ax1.hlines(
    y=np.zeros(2),
    xmin=stations_cor[0, :],
    xmax=stations_cor[1, :],
    colors="blue",
    linestyles="solid",
    linewidth=8,
    label="车站",
    alpha=0.5,
)
ax1.hlines(
    y=np.zeros(2),
    xmin=az_begin,
    xmax=az_end,
    colors="yellow",
    linestyles="solid",
    linewidth=8,
    label="加速区",
    alpha=0.5,
)
ax1.plot(
    levi_curves_pos_con,
    levi_curves_speed_con * 3.6,
    label="安全悬浮曲线",
    color="blue",
    linestyle="dashed",
    alpha=0.7,
)
ax1.plot(
    brake_curves_pos_con,
    brake_curves_speed_con * 3.6,
    label="安全制动曲线",
    color="red",
    linestyle="dashed",
    alpha=0.7,
)
ax1.plot(
    min_curves_pos_con,
    min_curves_speed_con * 3.6,
    label="最小速度曲线",
    color="blue",
    linestyle="solid",
    alpha=0.7,
)
ax1.plot(
    max_curves_pos_con,
    max_curves_speed_con * 3.6,
    label="最大速度曲线",
    color="red",
    linestyle="solid",
    alpha=0.7,
)
ax1.set_xlim((0.0, 30000.0))
ax1.set_ylim((0.0, 500.0))
ax1.set_xlabel(r"位置$s\left( m \right)$")
ax1.set_ylabel(r"速度$v\left( km/h \right)$")
ax1.set_title("区间限速与安全防护曲线")
ax1.legend()


# 绘制危险交叉点后的安全防护曲线和围成的危险速度域
fig2 = plt.figure(dpi=100)
ax2 = fig2.add_subplot()

safeguard = SafeGuardUtility(
    speed_limits=speed_limits,
    speed_limit_intervals=speed_limit_intervals,
    min_curves_list=min_curves_list,
    max_curves_list=max_curves_list,
    gamma=0.99,
)

# 绘制区间限速、危险交叉点、部分安全防护曲线和围成的危险速度域
safeguard.Render(ax=ax2)

# 绘制辅助停车区、车站
ax2.hlines(
    y=np.zeros_like(aps[:-1]),
    xmin=aps[:-1],
    xmax=dps[1:-1],
    colors="green",
    linestyles="solid",
    linewidth=8,
    label="辅助停车区",
    alpha=0.7,
)
ax2.hlines(
    y=np.zeros(2),
    xmin=stations_cor[0, :],
    xmax=stations_cor[1, :],
    colors="blue",
    linestyles="solid",
    linewidth=8,
    label="车站",
    alpha=0.5,
)
ax2.hlines(
    y=np.zeros(2),
    xmin=az_begin,
    xmax=az_end,
    colors="yellow",
    linestyles="solid",
    linewidth=8,
    label="加速区",
    alpha=0.5,
)

ax2.set_xlim((0.0, 30000.0))
ax2.set_ylim((0.0, 500.0))
ax2.set_xlabel(r"位置$s\left( m \right)$")
ax2.set_ylabel(r"速度$v\left( km/h \right)$")
ax2.set_title("区间限速与危险速度域")
ax2.legend()

plt.show()
