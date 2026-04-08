import matplotlib.pyplot as plt
import numpy as np

from model.safe_guard_utility import SafeGuardUtility
from utils.curve_plot import concatenate_curves_with_NaN
from utils.data_loader import (
    load_acceleration_zones,
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_speed_limits,
    load_stations,
)
from utils.plot_utils import set_chinese_font

# 辅助停车区
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

levi_curves_list, brake_curves_list, min_curves_list, max_curves_list = (
    load_safeguard_curves(
        "levi_curves_list",
        "brake_curves_list",
        "min_curves_list",
        "max_curves_list",
    )
)

# 将所有的坐标值数组列表合并为一个数组并在相邻段之间插入np.nan
levi_curves_pos_con, levi_curves_speed_con = concatenate_curves_with_NaN(
    levi_curves_list
)
brake_curves_pos_con, brake_curves_speed_con = concatenate_curves_with_NaN(
    brake_curves_list
)
min_curves_pos_con, min_curves_speed_con = concatenate_curves_with_NaN(min_curves_list)
max_curves_pos_con, max_curves_speed_con = concatenate_curves_with_NaN(max_curves_list)

set_chinese_font()

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
    y=np.zeros_like(accessible_points),
    xmin=accessible_points,
    xmax=dangerous_points,
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
    xmin=acceleration_zone_start,
    xmax=acceleration_zone_end,
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
    factor=0.99,
)

# 绘制区间限速、危险交叉点、部分安全防护曲线和围成的危险速度域
safeguard.render(ax=ax2)

# 绘制辅助停车区、车站
ax2.hlines(
    y=np.zeros_like(accessible_points[:-1]),
    xmin=accessible_points[:-1],
    xmax=dangerous_points[:-1],
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
    xmin=acceleration_zone_start,
    xmax=acceleration_zone_end,
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
