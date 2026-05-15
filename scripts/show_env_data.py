import matplotlib.pyplot as plt
import numpy as np

from model.ocs import SafeGuardUtility
from utils.data_loader import (
    load_acceleration_zones,
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations,
)
from utils.plot_utils import set_global_plot_style

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
stations_cor = np.array([
    [longyang_start, putong_start],
    [longyang_end, putong_end],
])

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

safeguard = SafeGuardUtility(
    speed_limits=speed_limits,
    speed_limit_intervals=speed_limit_intervals,
    levi_curves_list=levi_curves_list,
    brake_curves_list=brake_curves_list,
    min_curves_list=min_curves_list,
    max_curves_list=max_curves_list,
    factor=0.99,
)

set_global_plot_style(
    font_preset="sci",
    preferred_font="Calibri",
    title_font_size=8.0,
    axis_label_font_size=8.0,
    tick_font_size=8.0,
    legend_font_size=8.0,
    figure_dpi=150.0,
    savefig_dpi=300.0,
)

# 绘制区间限速、安全悬浮曲线、安全制动曲线、最小速度曲线
# 最大速度曲线、辅助停车区、车站、加速区
fig1, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
)
safeguard.render(ax=ax1, layers=SafeGuardUtility.FULL_CURVE_VIEW_LAYERS)
ax1.hlines(
    y=np.zeros_like(accessible_points),
    xmin=accessible_points,
    xmax=dangerous_points,
    colors="green",
    linestyles="solid",
    linewidth=8,
    label="Auxiliary Stopping Area",
    alpha=0.7,
)
ax1.hlines(
    y=np.zeros(2),
    xmin=stations_cor[0, :],
    xmax=stations_cor[1, :],
    colors="blue",
    linestyles="solid",
    linewidth=8,
    label="Station",
    alpha=0.5,
)
ax1.hlines(
    y=np.zeros(2),
    xmin=acceleration_zone_start,
    xmax=acceleration_zone_end,
    colors="yellow",
    linestyles="solid",
    linewidth=8,
    label="Acceleration Zone",
    alpha=0.5,
)
ax1.set_xlim((0.0, 30000.0))
ax1.set_ylim((0.0, 500.0))
ax1.set_ylabel("Speed(km/h)")
ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5)

# 绘制轨道坡度
slopes, slope_intervals = load_slopes()
ax2.stairs(
    values=slopes,
    edges=slope_intervals,
    color="saddlebrown",
    linewidth=1.0,
    fill=True,
    alpha=0.8,
    label="Slope",
)
ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
ax2.set_xlim((0.0, 30000.0))
ax2.set_xlabel("Position (m)")
ax2.set_ylabel("Slope (‰)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.show()
