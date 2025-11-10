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
from utils.curve import CalRegions, ConcatenateCurvesWithNaN
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

levi_curves_list = cal_SGC.CalLeviCurves(aps, vehicle, 1)
brake_curves_list = cal_SGC.CalBrakeCurves(dps, vehicle, 1)


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 将所有的坐标值数组列表合并为一个数组并在相邻段之间插入np.nan
levi_curves_pos_con, levi_curves_speed_con = ConcatenateCurvesWithNaN(levi_curves_list)
brake_curves_pos_con, brake_curves_speed_con = ConcatenateCurvesWithNaN(
    brake_curves_list
)
# 绘制区间限速、安全悬浮曲线、安全制动曲线、辅助停车区、车站、加速区
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
    alpha=0.7,
)
ax1.plot(
    brake_curves_pos_con,
    brake_curves_speed_con * 3.6,
    label="安全制动曲线",
    color="red",
    alpha=0.7,
)
ax1.set_xlim((0.0, 30000.0))
ax1.set_ylim((0.0, 500.0))
ax1.set_xlabel(r"位置$s\left( m \right)$")
ax1.set_ylabel(r"速度$v\left( km/h \right)$")
ax1.set_title("区间限速与安全防护曲线")
ax1.legend()


# 计算危险交叉点和危险交叉点之后的部分防护曲线
idp_points, levi_curves_part, brake_curves_part = CalRegions(
    levi_curves_list, brake_curves_list[:-1]
)
np.save(file="data/rail/safeguard/idp_points.npy", arr=idp_points)

with open("data/rail/safeguard/levi_curves_part.pkl", "wb") as f:
    pickle.dump(levi_curves_part, f)

with open("data/rail/safeguard/brake_curves_part.pkl", "wb") as f:
    pickle.dump(brake_curves_part, f)

# 读取危险交叉点和危险交叉点之后的部分防护曲线
# idp_points = np.load(file='data/rail/safeguard/idp_points.npy')

# with open('data/rail/safeguard/levi_curves_part.pkl', 'rb') as f:
#     levi_curves_part = pickle.load(f)

# with open('data/rail/safeguard/brake_curves_part.pkl', 'rb') as f:
#     brake_curves_part = pickle.load(f)

# 将组成危险速度域曲线对的数据点个数对齐并保存
# levi_curves_part_padded, brake_curves_part_padded = Padding2CurvesList(
#     levi_curves_part, brake_curves_part
# )

# with open("data/rail/safeguard/levi_curves_part_padded.pkl", "wb") as f:
#     pickle.dump(levi_curves_part_padded, f)

# with open("data/rail/safeguard/brake_curves_part_padded.pkl", "wb") as f:
#     pickle.dump(brake_curves_part_padded, f)


# 读取对齐后的防护曲线
# with open('data/rail/safeguard/levi_curves_part_padded.pkl', 'rb') as f:
#     levi_curves_part_padded = pickle.load(f)

# with open('data/rail/safeguard/brake_curves_part_padded.pkl', 'rb') as f:
#     brake_curves_part_padded = pickle.load(f)

# 绘制危险交叉点后的安全防护曲线和围成的危险速度域
fig2 = plt.figure(dpi=100)
ax2 = fig2.add_subplot()

safeguard = SafeGuardUtility(
    speed_limits=speed_limits,
    speed_limit_intervals=speed_limit_intervals,
    idp_points=idp_points,
    levi_curves_part_list=levi_curves_part,
    brake_curves_part_list=brake_curves_part,
)

# 绘制区间限速、危险交叉点、部分安全防护曲线和围成的危险速度域
safeguard.render(ax=ax2)

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

# 在危险速度域图像中显示光标的坐标
# 定义一个文本对象，用于显示坐标
# coord_text = ax2.text(
#     0.02,
#     0.98,
#     "",
#     transform=ax2.transAxes,
#     verticalalignment="top",
#     fontsize=12,
#     bbox=dict(facecolor="white", alpha=0.7),
# )

# def on_mouse_move(event):
#     # 检查使劲按是否发生在当前坐标轴内
#     if event.inaxes == ax2:
#         x_val = event.xdata
#         y_val = event.ydata
#         coord_text.set_text(f"x = {x_val:.2f}, y = {y_val:.2f}")
#     else:
#         coord_text.set_text("")
#     fig2.canvas.draw_idle()  # 刷新画布

# 绑定鼠标移动事件
# fig2.canvas.mpl_connect("motion_notify_event", on_mouse_move)

plt.show()
