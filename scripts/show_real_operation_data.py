import matplotlib.pyplot as plt
import numpy as np

from model.vehicle.vehicle import VehicleInfo
from model.common.energy_consumption_calculator import ECC
from model.track.track import TrackInfo, TrackProfile
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_excel,
    load_slopes,
    load_speed_limits,
)
from utils.plot_utils import set_chinese_font

# 绘制龙阳路到浦东国际机场的实际运行速度-里程曲线
raw_data = load_excel(
    "data/operation/a_longyang_to_airport.xlsx",
    sheet_name="a轨_双端两步4节_龙阳－机场",
    header=0,
    dtype=np.float32,
)

distance = raw_data["里程(km)"][1:]
v_km = raw_data["速度(km/h)"][1:]
accelerate = raw_data["加速度(m/s2)"][1:]
travel_time = raw_data["时间(s)"][1:]

set_chinese_font()

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(distance, v_km, label="实际运行速度随里程变化曲线", color="blue")
ax1.set_xlabel(r"里程($km$)")
ax1.set_ylabel(r"速度($km/h$)")
ax1.set_title("龙阳路到浦东国际机场实际运行速度-里程曲线")

# 绘制加速度随里程变化曲线
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(distance, accelerate, label="实际加速度随里程变化曲线", color="green")
ax2.set_xlabel(r"里程($km$)")
ax2.set_ylabel(r"加速度($m/s^2$)")
ax2.set_title("龙阳路到浦东国际机场实际加速度-里程曲线")

slopes, slope_intervals = load_slopes()
speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)
accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()


track = TrackInfo(
    slopes,
    slope_intervals,
    speed_limits.tolist(),
    speed_limit_intervals,
    accessible_points,
    dangerous_points,
)
trackprofile = TrackProfile(track=track)

vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)

tec = ECC(
    R_m=0.2796, L_d=0.0002, R_k=50.0, L_k=0.000142, Tau=0.258, Psi_fd=3.9629, k_c=0.8
)

propulsion_energy_consumption, leviation_energy_consumption = (
    tec.calc_energy_cumulative(
        pos_arr=distance.to_numpy(dtype=np.float64),
        speed_arr=v_km.to_numpy(dtype=np.float64) / 3.6,
        acc_arr=accelerate.to_numpy(dtype=np.float64),
        vehicle=vehicle,
        trackprofile=trackprofile,
        travel_time_arr=travel_time.to_numpy(dtype=np.float64),
    )
)
# 绘制列车能耗随里程变化曲线
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(
    distance,
    propulsion_energy_consumption,
    label="实际牵引能耗随里程变化曲线",
    color="red",
)
ax3.plot(
    distance,
    leviation_energy_consumption,
    label="实际悬浮能耗随里程变化曲线",
    color="green",
)
ax3.set_xlabel(r"里程($km$)")
ax3.set_ylabel(r"能耗($kJ$)")
ax3.legend()
ax3.set_title("龙阳路到浦东国际机场实际能耗-里程曲线")

plt.show()
