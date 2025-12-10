import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.Vehicle import Vehicle
from utils.CalcEnergy import CalcEnergyCumul
from model.Track import Track, TrackProfile
from utils.misc import SetChineseFont

# 绘制龙阳路到浦东国际机场的实际运行速度-里程曲线
raw_data = pd.read_excel(
    "data/operation/a_longyang_to_airport.xlsx",
    sheet_name="a轨_双端两步4节_龙阳－机场",
    header=0,
    dtype=np.float32,
)

distance = raw_data["里程(km)"][1:]
v_km = raw_data["速度(km/h)"][1:]
accelerate = raw_data["加速度(m/s2)"][1:]
travel_time = raw_data["时间(s)"][1:]

SetChineseFont()

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

# 坡度，百分位
with open("data/rail/raw/slopes.json", "r", encoding="utf-8") as f:
    slope_data = json.load(f)
    slopes = slope_data["slopes"]
    slope_intervals = slope_data["intervals"]

# 区间限速
with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
    speed_limit_data = json.load(f)
    speed_limits = speed_limit_data["speed_limits"]
    speed_limits = np.asarray(speed_limits) / 3.6
    speed_limit_intervals = speed_limit_data["intervals"]


track = Track(slopes, slope_intervals, speed_limits.tolist(), speed_limit_intervals)
trackprofile = TrackProfile(track=track)

vehicle = Vehicle(mass=317.5, numoftrainsets=5, length=128.5)

mechanic_energy_consumption, levi_energy_consumption = CalcEnergyCumul(
    pos_arr=distance.to_numpy(dtype=np.float64),
    speed_arr=v_km.to_numpy(dtype=np.float64) / 3.6,
    acc_arr=accelerate.to_numpy(dtype=np.float64),
    vehicle=vehicle,
    trackprofile=trackprofile,
    travel_time_arr=travel_time.to_numpy(dtype=np.float64),
)
# 绘制列车能耗随里程变化曲线
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(
    distance,
    mechanic_energy_consumption,
    label="实际机械能耗随里程变化曲线",
    color="red",
)
ax3.plot(
    distance,
    levi_energy_consumption,
    label="实际悬浮能耗随里程变化曲线",
    color="yellow",
)
ax3.set_xlabel(r"里程($km$)")
ax3.set_ylabel(r"能耗($kJ$)")
ax3.legend()
ax3.set_title("龙阳路到浦东国际机场实际能耗-里程曲线")

plt.show()
