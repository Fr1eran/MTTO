from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from model.common.energy_consumption_calculator import ECC
from model.ocs import SafeGuardUtility
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from scripts.transform_real_operation_curve import (
    DEFAULT_INPUT_FILE,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_SHEET_NAME,
    load_real_operation_curve,
)
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
    resolve_project_path,
)
from utils.plot_utils import apply_sci_figure_layout, set_chinese_font
from utils.scenario import build_safeguard_utility

ALIGNED_CURVE_REQUIRED_KEYS = ("position_m", "speed_mps", "acc_mps2", "time_s")


def _format_meter_axis_as_km(ax: Axes) -> None:
    def _km_formatter(x: float, _pos: object) -> str:
        return f"{x / 1000:g}"

    ax.xaxis.set_major_formatter(FuncFormatter(_km_formatter))
    _ = ax.set_xlabel(r"里程($km$)")


def load_operation_samples() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    aligned_curve_path = resolve_project_path(DEFAULT_OUTPUT_FILE)
    if aligned_curve_path.is_file():
        with np.load(aligned_curve_path, allow_pickle=False) as curve_data:
            missing_keys = [
                key
                for key in ALIGNED_CURVE_REQUIRED_KEYS
                if key not in curve_data.files
            ]
            if missing_keys:
                raise ValueError(
                    f"Aligned operation curve missing arrays: {missing_keys}"
                )
            distance_m = np.asarray(curve_data["position_m"], dtype=np.float64)
            speed_mps = np.asarray(curve_data["speed_mps"], dtype=np.float64)
            acceleration = np.asarray(curve_data["acc_mps2"], dtype=np.float64)
            travel_time_s = np.asarray(curve_data["time_s"], dtype=np.float64)
    else:
        start_position_m, target_position_m = load_stations_goal_positions()
        curve_arrays = load_real_operation_curve(
            input_file=DEFAULT_INPUT_FILE,
            sheet_name=DEFAULT_SHEET_NAME,
            start_position_m=start_position_m,
            target_position_m=target_position_m,
        )
        distance_m = np.asarray(curve_arrays["position_m"], dtype=np.float64)
        speed_mps = np.asarray(curve_arrays["speed_mps"], dtype=np.float64)
        acceleration = np.asarray(curve_arrays["acc_mps2"], dtype=np.float64)
        travel_time_s = np.asarray(curve_arrays["time_s"], dtype=np.float64)

    speed_kmh = speed_mps * 3.6
    return distance_m, speed_kmh, acceleration, travel_time_s


def build_track() -> TrackInfo:
    slopes, slope_intervals = load_slopes()
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)
    accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()
    return TrackInfo(
        slopes=slopes,
        slope_intervals=slope_intervals,
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        ASA_aps=accessible_points,
        ASA_dps=dangerous_points,
    )


def print_operation_summary(
    *,
    distance_m: np.ndarray,
    speed_mps: np.ndarray,
    travel_time_s: np.ndarray,
    propulsion_energy_consumption: np.ndarray,
    leviation_energy_consumption: np.ndarray,
) -> None:
    total_time_s = float(travel_time_s[-1] - travel_time_s[0])
    propulsion_energy_kj = float(propulsion_energy_consumption[-1])
    leviation_energy_kj = float(leviation_energy_consumption[-1])
    total_energy_kj = propulsion_energy_kj + leviation_energy_kj

    print("实际运行曲线统计（重标定后位置坐标）:")
    print(f"  样本数: {distance_m.size}")
    print(f"  起点位置: {float(distance_m[0]):.3f} m")
    print(f"  终点位置: {float(distance_m[-1]):.3f} m")
    print(f"  实际运行时间: {total_time_s:.3f} s")
    print(f"  初始速度: {float(speed_mps[0]):.3f} m/s")
    print(f"  终点速度: {float(speed_mps[-1]):.3f} m/s")
    print(f"  牵引能耗: {propulsion_energy_kj:.3f} kJ")
    print(f"  悬浮能耗: {leviation_energy_kj:.3f} kJ")
    print(f"  总能耗: {total_energy_kj:.3f} kJ")


def main() -> None:
    # 线路/防护/能耗模型均使用 m 与 m/s；绘图刻度再格式化成 km。
    distance_m, speed_kmh, acceleration, travel_time_s = load_operation_samples()
    speed_mps = speed_kmh / 3.6

    set_chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    safeguardutility = build_safeguard_utility()

    fig1, ax1 = plt.subplots()
    _ = ax1.plot(
        distance_m,
        speed_kmh,
        label="重标定后实际运行速度随里程变化曲线",
        color="blue",
    )
    safeguardutility.render(ax=ax1, layers=SafeGuardUtility.DANGER_VIEW_LAYERS)
    _format_meter_axis_as_km(ax1)
    _ = ax1.set_ylabel(r"速度($km/h$)")
    _ = ax1.set_title("龙阳路到浦东国际机场重标定后实际运行速度-里程曲线")
    ax1.grid(True, alpha=0.3)
    _ = ax1.legend()

    fig2, ax2 = plt.subplots()
    _ = ax2.plot(
        distance_m,
        acceleration,
        label="重标定后实际加速度随里程变化曲线",
        color="green",
    )
    _format_meter_axis_as_km(ax2)
    _ = ax2.set_ylabel(r"加速度($m/s^2$)")
    _ = ax2.set_title("龙阳路到浦东国际机场重标定后实际加速度-里程曲线")
    ax2.grid(True, alpha=0.3)
    _ = ax2.legend()

    track = build_track()
    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    tec = ECC(
        R_m=0.2796,
        L_d=0.0002,
        R_k=50.0,
        L_k=0.000142,
        Tau=0.258,
        Psi_fd=3.9629,
        k_c=0.8,
    )

    propulsion_energy_consumption, leviation_energy_consumption = (
        tec.calc_energy_cumulative(
            pos_arr=distance_m,
            speed_arr=speed_mps,
            acc_arr=acceleration,
            vehicle=vehicle,
            track=track,
            travel_time_arr=travel_time_s,
        )
    )

    print_operation_summary(
        distance_m=distance_m,
        speed_mps=speed_mps,
        travel_time_s=travel_time_s,
        propulsion_energy_consumption=propulsion_energy_consumption,
        leviation_energy_consumption=leviation_energy_consumption,
    )

    fig3, ax3 = plt.subplots()
    _ = ax3.plot(
        distance_m,
        propulsion_energy_consumption,
        label="重标定后实际牵引能耗随里程变化曲线",
        color="red",
    )
    _ = ax3.plot(
        distance_m,
        leviation_energy_consumption,
        label="重标定后实际悬浮能耗随里程变化曲线",
        color="green",
    )
    _format_meter_axis_as_km(ax3)
    _ = ax3.set_ylabel(r"能耗($kJ$)")
    _ = ax3.legend()
    ax3.grid(True, alpha=0.3)
    _ = ax3.set_title("龙阳路到浦东国际机场重标定后实际能耗-里程曲线")

    for fig in (fig1, fig2, fig3):
        apply_sci_figure_layout(fig, columns=2, height_in=3.4)

    plt.show()


if __name__ == "__main__":
    main()
