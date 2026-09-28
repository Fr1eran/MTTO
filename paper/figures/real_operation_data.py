from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from paper.figures import load_paper_scenario, load_paper_task
from paper.plotting.profiles import (
    DANGER_VIEW_LAYERS,
    render_safeguard,
)
from paper.plotting.style import (
    apply_sci_figure_layout,
    apply_sci_grid,
    set_chinese_font,
)
from paper.real_operation import real_operation_profile


def _format_meter_axis_as_km(ax: Axes) -> None:
    def _km_formatter(x: float, _pos: object) -> str:
        return f"{x / 1000:g}"

    ax.xaxis.set_major_formatter(FuncFormatter(_km_formatter))
    _ = ax.set_xlabel(r"里程($km$)")


def print_operation_summary(
    *,
    distance_m: np.ndarray,
    speed_mps: np.ndarray,
    travel_time_s: np.ndarray,
    propulsion_energy_consumption: np.ndarray,
    leviation_energy_consumption: np.ndarray,
) -> None:
    total_time_s = float(travel_time_s[-1] - travel_time_s[0])
    propulsion_energy_kwh = float(propulsion_energy_consumption[-1]) / 3600.0
    leviation_energy_kwh = float(leviation_energy_consumption[-1]) / 3600.0
    total_energy_kwh = propulsion_energy_kwh + leviation_energy_kwh

    print("实际运行曲线统计（重标定后位置坐标）:")
    print(f"  样本数: {distance_m.size}")
    print(f"  起点位置: {float(distance_m[0]):.3f} m")
    print(f"  终点位置: {float(distance_m[-1]):.3f} m")
    print(f"  实际运行时间: {total_time_s:.3f} s")
    print(f"  初始速度: {float(speed_mps[0]):.3f} m/s")
    print(f"  终点速度: {float(speed_mps[-1]):.3f} m/s")
    print(f"  牵引能耗: {propulsion_energy_kwh:.3f} kWh")
    print(f"  悬浮能耗: {leviation_energy_kwh:.3f} kWh")
    print(f"  总能耗: {total_energy_kwh:.3f} kWh")


def main() -> None:
    # 线路/防护/能耗模型均使用 m 与 m/s；绘图刻度再格式化成 km。
    scenario = load_paper_scenario()
    task = load_paper_task()
    profile = real_operation_profile(scenario, task)
    distance_m = profile.position_m
    speed_mps = profile.speed_mps
    speed_kmh = speed_mps * 3.6
    acceleration = np.concatenate(
        (
            profile.segment_acceleration_mps2[:1],
            profile.segment_acceleration_mps2,
        )
    )
    travel_time_s = profile.time_s

    set_chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    safeguardutility = scenario.safeguard

    fig1, ax1 = plt.subplots()
    _ = ax1.plot(
        distance_m,
        speed_kmh,
        label="重标定后实际运行速度随里程变化曲线",
        color="blue",
    )
    render_safeguard(safeguardutility, ax=ax1, layers=DANGER_VIEW_LAYERS)
    _format_meter_axis_as_km(ax1)
    _ = ax1.set_ylabel(r"速度($km/h$)")
    _ = ax1.set_title("龙阳路到浦东国际机场重标定后实际运行速度-里程曲线")
    apply_sci_grid(ax1)
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
    apply_sci_grid(ax2)
    _ = ax2.legend()

    propulsion_energy_consumption = profile.propulsion_energy_kj
    leviation_energy_consumption = profile.levitation_energy_kj

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
        propulsion_energy_consumption / 3600.0,
        label="重标定后实际牵引能耗随里程变化曲线",
        color="red",
    )
    _ = ax3.plot(
        distance_m,
        leviation_energy_consumption / 3600.0,
        label="重标定后实际悬浮能耗随里程变化曲线",
        color="green",
    )
    _format_meter_axis_as_km(ax3)
    _ = ax3.set_ylabel(r"能耗($kWh$)")
    _ = ax3.legend()
    apply_sci_grid(ax3)
    _ = ax3.set_title("龙阳路到浦东国际机场重标定后实际能耗-里程曲线")

    for fig in (fig1, fig2, fig3):
        apply_sci_figure_layout(fig, columns=2, height_in=3.4)

    plt.show()


if __name__ == "__main__":
    main()
