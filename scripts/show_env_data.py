import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from model.ocs import SafeGuardUtility
from utils.data_loader import (
    load_acceleration_zones,
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations,
)
from utils.plot_utils import (
    SCI_EXPORT_PAD_INCHES,
    apply_sci_figure_layout,
    save_sci_figure,
    set_global_plot_style,
)


@dataclass
class TrackEnvironmentData:
    accessible_points: NDArray[np.float64]
    dangerous_points: NDArray[np.float64]
    stations_cor: NDArray[np.float64]
    acceleration_zone_start: float
    acceleration_zone_end: float
    speed_limits: NDArray[np.float64]
    speed_limit_intervals: NDArray[np.float64]
    safeguard: SafeGuardUtility
    slopes: NDArray[np.float64]
    slope_intervals: NDArray[np.float64]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show environment data and safeguard curves."
    )
    _ = parser.add_argument(
        "--view",
        choices=["overview", "full-curves", "danger-region", "all"],
        default="overview",
        help=(
            "View mode to display: "
            "'overview' (default): Combined full curves & track slope; "
            "'full-curves': Full safeguard curves with track infrastructure; "
            "'danger-region': Dangerous speed regions and intersecting points; "
            "'all': Display all three figures."
        ),
    )
    _ = parser.add_argument(
        "--output-file",
        type=Path,
        help=(
            "Path for saving a compact paper-ready figure. "
            "If omitted, only show the figure."
        ),
    )
    _ = parser.add_argument(
        "--dpi",
        type=float,
        default=300.0,
        help="DPI used when saving the figure.",
    )
    _ = parser.add_argument(
        "--pad-inches",
        type=float,
        default=SCI_EXPORT_PAD_INCHES,
        help="Padding around the tight saved figure.",
    )
    _ = parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save without opening the interactive display window.",
    )
    return parser.parse_args()


def set_plot_style():
    _ = set_global_plot_style(
        font_preset="sci",
        preferred_font="Times New Roman",
        title_font_size=8.0,
        axis_label_font_size=8.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=150.0,
        savefig_dpi=300.0,
    )


def load_track_environment_data() -> TrackEnvironmentData:
    accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()

    stations_data = load_stations()
    longyang_start = stations_data["start_station"]["start"]
    longyang_end = stations_data["start_station"]["end"]
    putong_start = stations_data["end_station"]["start"]
    putong_end = stations_data["end_station"]["end"]
    stations_cor = np.array(
        [
            [longyang_start, putong_start],
            [longyang_end, putong_end],
        ],
        dtype=np.float64,
    )

    acceleration_zone_data = load_acceleration_zones()
    acceleration_zone_start = float(acceleration_zone_data["uplink"]["start"])
    acceleration_zone_end = float(acceleration_zone_data["uplink"]["end"])

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

    slopes, slope_intervals = load_slopes()

    return TrackEnvironmentData(
        accessible_points=np.asarray(accessible_points, dtype=np.float64),
        dangerous_points=np.asarray(dangerous_points, dtype=np.float64),
        stations_cor=stations_cor,
        acceleration_zone_start=acceleration_zone_start,
        acceleration_zone_end=acceleration_zone_end,
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        safeguard=safeguard,
        slopes=np.asarray(slopes, dtype=np.float64),
        slope_intervals=np.asarray(slope_intervals, dtype=np.float64),
    )


def _draw_infrastructure_hlines(
    ax,
    data: TrackEnvironmentData,
    *,
    exclude_last_asa: bool = False,
) -> None:
    aps = data.accessible_points[:-1] if exclude_last_asa else data.accessible_points
    dps = data.dangerous_points[:-1] if exclude_last_asa else data.dangerous_points
    ax.hlines(
        y=np.zeros_like(aps),
        xmin=aps,
        xmax=dps,
        colors="green",
        linestyles="solid",
        linewidth=8,
        label="Auxiliary stopping area",
        alpha=0.7,
    )
    ax.hlines(
        y=np.zeros(2),
        xmin=data.stations_cor[0, :],
        xmax=data.stations_cor[1, :],
        colors="blue",
        linestyles="solid",
        linewidth=8,
        label="Station",
        alpha=0.5,
    )
    ax.hlines(
        y=np.zeros(2),
        xmin=data.acceleration_zone_start,
        xmax=data.acceleration_zone_end,
        colors="yellow",
        linestyles="solid",
        linewidth=8,
        label="Acceleration zone",
        alpha=0.5,
    )


def create_overview_figure(data: TrackEnvironmentData) -> Figure:
    """创建综合环境视图：上方为全量防护曲线与设施，下方为轨道坡度阶梯图。"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    apply_sci_figure_layout(
        fig,
        columns=2,
        height_in=4.2,
        left=0.09,
        bottom=0.13,
        top=0.82,
        hspace=0.18,
    )
    data.safeguard.render(ax=ax1, layers=SafeGuardUtility.FULL_CURVE_VIEW_LAYERS)
    _draw_infrastructure_hlines(ax1, data, exclude_last_asa=False)

    ax1.set_xlim((0.0, 30000.0))
    ax1.set_ylim((0.0, 500.0))
    ax1.set_ylabel("Speed (km/h)")

    handles, labels = ax1.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles, strict=False))
    legend_items = [
        "Track speed limit",
        "Maximum speed curve",
        "Safe levitation curve",
        "Auxiliary stopping area",
        "Safe braking curve",
        "Station",
        "Minimum speed curve",
        "Acceleration zone",
    ]
    missing_labels = [
        source_label
        for source_label in legend_items
        if source_label not in handle_by_label
    ]
    if missing_labels:
        raise RuntimeError(f"Missing legend labels: {missing_labels}")

    ordered_handles = [handle_by_label[source_label] for source_label in legend_items]
    ordered_labels = [display_label for display_label in legend_items]

    ax1.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
    )
    ax1.grid(True, alpha=0.3)

    # 绘制轨道坡度
    ax2.stairs(
        values=data.slopes,
        edges=data.slope_intervals,
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
    _ = ax1.text(
        0.02,
        0.98,
        "(a)",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    _ = ax2.text(
        0.02,
        0.98,
        "(b)",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    return fig


def create_full_curves_figure(data: TrackEnvironmentData) -> Figure:
    """创建全量安全防护曲线视图（Safe levitation, safe braking, min/max curves）。"""
    fig, ax = plt.subplots()
    apply_sci_figure_layout(fig, columns=2, height_in=3.2)
    data.safeguard.render(ax=ax, layers=SafeGuardUtility.FULL_CURVE_VIEW_LAYERS)
    _draw_infrastructure_hlines(ax, data, exclude_last_asa=False)

    ax.set_xlim((0.0, 30000.0))
    ax.set_ylim((0.0, 500.0))
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    ax.grid(True, alpha=0.3)

    return fig


def create_danger_region_figure(data: TrackEnvironmentData) -> Figure:
    """创建危险速度域视图（局部防护曲线、危险交叉点散点与危险区域填充）。"""
    fig, ax = plt.subplots()
    apply_sci_figure_layout(fig, columns=2, height_in=3.2)
    data.safeguard.render(ax=ax, layers=SafeGuardUtility.DANGER_VIEW_LAYERS)
    _draw_infrastructure_hlines(ax, data, exclude_last_asa=True)

    ax.set_xlim((0.0, 30000.0))
    ax.set_ylim((0.0, 500.0))
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.grid(True, alpha=0.3)

    return fig


def create_figures(
    view_mode: str, data: TrackEnvironmentData
) -> dict[str, Figure]:
    """根据 view_mode 构建并返回对应的 Figure 字典。"""
    if view_mode == "overview":
        return {"overview": create_overview_figure(data)}
    if view_mode == "full-curves":
        return {"full_curves": create_full_curves_figure(data)}
    if view_mode == "danger-region":
        return {"danger_region": create_danger_region_figure(data)}
    if view_mode == "all":
        return {
            "overview": create_overview_figure(data),
            "full_curves": create_full_curves_figure(data),
            "danger_region": create_danger_region_figure(data),
        }
    raise ValueError(f"Unsupported view mode: {view_mode}")


def save_compact_figures(
    figures: dict[str, Figure],
    output_file: Path,
    dpi: float,
    pad_inches: float,
) -> list[Path]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_file.suffix if output_file.suffix else ".png"
    saved_paths: list[Path] = []

    if len(figures) == 1:
        single_fig = next(iter(figures.values()))
        target_path = output_file.with_suffix(suffix)
        _ = save_sci_figure(single_fig, target_path, dpi=dpi, pad_inches=pad_inches)
        saved_paths.append(target_path)
    else:
        base_stem = output_file.stem
        for key, fig in figures.items():
            target_path = output_file.parent / f"{base_stem}_{key}{suffix}"
            _ = save_sci_figure(fig, target_path, dpi=dpi, pad_inches=pad_inches)
            saved_paths.append(target_path)

    return saved_paths


def main():
    args = parse_args()
    set_plot_style()
    data = load_track_environment_data()
    figures = create_figures(args.view, data)

    if args.output_file is not None:
        saved_files = save_compact_figures(
            figures,
            args.output_file,
            dpi=args.dpi,
            pad_inches=args.pad_inches,
        )
        for saved_file in saved_files:
            print(f"Saved figure to {saved_file}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
