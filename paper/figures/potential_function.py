import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from mtto.rl.rewards import (
    PUNCTUALITY_POTENTIAL_SIGMA_S,
    SAFETY_RESERVE_UPPER_SCALE,
    RewardCalculator,
    RewardConfig,
    braking_reserve_steps,
    punctuality_potential_from_error_array,
    safety_potential_array,
    traction_reserve_steps,
)
from mtto.workflows.train import build_env_references
from paper.figures import load_paper_scenario, load_paper_task
from paper.plotting.style import (
    PAPER_LEGEND_FONT_SIZE,
    add_panel_label,
    apply_paper_style,
    apply_sci_grid,
    save_sci_figure,
    sci_figure_size,
)

SAFETY_POTENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "mtto_safety_penalty",
    [
        (0.00, "#E65100"),
        (0.35, "#F57C00"),
        (0.65, "#FFA726"),
        (0.88, "#FFD54F"),
        (0.97, "#FFF8E1"),
        (1.00, "#FFFDF9"),
    ],
)
SAFETY_POTENTIAL_CMAP.set_bad(color="white", alpha=1.0)
# Paper control period: the safety potential looks one control period ahead.
STEP_TIME_S = 1.0
# The maximum penalty attainable along any trajectory is max(K_u, K_l) = 3.0,
# as upper and lower boundaries cannot be approached simultaneously.
SAFETY_POTENTIAL_VMIN = -float(SAFETY_RESERVE_UPPER_SCALE)
PUNCTUALITY_POTENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "mtto_punctuality_penalty",
    [
        (0.00, "#004B87"),
        (0.30, "#1270AE"),
        (0.60, "#4298CB"),
        (0.80, "#8AC5E6"),
        (0.92, "#CDE7F5"),
        (1.00, "#FFFFFF"),
    ],
)
PUNCTUALITY_POTENTIAL_CMAP.set_bad(color="white", alpha=1.0)


@dataclass(frozen=True)
class _SafetyPotentialField:
    """第 7 个辅助停车区内安全势函数的位置、速度网格与边界。"""

    pos_array: np.ndarray
    speed_array_mps: np.ndarray
    position_grid: np.ndarray
    speed_grid_mps: np.ndarray
    min_speed_grid_mps: np.ndarray
    max_speed_grid_mps: np.ndarray
    min_speed_profile_mps: np.ndarray
    max_speed_profile_mps: np.ndarray
    feasible_mask: np.ndarray


@dataclass(frozen=True)
class _PunctualityPotentialField:
    """Full-route punctuality-potential grid derived from runtime semantics."""

    position_m: np.ndarray
    redundant_time_s: np.ndarray
    position_grid_m: np.ndarray
    redundant_time_grid_s: np.ndarray
    reference_slack_s: np.ndarray
    potential: np.ndarray


def interp_with_constant_fill(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    query: NDArray[np.floating] | float,
    left_value: float,
    right_value: float,
) -> NDArray[np.floating] | np.floating:
    """使用 numpy.interp 做线性插值，并在区间外用常量填充。"""
    return np.interp(
        query,
        x,
        y,
        left=left_value,
        right=right_value,
    )


def _build_safety_potential_field(
    *,
    position_points: int = 1200,
    speed_points: int = 800,
    position_window_m: tuple[float, float] | None = None,
    speed_window_mps: tuple[float, float] | None = None,
) -> _SafetyPotentialField:
    """构造第 7 个辅助停车区内安全势函数的状态域。

    The speed profiles always span the whole stopping area, so the one-period
    look-ahead stays exact; the optional windows only restrict the grid.
    """
    scenario = load_paper_scenario()
    min_curves_list = scenario.safeguard.min_curves
    max_curves_list = scenario.safeguard.max_curves
    min_curve = min_curves_list[6]
    max_curve = max_curves_list[7]
    min_curve_pos, min_curve_speed = min_curve[0, :], min_curve[1, :]
    max_curve_pos, max_curve_speed = max_curve[0, :], max_curve[1, :]

    pos_array = np.linspace(
        float(max_curve_pos[0]),
        float(max_curve_pos[-1]),
        position_points,
    )
    min_speed_profile_mps = np.maximum(
        interp_with_constant_fill(
            min_curve_pos,
            min_curve_speed,
            pos_array,
            left_value=0.0,
            right_value=0.0,
        ),
        0.0,
    )
    track_speed_limits_mps = scenario.line.speed_limits
    speed_limit_intervals_m = scenario.line.speed_limit_intervals
    track_limit_indices = np.clip(
        np.searchsorted(speed_limit_intervals_m, pos_array, side="right") - 1,
        0,
        track_speed_limits_mps.size - 1,
    )
    track_speed_profile_mps = track_speed_limits_mps[track_limit_indices]
    safeguard_max_profile_mps = interp_with_constant_fill(
        max_curve_pos,
        max_curve_speed,
        pos_array,
        left_value=np.inf,
        right_value=float(max_curve_speed[-1]),
    )
    max_speed_profile_mps = np.maximum(
        np.minimum(track_speed_profile_mps, safeguard_max_profile_mps),
        0.0,
    )
    speed_low, speed_high = speed_window_mps or (
        0.0,
        float(np.max(max_speed_profile_mps)),
    )
    speed_array_mps = np.linspace(speed_low, speed_high, speed_points)
    in_window = np.ones(pos_array.shape, dtype=bool)
    if position_window_m is not None:
        in_window = (pos_array >= position_window_m[0]) & (
            pos_array <= position_window_m[1]
        )
    position_grid, speed_grid_mps = np.meshgrid(pos_array[in_window], speed_array_mps)
    min_speed_grid_mps = np.broadcast_to(
        min_speed_profile_mps[in_window], position_grid.shape
    )
    max_speed_grid_mps = np.broadcast_to(
        max_speed_profile_mps[in_window], position_grid.shape
    )
    feasible_mask = (speed_grid_mps >= min_speed_grid_mps) & (
        speed_grid_mps <= max_speed_grid_mps
    )
    return _SafetyPotentialField(
        pos_array=pos_array,
        speed_array_mps=speed_array_mps,
        position_grid=position_grid,
        speed_grid_mps=speed_grid_mps,
        min_speed_grid_mps=min_speed_grid_mps,
        max_speed_grid_mps=max_speed_grid_mps,
        min_speed_profile_mps=min_speed_profile_mps,
        max_speed_profile_mps=max_speed_profile_mps,
        feasible_mask=feasible_mask,
    )


def _calculate_safety_potential(
    field: _SafetyPotentialField,
) -> np.ndarray:
    """只在速度上下限约束内，按运行时的制动储备公式计算安全势函数。"""
    vehicle = load_paper_scenario().vehicle
    # Farthest position reachable within one control period, per grid point.
    ahead_m = (
        field.position_grid
        + field.speed_grid_mps * STEP_TIME_S
        + 0.5 * vehicle.max_acc * STEP_TIME_S**2
    )
    shape = field.position_grid.shape
    max_ahead_grid = np.interp(ahead_m, field.pos_array, field.max_speed_profile_mps)
    min_ahead_grid = np.interp(ahead_m, field.pos_array, field.min_speed_profile_mps)

    mask = field.feasible_mask
    speed = field.speed_grid_mps[mask]
    braking = np.vectorize(braking_reserve_steps, otypes=[np.float64])(
        speed,
        field.max_speed_grid_mps[mask],
        max_ahead_grid[mask],
        vehicle.max_dec_abs,
        STEP_TIME_S,
    )
    traction = np.vectorize(traction_reserve_steps, otypes=[np.float64])(
        speed,
        field.min_speed_grid_mps[mask],
        min_ahead_grid[mask],
        vehicle.max_acc,
        STEP_TIME_S,
    )
    values = np.full(shape, np.nan)
    values[mask] = safety_potential_array(braking, traction)
    return values


def _build_punctuality_potential_field(
    *,
    schedule_time_s: float | None = None,
    position_points: int = 600,
    redundant_time_points: int = 400,
) -> _PunctualityPotentialField:
    """Build the full-route field used by the runtime punctuality potential."""
    if schedule_time_s is not None and (
        not np.isfinite(schedule_time_s) or schedule_time_s <= 0.0
    ):
        raise ValueError("schedule_time_s must be finite and positive")
    scenario = load_paper_scenario()
    task = load_paper_task(schedule_time_s=schedule_time_s)
    _, normalization = build_env_references(scenario, task)
    calculator = RewardCalculator(
        normalization,
        gamma=0.998,
        step_time_s=STEP_TIME_S,
        reward_config=RewardConfig(enable_potential_punctuality=True),
    )
    position_m = np.linspace(
        task.start_position_m,
        task.target_position_m,
        position_points,
    )
    reference_slack_s = np.asarray(
        [
            calculator.reference_punctuality_slack(
                float(pos), schedule_time_s=task.schedule_time_s, task=task
            )
            for pos in position_m
        ],
        dtype=np.float64,
    )
    margin_s = 3.0 * PUNCTUALITY_POTENTIAL_SIGMA_S
    redundant_time_s = np.linspace(
        float(np.min(reference_slack_s) - margin_s),
        float(np.max(reference_slack_s) + margin_s),
        redundant_time_points,
    )
    position_grid_m, redundant_time_grid_s = np.meshgrid(position_m, redundant_time_s)
    error_s = redundant_time_grid_s - reference_slack_s[np.newaxis, :]
    potential = punctuality_potential_from_error_array(error_s)
    return _PunctualityPotentialField(
        position_m=position_m,
        redundant_time_s=redundant_time_s,
        position_grid_m=position_grid_m,
        redundant_time_grid_s=redundant_time_grid_s,
        reference_slack_s=reference_slack_s,
        potential=potential,
    )


def _draw_punctuality_potential(
    ax: Axes,
    field: _PunctualityPotentialField,
) -> tuple[object, Line2D]:
    mesh = ax.pcolormesh(
        field.position_grid_m / 1000.0,
        field.redundant_time_grid_s,
        field.potential,
        cmap=PUNCTUALITY_POTENTIAL_CMAP,
        shading="auto",
        vmin=float(np.min(field.potential)),
        vmax=0.0,
        rasterized=True,
    )
    reference_line = ax.plot(
        field.position_m / 1000.0,
        field.reference_slack_s,
        color="black",
        linestyle="--",
        linewidth=1.4,
    )[0]
    ax.set_xlim(field.position_m[0] / 1000.0, field.position_m[-1] / 1000.0)
    ax.set_ylim(field.redundant_time_s[0], field.redundant_time_s[-1])
    return mesh, reference_line


def _draw_safety_potential(ax: Axes, field: _SafetyPotentialField) -> object:
    """Colour the safety potential over position (km) and speed (km/h)."""
    return ax.pcolormesh(
        field.position_grid / 1000.0,
        field.speed_grid_mps * 3.6,
        _calculate_safety_potential(field),
        cmap=SAFETY_POTENTIAL_CMAP,
        shading="auto",
        vmin=SAFETY_POTENTIAL_VMIN,
        vmax=0.0,
        rasterized=True,
    )


def _plot_safety_boundaries(
    ax: Axes,
    field: _SafetyPotentialField,
) -> tuple[Line2D, Line2D]:
    """绘制安全势函数使用的速度上下边界。"""
    min_speed_line = ax.plot(
        field.pos_array / 1000.0,
        field.min_speed_profile_mps * 3.6,
        color="tab:blue",
        linewidth=1.2,
    )[0]
    max_speed_line = ax.plot(
        field.pos_array / 1000.0,
        field.max_speed_profile_mps * 3.6,
        color="tab:red",
        linewidth=1.2,
    )[0]
    _ = ax.set_xlim(field.pos_array[0] / 1000.0, field.pos_array[-1] / 1000.0)
    _ = ax.set_ylim(0.0, field.speed_array_mps[-1] * 3.6)
    return min_speed_line, max_speed_line


def _apply_minimal_axis_style(ax: Axes) -> None:
    ax.grid(False)
    ax.set_axis_on()
    ax.axison = True


def _apply_transparent_background(fig: Figure) -> None:
    """设置不透明纯白背景以符合期刊规范并避免 PDF 产生透明对象。"""
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.patch.set_alpha(1.0)

        # 3D 坐标轴 pane 设置不透明纯白。
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis_obj = getattr(ax, axis_name, None)
            pane = getattr(axis_obj, "pane", None)
            if pane is not None:
                pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
                pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))


# Zoom windows of the paper figure (positions in m, speeds in km/h): one
# position stretch, once near the maximum and once near the minimum speed curve.
ZOOM_POSITION_WINDOW_M = (13_800.0, 14_300.0)
ZOOM_UPPER_SPEED_WINDOW_KMH = (290.0, 350.0)
ZOOM_LOWER_SPEED_WINDOW_KMH = (90.0, 150.0)
# Inset placement in axes fractions of panel (a): the upper zoom sits in the
# empty corner above the maximum speed curve, the lower zoom in the zero-potential
# interior between the curves.
ZOOM_UPPER_INSET_BOUNDS = (0.69, 0.61, 0.28, 0.36)
ZOOM_LOWER_INSET_BOUNDS = (0.07, 0.38, 0.29, 0.38)


def plot_safety_potential_heatmap_speed(*, minimal: bool = False) -> Figure:
    """以第 7 个辅助停车区绘制独立安全势函数（含放大插图）。"""
    field = _build_safety_potential_field()
    fig, ax = plt.subplots(figsize=sci_figure_size(columns="text", height_in=3.1))
    safety_mesh = _draw_safety_potential(ax, field)
    min_speed_line, max_speed_line = _plot_safety_boundaries(ax, field)

    position_window_km = tuple(value / 1000.0 for value in ZOOM_POSITION_WINDOW_M)
    for bounds, speed_window, zoom_label, label_pos in (
        (
            ZOOM_UPPER_INSET_BOUNDS,
            ZOOM_UPPER_SPEED_WINDOW_KMH,
            r"Zoom: $v_{\max}$",
            (0.06, 0.16),
        ),
        (
            ZOOM_LOWER_INSET_BOUNDS,
            ZOOM_LOWER_SPEED_WINDOW_KMH,
            r"Zoom: $v_{\min}$",
            (0.06, 0.82),
        ),
    ):
        inset = ax.inset_axes(bounds)
        field_inset = _build_safety_potential_field(
            position_points=20_000,
            speed_points=300,
            position_window_m=ZOOM_POSITION_WINDOW_M,
            speed_window_mps=(speed_window[0] / 3.6, speed_window[1] / 3.6),
        )
        _draw_safety_potential(inset, field_inset)
        _plot_safety_boundaries(inset, field_inset)
        inset.set_xlim(*position_window_km)
        inset.set_ylim(*speed_window)
        inset.tick_params(labelsize=PAPER_LEGEND_FONT_SIZE)
        if not minimal:
            inset.text(
                label_pos[0],
                label_pos[1],
                zoom_label,
                transform=inset.transAxes,
                fontsize=PAPER_LEGEND_FONT_SIZE,
                va="baseline",
            )
            ax.indicate_inset_zoom(inset, edgecolor="gray")

    if minimal:
        _apply_minimal_axis_style(ax)
    else:
        fig.subplots_adjust(top=0.96, bottom=0.15, left=0.12, right=0.90)
        ax.set_xlabel("Position (km)")
        ax.set_ylabel("Speed (km/h)")
        apply_sci_grid(ax)
        ax.legend(
            (min_speed_line, max_speed_line),
            (r"$v_{\min}(x)$", r"$v_{\max}(x)$"),
            loc="lower left",
            frameon=False,
        )
        safety_bar = fig.colorbar(
            safety_mesh,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            fraction=0.04,
        )
        safety_bar.set_label(r"$\Phi_{\mathrm{safety}}$")

    _apply_transparent_background(fig)
    return fig


PLOT_TYPE_CHOICES: tuple[str, ...] = (
    "punctuality",
    "safety-punctuality",
    "safety",
    "all",
)
FIGURE_FILENAMES = {
    "punctuality": "punctuality_potential.pdf",
    "safety-punctuality": "safety_punctuality_potential.pdf",
    "safety": "safety_potential.pdf",
}
FIGURE_FILENAMES_MINIMAL = {
    "punctuality": "punctuality_potential_minimal.tiff",
    "safety-punctuality": "safety_punctuality_potential_minimal.tiff",
    "safety": "safety_potential_minimal.tiff",
}


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="展示并可选保存势函数图。")
    _ = parser.add_argument(
        "--plot-type",
        choices=PLOT_TYPE_CHOICES,
        default="safety-punctuality",
        help="选择展示哪种势函数图。",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="固定名称 PDF 的输出目录；不传时仅展示图像。",
    )
    _ = parser.add_argument(
        "--schedule-time-s",
        type=float,
        default=None,
        help="准点势函数使用的计划运行时间（秒）。",
    )
    _ = parser.add_argument(
        "--minimal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="极简图形模式：仅保留核心数据图元，移除文字与辅助标注。",
    )
    _ = parser.add_argument(
        "--no-show",
        action="store_true",
        help="保存图像但不打开交互式展示窗口。",
    )
    return parser


def _validate_cli_args(cli_args: argparse.Namespace) -> None:
    if cli_args.output_dir is not None and str(cli_args.output_dir).strip() == "":
        raise ValueError("--output-dir must not be empty")
    if cli_args.schedule_time_s is not None:
        if not np.isfinite(cli_args.schedule_time_s) or cli_args.schedule_time_s <= 0.0:
            raise ValueError("--schedule-time-s must be finite and positive")


def _resolve_plotter(
    plot_type: str,
    *,
    minimal: bool,
    schedule_time_s: float | None = None,
) -> Callable[[], Figure]:
    plotters: dict[str, Callable[[], Figure]] = {
        "punctuality": lambda: plot_punctuality_potential(
            schedule_time_s=schedule_time_s, minimal=minimal
        ),
        "safety-punctuality": lambda: plot_safety_punctuality_potentials(
            schedule_time_s=schedule_time_s, minimal=minimal
        ),
        "safety": lambda: plot_safety_potential_heatmap_speed(minimal=minimal),
    }
    return plotters[plot_type]


def plot_punctuality_potential(
    *,
    schedule_time_s: float | None = None,
    minimal: bool = False,
) -> Figure:
    """Plot the runtime punctuality potential over the complete route."""
    field = _build_punctuality_potential_field(schedule_time_s=schedule_time_s)
    fig, ax = plt.subplots(figsize=sci_figure_size(columns="text", height_in=2.5))
    mesh, reference_line = _draw_punctuality_potential(ax, field)
    if minimal:
        _apply_minimal_axis_style(ax)
    else:
        fig.subplots_adjust(top=0.96, bottom=0.18, left=0.12, right=0.90)
        ax.set(
            xlabel="Position (km)",
            ylabel=r"Theoretical time margin $\rho$ (s)",
        )
        apply_sci_grid(ax)
        ax.legend(
            (reference_line,),
            (r"$\rho_{\mathrm{ref}}$",),
            loc="upper right",
            frameon=False,
        )
        punctuality_bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04)
        punctuality_bar.set_label(r"$\Phi_{\mathrm{punct}}$")
    _apply_transparent_background(fig)
    return fig


def plot_safety_punctuality_potentials(
    *,
    schedule_time_s: float | None = None,
    minimal: bool = False,
) -> Figure:
    """Safety potential over one stopping area with two zoom insets (a) above the
    punctuality potential over the whole route (b)."""
    safety_field = _build_safety_potential_field()
    punctuality_field = _build_punctuality_potential_field(
        schedule_time_s=schedule_time_s
    )
    fig, (ax_safety, ax_punctuality) = plt.subplots(
        2,
        1,
        figsize=sci_figure_size(columns="text", height_in=5.4),
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )
    safety_mesh = _draw_safety_potential(ax_safety, safety_field)
    min_speed_line, max_speed_line = _plot_safety_boundaries(ax_safety, safety_field)
    punctuality_mesh, reference_line = _draw_punctuality_potential(
        ax_punctuality, punctuality_field
    )
    position_window_km = tuple(value / 1000.0 for value in ZOOM_POSITION_WINDOW_M)
    for bounds, speed_window, zoom_label, label_pos in (
        (
            ZOOM_UPPER_INSET_BOUNDS,
            ZOOM_UPPER_SPEED_WINDOW_KMH,
            r"Zoom: $v_{\max}$",
            (0.06, 0.16),
        ),
        (
            ZOOM_LOWER_INSET_BOUNDS,
            ZOOM_LOWER_SPEED_WINDOW_KMH,
            r"Zoom: $v_{\min}$",
            (0.06, 0.82),
        ),
    ):
        inset = ax_safety.inset_axes(bounds)
        field = _build_safety_potential_field(
            position_points=20_000,
            speed_points=300,
            position_window_m=ZOOM_POSITION_WINDOW_M,
            speed_window_mps=(speed_window[0] / 3.6, speed_window[1] / 3.6),
        )
        _draw_safety_potential(inset, field)
        _plot_safety_boundaries(inset, field)
        inset.set_xlim(*position_window_km)
        inset.set_ylim(*speed_window)
        inset.tick_params(labelsize=PAPER_LEGEND_FONT_SIZE)
        if not minimal:
            inset.text(
                label_pos[0],
                label_pos[1],
                zoom_label,
                transform=inset.transAxes,
                fontsize=PAPER_LEGEND_FONT_SIZE,
                verticalalignment="center",
                bbox=dict(
                    boxstyle="square,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
            )
        ax_safety.indicate_inset_zoom(inset, edgecolor="black", linewidth=0.8)
    if minimal:
        for axis in fig.axes:
            _apply_minimal_axis_style(axis)
    else:
        fig.subplots_adjust(top=0.985, bottom=0.08, left=0.12, right=0.89, hspace=0.30)
        ax_safety.set(xlabel="Position (km)", ylabel="Speed (km/h)")
        ax_punctuality.set(
            xlabel="Position (km)", ylabel=r"Theoretical time margin $\rho$ (s)"
        )
        for axis in fig.axes:
            apply_sci_grid(axis)
        ax_safety.legend(
            (min_speed_line, max_speed_line),
            (r"$v_{\min}(x)$", r"$v_{\max}(x)$"),
            loc="lower left",
            frameon=False,
        )
        ax_punctuality.legend(
            (reference_line,),
            (r"$\rho_{\mathrm{ref}}$",),
            loc="upper right",
            frameon=False,
        )
        safety_bar = fig.colorbar(safety_mesh, ax=ax_safety, pad=0.02, fraction=0.04)
        safety_bar.set_label(r"$\Phi_{\mathrm{safety}}$")
        punctuality_bar = fig.colorbar(
            punctuality_mesh, ax=ax_punctuality, pad=0.02, fraction=0.04
        )
        punctuality_bar.set_label(r"$\Phi_{\mathrm{punct}}$")
        add_panel_label(ax_safety, "(a)")
        add_panel_label(ax_punctuality, "(b)")
    _apply_transparent_background(fig)
    return fig


def _save_compact_figure(
    figure: Figure,
    output_dir: Path,
    *,
    plot_type: str,
    minimal: bool = False,
) -> Path:
    if minimal:
        output_path = output_dir / FIGURE_FILENAMES_MINIMAL[plot_type]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=1200.0,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        return output_path
    return save_sci_figure(
        figure, output_dir / FIGURE_FILENAMES[plot_type], transparent=False
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    cli_args = parser.parse_args(argv)

    try:
        _validate_cli_args(cli_args)
    except ValueError as exc:
        parser.error(str(exc))

    apply_paper_style()
    plot_types = (
        ["safety", "punctuality"]
        if cli_args.plot_type == "all"
        else [cli_args.plot_type]
    )

    for ptype in plot_types:
        figure = _resolve_plotter(
            ptype,
            minimal=cli_args.minimal,
            schedule_time_s=cli_args.schedule_time_s,
        )()

        if cli_args.output_dir is not None:
            output_path = _save_compact_figure(
                figure,
                cli_args.output_dir,
                plot_type=ptype,
                minimal=cli_args.minimal,
            )
            print(f"图像已保存: {output_path}")

        if not cli_args.no_show:
            plt.show()
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
