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

from rl.operational_stepper import OperationalStepper
from rl.reward_calculator import (
    PUNCTUALITY_POTENTIAL_SCALE,
    PUNCTUALITY_POTENTIAL_SIGMA_S,
    RewardCalculator,
    RewardConfig,
    punctuality_potential_from_error,
)
from utils.data_loader import load_safeguard_curves, load_speed_limits
from utils.plot_utils import (
    add_panel_label,
    apply_sci_curve_style,
    apply_sci_grid,
    save_sci_figure,
    sci_figure_size,
)
from utils.scenario import build_scenario

DEFAULT_PUNCTUALITY_SCHEDULE_TIME_S = 465.0

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
PUNCTUALITY_POTENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "mtto_punctuality_penalty",
    [
        (0.00, "#005596"),
        (0.35, "#1B8FD1"),
        (0.65, "#68AFD5"),
        (0.85, "#BCE0F0"),
        (0.96, "#F0F8FC"),
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


def _potential_safety_speed(
    speed: NDArray[np.floating] | float,
    min_speed: NDArray[np.floating] | float,
    max_speed: NDArray[np.floating] | float,
) -> NDArray[np.float64] | float:
    scale = 0.5
    steepness = 8.0
    span = np.maximum(max_speed - min_speed, 1.0)

    upper_exponent = steepness * (max_speed - speed) / span
    upper_tail = np.exp(-np.abs(upper_exponent))
    upper_risk = np.where(
        upper_exponent >= 0.0,
        2.0 * upper_tail / (1.0 + upper_tail),
        2.0 / (1.0 + upper_tail),
    )

    lower_exponent = steepness * (speed - min_speed) / span
    lower_tail = np.exp(-np.abs(lower_exponent))
    lower_risk = np.where(
        min_speed > 0.0,
        np.where(
            lower_exponent >= 0.0,
            2.0 * lower_tail / (1.0 + lower_tail),
            2.0 / (1.0 + lower_tail),
        ),
        0.0,
    )

    return -scale * (upper_risk + lower_risk)


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
) -> _SafetyPotentialField:
    """构造第 7 个辅助停车区内安全势函数的状态域。"""
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list", "max_curves_list"
    )
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
    track_speed_limits_mps, speed_limit_intervals_m = load_speed_limits(to_mps=True)
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
    speed_array_mps = np.linspace(
        0.0,
        float(np.max(max_speed_profile_mps)),
        speed_points,
    )
    position_grid, speed_grid_mps = np.meshgrid(pos_array, speed_array_mps)
    min_speed_grid_mps = np.broadcast_to(min_speed_profile_mps, position_grid.shape)
    max_speed_grid_mps = np.broadcast_to(max_speed_profile_mps, position_grid.shape)
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
    """只在速度上下限约束内计算安全势函数。"""
    safety_potential = np.full(field.position_grid.shape, np.nan)
    safety_potential[field.feasible_mask] = _potential_safety_speed(
        field.speed_grid_mps[field.feasible_mask],
        field.min_speed_grid_mps[field.feasible_mask],
        field.max_speed_grid_mps[field.feasible_mask],
    )
    return safety_potential


def _build_punctuality_potential_field(
    *,
    schedule_time_s: float = DEFAULT_PUNCTUALITY_SCHEDULE_TIME_S,
    position_points: int = 600,
    redundant_time_points: int = 400,
) -> _PunctualityPotentialField:
    """Build the full-route field used by the runtime punctuality potential."""
    if not np.isfinite(schedule_time_s) or schedule_time_s <= 0.0:
        raise ValueError("schedule_time_s must be finite and positive")
    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=float(schedule_time_s)
    )
    stepper = OperationalStepper(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        step_distance_m=30.0,
    )
    calculator = RewardCalculator(
        train_service,
        max_episode_steps=stepper.required_episode_steps,
        whole_distance_m=stepper.whole_distance_m,
        max_energy_consumption_kj=stepper.max_energy_consumption_kj,
        gamma=0.998,
        reward_config=RewardConfig(enable_potential_punctuality=True),
        initial_min_operation_time_s=stepper.initial_min_operation_time_s,
    )
    position_m = np.linspace(
        train_service.start_position,
        train_service.target_position,
        position_points,
    )
    reference_slack_s = np.asarray(
        [calculator.reference_punctuality_slack(float(pos)) for pos in position_m],
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
    potential = punctuality_potential_from_error(error_s)
    assert isinstance(potential, np.ndarray)
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
        field.position_grid_m,
        field.redundant_time_grid_s,
        field.potential,
        cmap=PUNCTUALITY_POTENTIAL_CMAP,
        shading="auto",
        vmin=-PUNCTUALITY_POTENTIAL_SCALE,
        vmax=0.0,
        rasterized=True,
    )
    reference_line = ax.plot(
        field.position_m,
        field.reference_slack_s,
        color="black",
        linestyle="--",
        linewidth=1.4,
    )[0]
    ax.set_xlim(field.position_m[0], field.position_m[-1])
    ax.set_ylim(field.redundant_time_s[0], field.redundant_time_s[-1])
    return mesh, reference_line


def _plot_safety_boundaries(
    ax: Axes,
    field: _SafetyPotentialField,
) -> tuple[Line2D, Line2D]:
    """绘制安全势函数使用的速度上下边界。"""
    min_speed_line = ax.plot(
        field.pos_array,
        field.min_speed_profile_mps * 3.6,
        color="tab:blue",
        linewidth=1.2,
    )[0]
    max_speed_line = ax.plot(
        field.pos_array,
        field.max_speed_profile_mps * 3.6,
        color="tab:red",
        linewidth=1.2,
    )[0]
    _ = ax.set_xlim(field.pos_array[0], field.pos_array[-1])
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


def plot_safety_potential_heatmap_speed(*, minimal: bool = False) -> Figure:
    """以第 7 个辅助停车区绘制与联合图一致的安全势函数。"""
    field = _build_safety_potential_field()
    safety_potential = _calculate_safety_potential(field)
    fig, ax = plt.subplots(figsize=sci_figure_size(columns=1, height_in=2.7))
    safety_mesh = ax.pcolormesh(
        field.position_grid,
        field.speed_grid_mps * 3.6,
        safety_potential,
        cmap=SAFETY_POTENTIAL_CMAP,
        shading="auto",
        vmin=-1.0,
        vmax=0.0,
        rasterized=True,
    )
    min_speed_line, max_speed_line = _plot_safety_boundaries(ax, field)

    if minimal:
        _apply_minimal_axis_style(ax)
    else:
        fig.subplots_adjust(top=0.85, bottom=0.13, left=0.13, right=0.88)
        _ = ax.set_xlabel("Position (m)")
        _ = ax.set_ylabel("Speed (km/h)")
        apply_sci_grid(ax)
        _ = fig.legend(
            (min_speed_line, max_speed_line),
            (r"$v_{\min}(x)$", r"$v_{\max}(x)$"),
            loc="upper center",
            ncols=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.925),
        )
        _ = fig.colorbar(
            safety_mesh,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            fraction=0.046,
        )

    _apply_transparent_background(fig)
    return fig


PLOT_TYPE_CHOICES: tuple[str, ...] = (
    "punctuality",
    "safety-punctuality",
    "safety",
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
        default=DEFAULT_PUNCTUALITY_SCHEDULE_TIME_S,
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
    if not np.isfinite(cli_args.schedule_time_s) or cli_args.schedule_time_s <= 0.0:
        raise ValueError("--schedule-time-s must be finite and positive")


def _resolve_plotter(
    plot_type: str,
    *,
    minimal: bool,
    schedule_time_s: float = DEFAULT_PUNCTUALITY_SCHEDULE_TIME_S,
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
    schedule_time_s: float = DEFAULT_PUNCTUALITY_SCHEDULE_TIME_S,
    minimal: bool = False,
) -> Figure:
    """Plot the runtime punctuality potential over the complete route."""
    field = _build_punctuality_potential_field(schedule_time_s=schedule_time_s)
    fig, ax = plt.subplots(figsize=sci_figure_size(columns=1, height_in=2.8))
    mesh, reference_line = _draw_punctuality_potential(ax, field)
    if minimal:
        _apply_minimal_axis_style(ax)
    else:
        fig.subplots_adjust(top=0.85, bottom=0.17, left=0.18, right=0.88)
        ax.set(
            xlabel="Position (m)",
            ylabel="Redundant operation time (s)",
        )
        apply_sci_grid(ax)
        fig.legend(
            (reference_line,),
            (r"$\rho_{\mathrm{ref}}$",),
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.5, 0.94),
        )
        _ = fig.colorbar(mesh, ax=ax, pad=0.03, fraction=0.06)
    _apply_transparent_background(fig)
    return fig


def plot_safety_punctuality_potentials(
    *,
    schedule_time_s: float = DEFAULT_PUNCTUALITY_SCHEDULE_TIME_S,
    minimal: bool = False,
) -> Figure:
    """Plot safety and punctuality potentials as a double-column comparison."""
    safety_field = _build_safety_potential_field()
    safety_potential = _calculate_safety_potential(safety_field)
    punctuality_field = _build_punctuality_potential_field(
        schedule_time_s=schedule_time_s
    )
    fig, (ax_safety, ax_punctuality) = plt.subplots(
        1,
        2,
        figsize=sci_figure_size(columns=2, height_in=2.45),
    )
    safety_mesh = ax_safety.pcolormesh(
        safety_field.position_grid,
        safety_field.speed_grid_mps * 3.6,
        safety_potential,
        cmap=SAFETY_POTENTIAL_CMAP,
        shading="auto",
        vmin=-1.0,
        vmax=0.0,
        rasterized=True,
    )
    min_speed_line, max_speed_line = _plot_safety_boundaries(ax_safety, safety_field)
    punctuality_mesh, reference_line = _draw_punctuality_potential(
        ax_punctuality, punctuality_field
    )
    if minimal:
        _apply_minimal_axis_style(ax_safety)
        _apply_minimal_axis_style(ax_punctuality)
    else:
        fig.subplots_adjust(top=0.87, bottom=0.20, left=0.095, right=0.945, wspace=0.55)
        ax_safety.set(xlabel="Position (m)", ylabel="Speed (km/h)")
        ax_punctuality.set(xlabel="Position (m)", ylabel="Redundant operation time (s)")
        for axis in (ax_safety, ax_punctuality):
            apply_sci_grid(axis)
        fig.legend(
            (min_speed_line, max_speed_line, reference_line),
            (r"$v_{\min}(x)$", r"$v_{\max}(x)$", r"$\rho_{\mathrm{ref}}$"),
            loc="upper center",
            ncols=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
        )
        _ = fig.colorbar(safety_mesh, ax=ax_safety, pad=0.02, fraction=0.046)
        _ = fig.colorbar(punctuality_mesh, ax=ax_punctuality, pad=0.02, fraction=0.046)
    for panel_label, axis in (("(a)", ax_safety), ("(b)", ax_punctuality)):
        add_panel_label(axis, panel_label)
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

    apply_sci_curve_style()
    figure = _resolve_plotter(
        cli_args.plot_type,
        minimal=cli_args.minimal,
        schedule_time_s=cli_args.schedule_time_s,
    )()

    if cli_args.output_dir is not None:
        output_path = _save_compact_figure(
            figure,
            cli_args.output_dir,
            plot_type=cli_args.plot_type,
            minimal=cli_args.minimal,
        )
        print(f"图像已保存: {output_path}")

    if not cli_args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
