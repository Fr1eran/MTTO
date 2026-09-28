from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from mtto.domain.safeguard import Safeguard
from mtto.domain.safeguard.geometry import (
    cal_regions,
    concatenate_curves_list,
    pad_2curve_lists,
)
from mtto.domain.scenario import Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.io.scenario import load_scenario
from paper.plotting.style import (
    SCI_GRID_ALPHA,
    VIS_HARD_LIMIT_RED,
    VIS_SAFE_BLUE,
    apply_sci_grid,
)

ROOT = Path(__file__).resolve().parents[2]


def concatenate_curves_with_NaN(
    curves_set: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not curves_set:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    total_len = sum(curve.shape[1] + 1 for curve in curves_set)
    out = np.full((2, total_len), np.nan, dtype=np.float64)

    curr = 0
    for curve in curves_set:
        n = curve.shape[1]
        out[:, curr : curr + n] = curve
        curr += n + 1

    return out[0], out[1]


def draw_regions(
    ax: Axes,
    above_curves_list: list[NDArray[np.float64]],
    below_curves_list: list[NDArray[np.float64]],
    label: str,
    color: str,
    alpha: float,
) -> None:
    if not above_curves_list or not below_curves_list:
        return

    above_curves_x_con, above_curves_y_con = concatenate_curves_list(above_curves_list)
    _below_curves_x_con, below_curves_y_con = concatenate_curves_list(below_curves_list)

    above_curves_y_kmh = above_curves_y_con * 3.6
    below_curves_y_kmh = below_curves_y_con * 3.6

    _ = ax.fill_between(
        above_curves_x_con,
        above_curves_y_kmh,
        below_curves_y_kmh,
        where=(above_curves_y_kmh > below_curves_y_kmh),
        interpolate=False,
        step="pre",
        label=label,
        color=color,
        alpha=alpha,
    )


DANGER_VIEW_LAYERS: tuple[str, ...] = (
    "speed_limit",
    "danger_region",
    "min_curve_part",
    "max_curve_part",
)
FULL_CURVE_VIEW_LAYERS: tuple[str, ...] = (
    "speed_limit",
    "levi_curve_full",
    "brake_curve_full",
    "min_curve_full",
    "max_curve_full",
)
_LAYER_RENDER_ORDER: tuple[str, ...] = (
    "speed_limit",
    "danger_region",
    "min_curve_part",
    "max_curve_part",
    "levi_curve_full",
    "brake_curve_full",
    "min_curve_full",
    "max_curve_full",
    "idp_points",
)
_REGION_RENDER_LAYERS: frozenset[str] = frozenset(
    {
        "danger_region",
        "min_curve_part",
        "max_curve_part",
        "idp_points",
    }
)
_FULL_CURVE_RENDER_LAYERS: frozenset[str] = frozenset(
    {
        "levi_curve_full",
        "brake_curve_full",
        "min_curve_full",
        "max_curve_full",
    }
)
_MUTUALLY_EXCLUSIVE_LAYER_PAIRS: tuple[tuple[str, str], ...] = (
    ("min_curve_part", "min_curve_full"),
    ("max_curve_part", "max_curve_full"),
)
_VALID_RENDER_LAYERS: frozenset[str] = frozenset(_LAYER_RENDER_ORDER)


def _get_speed_scale(speed_unit: str) -> float:
    if speed_unit == "km/h":
        return 3.6
    if speed_unit == "m/s":
        return 1.0
    raise ValueError("speed_unit must be either 'm/s' or 'km/h'")


def _normalize_render_layers(layers: Sequence[str] | None) -> tuple[str, ...]:
    if layers is None:
        return DANGER_VIEW_LAYERS

    normalized_layers = tuple(layers)
    normalized_layer_set = set(normalized_layers)
    unknown_layers = [
        layer for layer in normalized_layers if layer not in _VALID_RENDER_LAYERS
    ]
    if unknown_layers:
        raise ValueError(
            "Unknown render layers: "
            + f"{unknown_layers}. Supported layers: "
            + f"{sorted(_VALID_RENDER_LAYERS)}"
        )

    for layer_a, layer_b in _MUTUALLY_EXCLUSIVE_LAYER_PAIRS:
        if layer_a in normalized_layer_set and layer_b in normalized_layer_set:
            raise ValueError(
                f"Render layers '{layer_a}' and '{layer_b}' are mutually exclusive"
            )

    return normalized_layers


def _plot_curve(
    ax: Axes,
    *,
    pos: NDArray[np.float64],
    speed: NDArray[np.float64],
    speed_scale: float,
    label: str,
    color: str,
    linestyle: str = "solid",
    alpha: float = 1.0,
    linewidth: float = 2.0,
) -> None:
    _ = ax.plot(
        pos,
        speed * speed_scale,
        label=label,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=linewidth,
    )


def render_safeguard(
    safeguard: Safeguard,
    ax: Axes,
    *,
    layers: Sequence[str] | None = None,
    speed_unit: Literal["km/h", "m/s"] = "km/h",
) -> None:
    """按图层绘制防护曲线和危险域。

    Args:
        safeguard: Safeguard 实例。
        ax: Matplotlib 坐标轴。
        layers: 需要绘制的图层序列。
            - None 时使用 `DANGER_VIEW_LAYERS`。
            - 允许混合选择危险域图层和完整曲线图层。
            - 互斥约束: `min_curve_part` 与 `min_curve_full` 不能同时出现;
              `max_curve_part` 与 `max_curve_full` 不能同时出现。
        speed_unit: 速度显示单位, 仅支持 "m/s" 与 "km/h"。
    """
    selected_layers = _normalize_render_layers(layers)
    if not selected_layers:
        return

    speed_scale = _get_speed_scale(speed_unit)
    selected_layer_set = set(selected_layers)

    if any(layer in _REGION_RENDER_LAYERS for layer in selected_layers):
        idp_points, min_parts, max_parts = cal_regions(
            list(safeguard.min_curves),
            list(safeguard.max_curves)[:-1],
        )
        min_padded, max_padded = pad_2curve_lists(min_parts, max_parts)
        min_parts_pos_con, min_parts_speed_con = concatenate_curves_with_NaN(min_padded)
        max_parts_pos_con, max_parts_speed_con = concatenate_curves_with_NaN(max_padded)
        idp_points_x = np.asarray(idp_points[0, :], dtype=np.float64)
        idp_points_y = np.asarray(idp_points[1, :], dtype=np.float64)

    if any(layer in _FULL_CURVE_RENDER_LAYERS for layer in selected_layers):
        levi_pos_con, levi_speed_con = concatenate_curves_with_NaN(
            list(safeguard.levi_curves)
        )
        brake_pos_con, brake_speed_con = concatenate_curves_with_NaN(
            list(safeguard.brake_curves)
        )
        min_pos_con, min_speed_con = concatenate_curves_with_NaN(
            list(safeguard.min_curves)
        )
        max_pos_con, max_speed_con = concatenate_curves_with_NaN(
            list(safeguard.max_curves)
        )

    for layer in _LAYER_RENDER_ORDER:
        if layer not in selected_layer_set:
            continue

        if layer == "speed_limit":
            _ = ax.step(
                safeguard.speed_limit_intervals[:-1],
                safeguard.speed_limits * speed_scale,
                where="post",
                color=VIS_HARD_LIMIT_RED,
                linestyle="dashdot",
                label="Track speed limit",
                linewidth=1.5,
            )
        elif layer == "danger_region":
            draw_regions(
                ax=ax,
                above_curves_list=min_padded,
                below_curves_list=max_padded,
                label="Dangerous speed region",
                color="#FFBABA",
                alpha=1.0,
            )
        elif layer == "min_curve_part":
            _plot_curve(
                ax=ax,
                pos=min_parts_pos_con,
                speed=min_parts_speed_con,
                speed_scale=speed_scale,
                label="Minimum speed curve",
                color=VIS_SAFE_BLUE,
                linewidth=1.2,
            )
        elif layer == "max_curve_part":
            _plot_curve(
                ax=ax,
                pos=max_parts_pos_con,
                speed=max_parts_speed_con,
                speed_scale=speed_scale,
                label="Maximum speed curve",
                color=VIS_HARD_LIMIT_RED,
                linewidth=1.2,
            )
        elif layer == "levi_curve_full":
            _plot_curve(
                ax=ax,
                pos=levi_pos_con,
                speed=levi_speed_con,
                speed_scale=speed_scale,
                label="Safe levitation curve",
                color=VIS_SAFE_BLUE,
                linestyle="dashed",
                linewidth=1.2,
            )
        elif layer == "brake_curve_full":
            _plot_curve(
                ax=ax,
                pos=brake_pos_con,
                speed=brake_speed_con,
                speed_scale=speed_scale,
                label="Safe braking curve",
                color=VIS_HARD_LIMIT_RED,
                linestyle="dashed",
                linewidth=1.2,
            )
        elif layer == "min_curve_full":
            _plot_curve(
                ax=ax,
                pos=min_pos_con,
                speed=min_speed_con,
                speed_scale=speed_scale,
                label="Minimum speed curve",
                color=VIS_SAFE_BLUE,
                linewidth=1.2,
            )
        elif layer == "max_curve_full":
            _plot_curve(
                ax=ax,
                pos=max_pos_con,
                speed=max_speed_con,
                speed_scale=speed_scale,
                label="Maximum speed curve",
                color=VIS_HARD_LIMIT_RED,
                linewidth=1.2,
            )
        elif layer == "idp_points":
            _ = ax.scatter(
                x=idp_points_x,
                y=idp_points_y * speed_scale,
                color="black",
                label="Intersecting dangerous point",
                linewidths=0.5,
            )


def render_trajectory_on_axes(
    *,
    ax: Any,
    pos_arr: Any,
    speed_arr: Any,
    task: Task | None = None,
    no_safeguard: bool = False,
    factor: float,
    curve_color: str = "blue",
    curve_label: str | None = None,
    safeguard: Safeguard | None = None,
    render_endpoints: bool = True,
    speed_scale: float = 3.6,
    alpha: float = 1.0,
    linewidth: float = 1.5,
    xlim: tuple[float, float] | None = (0.0, 30000.0),
    ylim: tuple[float, float] | None = (0.0, 500.0),
    xlabel: str | None = "Position (m)",
    ylabel: str | None = "Speed (km/h)",
    grid: bool = True,
    grid_alpha: float = SCI_GRID_ALPHA,
) -> None:
    """在给定的 matplotlib Axes 上渲染列车速度轨迹曲线、安全防护边界及停站标记。

    Args:
        ax: matplotlib Axes 对象。
        pos_arr: 位置数组 (m)。
        speed_arr: 速度数组 (m/s 或基础速度单位)。
        task: 运行任务 Task（用于起点终点标记）。
        no_safeguard: True 时跳过安全防护边界渲染。
        factor: 安全系数，用于构建 Safeguard（若未显式传入 safeguard）。
        curve_color: 轨迹曲线颜色。
        curve_label: 曲线图例标签。
        safeguard: 预构建的 Safeguard 实例；若为 None 且
            no_safeguard=False，则自动构建。
        render_endpoints: 是否在轨迹起点和终点处绘制散点标记。
        speed_scale: 速度缩放系数，默认 3.6 (m/s -> km/h)。
        alpha: 曲线透明度。
        linewidth: 曲线线宽。
        xlim: x 轴显示范围，None 时不设置。
        ylim: y 轴显示范围，None 时不设置。
        xlabel: x 轴标签，None 时不设置。
        ylabel: y 轴标签，None 时不设置。
        grid: 是否开启网格。
        grid_alpha: 网格透明度。
    """
    if not no_safeguard:
        if safeguard is None:
            scenario = load_scenario(
                ROOT / "paper/specs/scenario.toml", ROOT / "paper/data/line"
            )
            if scenario.safeguard.params.factor == factor:
                resolved_safeguard = scenario.safeguard
            else:
                from dataclasses import replace

                from mtto.domain.safeguard import build_safeguard

                new_params = replace(scenario.safeguard.params, factor=factor)
                resolved_safeguard = build_safeguard(
                    params=new_params,
                    line=scenario.line,
                    levi_curves=scenario.safeguard.levi_curves,
                    brake_curves=scenario.safeguard.brake_curves,
                    min_curves=scenario.safeguard.min_curves,
                    max_curves=scenario.safeguard.max_curves,
                )
        else:
            resolved_safeguard = safeguard

        render_safeguard(resolved_safeguard, ax=ax, layers=DANGER_VIEW_LAYERS)

    pos = np.asarray(pos_arr, dtype=np.float64)
    speed = np.asarray(speed_arr, dtype=np.float64)

    plot_kwargs: dict[str, Any] = {
        "color": curve_color,
        "alpha": alpha,
        "linewidth": linewidth,
    }
    if curve_label is not None:
        plot_kwargs["label"] = curve_label

    ax.plot(pos, speed * speed_scale, **plot_kwargs)

    if render_endpoints and task is not None:
        ax.scatter(
            float(task.start_position_m),
            0.0,
            marker="o",
            color="green",
            s=40,
            alpha=1.0,
            label="start",
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.scatter(
            float(task.target_position_m),
            0.0,
            marker="o",
            color="red",
            s=40,
            alpha=1.0,
            label="end",
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid:
        apply_sci_grid(ax, alpha=grid_alpha)


def _get_rl_trajectory_display_name(*, is_best: bool) -> str:
    return "RL best trajectory" if is_best else "RL final trajectory"


def render_rl_curve_on_axes(
    *,
    ax: Any,
    profile: SpeedProfile,
    task: Task | None = None,
    is_best: bool = False,
    no_safeguard: bool = False,
    factor: float,
    curve_color: str = "blue",
    curve_label: str | None = None,
    safeguard: Safeguard | None = None,
    render_endpoints: bool = True,
) -> None:
    """在给定的 matplotlib Axes 上渲染 RL 速度曲线及安全防护边界。

    Args:
        ax: matplotlib Axes 对象。
        profile: 列车速度曲线 SpeedProfile。
        task: 运行任务 Task（用于起点终点标记）。
        is_best: 是否为最优策略轨迹（影响默认图例）。
        no_safeguard: True 时跳过安全防护边界渲染。
        factor: 安全系数。
        curve_color: 曲线颜色。
        curve_label: 图例标签，None 时自动生成。
        safeguard: 预构建的 Safeguard，None 时按 factor 构建。
        render_endpoints: 是否绘制起点与终点散点标记。
    """
    render_trajectory_on_axes(
        ax=ax,
        pos_arr=profile.position_m,
        speed_arr=profile.speed_mps,
        task=task,
        no_safeguard=no_safeguard,
        factor=factor,
        curve_color=curve_color,
        curve_label=curve_label or _get_rl_trajectory_display_name(is_best=is_best),
        safeguard=safeguard,
        render_endpoints=render_endpoints,
    )


def render_dp_curve_on_axes(
    *,
    ax: Any,
    profile: SpeedProfile,
    task: Task | None = None,
    no_safeguard: bool = False,
    factor: float,
    curve_color: str = "tab:red",
    curve_label: str | None = None,
    safeguard: Safeguard | None = None,
    render_endpoints: bool = True,
) -> None:
    """在给定的 matplotlib Axes 上渲染 DP 速度曲线及安全防护边界。

    Args:
        ax: matplotlib Axes 对象。
        profile: 列车速度曲线 SpeedProfile。
        task: 运行任务 Task（用于起点终点标记）。
        no_safeguard: True 时跳过安全防护边界渲染。
        factor: 安全系数。
        curve_color: 曲线颜色，默认 "tab:red"。
        curve_label: 图例标签，None 时使用 "DP optimized speed curve"。
        safeguard: 预构建的 Safeguard，None 时按 factor 构建。
        render_endpoints: 是否绘制起点与终点散点标记。
    """
    render_trajectory_on_axes(
        ax=ax,
        pos_arr=profile.position_m,
        speed_arr=profile.speed_mps,
        task=task,
        no_safeguard=no_safeguard,
        factor=factor,
        curve_color=curve_color,
        curve_label=curve_label or "DP optimized speed curve",
        safeguard=safeguard,
        render_endpoints=render_endpoints,
    )
