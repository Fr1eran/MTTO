import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from utils.data_loader import load_safeguard_curves, load_speed_limits
from utils.plot_utils import set_global_plot_style


@dataclass(frozen=True)
class _SeventhAuxiliaryStopField:
    """第 7 个辅助停车区中两类势函数共用的位置、速度网格与边界。"""

    target_pos: float
    pos_array: np.ndarray
    speed_array_mps: np.ndarray
    position_grid: np.ndarray
    speed_grid_mps: np.ndarray
    min_speed_grid_mps: np.ndarray
    max_speed_grid_mps: np.ndarray
    min_speed_profile_mps: np.ndarray
    max_speed_profile_mps: np.ndarray
    feasible_mask: np.ndarray


def _potential_safety_position(
    pos: NDArray[np.floating],
    min_pos: NDArray[np.floating],
    max_pos: NDArray[np.floating],
    target_pos: float,
) -> NDArray[np.float64]:
    distance_to_target = np.abs(target_pos - pos)
    center_pos = (max_pos + min_pos) / 2.0
    safe_margin = (max_pos - min_pos) / 2.0

    norm_pos_diff = (pos - center_pos) / safe_margin
    spatial_log_arg = 1.1 - norm_pos_diff**2
    in_pos_band = (pos >= min_pos) & (pos <= max_pos)
    valid_mask_spatial = in_pos_band & (spatial_log_arg > 0.0)
    phi_base = np.full_like(norm_pos_diff, np.nan, dtype=np.float64)
    phi_base[valid_mask_spatial] = 2.0 * np.log(spatial_log_arg[valid_mask_spatial])

    scale = 1.0 + 1.0 * np.exp(-0.001 * distance_to_target)

    return scale * phi_base


def _smooth_softplus_risk(
    z: NDArray[np.floating] | float,
    alpha: float,
) -> NDArray[np.float64] | float:
    x = alpha * z
    return np.where(
        x > 20.0,
        z,
        np.log1p(np.exp(np.minimum(x, 20.0))) / alpha,
    )


def _potential_safety_speed(
    pos: NDArray[np.floating],
    speed: NDArray[np.floating],
    min_speed: NDArray[np.floating],
    max_speed: NDArray[np.floating],
    target_pos: float,
) -> NDArray[np.float64]:
    del pos, target_pos

    K_Safety = 1.0
    speed_band = max_speed - min_speed
    safety_buffer = np.clip(0.15 * speed_band, 1.0, 5.0)

    alpha = 3.0

    margin_max = max_speed - speed
    z_max = 1.0 - margin_max / safety_buffer
    smooth_z_max = _smooth_softplus_risk(z_max, alpha)
    phi_upper = -(smooth_z_max**2)

    margin_min = speed - min_speed
    z_min = 1.0 - margin_min / safety_buffer
    smooth_z_min = _smooth_softplus_risk(z_min, alpha)
    phi_lower = np.where(
        min_speed > 0.0,
        -(smooth_z_min**2),
        0.0,
    )

    return K_Safety * (phi_upper + phi_lower)


def _potential_stopping(
    pos: NDArray[np.floating] | float,
    speed: NDArray[np.floating] | float,
    target_pos: float,
    max_speed_mps: NDArray[np.floating] | float,
    *,
    distance_scale_m: float = 1500.0,
    potential_scale: float = 1.0,
    max_exp: float = 2.0,
) -> NDArray[np.float64]:
    """Stopping-potential reference used only for visualization and analysis."""
    if distance_scale_m <= 0.0:
        raise ValueError("distance_scale_m must be positive")
    if potential_scale <= 0.0:
        raise ValueError("potential_scale must be positive")
    if max_exp <= 0.0:
        raise ValueError("max_exp must be positive")

    distance_array = np.abs(np.asarray(pos, dtype=np.float64) - target_pos)
    speed_array = np.asarray(speed, dtype=np.float64)
    local_max_speed = np.maximum(np.asarray(max_speed_mps, dtype=np.float64), 0.0)
    speed_scale = 0.5 * local_max_speed + 1.0
    normalized_error = np.minimum(
        distance_array / distance_scale_m + speed_array / speed_scale, max_exp
    )
    clipped_exponential = np.exp(-normalized_error)

    return (-potential_scale * (1.0 - clipped_exponential)).astype(np.float64)


def infer_position_from_speed(
    curve_pos: NDArray[np.floating],
    curve_speed: NDArray[np.floating],
    target_speed: NDArray[np.floating] | float,
) -> NDArray[np.floating] | np.floating:
    """在速度曲线单调递减时，根据目标速度反推对应位置。"""
    return np.interp(
        target_speed,
        curve_speed[::-1],
        curve_pos[::-1],
        left=float(curve_pos[-1]),
        right=float(curve_pos[0]),
    )


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


def _build_seventh_auxiliary_stop_field(
    *,
    position_points: int = 1200,
    speed_points: int = 800,
) -> _SeventhAuxiliaryStopField:
    """构造第 7 个辅助停车区内安全势函数与停站势函数共用的状态域。"""
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list", "max_curves_list"
    )
    target_pos = 17828.0
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
    return _SeventhAuxiliaryStopField(
        target_pos=target_pos,
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


def _calculate_guidance_potentials(
    field: _SeventhAuxiliaryStopField,
) -> tuple[np.ndarray, np.ndarray]:
    """只在速度上下限约束内计算安全势函数与停站势函数。"""
    safety_potential = np.full(field.position_grid.shape, np.nan)
    safety_potential[field.feasible_mask] = _potential_safety_speed(
        field.position_grid[field.feasible_mask],
        field.speed_grid_mps[field.feasible_mask],
        field.min_speed_grid_mps[field.feasible_mask],
        field.max_speed_grid_mps[field.feasible_mask],
        field.target_pos,
    )
    stopping_potential = np.full(field.position_grid.shape, np.nan)
    stopping_potential[field.feasible_mask] = _potential_stopping(
        field.position_grid[field.feasible_mask],
        field.speed_grid_mps[field.feasible_mask],
        field.target_pos,
        field.max_speed_grid_mps[field.feasible_mask],
    )
    return safety_potential, stopping_potential


def _plot_guidance_boundaries(
    ax: Axes, field: _SeventhAuxiliaryStopField
) -> tuple[Line2D, Line2D, Line2D]:
    """绘制与联合图一致的速度边界与目标位置。"""
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
    target_position_line = ax.axvline(
        field.target_pos,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    _ = ax.set_xlim(field.pos_array[0], field.pos_array[-1])
    _ = ax.set_ylim(0.0, field.speed_array_mps[-1] * 3.6)
    return min_speed_line, max_speed_line, target_position_line


def plot_guidance_potentials_wide(*, minimal: bool = False) -> Figure:
    """在第 7 个辅助停车区并排展示安全势函数与停站势函数。"""
    field = _build_seventh_auxiliary_stop_field()
    safety_potential, stopping_potential = _calculate_guidance_potentials(field)

    if minimal:
        fig = plt.figure(figsize=(12.8, 5.8))
        grid = fig.add_gridspec(1, 2, wspace=0.12)
        ax_safety = fig.add_subplot(grid[0, 0])
        ax_stopping = fig.add_subplot(grid[0, 1], sharex=ax_safety, sharey=ax_safety)
    else:
        fig = plt.figure(figsize=(12.8, 5.9))
        grid = fig.add_gridspec(1, 2, wspace=0.12)
        ax_safety = fig.add_subplot(grid[0, 0])
        ax_stopping = fig.add_subplot(grid[0, 1], sharex=ax_safety, sharey=ax_safety)
        fig.subplots_adjust(top=0.85, bottom=0.17, left=0.07, right=0.98)

    safety_mesh = ax_safety.pcolormesh(
        field.position_grid,
        field.speed_grid_mps * 3.6,
        safety_potential,
        cmap=plt.get_cmap("Spectral"),
        shading="auto",
        vmin=-1.0,
        vmax=0.0,
    )
    stopping_mesh = ax_stopping.pcolormesh(
        field.position_grid,
        field.speed_grid_mps * 3.6,
        stopping_potential,
        cmap=plt.get_cmap("viridis"),
        shading="auto",
        vmin=-(1.0 - np.exp(-2.0)),
        vmax=0.0,
    )

    min_speed_line, max_speed_line, target_position_line = _plot_guidance_boundaries(
        ax_safety, field
    )
    _ = _plot_guidance_boundaries(ax_stopping, field)

    if minimal:
        _apply_minimal_axis_style(ax_safety)
        _apply_minimal_axis_style(ax_stopping)
    else:
        _ = ax_safety.set_xlabel("Position (m)")
        _ = ax_stopping.set_xlabel("Position (m)")
        _ = ax_safety.set_ylabel("Velocity (km/h)")
        ax_stopping.tick_params(axis="y", which="both", left=False, labelleft=False)
        for ax in (ax_safety, ax_stopping):
            ax.grid(True, alpha=0.3, linestyle=":")
        _ = fig.legend(
            (min_speed_line, max_speed_line, target_position_line),
            (r"$v_{\min}(x)$", r"$v_{\max}(x)$", "Target position"),
            loc="upper center",
            ncols=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.925),
        )
        _ = fig.colorbar(
            safety_mesh,
            ax=ax_safety,
            orientation="vertical",
            pad=0.02,
            fraction=0.046,
        )
        _ = fig.colorbar(
            stopping_mesh,
            ax=ax_stopping,
            orientation="vertical",
            pad=0.02,
            fraction=0.046,
        )
        for panel_label, ax in (("(a)", ax_safety), ("(b)", ax_stopping)):
            bounds = ax.get_position()
            _ = fig.text(
                (bounds.x0 + bounds.x1) / 2.0,
                bounds.y0 - 0.11,
                panel_label,
                ha="center",
                va="top",
            )

    _apply_transparent_background(fig)
    return fig


def _apply_minimal_axis_style(ax: Axes) -> None:
    ax.grid(False)
    ax.set_axis_on()
    ax.axison = True


def _apply_transparent_background(fig: Figure) -> None:
    """让图窗与所有坐标轴背景透明，便于嵌入论文架构图。"""
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)

        # 3D 坐标轴 pane 默认非透明，需单独处理。
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis_obj = getattr(ax, axis_name, None)
            pane = getattr(axis_obj, "pane", None)
            if pane is not None:
                pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


def plot_safety_potential_heatmap_speed(*, minimal: bool = False) -> Figure:
    """以第 7 个辅助停车区绘制与联合图一致的安全势函数。"""
    field = _build_seventh_auxiliary_stop_field()
    safety_potential, _ = _calculate_guidance_potentials(field)
    fig, ax = plt.subplots(figsize=(6.4, 5.9))
    safety_mesh = ax.pcolormesh(
        field.position_grid,
        field.speed_grid_mps * 3.6,
        safety_potential,
        cmap=plt.get_cmap("Spectral"),
        shading="auto",
        vmin=-1.0,
        vmax=0.0,
    )
    min_speed_line, max_speed_line, target_position_line = _plot_guidance_boundaries(
        ax, field
    )

    if minimal:
        _apply_minimal_axis_style(ax)
    else:
        fig.subplots_adjust(top=0.85, bottom=0.13, left=0.13, right=0.88)
        _ = ax.set_xlabel("Position (m)")
        _ = ax.set_ylabel("Velocity (km/h)")
        ax.grid(True, alpha=0.3, linestyle=":")
        _ = fig.legend(
            (min_speed_line, max_speed_line, target_position_line),
            (r"$v_{\min}(x)$", r"$v_{\max}(x)$", "Target position"),
            loc="upper center",
            ncols=3,
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


def plot_safety_potential_heatmap_position(*, minimal: bool = False) -> Figure:
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list", "max_curves_list"
    )

    # 仍然以第7个辅助停车区为示例。
    target_pos = 17828.0

    upper_speed = 150.0 / 3.6
    lower_speed = 0.0

    min_curve_pos = min_curves_list[6][0, :]
    min_curve_speed = min_curves_list[6][1, :]
    max_curve_pos = max_curves_list[7][0, :]
    max_curve_speed = max_curves_list[7][1, :]

    # 这里将速度视为自变量，按速度从 0 到 200 km/h 反算位置边界。
    speed_array_ms = np.linspace(lower_speed, upper_speed, 2000)
    pos_from_min_curve = infer_position_from_speed(
        min_curve_pos, min_curve_speed, speed_array_ms
    )
    pos_from_max_curve = infer_position_from_speed(
        max_curve_pos, max_curve_speed, speed_array_ms
    )

    safe_center_pos_array = (pos_from_min_curve + pos_from_max_curve) / 2.0

    pos_lower_1d = np.minimum(pos_from_min_curve, pos_from_max_curve)
    pos_upper_1d = np.maximum(pos_from_min_curve, pos_from_max_curve)

    pos_left_bound = float(np.min(pos_lower_1d))
    pos_right_bound = float(np.max(pos_upper_1d))

    pos_array = np.linspace(pos_left_bound, pos_right_bound, 2000)

    POS, SPEED = np.meshgrid(pos_array, speed_array_ms)
    POS_LOWER = np.tile(pos_lower_1d[:, None], (1, pos_array.size))
    POS_UPPER = np.tile(pos_upper_1d[:, None], (1, pos_array.size))

    POTENTIAL = _potential_safety_position(POS, POS_LOWER, POS_UPPER, target_pos)
    in_pos_band = (POS >= POS_LOWER) & (POS <= POS_UPPER)
    POTENTIAL_MASKED = np.where(in_pos_band, POTENTIAL, np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.get_cmap("Spectral")

    c = ax.pcolormesh(
        POS,
        SPEED * 3.6,
        POTENTIAL_MASKED,
        cmap=cmap,
        shading="auto",
        vmin=-8.0,
        vmax=0.0,
    )

    _ = ax.set_xlim(pos_left_bound - 1000, pos_right_bound + 1000)
    _ = ax.set_ylim(lower_speed * 3.6, upper_speed * 3.6)

    if minimal:
        _ = ax.plot(
            pos_from_max_curve,
            speed_array_ms * 3.6,
            color="red",
            linewidth=1,
        )
        _ = ax.plot(
            pos_from_min_curve,
            speed_array_ms * 3.6,
            color="blue",
            linewidth=1,
        )
        _apply_minimal_axis_style(ax)
    else:
        _ = ax.plot(
            pos_from_max_curve,
            speed_array_ms * 3.6,
            color="red",
            linewidth=1,
            label="maximum speed curve",
        )
        _ = ax.plot(
            pos_from_min_curve,
            speed_array_ms * 3.6,
            color="blue",
            linewidth=1,
            label="minimum speed curve",
        )
        _ = ax.plot(
            safe_center_pos_array,
            speed_array_ms * 3.6,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="safe center",
        )

        _ = fig.colorbar(c, ax=ax, extend="min")

        _ = ax.set_xlabel("Position (m)")
        _ = ax.set_ylabel("Speed (km/h)")
        _ = ax.legend(loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

    _apply_transparent_background(fig)
    plt.tight_layout()
    return fig


def plot_stopping_potential_heatmap(
    view_mode: str = "2d",
    *,
    minimal: bool = False,
) -> Figure:
    """以第 7 个辅助停车区绘制与联合图一致的停站势函数。"""
    mode = str(view_mode).lower().strip()
    if mode not in {"2d", "3d"}:
        raise ValueError("view_mode 仅支持 '2d' 或 '3d'")

    field = _build_seventh_auxiliary_stop_field()
    _, stopping_potential = _calculate_guidance_potentials(field)

    if mode == "3d":
        fig = plt.figure(figsize=(8.0, 5.9))
        ax = fig.add_subplot(111, projection="3d")
        surface_step = 4
        _ = ax.plot_surface(
            field.position_grid[::surface_step, ::surface_step],
            (field.speed_grid_mps * 3.6)[::surface_step, ::surface_step],
            stopping_potential[::surface_step, ::surface_step],
            cmap=plt.get_cmap("viridis"),
            linewidth=0,
            antialiased=False,
            vmin=-(1.0 - np.exp(-2.0)),
            vmax=0.0,
        )
        _ = ax.plot(
            field.pos_array,
            field.min_speed_profile_mps * 3.6,
            np.zeros_like(field.pos_array),
            color="tab:blue",
            linewidth=1.2,
        )
        _ = ax.plot(
            field.pos_array,
            field.max_speed_profile_mps * 3.6,
            np.zeros_like(field.pos_array),
            color="tab:red",
            linewidth=1.2,
        )
        ax.set_xlim(field.pos_array[0], field.pos_array[-1])
        ax.set_ylim(0.0, field.speed_array_mps[-1] * 3.6)
        ax.set_zlim(-(1.0 - np.exp(-2.0)), 0.0)
        ax.view_init(elev=28, azim=-130)
        if minimal:
            _apply_minimal_axis_style(ax)
        else:
            _ = ax.set_xlabel("Position (m)")
            _ = ax.set_ylabel("Velocity (km/h)")
            _ = ax.set_zlabel("Stopping potential")
        _apply_transparent_background(fig)
        return fig

    fig, ax = plt.subplots(figsize=(6.4, 5.9))
    stopping_mesh = ax.pcolormesh(
        field.position_grid,
        field.speed_grid_mps * 3.6,
        stopping_potential,
        cmap=plt.get_cmap("viridis"),
        shading="auto",
        vmin=-(1.0 - np.exp(-2.0)),
        vmax=0.0,
    )
    min_speed_line, max_speed_line, target_position_line = _plot_guidance_boundaries(
        ax, field
    )

    if minimal:
        _apply_minimal_axis_style(ax)
    else:
        fig.subplots_adjust(top=0.85, bottom=0.13, left=0.13, right=0.88)
        _ = ax.set_xlabel("Position (m)")
        _ = ax.set_ylabel("Velocity (km/h)")
        ax.grid(True, alpha=0.3, linestyle=":")
        _ = fig.legend(
            (min_speed_line, max_speed_line, target_position_line),
            (r"$v_{\min}(x)$", r"$v_{\max}(x)$", "Target position"),
            loc="upper center",
            ncols=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.925),
        )
        _ = fig.colorbar(
            stopping_mesh,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            fraction=0.046,
        )

    _apply_transparent_background(fig)
    return fig


def plot_stopping_potential_slices(*, minimal: bool = False) -> Figure:
    """
    绘制停站势函数在距离维与速度维上的一维切片，便于调参。
    """

    target_pos = 29270.046
    _, max_curves_list = load_safeguard_curves("min_curves_list", "max_curves_list")
    final_stop_max_curve = max_curves_list[9]
    max_speed_mps = float(
        interp_with_constant_fill(
            final_stop_max_curve[0, :],
            final_stop_max_curve[1, :],
            target_pos,
            left_value=float(final_stop_max_curve[1, 0]),
            right_value=float(final_stop_max_curve[1, -1]),
        )
    )

    distance_scale_m = 1500.0
    speed_scale_mps = 0.5 * max_speed_mps + 1.0
    max_exp = 2.0

    distance_error_array = np.linspace(-7500.0, 7500.0, 1800)
    pos_array = target_pos + distance_error_array
    speed_array_ms = np.linspace(0.0, max(max_speed_mps, 2.5 * speed_scale_mps), 1200)

    potential_vs_distance = _potential_stopping(
        pos=pos_array,
        speed=0.0,
        target_pos=target_pos,
        max_speed_mps=max_speed_mps,
    )
    potential_vs_speed = _potential_stopping(
        pos=target_pos,
        speed=speed_array_ms,
        target_pos=target_pos,
        max_speed_mps=max_speed_mps,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(distance_error_array, potential_vs_distance, color="tab:orange")

    axes[1].plot(speed_array_ms * 3.6, potential_vs_speed, color="tab:red")

    if minimal:
        _apply_minimal_axis_style(axes[0])
        _apply_minimal_axis_style(axes[1])
    else:
        axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        position_plateau_m = max_exp * distance_scale_m
        axes[0].axvline(position_plateau_m, color="gray", linestyle=":", linewidth=1.2)
        axes[0].axvline(-position_plateau_m, color="gray", linestyle=":", linewidth=1.2)
        axes[0].set_xlabel("stopping error (m)")
        axes[0].set_ylabel(r"$\Phi_D$")
        axes[0].set_title("v = 0")
        axes[0].grid(True, alpha=0.3, linestyle=":")

        axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        speed_plateau_mps = max_exp * speed_scale_mps
        axes[1].axvline(
            speed_plateau_mps * 3.6,
            color="gray",
            linestyle=":",
            linewidth=1.2,
        )
        axes[1].set_xlabel("speed (km/h)")
        axes[1].set_ylabel(r"$\Phi_D$")
        axes[1].set_title(rf"d = 0, $v_{{\max}}={max_speed_mps * 3.6:.1f}$ km/h")
        axes[1].grid(True, alpha=0.3, linestyle=":")

        _ = fig.suptitle(r"Clipped Laplace stopping-potential slices", fontsize=13)
    _apply_transparent_background(fig)
    plt.tight_layout()
    return fig


PLOT_TYPE_CHOICES: tuple[str, ...] = (
    "safety-speed",
    "safety-position",
    "stopping-heatmap",
    "stopping-slices",
    "guidance-wide",
)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="展示并可选保存势函数图。")
    _ = parser.add_argument(
        "--plot-type",
        choices=PLOT_TYPE_CHOICES,
        default="stopping-heatmap",
        help="选择展示哪种势函数图。",
    )
    _ = parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="输出紧凑版图像路径（如 output/potential.png）。不传时仅展示图像。",
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
    if cli_args.output_file is not None and str(cli_args.output_file).strip() == "":
        raise ValueError("--output-file must not be empty")


def _resolve_plotter(plot_type: str, *, minimal: bool) -> Callable[[], Figure]:
    plotters: dict[str, Callable[[], Figure]] = {
        "safety-speed": lambda: plot_safety_potential_heatmap_speed(minimal=minimal),
        "safety-position": lambda: plot_safety_potential_heatmap_position(
            minimal=minimal
        ),
        "stopping-heatmap": lambda: plot_stopping_potential_heatmap(
            view_mode="2d",
            minimal=minimal,
        ),
        "stopping-slices": lambda: plot_stopping_potential_slices(minimal=minimal),
        "guidance-wide": lambda: plot_guidance_potentials_wide(minimal=minimal),
    }
    return plotters[plot_type]


def _apply_plot_style() -> None:
    _ = set_global_plot_style(
        font_preset="sci",
        preferred_font="Times New Roman",
        title_font_size=12.0,
        axis_label_font_size=12.0,
        tick_font_size=12.0,
        legend_font_size=12.0,
        figure_dpi=100.0,
        savefig_dpi=300.0,
    )


def _save_compact_figure(figure: Figure, output_file: Path) -> Path:
    if output_file.suffix == "":
        output_file = output_file.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_file,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    return output_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    cli_args = parser.parse_args(argv)

    try:
        _validate_cli_args(cli_args)
    except ValueError as exc:
        parser.error(str(exc))

    _apply_plot_style()
    figure = _resolve_plotter(cli_args.plot_type, minimal=cli_args.minimal)()

    if cli_args.output_file is not None:
        output_path = _save_compact_figure(figure, cli_args.output_file)
        print(f"图像已保存: {output_path}")

    if not cli_args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
