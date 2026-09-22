from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import fontManager

from utils.type_utils import as_float

MM_PER_INCH = 25.4
SCI_SINGLE_COLUMN_WIDTH_IN = 85.0 / MM_PER_INCH
SCI_DOUBLE_COLUMN_WIDTH_IN = 170.0 / MM_PER_INCH
SCI_EXPORT_DPI = 1200.0
SCI_EXPORT_SUFFIX = ".pdf"
PROJECT_PRIMARY_FONT = "Arial"
SCI_LINE_WIDTH = 1.6
SCI_GRID_COLOR = "#D9D9D9"
SCI_GRID_LINESTYLE = "--"
SCI_GRID_LINE_WIDTH = 0.6
SCI_GRID_ALPHA = 1.0
SCI_BAND_ALPHA = 0.1


def sci_tint_color(
    color: str, factor: float = 0.2
) -> tuple[float, float, float, float]:
    """Convert an RGB color to an RGBA tuple with alpha transparency.

    Provides transparent fill bands for uncertainty shading so overlapping
    regions and underlying grid lines remain clearly distinguishable.
    """
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(color)
    return (*rgb, factor)


SCI_SERIES_LINE_STYLES: tuple[dict[str, str], ...] = (
    {"linestyle": "-", "marker": "o"},
    {"linestyle": "--", "marker": "s"},
    {"linestyle": ":", "marker": "^"},
    {"linestyle": "-.", "marker": "D"},
)

# Paper-wide visual identity: colors are assigned by meaning, never series order.
VIS_HARD_LIMIT_RED = "#FF1800"
VIS_DANGER_CORAL = "#FF7667"
VIS_SAFE_BLUE = "#0072B2"
VIS_PROPOSED_ORANGE = "#ED7D31"
VIS_DP_BLACK = "#181818"
VIS_ACTUAL_PURPLE = "#7B61A8"
VIS_PPO_GRAY = "#7F8C8D"
VIS_DSPL_MAGENTA = "#CC79A7"
VIS_ASA_MINT = "#BBEAD1"
VIS_STATION_LAVENDER = "#AC8EC6"
VIS_ACCEL_CREAM = "#FEE598"

CHINESE_FONT_CANDIDATES: tuple[str, ...] = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Noto Sans CJK JP",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "STHeiti",
)

SCI_ENGLISH_FONT_CANDIDATES: tuple[str, ...] = (
    "Arial",
    "Helvetica",
    "Arial Nova",
    "Nimbus Sans",
    "Liberation Sans",
    "DejaVu Sans",
)

COMMERCIAL_FONT_FALLBACKS: dict[str, tuple[str, ...]] = {
    "Arial": ("Liberation Sans", "Nimbus Sans", "DejaVu Sans"),
    "Helvetica": ("Nimbus Sans", "Liberation Sans", "DejaVu Sans"),
}

# Backward-compatible alias: kept for callers importing this symbol directly.
DEFAULT_FONT_CANDIDATES: tuple[str, ...] = CHINESE_FONT_CANDIDATES


def sci_column_width_in(columns: Literal[1, 2]) -> float:
    """Return the standard 85 mm / 170 mm SCI column width in inches."""
    if columns == 1:
        return SCI_SINGLE_COLUMN_WIDTH_IN
    if columns == 2:
        return SCI_DOUBLE_COLUMN_WIDTH_IN
    raise ValueError(f"columns must be 1 or 2, got {columns!r}")


def sci_figure_size(
    *,
    columns: Literal[1, 2],
    height_in: float,
) -> tuple[float, float]:
    """Build a fixed physical figure size for manuscript-ready graphics."""
    if height_in <= 0.0:
        raise ValueError(f"height_in must be positive, got {height_in!r}")
    return (sci_column_width_in(columns), float(height_in))


def apply_sci_figure_layout(
    fig: plt.Figure,
    *,
    columns: Literal[1, 2],
    height_in: float,
    left: float = 0.12,
    right: float = 0.98,
    bottom: float = 0.14,
    top: float = 0.96,
    wspace: float | None = None,
    hspace: float | None = None,
) -> None:
    """Set fixed manuscript dimensions and explicit compact subplot margins."""
    fig.set_size_inches(*sci_figure_size(columns=columns, height_in=height_in))
    kwargs: dict[str, float] = {
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
    }
    if wspace is not None:
        kwargs["wspace"] = wspace
    if hspace is not None:
        kwargs["hspace"] = hspace
    fig.subplots_adjust(**kwargs)


def save_sci_figure(
    fig: plt.Figure,
    output_file: str | Path,
    *,
    transparent: bool = False,
) -> Path:
    """Save a PDF at the figure's exact physical manuscript dimensions."""
    path = Path(output_file).with_suffix(SCI_EXPORT_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, float | str | bool] = {
        "dpi": SCI_EXPORT_DPI,
    }
    if transparent:
        save_kwargs.update(
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
    fig.savefig(path, **save_kwargs)
    return path


def _pick_first_available_font(font_candidates: Sequence[str]) -> str | None:
    available = {font.name for font in fontManager.ttflist}
    for name in font_candidates:
        if name in available:
            return name
    return None


def _pick_selected_or_first_available_font(
    font_candidates: Sequence[str],
    preferred_font: str | None,
    allow_fallback: bool = True,
) -> str | None:
    """先选择指定字体，再依次选用可选的开源备用字体，最后才是备选字体。"""

    if preferred_font is not None:
        selected = _pick_first_available_font((preferred_font,))
        if selected is not None:
            return selected

        fallback_candidates = (
            COMMERCIAL_FONT_FALLBACKS.get(preferred_font, ()) if allow_fallback else ()
        )
        selected = _pick_first_available_font(fallback_candidates)
        if selected is not None:
            return selected

        if preferred_font not in font_candidates:
            if fallback_candidates:
                tried_fonts = (preferred_font,) + fallback_candidates
                raise ValueError(
                    f"preferred_font={preferred_font!r} "
                    + f"不在候选字体中: {tuple(font_candidates)!r}, "
                    + f"且已尝试替代字体仍不可用: {tried_fonts!r}"
                )
            raise ValueError(
                f"preferred_font={preferred_font!r} "
                + f"不在候选字体中: {tuple(font_candidates)!r}"
            )

        if fallback_candidates:
            tried_fonts = (preferred_font,) + fallback_candidates
            raise ValueError(
                f"preferred_font={preferred_font!r} 在当前系统不可用，"
                + f"且已尝试替代字体仍不可用: {tried_fonts!r}"
            )

        raise ValueError(
            f"preferred_font={preferred_font!r} 在当前系统不可用，请先安装该字体。"
        )

    return _pick_first_available_font(font_candidates)


def _resolve_font_candidates(
    font_preset: Literal["auto", "zh", "sci"],
    custom_font_candidates: Sequence[str] | None,
) -> tuple[str, ...]:
    if custom_font_candidates:
        return tuple(custom_font_candidates)

    if font_preset == "zh":
        return CHINESE_FONT_CANDIDATES

    if font_preset == "sci":
        return SCI_ENGLISH_FONT_CANDIDATES

    # auto: prioritize SCI English fonts, but still keep Chinese fallback.
    return SCI_ENGLISH_FONT_CANDIDATES + CHINESE_FONT_CANDIDATES


def _apply_font_family(
    font_candidates: Sequence[str], preferred_font: str | None
) -> tuple[str | None, tuple[str, ...]]:
    selected_font = _pick_selected_or_first_available_font(
        font_candidates,
        preferred_font,
    )
    selected_cjk_font = _pick_first_available_font(CHINESE_FONT_CANDIDATES)
    font_family = tuple(
        dict.fromkeys(
            font for font in (selected_font, selected_cjk_font) if font is not None
        )
    )
    if font_family:
        plt.rcParams["font.family"] = list(font_family)
    if selected_font is not None:
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = selected_font
        plt.rcParams["mathtext.it"] = f"{selected_font}:italic"
        plt.rcParams["mathtext.bf"] = f"{selected_font}:bold"
        plt.rcParams["mathtext.sf"] = selected_font
        plt.rcParams["mathtext.fallback"] = "stixsans"
    return selected_font, font_family


def set_global_plot_style(
    *,
    base_font_size: float = 12.0,
    title_font_size: float | None = None,
    axis_label_font_size: float | None = None,
    tick_font_size: float | None = None,
    legend_font_size: float | None = None,
    figure_dpi: float = 150.0,
    # line_width: float = 1.5,
    # grid_alpha: float = 0.3,
    # grid_line_style: str = ":",
    unicode_minus: bool = False,
    font_preset: Literal["auto", "zh", "sci"] = "auto",
    preferred_font: str | None = PROJECT_PRIMARY_FONT,
    font_candidates: Sequence[str] | None = None,
) -> dict[str, float | str | tuple[str, ...] | None]:
    """Apply a consistent Matplotlib style for the whole project.

    This function is intended to be called once at script startup so all
    subsequent figures share the same font family, font sizes and DPI.

    Args:
        font_preset: 预设候选字体集合。"sci" 为英文字体优先，"zh" 为中文字体优先。
        preferred_font: 用户指定字体名。若该字体不可用，会尝试开源兼容替代。
        font_candidates: 自定义候选字体。若传入则覆盖 font_preset 对应集合。
    """

    chosen_candidates = _resolve_font_candidates(font_preset, font_candidates)
    selected_font, font_family = _apply_font_family(
        chosen_candidates,
        preferred_font,
    )

    effective_title_size = (
        title_font_size if title_font_size is not None else base_font_size + 2.0
    )
    effective_axis_label_size = (
        axis_label_font_size if axis_label_font_size is not None else base_font_size
    )
    effective_tick_size = (
        tick_font_size if tick_font_size is not None else max(base_font_size - 1.0, 1.0)
    )
    effective_legend_size = (
        legend_font_size
        if legend_font_size is not None
        else max(base_font_size - 1.0, 1.0)
    )

    plt.rcParams["axes.unicode_minus"] = unicode_minus

    plt.rcParams["figure.dpi"] = figure_dpi
    plt.rcParams["savefig.dpi"] = SCI_EXPORT_DPI
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.rcParams["font.size"] = base_font_size
    plt.rcParams["axes.titlesize"] = effective_title_size
    plt.rcParams["figure.titlesize"] = effective_title_size
    plt.rcParams["axes.labelsize"] = effective_axis_label_size
    plt.rcParams["xtick.labelsize"] = effective_tick_size
    plt.rcParams["ytick.labelsize"] = effective_tick_size
    plt.rcParams["legend.fontsize"] = effective_legend_size

    plt.rcParams["lines.linewidth"] = SCI_LINE_WIDTH
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["grid.color"] = SCI_GRID_COLOR
    plt.rcParams["grid.linestyle"] = SCI_GRID_LINESTYLE
    plt.rcParams["grid.linewidth"] = SCI_GRID_LINE_WIDTH
    plt.rcParams["grid.alpha"] = SCI_GRID_ALPHA

    return {
        "font": selected_font,
        "font_family": font_family,
        "base_font_size": base_font_size,
        "title_font_size": effective_title_size,
        "axis_label_font_size": effective_axis_label_size,
        "tick_font_size": effective_tick_size,
        "legend_font_size": effective_legend_size,
        "figure_dpi": figure_dpi,
        "savefig_dpi": SCI_EXPORT_DPI,
        # "line_width": line_width,
        # "grid_alpha": grid_alpha,
        # "grid_line_style": grid_line_style,
        "unicode_minus": unicode_minus,
        "font_preset": font_preset,
        "preferred_font": preferred_font,
    }


def set_chinese_font() -> None:
    """Apply the project Arial-first stack with a sans-serif CJK fallback."""
    _apply_font_family(SCI_ENGLISH_FONT_CANDIDATES, PROJECT_PRIMARY_FONT)


def apply_sci_curve_style(
    *,
    title_font_size: float = 8.0,
    axis_label_font_size: float = 8.0,
    tick_font_size: float = 8.0,
    legend_font_size: float = 8.0,
    figure_dpi: float = 150.0,
    preferred_font: str = PROJECT_PRIMARY_FONT,
    font_preset: Literal["auto", "zh", "sci"] = "sci",
) -> dict[str, float | str | tuple[str, ...] | None]:
    """设置科学论文/技术报告轨迹与评估曲线绘图的全局 matplotlib 样式。"""
    return set_global_plot_style(
        font_preset=font_preset,
        preferred_font=preferred_font,
        title_font_size=title_font_size,
        axis_label_font_size=axis_label_font_size,
        tick_font_size=tick_font_size,
        legend_font_size=legend_font_size,
        figure_dpi=figure_dpi,
    )


def apply_sci_grid(ax: Any, *, alpha: float = SCI_GRID_ALPHA) -> None:
    """Enable the project-wide light-gray dashed grid on an axes."""
    ax.set_axisbelow(True)
    ax.grid(
        True,
        color=SCI_GRID_COLOR,
        linestyle=SCI_GRID_LINESTYLE,
        linewidth=SCI_GRID_LINE_WIDTH,
        alpha=alpha,
    )


def add_panel_label(
    ax: Any,
    label: str,
    *,
    x: float = 0.02,
    y: float = 0.98,
    fontsize: float = 10.0,
    fontweight: str = "bold",
    ha: str = "left",
    va: str = "top",
    **kwargs: Any,
) -> Any:
    """在 matplotlib Axes 上添加加粗面板子图标签 (如 "(a)", "(b)")。

    支持位置传参与关键字传参。

    Args:
        ax: matplotlib Axes 对象。
        label: 标签文本。
        x: 相对坐标 x (默认 0.02)。
        y: 相对坐标 y (默认 0.98)。
        fontsize: 字体大小 (默认 10.0)。
        fontweight: 字体粗细 (默认 "bold")。
        ha: 水平对齐方式 (默认 "left")。
        va: 垂直对齐方式 (默认 "top")。
        **kwargs: 额外传递给 ax.text 的参数。

    Returns:
        matplotlib Text 实例。
    """
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        **kwargs,
    )


def render_trajectory_on_axes(
    *,
    ax: Any,
    pos_arr: Any,
    speed_arr: Any,
    metrics: Any = None,
    no_safeguard: bool = False,
    factor: float = 0.99,
    curve_color: str = "blue",
    curve_label: str | None = None,
    safeguard: Any = None,
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
        metrics: 轨迹指标字典或包含 to_display_mapping 的契约对象。
        no_safeguard: True 时跳过安全防护边界渲染。
        factor: 安全系数，用于构建 SafeGuardUtility（若未显式传入 safeguard）。
        curve_color: 轨迹曲线颜色。
        curve_label: 曲线图例标签。
        safeguard: 预构建的 SafeGuardUtility 实例；若为 None 且
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
            from utils.scenario import build_safeguard_utility

            resolved_safeguard = build_safeguard_utility(factor)
        else:
            resolved_safeguard = safeguard
        from model.ocs import SafeGuardUtility

        resolved_safeguard.render(ax=ax, layers=SafeGuardUtility.DANGER_VIEW_LAYERS)

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

    if render_endpoints and metrics is not None:
        display_metrics: Mapping[str, Any] | None = None
        if hasattr(metrics, "to_display_mapping"):
            display_metrics = metrics.to_display_mapping()
        elif isinstance(metrics, Mapping):
            display_metrics = metrics

        if display_metrics is not None:
            start_position = as_float(display_metrics.get("start_position_m"))
            target_position = as_float(display_metrics.get("target_position_m"))

            if start_position is not None:
                ax.scatter(
                    start_position,
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
            if target_position is not None:
                ax.scatter(
                    target_position,
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
