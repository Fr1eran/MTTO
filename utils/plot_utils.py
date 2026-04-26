import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from collections.abc import Sequence
from typing import Literal


CHINESE_FONT_CANDIDATES: tuple[str, ...] = (
    "SimHei",
    "Microsoft YaHei",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "Source Han Sans CN",
    "Source Han Sans SC",
    "STHeiti",
)

SCI_ENGLISH_FONT_CANDIDATES: tuple[str, ...] = (
    "Arial",
    "Calibri",
    "Helvetica",
    "Arial Nova",
    "Nimbus Sans",
    "Liberation Sans",
    "DejaVu Sans",
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "TeX Gyre Termes",
    "STIX Two Text",
    "STIXGeneral",
    "CMU Serif",
    "DejaVu Serif",
)

# Backward-compatible alias: kept for callers importing this symbol directly.
DEFAULT_FONT_CANDIDATES: tuple[str, ...] = CHINESE_FONT_CANDIDATES


def _pick_first_available_font(font_candidates: Sequence[str]) -> str | None:
    available = {font.name for font in fontManager.ttflist}
    for name in font_candidates:
        if name in available:
            return name
    return None


def _pick_selected_or_first_available_font(
    font_candidates: Sequence[str],
    preferred_font: str | None,
) -> str | None:
    """Pick user-selected font first, otherwise fallback to first available candidate."""

    if preferred_font is not None:
        if preferred_font not in font_candidates:
            raise ValueError(
                f"preferred_font={preferred_font!r} 不在候选字体中: {tuple(font_candidates)!r}"
            )

        selected = _pick_first_available_font((preferred_font,))
        if selected is None:
            raise ValueError(
                f"preferred_font={preferred_font!r} 在当前系统不可用，请先安装该字体。"
            )
        return selected

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


def set_global_plot_style(
    *,
    base_font_size: float = 12.0,
    title_font_size: float | None = None,
    axis_label_font_size: float | None = None,
    tick_font_size: float | None = None,
    legend_font_size: float | None = None,
    figure_dpi: float = 150.0,
    savefig_dpi: float = 300.0,
    # line_width: float = 1.5,
    # grid_alpha: float = 0.3,
    # grid_line_style: str = ":",
    unicode_minus: bool = False,
    font_preset: Literal["auto", "zh", "sci"] = "auto",
    preferred_font: str | None = None,
    font_candidates: Sequence[str] | None = None,
) -> dict[str, float | str | None]:
    """Apply a consistent Matplotlib style for the whole project.

    This function is intended to be called once at script startup so all
    subsequent figures share the same font family, font sizes and DPI.

    Args:
        font_preset: 预设候选字体集合。"sci" 为英文字体优先，"zh" 为中文字体优先。
        preferred_font: 用户指定字体名。需位于候选字体中且在系统可用。
        font_candidates: 自定义候选字体。若传入则覆盖 font_preset 对应集合。
    """

    chosen_candidates = _resolve_font_candidates(font_preset, font_candidates)
    selected_font = _pick_selected_or_first_available_font(
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

    if selected_font is not None:
        plt.rcParams["font.family"] = [selected_font]

    # Prefer journal-friendly math glyphs when using SCI style.
    if font_preset == "sci":
        plt.rcParams["mathtext.fontset"] = "stix"

    plt.rcParams["axes.unicode_minus"] = unicode_minus

    plt.rcParams["figure.dpi"] = figure_dpi
    plt.rcParams["savefig.dpi"] = savefig_dpi

    plt.rcParams["font.size"] = base_font_size
    plt.rcParams["axes.titlesize"] = effective_title_size
    plt.rcParams["figure.titlesize"] = effective_title_size
    plt.rcParams["axes.labelsize"] = effective_axis_label_size
    plt.rcParams["xtick.labelsize"] = effective_tick_size
    plt.rcParams["ytick.labelsize"] = effective_tick_size
    plt.rcParams["legend.fontsize"] = effective_legend_size

    # plt.rcParams["lines.linewidth"] = line_width
    # plt.rcParams["grid.alpha"] = grid_alpha
    # plt.rcParams["grid.linestyle"] = grid_line_style

    return {
        "font": selected_font,
        "base_font_size": base_font_size,
        "title_font_size": effective_title_size,
        "axis_label_font_size": effective_axis_label_size,
        "tick_font_size": effective_tick_size,
        "legend_font_size": effective_legend_size,
        "figure_dpi": figure_dpi,
        "savefig_dpi": savefig_dpi,
        # "line_width": line_width,
        # "grid_alpha": grid_alpha,
        # "grid_line_style": grid_line_style,
        "unicode_minus": unicode_minus,
        "font_preset": font_preset,
        "preferred_font": preferred_font,
    }


def set_chinese_font() -> None:
    selected_font = _pick_first_available_font(CHINESE_FONT_CANDIDATES)
    if selected_font is not None:
        plt.rcParams["font.family"] = [selected_font]
