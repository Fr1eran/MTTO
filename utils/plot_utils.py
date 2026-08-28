from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

MM_PER_INCH = 25.4
SCI_SINGLE_COLUMN_WIDTH_IN = 85.0 / MM_PER_INCH
SCI_DOUBLE_COLUMN_WIDTH_IN = 180.0 / MM_PER_INCH
SCI_EXPORT_DPI = 300.0
SCI_EXPORT_PAD_INCHES = 0.02

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

COMMERCIAL_FONT_FALLBACKS: dict[str, tuple[str, ...]] = {
    "Calibri": ("Carlito", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"),
    "Arial": ("Liberation Sans", "Nimbus Sans", "DejaVu Sans"),
    "Helvetica": ("Nimbus Sans", "Liberation Sans", "DejaVu Sans"),
    "Times New Roman": (
        "Liberation Serif",
        "Nimbus Roman",
        "TeX Gyre Termes",
        "DejaVu Serif",
    ),
}

# Backward-compatible alias: kept for callers importing this symbol directly.
DEFAULT_FONT_CANDIDATES: tuple[str, ...] = CHINESE_FONT_CANDIDATES


def sci_column_width_in(columns: Literal[1, 2]) -> float:
    """Return the standard 85 mm / 180 mm SCI column width in inches."""
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
    dpi: float = SCI_EXPORT_DPI,
    pad_inches: float = SCI_EXPORT_PAD_INCHES,
    transparent: bool = False,
) -> Path:
    """Save a compact high-resolution figure with a controlled outer margin."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, float | str | bool] = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "pad_inches": pad_inches,
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
        preferred_font: 用户指定字体名。若该字体不可用，会尝试开源兼容替代。
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
