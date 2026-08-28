import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from utils.plot_utils import (
    SCI_EXPORT_PAD_INCHES,
    apply_sci_figure_layout,
    save_sci_figure,
    set_global_plot_style,
)

MAX_STOP_ERROR_M = 0.3
MAX_TIME_ERROR_S = 10.0
PUNCTUALITY_DECAY_TIME_S = 30.0
STOPPING_ERROR_MAX_M = 10.0
PUNCTUALITY_ERROR_MAX_S = 140.0


def current_stopping_score(
    abs_stop_error_m: NDArray[np.floating] | float,
) -> NDArray[np.float64] | np.float64:
    """Mirror MTTOEnv._calc_stopping_score for positive absolute stop error."""
    abs_stop_error_m = np.asarray(abs_stop_error_m, dtype=np.float64)
    beta = 0.8
    delta = np.maximum(0.0, abs_stop_error_m - MAX_STOP_ERROR_M)
    return 1.0 / (1.0 + (delta / beta) ** 2)


def current_punctuality_score(
    abs_time_error_s: NDArray[np.floating] | float,
) -> NDArray[np.float64] | np.float64:
    """Mirror MTTOEnv._calc_punctuality_score for positive absolute time error."""
    abs_time_error_s = np.asarray(abs_time_error_s, dtype=np.float64)
    return np.exp(-abs_time_error_s / PUNCTUALITY_DECAY_TIME_S)


def visualize_stopping_score_function():
    x_values = np.linspace(0, STOPPING_ERROR_MAX_M, 1000)
    rewards = current_stopping_score(x_values)

    fig, ax_score = plt.subplots()

    _ = ax_score.plot(
        x_values,
        rewards,
        label=r"$f_s(x)=\frac{1}{1+\max(0,x-x_1)^2}$",
        color="blue",
        linewidth=2.5,
    )

    _ = ax_score.axvline(
        x=MAX_STOP_ERROR_M,
        color="green",
        linestyle=":",
        label=rf"$x_1 = {MAX_STOP_ERROR_M}\,\mathrm{{m}}$",
    )
    _ = ax_score.axhline(y=0, color="black", linewidth=1)
    _ = ax_score.set_ylabel("stopping score", fontsize=12)
    _ = ax_score.set_ylim(0.0, 1.05)
    ax_score.grid(True, alpha=0.3)
    _ = ax_score.legend(loc="upper right", fontsize=11)

    _ = ax_score.set_xlabel(r"$|\Delta x|$ / m", fontsize=12)
    _ = ax_score.set_xlim(0, STOPPING_ERROR_MAX_M)

    apply_sci_figure_layout(fig, columns=1, height_in=2.6)
    return fig


def visualize_punctuality_score_function():
    x_values = np.linspace(0, PUNCTUALITY_ERROR_MAX_S, 1000)
    rewards = current_punctuality_score(x_values)

    fig, ax_score = plt.subplots()

    _ = ax_score.plot(
        x_values,
        rewards,
        label=r"$f_t(x)=\exp\left(-x/60\right)$",
        color="blue",
        linewidth=2.5,
    )
    _ = ax_score.axhline(y=0, color="black", linewidth=1)
    _ = ax_score.set_ylabel("punctuality score", fontsize=12)
    _ = ax_score.set_ylim(0.0, 1.05)
    _ = ax_score.set_xlim(0, PUNCTUALITY_ERROR_MAX_S)
    ax_score.grid(True, alpha=0.3)
    _ = ax_score.legend(loc="upper right", fontsize=11)

    _ = ax_score.set_xlabel(r"$|\Delta t|$ / s", fontsize=12)

    apply_sci_figure_layout(fig, columns=1, height_in=2.6)
    return fig


def visualize_combined_score_functions():
    """在同一画幅中展示当前训练环境使用的停站和准点评分函数。"""

    fig, (ax1, ax2) = plt.subplots(1, 2)

    def add_panel_label(ax: Axes, label: str) -> None:
        _ = ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    # ---- 左子图：停站分数 ----
    x_stop = np.linspace(0, STOPPING_ERROR_MAX_M, 1000)

    ax1.plot(
        x_stop,
        current_stopping_score(x_stop),
        label=r"$f_{\mathrm{S}}\left( x \right)$",
        color="blue",
        linewidth=2,
    )
    ax1.axvline(
        x=MAX_STOP_ERROR_M,
        color="magenta",
        linestyle=":",
        label=f"{MAX_STOP_ERROR_M}m",
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.0, STOPPING_ERROR_MAX_M)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel(r"$|\Delta x|$")
    ax1.set_ylabel("stopping score")
    ax1.legend()
    add_panel_label(ax1, "(a)")

    # ---- 右子图：准时分 ----
    x_punct = np.linspace(0, PUNCTUALITY_ERROR_MAX_S, 1000)

    ax2.plot(
        x_punct,
        current_punctuality_score(x_punct),
        label=r"$f_{\mathrm{T}}\left( x \right)$",
        color="blue",
        linewidth=2,
    )
    ax2.axvline(
        x=MAX_TIME_ERROR_S,
        color="magenta",
        linestyle=":",
        label=f"{MAX_TIME_ERROR_S}s",
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, PUNCTUALITY_ERROR_MAX_S)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel(r"$|\Delta t|$")
    ax2.set_ylabel("punctuality score")
    ax2.legend()
    add_panel_label(ax2, "(b)")

    apply_sci_figure_layout(
        fig,
        columns=2,
        height_in=3.0,
        left=0.10,
        bottom=0.18,
        top=0.95,
        wspace=0.30,
    )
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize score functions and optionally save a compact figure."
    )
    _ = parser.add_argument(
        "--plot",
        choices=("combined", "stopping", "punctuality"),
        default="combined",
        help="Score function figure to display.",
    )
    _ = parser.add_argument(
        "--output-file",
        type=Path,
        help=(
            "Path for saving a compact paper-ready figure. "
            "If omitted, only display the figure."
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


def save_compact_figure(
    fig: Figure,
    output_file: Path,
    dpi: float,
    pad_inches: float,
) -> Path:
    if output_file.suffix == "":
        output_file = output_file.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return save_sci_figure(fig, output_file, dpi=dpi, pad_inches=pad_inches)


if __name__ == "__main__":
    args = parse_args()
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
    visualizers = {
        "combined": visualize_combined_score_functions,
        "stopping": visualize_stopping_score_function,
        "punctuality": visualize_punctuality_score_function,
    }
    fig = visualizers[args.plot]()

    if args.output_file is not None:
        output_file = save_compact_figure(
            fig,
            args.output_file,
            args.dpi,
            args.pad_inches,
        )
        print(f"Saved compact figure to {output_file}")

    if not args.no_show:
        plt.show()
