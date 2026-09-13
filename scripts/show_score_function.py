"""Visualize terminal stopping and punctuality score functions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from model.ocs import TrainService
from rl.reward_calculator import RewardCalculator
from utils.plot_utils import (
    VIS_HARD_LIMIT_RED,
    VIS_SAFE_BLUE,
    add_panel_label,
    apply_sci_figure_layout,
    apply_sci_grid,
    save_sci_figure,
    set_global_plot_style,
)

FIGURE_FILENAMES = {
    "combined": "score_functions.pdf",
    "stopping": "stopping_score_function.pdf",
    "punctuality": "punctuality_score_function.pdf",
}


def create_default_reward_calculator(
    *,
    schedule_time_s: float = 0.0,
    max_stop_error_m: float = 0.3,
    max_arr_time_error_s: float = TrainService.DEFAULT_MAX_ARR_TIME_ERROR_S,
) -> RewardCalculator:
    """Create a standard RewardCalculator instance for visualization."""
    train_service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=schedule_time_s,
        max_acc_change=0.75,
        max_stop_error=max_stop_error_m,
        max_arr_time_error_s=max_arr_time_error_s,
    )
    return RewardCalculator(
        train_service,
        max_episode_steps=100,
        whole_distance_m=100.0,
        max_energy_consumption_kj=100.0,
        gamma=0.998,
    )


DEFAULT_CALCULATOR = create_default_reward_calculator()
MAX_STOP_ERROR_M = DEFAULT_CALCULATOR.train_service.max_stop_error
MAX_TIME_ERROR_S = DEFAULT_CALCULATOR.train_service.max_arr_time_error_s
STOPPING_ERROR_MAX_M = 10.0
PUNCTUALITY_ERROR_MAX_S = 140.0
PUNCTUALITY_DECAY_TIME_S = RewardCalculator.PUNCTUALITY_DECAY_TIME_S
STOPPING_SCORE_BETA = RewardCalculator.STOPPING_SCORE_BETA


def current_stopping_score(
    abs_stop_error_m: NDArray[np.floating] | float,
    calculator: RewardCalculator | None = None,
) -> NDArray[np.float64] | np.float64:
    """Evaluate stopping score by delegating directly to RewardCalculator."""
    calc = calculator or DEFAULT_CALCULATOR
    if isinstance(abs_stop_error_m, (int, float, np.floating)):
        return float(calc.stopping_score(float(abs_stop_error_m)))
    arr = np.asarray(abs_stop_error_m, dtype=np.float64)
    values = np.fromiter(
        (calc.stopping_score(float(x)) for x in arr.flat),
        dtype=np.float64,
        count=arr.size,
    ).reshape(arr.shape)
    return values


def current_punctuality_score(
    abs_time_error_s: NDArray[np.floating] | float,
    calculator: RewardCalculator | None = None,
) -> NDArray[np.float64] | np.float64:
    """Evaluate punctuality score by delegating directly to RewardCalculator."""
    calc = calculator or DEFAULT_CALCULATOR
    sched = calc.train_service.schedule_time
    if isinstance(abs_time_error_s, (int, float, np.floating)):
        return float(calc.punctuality_score(sched + float(abs_time_error_s)))
    arr = np.asarray(abs_time_error_s, dtype=np.float64)
    values = np.fromiter(
        (calc.punctuality_score(sched + float(x)) for x in arr.flat),
        dtype=np.float64,
        count=arr.size,
    ).reshape(arr.shape)
    return values


def visualize_stopping_score_function(
    calculator: RewardCalculator | None = None,
) -> Figure:
    calc = calculator or DEFAULT_CALCULATOR
    max_stop_error_m = calc.train_service.max_stop_error
    x_values = np.linspace(0, STOPPING_ERROR_MAX_M, 1000)
    rewards = current_stopping_score(x_values, calculator=calc)

    fig, ax_score = plt.subplots()

    _ = ax_score.plot(
        x_values,
        rewards,
        label=rf"$f_s(x)=\frac{{1}}{{1+(\max(0,x-x_1)/{calc.STOPPING_SCORE_BETA})^2}}$",
        color=VIS_SAFE_BLUE,
        linewidth=1.8,
    )

    _ = ax_score.axvline(
        x=max_stop_error_m,
        color=VIS_HARD_LIMIT_RED,
        linestyle="--",
        linewidth=1.2,
        label=rf"$x_1 = {max_stop_error_m}\,\mathrm{{m}}$",
    )
    _ = ax_score.axhline(y=0, color="black", linewidth=1)
    _ = ax_score.set_ylabel("Stopping score", fontsize=12)
    _ = ax_score.set_ylim(0.0, 1.15)
    apply_sci_grid(ax_score)
    _ = ax_score.legend(loc="upper right", fontsize=11, frameon=False)

    _ = ax_score.set_xlabel(r"$|\Delta x|\ (\mathrm{m})$", fontsize=12)
    _ = ax_score.set_xlim(0, STOPPING_ERROR_MAX_M)

    apply_sci_figure_layout(fig, columns=1, height_in=2.6)
    return fig


def visualize_punctuality_score_function(
    calculator: RewardCalculator | None = None,
) -> Figure:
    calc = calculator or DEFAULT_CALCULATOR
    max_time_error_s = calc.train_service.max_arr_time_error_s
    x_values = np.linspace(0, PUNCTUALITY_ERROR_MAX_S, 1000)
    rewards = current_punctuality_score(x_values, calculator=calc)

    fig, ax_score = plt.subplots()

    _ = ax_score.plot(
        x_values,
        rewards,
        label=rf"$f_t(x)=\exp\left(-x/{calc.PUNCTUALITY_DECAY_TIME_S:.0f}\right)$",
        color=VIS_SAFE_BLUE,
        linewidth=1.8,
    )
    _ = ax_score.axvline(
        x=max_time_error_s,
        color=VIS_HARD_LIMIT_RED,
        linestyle="--",
        linewidth=1.2,
        label=rf"$t_1 = {max_time_error_s:.0f}\,\mathrm{{s}}$",
    )
    _ = ax_score.axhline(y=0, color="black", linewidth=1)
    _ = ax_score.set_ylabel("Punctuality score", fontsize=12)
    _ = ax_score.set_ylim(0.0, 1.15)
    _ = ax_score.set_xlim(0, PUNCTUALITY_ERROR_MAX_S)
    apply_sci_grid(ax_score)
    _ = ax_score.legend(loc="upper right", fontsize=11, frameon=False)

    _ = ax_score.set_xlabel(r"$|\Delta t|\ (\mathrm{s})$", fontsize=12)

    apply_sci_figure_layout(fig, columns=1, height_in=2.6)
    return fig


def visualize_combined_score_functions(
    calculator: RewardCalculator | None = None,
) -> Figure:
    """在同一画幅中展示当前训练环境使用的停站和准点评分函数。"""
    calc = calculator or DEFAULT_CALCULATOR
    max_stop_error_m = calc.train_service.max_stop_error
    max_time_error_s = calc.train_service.max_arr_time_error_s

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # ---- 左子图：停站分数 ----
    x_stop = np.linspace(0, STOPPING_ERROR_MAX_M, 1000)

    ax1.plot(
        x_stop,
        current_stopping_score(x_stop, calculator=calc),
        label=r"$f_{\mathrm{S}}(|\Delta x|)$",
        color=VIS_SAFE_BLUE,
        linewidth=1.8,
    )
    ax1.axvline(
        x=max_stop_error_m,
        color=VIS_HARD_LIMIT_RED,
        linestyle="--",
        linewidth=1.2,
        label=rf"$\epsilon_x = {max_stop_error_m}\,\mathrm{{m}}$",
    )
    apply_sci_grid(ax1)
    ax1.set_xlim(0.0, STOPPING_ERROR_MAX_M)
    ax1.set_ylim(0.0, 1.15)
    ax1.set_xlabel(r"$|\Delta x|\ (\mathrm{m})$")
    ax1.set_ylabel("Stopping score")
    ax1.legend(loc="upper right", frameon=False)
    add_panel_label(ax1, "(a)")

    # ---- 右子图：准时分 ----
    x_punct = np.linspace(0, PUNCTUALITY_ERROR_MAX_S, 1000)

    ax2.plot(
        x_punct,
        current_punctuality_score(x_punct, calculator=calc),
        label=r"$f_{\mathrm{T}}(|\Delta t|)$",
        color=VIS_SAFE_BLUE,
        linewidth=1.8,
    )
    ax2.axvline(
        x=max_time_error_s,
        color=VIS_HARD_LIMIT_RED,
        linestyle="--",
        linewidth=1.2,
        label=rf"$\epsilon_t = {max_time_error_s:.0f}\,\mathrm{{s}}$",
    )
    apply_sci_grid(ax2)
    ax2.set_xlim(0, PUNCTUALITY_ERROR_MAX_S)
    ax2.set_ylim(0.0, 1.15)
    ax2.set_xlabel(r"$|\Delta t|\ (\mathrm{s})$")
    ax2.set_ylabel("Punctuality score")
    ax2.legend(loc="upper right", frameon=False)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        "--output-dir",
        type=Path,
        help=(
            "Directory for the fixed-name paper-ready PDF. If omitted, only "
            "display the figure."
        ),
    )
    _ = parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save without opening the interactive display window.",
    )
    return parser.parse_args(argv)


def save_compact_figure(
    fig: Figure,
    output_dir: Path,
    *,
    plot_type: str,
) -> Path:
    return save_sci_figure(fig, output_dir / FIGURE_FILENAMES[plot_type])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _ = set_global_plot_style(
        font_preset="sci",
        preferred_font="Arial",
        title_font_size=8.0,
        axis_label_font_size=8.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=150.0,
    )
    visualizers = {
        "combined": visualize_combined_score_functions,
        "stopping": visualize_stopping_score_function,
        "punctuality": visualize_punctuality_score_function,
    }
    fig = visualizers[args.plot]()

    if args.output_dir is not None:
        output_file = save_compact_figure(fig, args.output_dir, plot_type=args.plot)
        print(f"Saved compact figure to {output_file}")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
