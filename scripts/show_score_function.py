import matplotlib.pyplot as plt
import numpy as np

from utils.plot_utils import set_global_plot_style
from utils.score_function import SigmoidVariant


def visualize_stopping_score_function():
    stopping_score_func = SigmoidVariant(x1=0.3, x2=3.0, c=10.0)

    x_values = np.linspace(0, 4.0, 1000)

    rewards = stopping_score_func(x_values)
    gradients = stopping_score_func.gradient(x_values)

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, rewards, label=r"$f(x)$", color="blue", linewidth=2.5)

    gradient_magnitude = np.abs(gradients)
    plt.plot(
        x_values,
        gradient_magnitude * 3,
        label=r"$f'(x)$",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    # 标记关键点和参考线
    plt.axvline(
        x=stopping_score_func.x1,
        color="green",
        linestyle=":",
        label=f"$x_1 = {stopping_score_func.x1}$",
    )
    plt.axvline(
        x=stopping_score_func.x2,
        color="purple",
        linestyle=":",
        label=f"$x_2 = {stopping_score_func.x2}$",
    )
    plt.axhline(y=0, color="black", linewidth=1)

    # 图表设置
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$y$", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=11)

    # 限制 y 轴范围便于观察
    # plt.ylim(-0.1, 1.2)
    plt.xlim(0, 4.0)

    plt.tight_layout()
    plt.show()


def visualize_punctuality_score_function():
    schedule_time = 430.0
    max_arr_time_ratio = 0.05
    punctuality_score_func = SigmoidVariant(
        x1=schedule_time * max_arr_time_ratio,
        x2=schedule_time * max_arr_time_ratio * 5,
        c=8.0,
    )

    x_values = np.linspace(0, 140.0, 1000)

    rewards = punctuality_score_func(x_values)
    gradients = punctuality_score_func.gradient(x_values)

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, rewards, label=r"$f(x)$", color="blue", linewidth=2.5)

    gradient_magnitude = np.abs(gradients)
    plt.plot(
        x_values,
        gradient_magnitude * 3,
        label=r"$f^{\prime}\left( x \right)$",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    # 标记关键点和参考线
    plt.axvline(
        x=punctuality_score_func.x1,
        color="green",
        linestyle=":",
        label=f"$x_1 = {punctuality_score_func.x1}$",
    )
    plt.axvline(
        x=punctuality_score_func.x2,
        color="purple",
        linestyle=":",
        label=f"$x_2 = {punctuality_score_func.x2}$",
    )
    plt.axhline(y=0, color="black", linewidth=1)

    # 图表设置
    plt.xlabel(r"$x$", fontsize=12)
    plt.ylabel(r"$y$", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=11)

    # 限制 y 轴范围便于观察
    plt.ylim(-0.1, 1.2)
    plt.xlim(0, 140.0)

    plt.tight_layout()
    plt.show()


def visualize_combined_score_functions():
    """在同一画幅中上下布局展示停站分数和准时分数的函数图像和导函数图像"""
    stopping_score_func = SigmoidVariant(x1=0.3, x2=3.0, c=5.0)
    schedule_time = 430.0
    max_arr_time_ratio = 0.05
    punctuality_score_func = SigmoidVariant(
        x1=schedule_time * max_arr_time_ratio,
        x2=schedule_time * max_arr_time_ratio * 2,
        c=6.0,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))

    # ---- 上子图：停站分数 ----
    x_stop = np.linspace(0, 4.0, 1000)
    rewards_stop = stopping_score_func(x_stop)
    gradients_stop = stopping_score_func.gradient(x_stop)

    ax1.plot(x_stop, rewards_stop, label=r"$f(x)$", color="blue", linewidth=2)
    ax1.plot(
        x_stop,
        np.abs(gradients_stop) * 3,
        label=r"$f'(x)$",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    ax1.axvline(
        x=stopping_score_func.x1,
        color="green",
        linestyle=":",
        label=r"$x_1$",
    )
    ax1.axvline(
        x=stopping_score_func.x2,
        color="purple",
        linestyle=":",
        label=r"$x_2$",
    )
    ax1.axhline(y=0, color="black", linewidth=1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4.0)
    ax1.text(
        0.5,
        -0.20,
        "(a)",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontweight="normal",
    )

    # ---- 下子图：准时分 ----
    x_punct = np.linspace(0, 100.0, 1000)
    rewards_punct = punctuality_score_func(x_punct)
    gradients_punct = punctuality_score_func.gradient(x_punct)

    ax2.plot(x_punct, rewards_punct, label=r"$f(x)$", color="blue", linewidth=2)
    ax2.plot(
        x_punct,
        np.abs(gradients_punct) * 3,
        color="red",
        linestyle="--",
        linewidth=2,
    )
    ax2.axvline(
        x=punctuality_score_func.x1,
        color="green",
        linestyle=":",
    )
    ax2.axvline(
        x=punctuality_score_func.x2,
        color="purple",
        linestyle=":",
    )
    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 140.0)
    ax2.text(
        0.5,
        -0.20,
        "(b)",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontweight="normal",
    )

    # 共用图例，置于整张画布正上方
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        bbox_transform=fig.transFigure,
        ncol=len(handles),
        fontsize=12,
        frameon=True,
    )

    fig.subplots_adjust(top=0.90, bottom=0.15, hspace=0.55)
    plt.show()


if __name__ == "__main__":
    set_global_plot_style(
        font_preset="sci",
        preferred_font="Calibri",
        title_font_size=12.0,
        axis_label_font_size=12.0,
        tick_font_size=10.0,
        legend_font_size=12.0,
        figure_dpi=150.0,
        savefig_dpi=300.0,
    )
    visualize_combined_score_functions()

