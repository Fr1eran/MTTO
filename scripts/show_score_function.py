import numpy as np
import matplotlib.pyplot as plt

from utils.score_function import SigmoidVariant


def visualize_docking_score_function():
    docking_score_func = SigmoidVariant(x1=0.3, x2=3.0, c=6.0)

    x_values = np.linspace(0, 4.0, 1000)

    rewards = docking_score_func(x_values)
    gradients = docking_score_func.gradient(x_values)

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
        x=docking_score_func.x1,
        color="green",
        linestyle=":",
        label=f"$x_1 = {docking_score_func.x1}$",
    )
    plt.axvline(
        x=docking_score_func.x2,
        color="purple",
        linestyle=":",
        label=f"$x_2 = {docking_score_func.x2}$",
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
    punctuality_score_func = SigmoidVariant(x1=22.0, x2=120.0, c=7.0)

    x_values = np.linspace(0, 140.0, 1000)

    rewards = punctuality_score_func(x_values)
    gradients = punctuality_score_func.gradient(x_values)

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


if __name__ == "__main__":
    # visualize_docking_score_function()
    visualize_punctuality_score_function()
