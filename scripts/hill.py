import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from numpy.typing import NDArray


def run_workbench():
    # 1. 初始化画布与布局 (留出底部空间给滑动条)
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(bottom=0.25)

    # 2. 设置初始参数
    init_x1 = 3.0
    init_beta = 4.0

    # 3. 生成自变量数据 (x 轴)
    x = np.linspace(0, 15, 1000)

    # 4. 计算得分与梯度的核心函数
    def calculate_curves(
        x: NDArray[np.floating], x1: float, beta: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        delta = np.maximum(0.0, x - x1)
        # 计算得分
        score = 1.0 / (1.0 + (delta / beta) ** 2)
        # 计算梯度绝对值 (分段解析导数)
        denom = (1.0 + (delta / beta) ** 2) ** 2
        grad = np.where(x > x1, (2.0 * delta) / (beta**2 * denom), 0.0)
        return score, grad

    score, grad = calculate_curves(x, init_x1, init_beta)

    # 5. 绘制初始曲线 (完美复刻你的配色意图)
    (line_score,) = ax.plot(x, score, "b-", lw=2.5, label="Stopping Score (f(x))")
    (line_grad,) = ax.plot(x, grad, "r--", lw=2.5, label="Gradient Magnitude (|f'(x)|)")

    # 绘制辅助指示线
    v_line_x1 = ax.axvline(
        x=init_x1, color="g", linestyle=":", lw=2, label="Deadzone Boundary (x1)"
    )
    peak_x = init_x1 + init_beta / np.sqrt(3.0)
    v_line_peak = ax.axvline(
        x=peak_x,
        color="purple",
        linestyle=":",
        lw=2,
        label="Peak Gradient Position",
    )

    # 6. 美化图表样式
    _ = ax.set_xlim(0, 15)
    _ = ax.set_ylim(0, 1.1)
    _ = ax.set_xlabel(r"$\Delta x$", fontsize=14)
    _ = ax.set_ylabel("Score / Gradient", fontsize=12)
    ax.grid(True, linestyle="-", alpha=0.3)
    _ = ax.legend(loc="upper right", fontsize=10)

    # 动态信息文本展示
    info_text = ax.text(
        0.5,
        0.2,
        "",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    def update_info_text(x1: float, beta: float):
        peak_pos = x1 + beta / np.sqrt(3.0)
        max_grad = (3.0 * np.sqrt(3.0)) / (8.0 * beta)
        info_text.set_text(
            "Parameters:\n"
            + f" ↳ x1 (Deadzone) = {x1:.2f}\n"
            + f" ↳ beta (Scale) = {beta:.2f}\n"
            + "Metrics:\n"
            + f" ↳ Max Gradient Location = {peak_pos:.2f}\n"
            + f" ↳ Max Gradient Value = {max_grad:.3f}"
        )

    update_info_text(init_x1, init_beta)

    # 7. 创建滑动条组件
    ax_x1 = plt.axes((0.15, 0.12, 0.7, 0.03))
    ax_beta = plt.axes((0.15, 0.05, 0.7, 0.03))

    slider_x1 = Slider(
        ax_x1, "x1 (Threshold)", 0.0, 8.0, valinit=init_x1, valfmt="%.2f", color="green"
    )
    slider_beta = Slider(
        ax_beta,
        "beta (Scale)",
        0.5,
        10.0,
        valinit=init_beta,
        valfmt="%.2f",
        color="red",
    )

    # 8. 回调更新函数
    def update(_val: float):
        x1 = slider_x1.val
        beta = slider_beta.val

        # 重新计算并更新曲线数据
        new_score, new_grad = calculate_curves(x, x1, beta)
        line_score.set_ydata(new_score)
        line_grad.set_ydata(new_grad)

        # 更新辅助垂直线位置
        v_line_x1.set_xdata([x1, x1])
        new_peak_x = x1 + beta / np.sqrt(3.0)
        v_line_peak.set_xdata([new_peak_x, new_peak_x])

        # 更新文本面板
        update_info_text(x1, beta)

        # 重新渲染画布
        fig.canvas.draw_idle()

    _ = slider_x1.on_changed(update)
    _ = slider_beta.on_changed(update)

    plt.show()


if __name__ == "__main__":
    run_workbench()
