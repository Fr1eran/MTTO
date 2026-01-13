import json
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.Track import Track
from model.SafeGuard import SafeGuardUtility
from model.Vehicle import Vehicle
from model.Task import Task
from model.ORS import ORS
from model.ECC import ECC
from utils.misc import SetChineseFont

# 坡度，百分位
with open("data/rail/raw/slopes.json", "r", encoding="utf-8") as f:
    slope_data = json.load(f)
    slopes = slope_data["slopes"]
    slope_intervals = slope_data["intervals"]

# 区间限速
with open("data/rail/raw/speed_limits.json", "r", encoding="utf-8") as f:
    speedlimit_data = json.load(f)
    speed_limits = speedlimit_data["speed_limits"]
    speed_limits = np.asarray(speed_limits) / 3.6
    speed_limit_intervals = speedlimit_data["intervals"]

# 车站
with open("data/rail/raw/stations.json", "r", encoding="utf-8") as f:
    stations_data = json.load(f)
    ly_zp = stations_data["LY"]["zp"]
    pa_zp = stations_data["PA"]["zp"]

with open("data/rail/safeguard/levi_curves_list.pkl", "rb") as f:
    levi_curves_list = pickle.load(f)
with open("data/rail/safeguard/brake_curves_list.pkl", "rb") as f:
    brake_curves_list = pickle.load(f)


gamma: float = 0.99
sgu = SafeGuardUtility(
    speed_limits=speed_limits,
    speed_limit_intervals=speed_limit_intervals,
    min_curves_list=levi_curves_list,
    max_curves_list=brake_curves_list,
    gamma=gamma,
)

track = Track(slopes, slope_intervals, speed_limits.tolist(), speed_limit_intervals)
vehicle = Vehicle(
    mass=317.5,
    numoftrainsets=5,
    length=128.5,
    max_acc=1.0,
    max_dec=1.0,
    levi_power_per_mass=1.7,
)
task = Task(
    start_position=ly_zp,
    start_speed=0.0,
    target_position=pa_zp,
    schedule_time=440.0,
    max_acc_change=0.75,
    max_arr_time_error=120.0,
    max_stop_error=0.3,
)
ors = ORS(vehicle=vehicle, track=track, task=task, gamma=gamma)
ecc = ECC(
    R_m=0.2796,
    L_d=0.0002,
    R_k=50.0,
    L_k=0.000142,
    Tau=0.258,
    Psi_fd=3.9629,
    k_c=0.8,
)

# 初始化位置和速度
begin_pos = ly_zp
begin_speed = 0.0
max_speed = float(np.max(speed_limits))

end_pos = task.target_position
end_speed = 0.0

# 设置matplotlib
SetChineseFont()

# 创建初始图形（只创建一次）
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制静态元素（区间限速、危险速度域和终点等）
sgu.Render(ax=ax)

ax.scatter(
    end_pos,
    end_speed * 3.6,
    marker="o",
    color="red",
    s=100,
    alpha=0.8,
    label="终点",
    zorder=5,
    edgecolors="black",
    linewidths=1.5,
)

ax.set_xlim((0.0, 30000.0))
ax.set_ylim((0.0, 500.0))
ax.set_xlabel(r"位置$s\left( m \right)$")
ax.set_ylabel(r"速度$v\left( km/h \right)$")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

# 初始化动态元素的引用（起点和运行曲线）
start_point = None
curve_line = None

# 添加说明文本
instructions = (
    "键盘命令：\n"
    "  Y - 随机生成起点并绘制曲线\n"
    "  I - 手动输入起点参数并绘制曲线\n"
    "  P - 在控制台打印当前参数\n"
    "  Q - 退出程序"
)
fig.text(
    0.02,
    0.98,
    instructions,
    transform=fig.transFigure,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# 更新标题显示命令提示
ax.set_title(
    "极限操作模式下的最短运行时间曲线\n(请在图形窗口中按 Y/I/P/Q 键操作)", fontsize=12
)


# 绘制曲线的通用函数
def draw_curve(pos, speed) -> bool:
    global start_point, curve_line, info_text

    try:
        # 计算最短运行时间曲线
        min_curve_pos_array, min_curve_speed_array = ors.CalMinRuntimeCurve(
            begin_pos=pos, begin_speed=speed
        )

        # 计算最速操作模式下的总运行时间和能耗
        PEC, LEC, total_operation_time = ors.CalRefEnergyAndOperationTime(
            begin_pos=pos, begin_speed=speed, distance=end_pos - pos, ecc=ecc
        )
        total_energy = PEC + LEC

        # 移除旧的起点和曲线（如果存在）
        if start_point is not None:
            start_point.remove()
        if curve_line is not None:
            curve_line.remove()

        # 绘制新的起点
        start_point = ax.scatter(
            pos,
            speed * 3.6,
            marker="o",
            color="yellow",
            s=100,
            alpha=0.8,
            label="起点",
            zorder=5,
            edgecolors="black",
            linewidths=1.5,
        )

        # 绘制新的最短运行时间曲线
        (curve_line,) = ax.plot(
            min_curve_pos_array,
            min_curve_speed_array * 3.6,
            label="最短运行时间曲线",
            color="blue",
            alpha=0.7,
            linewidth=2,
        )

        # 更新标题
        ax.set_title(
            f"极限操作模式下的最短运行时间曲线\n起点: ({pos:.2f}m, {speed:.2f}m/s) 运行能耗: {total_energy:.2f} 运行时间: {total_operation_time:.2f}\n按 Y/I/P/Q 操作",
            fontsize=12,
        )

        # 更新图例
        ax.legend(loc="upper right")

        fig.canvas.draw_idle()
        return True

    except Exception as e:
        print(f"计算或绘制时出错: {e}")
        return False


# 定义键盘事件处理函数
def on_key(event):
    global begin_pos, begin_speed, start_point, curve_line, info_text

    if event.key is None:
        return

    key = event.key.lower()

    if key == "y":
        # 随机初始化并绘制曲线
        begin_pos = float(np.random.uniform(ly_zp, pa_zp))
        begin_speed = float(np.random.uniform(0.0, max_speed))
        while sgu.DetectDanger(pos=begin_pos, speed=begin_speed):
            begin_pos = float(np.random.uniform(ly_zp, pa_zp))
            begin_speed = float(np.random.uniform(0.0, max_speed))

        if not draw_curve(begin_pos, begin_speed):
            print("[I] 曲线绘制失败！")

    elif key == "i":
        # 手动输入起点位置和速度
        print("\n" + "=" * 50)
        print("[I] 手动输入起点参数")
        print("=" * 50)

        try:
            # 输入位置
            pos_input = input(
                f"请输入起点位置 (m) [范围: {ly_zp:.2f} - {pa_zp:.2f}]: "
            ).strip()
            if not pos_input:
                print("[I] 取消操作")
                return

            input_pos = float(pos_input)
            if input_pos < ly_zp or input_pos > pa_zp:
                print(f"[I] 错误：位置超出范围 ({ly_zp:.2f} - {pa_zp:.2f})")
                return

            # 输入速度
            speed_input = input(
                f"请输入起点速度 (m/s) [范围: 0 - {max_speed:.2f}]: "
            ).strip()
            if not speed_input:
                print("[I] 取消操作")
                return

            input_speed = float(speed_input)

            if input_speed < 0 or input_speed > max_speed:
                print(f"[I] 错误：速度超出范围 (0 - {max_speed * 3.6:.2f} km/h)")
                return

            # 检查是否在危险区域
            if sgu.DetectDanger(pos=input_pos, speed=input_speed):
                print("[I] 警告：该起点位于危险速度域内！")
                confirm = input("[I] 是否继续绘制？(y/n): ").strip().lower()
                if confirm != "y":
                    print("[I] 已取消操作")
                    return

            # 更新全局变量
            begin_pos = input_pos
            begin_speed = input_speed

            print(f"[I] 使用起点位置: {begin_pos:.2f} m")
            print(f"[I] 使用起点速度: {begin_speed:.2f} m/s")

            if draw_curve(begin_pos, begin_speed):
                print("[I] 曲线绘制完成！")

        except ValueError:
            print("输入错误：请输入有效的数字")
        except Exception as e:
            print(f"[I] 发生错误: {e}")

    elif key == "p":
        # 输出当前参数
        print("\n" + "=" * 50)
        print("[P] 当前参数：")
        print(f"    起点位置 (begin_pos): {begin_pos:.2f} m")
        print(f"    起点速度 (begin_speed): {begin_speed:.2f} m/s")
        print("=" * 50)

    elif key == "q":
        # 退出程序
        print("\n[Q] 正在退出程序...")
        plt.close(fig)
        print("[Q] 程序已退出。")


# 连接键盘事件
fig.canvas.mpl_connect("key_press_event", on_key)

# 显示图形并阻塞
plt.show()
