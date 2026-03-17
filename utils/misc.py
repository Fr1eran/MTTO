import numpy as np
from numpy.typing import NDArray
from typing import Union, overload, Sequence, Any
from matplotlib.font_manager import fontManager
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json


Numeric = Union[float, np.number, NDArray]


def SaveCurveAndMetrics(
    pos_arr: Sequence[float] | NDArray,
    speed_arr: Sequence[float] | NDArray,
    output_path: str,
    metrics: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    将曲线数据保存为NPZ格式, 并将性能指标保存为同目录下JSON文件。

    Args:
        pos_arr: 曲线位置序列
        speed_arr: 曲线速度序列
        output_npz_path: 数据存放路径
        metrics: 性能指标

    Returns:
        npz_path: 曲线数据存放路径
        metrics_json_path: 性能指标存放路径
    """
    pos = np.asarray(pos_arr, dtype=np.float32)
    speed = np.asarray(speed_arr, dtype=np.float32)
    created_at = datetime.now().isoformat(timespec="seconds")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(output_path))[0]
    metrics_json_path = os.path.join(output_dir, f"{base_name}_metrics.json")

    np.savez_compressed(
        output_path,
        pos_m=pos,
        speed_mps=speed,
        created_at=np.asarray([created_at], dtype=str),
    )

    metrics_payload: dict[str, Any] = {"created_at": created_at}
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, np.generic):
                metrics_payload[key] = value.item()
            else:
                metrics_payload[key] = value

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    return output_path, metrics_json_path


@overload
def GetIntervalIndex(pos: float | np.number, interval_points: NDArray) -> np.intp: ...


@overload
def GetIntervalIndex(pos: NDArray, interval_points: NDArray) -> NDArray[np.intp]: ...


def GetIntervalIndex(
    pos: Numeric, interval_points: NDArray
) -> np.intp | NDArray[np.intp]:
    """
    获得当前位置所处区间的索引
    Args:
        pos: 当前位置
        intervals: 区间端点
    """
    pos = np.asarray(pos, dtype=np.float64)
    interval_points = np.asarray(interval_points, dtype=np.float64)
    idx = np.searchsorted(interval_points, pos, side="right") - 1
    return idx


def FindSpeedRiseEntryAndFallExit(
    speed_limits,
    interval_points,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    返回所有限速开始上升的入口点列表和所有限速下降的出口点列表。
    """
    sl = np.asarray(speed_limits)
    pts = np.asarray(interval_points)

    diff = np.diff(sl)

    # diff 长度为 len(sl)-1，对应边界索引 0..len(sl)-2，边界位置为 pts[1..-2]
    n = sl.size
    if n < 2:
        return [], []

    # 默认索引范围为全部区间 [0, n-1]
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = n - 1

    diff_indices = np.arange(n - 1)
    idx_mask = (diff_indices >= int(start_idx)) & (diff_indices < int(end_idx))

    # 上升入口点：diff > 0 且在 idx_mask 范围内 -> boundary at pts[i+1]
    inc_idxs = np.where((diff > 0) & idx_mask)[0]
    ascend_begin_pos = [(float(pts[i + 1]), float(speed_limits[i])) for i in inc_idxs]

    # 下降出口点：diff < 0 且在 idx_mask 范围内
    dec_idxs = np.where((diff < 0) & idx_mask)[0]
    descend_end_pos = [
        (float(pts[j + 1]), float(speed_limits[j + 1])) for j in dec_idxs
    ]

    return ascend_begin_pos, descend_end_pos


def SetChineseFont():
    candidates = [
        "SimHei",  # common on Windows
        "Microsoft YaHei",
        "Noto Sans CJK JP",  # common on many Linux distros
        "WenQuanYi Zen Hei",
        "Source Han Sans CN",
        "Source Han Sans SC",
        "STHeiti",  # macOS fallback
    ]
    available = {f.name for f in fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break


if __name__ == "__main__":
    speed_limits = [60, 100, 150, 120, 80]
    interval_points = [0, 100, 200, 300, 400, 500]
    asc, desc = FindSpeedRiseEntryAndFallExit(
        speed_limits=speed_limits,
        interval_points=interval_points,
        start_idx=1,
        end_idx=4,
    )
    print(f"asc: {asc}")
    print(f"desc: {desc}")
    # speed_limits = [60, 100]
    # interval_points = [0, 100, 200]
    # asc, desc = FindSpeedRiseEntryAndFallExit(
    #     speed_limits=speed_limits,
    #     interval_points=interval_points,
    #     start_idx=0,
    #     end_idx=2,
    # )
    # print(f"asc: {asc}")
    # print(f"desc: {desc}")
