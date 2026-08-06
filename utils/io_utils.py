import json
import os
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray


def format_float_token(value: float, *, decimals: int = 10) -> str:
    """将浮点数格式化为路径安全的 token 字符串。

    小数点替换为 'p'，负号替换为 'neg'，末尾零删除。
    例如: 430.0 → '430p0', 0.1 → '0p1', -1.5 → 'neg1p5'。

    Args:
        value: 有限浮点数。
        decimals: 四舍五入后保留的小数位数，默认 10。

    Raises:
        ValueError: value 不是有限数时抛出。
    """
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    token = f"{round(float(value), decimals):.{decimals}f}".rstrip("0").rstrip(".")
    if token in {"", "-0", "0"}:
        token = "0"
    if "." not in token:
        token = f"{token}.0"
    return token.replace("-", "neg").replace(".", "p")


@lru_cache(maxsize=32)
def _LoadJsonCached(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_optimized_curve_and_metrics(
    npz_path: str,
    metrics_path: str | None = None,
    *,
    dtype: np.dtype | type[np.floating] = np.float32,
    use_metrics_cache: bool = True,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    dict[str, Any],
]:
    """Load optimized trajectory arrays and metrics payload."""
    with np.load(npz_path, allow_pickle=False) as npz_data:
        keys = set(npz_data.files)
        pos_key = "pos_m" if "pos_m" in keys else "pos"
        speed_key = "speed_mps" if "speed_mps" in keys else "speed"
        pos_arr = np.asarray(npz_data[pos_key], dtype=dtype)
        speed_arr = np.asarray(npz_data[speed_key], dtype=dtype)

    if metrics_path is None:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        metrics_path = os.path.join(
            os.path.dirname(npz_path), f"{base_name}_metrics.json"
        )

    metrics: dict[str, Any] = {}
    if os.path.exists(metrics_path):
        if use_metrics_cache:
            metrics = dict(_LoadJsonCached(metrics_path))
        else:
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)

    return pos_arr, speed_arr, metrics


def load_curve_with_cum_time_and_metrics(
    npz_path: str,
    metrics_path: str | None = None,
    *,
    dtype: np.dtype | type[np.floating] = np.float32,
    use_metrics_cache: bool = True,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    dict[str, Any],
]:
    """Load trajectory arrays, including cumulative time when available."""
    pos_arr, speed_arr, metrics = load_optimized_curve_and_metrics(
        npz_path=npz_path,
        metrics_path=metrics_path,
        dtype=dtype,
        use_metrics_cache=use_metrics_cache,
    )
    with np.load(npz_path, allow_pickle=False) as npz_data:
        if "cum_time_s" in npz_data.files:
            cum_time_arr = np.asarray(npz_data["cum_time_s"], dtype=dtype)
        else:
            from utils.trajectory import recover_time_axis_from_trajectory

            cum_time_arr = recover_time_axis_from_trajectory(
                pos_arr,
                speed_arr,
            ).astype(dtype, copy=False)

    return pos_arr, speed_arr, cum_time_arr, metrics


def save_curve_and_metrics(
    pos_arr: Sequence[float] | NDArray[np.floating],
    speed_arr: Sequence[float] | NDArray[np.floating],
    output_path: str,
    metrics: dict[str, Any] | None = None,
    extra_arrays: dict[str, Sequence[float] | NDArray[np.floating]] | None = None,
) -> tuple[str, str]:
    """Save trajectory arrays to NPZ and metrics payload to JSON."""
    pos = np.asarray(pos_arr, dtype=np.float32)
    speed = np.asarray(speed_arr, dtype=np.float32)
    created_at = datetime.now().isoformat(timespec="seconds")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(output_path))[0]
    metrics_json_path = os.path.join(output_dir, f"{base_name}_metrics.json")

    npz_payload: dict[str, NDArray[np.floating] | NDArray[np.str_]] = {
        "pos_m": pos,
        "speed_mps": speed,
        "created_at": np.asarray([created_at], dtype=np.str_),
    }
    if extra_arrays:
        for key, value in extra_arrays.items():
            npz_payload[str(key)] = np.asarray(value, dtype=np.float32)

    np.savez_compressed(output_path, allow_pickle=True, **npz_payload)

    metrics_payload: dict[str, object] = {"created_at": created_at}
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, np.generic):
                metrics_payload[key] = value.item()
            else:
                metrics_payload[key] = value

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    return output_path, metrics_json_path
