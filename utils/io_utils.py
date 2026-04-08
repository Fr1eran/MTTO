import json
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@lru_cache(maxsize=32)
def _LoadJsonCached(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_optimized_curve_and_metrics(
    npz_path: str,
    metrics_path: str | None = None,
    *,
    dtype: np.dtype | type[np.floating] = np.float32,
    use_metrics_cache: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
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
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

    return pos_arr, speed_arr, metrics


def save_curve_and_metrics(
    pos_arr: Sequence[float] | NDArray,
    speed_arr: Sequence[float] | NDArray,
    output_path: str,
    metrics: dict[str, Any] | None = None,
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
