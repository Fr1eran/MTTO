import json
import os
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contracts.common import JSONValue, as_json_value
from contracts.evaluation import (
    EvaluationArtifact,
    EvaluationHistory,
    EvaluationMetrics,
    TrajectoryData,
)


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


def _resolve_metrics_path(npz_path: str, metrics_path: str | None) -> str:
    if metrics_path is not None:
        return metrics_path
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    return os.path.join(os.path.dirname(npz_path), f"{base_name}_metrics.json")


def _load_metrics(
    metrics_path: str,
    *,
    use_metrics_cache: bool,
) -> dict[str, Any]:
    if not os.path.exists(metrics_path):
        return {}
    if use_metrics_cache:
        return dict(_LoadJsonCached(metrics_path))
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def _load_curve_arrays(
    npz_path: str,
    *,
    dtype: np.dtype | type[np.floating],
    include_cum_time: bool,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating] | None,
]:
    with np.load(npz_path, allow_pickle=False) as npz_data:
        keys = set(npz_data.files)
        pos_key = "pos_m" if "pos_m" in keys else "pos"
        speed_key = "speed_mps" if "speed_mps" in keys else "speed"
        pos_arr = np.asarray(npz_data[pos_key], dtype=dtype)
        speed_arr = np.asarray(npz_data[speed_key], dtype=dtype)
        cum_time_arr = (
            np.asarray(npz_data["cum_time_s"], dtype=dtype)
            if include_cum_time and "cum_time_s" in keys
            else None
        )
    return pos_arr, speed_arr, cum_time_arr


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
    pos_arr, speed_arr, _ = _load_curve_arrays(
        npz_path,
        dtype=dtype,
        include_cum_time=False,
    )
    metrics = _load_metrics(
        _resolve_metrics_path(npz_path, metrics_path),
        use_metrics_cache=use_metrics_cache,
    )
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
    pos_arr, speed_arr, cum_time_arr = _load_curve_arrays(
        npz_path,
        dtype=dtype,
        include_cum_time=True,
    )
    if cum_time_arr is None:
        from utils.trajectory import recover_time_axis_from_trajectory

        cum_time_arr = recover_time_axis_from_trajectory(
            pos_arr,
            speed_arr,
        ).astype(dtype, copy=False)

    metrics = _load_metrics(
        _resolve_metrics_path(npz_path, metrics_path),
        use_metrics_cache=use_metrics_cache,
    )
    return pos_arr, speed_arr, cum_time_arr, metrics


def save_curve_and_metrics(
    pos_arr: Sequence[float] | NDArray[np.floating],
    speed_arr: Sequence[float] | NDArray[np.floating],
    output_path: str,
    metrics: dict[str, Any] | None = None,
    extra_arrays: dict[str, Sequence[float] | NDArray[np.floating]] | None = None,
    metrics_path: str | None = None,
) -> tuple[str, str]:
    """Save trajectory arrays to NPZ and metrics payload to JSON."""
    pos = np.asarray(pos_arr, dtype=np.float32)
    speed = np.asarray(speed_arr, dtype=np.float32)
    created_at = datetime.now().isoformat(timespec="seconds")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(output_path))[0]
    metrics_json_path = (
        metrics_path
        if metrics_path is not None
        else os.path.join(output_dir, f"{base_name}_metrics.json")
    )
    metrics_parent = os.path.dirname(metrics_json_path)
    if metrics_parent:
        os.makedirs(metrics_parent, exist_ok=True)

    npz_payload: dict[str, NDArray[np.floating] | NDArray[np.str_]] = {
        "pos_m": pos,
        "speed_mps": speed,
        "created_at": np.asarray([created_at], dtype=np.str_),
    }
    if extra_arrays:
        for key, value in extra_arrays.items():
            npz_payload[str(key)] = np.asarray(value, dtype=np.float32)

    np.savez_compressed(output_path, **npz_payload)

    metrics_payload: dict[str, object] = {"created_at": created_at}
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, np.generic):
                metrics_payload[key] = value.item()
            else:
                metrics_payload[key] = value

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    _LoadJsonCached.cache_clear()
    return output_path, metrics_json_path


def save_evaluation_artifact(
    artifact: EvaluationArtifact,
    output_path: str,
    *,
    extra_metadata: dict[str, object] | None = None,
    metrics_path: str | None = None,
) -> tuple[str, str]:
    """Persist the canonical RL evaluation artifact.

    Generic curve writers remain available for DP and legacy analysis files;
    this entry point is the only writer used for versioned RL evaluation
    metrics.  Optional context is deliberately stored under ``extensions`` so
    it cannot collide with a required metric field.
    """
    metrics = artifact.metrics
    if extra_metadata:
        extensions: dict[str, JSONValue] = dict(metrics.extensions)
        for key, value in extra_metadata.items():
            if hasattr(value, "to_mapping"):
                value = value.to_mapping()  # type: ignore[union-attr]
            extensions[str(key)] = as_json_value(value, field=f"extensions.{key}")
        metrics = replace(metrics, extensions=extensions)

    created_at = metrics.created_at or datetime.now().isoformat(timespec="seconds")
    metrics = replace(metrics, created_at=created_at)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    metrics_json_path = _resolve_metrics_path(output_path, metrics_path)
    metrics_parent = os.path.dirname(metrics_json_path)
    if metrics_parent:
        os.makedirs(metrics_parent, exist_ok=True)

    np.savez_compressed(output_path, **artifact.trajectory.to_npz_mapping())
    with open(metrics_json_path, "w", encoding="utf-8") as file_obj:
        json.dump(metrics.to_mapping(), file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    _LoadJsonCached.cache_clear()
    return output_path, metrics_json_path


def load_evaluation_artifact(
    npz_path: str,
    metrics_path: str | None = None,
    *,
    dtype: np.dtype | type[np.floating] = np.float32,
    use_metrics_cache: bool = True,
) -> EvaluationArtifact:
    """Load a canonical RL evaluation artifact with strict schema checks."""
    resolved_metrics_path = _resolve_metrics_path(npz_path, metrics_path)
    if not os.path.isfile(resolved_metrics_path):
        raise FileNotFoundError(
            f"Evaluation metrics file not found: {resolved_metrics_path}"
        )

    with np.load(npz_path, allow_pickle=False) as npz_data:
        required_keys = {
            "pos_m",
            "speed_mps",
            "safety_violation_positions_m",
        }
        present_keys = set(npz_data.files)
        missing = sorted(required_keys - present_keys)
        if missing:
            raise ValueError(
                f"Evaluation artifact is missing NPZ arrays: {', '.join(missing)}"
            )
        unknown = sorted(present_keys - required_keys)
        if unknown:
            raise ValueError(
                f"Evaluation artifact contains unknown NPZ arrays: {', '.join(unknown)}"
            )
        trajectory = TrajectoryData(
            position_m=np.asarray(npz_data["pos_m"], dtype=dtype),
            speed_mps=np.asarray(npz_data["speed_mps"], dtype=dtype),
            safety_violation_positions_m=np.asarray(
                npz_data["safety_violation_positions_m"], dtype=dtype
            ),
        )

    metrics_payload = _load_metrics(
        resolved_metrics_path,
        use_metrics_cache=use_metrics_cache,
    )
    if not metrics_payload:
        raise ValueError(
            f"Evaluation metrics payload is empty: {resolved_metrics_path}"
        )
    metrics = EvaluationMetrics.from_mapping(metrics_payload)
    return EvaluationArtifact(metrics=metrics, trajectory=trajectory)


def load_evaluation_metrics(
    metrics_path: str,
    *,
    use_metrics_cache: bool = True,
) -> EvaluationMetrics:
    """Load only the strict metrics component of a canonical artifact."""
    metrics_payload = _load_metrics(
        metrics_path,
        use_metrics_cache=use_metrics_cache,
    )
    if not metrics_payload:
        raise FileNotFoundError(f"Evaluation metrics file not found: {metrics_path}")
    return EvaluationMetrics.from_mapping(metrics_payload)


def save_evaluation_history(history: EvaluationHistory, output_path: str) -> str:
    """Persist the typed periodic evaluation history as a versioned NPZ."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(output_path, **history.to_npz_mapping())
    return output_path


def load_evaluation_history(
    input_path: str,
    *,
    use_copy: bool = True,
) -> EvaluationHistory:
    """Load a versioned periodic evaluation history with strict validation."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Evaluation history file not found: {input_path}")
    with np.load(input_path, allow_pickle=False) as data:
        payload = {
            key: np.asarray(data[key]).copy() if use_copy else data[key]
            for key in data.files
        }
    return EvaluationHistory.from_npz_mapping(payload)
