from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from dp.core import VariableSpacingDPOptimizer
from model.ocs import SafeGuardUtility
from model.ocs.train_service import TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.io_utils import load_curve_with_cum_time_and_metrics, save_curve_and_metrics
from utils.scenario import build_safeguard_utility
from utils.trajectory import (
    OptimizedCurveArtifact,
    compute_comfort_metrics_from_trajectory,
)

__all__ = [
    # 常量
    "DP_CURVE_FILENAME",
    "DP_DEFAULT_SEARCH_DIR",
    # 轨迹
    "resolve_dp_curve_artifact",
    "load_dp_curve_artifact",
    "compute_dp_reference_curve",
    # 可视化
    "render_dp_curve_on_axes",
]

DP_CURVE_FILENAME = "optimized_speed_curve.npz"
DP_DEFAULT_SEARCH_DIR = "output/optimal/dp"


def _metric_as_float(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _find_latest_named_file(*, search_dir: str, file_name: str) -> Path:
    """在目录下递归搜索匹配 file_name 的最新文件（按修改时间降序）。"""
    search_root = Path(search_dir)
    if not search_root.is_dir():
        raise FileNotFoundError(f"Search directory does not exist: {search_dir}")

    matches = sorted(
        (path for path in search_root.rglob(file_name) if path.is_file()),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{file_name}' under directory: {search_dir}"
        )

    if len(matches) > 1:
        print(
            f"Found {len(matches)} '{file_name}' files under '{search_dir}', "
            f"using latest: {matches[0]}"
        )
    return matches[0]


def resolve_dp_curve_artifact(*, curve_dir: str) -> OptimizedCurveArtifact:
    """在 DP 优化输出目录中定位速度曲线产物。

    Args:
        curve_dir: DP 输出根目录，递归搜索最新的 optimized_speed_curve.npz。

    Returns:
        ResolvedDPCurveArtifact 实例。

    Raises:
        FileNotFoundError: 未找到曲线文件或对应的 metrics 文件。
    """
    curve_path = _find_latest_named_file(
        search_dir=curve_dir,
        file_name=DP_CURVE_FILENAME,
    )
    metrics_file_name = curve_path.stem + "_metrics.json"
    metrics_path = curve_path.with_name(metrics_file_name)
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Could not find '{metrics_file_name}' in the same directory as "
            f"curve file: {curve_path}"
        )
    return OptimizedCurveArtifact(
        npz_path=str(curve_path),
        metrics_path=str(metrics_path),
    )


def load_dp_curve_artifact(
    artifact: OptimizedCurveArtifact,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """加载 DP 轨迹产物中的位置数组、速度数组、累计时间数组和指标字典。

    Args:
        artifact: 由 resolve_dp_curve_artifact 返回的产物定位信息。

    Returns:
        (pos_arr, speed_arr, cum_time_arr, metrics) 四元组。
    """
    pos_arr, speed_arr, cum_time_arr, metrics = load_curve_with_cum_time_and_metrics(
        npz_path=artifact.npz_path,
        metrics_path=artifact.metrics_path,
        dtype=np.float32,
        use_metrics_cache=True,
    )

    return pos_arr, speed_arr, cum_time_arr, metrics


def compute_dp_reference_curve(
    *,
    vehicle: VehicleInfo,
    track: TrackInfo,
    safeguard_utility: SafeGuardUtility,
    train_service: TrainService,
    output_dir: str | Path,
    start_position: float,
    start_speed: float,
    target_position: float,
    schedule_time_s: float,
    target_speed: float = 0.0,
    max_speed: float | None = None,
    delta_speed_mps: float = 0.1,
    max_outer_iterations: int = 100,
    precompute_mode: str = "parallel",
    precompute_workers: int | None = 4,
    precompute_chunk_size: int | None = None,
    mp_start_method: str | None = None,
    stage_division: str = "variable",
    uniform_step_size: float = 100.0,
    sub_stage_count: int = 30,
    skip_disk_cache: bool = False,
    show_precompute_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Compute, save, and return a DP trajectory usable as an ORS reference."""
    if abs(float(target_speed)) > 1e-9:
        raise ValueError("VariableSpacingDPOptimizer currently requires target_speed=0")

    task_train_service = replace(
        train_service,
        start_position=float(start_position),
        start_speed=float(start_speed),
        target_position=float(target_position),
        schedule_time=float(schedule_time_s),
    )
    optimizer = VariableSpacingDPOptimizer(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=task_train_service,
        show_precompute_progress=show_precompute_progress,
        precompute_mode=precompute_mode,  # type: ignore[arg-type]
        precompute_workers=precompute_workers,
        precompute_chunk_size=precompute_chunk_size,
        mp_start_method=mp_start_method,
        stage_division=stage_division,  # type: ignore[arg-type]
        uniform_step_size=uniform_step_size,
        sub_stage_count=sub_stage_count,
        skip_disk_cache=skip_disk_cache,
    )
    result = optimizer.optimize(
        max_speed=float(vehicle.max_speed if max_speed is None else max_speed),
        delta_speed=float(delta_speed_mps),
        max_iters=int(max_outer_iterations),
    )
    if result is None:
        raise RuntimeError("DP optimization did not find a feasible trajectory")

    comfort_metrics = compute_comfort_metrics_from_trajectory(
        pos_arr=result["pos"],
        speed_arr=result["speed"],
        max_acc_change=task_train_service.max_acc_change,
    )
    metrics: dict[str, object] = {
        "target_time_s": float(task_train_service.schedule_time),
        "total_time_s": float(result["total_time"]),
        "time_error_s": float(task_train_service.schedule_time - result["total_time"]),
        "total_energy_kj": float(result["total_energy"]),
        "start_position_m": float(task_train_service.start_position),
        "start_speed_mps": float(task_train_service.start_speed),
        "target_position_m": float(task_train_service.target_position),
        "target_speed_mps": float(target_speed),
        **comfort_metrics,
    }

    output_path = Path(output_dir) / DP_CURVE_FILENAME
    save_curve_and_metrics(
        pos_arr=result["pos"],
        speed_arr=result["speed"],
        output_path=str(output_path),
        extra_arrays={"cum_time_s": result["cum_time_s"]},
        metrics=metrics,
    )

    return (
        np.asarray(result["pos"], dtype=np.float32),
        np.asarray(result["speed"], dtype=np.float32),
        np.asarray(result["cum_time_s"], dtype=np.float32),
        metrics,
    )


def render_dp_curve_on_axes(
    *,
    ax: Any,
    pos_arr: np.ndarray,
    speed_arr: np.ndarray,
    metrics: dict[str, object],
    no_safeguard: bool,
    factor: float,
    curve_color: str = "tab:red",
    curve_label: str | None = None,
    safeguard: SafeGuardUtility | None = None,
) -> None:
    """在给定的 matplotlib Axes 上渲染 DP 速度曲线及安全防护边界。

    Args:
        ax: matplotlib Axes 对象。
        pos_arr: 位置数组 (m)。
        speed_arr: 速度数组 (m/s)。
        metrics: 轨迹指标字典。
        no_safeguard: True 时跳过安全防护边界渲染。
        factor: 安全系数。
        curve_color: 曲线颜色，默认 "tab:red"。
        curve_label: 图例标签，None 时使用 "DP optimized speed curve"。
        safeguard: 预构建的 SafeGuardUtility，None 时按 factor 构建。
    """
    if not no_safeguard:
        resolved_safeguard = (
            safeguard if safeguard is not None else build_safeguard_utility(factor)
        )
        resolved_safeguard.render(ax=ax, layers=SafeGuardUtility.DANGER_VIEW_LAYERS)

    ax.plot(
        pos_arr,
        speed_arr * 3.6,
        color=curve_color,
        alpha=0.85,
        linewidth=1.5,
        label=curve_label or "DP optimized speed curve",
    )

    start_position = _metric_as_float(metrics.get("start_position_m"))
    target_position = _metric_as_float(metrics.get("target_position_m"))

    if start_position is not None:
        ax.scatter(
            start_position,
            0.0,
            marker="o",
            color="green",
            s=40,
            alpha=0.85,
            label="start",
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )
    if target_position is not None:
        ax.scatter(
            target_position,
            0.0,
            marker="o",
            color="red",
            s=40,
            alpha=0.85,
            label="end",
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_xlim((0.0, 30000.0))
    ax.set_ylim((0.0, 500.0))
    ax.grid(True, alpha=0.3)
