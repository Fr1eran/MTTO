from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from model.ocs import SafeGuardUtility
from utils.io_utils import load_optimized_curve_and_metrics
from utils.scenario import build_safeguard_utility
from utils.trajectory import OptimizedCurveArtifact

__all__ = [
    # 常量
    "DP_CURVE_FILENAME",
    "DP_DEFAULT_SEARCH_DIR",
    # 轨迹
    "resolve_dp_curve_artifact",
    "load_dp_curve_artifact",
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
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """加载 DP 轨迹产物中的位置数组、速度数组和指标字典。

    Args:
        artifact: 由 resolve_dp_curve_artifact 返回的产物定位信息。

    Returns:
        (pos_arr, speed_arr, metrics) 三元组。
    """
    pos_arr, speed_arr, metrics = load_optimized_curve_and_metrics(
        npz_path=artifact.npz_path,
        metrics_path=artifact.metrics_path,
        dtype=np.float32,
        use_metrics_cache=True,
    )
    return pos_arr, speed_arr, metrics


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
