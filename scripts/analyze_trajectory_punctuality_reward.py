from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from dp.experiment_utils import (
    DP_DEFAULT_SEARCH_DIR,
    load_dp_curve_artifact,
    resolve_dp_curve_artifact,
)
from model.common import ORS
from rl.experiment_utils import (
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_SCHEDULE_TIME_S,
    RL_DEFAULT_SEARCH_DIR,
    RL_TRAJECTORY_SOURCE_CHOICES,
    load_rl_curve_artifact,
    resolve_rl_curve_artifact,
)
from utils.plot_utils import set_global_plot_style
from utils.scenario import build_scenario
from utils.trajectory import OptimizedCurveArtifact, recover_time_axis_from_trajectory

TRAJECTORY_KIND_CHOICES: tuple[str, ...] = ("dp", "rl")
POTENTIAL_VERSION_CHOICES: tuple[str, ...] = ("v34",)

OPERATION_TIME_KEYS: tuple[str, ...] = (
    "cum_time_s",
    "operation_time_s",
    "operation_time",
)
REDUNDANT_TIME_KEYS: tuple[str, ...] = (
    "redundant_operation_time_s",
    "redundant_operation_time",
    "ref_redundant_operation_time_s",
    "ref_redundant_operation_time",
    "redundant_time_s",
)


@dataclass(frozen=True)
class TimeSeries:
    values: NDArray[np.float64]
    source: str
    key: str | None = None


@dataclass(frozen=True)
class TrajectoryData:
    artifact: OptimizedCurveArtifact
    trajectory_kind: str
    trajectory_source: str
    pos_arr: NDArray[np.float64]
    speed_arr: NDArray[np.float64]
    operation_time_arr: NDArray[np.float64]
    operation_time_source: str
    metrics: dict[str, object]


@dataclass(frozen=True)
class TargetContext:
    schedule_time_s: float
    start_position_m: float
    start_speed_mps: float
    target_position_m: float
    target_speed_mps: float


@dataclass(frozen=True)
class RewardAnalysis:
    redundant_time_arr: NDArray[np.float64]
    redundant_time_source: str
    redundant_time_key: str | None
    max_redundant_time_s: float
    punctuality_error_arr: NDArray[np.float64]
    phi_arr: NDArray[np.float64]
    reward_arr: NDArray[np.float64]
    reward_x_arr: NDArray[np.float64]


def _as_1d_float_array(
    values: Sequence[float] | NDArray[Any],
    name: str,
) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_same_length(
    named_arrays: Sequence[tuple[str, NDArray[np.float64]]],
) -> None:
    sizes = {arr.size for _name, arr in named_arrays}
    if len(sizes) != 1:
        names = ", ".join(name for name, _arr in named_arrays)
        raise ValueError(f"{names} must have the same length")


def _metric_as_float(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _resolve_schedule_time(metrics: dict[str, object]) -> float:
    target_time = _metric_as_float(metrics, "target_time_s")
    if target_time is not None and target_time > 0.0:
        return target_time
    return float(DEFAULT_SCHEDULE_TIME_S)


def _find_1d_array_in_npz(
    npz_path: str | Path,
    *,
    keys: Sequence[str],
    expected_size: int,
    explicit_key: str | None = None,
) -> TimeSeries | None:
    with np.load(npz_path, allow_pickle=False) as npz_data:
        available_keys = tuple(npz_data.files)
        if explicit_key is not None:
            if explicit_key not in npz_data.files:
                raise ValueError(
                    f"NPZ file does not contain key {explicit_key!r}. "
                    f"Available keys: {available_keys!r}"
                )
            selected_key = explicit_key
        else:
            selected_key = next((key for key in keys if key in npz_data.files), None)
            if selected_key is None:
                return None

        arr = _as_1d_float_array(npz_data[selected_key], selected_key)

    if arr.size != int(expected_size):
        raise ValueError(
            f"Array '{selected_key}' length {arr.size} does not match trajectory "
            f"length {int(expected_size)}"
        )
    return TimeSeries(
        values=arr,
        source=f"npz:{selected_key}",
        key=selected_key,
    )


def load_or_recover_operation_time(
    *,
    npz_path: str | Path,
    pos_arr: Sequence[float] | NDArray[Any],
    speed_arr: Sequence[float] | NDArray[Any],
) -> TimeSeries:
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    speed = _as_1d_float_array(speed_arr, "speed_arr")
    _validate_same_length((("pos_arr", pos), ("speed_arr", speed)))

    loaded = _find_1d_array_in_npz(
        npz_path,
        keys=OPERATION_TIME_KEYS,
        expected_size=pos.size,
    )
    if loaded is not None:
        return loaded

    recovered = recover_time_axis_from_trajectory(pos, speed).astype(
        np.float64,
        copy=False,
    )
    return TimeSeries(
        values=recovered,
        source="recovered_from_position_speed",
        key=None,
    )


def reconstruct_redundant_operation_time(
    *,
    pos_arr: Sequence[float] | NDArray[Any],
    speed_arr: Sequence[float] | NDArray[Any],
    operation_time_arr: Sequence[float] | NDArray[Any],
    schedule_time_s: float,
    target_position_m: float,
    target_speed_mps: float,
    min_remaining_time_fn: Callable[[float, float, float, float], float],
) -> NDArray[np.float64]:
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    speed = _as_1d_float_array(speed_arr, "speed_arr")
    operation_time = _as_1d_float_array(operation_time_arr, "operation_time_arr")
    _validate_same_length((
        ("pos_arr", pos),
        ("speed_arr", speed),
        ("operation_time_arr", operation_time),
    ))

    min_remaining_arr = np.asarray(
        [
            min_remaining_time_fn(
                float(pos_val),
                float(speed_val),
                float(target_position_m),
                float(target_speed_mps),
            )
            for pos_val, speed_val in zip(pos, speed, strict=False)
        ],
        dtype=np.float64,
    )
    if min_remaining_arr.ndim != 1 or min_remaining_arr.size != pos.size:
        raise ValueError("min_remaining_time_fn must return one scalar per sample")
    if not np.all(np.isfinite(min_remaining_arr)):
        raise ValueError("min_remaining_time_fn returned non-finite values")
    return float(schedule_time_s) - operation_time - min_remaining_arr


def load_or_reconstruct_redundant_operation_time(
    *,
    npz_path: str | Path,
    pos_arr: Sequence[float] | NDArray[Any],
    speed_arr: Sequence[float] | NDArray[Any],
    operation_time_arr: Sequence[float] | NDArray[Any],
    schedule_time_s: float,
    target_position_m: float,
    target_speed_mps: float,
    redundant_key: str | None,
    min_remaining_time_fn: Callable[[float, float, float, float], float],
) -> TimeSeries:
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    loaded = _find_1d_array_in_npz(
        npz_path,
        keys=REDUNDANT_TIME_KEYS,
        expected_size=pos.size,
        explicit_key=redundant_key,
    )
    if loaded is not None:
        return loaded

    reconstructed = reconstruct_redundant_operation_time(
        pos_arr=pos,
        speed_arr=speed_arr,
        operation_time_arr=operation_time_arr,
        schedule_time_s=schedule_time_s,
        target_position_m=target_position_m,
        target_speed_mps=target_speed_mps,
        min_remaining_time_fn=min_remaining_time_fn,
    )
    return TimeSeries(
        values=reconstructed,
        source="reconstructed_from_operation_time",
        key=None,
    )


def compute_punctuality_error_v34(
    pos_arr: Sequence[float] | NDArray[Any],
    redundant_operation_time_arr: Sequence[float] | NDArray[Any],
    *,
    start_position_m: float,
    target_position_m: float,
    max_redundant_time_s: float,
) -> NDArray[np.float64]:
    """Compute the v34 redundancy error used by the environment potential.

    This mirrors ``MTTOEnv._potential_punctuality_v34``:
    ``e_r = redundant_time - r_exp`` and
    ``r_exp = max_redundant_time * (1 - progress)``.
    """
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    redundant = _as_1d_float_array(
        redundant_operation_time_arr,
        "redundant_operation_time_arr",
    )
    _validate_same_length((
        ("pos_arr", pos),
        ("redundant_operation_time_arr", redundant),
    ))
    whole_distance = abs(float(target_position_m) - float(start_position_m))
    if not math.isfinite(whole_distance) or whole_distance <= 0.0:
        raise ValueError("whole_distance_m must be finite and positive")
    if not math.isfinite(float(max_redundant_time_s)) or max_redundant_time_s <= 0.0:
        raise ValueError("max_redundant_time_s must be finite and positive")

    dist_to_target = float(target_position_m) - pos
    progress = np.minimum(
        1.0,
        1.0 - dist_to_target / whole_distance,
    )
    expected_redundant = float(max_redundant_time_s) * (1.0 - progress)
    return redundant - expected_redundant


def punctuality_potential_v34(
    pos_arr: Sequence[float] | NDArray[Any],
    redundant_operation_time_arr: Sequence[float] | NDArray[Any],
    *,
    start_position_m: float,
    target_position_m: float,
    max_redundant_time_s: float,
) -> NDArray[np.float64]:
    """Vectorized equivalent of ``MTTOEnv._potential_punctuality_v34``."""
    punctuality_error = compute_punctuality_error_v34(
        pos_arr,
        redundant_operation_time_arr,
        start_position_m=start_position_m,
        target_position_m=target_position_m,
        max_redundant_time_s=max_redundant_time_s,
    )
    return -0.10 * np.abs(punctuality_error)


def compute_dense_reward_from_potential(
    phi_arr: Sequence[float] | NDArray[Any],
    *,
    gamma: float,
) -> NDArray[np.float64]:
    phi = _as_1d_float_array(phi_arr, "phi_arr")
    if not math.isfinite(float(gamma)):
        raise ValueError("gamma must be finite")
    if phi.size < 2:
        return np.asarray([], dtype=np.float64)
    return float(gamma) * phi[1:] - phi[:-1]


def summarize_reward_statistics(
    reward_arr: Sequence[float] | NDArray[Any],
    *,
    zero_eps: float = 1e-9,
) -> dict[str, float | int | None]:
    reward = _as_1d_float_array(reward_arr, "reward_arr")
    if zero_eps < 0.0 or not math.isfinite(float(zero_eps)):
        raise ValueError("zero_eps must be finite and non-negative")

    positive_count = int(np.count_nonzero(reward > zero_eps))
    negative_count = int(np.count_nonzero(reward < -zero_eps))
    return {
        "sample_count": int(reward.size),
        "sum": float(np.sum(reward)),
        "mean": float(np.mean(reward)),
        "std": float(np.std(reward)),
        "min": float(np.min(reward)),
        "max": float(np.max(reward)),
        "positive_fraction": float(positive_count / reward.size),
        "negative_fraction": float(negative_count / reward.size),
    }


def _compute_segment_x_axis(pos_arr: NDArray[np.float64]) -> NDArray[np.float64]:
    if pos_arr.size < 2:
        return np.asarray([], dtype=np.float64)
    return pos_arr[1:].astype(np.float64, copy=False)


def plot_punctuality_reward_analysis(
    *,
    pos_arr: Sequence[float] | NDArray[Any],
    operation_time_arr: Sequence[float] | NDArray[Any],
    redundant_operation_time_arr: Sequence[float] | NDArray[Any],
    punctuality_error_arr: Sequence[float] | NDArray[Any],
    phi_arr: Sequence[float] | NDArray[Any],
    reward_arr: Sequence[float] | NDArray[Any],
) -> Figure:
    pos = _as_1d_float_array(pos_arr, "pos_arr")
    operation_time = _as_1d_float_array(operation_time_arr, "operation_time_arr")
    redundant = _as_1d_float_array(
        redundant_operation_time_arr,
        "redundant_operation_time_arr",
    )
    punctuality_error = _as_1d_float_array(
        punctuality_error_arr,
        "punctuality_error_arr",
    )
    phi = _as_1d_float_array(phi_arr, "phi_arr")
    reward = np.asarray(reward_arr, dtype=np.float64)
    if reward.ndim != 1:
        raise ValueError("reward_arr must be a 1-D array")
    if not np.all(np.isfinite(reward)):
        raise ValueError("reward_arr must contain only finite values")
    _validate_same_length((
        ("pos_arr", pos),
        ("operation_time_arr", operation_time),
        ("redundant_operation_time_arr", redundant),
        ("punctuality_error_arr", punctuality_error),
        ("phi_arr", phi),
    ))
    if reward.size != max(0, pos.size - 1):
        raise ValueError("reward_arr length must be len(pos_arr) - 1")

    reward_x = _compute_segment_x_axis(pos)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10, 12.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0, 1.0]},
    )
    ax_time, ax_redundant, ax_error, ax_phi, ax_reward = axes

    ax_time.plot(pos, operation_time, color="#2563eb", linewidth=1.4)
    ax_time.set_title("Operation time over trajectory")
    ax_time.set_ylabel("Operation time (s)")
    ax_time.grid(True, alpha=0.3)

    ax_redundant.plot(pos, redundant, color="#16a34a", linewidth=1.4)
    ax_redundant.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_redundant.set_title("Redundant operation time")
    ax_redundant.set_ylabel("Redundant time (s)")
    ax_redundant.grid(True, alpha=0.3)

    ax_error.plot(pos, punctuality_error, color="#dc2626", linewidth=1.4)
    ax_error.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_error.set_title("Punctuality error $e_r$ (v34)")
    ax_error.set_ylabel("$e_r$ (s)")
    ax_error.grid(True, alpha=0.3)

    ax_phi.plot(pos, phi, color="#f97316", linewidth=1.4)
    ax_phi.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_phi.set_title("Punctuality potential v34")
    ax_phi.set_ylabel("Phi")
    ax_phi.grid(True, alpha=0.3)

    ax_reward.plot(reward_x, reward, color="#9333ea", linewidth=1.4)
    ax_reward.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_reward.set_title("Differential dense reward (v34)")
    ax_reward.set_xlabel("Position (m)")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _load_trajectory_data(
    *,
    trajectory_kind: str,
    dp_curve_dir: str,
    rl_curve_dir: str,
    trajectory_source: str,
) -> TrajectoryData:
    if trajectory_kind == "dp":
        artifact = resolve_dp_curve_artifact(curve_dir=dp_curve_dir)
        pos_arr, speed_arr, _cum_time_arr, metrics = load_dp_curve_artifact(artifact)
        artifact_source = "dp"
    elif trajectory_kind == "rl":
        artifact = resolve_rl_curve_artifact(
            curve_dir=rl_curve_dir,
            trajectory_source=trajectory_source,
        )
        pos_arr, speed_arr, metrics = load_rl_curve_artifact(artifact)
        artifact_source = trajectory_source
    else:
        choices = ", ".join(TRAJECTORY_KIND_CHOICES)
        raise ValueError(
            f"Unknown trajectory kind '{trajectory_kind}'. Choices: {choices}"
        )

    pos = _as_1d_float_array(pos_arr, "pos_arr")
    speed = _as_1d_float_array(speed_arr, "speed_arr")
    _validate_same_length((("pos_arr", pos), ("speed_arr", speed)))
    operation_time = load_or_recover_operation_time(
        npz_path=artifact.npz_path,
        pos_arr=pos,
        speed_arr=speed,
    )
    return TrajectoryData(
        artifact=artifact,
        trajectory_kind=trajectory_kind,
        trajectory_source=artifact_source,
        pos_arr=pos,
        speed_arr=speed,
        operation_time_arr=operation_time.values,
        operation_time_source=operation_time.source,
        metrics=metrics,
    )


def _resolve_target_context(data: TrajectoryData) -> TargetContext:
    schedule_time = _resolve_schedule_time(data.metrics)
    start_position = _metric_as_float(data.metrics, "start_position_m")
    if start_position is None:
        start_position = float(data.pos_arr[0])
    start_speed = _metric_as_float(data.metrics, "start_speed_mps")
    if start_speed is None:
        start_speed = float(data.speed_arr[0])
    target_position = _metric_as_float(data.metrics, "target_position_m")
    if target_position is None:
        target_position = float(data.pos_arr[-1])
    target_speed = _metric_as_float(data.metrics, "target_speed_mps")
    if target_speed is None:
        target_speed = 0.0
    return TargetContext(
        schedule_time_s=float(schedule_time),
        start_position_m=float(start_position),
        start_speed_mps=float(start_speed),
        target_position_m=float(target_position),
        target_speed_mps=float(target_speed),
    )


def _compute_default_max_redundant_time(
    *,
    context: TargetContext,
    min_remaining_time_fn: Callable[[float, float, float, float], float],
) -> float:
    """Match MTTOEnv.max_redundant_operation_time at the initial state."""
    min_operation_time = min_remaining_time_fn(
        context.start_position_m,
        context.start_speed_mps,
        context.target_position_m,
        context.target_speed_mps,
    )
    max_redundant = context.schedule_time_s - float(min_operation_time)
    if not math.isfinite(max_redundant) or max_redundant <= 0.0:
        raise ValueError(
            "Computed max_redundant_time_s must be finite and positive; "
            "provide --max-redundant-s to override it."
        )
    return max_redundant


def _analyze_trajectory(
    *,
    data: TrajectoryData,
    context: TargetContext,
    redundant_key: str | None,
    gamma: float,
    max_redundant_s: float | None,
    min_remaining_time_fn: Callable[[float, float, float, float], float],
) -> RewardAnalysis:
    redundant_series = load_or_reconstruct_redundant_operation_time(
        npz_path=data.artifact.npz_path,
        pos_arr=data.pos_arr,
        speed_arr=data.speed_arr,
        operation_time_arr=data.operation_time_arr,
        schedule_time_s=context.schedule_time_s,
        target_position_m=context.target_position_m,
        target_speed_mps=context.target_speed_mps,
        redundant_key=redundant_key,
        min_remaining_time_fn=min_remaining_time_fn,
    )
    resolved_max_redundant = (
        float(max_redundant_s)
        if max_redundant_s is not None
        else _compute_default_max_redundant_time(
            context=context,
            min_remaining_time_fn=min_remaining_time_fn,
        )
    )
    punctuality_error_arr = compute_punctuality_error_v34(
        data.pos_arr,
        redundant_series.values,
        start_position_m=context.start_position_m,
        target_position_m=context.target_position_m,
        max_redundant_time_s=resolved_max_redundant,
    )
    phi_arr = punctuality_potential_v34(
        data.pos_arr,
        redundant_series.values,
        start_position_m=context.start_position_m,
        target_position_m=context.target_position_m,
        max_redundant_time_s=resolved_max_redundant,
    )
    reward_arr = compute_dense_reward_from_potential(phi_arr, gamma=gamma)
    return RewardAnalysis(
        redundant_time_arr=redundant_series.values,
        redundant_time_source=redundant_series.source,
        redundant_time_key=redundant_series.key,
        max_redundant_time_s=resolved_max_redundant,
        punctuality_error_arr=punctuality_error_arr,
        phi_arr=phi_arr,
        reward_arr=reward_arr,
        reward_x_arr=_compute_segment_x_axis(data.pos_arr),
    )


def save_compact_figure(
    fig: Figure,
    output_file: Path,
    dpi: float,
    pad_inches: float,
) -> Path:
    if output_file.suffix == "":
        output_file = output_file.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    return output_file


def write_analysis_csv(
    *,
    output_path: Path,
    data: TrajectoryData,
    analysis: RewardAnalysis,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "position_m",
                "speed_mps",
                "operation_time_s",
                "redundant_operation_time_s",
                "punctuality_error_e_r_s",
                "punctuality_phi",
                "dense_reward_to_sample",
            ],
        )
        writer.writeheader()
        for idx, values in enumerate(
            zip(
                data.pos_arr,
                data.speed_arr,
                data.operation_time_arr,
                analysis.redundant_time_arr,
                analysis.punctuality_error_arr,
                analysis.phi_arr,
                strict=True,
            )
        ):
            reward_value = "" if idx == 0 else float(analysis.reward_arr[idx - 1])
            writer.writerow({
                "position_m": float(values[0]),
                "speed_mps": float(values[1]),
                "operation_time_s": float(values[2]),
                "redundant_operation_time_s": float(values[3]),
                "punctuality_error_e_r_s": float(values[4]),
                "punctuality_phi": float(values[5]),
                "dense_reward_to_sample": reward_value,
            })
    return output_path


def write_json_summary(output_path: Path, payload: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def print_reward_summary(summary: dict[str, float | int | None]) -> None:
    print("Dense reward summary:")
    for key in [
        "sample_count",
        "sum",
        "mean",
        "std",
        "min",
        "max",
        "positive_fraction",
        "negative_fraction",
    ]:
        value = summary.get(key)
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze punctuality potential and differential dense reward for a "
            "saved DP or RL trajectory."
        )
    )
    parser.add_argument(
        "--trajectory-kind",
        choices=TRAJECTORY_KIND_CHOICES,
        default="dp",
        help="Trajectory artifact kind to analyze.",
    )
    parser.add_argument(
        "--dp-curve-dir",
        default=DP_DEFAULT_SEARCH_DIR,
        help="Directory used to recursively search DP trajectory artifacts.",
    )
    parser.add_argument(
        "--rl-curve-dir",
        default=RL_DEFAULT_SEARCH_DIR,
        help="Directory used to recursively search RL trajectory artifacts.",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=RL_TRAJECTORY_SOURCE_CHOICES,
        default="best",
        help="RL trajectory source used when --trajectory-kind=rl.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_REWARD_DISCOUNT,
        help="Discount factor in gamma * phi_t - phi_{t-1}.",
    )
    parser.add_argument(
        "--redundant-key",
        help="Optional NPZ key to read as redundant operation time.",
    )
    parser.add_argument(
        "--max-redundant-s",
        type=float,
        help=(
            "Override the initial max redundant operation time used by the "
            "v34 potential."
        ),
    )
    parser.add_argument(
        "--potential-version",
        choices=POTENTIAL_VERSION_CHOICES,
        default="v34",
        help="Punctuality potential implementation to analyze.",
    )
    parser.add_argument(
        "--zero-eps",
        type=float,
        default=1e-9,
        help="Tolerance for positive/negative reward fraction statistics.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional path for saving the combined figure.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for saving per-sample analysis rows.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for saving summary statistics.",
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=300.0,
        help="DPI used when saving the figure.",
    )
    parser.add_argument(
        "--pad-inches",
        type=float,
        default=0.03,
        help="Padding around the tight saved figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive display window.",
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    try:
        data = _load_trajectory_data(
            trajectory_kind=args.trajectory_kind,
            dp_curve_dir=args.dp_curve_dir,
            rl_curve_dir=args.rl_curve_dir,
            trajectory_source=args.trajectory_source,
        )
        context = _resolve_target_context(data)
        vehicle, track, safeguard_utility, _train_service = build_scenario(
            schedule_time_s=context.schedule_time_s
        )
        ors = ORS(vehicle=vehicle, track=track, factor=safeguard_utility.gamma)
        analysis = _analyze_trajectory(
            data=data,
            context=context,
            redundant_key=args.redundant_key,
            gamma=args.gamma,
            max_redundant_s=args.max_redundant_s,
            min_remaining_time_fn=ors.calc_min_operation_time,
        )
        reward_summary = summarize_reward_statistics(
            analysis.reward_arr,
            zero_eps=args.zero_eps,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Using trajectory file: {data.artifact.npz_path}")
    print(f"Using metrics file: {data.artifact.metrics_path}")
    print(f"Trajectory kind: {data.trajectory_kind}")
    print(f"Trajectory source: {data.trajectory_source}")
    print(f"Potential version: {args.potential_version}")
    print(f"Gamma: {float(args.gamma):.6f}")
    print(f"Operation time source: {data.operation_time_source}")
    print(f"Redundant time source: {analysis.redundant_time_source}")
    print(f"Max redundant time: {analysis.max_redundant_time_s:.6f} s")
    print_reward_summary(reward_summary)

    payload = {
        "trajectory_file": data.artifact.npz_path,
        "metrics_file": data.artifact.metrics_path,
        "trajectory_kind": data.trajectory_kind,
        "trajectory_source": data.trajectory_source,
        "potential_version": args.potential_version,
        "gamma": float(args.gamma),
        "operation_time_source": data.operation_time_source,
        "redundant_time_source": analysis.redundant_time_source,
        "redundant_time_key": analysis.redundant_time_key,
        "max_redundant_time_s": analysis.max_redundant_time_s,
        "schedule_time_s": context.schedule_time_s,
        "target_position_m": context.target_position_m,
        "target_speed_mps": context.target_speed_mps,
        "reward_statistics": reward_summary,
    }

    if args.output_json is not None:
        output_json = write_json_summary(args.output_json, payload)
        print(f"Saved JSON summary to {output_json}")

    if args.output_csv is not None:
        output_csv = write_analysis_csv(
            output_path=args.output_csv,
            data=data,
            analysis=analysis,
        )
        print(f"Saved per-sample CSV to {output_csv}")

    set_global_plot_style(
        font_preset="sci",
        preferred_font="Calibri",
        title_font_size=9.0,
        axis_label_font_size=9.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        figure_dpi=150.0,
        savefig_dpi=300.0,
    )
    fig = plot_punctuality_reward_analysis(
        pos_arr=data.pos_arr,
        operation_time_arr=data.operation_time_arr,
        redundant_operation_time_arr=analysis.redundant_time_arr,
        punctuality_error_arr=analysis.punctuality_error_arr,
        phi_arr=analysis.phi_arr,
        reward_arr=analysis.reward_arr,
    )

    if args.output_file is not None:
        output_file = save_compact_figure(
            fig,
            args.output_file,
            args.dpi,
            args.pad_inches,
        )
        print(f"Saved compact figure to {output_file}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
