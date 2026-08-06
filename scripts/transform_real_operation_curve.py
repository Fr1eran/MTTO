from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from utils.data_loader import load_excel, load_stations_goal_positions

DEFAULT_INPUT_FILE = "data/operation/a_longyang_to_airport.xlsx"
DEFAULT_SHEET_NAME = "a轨_双端两步4节_龙阳－机场"
DEFAULT_OUTPUT_FILE = "output/real_operation/aligned_real_operation_curve.npz"
REQUIRED_COLUMNS = ("里程(km)", "速度(km/h)", "加速度(m/s2)", "时间(s)")
MIN_TIME_STEP_S = 1e-9


def _as_1d_float_array(
    name: str, values: NDArray[np.floating] | Sequence[float]
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return array


def recompute_time_and_acceleration(
    *,
    position_m: NDArray[np.float64],
    speed_mps: NDArray[np.float64],
    source_time_s: NDArray[np.float64],
    position_scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    position = _as_1d_float_array("position_m", position_m)
    speed = _as_1d_float_array("speed_mps", speed_mps)
    source_time = _as_1d_float_array("source_time_s", source_time_s)
    if not (position.size == speed.size == source_time.size):
        raise ValueError("position_m, speed_mps, and source_time_s must match length")
    if position.size < 2:
        raise ValueError("operation curve must contain at least two valid samples")

    delta_pos = np.diff(position)
    avg_speed = 0.5 * (speed[:-1] + speed[1:])
    source_dt = np.diff(source_time)
    positive_source_dt = source_dt[source_dt > MIN_TIME_STEP_S]
    default_dt = (
        float(np.median(positive_source_dt)) if positive_source_dt.size else 0.0
    )
    fallback_dt = np.where(
        source_dt > MIN_TIME_STEP_S,
        source_dt,
        default_dt,
    ) * float(position_scale)

    segment_dt = np.where(
        (delta_pos > 0.0) & (avg_speed > MIN_TIME_STEP_S),
        delta_pos / avg_speed,
        fallback_dt,
    )
    if np.any(segment_dt < 0.0):
        raise ValueError("recomputed segment time must be non-negative")

    time_s = np.concatenate([[0.0], np.cumsum(segment_dt)])
    delta_speed = np.diff(speed)
    segment_acc = np.divide(
        delta_speed,
        segment_dt,
        out=np.zeros_like(delta_speed),
        where=segment_dt > MIN_TIME_STEP_S,
    )
    acc_mps2 = np.empty_like(speed)
    acc_mps2[0] = segment_acc[0]
    acc_mps2[1:] = segment_acc
    return time_s.astype(np.float64, copy=False), acc_mps2.astype(
        np.float64,
        copy=False,
    )


def transform_operation_curve_arrays(
    *,
    source_position_m: NDArray[np.floating] | Sequence[float],
    speed_kmh: NDArray[np.floating] | Sequence[float],
    acc_mps2: NDArray[np.floating] | Sequence[float],
    time_s: NDArray[np.floating] | Sequence[float],
    start_position_m: float,
    target_position_m: float,
) -> dict[str, NDArray[np.float64] | np.float64]:
    source_position = _as_1d_float_array("source_position_m", source_position_m)
    speed_kmh_arr = _as_1d_float_array("speed_kmh", speed_kmh)
    acc_arr = _as_1d_float_array("acc_mps2", acc_mps2)
    time_arr = _as_1d_float_array("time_s", time_s)

    sample_count = source_position.size
    if sample_count < 2:
        raise ValueError("operation curve must contain at least two valid samples")
    if not (
        speed_kmh_arr.size == sample_count
        and acc_arr.size == sample_count
        and time_arr.size == sample_count
    ):
        raise ValueError(
            "source_position_m, speed_kmh, acc_mps2, and time_s must "
            "have the same length"
        )
    if not np.all(np.isfinite(source_position)):
        raise ValueError("source_position_m must contain only finite values")
    if not np.all(np.isfinite(speed_kmh_arr)):
        raise ValueError("speed_kmh must contain only finite values")
    if not np.all(np.isfinite(acc_arr)):
        raise ValueError("acc_mps2 must contain only finite values")
    if not np.all(np.isfinite(time_arr)):
        raise ValueError("time_s must contain only finite values")

    start_position = float(start_position_m)
    target_position = float(target_position_m)
    if not np.isfinite(start_position) or not np.isfinite(target_position):
        raise ValueError("start_position_m and target_position_m must be finite")
    if target_position <= start_position:
        raise ValueError("target_position_m must be greater than start_position_m")

    source_start = float(source_position[0])
    source_target = float(source_position[-1])
    source_span = source_target - source_start
    if source_span <= 0.0:
        raise ValueError("source position span must be positive")

    target_span = target_position - start_position
    position_scale = target_span / source_span
    progress = (source_position - source_start) / source_span
    position_m = start_position + progress * target_span
    speed_mps = (speed_kmh_arr / 3.6).astype(np.float64, copy=False)
    recomputed_time_s, recomputed_acc_mps2 = recompute_time_and_acceleration(
        position_m=position_m,
        speed_mps=speed_mps,
        source_time_s=time_arr,
        position_scale=position_scale,
    )

    return {
        "position_m": position_m.astype(np.float64, copy=False),
        "speed_mps": speed_mps,
        "acc_mps2": recomputed_acc_mps2,
        "time_s": recomputed_time_s,
        "source_position_m": source_position.astype(np.float64, copy=False),
        "source_time_s": time_arr.astype(np.float64, copy=False),
        "source_acc_mps2": acc_arr.astype(np.float64, copy=False),
        "start_position_m": np.float64(start_position),
        "target_position_m": np.float64(target_position),
        "source_start_position_m": np.float64(source_start),
        "source_target_position_m": np.float64(source_target),
        "position_scale": np.float64(position_scale),
    }


def load_real_operation_curve(
    *,
    input_file: str | Path,
    sheet_name: str,
    start_position_m: float,
    target_position_m: float,
) -> dict[str, NDArray[np.float64] | np.float64]:
    raw_data = load_excel(
        input_file,
        sheet_name=sheet_name,
        header=0,
        dtype=np.float64,
    )
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in raw_data.columns
    ]
    if missing_columns:
        raise ValueError(f"Excel data missing columns: {missing_columns}")

    samples = raw_data.loc[:, REQUIRED_COLUMNS].dropna()
    source_position_m = samples["里程(km)"].to_numpy(dtype=np.float64) * 1000.0
    speed_kmh = samples["速度(km/h)"].to_numpy(dtype=np.float64)
    acc_mps2 = samples["加速度(m/s2)"].to_numpy(dtype=np.float64)
    time_s = samples["时间(s)"].to_numpy(dtype=np.float64)

    return transform_operation_curve_arrays(
        source_position_m=source_position_m,
        speed_kmh=speed_kmh,
        acc_mps2=acc_mps2,
        time_s=time_s,
        start_position_m=start_position_m,
        target_position_m=target_position_m,
    )


def save_transformed_operation_curve(
    output_file: str | Path,
    curve_arrays: dict[str, NDArray[np.float64] | np.float64],
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        allow_pickle=True,
        **{
            key: np.asarray(value, dtype=np.float64)
            for key, value in curve_arrays.items()
        },
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将实际运行曲线线性重标定到指定起点和终点，并保存为 NPZ 数组。"
    )
    _ = parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="实际运行数据 Excel 文件路径。",
    )
    _ = parser.add_argument(
        "--sheet-name",
        default=DEFAULT_SHEET_NAME,
        help="实际运行数据所在工作表名称。",
    )
    _ = parser.add_argument(
        "--start-position",
        type=float,
        default=None,
        help="目标起点位置 (m)。默认读取 stations.json 的 start_station.target。",
    )
    _ = parser.add_argument(
        "--target-position",
        type=float,
        default=None,
        help="目标终点位置 (m)。默认读取 stations.json 的 end_station.target。",
    )
    _ = parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="输出 NPZ 文件路径。",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    default_start, default_target = load_stations_goal_positions()
    start_position_m = (
        default_start if args.start_position is None else float(args.start_position)
    )
    target_position_m = (
        default_target if args.target_position is None else float(args.target_position)
    )

    try:
        curve_arrays = load_real_operation_curve(
            input_file=args.input_file,
            sheet_name=args.sheet_name,
            start_position_m=start_position_m,
            target_position_m=target_position_m,
        )
        output_path = save_transformed_operation_curve(args.output_file, curve_arrays)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Saved transformed operation curve to: {output_path}")
    print(
        "Source position: "
        + f"{float(curve_arrays['source_start_position_m']):.3f} m -> "
        + f"{float(curve_arrays['source_target_position_m']):.3f} m"
    )
    print(
        "Target position: "
        + f"{float(curve_arrays['start_position_m']):.3f} m -> "
        + f"{float(curve_arrays['target_position_m']):.3f} m"
    )
    print(f"Position scale: {float(curve_arrays['position_scale']):.9f}")
    print(f"Samples: {np.asarray(curve_arrays['position_m']).size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
