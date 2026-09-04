from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from model.ocs import SafeGuardCurves
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from utils.data_loader import (
    load_acceleration_zones,
    load_auxiliary_stopping_areas_ap_and_dp,
    load_slopes,
    load_speed_limits,
    resolve_project_path,
)

DEFAULT_OUTPUT_DIR = Path("output/safeguardcurves")
DEFAULT_VEHICLE_LENGTH_M = 128.5
METADATA_FILENAME = "metadata.json"
CURVE_FILENAMES = {
    "levi_curves_list": "levi_curves_list.pkl",
    "brake_curves_list": "brake_curves_list.pkl",
    "min_curves_list": "min_curves_list.pkl",
    "max_curves_list": "max_curves_list.pkl",
}


@dataclass(frozen=True)
class SafeguardCurveConfig:
    distance_step_m: float = 1.0
    mass_tonnes: float = 317.5
    trainset_count: int = 5
    max_acceleration_mps2: float = 1.0
    max_deceleration_mps2: float = 1.0
    position_error_m: float = 1.0
    speed_error_mps: float = 0.1
    traction_cutoff_delay_s: float = 0.5
    vortex_brake_delay_s: float = 0.5
    min_curve_position_offset_m: float = 0.0
    include_acceleration_zone_end: bool = True


@dataclass(frozen=True)
class CalculationInputs:
    calculator: SafeGuardCurves
    vehicle: VehicleInfo
    accessible_points: NDArray[np.float64]
    dangerous_points: NDArray[np.float64]


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算并保存上海磁浮示范线的安全防护曲线。"
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"防护曲线输出目录；相对路径按项目根目录解析。默认: {DEFAULT_OUTPUT_DIR}",
    )
    _ = parser.add_argument(
        "--distance-step-m",
        type=float,
        default=1.0,
        help="曲线计算的距离步长(m)。默认: 1.0",
    )
    _ = parser.add_argument(
        "--mass-tonnes",
        type=float,
        default=317.5,
        help="车辆质量(t)。默认: 317.5",
    )
    _ = parser.add_argument(
        "--trainset-count",
        type=int,
        default=5,
        help="列车编组数量。默认: 5",
    )
    _ = parser.add_argument(
        "--max-acceleration-mps2",
        type=float,
        default=1.0,
        help="最大加速度(m/s^2)。默认: 1.0",
    )
    _ = parser.add_argument(
        "--max-deceleration-mps2",
        type=float,
        default=1.0,
        help="最大减速度的正值大小(m/s^2)。默认: 1.0",
    )
    _ = parser.add_argument(
        "--position-error-m",
        type=float,
        default=1.0,
        help="里程测量误差的非负大小(m)。默认: 1.0",
    )
    _ = parser.add_argument(
        "--speed-error-mps",
        type=float,
        default=0.1,
        help="速度测量误差的非负大小(m/s)。默认: 0.1",
    )
    _ = parser.add_argument(
        "--traction-cutoff-delay-s",
        type=float,
        default=0.5,
        help="牵引切断命令执行延时(s)。默认: 0.5",
    )
    _ = parser.add_argument(
        "--vortex-brake-delay-s",
        type=float,
        default=0.5,
        help="牵引切断后涡流制动启动延时(s)。默认: 0.5",
    )
    _ = parser.add_argument(
        "--min-curve-position-offset-m",
        type=float,
        default=0.0,
        help="最小速度曲线相对目标点的位置偏移(m)。默认: 0.0",
    )
    _ = parser.add_argument(
        "--include-acceleration-zone-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否将上行加速区终点作为额外危险点。默认启用。",
    )
    _ = parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖输出目录中已有的防护曲线产物。",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅校验并展示有效配置和计划产物，不执行计算或写文件。",
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    positive_values = {
        "--distance-step-m": args.distance_step_m,
        "--mass-tonnes": args.mass_tonnes,
        "--max-acceleration-mps2": args.max_acceleration_mps2,
        "--max-deceleration-mps2": args.max_deceleration_mps2,
    }
    for option, value in positive_values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{option} must be a finite value > 0")

    if args.trainset_count < 1:
        raise ValueError("--trainset-count must be >= 1")

    nonnegative_values = {
        "--position-error-m": args.position_error_m,
        "--speed-error-mps": args.speed_error_mps,
        "--traction-cutoff-delay-s": args.traction_cutoff_delay_s,
        "--vortex-brake-delay-s": args.vortex_brake_delay_s,
    }
    for option, value in nonnegative_values.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{option} must be a finite value >= 0")

    if not np.isfinite(args.min_curve_position_offset_m):
        raise ValueError("--min-curve-position-offset-m must be finite")


def _config_from_args(args: argparse.Namespace) -> SafeguardCurveConfig:
    return SafeguardCurveConfig(
        distance_step_m=args.distance_step_m,
        mass_tonnes=args.mass_tonnes,
        trainset_count=args.trainset_count,
        max_acceleration_mps2=args.max_acceleration_mps2,
        max_deceleration_mps2=args.max_deceleration_mps2,
        position_error_m=args.position_error_m,
        speed_error_mps=args.speed_error_mps,
        traction_cutoff_delay_s=args.traction_cutoff_delay_s,
        vortex_brake_delay_s=args.vortex_brake_delay_s,
        min_curve_position_offset_m=args.min_curve_position_offset_m,
        include_acceleration_zone_end=args.include_acceleration_zone_end,
    )


def build_calculation_inputs(config: SafeguardCurveConfig) -> CalculationInputs:
    slopes, slope_intervals = load_slopes()
    speed_limits, speed_limit_intervals = load_speed_limits(to_mps=True)
    accessible_points_raw, dangerous_points_raw = (
        load_auxiliary_stopping_areas_ap_and_dp()
    )
    accessible_points = np.asarray(accessible_points_raw, dtype=np.float64)
    dangerous_points = np.asarray(dangerous_points_raw, dtype=np.float64)

    if config.include_acceleration_zone_end:
        acceleration_zone_end = float(load_acceleration_zones()["uplink"]["end"])
        dangerous_points = np.insert(dangerous_points, 0, acceleration_zone_end)

    track = TrackInfo(
        slopes=slopes,
        slope_intervals=slope_intervals,
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        ASA_aps=accessible_points.tolist(),
        ASA_dps=dangerous_points.tolist(),
    )
    vehicle = VehicleInfo(
        mass=config.mass_tonnes,
        numoftrainsets=config.trainset_count,
        length=DEFAULT_VEHICLE_LENGTH_M,
        max_acc=config.max_acceleration_mps2,
        max_dec=-config.max_deceleration_mps2,
    )
    return CalculationInputs(
        calculator=SafeGuardCurves(track=track),
        vehicle=vehicle,
        accessible_points=accessible_points,
        dangerous_points=dangerous_points,
    )


def calculate_curves(
    config: SafeguardCurveConfig,
    inputs: CalculationInputs,
) -> dict[str, list[NDArray[np.float64]]]:
    levi_curves, min_curves = inputs.calculator.calc_levi_and_min_curves(
        apoffsets=inputs.accessible_points,
        vehicle=inputs.vehicle,
        ds=config.distance_step_m,
        pos_error=config.position_error_m,
        speed_error=config.speed_error_mps,
        pos_offset=config.min_curve_position_offset_m,
        delay_time_until_DPS_done=config.traction_cutoff_delay_s,
    )
    brake_curves, max_curves = inputs.calculator.calc_brake_and_max_curves(
        dpoffsets=inputs.dangerous_points,
        vehicle=inputs.vehicle,
        ds=config.distance_step_m,
        pos_error=-config.position_error_m,
        speed_error=-config.speed_error_mps,
        delay_time_until_DPS_done=config.traction_cutoff_delay_s,
        delay_time_until_VB_begin=config.vortex_brake_delay_s,
    )
    return {
        "levi_curves_list": levi_curves,
        "brake_curves_list": brake_curves,
        "min_curves_list": min_curves,
        "max_curves_list": max_curves,
    }


def _managed_paths(output_dir: Path) -> tuple[Path, ...]:
    filenames = (*CURVE_FILENAMES.values(), METADATA_FILENAME)
    return tuple(output_dir / filename for filename in filenames)


def _existing_managed_paths(output_dir: Path) -> list[Path]:
    return [path for path in _managed_paths(output_dir) if path.exists()]


def _build_metadata(
    *,
    config: SafeguardCurveConfig,
    inputs: CalculationInputs,
    curves: Mapping[str, Sequence[NDArray[np.float64]]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator": "scripts.calc_and_save_safeguard_curves",
        "parameters": asdict(config),
        "vehicle": {"length_m": DEFAULT_VEHICLE_LENGTH_M},
        "scenario": {
            "accessible_points_m": inputs.accessible_points.tolist(),
            "dangerous_points_m": inputs.dangerous_points.tolist(),
        },
        "artifacts": {
            name: {
                "filename": CURVE_FILENAMES[name],
                "curve_count": len(curves[name]),
            }
            for name in CURVE_FILENAMES
        },
    }


def _write_pickle_temporary(value: object, target: Path) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            pickle.dump(value, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_json_temporary(value: Mapping[str, Any], target: Path) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def save_curves(
    *,
    curves: Mapping[str, Sequence[NDArray[np.float64]]],
    metadata: Mapping[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    replacements: list[tuple[Path, Path]] = []
    try:
        for name, filename in CURVE_FILENAMES.items():
            target = output_dir / filename
            temporary = _write_pickle_temporary(curves[name], target)
            temporary_paths.append(temporary)
            replacements.append((temporary, target))

        metadata_target = output_dir / METADATA_FILENAME
        metadata_temporary = _write_json_temporary(metadata, metadata_target)
        temporary_paths.append(metadata_temporary)
        replacements.append((metadata_temporary, metadata_target))

        for temporary, target in replacements:
            os.replace(temporary, target)
            temporary_paths.remove(temporary)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def _print_run_summary(
    *,
    config: SafeguardCurveConfig,
    inputs: CalculationInputs,
    output_dir: Path,
    existing_paths: Sequence[Path],
) -> None:
    summary = {
        "output_dir": str(output_dir),
        "parameters": asdict(config),
        "accessible_point_count": int(inputs.accessible_points.size),
        "dangerous_point_count": int(inputs.dangerous_points.size),
        "planned_files": [path.name for path in _managed_paths(output_dir)],
        "existing_files": [path.name for path in existing_paths],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    try:
        _validate_cli_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = resolve_project_path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output-dir is not a directory: {output_dir}")

    config = _config_from_args(args)
    inputs = build_calculation_inputs(config)
    existing_paths = _existing_managed_paths(output_dir)
    _print_run_summary(
        config=config,
        inputs=inputs,
        output_dir=output_dir,
        existing_paths=existing_paths,
    )

    if args.dry_run:
        if existing_paths and not args.force:
            print("提示: 正式运行时需要指定 --force 才能覆盖已有产物。")
        return 0

    if existing_paths and not args.force:
        filenames = ", ".join(path.name for path in existing_paths)
        parser.error(f"output artifacts already exist; use --force: {filenames}")

    try:
        curves = calculate_curves(config, inputs)
        metadata = _build_metadata(config=config, inputs=inputs, curves=curves)
        save_curves(curves=curves, metadata=metadata, output_dir=output_dir)
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，已停止防护曲线计算。")
        return 130

    print(f"防护曲线已保存到: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
