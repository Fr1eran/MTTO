from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from dp.core import (
    ParallelPrecomputeExitedError,
    VariableSpacingDPOptimizer,
)
from utils.io_utils import format_float_token, save_curve_and_metrics
from utils.scenario import build_scenario
from utils.trajectory import compute_comfort_metrics_from_trajectory


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="复现变间距动态规划速度曲线，并支持状态转移图并行预计算。"
    )
    _ = parser.add_argument(
        "--output-root",
        type=str,
        default="output/optimal/dp",
        help="优化结果输出根目录。",
    )
    _ = parser.add_argument(
        "--schedule-time-s",
        type=float,
        default=430.0,
        help="规划运行时间(s)。",
    )
    _ = parser.add_argument(
        "--delta-speed-mps",
        type=float,
        default=0.1,
        help="速度搜索步长(m/s)。",
    )
    _ = parser.add_argument(
        "--max-outer-iterations",
        type=int,
        default=100,
        help="外层二分搜索最大迭代次数。",
    )
    _ = parser.add_argument(
        "--precompute-mode",
        choices=("serial", "parallel"),
        default="parallel",
        help="状态转移图预计算模式。serial 为单进程，parallel 为多进程。",
    )
    _ = parser.add_argument(
        "--precompute-workers",
        type=int,
        default=4,
        help="并行模式下的进程数。默认值为4",
    )
    _ = parser.add_argument(
        "--precompute-chunk-size",
        type=int,
        default=None,
        help="并行模式下每个任务块包含的阶段数。未指定时自动估计。",
    )
    _ = parser.add_argument(
        "--mp-start-method",
        choices=("spawn", "fork", "forkserver"),
        default=None,
        help="多进程启动方式。Windows 默认使用 spawn。",
    )
    _ = parser.add_argument(
        "--hide-precompute-progress",
        action="store_true",
        help="关闭状态转移图预计算进度显示。",
    )
    _ = parser.add_argument(
        "--stage-division",
        choices=("variable", "uniform"),
        default="uniform",
        help="阶段划分方式。variable 为基于临界点的变间距划分，uniform 为等间距划分。",
    )
    _ = parser.add_argument(
        "--uniform-step-size",
        type=float,
        default=30.0,
        help="等间距划分时的阶段步长(m)，仅 --stage-division uniform 时生效。",
    )
    _ = parser.add_argument(
        "--sub-stage-count",
        type=int,
        default=30,
        help="变间距划分时每个临界区间的子阶段数量。",
    )
    _ = parser.add_argument(
        "--skip-disk-cache",
        action="store_true",
        help="跳过磁盘缓存，每次强制重新计算状态转移图。",
    )
    return parser


def _resolve_output_dir(
    *,
    output_root: str,
    schedule_time_s: float,
    delta_speed_mps: float,
    stage_division: str,
    sub_stage_count: int,
    uniform_step_size: float,
) -> str:
    schedule_token = format_float_token(schedule_time_s)
    delta_token = format_float_token(delta_speed_mps)
    if stage_division == "uniform":
        div_token = f"uni{format_float_token(uniform_step_size)}"
    else:
        div_token = f"var{sub_stage_count}"
    return os.path.join(output_root, f"{schedule_token}_{delta_token}_{div_token}")


def _validate_cli_args(cli_args: argparse.Namespace) -> None:
    if not cli_args.output_root.strip():
        raise ValueError("--output-root must not be empty")
    if cli_args.schedule_time_s <= 0.0:
        raise ValueError("--schedule-time-s must be > 0")
    if cli_args.delta_speed_mps <= 0.0:
        raise ValueError("--delta-speed-mps must be > 0")
    if cli_args.max_outer_iterations < 1:
        raise ValueError("--max-outer-iterations must be >= 1")
    if cli_args.uniform_step_size <= 0.0:
        raise ValueError("--uniform-step-size must be > 0")
    if cli_args.sub_stage_count < 1:
        raise ValueError("--sub-stage-count must be >= 1")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    cli_args = parser.parse_args(argv)

    try:
        _validate_cli_args(cli_args)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = _resolve_output_dir(
        output_root=cli_args.output_root,
        schedule_time_s=cli_args.schedule_time_s,
        delta_speed_mps=cli_args.delta_speed_mps,
        stage_division=cli_args.stage_division,
        sub_stage_count=cli_args.sub_stage_count,
        uniform_step_size=cli_args.uniform_step_size,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"优化结果输出目录: {output_dir}")
    return _run_optimization(cli_args=cli_args, output_dir=output_dir)


def _run_optimization(*, cli_args: argparse.Namespace, output_dir: str) -> int:
    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=cli_args.schedule_time_s
    )

    VSDP = VariableSpacingDPOptimizer(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        show_precompute_progress=not cli_args.hide_precompute_progress,
        precompute_progress_desc="状态转移图预计算",
        precompute_mode=cli_args.precompute_mode,
        precompute_workers=cli_args.precompute_workers,
        precompute_chunk_size=cli_args.precompute_chunk_size,
        mp_start_method=cli_args.mp_start_method,
        stage_division=cli_args.stage_division,
        uniform_step_size=cli_args.uniform_step_size,
        sub_stage_count=cli_args.sub_stage_count,
        skip_disk_cache=cli_args.skip_disk_cache,
    )

    try:
        result = VSDP.optimize(
            max_speed=vehicle.max_speed,
            delta_speed=cli_args.delta_speed_mps,
            max_iters=cli_args.max_outer_iterations,
        )
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，已停止预计算/优化流程。")
        return 130
    except ParallelPrecomputeExitedError as exc:
        print(f"并行预计算流程已退出，程序结束。原因: {exc}")
        return 1

    if result is not None:
        comfort_metrics = compute_comfort_metrics_from_trajectory(
            pos_arr=result["pos"],
            speed_arr=result["speed"],
            max_acc_change=train_service.max_acc_change,
        )

        output_file = os.path.join(output_dir, "optimized_speed_curve.npz")
        saved_npz_path, saved_metrics_path = save_curve_and_metrics(
            pos_arr=result["pos"],
            speed_arr=result["speed"],
            output_path=output_file,
            extra_arrays={"cum_time_s": result["cum_time_s"]},
            metrics={
                "target_time_s": float(train_service.schedule_time),
                "total_time_s": float(result["total_time"]),
                "time_error_s": float(
                    train_service.schedule_time - result["total_time"]
                ),
                "total_energy_kj": float(result["total_energy"]),
                "start_position_m": float(train_service.start_position),
                "start_speed_mps": float(train_service.start_speed),
                "target_position_m": float(train_service.target_position),
                "target_speed_mps": 0.0,
                "max_step_distance_m": (
                    float(cli_args.uniform_step_size)
                    if cli_args.stage_division == "uniform"
                    else None
                ),
                "stage_division": cli_args.stage_division,
                **comfort_metrics,
            },
        )
        print(f"优化速度曲线已保存到: {saved_npz_path}")
        print(f"性能指标已保存到: {saved_metrics_path}")
        print(
            "舒适度指标: "
            + f"TAV={comfort_metrics['comfort_tav']:.4f} m/s^2, "
            + f"ER={comfort_metrics['comfort_er_pct']:.2f} %, "
            + f"RMS={comfort_metrics['comfort_rms']:.4f} m/s^2"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
