from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from mtto.domain.scenario import Task
from mtto.domain.speed_profile import SpeedProfile
from mtto.evaluation.quality import QualityReport
from mtto.io.artifacts import RunKind, read_completed_run, task_from_json
from paper.figures import load_paper_scenario
from paper.plotting.profiles import render_rl_curve_on_axes
from paper.plotting.style import (
    apply_sci_curve_style,
    apply_sci_figure_layout,
    save_sci_figure,
)

FIGURE_FILENAME = "rl_result.pdf"


def _print_metrics(
    *,
    profile: SpeedProfile,
    quality: QualityReport,
    task: Task,
    result: Any,
    record: Any,
    is_best: bool,
) -> None:
    source = (
        "best"
        if is_best
        else ("final" if record.kind == RunKind.RL_TRAIN else "evaluation")
    )
    display_metrics: dict[str, Any] = {
        "trajectory_source": source,
        "total_reward": result.total_reward if result is not None else None,
        "target_time_s": task.schedule_time_s,
        "total_time_s": quality.metrics.run_time_s,
        "time_error_s": quality.metrics.arrival_time_error_s,
        "start_position_m": task.start_position_m,
        "target_position_m": task.target_position_m,
        "final_position_m": float(profile.position_m[-1]),
        "stop_error_m": quality.metrics.stop_error_m,
        "total_energy_kj": quality.metrics.total_energy_kj,
        "total_energy_j": quality.metrics.total_energy_kj * 1000.0,
        "final_speed_mps": float(profile.speed_mps[-1]),
        "comfort_tav": quality.metrics.comfort_tav_mps2,
        "comfort_er_pct": quality.metrics.comfort_exceedance_pct,
        "comfort_rms": quality.metrics.comfort_rms_mps2,
        "episode_steps": result.steps if result is not None else None,
        "success": quality.completed,
        "precise_arrival": quality.precise_stop,
        "punctual_arrival": quality.punctual,
        "safety_violation_count": len(quality.audit.violations),
        "safe": quality.safe,
        "feasible": quality.feasible,
        "min_safety_margin_mps": quality.audit.min_margin_mps,
        "created_at": record.created_at,
    }
    print("Loaded metrics:")
    for key, val in display_metrics.items():
        if val is not None:
            print(f"  {key}: {val}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="加载并显示已保存的强化学习轨迹结果。")
    _ = parser.add_argument(
        "--rl-run",
        type=Path,
        required=True,
        help="包含 run.json 等新产物的 RL 训练或评估运行目录",
    )
    _ = parser.add_argument(
        "--rl-best",
        action="store_true",
        default=False,
        help="取训练运行中的最优策略产物 (best/)，仅对训练运行有效",
    )
    _ = parser.add_argument(
        "--no-safeguard",
        action="store_true",
        help="不绘制安全防护曲线",
    )
    _ = parser.add_argument(
        "--factor",
        type=float,
        default=0.99,
        help="在启用安全防护时用于渲染的速度上限因数。",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="仅打印指标，不绘制图窗。",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"可选图表保存目录；输出 {FIGURE_FILENAME}。",
    )
    _ = parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="不打开交互式显示图窗。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    try:
        completed_run = read_completed_run(args.rl_run)
    except Exception as exc:
        parser.error(f"Failed to read run from {args.rl_run}: {exc}")

    if completed_run.record.kind not in (RunKind.RL_TRAIN, RunKind.EVALUATION):
        parser.error(
            "Expected RL training or evaluation run, "
            f"got {completed_run.record.kind.value}"
        )

    if args.rl_best:
        if completed_run.record.kind != RunKind.RL_TRAIN:
            parser.error("--rl-best is only valid for RL training runs")
        if completed_run.payload.best is None:
            parser.error(
                f"--rl-best specified but {args.rl_run} has no best/ artifacts"
            )

    scenario = load_paper_scenario()
    if completed_run.record.scenario_hash != scenario.scenario_hash:
        parser.error(
            f"Scenario hash mismatch for run at {args.rl_run}: "
            f"expected {scenario.scenario_hash}, "
            f"got {completed_run.record.scenario_hash}"
        )

    task = task_from_json(completed_run.record.task)
    target_payload = (
        completed_run.payload.best if args.rl_best else completed_run.payload
    )
    profile = target_payload.profile
    quality = target_payload.quality
    result = target_payload.result

    _print_metrics(
        profile=profile,
        quality=quality,
        task=task,
        result=result,
        record=completed_run.record,
        is_best=args.rl_best,
    )

    prefix = (
        "RL 最优轨迹"
        if args.rl_best
        else (
            "RL 最终轨迹"
            if completed_run.record.kind == RunKind.RL_TRAIN
            else "RL 评估轨迹"
        )
    )
    status = f"{prefix}（完成任务）" if quality.completed else f"{prefix}（未完成任务）"
    print(status)

    if args.dry_run:
        return

    apply_sci_curve_style()
    fig, ax = plt.subplots()
    render_rl_curve_on_axes(
        ax=ax,
        profile=profile,
        task=task,
        is_best=args.rl_best,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
    )
    _ = ax.legend(loc="upper right")
    apply_sci_figure_layout(fig, columns=2, height_in=3.4)

    if args.output_dir is not None:
        saved_path = save_sci_figure(fig, args.output_dir / FIGURE_FILENAME)
        print(f"Saved figure to: {saved_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
