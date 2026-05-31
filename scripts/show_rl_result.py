import argparse

import matplotlib.pyplot as plt

from rl.experiment_utils import (
    RL_DEFAULT_SEARCH_DIR as _RL_DEFAULT_SEARCH_DIR,
    RL_TRAJECTORY_SOURCE_CHOICES as _RL_TRAJECTORY_SOURCE_CHOICES,
    apply_rl_curve_plot_style,
    get_rl_trajectory_status_text,
    load_rl_curve_artifact,
    render_rl_curve_on_axes,
    resolve_rl_curve_artifact,
)


def _print_metrics(metrics: dict[str, object]) -> None:
    if not metrics:
        print("No metrics file found.")
        return

    print("Loaded metrics:")
    for key in [
        "trajectory_source",
        "reward_profile_name",
        "total_reward",
        "target_time_s",
        "total_time_s",
        "time_error_s",
        "start_position_m",
        "target_position_m",
        "final_position_m",
        "stop_error_m",
        "total_energy_kj",
        "total_energy_j",
        "final_speed_mps",
        "comfort_tav",
        "comfort_er_pct",
        "comfort_rms",
        "episode_steps",
        "success",
        "selection_rule",
        "strict_stop_error_limit_m",
        "strict_time_error_limit_s",
        "strict_stop_requirement_met",
        "strict_time_requirement_met",
        "selection_comparison_key",
        "best_update_reason",
        "num_timesteps",
        "eval_trigger_mode",
        "eval_trigger_interval",
        "created_at",
    ]:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="加载并显示已保存的强化学习轨迹结果。")
    parser.add_argument(
        "--curve-dir",
        default=_RL_DEFAULT_SEARCH_DIR,
        help="用于搜索强化学习轨迹相关产物的路径",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=_RL_TRAJECTORY_SOURCE_CHOICES,
        default="best",
        help="选择加载哪条保存的轨迹: 'best' 'best_steps' 'best_episodes' 'final'",
    )
    parser.add_argument(
        "--no-safeguard",
        action="store_true",
        help="不绘制安全防护曲线",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.99,
        help="在启用安全防护时用于渲染的速度上限因数。",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅解析轨迹产物路径，不加载曲线数据或显示图窗。",
    )

    return parser


def plot_rl_curve(
    *,
    pos_arr,
    speed_arr,
    metrics: dict[str, object],
    no_safeguard: bool,
    factor: float,
) -> None:
    apply_rl_curve_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    render_rl_curve_on_axes(
        ax=ax,
        pos_arr=pos_arr,
        speed_arr=speed_arr,
        metrics=metrics,
        no_safeguard=no_safeguard,
        factor=factor,
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    try:
        artifact = resolve_rl_curve_artifact(
            curve_dir=args.curve_dir,
            trajectory_source=args.trajectory_source,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    print(f"Using curve file: {artifact.npz_path}")
    print(f"Using metrics file: {artifact.metrics_path}")

    if args.dry_run:
        print(
            "Dry run completed: trajectory artifact resolved; \
             skipped loading metrics and plotting."
        )
        return

    pos_arr, speed_arr, metrics = load_rl_curve_artifact(artifact)

    _print_metrics(metrics)
    status_text = get_rl_trajectory_status_text(metrics)
    if status_text is not None:
        print(status_text)

    plot_rl_curve(
        pos_arr=pos_arr,
        speed_arr=speed_arr,
        metrics=metrics,
        no_safeguard=args.no_safeguard,
        factor=args.factor,
    )


if __name__ == "__main__":
    main()
