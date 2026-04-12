from __future__ import annotations

import argparse

from rl.training_analysis import AnalysisConfig, run_training_analysis


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze MTTO training logs and generate LLM-ready reports.",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default="mtto_ppo_tensorboard_logs",
        help="TensorBoard logs root directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name under log-root. If omitted, the latest run is used.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="mtto_train_reports",
        help="Output directory for analysis artifacts.",
    )
    parser.add_argument(
        "--step-window-size",
        type=int,
        default=5000,
        help="Step-based snapshot window size.",
    )
    parser.add_argument(
        "--episode-window-size",
        type=int,
        default=20,
        help="Episode-based snapshot window size.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.1,
        help="EMA alpha used in convergence analysis.",
    )
    parser.add_argument(
        "--kl-threshold",
        type=float,
        default=0.03,
        help="Approx KL safety threshold.",
    )
    parser.add_argument(
        "--near-miss-threshold-mps",
        type=float,
        default=1.0,
        help="Near-miss threshold for safety margin diagnostics.",
    )
    parser.add_argument(
        "--position-bin-size-m",
        type=float,
        default=500.0,
        help="Position bin size for geographic failure distribution.",
    )
    parser.add_argument(
        "--critical-point-radius-m",
        type=float,
        default=300.0,
        help="Radius for slope transition and SPS critical-point attribution.",
    )
    parser.add_argument(
        "--top-k-spatial-bins",
        type=int,
        default=8,
        help="Top-K spatial risk bins rendered in report.",
    )
    parser.add_argument(
        "--top-k-critical-points",
        type=int,
        default=8,
        help="Top-K critical points rendered in report.",
    )
    parser.add_argument(
        "--report-bar-width",
        type=int,
        default=24,
        help="ASCII bar width used in report charts.",
    )
    parser.add_argument(
        "--training-log-interval",
        type=int,
        default=None,
        help="Training log interval used for this run, stored into analysis metadata.",
    )
    parser.add_argument(
        "--min-points-per-10k-steps",
        type=float,
        default=5.0,
        help="Minimum acceptable mean samples per 10k steps for sampling quality.",
    )
    parser.add_argument(
        "--min-unique-episodes",
        type=int,
        default=100,
        help="Minimum acceptable unique episodes for sampling quality.",
    )
    parser.add_argument(
        "--max-mean-step-gap",
        type=float,
        default=2048.0,
        help="Maximum acceptable mean step gap for sampling quality.",
    )
    parser.add_argument(
        "--sampling-quality-mode",
        type=str,
        choices=["warn_only", "strict_fail"],
        default="warn_only",
        help="Sampling quality gate mode: warn_only logs warning, strict_fail aborts analysis.",
    )
    parser.add_argument(
        "--export-csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export CSV artifacts. Disabled by default.",
    )
    parser.add_argument(
        "--include-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include raw step/episode snapshots in analysis payload.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = AnalysisConfig(
        step_window_size=args.step_window_size,
        episode_window_size=args.episode_window_size,
        ema_alpha=args.ema_alpha,
        kl_threshold=args.kl_threshold,
        near_miss_threshold_mps=args.near_miss_threshold_mps,
        position_bin_size_m=args.position_bin_size_m,
        critical_point_radius_m=args.critical_point_radius_m,
        top_k_spatial_bins=args.top_k_spatial_bins,
        top_k_critical_points=args.top_k_critical_points,
        report_bar_width=args.report_bar_width,
        training_log_interval=args.training_log_interval,
        min_points_per_10k_steps=args.min_points_per_10k_steps,
        min_unique_episodes=args.min_unique_episodes,
        max_mean_step_gap=args.max_mean_step_gap,
        sampling_quality_mode=args.sampling_quality_mode,
        export_csv=args.export_csv,
        include_snapshots=args.include_snapshots,
        output_root=args.output_root,
    )

    result = run_training_analysis(
        log_root=args.log_root,
        run_name=args.run_name,
        config=config,
    )

    output_paths = result.get("output_paths", {})
    print("Training analysis completed.")
    print(f"JSON snapshot: {output_paths.get('json_snapshot', 'N/A')}")
    print(f"Markdown report: {output_paths.get('markdown_report', 'N/A')}")
    if "summary_metrics_csv" in output_paths:
        print(f"Summary CSV: {output_paths.get('summary_metrics_csv', 'N/A')}")
    else:
        print("CSV exports: disabled")


if __name__ == "__main__":
    main()
