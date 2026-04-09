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
    print(f"Summary CSV: {output_paths.get('summary_metrics_csv', 'N/A')}")


if __name__ == "__main__":
    main()
