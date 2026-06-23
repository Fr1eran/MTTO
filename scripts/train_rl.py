import argparse

from rl.experiment_utils import (
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_REWARD_PROFILE_NAME,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    TrainingRunSpec,
    resolve_training_run_spec,
    reward_profile_names,
    train_single_experiment,
)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用可配置的日志记录和分析开关训练 MTTO PPO 策略",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output/optimal/rl/",
        help="训练结果输出根目录。",
    )
    parser.add_argument(
        "--schedule-time-s",
        type=float,
        default=430.0,
        help="规划运行时间(s)",
    )
    parser.add_argument(
        "--max-step-distance",
        type=float,
        default=100.0,
        help="训练环境相邻状态转移间的最大移动距离。",
    )
    parser.add_argument(
        "--reward-profile",
        type=str,
        choices=tuple(reward_profile_names()),
        default=DEFAULT_REWARD_PROFILE_NAME,
        help=(
            "奖励配置预设。basic 固定包含 energy/comfort；"
            "其余预设仅沿 safety/stopping/punctuality 三个 shaping 维度逐级打开。"
        ),
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default=None,
        help="附加实验标签，用于隔离输出目录与日志命名。",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["tune", "reproduce", "monitor_best", "best_only"],
        default="tune",
        help=(
            "算法运行模式。"
            "tune=全量调参与分析；"
            "reproduce=高效复现；"
            "monitor_best=基础监控+best-eval；"
            "best_only=仅保留best-eval。"
        ),
    )
    parser.add_argument(
        "--enable-tb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 Tensorboard 日志记录。",
    )
    parser.add_argument(
        "--enable-callback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 Tensorboard 回调。",
    )
    parser.add_argument(
        "--enable-monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 VecMonitor 包装器。",
    )
    parser.add_argument(
        "--enable-env-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用训练过程诊断信息收集功能。",
    )
    parser.add_argument(
        "--enable-auto-analysis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用训练后自动分析。",
    )
    parser.add_argument(
        "--enable-best-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用最佳轨迹评估。",
    )
    parser.add_argument(
        "--analysis-output-root",
        type=str,
        default="mtto_train_reports",
        help="训练后分析结果的输出目录。仅在启用日志记录功能时生效。",
    )
    parser.add_argument(
        "--analysis-min-points-per-10k-steps",
        type=float,
        default=5.0,
        help="每1万步最低可接受平均样本数。仅在启用日志记录功能时生效。",
    )
    parser.add_argument(
        "--analysis-min-unique-episodes",
        type=int,
        default=100,
        help="最低可接受唯一回合数。仅在启用日志记录功能时生效。",
    )
    parser.add_argument(
        "--analysis-sampling-quality-mode",
        type=str,
        choices=["warn_only", "strict_fail"],
        default="warn_only",
        help="自动分析的采样质量门控模式。仅在启用日志记录功能时生效。",
    )
    parser.add_argument(
        "--reward-discount",
        type=float,
        default=DEFAULT_REWARD_DISCOUNT,
        help="回报折扣因子。",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="训练环境数量。",
    )
    parser.add_argument(
        "--vec-env-type",
        type=str,
        choices=["dummy", "subproc"],
        default="subproc",
        help="向量化环境后端。subproc 在 num_envs > 1 时启用并行采样。",
    )
    parser.add_argument(
        "--rollout-steps-per-update",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        help="PPO rollout 步数。",
    )
    parser.add_argument(
        "--n-steps-per-env",
        type=int,
        default=None,
        help="PPO n_steps 步数。如果未指定，\
             则根据 rollout-steps-per-update 和 num-envs 计算得出。",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="PPO 总训练步数。",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="mtto_ppo_tb_logs",
        help="TensorBoard 日志输出根目录。",
    )
    parser.add_argument(
        "--tb-log-name",
        type=str,
        default=None,
        help="TensorBoard 运行名称；未指定时会根据运行模式和实验标识自动生成。",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="PPO log_interval。仅在启用日志记录功能时生效。`tune`模式下默认为1。",
    )
    parser.add_argument(
        "--tb-sample-interval-steps",
        type=int,
        default=1,
        help="Tensorboard 回调记录数据的最小间隔步数。",
    )
    parser.add_argument(
        "--env-diagnostics-interval-steps",
        type=int,
        default=None,
        help="环境诊断信息的记录间隔。默认与 tb-sample-interval-steps 一致。",
    )
    parser.add_argument(
        "--force-dump-interval-steps",
        type=int,
        default=0,
        help="(legacy)Tensorboard 回调中强制刷新数据缓存的间隔步数。",
    )
    parser.add_argument(
        "--tb-batch-dump-records",
        type=int,
        default=0,
        help="Tensorboard 事件缓冲区的记录上限。\
             达到上限后，将刷写文件；如果设置为0，则会在训练结束后一次刷写全部内容。",
    )
    parser.add_argument(
        "--best-eval-trigger-mode",
        type=str,
        choices=["steps", "episodes"],
        default="steps",
        help="最优评估回调的触发模式。",
    )
    parser.add_argument(
        "--best-eval-trigger-interval",
        type=int,
        default=100_000,
        help="根据 best-eval-trigger-mode 的设置，\
             以步数或回合数为单位的最佳评估触发间隔。",
    )
    parser.add_argument(
        "--best-eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在运行最佳评估回放时，使用确定性策略。",
    )
    parser.add_argument(
        "--enable-safety-curve",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在周期评估中记录 Dangerous State Rate 曲线。",
    )
    parser.add_argument(
        "--safety-eval-margin-threshold-mps",
        type=float,
        default=5.0,
        help="危险状态率评估使用的速度安全裕度阈值(m/s)。",
    )
    parser.add_argument(
        "--rollout-record-trigger-mode",
        type=str,
        choices=["steps", "episodes"],
        default="steps",
        help=(
            "EpisodeMetricsCollector 记录触发模式。"
            "steps=按训练步数采样滑动窗口均值；"
            "episodes=每个回合终止时记录该回合的原始 reward 与长度。"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="指定运行 PPO 算法的硬件设备，例如 'cpu' 或 'cuda'。",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅解析训练配置、输出路径与运行预设，不创建环境或启动训练。",
    )
    return parser


def print_training_run_spec(spec: TrainingRunSpec) -> None:
    run_metadata_path = spec.run_metadata_path

    print("Training runtime switches:")
    print(f"- run_mode={spec.run_mode}")
    print(f"- reward_profile={spec.reward_profile.name}")
    print(f"- reward_config={spec.run_metadata['reward_config']}")
    print(f"- enable_tb={spec.enable_tb}")
    print(f"- enable_callback={spec.enable_callback}")
    print(f"- enable_monitor={spec.enable_monitor}")
    print(f"- enable_env_diagnostics={spec.enable_env_diagnostics}")
    print(f"- enable_auto_analysis={spec.enable_auto_analysis}")
    print(f"- enable_best_eval={spec.enable_best_eval}")
    print(f"- reward_discount={spec.reward_discount}")
    print(f"- schedule_time_s={spec.schedule_time_s}")
    print(f"- max_step_distance={spec.max_step_distance}")
    print(f"- num_envs={spec.num_envs}")
    print(f"- vec_env_type={spec.resolved_vec_env_type}")
    if spec.use_subproc:
        print(f"- subproc_start_method={spec.subproc_start_method}")
    else:
        print("- subproc_start_method=(not applicable)")
    print(f"- n_steps_per_env={spec.n_steps_per_env}")
    print(f"- rollout_steps_per_update={spec.rollout_steps_per_update}")
    print(f"- output_dir={spec.output_dir}")
    print(f"- total_timesteps={spec.total_timesteps}")
    print(f"- tensorboard_log_dir={spec.tensorboard_log_dir}")
    print(f"- run_metadata_path={run_metadata_path}")
    if spec.enable_tb:
        print(f"- tb_log_name={spec.tb_log_name}")
        print(f"- log_interval={spec.log_interval}")
    else:
        print(f"- tb_log_name=ignored (resolved name would be {spec.tb_log_name})")
        print("- log_interval=ignored (logging disabled by current switches)")
    print(f"- tb_sample_interval_steps={spec.tb_sample_interval_steps}")
    print(f"- env_diagnostics_interval_steps={spec.env_diagnostics_interval_steps}")
    print(f"- force_dump_interval_steps={spec.force_dump_interval_steps}")
    print(f"- tb_batch_dump_records={spec.tb_batch_dump_records}")
    print(f"- best_eval_trigger_mode={spec.best_eval_trigger_mode}")
    print(f"- best_eval_trigger_interval={spec.best_eval_trigger_interval}")
    print(f"- best_eval_output_dir={spec.best_eval_output_dir}")
    print(f"- best_eval_deterministic={spec.best_eval_deterministic}")
    print(f"- enable_safety_curve={spec.enable_safety_curve}")
    print(f"- safety_eval_margin_threshold_mps={spec.safety_eval_margin_threshold_mps}")
    print(f"- rollout_record_trigger_mode={spec.rollout_record_trigger_mode}")
    print(f"- device={spec.device}")
    print(f"- seed={spec.seed}")
    print(f"- dry_run={spec.dry_run}")
    if spec.enable_auto_analysis and not spec.enable_tb:
        print(
            "- warning: enable_auto_analysis=True while enable_tb=False; "
            f"analysis will use existing logs in \
            {spec.tensorboard_log_dir} if available."
        )
    if spec.enable_auto_analysis and not spec.enable_callback:
        print(
            "- warning: enable_auto_analysis=True while enable_callback=False; "
            "analysis may miss high-frequency state/constraint diagnostics."
        )


def main() -> None:
    args = build_cli_parser().parse_args()
    spec = resolve_training_run_spec(args)
    print_training_run_spec(spec)

    if spec.dry_run:
        print(
            "Dry run completed: training configuration resolved; \
             skipped environment and model initialization."
        )
        return

    train_single_experiment(args, spec=spec)


if __name__ == "__main__":
    main()
