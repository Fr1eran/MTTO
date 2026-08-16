import argparse

from rl.experiment_utils import (
    DEFAULT_CURRICULUM_PROFILE_NAME,
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
    DEFAULT_NUM_ENVS,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_REWARD_PRESET_NAME,
    DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
    DEFAULT_SCHEDULE_TIME_S,
    DEFAULT_STEP_DISTANCE,
    DEFAULT_VEC_ENV_TYPE,
    VEC_ENV_TYPE_CHOICES,
    TrainingRunSpec,
    curriculum_profile_names,
    resolve_training_run_spec,
    reward_preset_names,
    train_single_experiment,
)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用可配置的日志记录和分析开关训练 MTTO PPO 策略",
    )
    _ = parser.add_argument(
        "--output-root",
        type=str,
        default="output/optimal/rl/",
        help="训练结果输出根目录。",
    )
    _ = parser.add_argument(
        "--schedule-time-s",
        type=float,
        default=DEFAULT_SCHEDULE_TIME_S,
        help="规划运行时间(s)",
    )
    _ = parser.add_argument(
        "--step-distance",
        "--max-step-distance",
        dest="step_distance",
        type=float,
        default=DEFAULT_STEP_DISTANCE,
        help="训练环境固定空间控制步长；--max-step-distance 为兼容别名。",
    )
    _ = parser.add_argument(
        "--reward-preset",
        type=str,
        choices=tuple(reward_preset_names()),
        default=DEFAULT_REWARD_PRESET_NAME,
        help=(
            "奖励配置预设。basic 固定包含 energy/comfort；"
            "basic_safety 额外启用安全 PBRS。"
        ),
    )
    _ = parser.add_argument(
        "--curriculum-profile",
        type=str,
        choices=tuple(curriculum_profile_names()),
        default=DEFAULT_CURRICULUM_PROFILE_NAME,
        help="初态课程预设。none 保持真实起点训练；dspdl 启用离散自步课程。",
    )
    _ = parser.add_argument(
        "--reference-curve-dir",
        type=str,
        default=None,
        help="启用课程时必填：包含任务匹配 DP 参考轨迹的目录。",
    )
    _ = parser.add_argument(
        "--experiment-tag",
        type=str,
        default=None,
        help="附加实验标签，用于隔离输出目录与日志命名。",
    )
    _ = parser.add_argument(
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
    _ = parser.add_argument(
        "--enable-tb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 Tensorboard 日志记录。",
    )
    _ = parser.add_argument(
        "--enable-monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 VecMonitor 包装器。",
    )
    _ = parser.add_argument(
        "--enable-auto-analysis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用训练后自动分析。",
    )
    _ = parser.add_argument(
        "--enable-best-evaluation-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用最佳轨迹评估。",
    )
    _ = parser.add_argument(
        "--analysis-output-root",
        type=str,
        default="mtto_train_reports",
        help="训练后分析结果的输出目录。仅在启用日志记录功能时生效。",
    )
    _ = parser.add_argument(
        "--analysis-min-points-per-10k-steps",
        type=float,
        default=5.0,
        help="每1万步最低可接受平均样本数。仅在启用日志记录功能时生效。",
    )
    _ = parser.add_argument(
        "--analysis-sampling-quality-mode",
        type=str,
        choices=["warn_only", "strict_fail"],
        default="warn_only",
        help="自动分析的采样质量门控模式。仅在启用日志记录功能时生效。",
    )
    _ = parser.add_argument(
        "--reward-discount",
        type=float,
        default=DEFAULT_REWARD_DISCOUNT,
        help="回报折扣因子。",
    )
    _ = parser.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help="训练环境数量。",
    )
    _ = parser.add_argument(
        "--vec-env-type",
        type=str,
        choices=VEC_ENV_TYPE_CHOICES,
        default=DEFAULT_VEC_ENV_TYPE,
        help="向量化环境后端；默认使用低开销的 DummyVecEnv。",
    )
    _ = parser.add_argument(
        "--rollout-steps-per-update",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS_PER_UPDATE,
        help="PPO rollout 步数。",
    )
    _ = parser.add_argument(
        "--n-steps-per-env",
        type=int,
        default=None,
        help="PPO n_steps 步数。如果未指定，\
             则根据 rollout-steps-per-update 和 num-envs 计算得出。",
    )
    _ = parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="PPO 总训练步数。",
    )
    _ = parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="mtto_ppo_tb_logs",
        help="TensorBoard 日志输出根目录。",
    )
    _ = parser.add_argument(
        "--tb-log-name",
        type=str,
        default=None,
        help="TensorBoard 运行名称；未指定时会根据运行模式和实验标识自动生成。",
    )
    _ = parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="PPO log_interval。仅在启用日志记录功能时生效。`tune`模式下默认为1。",
    )
    _ = parser.add_argument(
        "--evaluation-interval-rollouts",
        type=int,
        default=DEFAULT_EVALUATION_INTERVAL_ROLLOUTS,
        help="两次策略评估之间完成的 PPO rollout 数。",
    )
    _ = parser.add_argument(
        "--evaluation-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在运行最佳评估回放时，使用确定性策略。",
    )
    _ = parser.add_argument(
        "--evaluation-history-path",
        type=str,
        default=None,
        help="可选的策略评估历史 NPZ 输出路径。",
    )
    _ = parser.add_argument(
        "--enable-safety-truncation-histogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="记录 worker 汇总的安全截断位置直方图；tune 模式默认启用。",
    )
    _ = parser.add_argument(
        "--safety-truncation-bin-size-m",
        type=float,
        default=5000.0,
        help="安全截断位置直方图使用的位置分桶长度(m)。",
    )
    _ = parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    _ = parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="指定运行 PPO 算法的硬件设备，例如 'cpu' 或 'cuda'。",
    )
    _ = parser.add_argument(
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
    print(f"- reward_preset={spec.reward_preset.name}")
    print(f"- reward_config={spec.run_metadata['reward_config']}")
    print(f"- curriculum_profile={spec.curriculum_profile}")
    if spec.curriculum_profile == "dspdl":
        print(f"- reference_curve_dir={spec.reference_curve_dir}")
    print(f"- enable_tb={spec.enable_tb}")
    print(f"- enable_monitor={spec.enable_monitor}")
    print(f"- enable_auto_analysis={spec.enable_auto_analysis}")
    print(
        "- enable_best_evaluation_artifacts="
        + f"{spec.enable_best_evaluation_artifacts}"
    )
    print(f"- reward_discount={spec.reward_discount}")
    print(f"- schedule_time_s={spec.schedule_time_s}")
    print(f"- step_distance={spec.step_distance}")
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
    print(f"- reward_diagnostics_path={spec.reward_diagnostics_path}")
    print(
        "- evaluation_interval_rollouts="
        + f"{spec.evaluation_interval_rollouts}"
    )
    print(f"- best_eval_output_dir={spec.best_eval_output_dir}")
    print(f"- evaluation_deterministic={spec.evaluation_deterministic}")
    print(f"- evaluation_history_path={spec.evaluation_history_path}")
    print(
        "- enable_safety_truncation_histogram="
        + f"{spec.enable_safety_truncation_histogram}"
    )
    print(f"- safety_truncation_bin_size_m={spec.safety_truncation_bin_size_m}")
    print(f"- device={spec.device}")
    print(f"- seed={spec.seed}")
    print(f"- dry_run={spec.dry_run}")


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

    _ = train_single_experiment(args, spec=spec)


if __name__ == "__main__":
    main()
