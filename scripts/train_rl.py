import os
import argparse
from typing import Any, Callable
import numpy as np

from rl.callbacks import BestTrajectoryEvalCallback, TensorboardCallback
from rl.evaluation import build_single_eval_env
from rl.env_factory import make_env
from rl.training_analysis import AnalysisConfig, run_training_analysis
from model.vehicle import VehicleInfo
from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

_RL_MODEL_FILENAME = "ppo_mtto_model.zip"
_RL_VECNORMALIZE_FILENAME = "vecnormalize.pkl"


# 学习率线性衰减
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def ensure_parent_dir(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _build_env_initializer(
    *,
    vehicle: VehicleInfo,
    track: TrackInfo,
    safeguard_utility: SafeGuardUtility,
    train_service: TrainService,
    gamma: float,
    max_step_distance: float,
    enable_diagnostics: bool,
    diagnostics_interval_steps: int,
) -> Callable[[], Any]:
    def _init():
        return make_env(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            gamma=gamma,
            max_step_distance=max_step_distance,
            enable_diagnostics=enable_diagnostics,
            diagnostics_interval_steps=diagnostics_interval_steps,
        )

    return _init


def build_arg_parser() -> argparse.ArgumentParser:
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
        default=440.0,
        help="规划运行时间(s)",
    )
    parser.add_argument(
        "--max-step-distance",
        type=float,
        default=100.0,
        help="训练环境相邻状态转移间的最大移动距离。",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["tune", "reproduce", "eval"],
        default="tune",
        help="算法运行模式。'tune'启用日志分析与分析，'reproduce/eval'更注重效率。",
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
        "--analysis-max-mean-step-gap",
        type=float,
        default=2048.0,
        help="最大平均训练步间隔。仅在启用日志记录功能时生效。",
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
        default=0.99,
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
        "--subproc-start-method",
        type=str,
        choices=["spawn", "forkserver"],
        default="spawn",
        help="SubprocVecEnv 的多进程启动方法。",
    )
    parser.add_argument(
        "--rollout-steps-per-update",
        type=int,
        default=2048,
        help="PPO rollout 步数。",
    )
    parser.add_argument(
        "--n-steps-per-env",
        type=int,
        default=None,
        help="PPO n_steps 步数。如果未指定，则根据 rollout-steps-per-update 和 num-envs 计算得出。",
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
        default="mtto_ppo_tensorboard_logs",
        help="TensorBoard 日志输出根目录。",
    )
    parser.add_argument(
        "--tb-log-name",
        type=str,
        default="trainning_log",
        help="TensorBoard 日志文件名称。",
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
        help="Tensorboard 事件缓冲区的记录上限。达到上限后，将刷写文件；如果设置为0，则会在训练结束后一次刷写全部内容。",
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
        help="根据 best-eval-trigger-mode 的设置，以步数或回合数为单位的最佳评估触发间隔。",
    )
    parser.add_argument(
        "--best-eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在运行最佳评估回放时，使用确定性策略。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="指定运行 PPO 算法的硬件设备，例如 'cpu' 或 'cuda'。",
    )
    return parser


def resolve_run_mode(
    args: argparse.Namespace,
) -> tuple[str, bool, bool, bool, bool, bool, bool]:
    run_mode = args.run_mode
    defaults_by_mode = {
        "tune": {
            "tb": True,
            "callback": True,
            "monitor": True,
            "env_diagnostics": True,
            "analysis": True,
            "best_eval": True,
        },
        "reproduce": {
            "tb": False,
            "callback": False,
            "monitor": False,
            "env_diagnostics": False,
            "analysis": False,
            "best_eval": False,
        },
        "eval": {
            "tb": False,
            "callback": False,
            "monitor": False,
            "env_diagnostics": False,
            "analysis": False,
            "best_eval": False,
        },
    }
    mode_defaults = defaults_by_mode[run_mode]
    enable_tb = mode_defaults["tb"] if args.enable_tb is None else args.enable_tb
    enable_callback = (
        mode_defaults["callback"]
        if args.enable_callback is None
        else args.enable_callback
    )
    enable_monitor = (
        mode_defaults["monitor"] if args.enable_monitor is None else args.enable_monitor
    )
    enable_env_diagnostics = (
        mode_defaults["env_diagnostics"]
        if args.enable_env_diagnostics is None
        else args.enable_env_diagnostics
    )
    enable_auto_analysis = (
        mode_defaults["analysis"]
        if args.enable_auto_analysis is None
        else args.enable_auto_analysis
    )
    enable_best_eval = (
        mode_defaults["best_eval"]
        if args.enable_best_eval is None
        else args.enable_best_eval
    )

    if not enable_tb:
        enable_callback = False

    return (
        run_mode,
        enable_tb,
        enable_callback,
        enable_monitor,
        enable_env_diagnostics,
        enable_auto_analysis,
        enable_best_eval,
    )


def resolve_log_interval(
    args: argparse.Namespace, run_mode: str, enable_tb: bool
) -> int:
    defaults_by_mode = {
        "tune": 1,
        "reproduce": 5,
        "eval": 10,
    }
    if args.log_interval is not None:
        return max(1, int(args.log_interval))
    if not enable_tb:
        return 1
    return int(defaults_by_mode.get(run_mode, 1))


def _normalize_optional_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    return None if int(value) <= 0 else int(value)


def resolve_n_steps_per_env(args: argparse.Namespace, num_envs: int) -> int:
    if args.n_steps_per_env is not None:
        return max(1, int(args.n_steps_per_env))

    target_rollout_steps = max(1, int(args.rollout_steps_per_update))
    return max(1, int(np.ceil(target_rollout_steps / max(1, int(num_envs)))))


def build_scenario(
    *, schedule_time_s: float
) -> tuple[VehicleInfo, TrackInfo, SafeGuardUtility, TrainService]:
    slopes, slope_intervals = load_slopes()
    speed_limits, speed_limit_intervals = load_speed_limits(
        to_mps=True, dtype=np.float64
    )
    accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()
    longyang_start_position, putong_end_position = load_stations_goal_positions()
    levi_curves_list, brake_curves_list, min_curves_list, max_curves_list = (
        load_safeguard_curves(
            "levi_curves_list",
            "brake_curves_list",
            "min_curves_list",
            "max_curves_list",
        )
    )

    safeguard_utility = SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        levi_curves_list=levi_curves_list,
        brake_curves_list=brake_curves_list,
        min_curves_list=min_curves_list,
        max_curves_list=max_curves_list,
        factor=0.95,
    )
    track = TrackInfo(
        slopes=slopes,
        slope_intervals=slope_intervals,
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
        ASA_aps=accessible_points,
        ASA_dps=dangerous_points,
    )
    vehicle = VehicleInfo(mass=317.5, numoftrainsets=5, length=128.5)
    train_service = TrainService(
        start_position=longyang_start_position,
        start_speed=0.0,
        target_position=putong_end_position,
        schedule_time=schedule_time_s,
        max_acc_change=0.75,
        max_arr_time_error_ratio=5.0,
        max_stop_error=0.3,
    )

    return vehicle, track, safeguard_utility, train_service


def _format_float_token(value: float, *, decimals: int = 10) -> str:
    if not np.isfinite(value):
        raise ValueError("value must be finite")

    token = f"{round(float(value), decimals):.{decimals}f}".rstrip("0").rstrip(".")
    if token in {"", "-0", "0"}:
        token = "0"
    if "." not in token:
        token = f"{token}.0"
    return token.replace("-", "neg").replace(".", "p")


def _resolve_output_dir(
    *, output_root: str, schedule_time_s: float, max_step_distance: float
) -> str:
    schedule_token = _format_float_token(schedule_time_s)
    max_step_token = _format_float_token(max_step_distance)
    return os.path.join(output_root, f"{schedule_token}_{max_step_token}")


def main() -> None:
    args = build_arg_parser().parse_args()

    schedule_time_s = args.schedule_time_s
    ds = args.max_step_distance
    reward_discount = args.reward_discount

    output_root = args.output_root
    output_dir = _resolve_output_dir(
        output_root=output_root,
        schedule_time_s=schedule_time_s,
        max_step_distance=ds,
    )
    os.makedirs(output_dir, exist_ok=True)

    final_output_dir = os.path.join(output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    final_model_save_path = os.path.join(final_output_dir, _RL_MODEL_FILENAME)
    final_vecnormalize_save_path = os.path.join(
        final_output_dir,
        _RL_VECNORMALIZE_FILENAME,
    )
    best_eval_output_dir = os.path.join(
        output_dir, f"best_{args.best_eval_trigger_mode}"
    )

    (
        run_mode,
        enable_tb,
        enable_callback,
        enable_monitor,
        enable_env_diagnostics,
        enable_auto_analysis,
        enable_best_eval,
    ) = resolve_run_mode(args)
    log_interval = resolve_log_interval(args, run_mode, enable_tb)
    tb_sample_interval_steps = max(1, int(args.tb_sample_interval_steps))
    env_diagnostics_interval_steps = max(
        1,
        int(
            args.env_diagnostics_interval_steps
            if args.env_diagnostics_interval_steps is not None
            else tb_sample_interval_steps
        ),
    )
    force_dump_interval_steps = _normalize_optional_positive_int(
        args.force_dump_interval_steps
    )
    tb_batch_dump_records = _normalize_optional_positive_int(args.tb_batch_dump_records)
    num_envs = max(1, int(args.num_envs))
    n_steps_per_env = resolve_n_steps_per_env(args, num_envs)
    rollout_steps_per_update = n_steps_per_env * num_envs

    use_subproc = num_envs > 1 and args.vec_env_type == "subproc"
    resolved_vec_env_type = "subproc" if use_subproc else "dummy"

    print("Training runtime switches:")
    print(f"- run_mode={run_mode}")
    print(f"- enable_tb={enable_tb}")
    print(f"- enable_callback={enable_callback}")
    print(f"- enable_monitor={enable_monitor}")
    print(f"- enable_env_diagnostics={enable_env_diagnostics}")
    print(f"- enable_auto_analysis={enable_auto_analysis}")
    print(f"- enable_best_eval={enable_best_eval}")
    print(f"- reward_discount={reward_discount}")
    print(f"- schedule_time_s={schedule_time_s}")
    print(f"- max_step_distance={ds}")
    print(f"- num_envs={num_envs}")
    print(f"- vec_env_type={resolved_vec_env_type}")
    if use_subproc:
        print(f"- subproc_start_method={args.subproc_start_method}")
    print(f"- n_steps_per_env={n_steps_per_env}")
    print(f"- rollout_steps_per_update={rollout_steps_per_update}")
    print(f"- output_dir={output_dir}")
    print(f"- total_timesteps={args.total_timesteps}")
    print(f"- tensorboard_log_dir={args.tensorboard_log_dir}")
    print(f"- tb_log_name={args.tb_log_name}")
    if enable_tb:
        print(f"- log_interval={log_interval}")
    else:
        print("- log_interval=ignored (logging disabled by current switches)")
    print(f"- tb_sample_interval_steps={tb_sample_interval_steps}")
    print(f"- env_diagnostics_interval_steps={env_diagnostics_interval_steps}")
    print(f"- force_dump_interval_steps={force_dump_interval_steps}")
    print(f"- tb_batch_dump_records={tb_batch_dump_records}")
    print(f"- best_eval_trigger_mode={args.best_eval_trigger_mode}")
    print(f"- best_eval_trigger_interval={args.best_eval_trigger_interval}")
    print(f"- best_eval_output_dir={best_eval_output_dir}")
    print(f"- best_eval_deterministic={args.best_eval_deterministic}")
    print(f"- device={args.device}")
    if enable_auto_analysis and not enable_tb:
        print(
            "- warning: enable_auto_analysis=True while enable_tb=False; "
            f"analysis will use existing logs in {args.tensorboard_log_dir} if available."
        )

    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=schedule_time_s
    )

    # 创建训练环境
    env_initializers: list[Callable[[], Any]] = [
        _build_env_initializer(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            gamma=reward_discount,
            max_step_distance=ds,
            enable_diagnostics=enable_env_diagnostics,
            diagnostics_interval_steps=env_diagnostics_interval_steps,
        )
        for _ in range(num_envs)
    ]
    if use_subproc:
        venv_train = SubprocVecEnv(
            env_initializers,
            start_method=args.subproc_start_method,
        )
    else:
        venv_train = DummyVecEnv(env_initializers)

    if enable_monitor:
        venv_train = VecMonitor(venv_train)
    venv_train = VecNormalize(
        venv=venv_train, norm_obs=False, norm_reward=True, gamma=reward_discount
    )

    model = PPO(
        "MlpPolicy",
        venv_train,
        device=args.device,
        verbose=0,
        learning_rate=linear_schedule(3e-4),
        n_steps=n_steps_per_env,
        batch_size=256,
        n_epochs=15,
        gamma=reward_discount,  # 提高折扣因子
        gae_lambda=0.95,
        clip_range=0.2,  # 缩小clip区间, 防止策略突变
        # clip_range_vf=None,
        # normalize_advantage=True,
        ent_coef=0.01,  # 增加探索
        vf_coef=0.5,  # 增加价值函数权重
        max_grad_norm=0.5,
        tensorboard_log=args.tensorboard_log_dir if enable_tb else None,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # 改进网络结构
        ),
    )

    callbacks: list[BaseCallback] = []
    if enable_callback:
        callbacks.append(
            TensorboardCallback(
                tb_sample_interval_steps=tb_sample_interval_steps,
                force_dump_interval_steps=force_dump_interval_steps,
                batch_dump_records=tb_batch_dump_records,
            )
        )

    if enable_best_eval:
        callbacks.append(
            BestTrajectoryEvalCallback(
                eval_env=build_single_eval_env(
                    vehicle=vehicle,
                    track=track,
                    safeguard_utility=safeguard_utility,
                    train_service=train_service,
                    gamma=reward_discount,
                    max_step_distance=ds,
                    enable_diagnostics=False,
                    enable_trajectory_tracking=True,
                ),
                output_dir=best_eval_output_dir,
                eval_trigger_mode=args.best_eval_trigger_mode,
                eval_trigger_interval=max(1, int(args.best_eval_trigger_interval)),
                deterministic=args.best_eval_deterministic,
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    # 训练，并使用tensorboard记录回报和网络损失变化
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=log_interval,
        tb_log_name=args.tb_log_name,
        progress_bar=True,
    )
    model.save(final_model_save_path)
    venv_train.save(final_vecnormalize_save_path)
    venv_train.close()

    print("Training finished.")
    print(f"Final Model saved to: {final_model_save_path}")
    print(f"Final VecNormalize stats saved to: {final_vecnormalize_save_path}")
    if enable_best_eval:
        print(f"Best trajectory artifacts saved under: {best_eval_output_dir}")
    print("Run python -m scripts.evaluate_rl to evaluate the trained policy.")

    if enable_auto_analysis:
        try:
            analyze_config = AnalysisConfig(
                output_root=args.analysis_output_root,
                training_log_interval=log_interval if enable_tb else None,
                min_points_per_10k_steps=args.analysis_min_points_per_10k_steps,
                min_unique_episodes=args.analysis_min_unique_episodes,
                max_mean_step_gap=args.analysis_max_mean_step_gap,
                sampling_quality_mode=args.analysis_sampling_quality_mode,
            )
            analysis_result = run_training_analysis(
                log_root=args.tensorboard_log_dir,
                config=analyze_config,
            )
            output_paths = analysis_result.get("output_paths", {})
            print("Training analysis completed.")
            print(f"Analysis JSON: {output_paths.get('json_snapshot', 'N/A')}")
            print(f"Analysis report: {output_paths.get('markdown_report', 'N/A')}")
        except Exception as exc:
            print(f"Training analysis skipped due to error: {exc}")


if __name__ == "__main__":
    main()
