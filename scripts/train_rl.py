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
        description="Train MTTO PPO policy with configurable logging and analysis switches.",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["tune", "reproduce", "eval"],
        default="tune",
        help="Preset runtime mode. tune enables logging+analysis, reproduce/eval favor efficiency.",
    )
    parser.add_argument(
        "--enable-tb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override TensorBoard logging switch.",
    )
    parser.add_argument(
        "--enable-callback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override TensorboardCallback switch.",
    )
    parser.add_argument(
        "--enable-monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override VecMonitor switch.",
    )
    parser.add_argument(
        "--enable-env-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override MTTOEnv diagnostics collection switch.",
    )
    parser.add_argument(
        "--enable-analysis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override post-training analysis switch.",
    )
    parser.add_argument(
        "--enable-best-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable single-environment best trajectory evaluation during training.",
    )
    parser.add_argument(
        "--analysis-output-root",
        type=str,
        default="mtto_train_reports",
        help="Output directory for post-training analysis artifacts.",
    )
    parser.add_argument(
        "--analysis-min-points-per-10k-steps",
        type=float,
        default=5.0,
        help="Minimum acceptable mean samples per 10k steps for auto-analysis quality gate.",
    )
    parser.add_argument(
        "--analysis-min-unique-episodes",
        type=int,
        default=100,
        help="Minimum acceptable unique episodes for auto-analysis quality gate.",
    )
    parser.add_argument(
        "--analysis-max-mean-step-gap",
        type=float,
        default=2048.0,
        help="Maximum acceptable mean step gap for auto-analysis quality gate.",
    )
    parser.add_argument(
        "--analysis-sampling-quality-mode",
        type=str,
        choices=["warn_only", "strict_fail"],
        default="warn_only",
        help="Sampling quality gate mode for auto-analysis.",
    )
    parser.add_argument(
        "--reward-discount",
        type=float,
        default=0.99,
        help="Discount factor used by env and PPO.",
    )
    parser.add_argument(
        "--step-distance",
        type=float,
        default=100.0,
        help="Environment max_step_distance.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel training environments.",
    )
    parser.add_argument(
        "--vec-env-type",
        type=str,
        choices=["dummy", "subproc"],
        default="subproc",
        help="Vectorized environment backend. subproc enables parallel sampling when num_envs > 1.",
    )
    parser.add_argument(
        "--subproc-start-method",
        type=str,
        choices=["spawn", "forkserver"],
        default="spawn",
        help="Multiprocessing start method for SubprocVecEnv.",
    )
    parser.add_argument(
        "--rollout-steps-per-update",
        type=int,
        default=2048,
        help="Target rollout steps collected per PPO update across all envs.",
    )
    parser.add_argument(
        "--n-steps-per-env",
        type=int,
        default=None,
        help="Override PPO n_steps for each environment. If omitted, it is derived from rollout-steps-per-update and num-envs.",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="output/optimal/rl/final/ppo_mtto",
        help="Path prefix for saving PPO model (without .zip suffix).",
    )
    parser.add_argument(
        "--vecnormalize-save-path",
        type=str,
        default="output/optimal/rl/final/vecnormalize.pkl",
        help="Path for saving VecNormalize stats.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Total timesteps for PPO training.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="mtto_ppo_tensorboard_logs",
        help="TensorBoard log root directory.",
    )
    parser.add_argument(
        "--tb-log-name",
        type=str,
        default="trainning_log",
        help="TensorBoard run name used by model.learn.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Override PPO log_interval. Effective when logging is enabled; default is tune=1 and reproduce/eval are ignored under default no-logging switches.",
    )
    parser.add_argument(
        "--tb-sample-interval-steps",
        type=int,
        default=1,
        help="Minimum timesteps between callback TensorBoard records.",
    )
    parser.add_argument(
        "--env-diagnostics-interval-steps",
        type=int,
        default=None,
        help="Minimum env steps between diagnostics snapshot emission. Defaults to tb-sample-interval-steps.",
    )
    parser.add_argument(
        "--force-dump-interval-steps",
        type=int,
        default=0,
        help="Force a buffered TensorBoard flush every N timesteps in callback (<=0 disables).",
    )
    parser.add_argument(
        "--tb-batch-dump-records",
        type=int,
        default=0,
        help="Flush buffered TensorBoard events after this many sampled callback records (<=0 disables).",
    )
    parser.add_argument(
        "--best-eval-trigger-mode",
        type=str,
        choices=["steps", "episodes"],
        default="steps",
        help="Trigger best-eval callback by global timesteps or completed episodes.",
    )
    parser.add_argument(
        "--best-eval-interval",
        type=int,
        default=100_000,
        help="Best-eval interval in timesteps or episodes, depending on best-eval-trigger-mode.",
    )
    parser.add_argument(
        "--best-eval-output-dir",
        type=str,
        default="output/optimal/rl/best",
        help="Directory used to save best model, VecNormalize stats, and trajectory artifacts.",
    )
    parser.add_argument(
        "--best-eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy when running best-eval rollouts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device for PPO, e.g. cpu or cuda.",
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
        if args.enable_analysis is None
        else args.enable_analysis
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


def build_scenario() -> tuple[VehicleInfo, TrackInfo, SafeGuardUtility, TrainService]:
    slopes, slope_intervals = load_slopes()
    speed_limits, speed_limit_intervals = load_speed_limits(
        to_mps=True, dtype=np.float64
    )
    accessible_points, dangerous_points = load_auxiliary_stopping_areas_ap_and_dp()
    longyang_start_position, putong_end_position = load_stations_goal_positions()
    min_curves_list, max_curves_list = load_safeguard_curves(
        "min_curves_list", "max_curves_list"
    )

    safeguard_utility = SafeGuardUtility(
        speed_limits=speed_limits,
        speed_limit_intervals=speed_limit_intervals,
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
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error=60.0,
        max_stop_error=2.0,
    )

    return vehicle, track, safeguard_utility, train_service


def main() -> None:
    args = build_arg_parser().parse_args()

    reward_discount = args.reward_discount
    ds = args.step_distance
    model_save_path = args.model_save_path
    vecnormalize_save_path = args.vecnormalize_save_path

    ensure_parent_dir(model_save_path)
    ensure_parent_dir(vecnormalize_save_path)

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
    print(f"- step_distance={ds}")
    print(f"- num_envs={num_envs}")
    print(f"- vec_env_type={resolved_vec_env_type}")
    if use_subproc:
        print(f"- subproc_start_method={args.subproc_start_method}")
    print(f"- n_steps_per_env={n_steps_per_env}")
    print(f"- rollout_steps_per_update={rollout_steps_per_update}")
    print(f"- model_save_path={model_save_path}")
    print(f"- vecnormalize_save_path={vecnormalize_save_path}")
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
    print(f"- best_eval_interval={args.best_eval_interval}")
    print(f"- best_eval_output_dir={args.best_eval_output_dir}")
    print(f"- best_eval_deterministic={args.best_eval_deterministic}")
    print(f"- device={args.device}")
    if enable_auto_analysis and not enable_tb:
        print(
            "- warning: enable_auto_analysis=True while enable_tb=False; "
            f"analysis will use existing logs in {args.tensorboard_log_dir} if available."
        )

    vehicle, track, safeguard_utility, train_service = build_scenario()

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
                output_dir=args.best_eval_output_dir,
                trigger_mode=args.best_eval_trigger_mode,
                trigger_interval=max(1, int(args.best_eval_interval)),
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
    model.save(model_save_path)
    venv_train.save(vecnormalize_save_path)
    venv_train.close()

    print("Training finished.")
    print(f"Model saved to: {model_save_path}.zip")
    print(f"VecNormalize stats saved to: {vecnormalize_save_path}")
    if enable_best_eval:
        print(f"Best trajectory artifacts saved under: {args.best_eval_output_dir}")
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
