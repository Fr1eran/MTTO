import os
import argparse
import numpy as np

from rl.callbacks import TensorboardCallback
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


# 学习率线性衰减
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def ensure_parent_dir(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


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
        "--analysis-output-root",
        type=str,
        default="mtto_train_reports",
        help="Output directory for post-training analysis artifacts.",
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
        "--model-save-path",
        type=str,
        default="output/optimal/rl/ppo_mtto",
        help="Path prefix for saving PPO model (without .zip suffix).",
    )
    parser.add_argument(
        "--vecnormalize-save-path",
        type=str,
        default="output/optimal/rl/vecnormalize.pkl",
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
        "--device",
        type=str,
        default="cpu",
        help="Training device for PPO, e.g. cpu or cuda.",
    )
    return parser


def resolve_run_mode(
    args: argparse.Namespace,
) -> tuple[str, bool, bool, bool, bool, bool]:
    run_mode = args.run_mode
    defaults_by_mode = {
        "tune": {
            "tb": True,
            "callback": True,
            "monitor": True,
            "env_diagnostics": True,
            "analysis": True,
        },
        "reproduce": {
            "tb": False,
            "callback": False,
            "monitor": False,
            "env_diagnostics": False,
            "analysis": False,
        },
        "eval": {
            "tb": False,
            "callback": False,
            "monitor": False,
            "env_diagnostics": False,
            "analysis": False,
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

    if not enable_tb:
        enable_callback = False

    return (
        run_mode,
        enable_tb,
        enable_callback,
        enable_monitor,
        enable_env_diagnostics,
        enable_auto_analysis,
    )


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
        max_arr_time_error=120,
        max_stop_error=0.3,
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
    ) = resolve_run_mode(args)

    print("Training runtime switches:")
    print(f"- run_mode={run_mode}")
    print(f"- enable_tb={enable_tb}")
    print(f"- enable_callback={enable_callback}")
    print(f"- enable_monitor={enable_monitor}")
    print(f"- enable_env_diagnostics={enable_env_diagnostics}")
    print(f"- enable_auto_analysis={enable_auto_analysis}")
    print(f"- reward_discount={reward_discount}")
    print(f"- step_distance={ds}")
    print(f"- model_save_path={model_save_path}")
    print(f"- vecnormalize_save_path={vecnormalize_save_path}")
    print(f"- total_timesteps={args.total_timesteps}")
    print(f"- tensorboard_log_dir={args.tensorboard_log_dir}")
    print(f"- tb_log_name={args.tb_log_name}")
    print(f"- device={args.device}")
    if enable_auto_analysis and not enable_tb:
        print(
            "- warning: enable_auto_analysis=True while enable_tb=False; "
            f"analysis will use existing logs in {args.tensorboard_log_dir} if available."
        )

    vehicle, track, safeguard_utility, train_service = build_scenario()

    # 创建训练环境
    venv_train = DummyVecEnv(
        [
            lambda: make_env(
                vehicle=vehicle,
                track=track,
                safeguard_utility=safeguard_utility,
                train_service=train_service,
                gamma=reward_discount,
                max_step_distance=ds,
                enable_diagnostics=enable_env_diagnostics,
            )
        ]
    )
    if enable_monitor:
        venv_train = VecMonitor(venv_train)
    venv_train = VecNormalize(
        venv=venv_train, norm_obs=False, norm_reward=True, gamma=reward_discount
    )

    model = PPO(
        "MlpPolicy",
        venv_train,
        device=args.device,
        verbose=1,
        learning_rate=linear_schedule(3e-4),
        n_steps=2048,
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

    # 训练，并使用tensorboard记录回报和网络损失变化
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=TensorboardCallback() if enable_callback else None,
        log_interval=5,
        tb_log_name=args.tb_log_name,
    )
    model.save(model_save_path)
    venv_train.save(vecnormalize_save_path)
    venv_train.close()

    print("Training finished.")
    print(f"Model saved to: {model_save_path}.zip")
    print(f"VecNormalize stats saved to: {vecnormalize_save_path}")
    print("Run python -m scripts.evaluate_rl to evaluate the trained policy.")

    if enable_auto_analysis:
        try:
            analyze_config = AnalysisConfig(output_root=args.analysis_output_root)
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
