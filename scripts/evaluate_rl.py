import os
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from model.ocs import SafeGuardUtility, TrainService
from model.track import TrackInfo
from model.vehicle import VehicleInfo
from rl.env_factory import make_env
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)


def build_scenario() -> tuple[VehicleInfo, TrackInfo, SafeGuardUtility, TrainService]:
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
        schedule_time=440.0,
        max_acc_change=0.75,
        max_arr_time_error_ratio=60.0,
        max_stop_error=2.0,
    )

    return vehicle, track, safeguard_utility, train_service


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained MTTO PPO policy.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="output/optimal/rl/final/ppo_mtto_model",
        help="Path prefix of PPO model zip file (without .zip suffix).",
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=str,
        default="output/optimal/rl/final/vecnormalize.pkl",
        help="Path of VecNormalize stats file.",
    )
    parser.add_argument(
        "--reward-discount",
        type=float,
        default=0.99,
        help="Discount factor used to reconstruct evaluation environment.",
    )
    parser.add_argument(
        "--step-distance",
        type=float,
        default=100.0,
        help="Environment max_step_distance for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device for loading PPO model.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy during evaluation.",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable video recording for evaluation rollout.",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="mtto_eval_video",
        help="Output directory for evaluation videos.",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=10000,
        help="Maximum recorded video length in steps.",
    )
    parser.add_argument(
        "--video-trigger-step",
        type=int,
        default=0,
        help="Record video when step equals this value.",
    )
    parser.add_argument(
        "--enable-env-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable diagnostics collection in evaluation environment.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    reward_discount = args.reward_discount
    ds = args.step_distance
    model_save_path = args.model_path
    vecnormalize_save_path = args.vecnormalize_path

    model_zip_path = f"{model_save_path}.zip"
    if not os.path.exists(model_zip_path):
        raise FileNotFoundError(f"Model file not found: {model_zip_path}")
    if not os.path.exists(vecnormalize_save_path):
        raise FileNotFoundError(
            f"VecNormalize stats file not found: {vecnormalize_save_path}"
        )

    vehicle, track, safeguard_utility, task = build_scenario()

    venv_eval = DummyVecEnv([
        lambda: make_env(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=task,
            gamma=reward_discount,
            max_step_distance=ds,
            enable_diagnostics=args.enable_env_diagnostics,
            enable_trajectory_tracking=args.record_video,
            render_mode="rgb_array" if args.record_video else None,
        )
    ])

    venv_eval = VecNormalize.load(vecnormalize_save_path, venv_eval)
    venv_eval.training = False
    venv_eval.norm_reward = False

    if args.record_video:
        eval_name_prefix = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        venv_eval = VecVideoRecorder(
            venv_eval,
            video_folder=args.video_folder,
            record_video_trigger=lambda step: step == args.video_trigger_step,
            video_length=args.video_length,
            name_prefix=eval_name_prefix,
        )

    model = PPO.load(model_save_path, device=args.device)

    reward_after = 0.0
    energy_consumption = 0.0
    operation_time = 0.0
    position = 0.0
    comfort_tav = 0.0
    comfort_er_pct = 0.0
    comfort_rms = 0.0

    obs = venv_eval.reset()
    episode_over = False

    while not episode_over:
        if not isinstance(obs, np.ndarray):
            raise TypeError("VecEnv observation must be a numpy.ndarray for MlpPolicy.")
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, rewards, dones, infos = venv_eval.step(action)
        reward_after += float(rewards[0])
        episode_over = bool(dones[0])
        if episode_over:
            final_info = infos[0]
            basic_info: dict[str, object] = {}
            if isinstance(final_info, dict):
                basic_info_candidate = final_info.get("basic", {})
                if isinstance(basic_info_candidate, dict):
                    basic_info = basic_info_candidate

            energy_consumption = float(basic_info.get("energy_consumption", 0.0))
            operation_time = float(basic_info.get("operation_time", 0.0))
            position = float(basic_info.get("position", 0.0))
            comfort_tav = float(basic_info.get("comfort_tav", 0.0))
            comfort_er_pct = float(basic_info.get("comfort_er_pct", 0.0))
            comfort_rms = float(basic_info.get("comfort_rms", 0.0))

    venv_eval.close()

    print("Evaluate after train:")
    print(f"reward_after: {reward_after}")

    if (energy_consumption != 0.0) and (operation_time != 0.0):
        print(f"Total energy consumption:{energy_consumption}")
        print(f"Operation time:{operation_time}")
        print(f"Position: {position}")
        print(f"Comfort TAV: {comfort_tav:.4f} m/s\u00b2")
        print(f"Comfort ER:  {comfort_er_pct:.2f} %")
        print(f"Comfort RMS: {comfort_rms:.4f} m/s\u00b2")


if __name__ == "__main__":
    main()
