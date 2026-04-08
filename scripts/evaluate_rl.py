import os
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from model.ocs import SafeGuardUtility,TrainService
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


def main():
    reward_discount = 0.99
    ds = 100.0
    model_save_path = "output/optimal/rl/ppo_mtto"
    vecnormalize_save_path = "output/optimal/rl/vecnormalize.pkl"

    model_zip_path = f"{model_save_path}.zip"
    if not os.path.exists(model_zip_path):
        raise FileNotFoundError(f"Model file not found: {model_zip_path}")
    if not os.path.exists(vecnormalize_save_path):
        raise FileNotFoundError(
            f"VecNormalize stats file not found: {vecnormalize_save_path}"
        )

    vehicle, track, safeguard_utility, task = build_scenario()

    venv_eval = DummyVecEnv(
        [
            lambda: make_env(
                vehicle=vehicle,
                track=track,
                safeguard_utility=safeguard_utility,
                train_service=task,
                gamma=reward_discount,
                max_step_distance=ds,
                render_mode="rgb_array",
            )
        ]
    )

    venv_eval = VecNormalize.load(vecnormalize_save_path, venv_eval)
    venv_eval.training = False
    venv_eval.norm_reward = False

    eval_name_prefix = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    venv_eval = VecVideoRecorder(
        venv_eval,
        video_folder="mtto_eval_video",
        record_video_trigger=lambda step: step == 0,
        video_length=10000,
        name_prefix=eval_name_prefix,
    )

    model = PPO.load(model_save_path, device="cpu")

    reward_after = 0.0
    energy_consumption = 0.0
    operation_time = 0.0
    docking_position = 0.0

    obs = venv_eval.reset()
    episode_over = False

    while not episode_over:
        if not isinstance(obs, np.ndarray):
            raise TypeError("VecEnv observation must be a numpy.ndarray for MlpPolicy.")
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv_eval.step(action)
        reward_after += float(rewards[0])
        episode_over = bool(dones[0])
        if episode_over:
            final_info = infos[0]
            energy_consumption = float(
                final_info.get("current_energy_consumption", 0.0)
            )
            operation_time = float(final_info.get("current_operation_time", 0.0))
            docking_position = float(final_info.get("docking_position", 0.0))

    venv_eval.close()

    print("Evaluate after train:")
    print(f"reward_after: {reward_after}")

    if (energy_consumption != 0.0) and (operation_time != 0.0):
        print(f"Total energy consumption:{energy_consumption}")
        print(f"Operation time:{operation_time}")
        print(f"Docking position: {docking_position}")


if __name__ == "__main__":
    main()
