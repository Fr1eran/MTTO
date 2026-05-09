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
from rl.evaluation import is_success_within_train_service_limits
from utils.data_loader import (
    load_auxiliary_stopping_areas_ap_and_dp,
    load_safeguard_curves,
    load_slopes,
    load_speed_limits,
    load_stations_goal_positions,
)
from utils.io_utils import save_curve_and_metrics

_RL_FINAL_MODEL_FILENAME = "final_model.zip"
_RL_FINAL_VECNORMALIZE_FILENAME = "final_vecnormalize.pkl"


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
        schedule_time=430.0,
        max_acc_change=0.75,
        max_arr_time_error_ratio=5.0,
        max_stop_error=0.3,
    )

    return vehicle, track, safeguard_utility, train_service


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained MTTO PPO policy.",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default="output/optimal/rl/final/",
        help="Search Path of PPO model zip file and vecnormalize pkl file.",
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
        default=False,
        help="Enable video recording for evaluation rollout.",
    )
    parser.add_argument(
        "--save-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save trajectory NPZ and metrics JSON files.",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="mtto_eval_video",
        help="Output directory for evaluation videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for saved trajectory files.",
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
    load_dir = args.load_dir

    model_zip_path = os.path.join(load_dir, f"{_RL_FINAL_MODEL_FILENAME}")
    vecnormalize_pkl_path = os.path.join(load_dir, f"{_RL_FINAL_VECNORMALIZE_FILENAME}")
    if not os.path.exists(model_zip_path):
        raise FileNotFoundError(f"Model file not found: {model_zip_path}")
    if not os.path.exists(vecnormalize_pkl_path):
        raise FileNotFoundError(
            f"VecNormalize stats file not found: {vecnormalize_pkl_path}"
        )

    vehicle, track, safeguard_utility, train_service = build_scenario()

    venv_eval = DummyVecEnv([
        lambda: make_env(
            vehicle=vehicle,
            track=track,
            safeguard_utility=safeguard_utility,
            train_service=train_service,
            gamma=reward_discount,
            max_step_distance=ds,
            enable_diagnostics=args.enable_env_diagnostics,
            enable_trajectory_tracking=args.record_video,
            render_mode="rgb_array" if args.record_video else None,
        )
    ])

    venv_eval = VecNormalize.load(vecnormalize_pkl_path, venv_eval)
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

    model = PPO.load(model_zip_path, device=args.device)

    total_reward = 0.0
    episode_steps = 0
    trajectory_position_seq: list[float] = [0.0]
    trajectory_speed_seq: list[float] = [0.0]

    obs = venv_eval.reset()
    episode_over = False
    last_info: dict[str, object] = {}

    while not episode_over:
        if not isinstance(obs, np.ndarray):
            raise TypeError("VecEnv observation must be a numpy.ndarray for MlpPolicy.")
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, rewards, dones, infos = venv_eval.step(action)
        total_reward += float(rewards[0])
        episode_steps += 1
        episode_over = bool(dones[0])
        last_info = infos[0]
        if isinstance(last_info, dict):
            basic = last_info.get("basic")
            if isinstance(basic, dict):
                trajectory_position_seq.append(float(basic.get("position", 0.0)))
                trajectory_speed_seq.append(float(basic.get("speed", 0.0)))

    target_time_s = float(train_service.schedule_time)
    start_position_m = float(train_service.start_position)
    target_position_m = float(train_service.target_position)

    # VecEnv 在 done 后会自动 reset；终态指标优先从最后一步 info 快照读取。
    basic_info = last_info.get("basic") if isinstance(last_info, dict) else None
    basic_snapshot = basic_info if isinstance(basic_info, dict) else {}

    # 提取性能指标
    final_position_m = float(basic_snapshot.get("position", 0.0))
    final_speed_mps = float(basic_snapshot.get("speed", 0.0))
    total_time_s = float(basic_snapshot.get("operation_time", 0.0))
    total_energy_kj = float(basic_snapshot.get("energy_consumption", 0.0))
    total_energy_j = total_energy_kj * 1000.0
    stop_error_m = abs(target_position_m - final_position_m)
    time_error_s = total_time_s - target_time_s

    comfort_tav = float(basic_snapshot.get("comfort_tav", 0.0))
    comfort_er_pct = float(basic_snapshot.get("comfort_er_pct", 0.0))
    comfort_rms = float(basic_snapshot.get("comfort_rms", 0.0))

    success = is_success_within_train_service_limits(
        stop_error_m=stop_error_m,
        time_error_s=time_error_s,
        train_service=train_service,
    )

    saved_npz_path = ""
    saved_json_path = ""
    output_dir = args.output_dir if args.output_dir is not None else load_dir
    if args.save_trajectory:
        npz_path = os.path.join(output_dir, "final_trajectory.npz")
        trajectory_metrics = {
            "total_reward": total_reward,
            "target_time_s": target_time_s,
            "total_time_s": total_time_s,
            "time_error_s": time_error_s,
            "start_position_m": start_position_m,
            "target_position_m": target_position_m,
            "final_position_m": final_position_m,
            "stop_error_m": stop_error_m,
            "total_energy_kj": total_energy_kj,
            "total_energy_j": total_energy_j,
            "final_speed_mps": final_speed_mps,
            "comfort_tav": comfort_tav,
            "comfort_er_pct": comfort_er_pct,
            "comfort_rms": comfort_rms,
            "episode_steps": episode_steps,
            "success": success,
        }
        saved_npz_path, saved_json_path = save_curve_and_metrics(
            pos_arr=trajectory_position_seq,
            speed_arr=trajectory_speed_seq,
            output_path=npz_path,
            metrics=trajectory_metrics,
        )

    venv_eval.close()

    print("========== Evaluation Results ==========")
    print(f"  total_reward:       {total_reward:.6f}")
    print(f"  success:            {success}")
    print(f"  target_time_s:      {target_time_s:.2f}")
    print(f"  total_time_s:       {total_time_s:.2f}")
    print(f"  time_error_s:       {time_error_s:.2f}")
    print(f"  start_position_m:   {start_position_m:.2f}")
    print(f"  target_position_m:  {target_position_m:.2f}")
    print(f"  final_position_m:   {final_position_m:.2f}")
    print(f"  stop_error_m:       {stop_error_m:.4f}")
    print(f"  final_speed_mps:    {final_speed_mps:.4f}")
    print(f"  total_energy_kj:    {total_energy_kj:.4f}")
    print(f"  total_energy_j:     {total_energy_j:.4f}")
    print(f"  comfort_tav:        {comfort_tav:.4f} m/s²")
    print(f"  comfort_er_pct:     {comfort_er_pct:.2f} %")
    print(f"  comfort_rms:        {comfort_rms:.4f} m/s²")
    print(f"  episode_steps:      {episode_steps}")
    if saved_npz_path:
        print(f"  trajectory_npz:     {saved_npz_path}")
    if saved_json_path:
        print(f"  trajectory_json:    {saved_json_path}")
    print("=========================================")


if __name__ == "__main__":
    main()
