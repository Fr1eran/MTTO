import argparse
import os
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from rl.env_factory import make_env
from rl.evaluation import (
    PolicyEvaluationResult,
    classify_arrival_status,
    get_strict_stop_error_limit_m,
    get_strict_time_error_limit_s,
)
from rl.experiment_utils import (
    DEFAULT_MAX_STEP_DISTANCE,
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_REWARD_PROFILE_NAME,
    DEFAULT_SCHEDULE_TIME_S,
    RL_FINAL_MODEL_FILENAME,
    RL_FINAL_VECNORMALIZE_FILENAME,
    build_run_metadata,
    load_run_metadata,
    resolve_reward_profile,
    reward_profile_names,
)
from utils.io_utils import save_curve_and_metrics
from utils.scenario import build_scenario


def _get_initial_state(venv):
    values = venv.get_attr("state")
    if not values:
        raise RuntimeError("Could not read environment state")
    return values[0]


def build_initial_rollout_series(
    venv,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
]:
    state = _get_initial_state(venv)
    return (
        [float(state.position_m)],
        [float(state.speed_mps)],
        [float(state.operation_time_s)],
        [float(state.redundant_operation_time_s)],
    )


def plot_operation_time_series(
    operation_time_seq: Sequence[float],
    redundant_operation_time_seq: Sequence[float],
    target_time_s: float,
) -> None:
    if len(operation_time_seq) != len(redundant_operation_time_seq):
        raise ValueError(
            "operation_time_seq and redundant_operation_time_seq must have the same length"
        )
    if len(operation_time_seq) == 0:
        print("No operation-time samples collected; skipped time-series plot.")
        return

    import matplotlib.pyplot as plt

    step_axis = np.arange(len(operation_time_seq), dtype=np.int32)
    operation_time_arr = np.asarray(operation_time_seq, dtype=np.float32)
    redundant_operation_time_arr = np.asarray(
        redundant_operation_time_seq,
        dtype=np.float32,
    )

    fig, (ax_time, ax_redundant) = plt.subplots(
        2,
        1,
        figsize=(10, 6.5),
        sharex=True,
    )
    ax_time.plot(
        step_axis,
        operation_time_arr,
        color="#2563eb",
        linewidth=1.5,
        label="Operation time",
    )
    ax_time.axhline(
        target_time_s,
        color="#dc2626",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
        label="Schedule time",
    )
    ax_time.set_title("Operation time over evaluation rollout")
    ax_time.set_ylabel("Operation time (s)")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()

    ax_redundant.plot(
        step_axis,
        redundant_operation_time_arr,
        color="#16a34a",
        linewidth=1.5,
        label="Redundant operation time",
    )
    ax_redundant.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        label="No redundancy",
    )
    ax_redundant.set_title("Redundant operation time over evaluation rollout")
    ax_redundant.set_ylabel("Redundant operation time (s)")
    ax_redundant.grid(True, alpha=0.3)
    ax_redundant.legend()

    ax_redundant.set_xlabel("Agent step")

    fig.tight_layout()
    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="评估 MTTO PPO 策略",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default="output/optimal/rl/final/",
        help="PPO 模型文件和 vecnormalize 文件的搜索路径",
    )
    parser.add_argument(
        "--reward-discount",
        type=float,
        default=None,
        help="评估环境的折扣因子",
    )
    parser.add_argument(
        "--schedule-time-s",
        type=float,
        default=None,
        help="规划运行时间；未指定时优先从训练元数据读取。",
    )
    parser.add_argument(
        "--step-distance",
        type=float,
        default=None,
        help="评估环境的最大仿真位移步长",
    )
    parser.add_argument(
        "--reward-profile",
        type=str,
        choices=tuple(reward_profile_names()),
        default=None,
        help="奖励配置预设；未指定时优先从训练元数据读取。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="部署 PPO 模型的硬件设备",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="评估时使用确定性策略",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="记录评估过程视频",
    )
    parser.add_argument(
        "--save-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存轨迹数据",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="mtto_eval_video",
        help="评估视频保存路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="轨迹文件保存路径",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=10000,
        help="视频的最大状态转移步数",
    )
    parser.add_argument(
        "--video-trigger-step",
        type=int,
        default=0,
        help="当状态转移至该步时，开始保存视频",
    )
    parser.add_argument(
        "--enable-env-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="评估时收集诊断信息",
    )
    parser.add_argument(
        "--plot-operation-time-series",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="评估后展示运行时间、冗余运行时间和 e_r 随智能体步数变化的曲线。",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅解析评估配置、路径与训练元数据，不加载模型或运行 rollout。",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    load_dir = args.load_dir
    run_metadata = load_run_metadata(load_dir)

    schedule_time_s = float(
        args.schedule_time_s
        if args.schedule_time_s is not None
        else run_metadata.get("schedule_time_s", DEFAULT_SCHEDULE_TIME_S)
    )
    reward_discount = float(
        args.reward_discount
        if args.reward_discount is not None
        else run_metadata.get("reward_discount", DEFAULT_REWARD_DISCOUNT)
    )
    ds = float(
        args.step_distance
        if args.step_distance is not None
        else run_metadata.get("max_step_distance", DEFAULT_MAX_STEP_DISTANCE)
    )
    reward_profile = resolve_reward_profile(
        args.reward_profile
        or str(run_metadata.get("reward_profile_name", DEFAULT_REWARD_PROFILE_NAME))
    )
    reward_config = reward_profile.to_reward_config()
    output_dir = args.output_dir if args.output_dir is not None else load_dir

    model_zip_path = os.path.join(load_dir, RL_FINAL_MODEL_FILENAME)
    vecnormalize_pkl_path = os.path.join(load_dir, RL_FINAL_VECNORMALIZE_FILENAME)
    if args.dry_run:
        print("========== Evaluation Dry Run ==========")
        print(f"  load_dir:            {load_dir}")
        print(f"  output_dir:          {output_dir}")
        print(f"  reward_profile:      {reward_profile.name}")
        print(f"  reward_config:       {reward_config}")
        print(f"  schedule_time_s:     {schedule_time_s:.2f}")
        print(f"  step_distance:       {ds:.2f}")
        print(f"  reward_discount:     {reward_discount:.4f}")
        print(f"  deterministic:       {args.deterministic}")
        print(f"  record_video:        {args.record_video}")
        print(f"  save_trajectory:     {args.save_trajectory}")
        print(f"  plot_operation_time_series: {args.plot_operation_time_series}")
        print(f"  model_zip_path:      {model_zip_path}")
        print(f"  model_exists:        {os.path.exists(model_zip_path)}")
        print(f"  vecnormalize_path:   {vecnormalize_pkl_path}")
        print(f"  vecnormalize_exists: {os.path.exists(vecnormalize_pkl_path)}")
        print("========================================")
        return

    if not os.path.exists(model_zip_path):
        raise FileNotFoundError(f"Model file not found: {model_zip_path}")
    if not os.path.exists(vecnormalize_pkl_path):
        raise FileNotFoundError(
            f"VecNormalize stats file not found: {vecnormalize_pkl_path}"
        )

    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=schedule_time_s
    )

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
            reward_config=reward_config,
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

    obs = venv_eval.reset()
    (
        trajectory_position_seq,
        trajectory_speed_seq,
        operation_time_seq,
        redundant_operation_time_seq,
    ) = build_initial_rollout_series(venv_eval)
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
                operation_time_seq.append(float(basic.get("operation_time", 0.0)))
                redundant_operation_time_seq.append(
                    float(basic.get("redundant_operation_time", 0.0))
                )

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
    outcome_info = last_info.get("outcome") if isinstance(last_info, dict) else None
    outcome_snapshot = outcome_info if isinstance(outcome_info, dict) else {}
    truncated = bool(
        outcome_snapshot.get(
            "truncated",
            last_info.get("TimeLimit.truncated", False)
            if isinstance(last_info, dict)
            else False,
        )
    )
    terminated = bool(
        outcome_snapshot.get("terminated", episode_over and not truncated)
    )

    success, precise_arrival, punctual_arrival = classify_arrival_status(
        stop_error_m=stop_error_m,
        time_error_s=time_error_s,
        final_speed_mps=final_speed_mps,
        train_service=train_service,
        terminated=terminated,
        truncated=truncated,
    )

    saved_npz_path = ""
    saved_json_path = ""
    if args.save_trajectory:
        npz_path = os.path.join(output_dir, "final_trajectory.npz")
        trajectory_metadata = build_run_metadata(
            reward_profile=reward_profile,
            schedule_time_s=schedule_time_s,
            max_step_distance=ds,
            reward_discount=reward_discount,
            run_mode=str(run_metadata.get("run_mode")),
            experiment_tag=str(run_metadata.get("experiment_tag")),
            tensorboard_log_dir=str(run_metadata.get("tensorboard_log_dir")),
            tb_log_name=str(run_metadata.get("tb_log_name")),
            output_dir=str(run_metadata.get("output_dir")),
            final_output_dir=str(run_metadata.get("final_output_dir")),
            best_eval_output_dir=str(run_metadata.get("best_eval_output_dir")),
        )
        trajectory_metadata["trajectory_source"] = "final"
        trajectory_metadata["evaluation_load_dir"] = load_dir
        trajectory_metadata["deterministic"] = bool(args.deterministic)
        evaluation_result = PolicyEvaluationResult(
            success=success,
            precise_arrival=precise_arrival,
            punctual_arrival=punctual_arrival,
            total_reward=float(total_reward),
            total_time_s=total_time_s,
            target_time_s=target_time_s,
            total_energy_j=total_energy_j,
            total_energy_kj=total_energy_kj,
            start_position_m=start_position_m,
            target_position_m=target_position_m,
            final_position_m=final_position_m,
            final_speed_mps=final_speed_mps,
            stop_error_m=stop_error_m,
            time_error_s=time_error_s,
            strict_stop_error_limit_m=get_strict_stop_error_limit_m(train_service),
            strict_time_error_limit_s=get_strict_time_error_limit_s(train_service),
            comfort_tav=comfort_tav,
            comfort_er_pct=comfort_er_pct,
            comfort_rms=comfort_rms,
            terminated=terminated,
            truncated=truncated,
            episode_steps=episode_steps,
            trajectory_pos_m=np.asarray(trajectory_position_seq, dtype=np.float32),
            trajectory_speed_mps=np.asarray(trajectory_speed_seq, dtype=np.float32),
        )
        trajectory_metrics = evaluation_result.to_metrics()
        trajectory_metrics.update(trajectory_metadata)
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
    print(f"  precise_arrival:    {precise_arrival}")
    print(f"  punctual_arrival:   {punctual_arrival}")
    print(f"  reward_profile:     {reward_profile.name}")
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

    if args.plot_operation_time_series:
        plot_operation_time_series(
            operation_time_seq,
            redundant_operation_time_seq,
            target_time_s=target_time_s,
        )


if __name__ == "__main__":
    main()
