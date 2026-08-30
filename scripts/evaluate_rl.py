import argparse
import os
from collections.abc import Mapping, Sequence
from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from contracts.training import RunMetadata
from rl.evaluation import (
    PolicyEvaluationResult,
    build_single_eval_env,
    evaluate_and_save_final_policy,
    evaluate_policy_once,
)
from rl.experiment_utils import (
    DEFAULT_DEVICE,
    RL_FINAL_MODEL_FILENAME,
    RewardPreset,
    load_run_metadata,
    resolve_reward_preset,
    reward_preset_names,
)
from rl.operational_state import OperationalState
from utils.plot_utils import apply_sci_figure_layout
from utils.scenario import build_scenario


def _get_initial_state(env: VecEnv | gym.Env) -> OperationalState:
    get_attr = getattr(env, "get_attr", None)
    if callable(get_attr):
        values = get_attr("state")
        if not values:
            raise RuntimeError("Could not read environment state")
        return values[0]

    state = getattr(getattr(env, "unwrapped", env), "state", None)
    if state is None:
        raise RuntimeError("Could not read environment state")
    return state


def build_initial_rollout_series(
    env: VecEnv | gym.Env,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
]:
    state = _get_initial_state(env)
    return (
        [float(state.position_m)],
        [float(state.speed_mps)],
        [float(state.operation_time_s)],
        [float(state.redundant_operation_time_s)],
    )


class OperationTimeTrace(gym.Wrapper):
    """Collect operation-time diagnostics without owning evaluation semantics."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.position_seq: list[float] = []
        self.speed_seq: list[float] = []
        self.operation_time_seq: list[float] = []
        self.redundant_operation_time_seq: list[float] = []

    def _reset_trace(self) -> None:
        (
            self.position_seq,
            self.speed_seq,
            self.operation_time_seq,
            self.redundant_operation_time_seq,
        ) = build_initial_rollout_series(self.env)

    def _append_current_state(self) -> None:
        state = _get_initial_state(self.env)
        self.position_seq.append(float(state.position_m))
        self.speed_seq.append(float(state.speed_mps))
        self.operation_time_seq.append(float(state.operation_time_s))
        self.redundant_operation_time_seq.append(
            float(state.redundant_operation_time_s)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._reset_trace()
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._append_current_state()
        return observation, reward, terminated, truncated, info


def plot_operation_time_series(
    operation_time_seq: Sequence[float],
    redundant_operation_time_seq: Sequence[float],
    target_time_s: float,
) -> None:
    if len(operation_time_seq) != len(redundant_operation_time_seq):
        raise ValueError(
            "operation_time_seq and redundant_operation_time_seq must have "
            "the same length"
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

    fig, (ax_time, ax_redundant) = plt.subplots(2, 1, sharex=True)
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
    ax_redundant.set_ylabel("Redundant operation time (s)")
    ax_redundant.grid(True, alpha=0.3)
    ax_redundant.legend()

    ax_redundant.set_xlabel("Agent step")
    _ = ax_time.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_time.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    _ = ax_redundant.text(
        0.02,
        0.98,
        "(b)",
        transform=ax_redundant.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    apply_sci_figure_layout(
        fig,
        columns=1,
        height_in=4.0,
        left=0.20,
        bottom=0.14,
        top=0.96,
        hspace=0.24,
    )
    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="评估 MTTO PPO 策略",
    )
    _ = parser.add_argument(
        "--load-dir",
        type=str,
        default="output/optimal/rl/final/",
        help="PPO 模型文件所在目录",
    )
    _ = parser.add_argument(
        "--reward-discount",
        type=float,
        default=None,
        help="评估环境的折扣因子",
    )
    _ = parser.add_argument(
        "--schedule-time-s",
        type=float,
        default=None,
        help="规划运行时间；未指定时优先从训练元数据读取。",
    )
    _ = parser.add_argument(
        "--step-distance",
        "--max-step-distance",
        dest="step_distance",
        type=float,
        default=None,
        help="评估环境的固定空间控制步长；--max-step-distance 为兼容别名。",
    )
    _ = parser.add_argument(
        "--reward-preset",
        type=str,
        choices=tuple(reward_preset_names()),
        default=None,
        help="奖励配置预设；未指定时优先从训练元数据读取。",
    )
    _ = parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="部署 PPO 模型的硬件设备",
    )
    _ = parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="评估时使用确定性策略",
    )
    _ = parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="记录评估过程视频",
    )
    _ = parser.add_argument(
        "--save-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存轨迹数据",
    )
    _ = parser.add_argument(
        "--video-folder",
        type=str,
        default="mtto_eval_video",
        help="评估视频保存路径",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="轨迹文件保存路径",
    )
    _ = parser.add_argument(
        "--video-length",
        type=int,
        default=10000,
        help="视频的最大状态转移步数",
    )
    _ = parser.add_argument(
        "--video-trigger-step",
        type=int,
        default=0,
        help="当状态转移至该步时，开始保存视频",
    )
    _ = parser.add_argument(
        "--plot-operation-time-series",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="评估后展示运行时间、冗余运行时间和 e_r 随智能体步数变化的曲线。",
    )
    _ = parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅解析评估配置、路径与训练元数据，不加载模型或运行 rollout。",
    )
    return parser


def _run_evaluation(
    args: argparse.Namespace,
    *,
    load_dir: str,
    run_metadata: RunMetadata | Mapping[str, object],
    schedule_time_s: float,
    reward_discount: float,
    step_distance: float,
    reward_preset: RewardPreset,
    output_dir: str,
) -> tuple[PolicyEvaluationResult, str, str, OperationTimeTrace | None]:
    model_zip_path = os.path.join(load_dir, RL_FINAL_MODEL_FILENAME)
    if not os.path.exists(model_zip_path):
        raise FileNotFoundError(f"Model file not found: {model_zip_path}")

    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=schedule_time_s
    )
    evaluation_env = build_single_eval_env(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        gamma=reward_discount,
        step_distance=step_distance,
        enable_trajectory_tracking=args.save_trajectory or args.record_video,
        render_mode="rgb_array" if args.record_video else None,
        reward_config=reward_preset.config,
    )
    if args.plot_operation_time_series:
        time_trace = OperationTimeTrace(evaluation_env)
    else:
        time_trace = None
    if time_trace is not None:
        evaluation_env = time_trace

    if args.record_video:
        eval_name_prefix = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        evaluation_env = RecordVideo(
            evaluation_env,
            video_folder=args.video_folder,
            step_trigger=lambda step: step == args.video_trigger_step,
            video_length=args.video_length,
            name_prefix=eval_name_prefix,
        )

    saved_npz_path = ""
    saved_json_path = ""
    try:
        model = PPO.load(model_zip_path, device=args.device)
        evaluation_metadata = (
            run_metadata.to_mapping()
            if isinstance(run_metadata, RunMetadata)
            else dict(run_metadata)
        )
        evaluation_metadata["evaluation_load_dir"] = load_dir
        if args.save_trajectory:
            (
                evaluation_result,
                saved_npz_path,
                saved_json_path,
            ) = evaluate_and_save_final_policy(
                model,
                evaluation_env,
                output_path=os.path.join(output_dir, "final_trajectory.npz"),
                metadata=evaluation_metadata,
                deterministic=args.deterministic,
                metrics_path=os.path.join(output_dir, "metrics_final.json"),
            )
        else:
            evaluation_result = evaluate_policy_once(
                model,
                evaluation_env,
                deterministic=args.deterministic,
            )
    finally:
        evaluation_env.close()

    return evaluation_result, saved_npz_path, saved_json_path, time_trace


def _print_evaluation_results(
    result: PolicyEvaluationResult,
    *,
    reward_preset_name: str,
    saved_npz_path: str,
    saved_json_path: str,
) -> None:
    print("========== Evaluation Results ==========")
    print(f"  total_reward:       {result.total_reward:.6f}")
    print(f"  success:            {result.success}")
    print(f"  precise_arrival:    {result.precise_arrival}")
    print(f"  punctual_arrival:   {result.punctual_arrival}")
    print(f"  reward_preset:     {reward_preset_name}")
    print(f"  target_time_s:      {result.target_time_s:.2f}")
    print(f"  total_time_s:       {result.total_time_s:.2f}")
    print(f"  time_error_s:       {result.time_error_s:.2f}")
    print(f"  start_position_m:   {result.start_position_m:.2f}")
    print(f"  target_position_m:  {result.target_position_m:.2f}")
    print(f"  final_position_m:   {result.final_position_m:.2f}")
    print(f"  stop_error_m:       {result.stop_error_m:.4f}")
    print(f"  final_speed_mps:    {result.final_speed_mps:.4f}")
    print(f"  total_energy_kj:    {result.total_energy_kj:.4f}")
    print(f"  total_energy_j:     {result.total_energy_j:.4f}")
    print(f"  comfort_tav:        {result.comfort_tav:.4f} m/s²")
    print(f"  comfort_er_pct:     {result.comfort_er_pct:.2f} %")
    print(f"  comfort_rms:        {result.comfort_rms:.4f} m/s²")
    print(f"  episode_steps:      {result.episode_steps}")
    if saved_npz_path:
        print(f"  trajectory_npz:     {saved_npz_path}")
    if saved_json_path:
        print(f"  trajectory_json:    {saved_json_path}")
    print("=========================================")


def main() -> None:
    args = build_arg_parser().parse_args()

    load_dir = args.load_dir
    run_metadata = load_run_metadata(load_dir)

    schedule_time_s = float(
        args.schedule_time_s
        if args.schedule_time_s is not None
        else run_metadata.schedule_time_s
    )
    reward_discount = float(
        args.reward_discount
        if args.reward_discount is not None
        else run_metadata.reward_discount
    )
    step_distance = float(
        args.step_distance
        if args.step_distance is not None
        else run_metadata.step_distance
    )
    reward_preset = resolve_reward_preset(
        args.reward_preset or run_metadata.reward_preset_name
    )
    output_dir = args.output_dir if args.output_dir is not None else load_dir
    model_zip_path = os.path.join(load_dir, RL_FINAL_MODEL_FILENAME)

    if args.dry_run:
        print("========== Evaluation Dry Run ==========")
        print(f"  load_dir:            {load_dir}")
        print(f"  output_dir:          {output_dir}")
        print(f"  reward_preset:      {reward_preset.name}")
        print(f"  reward_config:       {reward_preset.config}")
        print(f"  schedule_time_s:     {schedule_time_s:.2f}")
        print(f"  step_distance:       {step_distance:.2f}")
        print(f"  reward_discount:     {reward_discount:.4f}")
        print(f"  deterministic:       {args.deterministic}")
        print(f"  record_video:        {args.record_video}")
        print(f"  save_trajectory:     {args.save_trajectory}")
        print(f"  plot_operation_time_series: {args.plot_operation_time_series}")
        print(f"  model_zip_path:      {model_zip_path}")
        print(f"  model_exists:        {os.path.exists(model_zip_path)}")
        print("========================================")
        return

    (
        evaluation_result,
        saved_npz_path,
        saved_json_path,
        time_trace,
    ) = _run_evaluation(
        args,
        load_dir=load_dir,
        run_metadata=run_metadata,
        schedule_time_s=schedule_time_s,
        reward_discount=reward_discount,
        step_distance=step_distance,
        reward_preset=reward_preset,
        output_dir=output_dir,
    )
    _print_evaluation_results(
        evaluation_result,
        reward_preset_name=reward_preset.name,
        saved_npz_path=saved_npz_path,
        saved_json_path=saved_json_path,
    )

    if args.plot_operation_time_series:
        if time_trace is None:
            raise RuntimeError("Operation-time trace was not initialized")
        plot_operation_time_series(
            time_trace.operation_time_seq,
            time_trace.redundant_operation_time_seq,
            target_time_s=evaluation_result.target_time_s,
        )


if __name__ == "__main__":
    main()
