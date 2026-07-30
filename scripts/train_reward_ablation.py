from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from rl.experiment_utils import (
    DEFAULT_REWARD_DISCOUNT,
    TrainingRunSpec,
    build_default_training_args,
    resolve_training_run_spec,
    reward_profile_names,
    train_single_experiment,
)

DEFAULT_ABLATION_REWARD_PROFILES: tuple[str, ...] = (
    "basic",
    "basic_safety",
    "basic_safety_stopping",
)
ABLATION_MANIFEST_FILENAME = "reward_ablation_manifest.json"


@dataclass(frozen=True)
class AblationRunEntry:
    reward_profile_name: str
    repeat_index: int
    seed: int | None
    experiment_tag: str | None
    train_args: argparse.Namespace
    training_run_spec: TrainingRunSpec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 PBRS 奖励消融实验（训练曲线记录模式固定为 steps）",
    )
    parser.add_argument(
        "--ablation-output-root",
        type=str,
        default="output/optimal/rl/reward_ablation",
        help="三种奖励方案消融实验的输出根目录。",
    )
    parser.add_argument(
        "--ablation-tag",
        type=str,
        default=None,
        help="批次标签; 若 repeats > 1, 会自动附加 repeat 标识。",
    )
    parser.add_argument(
        "--reward-profiles",
        nargs="+",
        choices=reward_profile_names(),
        default=None,
        help=(
            "要运行的奖励预设列表。\
            默认按 basic -> safety -> stopping 的三种消融情形执行。"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每种奖励情形的重复训练次数。未指定 seed-list 时生效。",
    )
    parser.add_argument(
        "--seed-list",
        nargs="+",
        type=int,
        default=None,
        help="显式指定每次重复训练使用的 seed; 指定后优先于 repeats/base-seed。",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="重复训练的基准 seed; 若 repeats > 1, \
             则按 base-seed + repeat_index 推导。",
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
        default=30.0,
        help="训练环境相邻状态转移间的最大移动距离。",
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
        default=2048,
        help="PPO rollout 步数。",
    )
    parser.add_argument(
        "--n-steps-per-env",
        type=int,
        default=None,
        help="PPO n_steps 步数。\
             如果未指定, 则根据 rollout-steps-per-update 和 num-envs 计算得出。",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="PPO 总训练步数。",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="mtto_ppo_tb_logs",
        help="TensorBoard 日志输出根目录。",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="PPO log_interval。未指定时沿用 monitor_best 语义下的默认值。",
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
        default=10_000,
        help="根据 best-eval-trigger-mode 的设置, \
             以步数或回合数为单位的最佳评估触发间隔。",
    )
    parser.add_argument(
        "--best-eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在运行最佳评估回放时, 使用确定性策略。",
    )
    parser.add_argument(
        "--safety-position-bin-size-m",
        type=float,
        default=5000.0,
        help="安全违规位置统计使用的位置分桶长度(m)。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="指定运行 PPO 算法的硬件设备, 例如 'cpu' 或 'cuda'。",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅展开实验运行矩阵并预览输出路径, 不启动训练。",
    )
    return parser


def resolve_ablation_reward_profiles(
    reward_profiles: list[str] | None,
) -> tuple[str, ...]:
    profiles = (
        reward_profiles if reward_profiles else list(DEFAULT_ABLATION_REWARD_PROFILES)
    )
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for profile_name in profiles:
        if profile_name in seen:
            continue
        seen.add(profile_name)
        ordered_unique.append(profile_name)
    return tuple(ordered_unique)


def resolve_ablation_seeds(args: argparse.Namespace) -> list[int | None]:
    if args.seed_list:
        return [int(seed) for seed in args.seed_list]

    repeats = max(1, int(args.repeats))
    if repeats == 1:
        return [int(args.base_seed)] if args.base_seed is not None else [None]

    if args.base_seed is None:
        raise ValueError("当 repeats > 1 时, 必须提供 --base-seed 或 --seed-list.")

    return [int(args.base_seed) + repeat_index for repeat_index in range(repeats)]


def _build_repeat_experiment_tag(
    *,
    ablation_tag: str | None,
    repeat_index: int,
    total_repeats: int,
) -> str | None:
    repeat_token = None if total_repeats == 1 else f"r{repeat_index + 1:02d}"
    if ablation_tag and repeat_token:
        return f"{ablation_tag}__{repeat_token}"
    if ablation_tag:
        return ablation_tag
    return repeat_token


def _build_base_train_args() -> argparse.Namespace:
    return build_default_training_args()


def _apply_ablation_overrides(
    base_args: argparse.Namespace,
    *,
    args: argparse.Namespace,
    reward_profile_name: str,
    repeat_index: int,
    total_repeats: int,
    seed: int | None,
) -> argparse.Namespace:
    train_args = argparse.Namespace(**vars(base_args))

    for field_name in (
        "schedule_time_s",
        "max_step_distance",
        "reward_discount",
        "num_envs",
        "vec_env_type",
        "rollout_steps_per_update",
        "n_steps_per_env",
        "total_timesteps",
        "tensorboard_log_dir",
        "log_interval",
        "best_eval_trigger_mode",
        "best_eval_trigger_interval",
        "best_eval_deterministic",
        "safety_position_bin_size_m",
        "device",
    ):
        setattr(train_args, field_name, getattr(args, field_name))

    train_args.output_root = args.ablation_output_root
    train_args.reward_profile = reward_profile_name
    train_args.experiment_tag = _build_repeat_experiment_tag(
        ablation_tag=args.ablation_tag,
        repeat_index=repeat_index,
        total_repeats=total_repeats,
    )
    train_args.run_mode = "monitor_best"
    train_args.enable_tb = None
    train_args.enable_callback = None
    train_args.enable_monitor = None
    train_args.enable_env_diagnostics = None
    train_args.enable_auto_analysis = None
    train_args.enable_best_eval = None
    train_args.enable_safety_violation_bins = True
    train_args.enable_env_diagnostics = True
    train_args.env_diagnostics_interval_steps = 1
    train_args.tb_log_name = None
    train_args.rollout_record_trigger_mode = "steps"
    train_args.seed = seed
    train_args.dry_run = False
    return train_args


def resolve_ablation_run_matrix(args: argparse.Namespace) -> list[AblationRunEntry]:
    reward_profiles = resolve_ablation_reward_profiles(args.reward_profiles)
    seeds = resolve_ablation_seeds(args)
    base_train_args = _build_base_train_args()

    run_entries: list[AblationRunEntry] = []
    total_repeats = len(seeds)
    for repeat_index, seed in enumerate(seeds):
        for reward_profile_name in reward_profiles:
            train_args = _apply_ablation_overrides(
                base_args=base_train_args,
                args=args,
                reward_profile_name=reward_profile_name,
                repeat_index=repeat_index,
                total_repeats=total_repeats,
                seed=seed,
            )
            training_run_spec = resolve_training_run_spec(train_args)
            run_entries.append(
                AblationRunEntry(
                    reward_profile_name=reward_profile_name,
                    repeat_index=repeat_index,
                    seed=seed,
                    experiment_tag=train_args.experiment_tag,
                    train_args=train_args,
                    training_run_spec=training_run_spec,
                )
            )
    return run_entries


def build_ablation_manifest(
    args: argparse.Namespace,
    run_entries: list[AblationRunEntry],
    *,
    statuses: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_runs: list[dict[str, Any]] = []
    status_map = statuses or {}
    for entry in run_entries:
        status_payload = status_map.get(
            (entry.reward_profile_name, entry.repeat_index),
            {"status": "pending"},
        )
        manifest_runs.append({
            "reward_profile_name": entry.reward_profile_name,
            "repeat_index": entry.repeat_index,
            "seed": entry.seed,
            "experiment_tag": entry.experiment_tag,
            "run_mode": entry.training_run_spec.run_mode,
            "output_dir": entry.training_run_spec.output_dir,
            "final_output_dir": entry.training_run_spec.final_output_dir,
            "best_eval_output_dir": entry.training_run_spec.best_eval_output_dir,
            "tb_log_name": entry.training_run_spec.tb_log_name,
            "run_metadata_path": entry.training_run_spec.run_metadata_path,
            **status_payload,
        })

    return {
        "ablation_output_root": args.ablation_output_root,
        "ablation_tag": args.ablation_tag,
        "reward_profiles": list(resolve_ablation_reward_profiles(args.reward_profiles)),
        "repeats": int(args.repeats),
        "seed_list": [entry.seed for entry in run_entries],
        "dry_run": bool(args.dry_run),
        "run_mode": "monitor_best",
        "runs": manifest_runs,
    }


def _write_ablation_manifest(output_root: str, manifest: dict[str, Any]) -> str:
    os.makedirs(output_root, exist_ok=True)
    manifest_path = os.path.join(output_root, ABLATION_MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=True, indent=2)
    return manifest_path


def _print_run_matrix(run_entries: list[AblationRunEntry]) -> None:
    print("Resolved ablation run matrix:")
    print("rollout_record_trigger_mode is fixed to 'steps' for reward ablation.")
    for index, entry in enumerate(run_entries, start=1):
        print(
            f"[{index}] profile={entry.reward_profile_name} \
             repeat={entry.repeat_index + 1}"
            f"seed={entry.seed} output_dir={entry.training_run_spec.output_dir} "
            f"tb_log_name={entry.training_run_spec.tb_log_name}"
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        run_entries = resolve_ablation_run_matrix(args)
    except ValueError as exc:
        parser.error(str(exc))

    _print_run_matrix(run_entries)
    if args.dry_run:
        print(
            "Dry run completed: ablation run matrix resolved; \
            skipped all training executions."
        )
        return

    statuses: dict[tuple[str, int], dict[str, Any]] = {}
    manifest_path = _write_ablation_manifest(
        args.ablation_output_root,
        build_ablation_manifest(args, run_entries, statuses=statuses),
    )

    for index, entry in enumerate(run_entries, start=1):
        print(
            f"Running ablation job {index}/{len(run_entries)}: "
            f"profile={entry.reward_profile_name}, \
            repeat={entry.repeat_index + 1}, seed={entry.seed}"
        )
        try:
            train_single_experiment(entry.train_args, spec=entry.training_run_spec)
            statuses[(entry.reward_profile_name, entry.repeat_index)] = {
                "status": "completed"
            }
        except Exception as exc:
            statuses[(entry.reward_profile_name, entry.repeat_index)] = {
                "status": "failed",
                "error_message": str(exc),
            }
            _write_ablation_manifest(
                args.ablation_output_root,
                build_ablation_manifest(args, run_entries, statuses=statuses),
            )
            raise

        _write_ablation_manifest(
            args.ablation_output_root,
            build_ablation_manifest(args, run_entries, statuses=statuses),
        )

    print(f"Ablation training completed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
