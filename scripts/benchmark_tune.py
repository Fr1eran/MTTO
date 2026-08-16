"""Measure the hot-path cost of tune-mode environment telemetry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from rl.env_factory import make_env
from rl.experiment_utils import (
    DEFAULT_REWARD_DISCOUNT,
    DEFAULT_SCHEDULE_TIME_S,
    DEFAULT_STEP_DISTANCE,
)
from utils.scenario import build_scenario


def run_environment_benchmark(*, steps: int, rollout_capacity: int) -> dict[str, float]:
    vehicle, track, safeguard_utility, train_service = build_scenario(
        schedule_time_s=DEFAULT_SCHEDULE_TIME_S
    )
    env = make_env(
        vehicle=vehicle,
        track=track,
        safeguard_utility=safeguard_utility,
        train_service=train_service,
        gamma=DEFAULT_REWARD_DISCOUNT,
        step_distance=DEFAULT_STEP_DISTANCE,
        reward_diagnostics_worker_rank=0,
        reward_diagnostics_rollout_capacity=rollout_capacity,
        compact_training_info=True,
    )
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        info_field_count = 0
        drain_count = 0
        start = time.perf_counter()
        for step_index in range(steps):
            _, _, terminated, truncated, info = env.step(action)
            info_field_count += len(info)
            if (step_index + 1) % rollout_capacity == 0:
                env.drain_reward_diagnostics()
                drain_count += 1
            if terminated or truncated:
                env.reset()
        elapsed = max(time.perf_counter() - start, 1e-12)
    finally:
        env.close()

    return {
        "steps": float(steps),
        "elapsed_s": elapsed,
        "steps_per_s": float(steps) / elapsed,
        "mean_info_top_level_fields": float(info_field_count) / float(steps),
        "rollout_drain_count": float(drain_count),
        "rollout_capacity": float(rollout_capacity),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark compact tune telemetry environment throughput."
    )
    _ = parser.add_argument("--steps", type=int, default=8192)
    _ = parser.add_argument("--rollout-capacity", type=int, default=2048)
    _ = parser.add_argument("--json-output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.steps <= 0 or args.rollout_capacity <= 0:
        raise ValueError("--steps and --rollout-capacity must be positive")
    result: dict[str, Any] = run_environment_benchmark(
        steps=int(args.steps),
        rollout_capacity=int(args.rollout_capacity),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        _ = args.json_output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
