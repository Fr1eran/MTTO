from types import SimpleNamespace
from typing import cast

import pytest
from stable_baselines3.common.vec_env import VecEnv

from scripts.evaluate_rl import build_arg_parser, build_initial_rollout_series


def test_evaluate_rl_cli_accepts_dry_run_and_shared_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--schedule-time-s",
            "430.0",
            "--reward-profile",
            "basic",
            "--reward-discount",
            "0.95",
            "--device",
            "cuda",
        ]
    )

    assert args.dry_run is True
    assert args.schedule_time_s == 430.0
    assert args.reward_profile == "basic"
    assert args.reward_discount == 0.95
    assert args.device == "cuda"


def test_evaluate_rl_cli_rejects_removed_punctuality_dense_reward_option() -> None:
    with pytest.raises(SystemExit):
        _ = build_arg_parser().parse_args(["--plot-punctuality-dense-reward"])


def test_build_initial_rollout_series_reads_reset_environment_state() -> None:
    class FakeVecEnv:
        values: dict[str, list[object]] = {
            "state": [
                SimpleNamespace(
                    position_m=123.0,
                    speed_mps=4.5,
                    operation_time_s=0.0,
                    redundant_operation_time_s=26.0,
                )
            ]
        }

        def get_attr(self, attr_name: str):
            return self.values[attr_name]

    (
        position_seq,
        speed_seq,
        operation_time_seq,
        redundant_operation_time_seq,
    ) = build_initial_rollout_series(cast(VecEnv, cast(object, FakeVecEnv())))

    assert position_seq == [123.0]
    assert speed_seq == [4.5]
    assert operation_time_seq == [0.0]
    assert redundant_operation_time_seq == [26.0]
