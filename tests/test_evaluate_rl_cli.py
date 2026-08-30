from types import SimpleNamespace
from typing import cast

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecEnv

import scripts.evaluate_rl as evaluate_rl
from scripts.evaluate_rl import build_arg_parser, build_initial_rollout_series


def test_evaluate_rl_cli_accepts_dry_run_and_shared_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--schedule-time-s",
            "430.0",
            "--reward-preset",
            "basic",
            "--reward-discount",
            "0.95",
            "--device",
            "cuda",
        ]
    )

    assert args.dry_run is True
    assert args.schedule_time_s == 430.0
    assert args.reward_preset == "basic"
    assert args.reward_discount == 0.95
    assert args.device == "cuda"


def test_evaluate_rl_cli_rejects_removed_punctuality_dense_reward_option() -> None:
    with pytest.raises(SystemExit):
        _ = build_arg_parser().parse_args(["--plot-punctuality-dense-reward"])


def test_evaluate_rl_cli_rejects_survival_reward_scale() -> None:
    with pytest.raises(SystemExit):
        _ = build_arg_parser().parse_args(["--survival-reward-scale", "50"])


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


def _run_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "save_trajectory": True,
        "record_video": False,
        "plot_operation_time_series": False,
        "device": "cpu",
        "deterministic": True,
        "video_folder": "videos",
        "video_length": 12,
        "video_trigger_step": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self.closed = False

    def reset(self, *, seed: int | None = None, options=None):
        del seed, options
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}

    def close(self) -> None:
        self.closed = True


def test_run_evaluation_delegates_saved_rollout_to_canonical_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_env = _FakeEnv()
    fake_result = SimpleNamespace()
    calls: dict[str, object] = {}
    model_path = tmp_path / "run" / "policy_final.zip"
    model_path.parent.mkdir()
    model_path.touch()

    monkeypatch.setattr(
        evaluate_rl,
        "build_scenario",
        lambda schedule_time_s: ("vehicle", "track", "safety", "service"),
    )

    def build_env(**kwargs):
        calls["env_kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(evaluate_rl, "build_single_eval_env", build_env)
    monkeypatch.setattr(
        evaluate_rl.PPO,
        "load",
        lambda path, device: calls.setdefault("model_load", (path, device)) or "model",
    )

    def save_evaluation(*args, **kwargs):
        calls["save_args"] = args
        calls["save_kwargs"] = kwargs
        return fake_result, "trajectory.npz", "trajectory_metrics.json"

    def unexpected_direct_evaluation(*args, **kwargs):
        raise AssertionError("saved evaluation must use the canonical save helper")

    monkeypatch.setattr(evaluate_rl, "evaluate_and_save_final_policy", save_evaluation)
    monkeypatch.setattr(
        evaluate_rl, "evaluate_policy_once", unexpected_direct_evaluation
    )

    result, npz_path, json_path, trace = evaluate_rl._run_evaluation(
        _run_args(),
        load_dir=str(tmp_path / "run"),
        run_metadata={"run_mode": "reproduce"},
        schedule_time_s=465.0,
        reward_discount=0.998,
        step_distance=30.0,
        reward_preset=SimpleNamespace(name="basic", config="reward-config"),
        output_dir=str(tmp_path / "artifacts"),
    )

    assert result is fake_result
    assert (npz_path, json_path) == ("trajectory.npz", "trajectory_metrics.json")
    assert trace is None
    assert fake_env.closed is True
    assert calls["model_load"] == (
        str(tmp_path / "run" / "policy_final.zip"),
        "cpu",
    )
    env_kwargs = cast(dict[str, object], calls["env_kwargs"])
    assert env_kwargs["enable_trajectory_tracking"] is True
    save_kwargs = cast(dict[str, object], calls["save_kwargs"])
    assert save_kwargs["output_path"] == str(
        tmp_path / "artifacts" / "final_trajectory.npz"
    )
    assert save_kwargs["deterministic"] is True
    assert cast(dict[str, object], save_kwargs["metadata"])[
        "evaluation_load_dir"
    ] == str(tmp_path / "run")


def test_run_evaluation_without_saving_uses_canonical_rollout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_env = _FakeEnv()
    fake_result = SimpleNamespace()
    calls: dict[str, object] = {}
    model_path = tmp_path / "run" / "policy_final.zip"
    model_path.parent.mkdir()
    model_path.touch()

    monkeypatch.setattr(
        evaluate_rl,
        "build_scenario",
        lambda schedule_time_s: ("vehicle", "track", "safety", "service"),
    )

    def build_env(**kwargs):
        calls["env_kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(evaluate_rl, "build_single_eval_env", build_env)
    monkeypatch.setattr(evaluate_rl.PPO, "load", lambda path, device: "model")
    monkeypatch.setattr(
        evaluate_rl,
        "evaluate_and_save_final_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("save helper must not run when saving is disabled")
        ),
    )

    def evaluate_once(*args, **kwargs):
        calls["evaluate_kwargs"] = kwargs
        return fake_result

    monkeypatch.setattr(evaluate_rl, "evaluate_policy_once", evaluate_once)

    result, npz_path, json_path, trace = evaluate_rl._run_evaluation(
        _run_args(save_trajectory=False),
        load_dir=str(tmp_path / "run"),
        run_metadata={},
        schedule_time_s=465.0,
        reward_discount=0.998,
        step_distance=30.0,
        reward_preset=SimpleNamespace(name="basic", config="reward-config"),
        output_dir=str(tmp_path / "artifacts"),
    )

    assert result is fake_result
    assert (npz_path, json_path) == ("", "")
    assert trace is None
    assert fake_env.closed is True
    env_kwargs = cast(dict[str, object], calls["env_kwargs"])
    assert env_kwargs["enable_trajectory_tracking"] is False
    assert cast(dict[str, object], calls["evaluate_kwargs"])["deterministic"] is True


def test_operation_time_trace_records_state_without_parsing_info() -> None:
    class TraceEnv(_FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.state = SimpleNamespace(
                position_m=10.0,
                speed_mps=2.0,
                operation_time_s=0.0,
                redundant_operation_time_s=4.0,
            )

        def reset(self, *, seed: int | None = None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            self.state.position_m = 10.0
            self.state.operation_time_s = 0.0
            return observation, info

        def step(self, action):
            self.state.position_m = 20.0
            self.state.operation_time_s = 3.5
            self.state.redundant_operation_time_s = 1.5
            return super().step(action)

    env = TraceEnv()
    trace = evaluate_rl.OperationTimeTrace(env)
    _ = trace.reset()
    _ = trace.step(np.zeros(1, dtype=np.float32))

    assert trace.position_seq == [10.0, 20.0]
    assert trace.operation_time_seq == [0.0, 3.5]
    assert trace.redundant_operation_time_seq == [4.0, 1.5]
    trace.close()
    assert env.closed is True


def test_run_evaluation_configures_single_env_video_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_env = _FakeEnv()
    fake_result = SimpleNamespace()
    calls: dict[str, object] = {}
    model_path = tmp_path / "run" / "policy_final.zip"
    model_path.parent.mkdir()
    model_path.touch()

    class FakeRecordVideo:
        def __init__(self, env, **kwargs):
            self.env = env
            self.kwargs = kwargs

        def close(self) -> None:
            self.env.close()

    monkeypatch.setattr(
        evaluate_rl,
        "build_scenario",
        lambda schedule_time_s: ("vehicle", "track", "safety", "service"),
    )

    def build_env(**kwargs):
        calls["env_kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(evaluate_rl, "build_single_eval_env", build_env)
    monkeypatch.setattr(evaluate_rl.PPO, "load", lambda path, device: "model")
    monkeypatch.setattr(evaluate_rl, "RecordVideo", FakeRecordVideo)

    def evaluate_once(model, env, *, deterministic):
        calls["evaluation_env"] = env
        calls["deterministic"] = deterministic
        return fake_result

    monkeypatch.setattr(evaluate_rl, "evaluate_policy_once", evaluate_once)

    result, _, _, _ = evaluate_rl._run_evaluation(
        _run_args(save_trajectory=False, record_video=True),
        load_dir=str(tmp_path / "run"),
        run_metadata={},
        schedule_time_s=465.0,
        reward_discount=0.998,
        step_distance=30.0,
        reward_preset=SimpleNamespace(name="basic", config="reward-config"),
        output_dir=str(tmp_path / "artifacts"),
    )

    wrapped_env = cast(FakeRecordVideo, calls["evaluation_env"])
    assert result is fake_result
    assert wrapped_env.env is fake_env
    assert wrapped_env.kwargs["video_folder"] == "videos"
    assert wrapped_env.kwargs["video_length"] == 12
    assert wrapped_env.kwargs["step_trigger"](3) is True
    assert wrapped_env.kwargs["step_trigger"](2) is False
    assert calls["deterministic"] is True
    assert fake_env.closed is True
