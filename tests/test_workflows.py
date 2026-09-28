"""Tests for workflows.train and workflows.evaluate."""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import mtto
from mtto.domain.scenario import Task
from mtto.dp.solver import VariableSpacingDPOptimizer
from mtto.io.artifacts import (
    ArtifactError,
    RunKind,
    file_sha256,
    read_completed_run,
    read_diagnostics,
    read_evaluations,
    task_to_json,
    write_diagnostics,
    write_evaluations,
)
from mtto.rl.diagnostics import (
    EvaluationHistory,
    RewardDiagnostics,
    SafetyTruncationHistogram,
    TrainingDiagnostics,
)
from mtto.rl.observation import POLICY_IO_VERSION
from mtto.workflows.dp import DPConfig, solve_dp
from mtto.workflows.evaluate import EvaluateConfig, evaluate
from mtto.workflows.train import (
    Progress,
    TrainConfig,
    build_env_references,
    derive_training_budget,
    train,
)
from paper.figures import load_paper_scenario, load_paper_task


def _make_dummy_reward_diagnostics(
    n_rollouts: int = 2, n_episodes: int = 3
) -> RewardDiagnostics:
    reward_names = np.asarray(
        [
            "safety",
            "energy",
            "comfort",
            "terminal_stopping",
            "terminal_punctuality",
            "survival",
            "truncation",
            "punctuality_shaping",
            "total",
        ]
    )
    return RewardDiagnostics(
        schema_version=np.asarray([5], dtype=np.int16),
        reward_names=reward_names,
        rollout_end_step=np.arange(1, n_rollouts + 1, dtype=np.int64) * 16,
        rollout_transition_count=np.full(n_rollouts, 16, dtype=np.int64),
        rollout_reward_sum=np.zeros((n_rollouts, 9), dtype=np.float64),
        rollout_reward_abs_sum=np.zeros((n_rollouts, 9), dtype=np.float64),
        rollout_reward_nonzero_count=np.zeros((n_rollouts, 9), dtype=np.int64),
        rollout_reward_cross_product=np.zeros((n_rollouts, 9, 9), dtype=np.float64),
        episode_end_step=np.arange(1, n_episodes + 1, dtype=np.int64) * 10,
        episode_worker_rank=np.zeros(n_episodes, dtype=np.int16),
        episode_index=np.arange(n_episodes, dtype=np.int64),
        episode_length=np.full(n_episodes, 10, dtype=np.int32),
        episode_termination_reason=np.full(n_episodes, 1, dtype=np.int8),
        episode_complete=np.ones(n_episodes, dtype=np.bool_),
        episode_reward_sums=np.zeros((n_episodes, 9), dtype=np.float64),
    )


def _make_dummy_safety_histogram(n_bins: int = 4) -> SafetyTruncationHistogram:
    bin_size = 5000.0
    starts = np.arange(n_bins, dtype=np.float64) * bin_size
    return SafetyTruncationHistogram(
        bin_start_m=starts,
        bin_end_m=starts + bin_size,
        safety_truncation_count=np.ones(n_bins, dtype=np.int64),
        low_safety_truncation_count=np.zeros(n_bins, dtype=np.int64),
        high_safety_truncation_count=np.ones(n_bins, dtype=np.int64),
        global_safety_truncation_share=np.full(n_bins, 1.0 / n_bins, dtype=np.float64),
        position_bin_size_m=np.asarray([bin_size], dtype=np.float64),
    )


def _make_dummy_evaluation_history(n_evals: int = 2) -> EvaluationHistory:
    return EvaluationHistory(
        training_steps=np.arange(1, n_evals + 1, dtype=np.int64) * 100,
        rollout_indices=np.arange(1, n_evals + 1, dtype=np.int64),
        total_reward=np.zeros(n_evals, dtype=np.float64),
        episode_steps=np.full(n_evals, 50, dtype=np.int64),
        success=np.ones(n_evals, dtype=np.bool_),
        safe=np.ones(n_evals, dtype=np.bool_),
        feasible=np.ones(n_evals, dtype=np.bool_),
        stop_error_m=np.full(n_evals, 0.1, dtype=np.float64),
        time_error_s=np.full(n_evals, 0.5, dtype=np.float64),
        total_energy_j=np.full(n_evals, 1000.0, dtype=np.float64),
        comfort_tav=np.full(n_evals, 0.2, dtype=np.float64),
        completed_training_episodes=np.arange(1, n_evals + 1, dtype=np.int64) * 5,
        scheduled_completed_training_episodes=np.arange(1, n_evals + 1, dtype=np.int64)
        * 5,
        route_completion_ratio=np.ones(n_evals, dtype=np.float64),
        safety_violation_positions_m=np.empty(0, dtype=np.float64),
        safety_violation_position_offsets=np.zeros(n_evals + 1, dtype=np.int64),
    )


def test_train_workflow_environment_steps(tmp_path: Path) -> None:
    scenario = load_paper_scenario()
    task = load_paper_task()
    out_dir = tmp_path / "run_env_steps"
    config = TrainConfig(
        reward_preset="basic_safety_punctuality",
        step_distance_m=100.0,
        gamma=0.998,
        budget_mode="environment_steps",
        training_episodes=None,
        training_rollouts=2,
        num_envs=1,
        n_steps_per_env=512,
        evaluation_interval_rollouts=1,
        evaluation_interval_episodes=None,
        evaluation_deterministic=True,
        keep_best=True,
        safety_truncation_bin_size_m=5000.0,
        device="cpu",
        seed=42,
    )
    result = train(scenario, task, config, out_dir)

    # 1. read_completed_run 可读
    completed = read_completed_run(out_dir)

    # 2. 组成符合 rl_train（含 diagnostics.npz、evaluations.npz，keep_best 时含 best/）
    assert (out_dir / "run.json").is_file()
    assert (out_dir / "policy.zip").is_file()
    assert (out_dir / "profile.npz").is_file()
    assert (out_dir / "quality.json").is_file()
    assert (out_dir / "result.json").is_file()
    assert (out_dir / "diagnostics.npz").is_file()
    assert (out_dir / "evaluations.npz").is_file()
    assert (out_dir / "best").is_dir()
    assert (out_dir / "best" / "policy.zip").is_file()
    assert (out_dir / "best" / "profile.npz").is_file()
    assert (out_dir / "best" / "quality.json").is_file()
    assert (out_dir / "best" / "result.json").is_file()

    # 3. result.json.training 与预算一致
    assert completed.payload.result is not None
    assert completed.payload.result.training is not None
    t = completed.payload.result.training
    assert t.actual_training_timesteps == 1024
    assert t.actual_training_rollouts == 2
    assert t.target_reached is True
    assert t.stop_reason == "environment_step_target"

    # 4. run.json 字段检查
    assert completed.record.policy_io_version == POLICY_IO_VERSION
    assert completed.record.mtto_version == mtto.__version__
    assert completed.record.scenario_hash == scenario.scenario_hash
    assert completed.record.task == task_to_json(task)

    # 5. policy_sha256 与文件一致
    assert completed.payload.result.policy_sha256 == file_sha256(out_dir / "policy.zip")
    assert completed.payload.best is not None
    assert completed.payload.best.result is not None
    assert completed.payload.best.result.policy_sha256 == file_sha256(
        out_dir / "best" / "policy.zip"
    )

    # 6. 返回的 TrainResult 与读回的产物一致
    assert result.run_id == completed.record.run_id
    assert result.result == completed.payload.result
    assert result.quality.metrics == completed.payload.quality.metrics
    assert result.quality.safe == completed.payload.quality.safe
    assert result.quality.feasible == completed.payload.quality.feasible
    assert result.quality.completed == completed.payload.quality.completed
    assert result.quality.precise_stop == completed.payload.quality.precise_stop
    assert result.quality.punctual == completed.payload.quality.punctual
    np.testing.assert_array_equal(
        result.profile.position_m, completed.payload.profile.position_m
    )
    assert result.best is not None
    assert result.best.result == completed.payload.best.result


def test_train_workflow_completed_episodes(tmp_path: Path) -> None:
    scenario = load_paper_scenario()
    task = load_paper_task()
    out_dir = tmp_path / "run_episodes"
    config = TrainConfig(
        reward_preset="basic_safety_punctuality",
        step_distance_m=1000.0,
        gamma=0.998,
        budget_mode="completed_episodes",
        training_episodes=2,
        training_rollouts=None,
        num_envs=1,
        n_steps_per_env=512,
        evaluation_interval_rollouts=None,
        evaluation_interval_episodes=None,
        evaluation_deterministic=True,
        keep_best=False,
        safety_truncation_bin_size_m=5000.0,
        device="cpu",
        seed=123,
    )
    result = train(scenario, task, config, out_dir)

    completed = read_completed_run(out_dir)
    assert not (out_dir / "best").exists()
    assert result.best is None
    assert completed.payload.best is None
    assert completed.payload.result is not None
    assert completed.payload.result.training is not None
    t = completed.payload.result.training
    assert t.actual_completed_episodes >= 2
    assert t.target_reached is True
    assert t.stop_reason == "completed_episode_target"


def test_train_and_evaluate_fail_if_output_dir_exists(tmp_path: Path) -> None:
    scenario = load_paper_scenario()
    task = load_paper_task()
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    sentinel = existing_dir / "sentinel.txt"
    sentinel.write_text("protect me", encoding="utf-8")

    train_config = TrainConfig(
        reward_preset="basic_safety_punctuality",
        step_distance_m=100.0,
        gamma=0.998,
        budget_mode="environment_steps",
        training_episodes=None,
        training_rollouts=1,
        num_envs=1,
        n_steps_per_env=512,
        evaluation_interval_rollouts=None,
        evaluation_interval_episodes=None,
        evaluation_deterministic=True,
        keep_best=False,
        safety_truncation_bin_size_m=5000.0,
        device="cpu",
        seed=1,
    )
    with pytest.raises(FileExistsError):
        train(scenario, task, train_config, existing_dir)

    assert sentinel.read_text(encoding="utf-8") == "protect me"
    assert list(existing_dir.iterdir()) == [sentinel]

    eval_config = EvaluateConfig(use_best=False, deterministic=True, device="cpu")
    with pytest.raises(FileExistsError):
        evaluate(scenario, task, tmp_path / "nonexistent", eval_config, existing_dir)

    assert sentinel.read_text(encoding="utf-8") == "protect me"
    assert list(existing_dir.iterdir()) == [sentinel]


def test_train_and_evaluate_do_not_create_output_dir_on_validation_failure(
    tmp_path: Path,
) -> None:
    scenario = load_paper_scenario()
    task = load_paper_task()

    valid_train_config = TrainConfig(
        reward_preset="basic_safety_punctuality",
        step_distance_m=100.0,
        gamma=0.998,
        budget_mode="environment_steps",
        training_episodes=None,
        training_rollouts=1,
        num_envs=1,
        n_steps_per_env=512,
        evaluation_interval_rollouts=None,
        evaluation_interval_episodes=None,
        evaluation_deterministic=True,
        keep_best=False,
        safety_truncation_bin_size_m=5000.0,
        device="cpu",
        seed=42,
    )

    # 1. train: task 无 schedule_time_s 时校验失败，不创建输出目录
    fail_dir_1 = tmp_path / "train_fail_task"
    bad_task = replace(task, schedule_time_s=None)
    with pytest.raises(ValueError, match="schedule_time_s must not be None"):
        train(scenario, bad_task, valid_train_config, fail_dir_1)
    assert not fail_dir_1.exists()

    # 2. train: config 字段非法（num_envs <= 0）时校验失败，不创建输出目录
    fail_dir_2 = tmp_path / "train_fail_envs"
    bad_config = replace(valid_train_config, num_envs=0)
    with pytest.raises(ValueError, match="num_envs must be positive"):
        train(scenario, task, bad_config, fail_dir_2)
    assert not fail_dir_2.exists()

    # 3. train: budget_mode 未知时校验失败，不创建输出目录
    fail_dir_3 = tmp_path / "train_fail_budget_mode"
    bad_budget = replace(valid_train_config, budget_mode="invalid_mode")
    with pytest.raises(ValueError, match="Unknown budget_mode"):
        train(scenario, task, bad_budget, fail_dir_3)
    assert not fail_dir_3.exists()

    # 准备基础源运行以供 evaluate 校验测试
    source_train_dir = tmp_path / "source_train_for_eval_failure_tests"
    train(scenario, task, valid_train_config, source_train_dir)

    eval_config = EvaluateConfig(use_best=False, deterministic=True, device="cpu")

    # 4. evaluate: policy_io_version 不符时校验失败，不创建输出目录
    corrupt_dir = tmp_path / "corrupt_train_version"
    shutil.copytree(source_train_dir, corrupt_dir)
    run_json = corrupt_dir / "run.json"
    data = json.loads(run_json.read_text(encoding="utf-8"))
    data["policy_io_version"] = 999
    run_json.write_text(json.dumps(data), encoding="utf-8")

    fail_eval_dir_1 = tmp_path / "eval_fail_version"
    with pytest.raises(ValueError, match="Policy IO version mismatch"):
        evaluate(scenario, task, corrupt_dir, eval_config, fail_eval_dir_1)
    assert not fail_eval_dir_1.exists()

    # 5. evaluate: use_best=True 但源运行没有 best/ 时校验失败，不创建输出目录
    fail_eval_dir_2 = tmp_path / "eval_fail_no_best"
    best_eval_config = EvaluateConfig(use_best=True, deterministic=True, device="cpu")
    with pytest.raises(ValueError, match="does not contain a best policy"):
        evaluate(scenario, task, source_train_dir, best_eval_config, fail_eval_dir_2)
    assert not fail_eval_dir_2.exists()

    # 6. evaluate: 源运行不存在时校验失败，不创建输出目录
    fail_eval_dir_3 = tmp_path / "eval_fail_nonexistent_src"
    with pytest.raises((FileNotFoundError, ArtifactError)):
        evaluate(
            scenario,
            task,
            tmp_path / "nonexistent_source",
            eval_config,
            fail_eval_dir_3,
        )
    assert not fail_eval_dir_3.exists()


def test_evaluate_workflow(tmp_path: Path) -> None:
    scenario = load_paper_scenario()
    task = load_paper_task()
    train_dir = tmp_path / "source_train"
    train_config = TrainConfig(
        reward_preset="basic_safety_punctuality",
        step_distance_m=100.0,
        gamma=0.998,
        budget_mode="environment_steps",
        training_episodes=None,
        training_rollouts=2,
        num_envs=1,
        n_steps_per_env=512,
        evaluation_interval_rollouts=1,
        evaluation_interval_episodes=None,
        evaluation_deterministic=True,
        keep_best=True,
        safety_truncation_bin_size_m=5000.0,
        device="cpu",
        seed=42,
    )
    train_res = train(scenario, task, train_config, train_dir)

    # 1. 评估最终模型
    eval_dir = tmp_path / "eval_final"
    eval_config = EvaluateConfig(use_best=False, deterministic=True, device="cpu")
    eval_res = evaluate(scenario, task, train_dir, eval_config, eval_dir)

    # 产物符合 evaluation 组成、目录中无 policy.zip
    assert (eval_dir / "run.json").is_file()
    assert (eval_dir / "profile.npz").is_file()
    assert (eval_dir / "quality.json").is_file()
    assert (eval_dir / "result.json").is_file()
    assert not (eval_dir / "policy.zip").exists()

    # policy_run_id 为源运行的 run_id
    assert eval_res.result.policy_run_id == train_res.run_id
    assert eval_res.result.policy_sha256 == file_sha256(train_dir / "policy.zip")
    assert eval_res.result.training is None

    eval_completed = read_completed_run(eval_dir)
    assert eval_completed.record.kind == RunKind.EVALUATION
    assert eval_completed.record.config["source_run_id"] == train_res.run_id
    assert eval_completed.record.config["source_policy_path"] == "policy.zip"

    # 2. 评估最优模型
    eval_best_dir = tmp_path / "eval_best"
    eval_best_config = EvaluateConfig(use_best=True, deterministic=True, device="cpu")
    eval_best_res = evaluate(scenario, task, train_dir, eval_best_config, eval_best_dir)
    assert eval_best_res.result.policy_run_id == train_res.run_id
    assert eval_best_res.result.policy_sha256 == file_sha256(
        train_dir / "best" / "policy.zip"
    )

    # 3. policy_io_version 不一致时报错
    corrupt_dir = tmp_path / "corrupt_train"
    shutil.copytree(train_dir, corrupt_dir)
    corrupt_run_json = corrupt_dir / "run.json"
    data = json.loads(corrupt_run_json.read_text(encoding="utf-8"))
    data["policy_io_version"] = 999
    corrupt_run_json.write_text(json.dumps(data), encoding="utf-8")

    fail_ver_dir = tmp_path / "eval_fail_ver"
    with pytest.raises(ValueError, match="Policy IO version mismatch"):
        evaluate(scenario, task, corrupt_dir, eval_config, fail_ver_dir)
    assert not fail_ver_dir.exists()

    # 4. 换一个任务（如不同到站时间）评估
    alt_task = load_paper_task(schedule_time_s=480.0)
    eval_alt_dir = tmp_path / "eval_alt"
    eval_alt_res = evaluate(scenario, alt_task, train_dir, eval_config, eval_alt_dir)
    eval_alt_completed = read_completed_run(eval_alt_dir)
    assert eval_alt_completed.record.task == task_to_json(alt_task)
    assert eval_alt_res.profile is not None


def test_train_progress_callback(tmp_path: Path) -> None:
    scenario = load_paper_scenario()
    task = load_paper_task()
    out_dir = tmp_path / "run_progress"
    config = TrainConfig(
        reward_preset="basic_safety_punctuality",
        step_distance_m=100.0,
        gamma=0.998,
        budget_mode="environment_steps",
        training_episodes=None,
        training_rollouts=3,
        num_envs=1,
        n_steps_per_env=512,
        evaluation_interval_rollouts=None,
        evaluation_interval_episodes=None,
        evaluation_deterministic=True,
        keep_best=False,
        safety_truncation_bin_size_m=5000.0,
        device="cpu",
        seed=1,
    )
    records: list[Progress] = []
    res = train(scenario, task, config, out_dir, progress=records.append)

    assert len(records) == 3
    assert res.result.training is not None
    assert (
        records[-1].training_timesteps == res.result.training.actual_training_timesteps
    )
    assert records[-1].total_timesteps == 1536


def test_diagnostics_roundtrip_and_strict_checks(tmp_path: Path) -> None:
    diag = TrainingDiagnostics(
        reward=_make_dummy_reward_diagnostics(),
        safety=_make_dummy_safety_histogram(),
    )
    path = tmp_path / "diagnostics.npz"
    write_diagnostics(path, diag)
    loaded = read_diagnostics(path)

    np.testing.assert_array_equal(
        loaded.reward.schema_version, diag.reward.schema_version
    )
    np.testing.assert_array_equal(
        loaded.reward.rollout_end_step, diag.reward.rollout_end_step
    )
    np.testing.assert_array_equal(loaded.safety.bin_start_m, diag.safety.bin_start_m)

    # 缺键
    with np.load(path) as d:
        raw = dict(d)
    missing_path = tmp_path / "missing.npz"
    bad_data = {k: v for k, v in raw.items() if k != "schema_version"}
    np.savez(missing_path, **bad_data)
    with pytest.raises(ArtifactError, match="missing keys"):
        read_diagnostics(missing_path)

    # 多键
    extra_path = tmp_path / "extra.npz"
    np.savez(extra_path, **raw, unexpected_array=np.zeros(3))
    with pytest.raises(ArtifactError, match="extra keys"):
        read_diagnostics(extra_path)

    # 类型不符
    type_path = tmp_path / "bad_type.npz"
    bad_type_data = dict(raw)
    bad_type_data["schema_version"] = np.asarray([5.0], dtype=np.float64)
    np.savez(type_path, **bad_type_data)
    with pytest.raises(ArtifactError, match="schema_version must have dtype int16"):
        read_diagnostics(type_path)


def test_evaluations_roundtrip_and_strict_checks(tmp_path: Path) -> None:
    history = _make_dummy_evaluation_history(n_evals=3)
    path = tmp_path / "evaluations.npz"
    write_evaluations(path, history)
    loaded = read_evaluations(path)

    np.testing.assert_array_equal(loaded.training_steps, history.training_steps)
    np.testing.assert_array_equal(loaded.total_reward, history.total_reward)

    with np.load(path) as d:
        raw = dict(d)

    # 缺键
    missing_path = tmp_path / "eval_missing.npz"
    bad_data = {k: v for k, v in raw.items() if k != "total_reward"}
    np.savez(missing_path, **bad_data)
    with pytest.raises(ArtifactError, match="missing keys"):
        read_evaluations(missing_path)

    # 多键
    extra_path = tmp_path / "eval_extra.npz"
    np.savez(extra_path, **raw, extra_metric=np.zeros(2))
    with pytest.raises(ArtifactError, match="extra keys"):
        read_evaluations(extra_path)

    # 类型不符
    type_path = tmp_path / "eval_bad_type.npz"
    bad_type = dict(raw)
    bad_type["training_steps"] = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    np.savez(type_path, **bad_type)
    with pytest.raises(ArtifactError, match="dtype"):
        read_evaluations(type_path)


@pytest.mark.parametrize(
    ("route_dist", "episodes", "num_envs", "step_dist", "rollout_steps"),
    [
        (29845.0, 7000, 8, 30.0, 8192),
        (29845.0, 7001, 8, 30.0, 8192),
        (29845.0, 5000, 8, 30.0, 8192),
        (29845.0, 7000, 8, 10.0, 8192),
        (29845.0, 7000, 8, 100.0, 8192),
        (10000.0, 100, 4, 25.0, 1024),
    ],
)
def test_derive_training_budget_equivalence(
    route_dist: float,
    episodes: int,
    num_envs: int,
    step_dist: float,
    rollout_steps: int,
) -> None:
    effective, max_steps, total_timesteps = derive_training_budget(
        route_distance_m=route_dist,
        training_episodes=episodes,
        num_envs=num_envs,
        step_distance_m=step_dist,
        rollout_steps_per_update=rollout_steps,
    )
    expected_effective = math.ceil(episodes / num_envs) * num_envs
    expected_max_steps = math.ceil(route_dist / step_dist)
    expected_total = (
        math.ceil(expected_effective * expected_max_steps / rollout_steps)
        * rollout_steps
    )
    assert effective == expected_effective
    assert max_steps == expected_max_steps
    assert total_timesteps == expected_total


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"route_distance_m": 0.0},
            "route_distance_m must be finite and positive",
        ),
        (
            {"route_distance_m": float("nan")},
            "route_distance_m must be finite and positive",
        ),
        ({"training_episodes": 0}, "training_episodes must be positive"),
        ({"num_envs": 0}, "num_envs must be positive"),
        (
            {"step_distance_m": 0.0},
            "step_distance_m must be finite and positive",
        ),
        ({"rollout_steps_per_update": 0}, "rollout_steps_per_update must be positive"),
    ],
)
def test_derive_training_budget_rejects_invalid_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    base_kwargs: dict[str, float | int] = {
        "route_distance_m": 29845.0,
        "training_episodes": 7000,
        "num_envs": 8,
        "step_distance_m": 30.0,
        "rollout_steps_per_update": 8192,
    }
    with pytest.raises(ValueError, match=match):
        derive_training_budget(**{**base_kwargs, **kwargs})


def test_build_env_references_consistency() -> None:
    scenario = load_paper_scenario()
    task = load_paper_task()
    lookup, normalization = build_env_references(scenario, task, 30.0)
    assert normalization.max_energy_consumption_kj > 0.0
    assert normalization.initial_min_operation_time_s > 0.0
    assert normalization.required_episode_steps == math.ceil(
        (task.target_position_m - task.start_position_m) / 30.0
    )
    assert lookup.pos_min_m == task.start_position_m
    assert lookup.speed_mps.size > 0


def test_solve_dp_workflow(tmp_path: Path) -> None:
    scenario = load_paper_scenario()
    task = Task(
        start_position_m=135.0,
        target_position_m=335.0,
        schedule_time_s=40.0,
        max_acc_change=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )
    config = DPConfig(
        delta_speed=1.0,
        stage_division="uniform",
        uniform_step_size=50.0,
        sub_stage_count=30,
        max_outer_iterations=100,
        precompute_mode="serial",
        precompute_workers=None,
        precompute_chunk_size=None,
    )
    out_dir = tmp_path / "dp_run"
    dp_res = solve_dp(scenario, task, config, out_dir, cache_dir=None)

    # 1. 产物目录含 run.json、profile.npz、quality.json，
    #    不含 policy.zip、result.json
    assert (out_dir / "run.json").is_file()
    assert (out_dir / "profile.npz").is_file()
    assert (out_dir / "quality.json").is_file()
    assert not (out_dir / "policy.zip").exists()
    assert not (out_dir / "result.json").exists()

    # 2. read_completed_run 严格读取
    completed = read_completed_run(out_dir)
    assert completed.record.kind == RunKind.DP_SOLVE
    assert completed.record.policy_io_version is None
    assert completed.record.scenario_hash == scenario.scenario_hash
    assert completed.record.task == task_to_json(task)
    assert completed.record.config == dataclasses.asdict(config)
    assert completed.payload.result is None

    # 3. 与直接调用 VariableSpacingDPOptimizer.optimize 结果逐元素相等
    optimizer = VariableSpacingDPOptimizer(
        scenario=scenario,
        task=task,
        cache_dir=None,
        delta_speed=1.0,
        uniform_step_size=50.0,
        precompute_mode="serial",
        show_precompute_progress=False,
    )
    direct_profile = optimizer.optimize(135.0, 0.0, 335.0, 0.0, 40.0)
    assert direct_profile is not None
    np.testing.assert_array_equal(dp_res.profile.position_m, direct_profile.position_m)
    np.testing.assert_array_equal(dp_res.profile.speed_mps, direct_profile.speed_mps)
    np.testing.assert_array_equal(dp_res.profile.time_s, direct_profile.time_s)
    np.testing.assert_array_equal(
        dp_res.profile.propulsion_energy_kj, direct_profile.propulsion_energy_kj
    )
    np.testing.assert_array_equal(
        dp_res.profile.levitation_energy_kj, direct_profile.levitation_energy_kj
    )


def test_solve_dp_directory_already_exists_and_validation_failure(
    tmp_path: Path,
) -> None:
    scenario = load_paper_scenario()
    task = Task(
        start_position_m=135.0,
        target_position_m=335.0,
        schedule_time_s=40.0,
        max_acc_change=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )
    config = DPConfig(
        delta_speed=1.0,
        stage_division="uniform",
        uniform_step_size=50.0,
        sub_stage_count=30,
        max_outer_iterations=100,
        precompute_mode="serial",
        precompute_workers=None,
        precompute_chunk_size=None,
    )

    # 1. 输出目录已存在时报错且不写入
    exists_dir = tmp_path / "already_exists"
    exists_dir.mkdir()
    sentinel_file = exists_dir / "keep.txt"
    sentinel_file.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        solve_dp(scenario, task, config, exists_dir)
    assert sentinel_file.read_text(encoding="utf-8") == "sentinel"
    assert not (exists_dir / "run.json").exists()

    # 2. schedule_time_s 为 None 时校验报错且不留下输出目录
    bad_task = replace(task, schedule_time_s=None)
    fail_dir = tmp_path / "fail_no_schedule"
    with pytest.raises(ValueError, match="schedule_time_s must not be None"):
        solve_dp(scenario, bad_task, config, fail_dir)
    assert not fail_dir.exists()


def test_solve_dp_cache_dir_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = load_paper_scenario()
    task = Task(
        start_position_m=135.0,
        target_position_m=335.0,
        schedule_time_s=40.0,
        max_acc_change=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )
    config = DPConfig(
        delta_speed=1.0,
        stage_division="uniform",
        uniform_step_size=50.0,
        sub_stage_count=30,
        max_outer_iterations=100,
        precompute_mode="serial",
        precompute_workers=None,
        precompute_chunk_size=None,
    )

    # 1. cache_dir=None 时不产生任何缓存文件
    out_dir_nocache = tmp_path / "out_nocache"
    solve_dp(scenario, task, config, out_dir_nocache, cache_dir=None)
    assert [p.name for p in tmp_path.iterdir()] == ["out_nocache"]

    # 2. 给定 cache_dir 时第一次写出缓存
    cache_dir = tmp_path / "dp_cache"
    out_dir_1 = tmp_path / "out_run1"
    res1 = solve_dp(scenario, task, config, out_dir_1, cache_dir=cache_dir)

    cache_subdirs = list(cache_dir.iterdir())
    assert len(cache_subdirs) == 1
    cache_subdir = cache_subdirs[0]
    assert (cache_subdir / "graph_data.pkl.gz").is_file()
    assert (cache_subdir / "metadata.json").is_file()

    # 验证缓存子目录名与 6a 后的命名规则一致 (v3_uni50p0_...)
    assert cache_subdir.name.startswith("v3_uni50p0_")

    # 3. 第二次求解（新优化器实例、新输出目录）从缓存读取
    def _forbidden_build_transition_graph(*args, **kwargs):
        raise AssertionError(
            "build_transition_graph should not be called when cache hits"
        )

    monkeypatch.setattr(
        "mtto.dp.solver.build_transition_graph",
        _forbidden_build_transition_graph,
    )

    out_dir_2 = tmp_path / "out_run2"
    res2 = solve_dp(scenario, task, config, out_dir_2, cache_dir=cache_dir)

    # 两次 profile 逐元素相等
    np.testing.assert_array_equal(res1.profile.position_m, res2.profile.position_m)
    np.testing.assert_array_equal(res1.profile.speed_mps, res2.profile.speed_mps)
    np.testing.assert_array_equal(res1.profile.time_s, res2.profile.time_s)
    np.testing.assert_array_equal(
        res1.profile.propulsion_energy_kj, res2.profile.propulsion_energy_kj
    )
    np.testing.assert_array_equal(
        res1.profile.levitation_energy_kj, res2.profile.levitation_energy_kj
    )


@pytest.mark.slow
def test_dp_serial_vs_parallel_precompute_equivalence(tmp_path: Path) -> None:
    scenario = load_paper_scenario()
    task = Task(
        start_position_m=135.0,
        target_position_m=335.0,
        schedule_time_s=40.0,
        max_acc_change=0.75,
        max_stop_error_m=0.3,
        max_arr_time_error_s=10.0,
    )
    config_serial = DPConfig(
        delta_speed=1.0,
        stage_division="uniform",
        uniform_step_size=50.0,
        sub_stage_count=30,
        max_outer_iterations=100,
        precompute_mode="serial",
        precompute_workers=None,
        precompute_chunk_size=None,
    )
    config_parallel = DPConfig(
        delta_speed=1.0,
        stage_division="uniform",
        uniform_step_size=50.0,
        sub_stage_count=30,
        max_outer_iterations=100,
        precompute_mode="parallel",
        precompute_workers=2,
        precompute_chunk_size=1,
    )

    res_serial = solve_dp(
        scenario, task, config_serial, tmp_path / "out_serial", cache_dir=None
    )
    res_parallel = solve_dp(
        scenario, task, config_parallel, tmp_path / "out_parallel", cache_dir=None
    )

    np.testing.assert_array_equal(
        res_serial.profile.position_m, res_parallel.profile.position_m
    )
    np.testing.assert_array_equal(
        res_serial.profile.speed_mps, res_parallel.profile.speed_mps
    )
    np.testing.assert_array_equal(
        res_serial.profile.time_s, res_parallel.profile.time_s
    )
    np.testing.assert_array_equal(
        res_serial.profile.propulsion_energy_kj,
        res_parallel.profile.propulsion_energy_kj,
    )
    np.testing.assert_array_equal(
        res_serial.profile.levitation_energy_kj,
        res_parallel.profile.levitation_energy_kj,
    )

    opt_serial = VariableSpacingDPOptimizer(
        scenario=scenario,
        task=task,
        cache_dir=None,
        delta_speed=1.0,
        uniform_step_size=50.0,
        precompute_mode="serial",
        show_precompute_progress=False,
    )
    opt_parallel = VariableSpacingDPOptimizer(
        scenario=scenario,
        task=task,
        cache_dir=None,
        delta_speed=1.0,
        uniform_step_size=50.0,
        precompute_mode="parallel",
        precompute_workers=2,
        precompute_chunk_size=1,
        show_precompute_progress=False,
    )
    graph_serial = opt_serial._prepare_transition_graph_cache(135.0, 335.0)
    graph_parallel = opt_parallel._prepare_transition_graph_cache(135.0, 335.0)

    assert graph_serial["total_valid_edges"] == graph_parallel["total_valid_edges"]
    assert graph_serial["total_valid_edges"] > 0
