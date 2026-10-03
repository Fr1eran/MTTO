from pathlib import Path

import pytest

from mtto.cli import main
from mtto.io.artifacts import RunKind, read_completed_run


@pytest.mark.parametrize(
    ("cmd", "expected_args"),
    [
        ([], ["train", "evaluate", "dp", "analyze-training"]),
        (
            ["train"],
            [
                "--scenario",
                "--line-dir",
                "--tasks",
                "--task",
                "--schedule-time",
                "--config",
                "--output-dir",
                "--tensorboard-dir",
                "--run-id",
            ],
        ),
        (
            ["evaluate"],
            [
                "--scenario",
                "--line-dir",
                "--tasks",
                "--task",
                "--schedule-time",
                "--policy-run",
                "--use-best",
                "--stochastic",
                "--device",
                "--output-dir",
                "--run-id",
            ],
        ),
        (
            ["dp"],
            [
                "--scenario",
                "--line-dir",
                "--tasks",
                "--task",
                "--schedule-time",
                "--config",
                "--output-dir",
                "--cache-dir",
                "--run-id",
            ],
        ),
        (
            ["analyze-training"],
            [
                "--train-run",
                "--tensorboard-root",
                "--tensorboard-run",
                "--run-name",
                "--output-root",
                "--config",
            ],
        ),
    ],
)
def test_cli_help(
    cmd: list[str],
    expected_args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([*cmd, "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    for arg_name in expected_args:
        assert arg_name in captured.out


def test_cli_dp_e2e(tmp_path: Path) -> None:
    tasks_content = """
[small_task]
start_position_m = 135.0
target_position_m = 335.0
schedule_time_s = 40.0
max_jerk_mps3 = 0.75
max_stop_error_m = 0.3
max_arr_time_error_s = 10.0
"""
    tasks_file = tmp_path / "tasks.toml"
    tasks_file.write_text(tasks_content, encoding="utf-8")

    dp_content = """
[dp]
delta_speed = 1.0
stage_division = "uniform"
uniform_step_size = 50.0
sub_stage_count = 1
max_outer_iterations = 1
precompute_mode = "serial"
"""
    dp_config_file = tmp_path / "dp_config.toml"
    dp_config_file.write_text(dp_content, encoding="utf-8")

    out_dir = tmp_path / "dp_out"

    ret = main(
        [
            "dp",
            "--scenario",
            "paper/specs/scenario.toml",
            "--line-dir",
            "paper/data/line",
            "--tasks",
            str(tasks_file),
            "--task",
            "small_task",
            "--config",
            str(dp_config_file),
            "--output-dir",
            str(out_dir),
            "--run-id",
            "test-dp-run-123",
        ]
    )
    assert ret == 0

    completed = read_completed_run(out_dir)
    assert completed.record.kind == RunKind.DP_SOLVE
    assert completed.record.run_id == "test-dp-run-123"
    assert completed.payload.profile is not None
    assert completed.payload.quality is not None


def test_cli_train_eval_analyze_e2e(tmp_path: Path) -> None:
    tasks_content = """
[small_task]
start_position_m = 135.0
target_position_m = 335.0
schedule_time_s = 40.0
max_jerk_mps3 = 0.75
max_stop_error_m = 0.3
max_arr_time_error_s = 10.0
"""
    tasks_file = tmp_path / "tasks.toml"
    tasks_file.write_text(tasks_content, encoding="utf-8")

    train_content = """
[train]
reward_preset = "basic"
step_time_s = 1.0
gamma = 0.99
budget_mode = "environment_steps"
training_rollouts = 1
num_envs = 1
n_steps_per_env = 512
evaluation_deterministic = true
keep_best = true
safety_truncation_bin_size_m = 50.0
device = "cpu"
seed = 42
"""
    train_config_file = tmp_path / "train_config.toml"
    train_config_file.write_text(train_content, encoding="utf-8")

    # 1. train
    train_out = tmp_path / "train_out"
    ret_train = main(
        [
            "train",
            "--scenario",
            "paper/specs/scenario.toml",
            "--line-dir",
            "paper/data/line",
            "--tasks",
            str(tasks_file),
            "--task",
            "small_task",
            "--config",
            str(train_config_file),
            "--output-dir",
            str(train_out),
            "--run-id",
            "test-train-run",
        ]
    )
    assert ret_train == 0
    completed_train = read_completed_run(train_out)
    assert completed_train.record.kind == RunKind.RL_TRAIN
    assert completed_train.record.run_id == "test-train-run"

    # 2. evaluate
    eval_out = tmp_path / "eval_out"
    ret_eval = main(
        [
            "evaluate",
            "--scenario",
            "paper/specs/scenario.toml",
            "--line-dir",
            "paper/data/line",
            "--tasks",
            str(tasks_file),
            "--task",
            "small_task",
            "--policy-run",
            str(train_out),
            "--output-dir",
            str(eval_out),
            "--run-id",
            "test-eval-run",
        ]
    )
    assert ret_eval == 0
    completed_eval = read_completed_run(eval_out)
    assert completed_eval.record.kind == RunKind.EVALUATION
    assert completed_eval.record.run_id == "test-eval-run"

    # 3. analyze-training
    analysis_root = tmp_path / "analysis_out"
    ret_analysis = main(
        [
            "analyze-training",
            "--train-run",
            str(train_out),
            "--output-root",
            str(analysis_root),
            "--config",
            "paper/specs/analysis.toml",
        ]
    )
    assert ret_analysis == 0
    report_md = analysis_root / train_out.name / "report.md"
    assert report_md.exists()
    assert report_md.read_text(encoding="utf-8") != ""


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tasks_content = """
[small_task]
start_position_m = 135.0
target_position_m = 335.0
schedule_time_s = 40.0
max_jerk_mps3 = 0.75
max_stop_error_m = 0.3
max_arr_time_error_s = 10.0
"""
    tasks_file = tmp_path / "tasks.toml"
    tasks_file.write_text(tasks_content, encoding="utf-8")

    dp_content = """
[dp]
delta_speed = 1.0
stage_division = "uniform"
uniform_step_size = 50.0
sub_stage_count = 1
max_outer_iterations = 1
precompute_mode = "serial"
"""
    dp_config_file = tmp_path / "dp_config.toml"
    dp_config_file.write_text(dp_content, encoding="utf-8")

    # 1. Output directory already exists -> exit code 1
    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir()
    ret = main(
        [
            "dp",
            "--scenario",
            "paper/specs/scenario.toml",
            "--line-dir",
            "paper/data/line",
            "--tasks",
            str(tasks_file),
            "--task",
            "small_task",
            "--config",
            str(dp_config_file),
            "--output-dir",
            str(existing_dir),
        ]
    )
    assert ret == 1
    _, err = capsys.readouterr()
    assert "Error:" in err
    assert "already exists" in err

    # 2. Task name does not exist -> exit code 1
    ret = main(
        [
            "dp",
            "--scenario",
            "paper/specs/scenario.toml",
            "--line-dir",
            "paper/data/line",
            "--tasks",
            str(tasks_file),
            "--task",
            "nonexistent_task",
            "--config",
            str(dp_config_file),
            "--output-dir",
            str(tmp_path / "nonexistent_task_out"),
        ]
    )
    assert ret == 1
    _, err = capsys.readouterr()
    assert "Error:" in err
    assert "nonexistent_task" in err
    assert "small_task" in err

    # 3. Config TOML missing required key -> exit code 1
    missing_key_content = """
[dp]
delta_speed = 1.0
stage_division = "uniform"
sub_stage_count = 1
max_outer_iterations = 1
precompute_mode = "serial"
"""
    missing_key_file = tmp_path / "missing_key.toml"
    missing_key_file.write_text(missing_key_content, encoding="utf-8")
    ret = main(
        [
            "dp",
            "--scenario",
            "paper/specs/scenario.toml",
            "--line-dir",
            "paper/data/line",
            "--tasks",
            str(tasks_file),
            "--task",
            "small_task",
            "--config",
            str(missing_key_file),
            "--output-dir",
            str(tmp_path / "missing_key_out"),
        ]
    )
    assert ret == 1
    _, err = capsys.readouterr()
    assert "Error:" in err
    assert "uniform_step_size" in err

    # 4. Config TOML extra key -> exit code 1
    extra_key_content = """
[dp]
delta_speed = 1.0
stage_division = "uniform"
uniform_step_size = 50.0
sub_stage_count = 1
max_outer_iterations = 1
precompute_mode = "serial"
unexpected_key = 123
"""
    extra_key_file = tmp_path / "extra_key.toml"
    extra_key_file.write_text(extra_key_content, encoding="utf-8")
    ret = main(
        [
            "dp",
            "--scenario",
            "paper/specs/scenario.toml",
            "--line-dir",
            "paper/data/line",
            "--tasks",
            str(tasks_file),
            "--task",
            "small_task",
            "--config",
            str(extra_key_file),
            "--output-dir",
            str(tmp_path / "extra_key_out"),
        ]
    )
    assert ret == 1
    _, err = capsys.readouterr()
    assert "Error:" in err
    assert "unexpected_key" in err

    # 5. Train config without a positive control period -> exit code 1
    train_content = """
[train]
reward_preset = "basic"
gamma = 0.99
budget_mode = "environment_steps"
training_rollouts = 1
num_envs = 1
n_steps_per_env = 512
evaluation_deterministic = true
keep_best = true
safety_truncation_bin_size_m = 50.0
device = "cpu"
"""
    for step_line, message in (
        ("", "Missing required config key in [train]: step_time_s"),
        ("step_time_s = 0.0\n", "step_time_s must be finite and positive"),
    ):
        train_config_file = tmp_path / "train_step_time.toml"
        train_config_file.write_text(train_content + step_line, encoding="utf-8")
        output_dir = tmp_path / "train_step_time_out"
        ret = main(
            [
                "train",
                "--scenario",
                "paper/specs/scenario.toml",
                "--line-dir",
                "paper/data/line",
                "--tasks",
                str(tasks_file),
                "--task",
                "small_task",
                "--config",
                str(train_config_file),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert ret == 1
        _, err = capsys.readouterr()
        assert message in err
        assert not output_dir.exists()


def test_cli_analyze_training_requires_config(tmp_path: Path) -> None:
    # --config is required (gate-3 ruling C): AnalysisConfig carries no
    # library defaults any more, argparse must reject a missing --config.
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "analyze-training",
                "--train-run",
                str(tmp_path / "train"),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
    assert exc_info.value.code == 2
