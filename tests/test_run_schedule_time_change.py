import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.run_schedule_time_change import (
    DEFAULT_DELTA_TIMES_S,
    DEFAULT_EVALUATE_LOAD_DIR,
    DEFAULT_OUTPUT_DIR,
    SUMMARY_FILENAME,
    _as_batch_observation,
    build_arg_parser,
    build_schedule_change_case,
    resolve_schedule_change_experiment_dir,
    should_trigger_schedule_change,
)


def test_as_batch_observation_preserves_environment_normalized_values() -> None:
    observation = _as_batch_observation([0.25, -0.5, 1.0])

    assert observation.dtype == np.float32
    assert observation.shape == (1, 3)
    np.testing.assert_allclose(observation, [[0.25, -0.5, 1.0]])


def test_cli_evaluate_and_show_load_dir_defaults_are_mode_specific() -> None:
    parser = build_arg_parser()

    evaluate_args = parser.parse_args(["evaluate"])
    show_args = parser.parse_args(["show"])

    assert evaluate_args.mode == "evaluate"
    assert evaluate_args.load_dir == DEFAULT_EVALUATE_LOAD_DIR
    assert evaluate_args.output_dir == DEFAULT_OUTPUT_DIR
    assert evaluate_args.delta_times_s == DEFAULT_DELTA_TIMES_S

    assert show_args.mode == "show"
    assert show_args.load_dir == DEFAULT_OUTPUT_DIR
    assert not hasattr(show_args, "output_dir")


def test_cli_parses_custom_delta_times() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(["evaluate", "--delta-times-s", "0,-5, 7.5"])

    assert args.delta_times_s == (0.0, -5.0, 7.5)


def test_schedule_change_case_labels_and_tokens() -> None:
    assert build_schedule_change_case(0.0).label == "Original"
    assert build_schedule_change_case(10.0).label == "Plus 10s"
    assert build_schedule_change_case(-20.0).label == "Minus 20s"
    assert build_schedule_change_case(10.0).token == "plus_10p0s"
    assert build_schedule_change_case(-20.0).token == "minus_20p0s"


def test_should_trigger_schedule_change_once_when_crossing_forward() -> None:
    assert (
        should_trigger_schedule_change(
            previous_position_m=700.0,
            current_position_m=850.0,
            change_distance_m=800.0,
            direction=1,
            already_triggered=False,
            delta_time_s=10.0,
        )
        is True
    )
    assert (
        should_trigger_schedule_change(
            previous_position_m=700.0,
            current_position_m=850.0,
            change_distance_m=800.0,
            direction=1,
            already_triggered=True,
            delta_time_s=10.0,
        )
        is False
    )


def test_should_not_trigger_original_case() -> None:
    assert (
        should_trigger_schedule_change(
            previous_position_m=700.0,
            current_position_m=850.0,
            change_distance_m=800.0,
            direction=1,
            already_triggered=False,
            delta_time_s=0.0,
        )
        is False
    )


def test_should_trigger_schedule_change_when_crossing_backward() -> None:
    assert (
        should_trigger_schedule_change(
            previous_position_m=850.0,
            current_position_m=700.0,
            change_distance_m=800.0,
            direction=-1,
            already_triggered=False,
            delta_time_s=-10.0,
        )
        is True
    )


def _write_summary(path: Path) -> None:
    path.mkdir(parents=True)
    _ = (path / SUMMARY_FILENAME).write_text(
        json.dumps({"cases": []}),
        encoding="utf-8",
    )


def test_resolve_experiment_dir_accepts_direct_experiment(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "20260101_000000"
    _write_summary(experiment_dir)

    assert resolve_schedule_change_experiment_dir(experiment_dir) == experiment_dir


def test_resolve_experiment_dir_selects_latest_child(tmp_path: Path) -> None:
    old_dir = tmp_path / "20260101_000000"
    new_dir = tmp_path / "20260102_000000"
    _write_summary(old_dir)
    _write_summary(new_dir)
    os.utime(old_dir, (1, 1))
    os.utime(new_dir, (2, 2))

    assert resolve_schedule_change_experiment_dir(tmp_path) == new_dir


def test_resolve_experiment_dir_errors_without_summary(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=SUMMARY_FILENAME):
        _ = resolve_schedule_change_experiment_dir(tmp_path)
