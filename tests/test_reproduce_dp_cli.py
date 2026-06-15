import os
from argparse import Namespace
from types import SimpleNamespace

import json
import pytest

from scripts.reproduce_dp import (
    _build_cli_parser,
    _resolve_output_dir,
    _run_optimization,
    _validate_cli_args,
    main,
)
from utils.io_utils import format_float_token

# ---- CLI defaults ----


def test_reproduce_dp_cli_defaults() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([])

    assert args.output_root == "output/optimal/dp"
    assert args.schedule_time_s == pytest.approx(430.0)
    assert args.delta_speed_mps == pytest.approx(0.1)
    assert args.max_outer_iterations == 100
    assert args.precompute_mode == "parallel"
    assert args.precompute_workers == 4
    assert args.stage_division == "variable"
    assert args.uniform_step_size == pytest.approx(100.0)
    assert args.sub_stage_count == 30
    assert args.skip_disk_cache is False


# ---- CLI explicit args ----


def test_reproduce_dp_cli_accepts_explicit_args() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([
        "--output-root",
        "output/custom",
        "--schedule-time-s",
        "500.5",
        "--delta-speed-mps",
        "0.05",
        "--max-outer-iterations",
        "50",
        "--precompute-mode",
        "parallel",
        "--precompute-workers",
        "4",
    ])

    assert args.output_root == "output/custom"
    assert args.schedule_time_s == pytest.approx(500.5)
    assert args.delta_speed_mps == pytest.approx(0.05)
    assert args.max_outer_iterations == 50
    assert args.precompute_mode == "parallel"
    assert args.precompute_workers == 4


def test_reproduce_dp_cli_stage_division_uniform() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([
        "--stage-division",
        "uniform",
        "--uniform-step-size",
        "25.0",
        "--sub-stage-count",
        "50",
        "--skip-disk-cache",
    ])

    assert args.stage_division == "uniform"
    assert args.uniform_step_size == pytest.approx(25.0)
    assert args.sub_stage_count == 50
    assert args.skip_disk_cache is True


def test_reproduce_dp_cli_stage_division_variable() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([
        "--stage-division",
        "variable",
        "--sub-stage-count",
        "20",
    ])

    assert args.stage_division == "variable"
    assert args.sub_stage_count == 20
    # uniform_step_size retains its default even when unused
    assert args.uniform_step_size == pytest.approx(100.0)


# ---- _format_float_token ----


@pytest.mark.parametrize(
    "value,expected",
    [
        (440.0, "440p0"),
        (0.1, "0p1"),
        (12.3456, "12p3456"),
        (10.0, "10p0"),
    ],
)
def test_format_float_token(value: float, expected: str) -> None:
    assert format_float_token(value) == expected


# ---- _resolve_output_dir ----


def test_resolve_output_dir_variable() -> None:
    path = _resolve_output_dir(
        output_root="output/optimal/dp",
        schedule_time_s=440.0,
        delta_speed_mps=0.1,
        stage_division="variable",
        sub_stage_count=30,
        uniform_step_size=10.0,
    )
    assert path == os.path.join("output/optimal/dp", "440p0_0p1_var30")


def test_resolve_output_dir_uniform() -> None:
    path = _resolve_output_dir(
        output_root="output/custom",
        schedule_time_s=500.0,
        delta_speed_mps=0.05,
        stage_division="uniform",
        sub_stage_count=30,
        uniform_step_size=25.0,
    )
    assert path == os.path.join("output/custom", "500p0_0p05_uni25p0")


def test_resolve_output_dir_different_divisions_produce_distinct_paths() -> None:
    """不同划分方式应输出到不同的目录。"""
    path_var = _resolve_output_dir(
        output_root="out",
        schedule_time_s=440.0,
        delta_speed_mps=0.1,
        stage_division="variable",
        sub_stage_count=30,
        uniform_step_size=10.0,
    )
    path_uni = _resolve_output_dir(
        output_root="out",
        schedule_time_s=440.0,
        delta_speed_mps=0.1,
        stage_division="uniform",
        sub_stage_count=30,
        uniform_step_size=10.0,
    )
    assert path_var != path_uni


# ---- _validate_cli_args ----


class TestValidateCliArgs:
    def test_accepts_valid_args(self) -> None:
        import argparse

        ns = argparse.Namespace(
            output_root="out",
            schedule_time_s=440.0,
            delta_speed_mps=0.1,
            max_outer_iterations=100,
            uniform_step_size=10.0,
            sub_stage_count=30,
        )
        _validate_cli_args(ns)  # 不应抛出异常

    @pytest.mark.parametrize(
        "field,value",
        [
            ("output_root", "   "),
            ("schedule_time_s", 0.0),
            ("schedule_time_s", -1.0),
            ("delta_speed_mps", 0.0),
            ("delta_speed_mps", -0.1),
            ("max_outer_iterations", 0),
            ("max_outer_iterations", -1),
            ("uniform_step_size", 0.0),
            ("uniform_step_size", -10.0),
            ("sub_stage_count", 0),
            ("sub_stage_count", -1),
        ],
    )
    def test_rejects_invalid_value(self, field: str, value: object) -> None:
        import argparse

        ns = argparse.Namespace(
            output_root="out",
            schedule_time_s=440.0,
            delta_speed_mps=0.1,
            max_outer_iterations=100,
            uniform_step_size=10.0,
            sub_stage_count=30,
        )
        setattr(ns, field, value)
        with pytest.raises(ValueError):
            _validate_cli_args(ns)


# ---- main() rejects invalid args ----


@pytest.mark.parametrize(
    "argv",
    [
        ["--schedule-time-s", "0"],
        ["--delta-speed-mps", "0"],
        ["--max-outer-iterations", "0"],
        ["--output-root", "   "],
        ["--uniform-step-size", "0"],
        ["--uniform-step-size", "-1"],
        ["--sub-stage-count", "0"],
        ["--sub-stage-count", "-1"],
    ],
)
def test_main_rejects_invalid_args(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    assert exc_info.value.code == 2


def test_run_optimization_metrics_include_start_and_target_speed(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeOptimizer:
        def __init__(self, **kwargs):
            del kwargs

        def optimize(self, *, max_speed: float, delta_speed: float, max_iters: int):
            del max_speed, delta_speed, max_iters
            return {
                "pos": [0.0, 20.0],
                "speed": [1.5, 0.0],
                "cum_time_s": [0.0, 9.0],
                "total_time": 9.0,
                "total_energy": 12.0,
            }

    train_service = SimpleNamespace(
        schedule_time=10.0,
        start_position=0.0,
        start_speed=1.5,
        target_position=20.0,
        max_acc_change=0.75,
    )
    monkeypatch.setattr(
        "scripts.reproduce_dp.build_scenario",
        lambda *, schedule_time_s: (
            SimpleNamespace(max_speed=30.0),
            object(),
            object(),
            train_service,
        ),
    )
    monkeypatch.setattr("scripts.reproduce_dp.VariableSpacingDPOptimizer", FakeOptimizer)
    monkeypatch.setattr(
        "scripts.reproduce_dp.compute_comfort_metrics_from_trajectory",
        lambda **kwargs: {
            "comfort_tav": 0.0,
            "comfort_er_pct": 0.0,
            "comfort_rms": 0.0,
        },
    )

    args = Namespace(
        schedule_time_s=10.0,
        hide_precompute_progress=True,
        precompute_mode="serial",
        precompute_workers=1,
        precompute_chunk_size=None,
        mp_start_method=None,
        stage_division="variable",
        uniform_step_size=100.0,
        sub_stage_count=30,
        skip_disk_cache=False,
        delta_speed_mps=0.1,
        max_outer_iterations=1,
    )

    assert _run_optimization(cli_args=args, output_dir=str(tmp_path)) == 0

    metrics_path = tmp_path / "optimized_speed_curve_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["start_speed_mps"] == pytest.approx(1.5)
    assert metrics["target_speed_mps"] == pytest.approx(0.0)
