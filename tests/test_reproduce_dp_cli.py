import os

import pytest

from scripts.reproduce_dp import (
    _build_cli_parser,
    _format_float_token,
    _resolve_output_dir,
    main,
)


def test_reproduce_dp_cli_defaults() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args([])

    assert args.output_root == "output/optimal/dp"
    assert args.schedule_time_s == pytest.approx(440.0)
    assert args.delta_speed_mps == pytest.approx(0.1)
    assert args.max_outer_iterations == 100
    assert args.precompute_mode == "serial"


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
    assert _format_float_token(value) == expected


def test_resolve_output_dir_uses_schedule_and_delta_tokens() -> None:
    path = _resolve_output_dir(
        output_root="output/optimal/dp",
        schedule_time_s=440.0,
        delta_speed_mps=0.1,
    )

    assert path == os.path.join("output/optimal/dp", "440p0_0p1")


@pytest.mark.parametrize(
    "argv",
    [
        ["--schedule-time-s", "0"],
        ["--delta-speed-mps", "0"],
        ["--max-outer-iterations", "0"],
        ["--output-root", "   "],
    ],
)
def test_main_rejects_invalid_args(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    assert exc_info.value.code == 2
