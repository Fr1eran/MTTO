from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pytest

import scripts.analyze_dp_redundancy_error as analyze_dp_redundancy_error
import scripts.analyze_sps_compliance as analyze_sps_compliance
import scripts.compare_speed_profiles as compare_speed_profiles
import scripts.show_dp_result as show_dp_result


@pytest.mark.parametrize(
    ("parser_factory", "prefix"),
    (
        (analyze_dp_redundancy_error._build_cli_parser, []),
        (analyze_sps_compliance._build_cli_parser, []),
        (
            compare_speed_profiles._build_cli_parser,
            ["--rl-model-dir", "output/model"],
        ),
        (show_dp_result._build_cli_parser, []),
    ),
)
def test_figure_clis_accept_only_output_directory(
    parser_factory: Callable[[], argparse.ArgumentParser],
    prefix: list[str],
) -> None:
    parser = parser_factory()
    assert isinstance(parser, argparse.ArgumentParser)
    output_dir = Path("output/figures")

    args = parser.parse_args([*prefix, "--output-dir", str(output_dir)])
    assert args.output_dir == output_dir

    with pytest.raises(SystemExit):
        parser.parse_args([*prefix, "--output-file", "custom.pdf"])


def test_fixed_generic_figure_filenames_are_pdf() -> None:
    assert analyze_dp_redundancy_error.FIGURE_FILENAME == "dp_redundancy_error.pdf"
    assert analyze_sps_compliance.COMPARE_FIGURE_FILENAME == "dp_rl_sps_compliance.pdf"
    assert set(analyze_sps_compliance.SINGLE_FIGURE_FILENAMES.values()) == {
        "dp_sps_compliance.pdf",
        "rl_sps_compliance.pdf",
    }
    assert compare_speed_profiles.FIGURE_FILENAME == "dp_rl_actual_comparison.pdf"
    assert show_dp_result.FIGURE_FILENAME == "dp_result.pdf"
