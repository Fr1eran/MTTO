from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import scripts.show_score_function as show_score
from model.ocs import TrainService
from rl.reward_calculator import RewardCalculator


def test_current_stopping_score_scalar_and_array():
    # Dead zone within max_stop_error (0.3m) evaluates to 1.0
    assert show_score.current_stopping_score(0.0) == 1.0
    assert show_score.current_stopping_score(0.2) == 1.0
    assert show_score.current_stopping_score(0.3) == 1.0

    # Outside dead zone: delta = 1.1 - 0.3 = 0.8 -> 1 / (1 + (0.8/0.8)^2) = 0.5
    assert show_score.current_stopping_score(1.1) == pytest.approx(0.5)

    # Negative stop error is handled via absolute value
    assert show_score.current_stopping_score(-1.1) == pytest.approx(0.5)

    # Array evaluation preserves shape and matches RewardCalculator
    arr = np.array([[0.0, 0.3], [1.1, 1.9]])
    res = show_score.current_stopping_score(arr)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2, 2)
    assert res[0, 0] == 1.0
    assert res[0, 1] == 1.0
    assert res[1, 0] == pytest.approx(0.5)


def test_current_punctuality_score_scalar_and_array():
    # Zero time error gives 1.0
    assert show_score.current_punctuality_score(0.0) == 1.0

    # At tau = 45.0s, score is exp(-1.0)
    assert show_score.current_punctuality_score(45.0) == pytest.approx(math.exp(-1.0))

    # At tau = 90.0s, score is exp(-2.0)
    assert show_score.current_punctuality_score(90.0) == pytest.approx(math.exp(-2.0))

    # Negative time error is handled via absolute value
    assert show_score.current_punctuality_score(-45.0) == pytest.approx(math.exp(-1.0))

    # Array evaluation preserves shape and matches RewardCalculator
    arr = np.linspace(0.0, 90.0, 5)
    res = show_score.current_punctuality_score(arr)
    assert isinstance(res, np.ndarray)
    assert res.shape == (5,)
    assert res[0] == 1.0
    assert res[-1] == pytest.approx(math.exp(-2.0))


def test_custom_calculator_injection():
    # Custom TrainService with max_stop_error = 1.0
    service = TrainService(
        start_position=0.0,
        target_position=100.0,
        schedule_time=50.0,
        max_acc_change=0.75,
        max_stop_error=1.0,
    )
    custom_calc = RewardCalculator(
        service,
        max_episode_steps=100,
        whole_distance_m=100.0,
        max_energy_consumption_kj=100.0,
        gamma=0.998,
    )

    # With max_stop_error=1.0, 0.8 is within dead zone
    assert show_score.current_stopping_score(0.8, calculator=custom_calc) == 1.0
    # At 1.8, delta = 0.8 -> 0.5
    assert show_score.current_stopping_score(
        1.8, calculator=custom_calc
    ) == pytest.approx(0.5)


def test_visualize_stopping_score_function():
    fig = show_score.visualize_stopping_score_function()
    assert fig is not None
    ax = fig.axes[0]
    lines = ax.get_lines()
    # Main curve line label includes beta=0.8
    assert r"\frac{1}{1+(\max(0,x-x_1)/0.8)^2}" in lines[0].get_label()
    # Vertical threshold line
    assert "x_1 = 0.3" in lines[1].get_label()


def test_visualize_punctuality_score_function():
    fig = show_score.visualize_punctuality_score_function()
    assert fig is not None
    ax = fig.axes[0]
    lines = ax.get_lines()
    # Punctuality curve line label includes 45s decay constant
    assert r"\exp\left(-x/45\right)" in lines[0].get_label()


def test_visualize_combined_score_functions():
    fig = show_score.visualize_combined_score_functions()
    assert fig is not None
    assert len(fig.axes) == 2


def test_cli_execution(tmp_path: Path):
    output_dir = tmp_path / "scores"
    ret = show_score.main(
        [
            "--plot",
            "combined",
            "--output-dir",
            str(output_dir),
            "--no-show",
        ]
    )
    assert ret == 0
    output_pdf = output_dir / "score_functions.pdf"
    assert output_pdf.is_file()
    assert output_pdf.stat().st_size > 0


@pytest.mark.parametrize("option", ["--dpi", "--pad-inches", "--output-file"])
def test_cli_rejects_removed_export_options(option: str):
    with pytest.raises(SystemExit):
        _ = show_score.parse_args([option, "300"])


def test_cli_plot_subcommands():
    assert show_score.main(["--plot", "stopping", "--no-show"]) == 0
    assert show_score.main(["--plot", "punctuality", "--no-show"]) == 0
