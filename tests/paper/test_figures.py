"""Run migrated paper figures (env data, potential/score functions) noninteractively."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import matplotlib
import numpy as np
import pytest
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

import paper.figures.potential_function as show_potential_function
import paper.figures.score_function as show_score
from mtto.domain.scenario import Task
from mtto.rl.rewards import RewardCalculator, RewardNormalization
from paper.figures import (
    env_data,
    load_paper_scenario,
    load_paper_task,
    min_operation_time_curve,
    potential_function,
    real_operation_data,
    score_function,
)
from paper.figures.env_data import (
    create_figures,
    load_track_environment_data,
    main,
    save_compact_figures,
)
from paper.real_operation import real_operation_profile

_PAPER_SCENARIO = load_paper_scenario()


@pytest.mark.parametrize(
    ("module", "arguments", "filename"),
    [
        (env_data, ("--view", "overview"), "env_overview.pdf"),
        (
            potential_function,
            ("--plot-type", "punctuality"),
            "punctuality_potential.pdf",
        ),
        (score_function, ("--plot", "combined"), "score_functions.pdf"),
    ],
)
def test_saved_domain_figure(
    module, arguments, filename, tmp_path, monkeypatch
) -> None:
    output = tmp_path / module.__name__.rsplit(".", 1)[-1]
    command = [module.__name__, *arguments, "--output-dir", str(output), "--no-show"]
    monkeypatch.setattr(sys, "argv", command)
    if module in (score_function, potential_function):
        assert module.main(command[1:]) == 0
    else:
        module.main()
    assert (output / filename).is_file()


def test_real_operation_figure_keeps_three_profiles(monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    before = set(plt.get_fignums())
    try:
        real_operation_data.main()
        figures = [plt.figure(number) for number in set(plt.get_fignums()) - before]
        assert len(figures) == 3
        assert all(len(figure.axes) == 1 for figure in figures)
        profile = real_operation_profile(load_paper_scenario(), load_paper_task())
        assert len(figures[0].axes[0].lines[0].get_xdata()) == profile.position_m.size
        assert len(figures[1].axes[0].lines[0].get_xdata()) == profile.position_m.size
        assert len(figures[2].axes[0].lines) == 2
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)


def test_min_operation_time_interactive_curve(monkeypatch) -> None:
    task = load_paper_task()
    answers = iter((str(task.target_position_m - 200.0), "10", "y"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    def press_key() -> None:
        figure = plt.gcf()
        figure.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", figure.canvas, key="i")
        )

    monkeypatch.setattr(plt, "show", press_key)
    before = set(plt.get_fignums())
    try:
        min_operation_time_curve.main()
        figures = [plt.figure(number) for number in set(plt.get_fignums()) - before]
        assert len(figures) == 1
        assert len(figures[0].axes) == 1
        assert any(
            line.get_label() == "最短运行时间曲线" for line in figures[0].axes[0].lines
        )
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)


def test_load_track_environment_data():
    data = load_track_environment_data()
    assert data.accessible_points.ndim == 1
    assert data.dangerous_points.ndim == 1
    assert data.stations_cor.shape == (2, 2)
    assert data.speed_limits.size > 0
    assert data.speed_limit_intervals.size > 0
    assert data.slopes.size > 0
    assert data.slope_intervals.size > 0


def test_create_figures_view_modes():
    data = load_track_environment_data()

    # overview
    figs_overview = create_figures("overview", data)
    assert set(figs_overview.keys()) == {"overview"}
    assert len(figs_overview["overview"].axes) == 2
    overview_axes = figs_overview["overview"].axes
    assert [
        text.get_text()
        for axis in overview_axes
        for text in axis.texts
        if text.get_text() in {"(a)", "(b)"}
    ] == ["(a)", "(b)"]

    # full-curves
    figs_full = create_figures("full-curves", data)
    assert set(figs_full.keys()) == {"full_curves"}
    assert len(figs_full["full_curves"].axes) == 1

    # danger-region
    figs_danger = create_figures("danger-region", data)
    assert set(figs_danger.keys()) == {"danger_region"}
    assert len(figs_danger["danger_region"].axes) == 1

    # all
    figs_all = create_figures("all", data)
    assert set(figs_all.keys()) == {"overview", "full_curves", "danger_region"}

    for fig_dict in [figs_overview, figs_full, figs_danger, figs_all]:
        for fig in fig_dict.values():
            plt.close(fig)


def test_save_compact_figures_single_and_multi(tmp_path: Path):
    data = load_track_environment_data()

    # 单图保存
    figs_single = create_figures("full-curves", data)
    out_single = tmp_path / "single"
    saved_single = save_compact_figures(
        figs_single,
        out_single,
    )
    assert len(saved_single) == 1
    assert saved_single[0] == out_single / "env_full_curves.pdf"
    assert (out_single / "env_full_curves.pdf").is_file()

    # 多图保存
    figs_multi = create_figures("all", data)
    out_multi = tmp_path / "multi"
    saved_multi = save_compact_figures(
        figs_multi,
        out_multi,
    )
    assert len(saved_multi) == 3
    assert (out_multi / "env_overview.pdf").is_file()
    assert (out_multi / "env_full_curves.pdf").is_file()
    assert (out_multi / "env_danger_region.pdf").is_file()

    for fig in figs_single.values():
        plt.close(fig)
    for fig in figs_multi.values():
        plt.close(fig)


def test_main_cli_execution_no_show(tmp_path: Path):
    output_dir = tmp_path / "cli_test"
    with patch(
        "sys.argv",
        [
            "show_env_data",
            "--view",
            "full-curves",
            "--output-dir",
            str(output_dir),
            "--no-show",
        ],
    ):
        main()

    assert (output_dir / "env_full_curves.pdf").is_file()


def _patch_compact_linspace(monkeypatch: pytest.MonkeyPatch, limit: int = 32) -> None:
    original_linspace = cast(
        Callable[..., NDArray[np.floating]], show_potential_function.np.linspace
    )

    def _compact_linspace(
        start: float | NDArray[np.floating],
        stop: float | NDArray[np.floating],
        num: float,
        *args: object,
        **kwargs: object,
    ) -> NDArray[np.floating]:
        compact_num = (
            min(int(num), limit) if int(num) in (1200, 800, 600, 400) else int(num)
        )
        return original_linspace(
            start,
            stop,
            compact_num,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(show_potential_function.np, "linspace", _compact_linspace)
    monkeypatch.setattr(
        show_potential_function, "load_paper_scenario", lambda: _PAPER_SCENARIO
    )


def _patch_mock_safeguard_curves(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_curve = np.asarray(
        [
            [0.0, 50.0, 100.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    min_curves_list = [dummy_curve.copy() for _ in range(9)]
    max_curves_list = [dummy_curve.copy() for _ in range(10)]

    min_curves_list[6] = np.asarray(
        [
            [10700.0, 17828.0, 18067.0],
            [20.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    max_curves_list[7] = np.asarray(
        [
            [10700.0, 17828.0, 18067.0],
            [55.0, 45.0, 0.0],
        ],
        dtype=np.float64,
    )
    min_curves_list[8] = np.asarray(
        [
            [29010.0, 29270.0, 29340.0],
            [12.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )
    max_curves_list[9] = np.asarray(
        [
            [29010.0, 29270.0, 29340.0],
            [25.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )

    real_scenario = _PAPER_SCENARIO
    custom_safeguard = replace(
        real_scenario.safeguard,
        speed_limits=np.asarray([60.0, 50.0, 30.0], dtype=np.float64),
        speed_limit_intervals=np.asarray(
            [0.0, 28000.0, 29000.0, 30000.0], dtype=np.float64
        ),
        min_curves=tuple(min_curves_list),
        max_curves=tuple(max_curves_list),
    )
    custom_line = replace(
        real_scenario.line,
        speed_limits=np.asarray([60.0, 50.0, 30.0], dtype=np.float64),
        speed_limit_intervals=np.asarray(
            [0.0, 28000.0, 29000.0, 30000.0], dtype=np.float64
        ),
    )
    patched_scenario = replace(
        real_scenario,
        safeguard=custom_safeguard,
        line=custom_line,
    )

    monkeypatch.setattr(
        show_potential_function,
        "load_paper_scenario",
        lambda: patched_scenario,
    )


def test_show_potential_function_cli_rejects_output_filename() -> None:
    parser = show_potential_function._build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-file", "figure.pdf"])


def test_plot_safety_speed_minimal_keeps_upper_and_lower_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    fig = show_potential_function.plot_safety_potential_heatmap_speed(minimal=True)
    ax = fig.axes[0]

    assert ax.axison is True
    assert len(ax.lines) == 2
    assert len(fig.axes) == 1
    show_potential_function.plt.close(fig)


def test_safety_speed_single_plot_has_boundary_legend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    fig = show_potential_function.plot_safety_potential_heatmap_speed(minimal=False)
    ax, colorbar_axis = fig.axes

    assert len(ax.lines) == 2
    assert len(fig.legends) == 1
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        r"$v_{\min}(x)$",
        r"$v_{\max}(x)$",
    ]
    assert colorbar_axis.get_position().x0 > ax.get_position().x1
    assert colorbar_axis.get_ylabel() == ""
    show_potential_function.plt.close(fig)


def test_safety_potential_field_masks_values_outside_both_speed_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    field = show_potential_function._build_safety_potential_field()

    assert np.all(
        field.speed_grid_mps[field.feasible_mask]
        >= field.min_speed_grid_mps[field.feasible_mask]
    )
    assert np.all(
        field.speed_grid_mps[field.feasible_mask]
        <= field.max_speed_grid_mps[field.feasible_mask]
    )
    assert np.any(field.speed_grid_mps < field.min_speed_grid_mps)
    assert np.any(field.speed_grid_mps > field.max_speed_grid_mps)


def test_apply_minimal_axis_style_keeps_3d_axis_on() -> None:
    fig = show_potential_function.plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    show_potential_function._apply_minimal_axis_style(ax)

    assert ax.axison is True
    show_potential_function.plt.close(fig)


def test_apply_transparent_background_sets_figure_and_axes_opaque() -> None:
    fig = show_potential_function.plt.figure()
    ax = fig.add_subplot(111)

    show_potential_function._apply_transparent_background(fig)

    assert fig.patch.get_alpha() == 1.0
    assert ax.patch.get_alpha() == 1.0
    show_potential_function.plt.close(fig)


def test_punctuality_field_uses_full_route_and_canonical_potential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    field = show_potential_function._build_punctuality_potential_field()

    assert field.position_m[0] == pytest.approx(135.0)
    assert field.position_m[-1] == pytest.approx(29270.046)
    np.testing.assert_allclose(
        field.potential,
        show_potential_function.punctuality_potential_from_error_array(
            field.redundant_time_grid_s - field.reference_slack_s[np.newaxis, :]
        ),
    )
    assert np.max(field.potential) <= 0.0


def test_punctuality_single_and_safety_combined_have_sci_widths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    single = show_potential_function.plot_punctuality_potential(minimal=False)
    combined = show_potential_function.plot_safety_punctuality_potentials(minimal=False)

    assert single.get_size_inches()[0] == pytest.approx(85.0 / 25.4)
    assert combined.get_size_inches()[0] == pytest.approx(408.0 / 72.27)
    assert len(single.axes) == 2
    # Two panels and their colour bars; the zooms are insets of panel (a).
    ax_safety, ax_punctuality = combined.axes[:2]
    assert len(combined.axes) == 4
    assert [text.get_text() for text in ax_safety.texts] == ["(a)"]
    assert [text.get_text() for text in ax_punctuality.texts] == ["(b)"]
    assert len(ax_safety.child_axes) == 2
    assert ax_safety.get_ylabel() == "Speed (km/h)"
    assert all(axis.get_xlabel() == "Position (km)" for axis in combined.axes[:2])
    assert combined.axes[1].get_ylabel() == r"Theoretical time margin $\rho$ (s)"
    safety_mesh = combined.axes[0].collections[0]
    punctuality_mesh = combined.axes[1].collections[0]
    assert safety_mesh.cmap.name == "mtto_safety_penalty"
    assert punctuality_mesh.cmap.name == "mtto_punctuality_penalty"
    assert sum(safety_mesh.cmap(1.0)[:3]) > sum(safety_mesh.cmap(0.0)[:3])
    assert sum(punctuality_mesh.cmap(1.0)[:3]) > sum(punctuality_mesh.cmap(0.0)[:3])
    assert combined.axes[0].lines[0].get_color() == "tab:blue"
    assert combined.axes[0].lines[1].get_color() == "tab:red"
    assert len(combined.axes[0].lines[0].get_path_effects()) == 0
    assert len(combined.axes[0].lines[1].get_path_effects()) == 0
    assert combined.axes[1].lines[0].get_color() == "black"
    assert combined.axes[1].lines[0].get_linestyle() == "--"
    show_potential_function.plt.close(single)
    show_potential_function.plt.close(combined)


@pytest.mark.parametrize(
    ("columns", "expected_width_mm"),
    ((1, 85.0), (2, 170.0)),
)
def test_sci_figure_size_uses_requested_physical_width(
    columns: int, expected_width_mm: float
) -> None:
    from paper.plotting.style import sci_figure_size

    width, height = sci_figure_size(
        columns=columns,
        height_in=2.75,  # type: ignore[arg-type]
    )

    assert width == pytest.approx(expected_width_mm / 25.4)
    assert height == pytest.approx(2.75)


@pytest.mark.parametrize(
    ("error_m", "expected"),
    [
        (0.0, 1.0),
        (0.1, 1.0 / (1.0 + (1.0 / 3.0) ** 4)),
        (0.3, 0.5),
        (-0.3, 0.5),
        (0.6, 1.0 / 17.0),
    ],
)
def test_current_stopping_score_is_one_on_target_and_half_at_tolerance(
    error_m: float, expected: float
) -> None:
    assert show_score.current_stopping_score(error_m) == pytest.approx(expected)


def test_current_stopping_score_is_smooth_and_decreasing_across_tolerance():
    x = np.linspace(0.0, 2.0, 20_001)
    score = show_score.current_stopping_score(x)
    slope = np.diff(score) / np.diff(x)
    assert isinstance(score, np.ndarray)
    assert np.all(slope <= 0.0)
    # No jump at the tolerance: neighbouring slopes around 0.3 m agree.
    i = int(np.searchsorted(x, 0.3))
    assert slope[i - 1] == pytest.approx(slope[i], rel=1e-2)
    assert slope[i] == pytest.approx(-4.0 / (4.0 * 0.3), rel=1e-2)


def test_current_punctuality_score_scalar_and_array():
    # Zero time error gives 1.0
    assert show_score.current_punctuality_score(0.0) == 1.0

    tau = show_score.PUNCTUALITY_DECAY_TIME_S
    assert show_score.current_punctuality_score(tau) == pytest.approx(math.exp(-1.0))
    assert show_score.current_punctuality_score(2 * tau) == pytest.approx(
        math.exp(-2.0)
    )

    # Negative time error is handled via absolute value
    assert show_score.current_punctuality_score(-tau) == pytest.approx(math.exp(-1.0))

    # Array evaluation preserves shape and matches RewardCalculator
    arr = np.linspace(0.0, 2 * tau, 5)
    res = show_score.current_punctuality_score(arr)
    assert isinstance(res, np.ndarray)
    assert res.shape == (5,)
    assert res[0] == 1.0
    assert res[-1] == pytest.approx(math.exp(-2.0))


def test_custom_calculator_injection():
    # Custom Task with max_stop_error_m = 1.0
    task = Task(
        start_position_m=0.0,
        target_position_m=100.0,
        schedule_time_s=50.0,
        max_jerk_mps3=0.75,
        max_stop_error_m=1.0,
        max_arr_time_error_s=10.0,
    )
    custom_calc = RewardCalculator(
        RewardNormalization(100.0, 0.0),
        gamma=0.998,
        step_time_s=1.0,
    )

    # The tolerance of the task is the half-score point.
    assert show_score.current_stopping_score(
        1.0, calculator=custom_calc, task=task
    ) == pytest.approx(0.5)


def test_visualize_stopping_score_function():
    fig = show_score.visualize_stopping_score_function()
    assert fig is not None
    ax = fig.axes[0]
    lines = ax.get_lines()
    # Main curve line label includes beta=0.3
    assert r"\frac{1}{1+(x/x_1)^{4}}" in lines[0].get_label()
    # Vertical threshold line
    assert "x_1 = 0.3" in lines[1].get_label()


def test_visualize_punctuality_score_function():
    fig = show_score.visualize_punctuality_score_function()
    assert fig is not None
    ax = fig.axes[0]
    lines = ax.get_lines()
    # Punctuality curve line label includes 45s decay constant
    assert r"\exp\left(-x/15\right)" in lines[0].get_label()


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
