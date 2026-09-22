from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import pytest
from numpy.typing import NDArray

matplotlib.use("Agg")

import scripts.show_potential_function as show_potential_function


class _FakeFigure:
    def __init__(self) -> None:
        self.saved_paths: list[Path] = []
        self.savefig_calls: list[dict[str, object]] = []

    def savefig(self, path: str | Path, *args: object, **kwargs: object) -> None:
        self.saved_paths.append(Path(path))
        self.savefig_calls.append({"args": args, "kwargs": kwargs})


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
        compact_num = min(int(num), limit) if int(num) > 256 else int(num)
        return original_linspace(
            start,
            stop,
            compact_num,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(show_potential_function.np, "linspace", _compact_linspace)


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

    def _fake_load_safeguard_curves(
        *_keys: object,
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        return min_curves_list, max_curves_list

    def _fake_load_speed_limits(
        **_kwargs: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            np.asarray([60.0, 50.0, 30.0], dtype=np.float64),
            np.asarray([0.0, 28000.0, 29000.0, 30000.0], dtype=np.float64),
        )

    monkeypatch.setattr(
        show_potential_function,
        "load_safeguard_curves",
        _fake_load_safeguard_curves,
    )
    monkeypatch.setattr(
        show_potential_function,
        "load_speed_limits",
        _fake_load_speed_limits,
    )


def test_show_potential_function_cli_defaults() -> None:
    parser = show_potential_function._build_cli_parser()
    args = parser.parse_args([])

    assert args.plot_type == "safety-punctuality"
    assert args.output_dir is None
    assert args.minimal is False
    assert args.schedule_time_s == pytest.approx(465.0)


@pytest.mark.parametrize("plot_type", show_potential_function.PLOT_TYPE_CHOICES)
def test_show_potential_function_cli_accepts_plot_type(plot_type: str) -> None:
    parser = show_potential_function._build_cli_parser()
    args = parser.parse_args(["--plot-type", plot_type])

    assert args.plot_type == plot_type


@pytest.mark.parametrize(
    "plot_type",
    (
        "punctuality-slack",
        "safety-position",
        "stopping-heatmap",
        "stopping-slices",
        "guidance-wide",
        "safety-speed",
    ),
)
def test_show_potential_function_cli_rejects_retired_plot_types(
    plot_type: str,
) -> None:
    parser = show_potential_function._build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--plot-type", plot_type])


def test_show_potential_function_cli_accepts_minimal_flags() -> None:
    parser = show_potential_function._build_cli_parser()

    args_minimal = parser.parse_args(["--minimal"])
    assert args_minimal.minimal is True

    args_no_minimal = parser.parse_args(["--no-minimal"])
    assert args_no_minimal.minimal is False


def test_show_potential_function_cli_rejects_output_filename() -> None:
    parser = show_potential_function._build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-file", "figure.pdf"])


def test_main_dispatches_selected_plot_type(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def _fake_resolve_plotter(plot_type: str, *, minimal: bool, schedule_time_s: float):
        called["plot_type"] = plot_type
        called["minimal"] = minimal
        called["schedule_time_s"] = schedule_time_s
        return lambda: _FakeFigure()

    monkeypatch.setattr(show_potential_function, "apply_sci_curve_style", lambda: None)
    monkeypatch.setattr(
        show_potential_function,
        "_resolve_plotter",
        _fake_resolve_plotter,
    )
    monkeypatch.setattr(show_potential_function.plt, "show", lambda: None)

    exit_code = show_potential_function.main(
        ["--plot-type", "safety-punctuality", "--minimal"]
    )

    assert exit_code == 0
    assert called == {
        "plot_type": "safety-punctuality",
        "minimal": True,
        "schedule_time_s": 465.0,
    }


@pytest.mark.parametrize(
    ("minimal", "expected_filename", "expected_kwargs"),
    (
        (False, "punctuality_potential.pdf", {"dpi": 1200.0}),
        (
            True,
            "punctuality_potential_minimal.tiff",
            {"dpi": 1200.0, "pil_kwargs": {"compression": "tiff_lzw"}},
        ),
    ),
)
def test_main_saves_figure_and_creates_parent_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    minimal: bool,
    expected_filename: str,
    expected_kwargs: dict[str, object],
) -> None:
    figure = _FakeFigure()
    output_dir = tmp_path / "nested"
    expected_output = output_dir / expected_filename

    def _fake_resolve_plotter(
        _plot_type: str, *, minimal: bool, schedule_time_s: float
    ) -> Callable[[], object]:
        del minimal, schedule_time_s
        return lambda: figure

    monkeypatch.setattr(show_potential_function, "apply_sci_curve_style", lambda: None)
    monkeypatch.setattr(
        show_potential_function,
        "_resolve_plotter",
        _fake_resolve_plotter,
    )
    monkeypatch.setattr(show_potential_function.plt, "show", lambda: None)

    cli_args = [
        "--plot-type",
        "punctuality",
        "--output-dir",
        str(output_dir),
    ]
    if minimal:
        cli_args.append("--minimal")

    exit_code = show_potential_function.main(cli_args)

    assert exit_code == 0
    assert output_dir.is_dir()
    assert figure.saved_paths == [expected_output]
    assert figure.savefig_calls[0]["kwargs"] == expected_kwargs


def test_main_does_not_save_when_save_flag_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figure = _FakeFigure()

    def _fake_resolve_plotter(
        _plot_type: str, *, minimal: bool, schedule_time_s: float
    ) -> Callable[[], object]:
        del minimal, schedule_time_s
        return lambda: figure

    monkeypatch.setattr(show_potential_function, "apply_sci_curve_style", lambda: None)
    monkeypatch.setattr(
        show_potential_function,
        "_resolve_plotter",
        _fake_resolve_plotter,
    )
    monkeypatch.setattr(show_potential_function.plt, "show", lambda: None)

    exit_code = show_potential_function.main(["--plot-type", "safety"])

    assert exit_code == 0
    assert figure.saved_paths == []


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


def test_safety_speed_matches_runtime_reward_calculator_formula() -> None:
    from rl.reward_calculator import RewardCalculator

    speed = np.asarray([8.0, 15.0, 24.0], dtype=np.float64)
    min_speed = np.asarray([5.0, 5.0, 5.0], dtype=np.float64)
    max_speed = np.asarray([25.0, 25.0, 25.0], dtype=np.float64)

    potential = show_potential_function._potential_safety_speed(
        speed,
        min_speed,
        max_speed,
    )

    assert np.all(potential <= 0.0)
    for s, mi, ma in zip(speed, min_speed, max_speed, strict=True):
        expected = RewardCalculator._potential_safety(
            speed_mps=s, min_speed_mps=mi, max_speed_mps=ma
        )
        actual = show_potential_function._potential_safety_speed(
            np.asarray([s], dtype=np.float64),
            np.asarray([mi], dtype=np.float64),
            np.asarray([ma], dtype=np.float64),
        )[0]
        assert np.isclose(expected, actual, atol=1e-12)


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
        show_potential_function.punctuality_potential_from_error(
            field.redundant_time_grid_s - field.reference_slack_s[np.newaxis, :]
        ),
    )
    assert np.max(field.potential) <= 0.0
    assert (
        np.min(field.potential) >= -show_potential_function.PUNCTUALITY_POTENTIAL_SCALE
    )


def test_punctuality_single_and_safety_combined_have_sci_widths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    single = show_potential_function.plot_punctuality_potential(minimal=False)
    combined = show_potential_function.plot_safety_punctuality_potentials(minimal=False)

    assert single.get_size_inches()[0] == pytest.approx(85.0 / 25.4)
    assert combined.get_size_inches()[0] == pytest.approx(170.0 / 25.4)
    assert len(single.axes) == 2
    assert len(combined.axes) == 4
    assert [text.get_text() for text in combined.axes[0].texts] == ["(a)"]
    assert [text.get_text() for text in combined.axes[1].texts] == ["(b)"]
    assert combined.axes[0].get_ylabel() == "Speed (km/h)"
    assert combined.axes[2].get_title() == ""
    assert combined.axes[3].get_title() == ""
    assert combined.axes[1].get_ylabel() == "Redundant operation time (s)"
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
    from utils.plot_utils import sci_figure_size

    width, height = sci_figure_size(
        columns=columns,
        height_in=2.75,  # type: ignore[arg-type]
    )

    assert width == pytest.approx(expected_width_mm / 25.4)
    assert height == pytest.approx(2.75)
