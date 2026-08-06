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

    assert args.plot_type == "stopping-heatmap"
    assert args.output_file is None
    assert args.minimal is False


@pytest.mark.parametrize("plot_type", show_potential_function.PLOT_TYPE_CHOICES)
def test_show_potential_function_cli_accepts_plot_type(plot_type: str) -> None:
    parser = show_potential_function._build_cli_parser()
    args = parser.parse_args(["--plot-type", plot_type])

    assert args.plot_type == plot_type


def test_show_potential_function_cli_accepts_minimal_flags() -> None:
    parser = show_potential_function._build_cli_parser()

    args_minimal = parser.parse_args(["--minimal"])
    assert args_minimal.minimal is True

    args_no_minimal = parser.parse_args(["--no-minimal"])
    assert args_no_minimal.minimal is False


def test_main_dispatches_selected_plot_type(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def _fake_resolve_plotter(plot_type: str, *, minimal: bool):
        called["plot_type"] = plot_type
        called["minimal"] = minimal
        return lambda: _FakeFigure()

    monkeypatch.setattr(show_potential_function, "_apply_plot_style", lambda: None)
    monkeypatch.setattr(
        show_potential_function,
        "_resolve_plotter",
        _fake_resolve_plotter,
    )
    monkeypatch.setattr(show_potential_function.plt, "show", lambda: None)

    exit_code = show_potential_function.main(
        ["--plot-type", "safety-position", "--minimal"]
    )

    assert exit_code == 0
    assert called == {"plot_type": "safety-position", "minimal": True}


def test_main_saves_figure_and_creates_parent_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figure = _FakeFigure()
    output_file = tmp_path / "nested" / "potential.png"

    def _fake_resolve_plotter(
        _plot_type: str, *, minimal: bool
    ) -> Callable[[], object]:
        del minimal
        return lambda: figure

    monkeypatch.setattr(show_potential_function, "_apply_plot_style", lambda: None)
    monkeypatch.setattr(
        show_potential_function,
        "_resolve_plotter",
        _fake_resolve_plotter,
    )
    monkeypatch.setattr(show_potential_function.plt, "show", lambda: None)

    exit_code = show_potential_function.main(
        [
            "--plot-type",
            "stopping-slices",
            "--output-file",
            str(output_file),
        ]
    )

    assert exit_code == 0
    assert output_file.parent.is_dir()
    assert figure.saved_paths == [output_file]
    assert figure.savefig_calls[0]["kwargs"] == {
        "transparent": True,
        "facecolor": "none",
        "edgecolor": "none",
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }


def test_main_does_not_save_when_save_flag_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figure = _FakeFigure()

    def _fake_resolve_plotter(
        _plot_type: str, *, minimal: bool
    ) -> Callable[[], object]:
        del minimal
        return lambda: figure

    monkeypatch.setattr(show_potential_function, "_apply_plot_style", lambda: None)
    monkeypatch.setattr(
        show_potential_function,
        "_resolve_plotter",
        _fake_resolve_plotter,
    )
    monkeypatch.setattr(show_potential_function.plt, "show", lambda: None)

    exit_code = show_potential_function.main(["--plot-type", "safety-speed"])

    assert exit_code == 0
    assert figure.saved_paths == []


def test_plot_stopping_potential_slices_minimal_keeps_axis_and_hides_annotations() -> (
    None
):
    fig = show_potential_function.plot_stopping_potential_slices(minimal=True)
    ax_left, ax_right = fig.axes

    assert getattr(fig, "_suptitle", None) is None
    assert ax_left.axison is True
    assert ax_right.axison is True
    assert len(ax_left.lines) == 1
    assert len(ax_right.lines) == 1
    show_potential_function.plt.close(fig)


def test_plot_stopping_potential_slices_default_keeps_annotations() -> None:
    fig = show_potential_function.plot_stopping_potential_slices(minimal=False)
    ax_left, ax_right = fig.axes

    assert getattr(fig, "_suptitle", None) is not None
    assert ax_left.axison is True
    assert ax_right.axison is True
    assert len(ax_left.lines) == 4
    assert len(ax_right.lines) == 3
    show_potential_function.plt.close(fig)


def test_plot_safety_speed_minimal_keeps_upper_and_lower_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    fig = show_potential_function.plot_safety_potential_heatmap_speed(minimal=True)
    ax = fig.axes[0]

    assert ax.axison is True
    assert len(ax.lines) == 3
    assert len(fig.axes) == 1
    show_potential_function.plt.close(fig)


def test_safety_speed_asymmetric_v3_is_position_decoupled() -> None:
    speed = np.asarray([8.0, 15.0, 24.0], dtype=np.float64)
    min_speed = np.asarray([5.0, 5.0, 5.0], dtype=np.float64)
    max_speed = np.asarray([25.0, 25.0, 25.0], dtype=np.float64)

    near_target = show_potential_function._potential_safety_speed_asymmetric_v3(
        np.asarray([990.0, 995.0, 1000.0], dtype=np.float64),
        speed,
        min_speed,
        max_speed,
        1000.0,
    )
    far_target = show_potential_function._potential_safety_speed_asymmetric_v3(
        np.asarray([0.0, 100.0, 200.0], dtype=np.float64),
        speed,
        min_speed,
        max_speed,
        30000.0,
    )

    np.testing.assert_allclose(near_target, far_target)
    assert np.all(near_target <= 0.0)


def test_plot_safety_position_minimal_keeps_upper_and_lower_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    fig = show_potential_function.plot_safety_potential_heatmap_position(minimal=True)
    ax = fig.axes[0]

    assert ax.axison is True
    assert len(ax.lines) == 2
    assert len(fig.axes) == 1
    show_potential_function.plt.close(fig)


def test_plot_stopping_heatmap_2d_minimal_skips_colorbar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    fig_minimal = show_potential_function.plot_stopping_potential_heatmap(
        view_mode="2d",
        minimal=True,
    )
    fig_default = show_potential_function.plot_stopping_potential_heatmap(
        view_mode="2d",
        minimal=False,
    )

    assert len(fig_minimal.axes) == 1
    assert len(fig_default.axes) == 2
    assert len(fig_minimal.axes[0].lines) == 3
    assert len(fig_default.axes[0].lines) == 3
    assert len(fig_default.legends) == 1

    show_potential_function.plt.close(fig_minimal)
    show_potential_function.plt.close(fig_default)


def test_safety_speed_single_plot_matches_guidance_wide_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    fig = show_potential_function.plot_safety_potential_heatmap_speed(minimal=False)
    ax, colorbar_axis = fig.axes

    assert len(ax.lines) == 3
    assert len(fig.legends) == 1
    assert colorbar_axis.get_position().x0 > ax.get_position().x1
    assert colorbar_axis.get_ylabel() == ""
    show_potential_function.plt.close(fig)


def test_plot_guidance_potentials_wide_uses_shared_final_stop_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    fig = show_potential_function.plot_guidance_potentials_wide(minimal=False)
    ax_safety, ax_stopping, *_colorbars = fig.axes

    assert len(fig.axes) == 4
    np.testing.assert_allclose(ax_safety.get_xlim(), ax_stopping.get_xlim())
    np.testing.assert_allclose(ax_safety.get_ylim(), ax_stopping.get_ylim())
    assert len(ax_safety.lines) == 3
    assert len(ax_stopping.lines) == 3
    assert [text.get_text() for text in fig.texts] == ["(a)", "(b)"]
    assert len(fig.legends) == 1
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        r"$v_{\min}(x)$",
        r"$v_{\max}(x)$",
        "Target position",
    ]
    safety_colorbar_axis, stopping_colorbar_axis = fig.axes[2:]
    assert safety_colorbar_axis.get_xlabel() == ""
    assert stopping_colorbar_axis.get_xlabel() == ""
    assert safety_colorbar_axis.get_position().x0 > ax_safety.get_position().x1
    assert stopping_colorbar_axis.get_position().x0 > ax_stopping.get_position().x1
    show_potential_function.plt.close(fig)


def test_plot_guidance_potentials_wide_minimal_hides_colorbars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    fig = show_potential_function.plot_guidance_potentials_wide(minimal=True)

    assert len(fig.axes) == 2
    assert all(axis.axison for axis in fig.axes)
    show_potential_function.plt.close(fig)


def test_final_stop_field_masks_values_outside_both_speed_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)
    _patch_mock_safeguard_curves(monkeypatch)

    field = show_potential_function._build_seventh_auxiliary_stop_field()

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


def test_stopping_potential_v3_uses_state_local_speed_limit() -> None:
    pos = np.asarray([100.0, 100.0], dtype=np.float64)
    speed = np.asarray([10.0, 10.0], dtype=np.float64)
    potential = show_potential_function._potential_stopping_v3(
        pos,
        speed,
        target_pos=100.0,
        max_speed_mps=np.asarray([10.0, 100.0], dtype=np.float64),
    )

    assert potential[0] == pytest.approx(10.0 * np.exp(-10.0 / 3.0))
    assert potential[1] > potential[0]


def test_apply_minimal_axis_style_keeps_3d_axis_on() -> None:
    fig = show_potential_function.plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    show_potential_function._apply_minimal_axis_style(ax)

    assert ax.axison is True
    show_potential_function.plt.close(fig)


def test_apply_transparent_background_sets_figure_and_axes_transparent() -> None:
    fig = show_potential_function.plt.figure()
    ax = fig.add_subplot(111)

    show_potential_function._apply_transparent_background(fig)

    assert fig.patch.get_alpha() == 0.0
    assert ax.patch.get_alpha() == 0.0
    show_potential_function.plt.close(fig)
