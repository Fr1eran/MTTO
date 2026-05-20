from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import scripts.show_potential_function as show_potential_function


class _FakeFigure:
    def __init__(self) -> None:
        self.saved_paths: list[Path] = []
        self.savefig_calls: list[dict[str, object]] = []

    def savefig(self, path: str | Path, *args, **kwargs) -> None:
        self.saved_paths.append(Path(path))
        self.savefig_calls.append({"args": args, "kwargs": kwargs})


def _patch_compact_linspace(monkeypatch: pytest.MonkeyPatch, limit: int = 32) -> None:
    original_linspace = show_potential_function.np.linspace

    def _compact_linspace(start, stop, num, *args, **kwargs):
        return original_linspace(start, stop, min(int(num), limit), *args, **kwargs)

    monkeypatch.setattr(show_potential_function.np, "linspace", _compact_linspace)


def _patch_mock_safeguard_curves(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_curve = np.asarray(
        [
            [0.0, 50.0, 100.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    min_curves_list = [dummy_curve.copy() for _ in range(8)]
    max_curves_list = [dummy_curve.copy() for _ in range(8)]

    min_curves_list[6] = np.asarray(
        [
            [0.0, 50.0, 100.0],
            [45.0, 30.0, 10.0],
        ],
        dtype=np.float64,
    )
    max_curves_list[7] = np.asarray(
        [
            [0.0, 50.0, 100.0],
            [55.0, 45.0, 20.0],
        ],
        dtype=np.float64,
    )
    monkeypatch.setattr(
        show_potential_function,
        "load_safeguard_curves",
        lambda *_keys: (min_curves_list, max_curves_list),
    )


def test_show_potential_function_cli_defaults() -> None:
    parser = show_potential_function._build_cli_parser()
    args = parser.parse_args([])

    assert args.plot_type == "docking-heatmap"
    assert args.save is False
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


def test_main_rejects_missing_output_file_when_save_enabled() -> None:
    with pytest.raises(SystemExit) as exc_info:
        show_potential_function.main(["--save"])

    assert exc_info.value.code == 2


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

    monkeypatch.setattr(show_potential_function, "_apply_plot_style", lambda: None)
    monkeypatch.setattr(
        show_potential_function,
        "_resolve_plotter",
        lambda _plot_type, *, minimal: (lambda: figure),
    )
    monkeypatch.setattr(show_potential_function.plt, "show", lambda: None)

    exit_code = show_potential_function.main(
        [
            "--plot-type",
            "docking-slices",
            "--save",
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

    monkeypatch.setattr(show_potential_function, "_apply_plot_style", lambda: None)
    monkeypatch.setattr(
        show_potential_function,
        "_resolve_plotter",
        lambda _plot_type, *, minimal: (lambda: figure),
    )
    monkeypatch.setattr(show_potential_function.plt, "show", lambda: None)

    exit_code = show_potential_function.main(["--plot-type", "punctuality-curve"])

    assert exit_code == 0
    assert figure.saved_paths == []


def test_plot_docking_potential_slices_minimal_keeps_axis_and_hides_annotations() -> None:
    fig = show_potential_function.plot_docking_potential_slices(minimal=True)
    ax_left, ax_right = fig.axes

    assert fig._suptitle is None
    assert ax_left.axison is True
    assert ax_right.axison is True
    assert len(ax_left.lines) == 1
    assert len(ax_right.lines) == 1
    show_potential_function.plt.close(fig)


def test_plot_docking_potential_slices_default_keeps_annotations() -> None:
    fig = show_potential_function.plot_docking_potential_slices(minimal=False)
    ax_left, ax_right = fig.axes

    assert fig._suptitle is not None
    assert ax_left.axison is True
    assert ax_right.axison is True
    assert len(ax_left.lines) == 4
    assert len(ax_right.lines) == 3
    show_potential_function.plt.close(fig)


def test_plot_punctuality_curve_minimal_has_axis_and_no_legend_or_reference_line() -> None:
    fig = show_potential_function.plot_punctuality_potential_curve(minimal=True)
    ax = fig.axes[0]

    assert ax.axison is True
    assert ax.get_legend() is None
    assert len(ax.lines) == 1
    show_potential_function.plt.close(fig)


def test_plot_punctuality_curve_default_keeps_legend_and_reference_line() -> None:
    fig = show_potential_function.plot_punctuality_potential_curve(minimal=False)
    ax = fig.axes[0]

    assert ax.axison is True
    assert ax.get_legend() is not None
    assert len(ax.lines) == 2
    show_potential_function.plt.close(fig)


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


def test_plot_docking_heatmap_2d_minimal_skips_colorbar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compact_linspace(monkeypatch)

    fig_minimal = show_potential_function.plot_docking_potential_heatmap(
        view_mode="2d",
        minimal=True,
    )
    fig_default = show_potential_function.plot_docking_potential_heatmap(
        view_mode="2d",
        minimal=False,
    )

    assert len(fig_minimal.axes) == 1
    assert len(fig_default.axes) == 2

    show_potential_function.plt.close(fig_minimal)
    show_potential_function.plt.close(fig_default)


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
