import matplotlib

matplotlib.use("Agg")
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from utils import plot_utils


@pytest.fixture(autouse=True)
def _restore_rcparams():
    original = plt.rcParams.copy()
    yield
    plt.rcParams.update(original)


def _set_available_fonts(
    monkeypatch: pytest.MonkeyPatch, font_names: list[str]
) -> None:
    fake_fonts = [SimpleNamespace(name=name) for name in font_names]
    monkeypatch.setattr(plot_utils.fontManager, "ttflist", fake_fonts)


def test_pick_selected_font_uses_preferred_when_available(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_available_fonts(monkeypatch, ["Calibri", "Carlito"])

    selected = plot_utils._pick_selected_or_first_available_font(
        plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
        preferred_font="Calibri",
    )

    assert selected == "Calibri"


def test_pick_selected_font_falls_back_to_carlito(monkeypatch: pytest.MonkeyPatch):
    _set_available_fonts(monkeypatch, ["Carlito"])

    selected = plot_utils._pick_selected_or_first_available_font(
        plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
        preferred_font="Calibri",
    )

    assert selected == "Carlito"


def test_pick_selected_font_raises_after_all_fallbacks_fail(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_available_fonts(monkeypatch, [])

    with pytest.raises(ValueError, match="已尝试替代字体仍不可用"):
        _ = plot_utils._pick_selected_or_first_available_font(
            plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
            preferred_font="Calibri",
        )


def test_pick_selected_font_raises_for_unknown_font(monkeypatch: pytest.MonkeyPatch):
    _set_available_fonts(monkeypatch, ["DejaVu Sans"])

    with pytest.raises(ValueError, match="不在候选字体中"):
        _ = plot_utils._pick_selected_or_first_available_font(
            plot_utils.SCI_ENGLISH_FONT_CANDIDATES,
            preferred_font="MyCommercialFont",
        )


def test_set_global_plot_style_applies_fallback_font(monkeypatch: pytest.MonkeyPatch):
    _set_available_fonts(monkeypatch, ["Carlito"])

    style = plot_utils.set_global_plot_style(
        font_preset="sci",
        preferred_font="Calibri",
    )

    assert style["font"] == "Carlito"
    assert plt.rcParams["font.family"][0] == "Carlito"
