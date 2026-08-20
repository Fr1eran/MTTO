from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.curve_geometry import (
    cal_regions,
    concatenate_curves_list,
    cut_curve_by_crosspoint,
    find_2curves_crosspoint,
    pad_2curve_lists,
    pad_2curves,
)
from utils.curve_plot import concatenate_curves_with_NaN, draw_regions


def test_concatenate_curves_with_nan_empty() -> None:
    x, y = concatenate_curves_with_NaN([])
    assert x.size == 0
    assert y.size == 0
    assert x.dtype == np.float64
    assert y.dtype == np.float64


def test_concatenate_curves_with_nan_normal() -> None:
    c1 = np.array([[0.0, 1.0], [10.0, 20.0]], dtype=np.float64)
    c2 = np.array([[2.0, 3.0, 4.0], [30.0, 40.0, 50.0]], dtype=np.float64)

    x, y = concatenate_curves_with_NaN([c1, c2])

    assert len(x) == 2 + 1 + 3 + 1
    assert len(y) == len(x)
    np.testing.assert_allclose(x[:2], [0.0, 1.0])
    assert np.isnan(x[2])
    np.testing.assert_allclose(x[3:6], [2.0, 3.0, 4.0])
    assert np.isnan(x[6])

    np.testing.assert_allclose(y[:2], [10.0, 20.0])
    assert np.isnan(y[2])
    np.testing.assert_allclose(y[3:6], [30.0, 40.0, 50.0])
    assert np.isnan(y[6])


def test_concatenate_curves_list_empty() -> None:
    x, y = concatenate_curves_list([])
    assert x.size == 0
    assert y.size == 0
    assert x.dtype == np.float64
    assert y.dtype == np.float64


def test_concatenate_curves_list_normal() -> None:
    c1 = np.array([[0.0, 1.0], [10.0, 20.0]], dtype=np.float64)
    c2 = np.array([[2.0, 3.0, 4.0], [30.0, 40.0, 50.0]], dtype=np.float64)

    x, y = concatenate_curves_list([c1, c2])

    assert x.shape == (5,)
    assert y.shape == (5,)
    np.testing.assert_allclose(x, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(y, [10.0, 20.0, 30.0, 40.0, 50.0])


def test_find_2curves_crosspoint_standard() -> None:
    # Line 1: y = 2x from x=0 to 10
    c1 = np.array([[0.0, 10.0], [0.0, 20.0]], dtype=np.float64)
    # Line 2: y = -x + 15 from x=0 to 10. Intersection at x=5, y=10.
    c2 = np.array([[0.0, 10.0], [15.0, 5.0]], dtype=np.float64)

    x_cross, y_cross = find_2curves_crosspoint(c1, c2)
    assert x_cross == pytest.approx(5.0, abs=1e-2)
    assert y_cross == pytest.approx(10.0, abs=1e-2)


def test_find_2curves_crosspoint_errors() -> None:
    # No x-domain overlap
    c1 = np.array([[0.0, 5.0], [0.0, 10.0]], dtype=np.float64)
    c2 = np.array([[6.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="overlapping x-domain"):
        _ = find_2curves_crosspoint(c1, c2)

    # Overlapping domain but parallel without intersection
    c3 = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    c4 = np.array([[0.0, 10.0], [10.0, 20.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="intersect"):
        _ = find_2curves_crosspoint(c3, c4)


def test_cut_curve_by_crosspoint() -> None:
    s = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    v = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)

    cut_s, cut_v = cut_curve_by_crosspoint(s, v, 3.5)
    np.testing.assert_allclose(cut_s, [4.0, 6.0, 8.0])
    np.testing.assert_allclose(cut_v, [30.0, 40.0, 50.0])


def test_cal_regions() -> None:
    # Above curve 1: y = 20 - x (x in 0..10)
    c_above = np.array([[0.0, 10.0], [20.0, 10.0]], dtype=np.float64)
    # Below curve 1: y = x (x in 0..10), cross at x=10, y=10
    c_below = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float64)

    cross_pts, above_parts, below_parts = cal_regions([c_above], [c_below])

    assert cross_pts.shape == (2, 1)
    assert cross_pts[0, 0] == pytest.approx(10.0, abs=1e-2)
    assert cross_pts[1, 0] == pytest.approx(10.0, abs=1e-2)
    assert len(above_parts) == 1
    assert len(below_parts) == 1


def test_cal_regions_mismatched_length_raises() -> None:
    c = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="same size"):
        _ = cal_regions([c], [])


def test_pad_2curves_and_pad_2curve_lists() -> None:
    c1_x = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    c1_y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    c2_x = np.array([0.0, 5.0], dtype=np.float64)
    c2_y = np.array([4.0, 5.0], dtype=np.float64)

    p1_x, p1_y, p2_x, p2_y = pad_2curves(c1_x, c1_y, c2_x, c2_y)
    assert len(p1_x) == 3
    assert len(p2_x) == 3
    assert p2_x[-1] == pytest.approx(10.0)
    assert p2_y[-1] == pytest.approx(5.0)

    # Test reverse direction padding
    p1_x_r, p1_y_r, p2_x_r, p2_y_r = pad_2curves(c2_x, c2_y, c1_x, c1_y)
    assert len(p1_x_r) == 3
    assert len(p2_x_r) == 3

    # Test list padding
    c1 = np.stack([c1_x, c1_y], axis=0)
    c2 = np.stack([c2_x, c2_y], axis=0)
    list1_p, list2_p = pad_2curve_lists([c1], [c2])
    assert len(list1_p) == 1
    assert len(list2_p) == 1
    assert list1_p[0].shape == (2, 3)
    assert list2_p[0].shape == (2, 3)

    with pytest.raises(ValueError, match="same size"):
        _ = pad_2curve_lists([c1], [])


def test_draw_regions_smoke() -> None:
    c_above = [np.array([[0.0, 5.0, 10.0], [20.0, 15.0, 10.0]], dtype=np.float64)]
    c_below = [np.array([[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]], dtype=np.float64)]

    fig, ax = plt.subplots()
    try:
        draw_regions(
            ax=ax,
            above_curves_list=c_above,
            below_curves_list=c_below,
            label="test_region",
            color="red",
            alpha=0.5,
        )
        # Empty list should simply no-op
        draw_regions(
            ax=ax,
            above_curves_list=[],
            below_curves_list=[],
            label="empty",
            color="blue",
            alpha=0.5,
        )
    finally:
        plt.close(fig)
