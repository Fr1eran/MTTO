import numpy as np
import pytest

from utils.score_function import SigmoidVariant


def test_sigmoid_variant_scalar_input_returns_float():
    fn = SigmoidVariant(x1=0.3, x2=30.0, c=10.0)

    x = 5.0
    reward = fn(x)
    grad = fn.gradient(x)

    assert isinstance(reward, float)
    assert isinstance(grad, float)

    exponent = np.clip(fn.k * (x - fn.xm), -500.0, 500.0)
    expected_reward = 1.0 / (1.0 + np.exp(exponent))
    expected_grad = -fn.k * expected_reward * (1.0 - expected_reward)

    assert np.isclose(reward, expected_reward, atol=1e-12)
    assert np.isclose(grad, expected_grad, atol=1e-12)


def test_sigmoid_variant_array_input_keeps_shape():
    fn = SigmoidVariant(x1=0.3, x2=30.0, c=10.0)

    x = np.linspace(0.0, 40.0, 1000, dtype=np.float64).reshape(20, 50)
    reward = fn(x)
    grad = fn.gradient(x)

    assert isinstance(reward, np.ndarray)
    assert isinstance(grad, np.ndarray)
    assert reward.shape == x.shape
    assert grad.shape == x.shape


def test_sigmoid_variant_list_input_returns_ndarray_with_same_dimension():
    fn = SigmoidVariant(x1=0.3, x2=30.0, c=10.0)

    x = [[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]]
    reward = fn(x)

    assert isinstance(reward, np.ndarray)
    assert reward.shape == (2, 3)


def test_sigmoid_variant_rejects_invalid_thresholds():
    with pytest.raises(AssertionError, match="x2"):
        SigmoidVariant(x1=1.0, x2=1.0)
