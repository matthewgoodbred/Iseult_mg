import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import fast_streamplot


def matplotlib_segments(x, y, u, v, density):
    fig, ax = plt.subplots()
    lines = ax.streamplot(x, y, u, v, density=density).lines
    plt.close(fig)
    return np.asarray(lines.get_segments())


def fast_segments(x, y, u, v, density):
    fig, ax = plt.subplots()
    result = fast_streamplot.streamplot(ax, x, y, u, v, density=density)
    n_arrows = len(result.arrows)
    plt.close(fig)
    return np.asarray(result.lines.get_segments()), n_arrows


def vortex():
    x = np.linspace(-3, 5, 200)
    y = np.linspace(-2, 2, 120)
    X, Y = np.meshgrid(x, y)
    return x, y, -Y + 0.3*np.sin(X), 0.5*X + np.cos(Y)


def noise_with_holes():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(40, 60))
    v = rng.normal(size=(40, 60))
    u[10:12, 20:25] = np.nan  # streamlines must stop at invalid data
    v[30, 5] = np.inf
    return np.arange(60.), 0.5*np.arange(40.), u, v


@pytest.mark.parametrize('field', [vortex, noise_with_holes])
@pytest.mark.parametrize('density', [1, 2, (1, 2)])
def test_same_lines_as_matplotlib(field, density):
    x, y, u, v = field()
    expected = matplotlib_segments(x, y, u, v, density)
    got, n_arrows = fast_segments(x, y, u, v, density)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)
    assert n_arrows > 0


def test_remove_takes_everything_off_the_axes():
    x, y, u, v = vortex()
    fig, ax = plt.subplots()
    before = len(ax.get_children())
    result = fast_streamplot.streamplot(ax, x, y, u, v)
    assert len(ax.get_children()) > before
    result.remove()
    assert len(ax.get_children()) == before
    result.remove()  # removing twice is harmless
    plt.close(fig)


def test_zero_field_has_no_lines():
    x = np.arange(10.)
    y = np.arange(8.)
    lines = fast_streamplot.trace(x, y, np.zeros((8, 10)), np.zeros((8, 10)))
    assert lines == []
