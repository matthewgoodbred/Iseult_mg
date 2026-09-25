import pathlib
import sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import phase_space
from phase_panel import PhasePanel


####
#
# phase_space
#
####

def params(**kwargs):
    return lambda name: kwargs[name]


def test_phase_axes_prefers_new_params():
    assert phase_space.phase_axes(params(phase_x='uy', phase_y='gamma',
                                         plot_axis=1, mom_dim=2)) == ('uy', 'gamma')


def test_phase_axes_falls_back_to_legacy_params():
    # A view saved before phase_x/phase_y existed
    assert phase_space.phase_axes(params(phase_x=None, phase_y=None,
                                         plot_axis=1, mom_dim=2)) == ('y', 'uz')
    assert phase_space.phase_axes(params(plot_axis=0, mom_dim=0)) == ('x', 'ux')


def test_momentum_keys_needed():
    assert phase_space.momentum_keys_needed(1, ('x', 'y')) == []
    assert phase_space.momentum_keys_needed(1, ('x', 'uy')) == ['ve']
    assert phase_space.momentum_keys_needed(0, ('ux', 'uz')) == ['ui', 'wi']
    assert phase_space.momentum_keys_needed(0, ('x', 'gamma')) == ['ui', 'vi', 'wi']
    assert phase_space.momentum_keys_needed(0, ('x', 'betay')) == ['ui', 'vi', 'wi']
    assert phase_space.momentum_keys_needed(0, ('x', 'y'), all_components=True) == ['ui', 'vi', 'wi']


def make_quantities(boost=None):
    data = {'ue': np.array([0.0, 1.0, 3.0]),
            've': np.array([0.0, 2.0, 0.0]),
            'we': np.array([0.0, 2.0, 4.0])}
    pos = {'x': np.array([1.0, 2.0, 3.0]), 'y': np.array([4.0, 5.0, 6.0])}
    return phase_space.ParticleQuantities(data.__getitem__, pos.get, 1, boost)


def test_quantities_unboosted():
    q = make_quantities()
    np.testing.assert_allclose(q('gamma'), [1.0, np.sqrt(10.0), np.sqrt(26.0)])
    np.testing.assert_allclose(q('uy'), [0.0, 2.0, 0.0])
    np.testing.assert_allclose(q('betax'), [0.0, 1.0/np.sqrt(10.0), 3.0/np.sqrt(26.0)])
    np.testing.assert_allclose(q('y'), [4.0, 5.0, 6.0])
    assert q('z') is None
    # |beta| < 1 for every particle
    beta2 = q('betax')**2 + q('betay')**2 + q('betaz')**2
    assert np.all(beta2 < 1)


def test_boost_matches_velocity_addition():
    boost = phase_space.boost_factors(0.6)
    big_gamma, beta = boost
    assert big_gamma == pytest.approx(1.25)
    q = make_quantities(boost)
    lab = make_quantities()
    # Invariant: gamma^2 - u^2 = 1 in the boosted frame too
    np.testing.assert_allclose(q('gamma')**2 - q('ux')**2 - q('uy')**2 - q('uz')**2, 1.0)
    # The transverse four-velocity is unchanged by a boost along x
    np.testing.assert_allclose(q('uz'), lab('uz'))
    # Relativistic velocity addition for vx
    vx = lab('betax')
    np.testing.assert_allclose(q('betax'), (vx - beta)/(1 - vx*beta))
    # Energy cuts are made in the lab frame
    np.testing.assert_allclose(q.lab_gamma(), lab('gamma'))


def test_boost_factor_conventions():
    assert phase_space.boost_factors(0.0) is None
    g, b = phase_space.boost_factors(2.0)
    assert g == 2.0 and b == pytest.approx(np.sqrt(0.75))
    g, b = phase_space.boost_factors(-2.0)
    assert g == 2.0 and b == pytest.approx(-np.sqrt(0.75))


def test_histogram_orientation_and_ranges():
    h = np.array([0.1, 0.1, 0.9])
    v = np.array([5.0, 5.0, -5.0])
    img, vrange, hrange, clim = phase_space.histogram(h, v, (0.0, 1.0), (-10.0, 10.0), 10, 4)
    # vertical along the first axis, as imshow expects
    assert img.shape == (4, 10)
    assert hrange == [0.0, 1.0] and vrange == [-10.0, 10.0]
    assert img[3, 1] == 1.0  # the two particles at (0.1, 5) are the max
    assert img[1, 9] == pytest.approx(0.5)
    assert clim == [pytest.approx(0.5), 1.0]


def test_histogram_with_no_particles():
    img, _, _, clim = phase_space.histogram(np.array([]), np.array([]), (0, 1), (0, 1), 5, 5)
    assert img.shape == (5, 5)
    assert clim == [0.1, 1]


def test_labels():
    assert phase_space.axis_label('ux', 0) == r'$\gamma_i\beta_{x,i}$'
    assert phase_space.axis_label('betaz', 1) == r'$\beta_{z,e}$'
    assert phase_space.axis_label('gamma', 1, boosted=True) == r'$\gamma\prime_e$'
    assert phase_space.axis_label('x', 1, boosted=True) == r'$x\prime\ [c/\omega_{\rm pe}]$'
    assert phase_space.axis_label('y', 1, boosted=True) == r'$y\ [c/\omega_{\rm pe}]$'


####
#
# The headless panel used for movies
#
####

class MockParent:
    def __init__(self):
        self.MainParamDict = {'ColorMap': 'viridis', 'NumOfRows': 1, 'NumOfCols': 1,
                              '2DSlicePlane': 0, 'SetxLim': False, 'SetyLim': False,
                              'LinkSpatial': 1, 'DoLorentzBoost': False, 'GammaBoost': 0.0}
        self.figure = None
        self.SubPlotList = None
        self.ion_color = 'r'
        self.electron_color = 'b'


class MockOutput:
    def __init__(self):
        self.c_omp = 2.0
        self.istep = 1.0
        self.me = 1.0
        self.mi = 16.0
        self.bx = np.zeros((1, 10, 20))
        self.xe = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        self.ye = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        self.ue = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.ve = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])
        self.we = np.zeros(5)
        self.che = np.ones(5)


def test_headless_panel_momentum_momentum():
    panel = PhasePanel(MockParent(), (0, 0), {'prtl_type': 1, 'phase_x': 'ux', 'phase_y': 'uy',
                                              'xbins': 8, 'pbins': 6, 'filter_by_viewport': False})
    panel.update_data(MockOutput())
    img, vrange, hrange, _ = panel.hist2d
    assert img.shape == (6, 8)
    assert hrange == [pytest.approx(0.1), pytest.approx(0.5)]
    assert vrange == [pytest.approx(-0.5), pytest.approx(-0.1)]
    assert img.count() == 5  # five particles in five different bins


def test_headless_panel_position_position():
    panel = PhasePanel(MockParent(), (0, 0), {'prtl_type': 1, 'phase_x': 'y', 'phase_y': 'x',
                                              'filter_by_viewport': False})
    panel.update_data(MockOutput())
    _, vrange, hrange, _ = panel.hist2d
    # The domain along each position: 10 cells in y, 20 in x, over c_omp = 2
    assert hrange == [0.0, 5.0]
    assert vrange == [0.0, 10.0]
    panel.IntRegionLines = []
    panel.UpdateLabelsandColors()
    assert panel.x_label.startswith('$y')


def test_headless_panel_missing_coordinate_falls_back_to_x():
    panel = PhasePanel(MockParent(), (0, 0), {'prtl_type': 1, 'phase_x': 'z', 'phase_y': 'gamma',
                                              'filter_by_viewport': False})
    panel.update_data(MockOutput())
    assert panel.shown_axes() == ('x', 'gamma')


def test_headless_panel_legacy_view():
    panel = PhasePanel(MockParent(), (0, 0), {'prtl_type': 1, 'mom_dim': 1,
                                              'filter_by_viewport': False})
    panel.update_data(MockOutput())
    assert panel.shown_axes() == ('x', 'uy')


def test_limited_range():
    get = params(set_h_min=True, h_min=2.0, set_h_max=False, h_max=9.0,
                 set_p_min=True, p_min=5.0, set_p_max=True, p_max=1.0)
    assert phase_space.limited_range((0.0, 10.0), get, 'h') == (2.0, 10.0)
    # An empty range is ignored
    assert phase_space.limited_range((0.0, 10.0), get, 'p') == (0.0, 10.0)
    assert phase_space.limits_key(get) == '_h_min_2.0_p_min_5.0_p_max_1.0'


def test_headless_panel_bins_over_set_limits():
    panel = PhasePanel(MockParent(), (0, 0), {'prtl_type': 1, 'phase_x': 'ux', 'phase_y': 'uy',
                                              'set_h_min': True, 'h_min': 0.25,
                                              'set_p_max': True, 'p_max': 0.0,
                                              'filter_by_viewport': False})
    panel.update_data(MockOutput())
    _, vrange, hrange, _ = panel.hist2d
    assert hrange == [0.25, pytest.approx(0.5)]
    assert vrange == [pytest.approx(-0.5), 0.0]
