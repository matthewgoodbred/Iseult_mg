import types

import numpy as np
import pytest

from moments import MomentsPanel


class FakeWrap:
    '''Just enough of SubPlotWrapper to hold a Moments panel's params.'''
    def __init__(self, parent, **params):
        self.parent = parent
        self.params = dict(MomentsPanel.plot_param_dict)
        self.params.update(params)
        self.PlotTypeDict = {'Moments': MomentsPanel}
        self.chartType = 'Moments'
        self.figure = None
        self.pos = (0, 0)
        self.renewed = 0

    def GetPlotParam(self, key):
        return self.params[key]

    def SetPlotParam(self, key, value, update_plot=True, NeedsRedraw=False):
        self.params[key] = value


def make_panel(**params):
    parent = types.SimpleNamespace(
        MainParamDict={'2DSlicePlane': 0, 'PrtlStride': 1},
        is_viewport_zoomed=lambda: False,
        get_active_viewport=lambda: None)
    wrap = FakeWrap(parent, **params)
    return MomentsPanel(parent, wrap)


@pytest.mark.parametrize('params, family, comps', [
    # Configs saved before the stress-energy tensor existed
    ({'m_type': 0, 'show_x': True, 'show_y': False, 'show_z': True}, 'beta', ['x', 'z']),
    ({'m_type': 1, 'show_x': False, 'show_y': True, 'show_z': False}, 'u', ['y']),
    ({'m_type': 2, 'show_x': True, 'show_y': True}, 'energy', ['ke', 'thermal']),
    # New configs
    ({'m_type': 3, 'components': 'xx,00,bogus'}, 'T', ['00', 'xx']),
    ({'m_type': 3, 'basis': 1, 'components': 'xx'}, 'T', ['00']),  # not in this basis
    ({'m_type': 0, 'components': 'none'}, 'beta', []),
])
def test_selected_components(params, family, comps):
    panel = make_panel(**params)
    assert panel.family() == family
    assert panel.selected_components() == comps


def test_two_d_shows_one_component_and_species():
    panel = make_panel(m_type=3, components='00,0x', twoD=1, species_2d=1)
    assert panel.selected_components() == ['00']
    assert panel.shown_species() == [1]
    assert panel.spatial_plot_axes() == ('x', 'y')


def test_changing_the_component_does_not_change_what_is_binned():
    '''Picking another component must reuse the cached particle sums.'''
    panel = make_panel(m_type=3, components='00')
    key = panel.binning_spec()['key_lab']
    for params in ({'components': 'xy'}, {'m_type': 0, 'components': 'x'},
                   {'normalization': 1}, {'show_ions': False}, {'mass_weight': True}):
        panel.FigWrap.params.update(params)
        assert panel.binning_spec()['key_lab'] == key


def test_field_aligned_components_ask_for_the_field_pass():
    panel = make_panel(m_type=3, components='00,parpar', basis=1)
    spec = panel.binning_spec()
    assert spec['field_aligned']
    # The field-aligned sums hold the lab ones too, so they serve a lab request
    lab = make_panel(m_type=3, components='00').binning_spec()
    assert MomentsPanel.cached({spec['key_fa']: 'fa sums'}, lab) == 'fa sums'
    assert MomentsPanel.cached({lab['key_lab']: 'lab sums'}, spec) is None


def test_binning_changes_change_the_key():
    panel = make_panel()
    key = panel.binning_spec()['key_lab']
    for params in ({'xbins': 50}, {'weighted': True}, {'plot_axis': 1}, {'twoD': 1}):
        other = make_panel(**params)
        assert other.binning_spec()['key_lab'] != key
