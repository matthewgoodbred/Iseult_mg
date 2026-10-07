'''Run the Ohm's-law and pressure-balance panels end to end on the sample data,
with a stand-in for MainApp, and draw them into a headless figure.'''
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import data_loading
from balance_panels import OhmsLawPanel, PressureBalancePanel

DATA = pathlib.Path(__file__).resolve().parent / 'data'
RUNS = {'v1': {'Param': DATA / 'tristan_v1' / 'param.041', 'Flds': DATA / 'tristan_v1' / 'flds.tot.041',
               'Prtl': DATA / 'tristan_v1' / 'prtl.tot.041'},
        'v2': {'Param': DATA / 'tristan_v2' / 'standard_structure' / 'params.00070',
               'Flds': DATA / 'tristan_v2' / 'standard_structure' / 'flds' / 'flds.tot.00070',
               'Prtl': DATA / 'tristan_v2' / 'standard_structure' / 'prtl' / 'prtl.tot.00070'}}
FALLBACKS = {'c': 0.45, 'ppc0': np.nan}


class FakeTimeStep:
    value = 1


class FakeApp:
    def __init__(self, run):
        self.PathDict = {kind: [path] for kind, path in RUNS[run].items()}
        self.TimeStep = FakeTimeStep()
        self.NewDirectory = False
        self.MainParamDict = {'Average1D': 0, 'PrtlStride': 1, '2DSlicePlane': 0, 'SetxLim': False,
                              'SetyLim': False, 'NumFontSize': 8, 'xLabelPad': 0, 'yLabelPad': 0,
                              'AxLabelSize': 8, 'legendLabelSize': 7}
        shape = data_loading.dataset_shape(RUNS[run]['Flds'], 'bx')
        self.MaxZInd, self.MaxYInd, self.MaxXInd = np.array(shape) - 1
        self.xSlice, self.ySlice, self.zSlice = self.MaxXInd // 2, self.MaxYInd // 2, 0
        self.DataDict = {}
        self.figure = Figure()
        self.gs0 = gridspec.GridSpec(1, 1)
        self.axes_extent = [4, 90, 0, 92]

    def load(self, keys):
        for key in keys:
            kind = 'Flds' if key in ('ex', 'ey', 'ez', 'bx', 'by', 'bz') else 'Param'
            try:
                self.DataDict[key] = data_loading.load_dataset(self.PathDict[kind][0], key)
            except KeyError:
                self.DataDict[key] = FALLBACKS[key]

    def GetSharedAxes(self, pos):
        return None, None


class FakeWrap:
    def __init__(self, app, panel_class, **params):
        self.parent = app
        self.PlotTypeDict = {'OhmsLaw': OhmsLawPanel, 'PressureBalance': PressureBalancePanel}
        self.chartType = [k for k, v in self.PlotTypeDict.items() if v is panel_class][0]
        self.figure = app.figure
        self.pos = (0, 0)
        self.params = dict(panel_class.plot_param_dict)
        self.params.update(params)
        self.graph = panel_class(app, self)

    def LoadKey(self, key):
        return self.parent.DataDict[key]

    def GetPlotParam(self, key):
        return self.params[key]

    def SetPlotParam(self, key, value, update_plot=True, NeedsRedraw=False):
        self.params[key] = value


def run_panel(run, panel_class, **params):
    app = FakeApp(run)
    wrap = FakeWrap(app, panel_class, **params)
    app.load(wrap.graph.set_plot_keys())
    wrap.graph.LoadData()
    wrap.graph.draw()
    return wrap.graph


@pytest.mark.parametrize('run', ['v1', 'v2'])
@pytest.mark.parametrize('params', [{}, {'plot_axis': 1}, {'time_deriv': True, 'split_pressure': True},
                                    {'species': 0, 'component': 2, 'nbins': 4, 'smooth': 1}])
def test_ohms_law_panel_draws(run, params):
    graph = run_panel(run, OhmsLawPanel, **params)
    keys = [s[0] for s in graph.series]
    assert 'E' in keys and 'residual' in keys
    # the only output has no neighbours, so there is no d/dt to show
    assert 'dpdt' not in keys
    assert len(graph.lines) == len(graph.series)
    for key, label, values, style in graph.series:
        assert np.shape(values) == (graph.stencil.nh,)
    assert 'Binned' in graph.status_text


@pytest.mark.parametrize('run', ['v1', 'v2'])
@pytest.mark.parametrize('params', [{}, {'mode': 1}, {'species': 3, 'split_pressure': True},
                                    {'mode': 1, 'force_terms': 'none', 'stress_terms': 'p_xx,total', 'species': 3},
                                    {'time_deriv': True, 'component': 1},
                                    {'integrate': True, 'species': 3, 'split_pressure': True},
                                    {'integrate': True, 'component': 1, 'force_terms': ','.join(
                                        PressureBalancePanel.TERM_NAMES)}])
def test_pressure_balance_panel_draws(run, params):
    graph = run_panel(run, PressureBalancePanel, **params)
    keys = [s[0] for s in graph.series]
    if params.get('mode') == 1:
        assert 'total' in keys
    else:
        assert 'residual' in keys and 'mag_pressure' in keys
    assert len(graph.lines) == len(graph.series)
    # the legend's labels are valid mathtext
    FigureCanvasAgg(graph.figure).draw()


def test_average_1d_drops_the_transverse_stencil():
    app = FakeApp('v2')
    app.MainParamDict['Average1D'] = 1
    wrap = FakeWrap(app, OhmsLawPanel)
    app.load(wrap.graph.set_plot_keys())
    wrap.graph.LoadData()
    assert wrap.graph.stencil.nv == 1


def test_sums_are_cached_across_panels():
    app = FakeApp('v2')
    first = FakeWrap(app, OhmsLawPanel)
    app.load(first.graph.set_plot_keys())
    first.graph.LoadData()
    second = FakeWrap(app, PressureBalancePanel)
    second.graph.LoadData()
    assert len(app._balance_cache) == 2   # the particle sums and the binned fields, once each
