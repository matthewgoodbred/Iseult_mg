'''The frame the user has zoomed into has to survive the plot changing.

Toggling something on a panel makes Iseult clear the figure and draw it again
from scratch, and stepping in time makes it refresh the panels in place. Both
are meant to leave the user looking at the same region they were looking at
before, which is what MainApp.SaveView and MainApp.LoadView are for.
'''

import gc
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import NavigationToolbar2
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from view_state import view_limits, view_from_limits

MAIN_APP = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'main_app.py'


def _view_mixin():
    '''SaveView and LoadView, lifted out of main_app.

    main_app cannot be imported here: it selects the QtAgg backend at import
    time, which needs a display. The two methods themselves only touch the
    toolbar and the panel list, so they are read out of the file and given a
    class of their own to live in.'''
    lines = MAIN_APP.read_text().split('\n')
    first = next(i for i, l in enumerate(lines) if l.strip().startswith('def SaveView'))
    last = next(i for i, l in enumerate(lines)
                if i > first and l.strip().startswith('def RenewCanvas'))
    namespace = {'view_limits': view_limits, 'view_from_limits': view_from_limits}
    exec('class ViewMixin:\n' + '\n'.join(lines[first:last]), namespace)
    return namespace['ViewMixin']


class FakeGraph:
    def __init__(self, axes):
        self.axes = axes


class FakeSubPlot:
    chartType = 'FieldsPlot'

    def __init__(self):
        self.graph = None
        self.Changedto1D = False
        self.Changedto2D = False
        self.PlotParamsDict = {'FieldsPlot': {'spatial_x': True}}


class FakeApp(_view_mixin()):
    '''One panel showing a domain that runs from 0 to 100 in both directions.'''

    DOMAIN = (0.0, 100.0, 0.0, 100.0)

    def __init__(self):
        self.MainParamDict = {'NumOfRows': 1, 'NumOfCols': 1, 'xLimsRelative': 0}
        self.f = plt.figure()
        self.toolbar = NavigationToolbar2(self.f.canvas)
        self.cbarList = []
        self.shock_loc = 0.0
        self.prev_shock_loc = 0.0
        self.SubPlotList = [[FakeSubPlot()]]
        self.prev_ctype_list = [['FieldsPlot']]
        self.diff_from_home = []
        # What the panels would frame if nobody had zoomed, i.e. what they set
        # from the data of the timestep they are showing.
        self.panel_limits = FakeApp.DOMAIN
        self._draw_panels()

    def _draw_panels(self):
        axes = self.f.add_subplot(111)
        axes.plot([0, 100], [0, 100])
        self._frame_domain(axes)
        self.SubPlotList[0][0].graph = FakeGraph(axes)

    def _frame_domain(self, axes):
        '''What a panel's own draw() or refresh() does: frame its own data.'''
        axes.set_xlim(self.panel_limits[0], self.panel_limits[1])
        axes.set_ylim(self.panel_limits[2], self.panel_limits[3])

    @property
    def axes(self):
        return self.SubPlotList[0][0].graph.axes

    def limits(self):
        return tuple(float(v) for v in self.axes.get_xlim() + self.axes.get_ylim())

    def redraw(self):
        '''The essentials of ReDrawCanvas(keep_view = True).

        A redraw is what a panel setting, e.g. turning streamlines off, asks
        for: the figure is cleared and every panel is built again.'''
        if self.toolbar._nav_stack() is None:
            return None
        self.SaveView()
        self.toolbar._nav_stack.clear()
        self.f.clf()
        gc.collect() # the toolbar holds its views by weak reference
        self.cbarList = []
        self._draw_panels()
        self.LoadView()
        return self.limits()

    def refresh(self):
        '''The essentials of RefreshCanvas(keep_view = True).

        A refresh is what stepping in time asks for: the axes stay, and each
        panel writes the new timestep's data and limits into them.'''
        if self.toolbar._nav_stack() is None:
            return None
        self.SaveView()
        self.toolbar._nav_stack.clear()
        self._frame_domain(self.axes)
        self.LoadView()
        return self.limits()

    def zoom(self, xlim, ylim):
        '''Zoom the way the toolbar's rubber band does, stack and all.'''
        if self.toolbar._nav_stack() is None:
            self.toolbar.push_current() # press_zoom records where we came from
        self.axes.set_xlim(*xlim)
        self.axes.set_ylim(*ylim)
        self.toolbar.push_current() # release_zoom records where we ended up


@pytest.fixture
def app():
    application = FakeApp()
    yield application
    plt.close(application.f)


def test_zoom_survives_repeated_redraws(app):
    # Every panel setting that is toggled is another redraw, so the frame has
    # to come back each time and not just the first.
    app.zoom((20, 30), (40, 50))
    for _ in range(3):
        assert app.redraw() == (20.0, 30.0, 40.0, 50.0)


def test_zoom_survives_redraws_and_refreshes_mixed(app):
    app.zoom((20, 30), (40, 50))
    assert app.refresh() == (20.0, 30.0, 40.0, 50.0)
    assert app.redraw() == (20.0, 30.0, 40.0, 50.0)
    assert app.refresh() == (20.0, 30.0, 40.0, 50.0)


def test_axis_the_user_left_alone_follows_the_data(app):
    # Only x is zoomed; y is left where the panel put it.
    app.zoom((20, 30), FakeApp.DOMAIN[2:])
    # The panel now wants a different y, as it would after the data changed.
    app.panel_limits = (0.0, 100.0, 200.0, 300.0)
    assert app.redraw() == (20.0, 30.0, 200.0, 300.0)


def test_later_zoom_replaces_the_earlier_one(app):
    app.zoom((20, 30), (40, 50))
    app.redraw()
    app.zoom((5, 7), (8, 9))
    assert app.redraw() == (5.0, 7.0, 8.0, 9.0)


def test_home_still_returns_to_the_whole_domain(app):
    app.zoom((20, 30), (40, 50))
    app.redraw()
    app.toolbar.home()
    assert app.limits() == FakeApp.DOMAIN


def test_unzoomed_figure_is_not_pinned(app):
    # Nothing has been zoomed, so there is no frame to keep and the panels are
    # free to choose their own limits.
    assert app.redraw() is None


def test_view_limits_reads_both_forms_of_view():
    assert view_limits({'xlim': (1, 2), 'ylim': (3, 4)}) == (1, 2, 3, 4)
    assert view_limits((1, 2, 3, 4)) == (1, 2, 3, 4)


def test_view_from_limits_turns_off_autoscale_only_where_the_user_zoomed():
    view = {'xlim': (0, 1), 'autoscalex_on': True,
            'ylim': (0, 1), 'autoscaley_on': True}
    rebuilt = view_from_limits(view, (5, 6, 7, 8), [True, True, False, False])
    assert rebuilt['xlim'] == (5, 6)
    assert rebuilt['ylim'] == (7, 8)
    assert rebuilt['autoscalex_on'] is False
    assert rebuilt['autoscaley_on'] is True
    assert view['xlim'] == (0, 1) # the view passed in is left alone
