"""1D lineouts are cut across the region the 2D panels show."""
import numpy as np

import plot_axes


class FakeMain:
    def __init__(self, shape=(1, 20, 40), average=False, viewport=None, plane=0):
        self.MaxZInd, self.MaxYInd, self.MaxXInd = np.array(shape) - 1
        self.xSlice, self.ySlice, self.zSlice = 0, 0, 0
        self.lineout_viewport = viewport
        self.MainParamDict = {'Average1D': int(average), '2DSlicePlane': plane,
                              'SetxLim': False, 'SetyLim': False}


class FakePanel:
    c_omp = 2.0
    istep = 1.0

    def __init__(self, main, plot_axis=0):
        self.parent = main
        self.params = {'plot_axis': plot_axis}

    def GetPlotParam(self, key):
        return self.params[key]


def field(shape=(1, 20, 40)):
    # Each value encodes its own (y, x) indices, so a cut shows where it was taken.
    z, y, x = np.indices(shape)
    return 1000.0 * y + x + 0.0 * z


def test_no_viewport_cuts_through_the_domain_centre():
    panel = FakePanel(FakeMain())
    # y runs 0..10 c/omega_pe, so the centre is y = 5, index 10
    assert plot_axes.lineout_window(panel)['y'] == (10, 10)
    np.testing.assert_array_equal(plot_axes.lineout(panel, field()), 10000.0 + np.arange(40))


def test_centre_of_viewport():
    view = (('x', 2.0, 8.0), ('y', 1.0, 3.0))
    panel = FakePanel(FakeMain(viewport=view))
    assert plot_axes.lineout_window(panel)['y'] == (4, 4)
    assert plot_axes.slice_location(panel) == 2.0
    # plotted against y, the cut is at the centre of the viewport's x
    panel_y = FakePanel(FakeMain(viewport=view), plot_axis=1)
    assert plot_axes.lineout_window(panel_y)['x'] == (10, 10)
    np.testing.assert_array_equal(plot_axes.lineout(panel_y, field()), 1000.0 * np.arange(20) + 10)


def test_average_over_viewport():
    view = (('x', 2.0, 8.0), ('y', 1.0, 3.0))
    panel = FakePanel(FakeMain(viewport=view, average=True))
    assert plot_axes.lineout_window(panel)['y'] == (2, 6)
    np.testing.assert_allclose(plot_axes.lineout(panel, field()), 4000.0 + np.arange(40))


def test_viewport_is_clipped_to_the_domain():
    view = (('x', -5.0, 30.0), ('y', -4.0, 2.0))
    panel = FakePanel(FakeMain(viewport=view, average=True))
    assert plot_axes.lineout_viewport(panel) == {'x': (0.0, 20.0), 'y': (0.0, 2.0)}
    assert plot_axes.lineout_window(panel)['y'] == (0, 4)


def test_out_of_plane_axis_stays_on_the_plane():
    main = FakeMain(shape=(8, 20, 40), average=True, viewport=(('x', 0.0, 20.0), ('y', 0.0, 10.0)))
    main.zSlice = 3
    window = plot_axes.lineout_window(FakePanel(main))
    assert window['z'] == (3, 3)
    assert window['y'] == (0, 19)
