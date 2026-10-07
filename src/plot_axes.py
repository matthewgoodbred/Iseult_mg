#!/usr/bin/env python
"""Helpers that let a panel be plotted against either physical spatial axis.

Iseult was originally written assuming that x is the primary axis: every 1D
lineout, phase-space histogram and moment was a function of x, and the
horizontal axis of a 2D panel was always the first of the two in-plane
coordinates.  This module breaks that assumption.  A panel records

    'plot_axis' -- the physical axis a 1D panel is a function of, and
    'rotate_90' -- whether a 2D panel is transposed so that the second
                   in-plane coordinate runs horizontally,

and everything that used to be hard-coded to x (labels, array slicing,
particle coordinates, which axes are shared between panels) is derived from
those two params here so that the panels stay in step with each other.
"""
import qt_compat as Tk
from qt_compat import ttk

import numpy as np

# The physical axes a 1D panel can be plotted against.
ONE_D_AXES = ('x', 'y')

AXIS_LABELS = {'x': r'$x\ [c/\omega_{\rm pe}]$',
               'y': r'$y\ [c/\omega_{\rm pe}]$',
               'z': r'$z\ [c/\omega_{\rm pe}]$'}

# The (horizontal, vertical) physical axes of each '2DSlicePlane' before rotation.
SLICE_PLANE_AXES = (('x', 'y'), ('x', 'z'), ('y', 'z'))

# Where each physical axis lives in the (z, y, x) field arrays.
FIELD_AXIS_INDEX = {'z': 0, 'y': 1, 'x': 2}

# Particle position keys, indexed by [prtl_type][axis]. prtl_type 0 is ions.
PRTL_POS_KEYS = {0: {'x': 'xi', 'y': 'yi', 'z': 'zi'},
                 1: {'x': 'xe', 'y': 'ye', 'z': 'ze'}}

# The Tristan v2 names of those datasets, needed to check whether the data
# actually contains a coordinate before we ask for it.
_V2_POS_NAMES = {'xi': 'x_2', 'yi': 'y_2', 'zi': 'z_2',
                 'xe': 'x_1', 'ye': 'y_1', 'ze': 'z_1'}


####
#
# Plot params
#
####

def add_axis_params(param_dictionary, two_d=True):
    """Add the axis-orientation params to a panel's default param dictionary.

    Parameters
    ----------
    param_dictionary : dict
        The plot_param_dict of the panel class.
    two_d : bool
        Whether the panel can be shown as a 2D image, and therefore needs the
        90 degree rotation toggle.
    """
    param_dictionary['plot_axis'] = 0  # An index into ONE_D_AXES
    if two_d:
        param_dictionary['rotate_90'] = False


def plot_axis_name(panel):
    """The physical axis a 1D panel is a function of."""
    try:
        return ONE_D_AXES[panel.GetPlotParam('plot_axis')]
    except (KeyError, IndexError):
        return 'x'


def is_rotated(panel):
    """Whether a 2D panel has its two spatial axes swapped."""
    try:
        return bool(panel.GetPlotParam('rotate_90'))
    except KeyError:
        return False


def two_d_axes(panel):
    """The (horizontal, vertical) physical axes of a 2D panel."""
    horiz, vert = SLICE_PLANE_AXES[panel.parent.MainParamDict['2DSlicePlane']]
    if is_rotated(panel):
        horiz, vert = vert, horiz
    return horiz, vert


def plot_axes_of(panel):
    """The (horizontal, vertical) physical axes of any panel.

    Either entry is None when that axis of the plot is not a spatial one, e.g.
    the vertical axis of a phase plot is momentum, so it is None. A panel whose
    'twoD' param does not mean 'a 2D map of space' -- a phase plot is drawn as
    an image but only one of its axes is spatial -- says so by defining
    spatial_plot_axes.
    """
    if hasattr(panel, 'spatial_plot_axes'):
        return panel.spatial_plot_axes()
    try:
        two_dimensional = panel.GetPlotParam('twoD')
    except KeyError:
        two_dimensional = False
    if two_dimensional:
        return two_d_axes(panel)
    return plot_axis_name(panel), None


####
#
# Slicing field arrays
#
####

def _clip(index, length):
    """Keep a slice index inside an array that may be smaller than expected."""
    return min(max(int(index), 0), max(length - 1, 0))


def axis_values(panel, arr, axis=None):
    """Grid coordinates in c/omega_pe along `axis` for a (z, y, x) field array."""
    axis = plot_axis_name(panel) if axis is None else axis
    return np.arange(arr.shape[FIELD_AXIS_INDEX[axis]]) / panel.c_omp * panel.istep


def averages_lineouts(panel):
    """Whether 1D lineouts are averaged over the viewport rather than cut through its centre."""
    return bool(panel.parent.MainParamDict['Average1D'])


def lineout_viewport(panel):
    """The region of the slice plane that 1D lineouts are taken from.

    A dict mapping each of the two in-plane physical axes to a (low, high)
    range in c/omega_pe. It is the region shown by the first 2D spatial panel,
    as recorded by MainApp before the panels are drawn, and any axis that does
    not constrain falls back to the main window's limits or the whole domain.
    """
    main = panel.parent
    shown = {axis: (low, high) for axis, low, high in getattr(main, 'lineout_viewport', None) or ()
             if axis is not None}
    ranges = {}
    for axis in SLICE_PLANE_AXES[main.MainParamDict['2DSlicePlane']]:
        low, high = shown.get(axis) or limits_for_axis(panel, axis) or full_extent(panel, axis)
        # A view panned past the edge of the domain holds no data out there.
        top = domain_extent(panel, axis)
        low, high = min(max(low, 0.0), top), min(max(high, 0.0), top)
        ranges[axis] = (min(low, high), max(low, high))
    return ranges


def lineout_window(panel, axis=None):
    """The field-array indices a 1D lineout along `axis` is taken over.

    A dict mapping each of the two axes across the lineout to an inclusive
    (first, last) index range. An axis in the slice plane is cut through the
    centre of the viewport, or spans the whole of it when lineouts are
    averaged; the axis out of the plane is held at the plane's own slice, so
    the lineout always lies in the plane the 2D panels show.
    """
    axis = plot_axis_name(panel) if axis is None else axis
    main = panel.parent
    ranges = lineout_viewport(panel)
    average = averages_lineouts(panel)
    scale = panel.c_omp / panel.istep  # indices per c/omega_pe
    window = {}
    for other in FIELD_AXIS_INDEX:
        if other == axis:
            continue
        if other not in ranges:
            index = getattr(main, other + 'Slice', 0)
            window[other] = (index, index)
            continue
        low, high = ranges[other]
        center = int(np.around(0.5 * (low + high) * scale))
        first, last = int(np.ceil(low * scale - 1e-9)), int(np.floor(high * scale + 1e-9))
        top = {'x': main.MaxXInd, 'y': main.MaxYInd, 'z': main.MaxZInd}[other]
        first, last, center = (min(max(i, 0), top) for i in (first, last, center))
        if not average or last < first:
            # A window narrower than a cell holds only its nearest node.
            first = last = center
        window[other] = (first, last)
    return window


def lineout(panel, arr, axis=None):
    """A 1D cut along `axis` through a (z, y, x) field array.

    The cut is taken across the region the 2D panels show, see lineout_window:
    through its centre, or averaged over it when the main 'Average1D' setting
    is on.
    """
    axis = plot_axis_name(panel) if axis is None else axis
    arr = np.asanyarray(arr)
    if arr.ndim == 1:
        # A user defined function is allowed to hand back a 1D array.
        return arr
    kept = FIELD_AXIS_INDEX[axis]
    index = [slice(None)] * arr.ndim
    for other, (first, last) in lineout_window(panel, axis).items():
        dim = FIELD_AXIS_INDEX[other]
        index[dim] = slice(_clip(first, arr.shape[dim]), _clip(last, arr.shape[dim]) + 1)
    return np.mean(arr[tuple(index)], axis=tuple(i for i in range(arr.ndim) if i != kept))


def two_d_slice(panel, arr):
    """The 2D image of `arr` for the current slice plane and rotation."""
    arr = np.asanyarray(arr)
    plane = panel.parent.MainParamDict['2DSlicePlane']
    if plane == 0:  # x-y plane
        img = arr[_clip(panel.parent.zSlice, arr.shape[0]), :, :]
    elif plane == 1:  # x-z plane
        img = arr[:, _clip(panel.parent.ySlice, arr.shape[1]), :]
    else:  # y-z plane
        img = arr[:, :, _clip(panel.parent.xSlice, arr.shape[2])]
    if is_rotated(panel):
        img = img.T
    return img


def slice_location(panel, axis=None):
    """Where, in c/omega_pe, the 1D cut along `axis` is taken from.

    A cut along x is centred on the y of the viewport, a cut along y on its x.
    """
    axis = plot_axis_name(panel) if axis is None else axis
    first, last = lineout_window(panel, axis)['y' if axis == 'x' else 'x']
    return 0.5 * (first + last) / panel.c_omp * panel.istep


####
#
# Particle positions
#
####

def available_position_keys(panel, prtl_type, axes):
    """Which of the position keys for `axes` the particle data actually holds."""
    keys = []
    try:
        import h5py
        prtl_file = panel.FigWrap.parent.PathDict['Prtl'][0]
        with h5py.File(prtl_file, 'r') as f:
            for axis in axes:
                key = PRTL_POS_KEYS[prtl_type][axis]
                if key in f or _V2_POS_NAMES[key] in f:
                    keys.append(key)
    except Exception:
        pass
    return keys


def load_positions(panel, prtl_type, axis, n_expected=None):
    """Particle positions along `axis` in c/omega_pe, or None if unavailable.

    Iseult substitutes a dummy array for a dataset that a file does not hold,
    so a position that does not line up with the momenta is treated as missing
    rather than being silently used.
    """
    try:
        coord = panel.FigWrap.LoadKey(PRTL_POS_KEYS[prtl_type][axis])
    except KeyError:
        return None
    coord = np.asanyarray(coord)
    if coord.ndim != 1:
        return None
    if n_expected is not None and coord.shape[0] != n_expected:
        return None
    return coord / panel.c_omp


def filter_by_viewport(panel, viewport, prtl_type, in_range):
    """Restrict `in_range` to the particles inside the 2D region `viewport`.

    `viewport` is the sequence of (axis, low, high) triples returned by
    MainApp.get_active_viewport. The mask is modified in place where possible;
    the returned value is the extent of the region along the panel's own plot
    axis, or None when the viewport does not constrain that axis.
    """
    own_axis = plot_axis_name(panel)
    own_range = None
    for axis, low, high in viewport:
        if axis is None:
            continue
        coord = load_positions(panel, prtl_type, axis, n_expected=len(in_range))
        if coord is None:
            continue
        in_range &= (coord >= low) & (coord <= high)
        if axis == own_axis:
            own_range = (low, high)
    return own_range


def viewport_key(viewport):
    """A short string identifying `viewport`, for the DataDict cache keys."""
    if viewport is None:
        return ''
    return '_vp' + ''.join(f'_{axis}_{low:.4f}_{high:.4f}'
                           for axis, low, high in viewport if axis is not None)


####
#
# Settings-panel widgets
#
####

def add_axis_buttons(frm, settings, panel, row, column=0, columnspan=2, two_d=True,
                     on_change=None):
    """Add the 'plot vs' radio buttons, and the 2D rotation toggle, to a settings pane.

    Parameters
    ----------
    frm : ttk.Frame
        The frame the settings pane lays its widgets out in.
    settings : Tk.Toplevel
        The settings window, which the Tk variables are hung off of so that
        they are not garbage collected.
    panel :
        The panel whose params the buttons change.
    row, column, columnspan :
        Where in `frm` to grid the controls.
    two_d : bool
        Whether to include the 90 degree rotation toggle.
    on_change : callable, optional
        Called with no arguments after the axis is changed, so that a settings
        pane can refresh any labels naming the axis it is plotted against.
    """
    frame = ttk.Frame(frm)

    ttk.Label(frame, text='Plot vs:').pack(side=Tk.LEFT, expand=0)
    settings.PlotAxisVar = Tk.IntVar()
    settings.PlotAxisVar.set(panel.GetPlotParam('plot_axis'))
    for i, name in enumerate(ONE_D_AXES):
        ttk.Radiobutton(frame,
                        text=name,
                        variable=settings.PlotAxisVar,
                        value=i,
                        command=lambda: _plot_axis_handler(settings, panel, on_change)).pack(side=Tk.LEFT, expand=0)

    if two_d:
        settings.Rotate90Var = Tk.IntVar()
        settings.Rotate90Var.set(panel.GetPlotParam('rotate_90'))
        ttk.Checkbutton(frame,
                        text='Rotate 2D 90 deg',
                        variable=settings.Rotate90Var,
                        command=lambda: _rotate_handler(settings, panel)).pack(side=Tk.LEFT, expand=0)

    frame.grid(row=row, column=column, columnspan=columnspan, sticky=Tk.W)
    return frame


def _plot_axis_handler(settings, panel, on_change=None):
    if settings.PlotAxisVar.get() != panel.GetPlotParam('plot_axis'):
        panel.SetPlotParam('plot_axis', settings.PlotAxisVar.get(), NeedsRedraw=True)
        if on_change is not None:
            on_change()


def _rotate_handler(settings, panel):
    if bool(settings.Rotate90Var.get()) != bool(panel.GetPlotParam('rotate_90')):
        panel.SetPlotParam('rotate_90', bool(settings.Rotate90Var.get()), NeedsRedraw=True)


def grid_values(panel, axis):
    """Grid coordinates in c/omega_pe along `axis` for the whole domain.

    Cached in the parent's DataDict so that every panel showing the same
    timestep agrees on them.
    """
    key = axis + 'axis_values'
    if key in panel.parent.DataDict:
        return panel.parent.DataDict[key]
    max_index = {'x': panel.parent.MaxXInd,
                 'y': panel.parent.MaxYInd,
                 'z': panel.parent.MaxZInd}[axis]
    values = np.arange(max_index + 1) / panel.c_omp * panel.istep
    panel.parent.DataDict[key] = np.copy(values)
    return values


def profile_values(panel):
    """Grid coordinates along the axis a 1D panel is plotted against."""
    return grid_values(panel, plot_axis_name(panel))


def image_kwargs(panel):
    """The imshow kwargs that set a 2D panel's aspect ratio."""
    return {} if panel.parent.MainParamDict['ImageAspect'] else {'aspect': 'auto'}


def lineout_data(panel, arr):
    """The (horizontal values, data) pair for a 1D lineout of `arr`.

    A user-defined quantity is allowed to come back as a plain 1D array along
    x, which there is no way to cut along another axis, so such an array stays
    a function of x whatever the panel is set to plot against.
    """
    arr = np.asanyarray(arr)
    if arr.ndim == 1:
        return grid_values(panel, 'x'), arr
    return profile_values(panel), lineout(panel, arr)


####
#
# Lines marking a fixed position, e.g. the shock or the FFT region
#
####

def marker_orientation(panel, axis):
    """How a marker at a fixed value of physical `axis` runs on this panel.

    'v' when `axis` is the panel's horizontal axis, so the marker is a vertical
    line, 'h' when it is the vertical axis, and None when the panel does not
    show that axis at all.
    """
    horiz, vert = plot_axes_of(panel)
    if horiz == axis:
        return 'v'
    if vert == axis:
        return 'h'
    return None


def shows_axis(panel, axis):
    """Whether `axis` is one of the panel's two plotted axes."""
    return marker_orientation(panel, axis) is not None


def add_marker_line(panel, axis, location, **kwargs):
    """A line marking a fixed value of physical `axis`, oriented to match the panel.

    A panel that does not show `axis` still gets a line object back, so that
    callers have something to keep and toggle, but it is hidden.
    """
    orientation = marker_orientation(panel, axis)
    if orientation == 'h':
        return panel.axes.axhline(location, **kwargs)
    line = panel.axes.axvline(location, **kwargs)
    if orientation is None:
        line.set_visible(False)
    return line


def move_marker_line(panel, line, axis, location):
    """Move a line made by add_marker_line. Returns whether the panel shows `axis`."""
    orientation = marker_orientation(panel, axis)
    if orientation == 'h':
        line.set_ydata([location, location])
    elif orientation == 'v':
        line.set_xdata([location, location])
    return orientation is not None


####
#
# The main window's limit settings
#
####

def limits_for_axis(panel, axis):
    """The limits the main settings window fixes for physical `axis`, if any.

    'Set xlim' and 'Set ylim' are ranges in the physical x and y, so they are
    looked up by coordinate rather than by which side of the plot they land on.
    """
    main = panel.parent.MainParamDict
    if axis == 'x' and main['SetxLim']:
        if main['xLimsRelative']:
            return (main['xLeft'] + panel.parent.shock_loc,
                    main['xRight'] + panel.parent.shock_loc)
        return (main['xLeft'], main['xRight'])
    if axis == 'y' and main['SetyLim']:
        return (main['yBottom'], main['yTop'])
    return None


def full_extent(panel, axis):
    """The whole simulation domain along `axis`, as a (low, high) pair.

    Every panel showing a coordinate falls back to this, so that panels which
    matplotlib cannot link still agree on what 'the whole domain' is.
    """
    if axis is None:
        return None
    return (0.0, domain_extent(panel, axis))


def apply_limits(panel, default_horiz=None, default_vert=None):
    """Set the panel's limits, preferring the main window's settings.

    An axis the user has not fixed falls back to `default_horiz` /
    `default_vert`, and a spatial axis with no default given falls back to the
    whole domain. A non-spatial axis with no default is left alone.
    """
    horiz, vert = plot_axes_of(panel)
    lims = (limits_for_axis(panel, horiz) if horiz is not None else None) \
        or default_horiz or full_extent(panel, horiz)
    if lims is not None:
        panel.axes.set_xlim(*lims)
    lims = (limits_for_axis(panel, vert) if vert is not None else None) \
        or default_vert or full_extent(panel, vert)
    if lims is not None:
        panel.axes.set_ylim(*lims)


def domain_extent(panel, axis):
    """The size of the whole simulation domain along `axis`, in c/omega_pe."""
    max_index = {'x': panel.parent.MaxXInd,
                 'y': panel.parent.MaxYInd,
                 'z': panel.parent.MaxZInd}[axis]
    return (max_index + 1) / panel.c_omp * panel.istep


####
#
# Vector fields in the plane of a 2D panel
#
####

# Which pair of components makes up the cross product for each axis, so that
# S_x = E_y B_z - E_z B_y and its cyclic permutations.
_CROSS_PAIRS = {'x': ('y', 'z'), 'y': ('z', 'x'), 'z': ('x', 'y')}


def poynting_component(load, axis):
    """One component of the Poynting flux E x B, in Iseult's field units.

    `load` takes a field key, e.g. 'ey', and gives back the whole array. The
    result is E x B without the c/4pi out front, so it carries the direction
    and the relative magnitude of the energy flux but not its absolute units.
    """
    first, second = _CROSS_PAIRS[axis]
    return (np.asanyarray(load('e' + first)) * np.asanyarray(load('b' + second))
            - np.asanyarray(load('e' + second)) * np.asanyarray(load('b' + first)))


def poynting_keys():
    """The field keys poynting_component needs for any axis."""
    return ['ex', 'ey', 'ez', 'bx', 'by', 'bz']


def in_plane_slices(panel, component):
    """The (horizontal, vertical) parts of a vector field on a 2D panel.

    `component` takes a physical axis name and gives back the whole array for
    that component. Both the choice of components and the slicing follow the
    panel's slice plane and rotation, so a rotated panel gets the component
    that really does point along its horizontal axis.
    """
    horiz, vert = two_d_axes(panel)
    return (two_d_slice(panel, component(horiz)),
            two_d_slice(panel, component(vert)))
