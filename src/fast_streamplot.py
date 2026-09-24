#!/usr/bin/env python
"""A compiled drop-in for the parts of `matplotlib.axes.Axes.streamplot` Iseult uses.

matplotlib traces every streamline one interpolation at a time in pure Python,
on masked arrays, which makes a single panel of streamlines cost a few tenths
of a second on every timestep. The algorithm is reproduced here step for step
(the same spiral of seed points, the same occupancy mask setting the line
spacing, the same adaptive RK12 integrator and the same arrow placement), so
the lines drawn are the same as matplotlib's, but the tracing runs in numba.

Only a uniform grid, a single colour and a single line width are supported,
which is all Iseult asks of streamplot.
"""
import numpy as np
from numba import njit
import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Integration outcomes of _field
_OK, _OUT_OF_BOUNDS, _TERMINATE = 0, 1, 2

_MAX_ERROR = 0.003 # matplotlib's RK12 error tolerance, in axes coordinates


class StreamplotSet:
    '''What streamplot returns: the lines, and the arrows half way along them.'''
    def __init__(self, lines, arrows):
        self.lines = lines
        self.arrows = arrows

    def remove(self):
        for artist in [self.lines] + list(self.arrows):
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass # already gone, e.g. the figure was cleared


@njit(cache=True)
def _interp(a, xi, yi):
    '''Bilinear interpolation of `a` at the grid coordinates (xi, yi).

    A NaN in any of the four surrounding cells makes the result NaN, which is
    how matplotlib's masked arrays end a streamline at invalid data.'''
    ny, nx = a.shape
    x = int(xi)
    y = int(yi)
    xn = x if x == nx - 1 else x + 1
    yn = y if y == ny - 1 else y + 1
    xt = xi - x
    yt = yi - y
    a0 = a[y, x] * (1 - xt) + a[y, xn] * xt
    a1 = a[yn, x] * (1 - xt) + a[yn, xn] * xt
    return a0 * (1 - yt) + a1 * yt


@njit(cache=True)
def _within(xi, yi, nx, ny):
    return 0 <= xi <= nx - 1 and 0 <= yi <= ny - 1


@njit(cache=True)
def _field(u, v, speed, xi, yi, sign):
    '''The unit-speed direction of the flow at (xi, yi), times `sign`.'''
    ny, nx = u.shape
    if not _within(xi, yi, nx, ny):
        return _OUT_OF_BOUNDS, 0.0, 0.0
    ds_dt = _interp(speed, xi, yi)
    if np.isnan(ds_dt) or ds_dt == 0:
        return _TERMINATE, 0.0, 0.0
    dt_ds = 1. / ds_dt
    ui = _interp(u, xi, yi)
    vi = _interp(v, xi, yi)
    if np.isnan(ui) or np.isnan(vi):
        return _TERMINATE, 0.0, 0.0
    return _OK, sign * (ui * dt_ds), sign * (vi * dt_ds)


@njit(cache=True)
def _grid2mask(xi, yi, x_grid2mask, y_grid2mask):
    # np.rint rounds halves to even, like the Python round() matplotlib uses.
    return int(np.rint(xi * x_grid2mask)), int(np.rint(yi * y_grid2mask))


@njit(cache=True)
def _mark(mask, state, cells, xm, ym):
    '''Enter mask cell (xm, ym). False if another streamline already owns it.

    state holds [current x cell, current y cell, number of cells entered by
    the streamline being traced]; cells records those cells so that a
    rejected streamline can hand them back.'''
    if state[0] != xm or state[1] != ym:
        if mask[ym, xm] != 0:
            return False
        cells[state[2], 0] = ym
        cells[state[2], 1] = xm
        state[2] += 1
        mask[ym, xm] = 1
        state[0] = xm
        state[1] = ym
    return True


@njit(cache=True)
def _append(buf, n, x, y):
    if n == buf.shape[0]:
        bigger = np.empty((2 * buf.shape[0], 2))
        bigger[:n] = buf[:n]
        buf = bigger
    buf[n, 0] = x
    buf[n, 1] = y
    return buf, n + 1


@njit(cache=True)
def _rk12(x0, y0, sign, u, v, speed, mask, state, cells,
          x_grid2mask, y_grid2mask, maxds, maxlength):
    '''Trace from (x0, y0) in one direction; matplotlib's _integrate_rk12.'''
    ny, nx = u.shape
    ds = maxds
    stotal = 0.0
    xi = x0
    yi = y0
    buf = np.empty((256, 2))
    n = 0
    while True:
        out_of_bounds = False
        k1x = k1y = k2x = k2y = 0.0
        if _within(xi, yi, nx, ny):
            buf, n = _append(buf, n, xi, yi)
            status, k1x, k1y = _field(u, v, speed, xi, yi, sign)
            if status == _OK:
                status, k2x, k2y = _field(u, v, speed, xi + ds * k1x, yi + ds * k1y, sign)
            if status == _TERMINATE:
                break
            out_of_bounds = status == _OUT_OF_BOUNDS
        else:
            out_of_bounds = True

        if out_of_bounds:
            # Take an Euler step to the boundary to finish the line neatly.
            if n > 0:
                xl = buf[n - 1, 0]
                yl = buf[n - 1, 1]
                status, cx, cy = _field(u, v, speed, xl, yl, sign)
                if status == _OK:
                    if cx == 0:
                        dsx = np.inf
                    elif cx < 0:
                        dsx = xl / -cx
                    else:
                        dsx = (nx - 1 - xl) / cx
                    if cy == 0:
                        dsy = np.inf
                    elif cy < 0:
                        dsy = yl / -cy
                    else:
                        dsy = (ny - 1 - yl) / cy
                    step = min(dsx, dsy)
                    buf, n = _append(buf, n, xl + cx * step, yl + cy * step)
                    stotal += step
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)
        error = np.hypot((dx2 - dx1) / (nx - 1), (dy2 - dy1) / (ny - 1))

        if error < _MAX_ERROR:
            xi += dx2
            yi += dy2
            if not _within(xi, yi, nx, ny):
                break
            xm, ym = _grid2mask(xi, yi, x_grid2mask, y_grid2mask)
            if not _mark(mask, state, cells, xm, ym):
                break
            if stotal + ds > maxlength:
                break
            stotal += ds

        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (_MAX_ERROR / error) ** 0.5)
    return stotal, buf, n


@njit(cache=True)
def _trace_all(u, v, mask_nx, mask_ny, minlength, maxlength):
    '''All streamlines of (u, v), given in grid units per unit time.

    Returns the points, in grid coordinates, of every streamline one after
    the other, and the index each streamline starts at.'''
    ny, nx = u.shape
    speed = np.sqrt((u / (nx - 1))**2 + (v / (ny - 1))**2)

    x_grid2mask = (mask_nx - 1) / (nx - 1)
    y_grid2mask = (mask_ny - 1) / (ny - 1)
    x_mask2grid = 1. / x_grid2mask
    y_mask2grid = 1. / y_grid2mask
    maxds = min(1. / mask_nx, 1. / mask_ny, 0.1)

    mask = np.zeros((mask_ny, mask_nx), dtype=np.int8)
    cells = np.empty((mask_nx * mask_ny, 2), dtype=np.int64)
    state = np.array([-1, -1, 0], dtype=np.int64)

    points = np.empty((4096, 2))
    n_points = 0
    starts = [0]

    # Seed points spiral inwards from the corner of the mask, so that the
    # streamlines that reach the edges are found first.
    xfirst, yfirst, xlast, ylast = 0, 1, mask_nx - 1, mask_ny - 1
    xm, ym = 0, 0
    direction = 0 # right, up, left, down
    for _ in range(mask_nx * mask_ny):
        if mask[ym, xm] == 0:
            x0 = xm * x_mask2grid
            y0 = ym * y_mask2grid
            state[2] = 0
            sxm, sym = _grid2mask(x0, y0, x_grid2mask, y_grid2mask)
            if _mark(mask, state, cells, sxm, sym):
                sb, back, nb = _rk12(x0, y0, -1.0, u, v, speed, mask, state, cells,
                                     x_grid2mask, y_grid2mask, maxds, maxlength)
                state[0] = sxm
                state[1] = sym
                sf, fwd, nf = _rk12(x0, y0, 1.0, u, v, speed, mask, state, cells,
                                    x_grid2mask, y_grid2mask, maxds, maxlength)
                if sb + sf > minlength:
                    for k in range(nb - 1, -1, -1):
                        points, n_points = _append(points, n_points, back[k, 0], back[k, 1])
                    for k in range(1, nf):
                        points, n_points = _append(points, n_points, fwd[k, 0], fwd[k, 1])
                    starts.append(n_points)
                else:
                    # Too short: give back every cell it took.
                    for k in range(state[2]):
                        mask[cells[k, 0], cells[k, 1]] = 0

        if direction == 0:
            xm += 1
            if xm >= xlast:
                xlast -= 1
                direction = 1
        elif direction == 1:
            ym += 1
            if ym >= ylast:
                ylast -= 1
                direction = 2
        elif direction == 2:
            xm -= 1
            if xm <= xfirst:
                xfirst += 1
                direction = 3
        else:
            ym -= 1
            if ym <= yfirst:
                yfirst += 1
                direction = 0

    return points[:n_points].copy(), np.array(starts, dtype=np.int64)


def trace(x, y, u, v, density=1, minlength=0.1, maxlength=4.0):
    '''The streamlines of (u, v) on the uniform grid with 1D coordinates x, y.

    Returns a list of (k, 2) arrays of data coordinates, one per streamline,
    as matplotlib's streamplot would trace them with integration_direction
    'both'.'''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != (len(y), len(x)) or v.shape != u.shape:
        raise ValueError("'u' and 'v' must match the shape of the (x, y) grid")
    if len(x) < 2 or len(y) < 2:
        raise ValueError("streamlines need at least a 2 x 2 grid")
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    mask_nx, mask_ny = (30 * np.broadcast_to(density, 2)).astype(int)

    # matplotlib masks inf as well as NaN; NaN is what _interp checks for.
    u = np.where(np.isfinite(u), u, np.nan) / dx
    v = np.where(np.isfinite(v), v, np.nan) / dy

    points, starts = _trace_all(u, v, int(mask_nx), int(mask_ny),
                                float(minlength), float(maxlength) / 2.)
    points[:, 0] = points[:, 0] * dx + x[0]
    points[:, 1] = points[:, 1] * dy + y[0]
    return [points[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def streamplot(axes, x, y, u, v, density=1, color=None, linewidth=None,
               arrowsize=1, arrowstyle='-|>', minlength=0.1, maxlength=4.0,
               zorder=None):
    '''Draw the streamlines of (u, v) on `axes`, like `axes.streamplot`.

    x and y are the 1D, evenly spaced coordinates of the columns and rows of
    u and v. Returns a StreamplotSet.'''
    if zorder is None:
        zorder = mlines.Line2D.zorder
    if color is None:
        color = axes._get_lines.get_next_color()
    if linewidth is None:
        linewidth = mpl.rcParams['lines.linewidth']
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lines = trace(x, y, u, v, density=density, minlength=minlength, maxlength=maxlength)

    transform = axes.transData
    if lines:
        segments = np.concatenate([np.stack([t[:-1], t[1:]], axis=1) for t in lines])
    else:
        segments = np.empty((0, 2, 2))
    lc = mcollections.LineCollection(segments, transform=transform, color=color,
                                     linewidth=linewidth, zorder=zorder)
    lc.sticky_edges.x[:] = [x[0], x[-1]]
    lc.sticky_edges.y[:] = [y[0], y[-1]]
    axes.add_collection(lc)

    arrows = []
    for t in lines:
        tx, ty = t[:, 0], t[:, 1]
        s = np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))
        n = np.searchsorted(s, s[-1] / 2.)
        arrow = mpatches.FancyArrowPatch(
            (tx[n], ty[n]), (np.mean(tx[n:n + 2]), np.mean(ty[n:n + 2])),
            transform=transform, arrowstyle=arrowstyle, mutation_scale=10 * arrowsize,
            color=color, linewidth=linewidth, zorder=zorder)
        # add_artist rather than add_patch: the arrows lie inside the lines,
        # so working out the data limits of each one is wasted effort.
        axes.add_artist(arrow)
        arrows.append(arrow)

    axes.autoscale_view()
    return StreamplotSet(lc, arrows)
