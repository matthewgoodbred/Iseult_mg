#!/usr/bin/env python
"""1D panels showing the terms of Ohm's law and of the pressure balance.

Both panels are cuts along x or y through the slice picked in the main
settings, and share one engine: the particles near the slice are binned into
the stencil of `fluid_balance.Stencil` with the stress-energy sums of
`stress_energy`, the fields are averaged over the same bins, and the
neighbouring outputs are binned the same way when time derivatives are asked
for. The binned sums of every timestep are cached, so stepping forward with
time derivatives on costs one new particle pass per step, and the two panels
showing the same slice share the work.

The physics, and the units, are described in `fluid_balance`.
"""
import time
from collections import OrderedDict
import qt_compat as Tk
from qt_compat import ttk

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import numpy as np

import data_loading
import fluid_balance as fb
import plot_axes
import stress_energy as se

COMPONENTS = ('x', 'y', 'z')
FIELD_KEYS = ('ex', 'ey', 'ez', 'bx', 'by', 'bz')
MOMENTA = {0: ('ui', 'vi', 'wi'), 1: ('ue', 've', 'we')}
SPECIES_NAMES = ('Ions', 'Electrons')
SPECIES_TEX = ('i', 'e')

# How many timesteps of binned sums to keep around, across all balance panels
_CACHE_SIZE = 64

# Fixed colours so a term looks the same whatever else is shown
TERM_COLORS = {'E': 'k', 'vxb': 'C0', 'ideal': 'C9', 'pressure': 'C3', 'inertia': 'C2', 'heat': 'C4',
               'dpdt': 'C1', 'rhs': '0.45', 'residual': 'C7',
               'mag_pressure': 'C0', 'mag_tension': 'C9', 'elec_pressure': 'C5',
               'elec_tension': 'C6', 'em_momentum': 'C8', 'total': 'k',
               'p_xx': 'C3', 'p_yy': 'C1', 'p_zz': 'C4'}
SPLIT_DASHES = {'x': (6, 2), 'y': (2, 2), 'z': (6, 2, 1, 2)}
SPECIES_DASHES = {'total': None, 0: (6, 2), 1: (1.5, 1.5)}


def _cache(parent):
    cache = getattr(parent, '_balance_cache', None)
    if cache is None:
        cache = OrderedDict()
        parent._balance_cache = cache
    return cache


def _cache_get(parent, key):
    cache = _cache(parent)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _cache_put(parent, key, value):
    cache = _cache(parent)
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)


def _scalar(value, default=np.nan):
    try:
        value = float(np.squeeze(value))
    except (TypeError, ValueError):
        return default
    return value


class BalancePanel:
    '''What the Ohm's-law and pressure-balance panels have in common.'''

    plot_param_dict = {'twoD': 0,
                       'spatial_x': True,
                       'spatial_y': False,
                       'component': 0,        # 0, 1, 2 = x, y, z
                       'nbins': 0,            # bins along the slice; 0 = auto, see build_stencil
                       'trans_width': 0.0,    # width of a bin across the slice [c/omega_pe]; 0 = auto
                       'smooth': 3,           # boxcar width, in bins along the slice
                       'time_deriv': True,    # d/dt from the neighbouring outputs
                       'split_pressure': False, # show d_j P_ij for each j as well
                       'show_legend': True,
                       'legend_outside': False, # in the colorbar slot rather than on the axes
                       'legend_loc': 'N/A',
                       'symmetric': False,
                       'set_v_min': False,
                       'set_v_max': False,
                       'v_min': -1.0,
                       'v_max': 1.0}
    plot_axes.add_axis_params(plot_param_dict, two_d=False)

    # the params that list which terms are shown, and their defaults, per mode
    TERM_PARAM = 'terms'

    def __init__(self, parent, figwrapper):
        self.settings_window = None
        self.FigWrap = figwrapper
        self.parent = parent
        self.ChartTypes = self.FigWrap.PlotTypeDict.keys()
        self.chartType = self.FigWrap.chartType
        self.figure = self.FigWrap.figure
        self.status_text = ''
        self.axes = None
        self.lines = {}
        self.legend = None

    ####
    #
    # What is shown
    #
    ####

    def component(self):
        c = self.GetPlotParam('component')
        return c if c in (0, 1, 2) else 0

    def available_terms(self):
        raise NotImplementedError

    def default_terms(self):
        raise NotImplementedError

    def term_param(self):
        return self.TERM_PARAM

    def selected_terms(self):
        allowed = self.available_terms()
        stored = self.GetPlotParam(self.term_param())
        if stored == 'none':
            return []
        chosen = stored.split(',') if stored else self.default_terms()
        return [t for t in allowed if t in chosen]

    def needs_time_neighbours(self):
        return bool(self.GetPlotParam('time_deriv'))

    ####
    #
    # Loading
    #
    ####

    def set_plot_keys(self):
        # The particles are read by the panel itself, at this and the
        # neighbouring timesteps, so only the parameters and the fields at
        # this timestep go through MainApp.
        return ['c_omp', 'istep', 'me', 'mi', 'qi', 'c', 'stride', 'ppc0', 'time'] + list(FIELD_KEYS)

    def _max_index(self, axis):
        return {'x': self.parent.MaxXInd, 'y': self.parent.MaxYInd, 'z': self.parent.MaxZInd}[axis]

    def _slice_index(self, axis):
        return getattr(self.parent, axis + 'Slice', 0)

    def extent_cells(self, axis):
        max_index = self._max_index(axis)
        return 1.0 if max_index == 0 else (max_index + 1) * self.istep

    # The automatic bin widths, in c/omega_pe. Output particles are usually
    # thinned by a large stride, so bins much finer than the skin depth hold
    # too few of them to take derivatives of.
    AUTO_BIN = 0.25
    AUTO_WIDTH = 0.5

    def build_stencil(self):
        '''The bins along and across the slice, and a key naming them.

        With 'nbins' at 0 the bins along the slice are AUTO_BIN wide, or one
        output grid cell if that is wider; with 'trans_width' at 0 the bins
        across it are AUTO_WIDTH wide, or as wide as the bins along it.'''
        axis = plot_axes.plot_axis_name(self)
        length = plot_axes.domain_extent(self, axis)
        spacing = self.istep / self.c_omp
        nh = int(self.GetPlotParam('nbins'))
        if nh <= 0:
            nh = max(1, int(round(length / max(spacing, self.AUTO_BIN))))
        h_edges = np.linspace(0.0, length, nh + 1)
        dh = length / nh
        width = float(self.GetPlotParam('trans_width'))
        width = max(dh, self.AUTO_WIDTH) if width <= 0 else width
        # Across the slice, the axis in the 2D plane is centred on the 2D
        # viewport, or averaged over all of it; the one out of the plane is
        # centred on the plane itself.
        averaged = plot_axes.averages_lineouts(self)
        view = plot_axes.lineout_viewport(self)
        trans = {}
        key_parts = ['balance', axis, str(nh)]
        for other in fb.AXES:
            if other == axis:
                continue
            if self._max_index(other) == 0:
                trans[other] = (None, None, 1)
                key_parts.append(f'{other}:all')
                continue
            if other in view:
                low, high = view[other]
                center = 0.5 * (low + high)
                if averaged and high - low > 0:
                    trans[other] = (center, high - low, 1)
                    key_parts.append(f'{other}:{low:.6g}..{high:.6g}')
                    continue
            else:
                center = self._slice_index(other) * self.istep / self.c_omp
            trans[other] = (center, width, 3)
            key_parts.append(f'{other}:{center:.6g}±{width:.6g}')
        key_parts.append(str(self.parent.MainParamDict['PrtlStride']))
        return fb.Stencil(axis, h_edges, trans), '|'.join(key_parts)

    def bin_volume_cells(self, stencil):
        volume = stencil.dh * self.c_omp
        for other in (stencil.t1, stencil.t2):
            center, width, n = stencil.trans[other]
            volume *= self.extent_cells(other) if width is None else width * self.c_omp
        return volume

    def _paths(self, kind):
        return self.parent.PathDict[kind]

    def _load_prtl(self, index, key):
        '''A particle array at output `index`, from memory when MainApp has it.'''
        if index == self.parent.TimeStep.value - 1 and key in self.parent.DataDict:
            return np.asarray(self.parent.DataDict[key])
        return np.asarray(data_loading.load_dataset(
            self._paths('Prtl')[index], key, slice(None, None, self.parent.MainParamDict['PrtlStride'])))

    def _positions(self, index, prtl_type, axes, n):
        '''Particle positions in c/omega_pe for each of `axes`, None where missing.'''
        available = set(plot_axes.available_position_keys(self, prtl_type, axes))
        out = {}
        for axis in axes:
            key = plot_axes.PRTL_POS_KEYS[prtl_type][axis]
            if key not in available:
                out[axis] = None
                continue
            pos = self._load_prtl(index, key)
            out[axis] = pos / self.c_omp if pos.ndim == 1 and len(pos) == n else None
        return out

    def particle_sums(self, index, stencil, key):
        '''The binned sums of both species at output `index`, cached.'''
        cache_key = (str(self._paths('Prtl')[index]), key)
        entry = _cache_get(self.parent, cache_key)
        if entry is not None:
            return entry
        start = time.perf_counter()
        axis = stencil.axis
        active = [a for a in (stencil.t1, stencil.t2) if stencil.trans[a][1] is not None]
        sums = []
        n_particles = 0
        for prtl_type in (0, 1):
            u, v, w = (self._load_prtl(index, k) for k in MOMENTA[prtl_type])
            n = len(u)
            n_particles += n
            pos = self._positions(index, prtl_type, [axis] + active, n)
            if pos[axis] is None:
                raise ValueError(f'The particle data has no {axis} positions to bin along.')
            vpos, mask = stencil.transverse_index(pos)
            if vpos is None:
                vpos = np.full(n, 0.5)
            s = se.bin_moments(u, v, w, pos[axis], (stencil.h_edges[0], stencil.h_edges[-1]), stencil.nh,
                               vpos=vpos, v_range=(0.0, float(stencil.nv)), nv=stencil.nv, mask=mask)
            sums.append(s.reshape(se.N_LAB, stencil.n2, stencil.n1, stencil.nh))
        entry = {'sums': sums, 'n_particles': n_particles,
                 'n_binned': int(sum(s[se.N_].sum() for s in sums)),
                 'seconds': time.perf_counter() - start}
        _cache_put(self.parent, cache_key, entry)
        return entry

    def _slab(self, stencil):
        '''The index ranges of the field grid the stencil needs, as (z, y, x) slices.'''
        slices, offsets = {}, {}
        spacing = self.istep / self.c_omp
        for axis in fb.AXES:
            size = self._max_index(axis) + 1
            center, width, n = stencil.trans.get(axis, (None, None, 1))
            if axis == stencil.axis or width is None:
                slices[axis], offsets[axis] = slice(0, size), 0
                continue
            lo = int(np.floor((center - 0.5 * n * width) / spacing)) - 1
            hi = int(np.ceil((center + 0.5 * n * width) / spacing)) + 2
            lo, hi = max(lo, 0), min(max(hi, lo + 1), size)
            slices[axis], offsets[axis] = slice(lo, hi), lo
        return (slices['z'], slices['y'], slices['x']), offsets

    def field_bins(self, index, stencil, key):
        '''E and B averaged over the stencil bins at output `index`, shape (3, n2, n1, nh) each.'''
        cache_key = (str(self._paths('Flds')[index]), key, 'fields')
        entry = _cache_get(self.parent, cache_key)
        if entry is not None:
            return entry
        spacing = self.istep / self.c_omp
        binned = {}
        slab, offsets = self._slab(stencil)
        if index == self.parent.TimeStep.value - 1:
            for k in FIELD_KEYS:
                binned[k] = fb.bin_grid(np.asarray(self.FigWrap.LoadKey(k))[slab], stencil, offsets, spacing)
        else:
            shape = tuple(s.stop - s.start for s in slab)
            path = self._paths('Flds')[index]
            for k in FIELD_KEYS:
                arr = np.asarray(data_loading.load_dataset(path, k, slab), dtype=np.float64).reshape(shape)
                binned[k] = fb.bin_grid(arr, stencil, offsets, spacing)
        entry = {'E': np.stack([binned['ex'], binned['ey'], binned['ez']]),
                 'B': np.stack([binned['bx'], binned['by'], binned['bz']])}
        _cache_put(self.parent, cache_key, entry)
        return entry

    def output_time(self, index):
        '''The time of output `index` in 1/omega_pe.'''
        if index == self.parent.TimeStep.value - 1:
            return _scalar(self.FigWrap.LoadKey('time'))
        cache_key = (str(self._paths('Param')[index]), 'time')
        value = _cache_get(self.parent, cache_key)
        if value is None:
            value = _scalar(data_loading.load_dataset(self._paths('Param')[index], 'time'))
            _cache_put(self.parent, cache_key, value)
        return value

    def LoadData(self):
        self.c_omp = _scalar(self.FigWrap.LoadKey('c_omp'))
        self.istep = _scalar(self.FigWrap.LoadKey('istep'))
        self.c = _scalar(self.FigWrap.LoadKey('c'), 0.45)
        me, mi = _scalar(self.FigWrap.LoadKey('me')), _scalar(self.FigWrap.LoadKey('mi'))
        qi = abs(_scalar(self.FigWrap.LoadKey('qi'), 1.0))
        self.masses = (mi, me)
        self.charges = (qi, -qi)
        self.has_ions = np.isfinite(mi) and mi > 0
        self.m_ref = mi if self.has_ions else me
        self.mass_unit = 'i' if self.has_ions else 'e'
        try:
            self.version = data_loading.tristan_version(self._paths('Param')[self.parent.TimeStep.value - 1])
        except Exception:
            self.version = 2

        stencil, key = self.build_stencil()
        self.stencil = stencil
        index = self.parent.TimeStep.value - 1
        smooth = int(self.GetPlotParam('smooth'))

        entry = self.particle_sums(index, stencil, key)
        self.sums = [fb.smooth(s, smooth) for s in entry['sums']]
        fields = self.field_bins(index, stencil, key)
        self.E = fb.smooth(fields['E'], smooth)
        self.B = fb.smooth(fields['B'], smooth)
        self.t_now = self.output_time(index)

        # The neighbouring outputs, for the time derivatives
        self.neighbours = {}
        n_outputs = len(self._paths('Prtl'))
        if self.needs_time_neighbours():
            for name, other in (('before', index - 1), ('after', index + 1)):
                if 0 <= other < n_outputs:
                    other_entry = self.particle_sums(other, stencil, key)
                    other_fields = self.field_bins(other, stencil, key)
                    self.neighbours[name] = {
                        't': self.output_time(other),
                        'sums': [fb.smooth(s, smooth) for s in other_entry['sums']],
                        'E': fb.smooth(other_fields['E'], smooth),
                        'B': fb.smooth(other_fields['B'], smooth)}

        status = (f"Binned {entry['n_binned']:,} of {entry['n_particles']:,} particles"
                  + (f" in {entry['seconds']:.2f} s" if entry['seconds'] > 0 else '')
                  + f": {stencil.nh} bins of {stencil.dh:.3g} c/ωpe along the slice")
        across = [a for a in (stencil.t1, stencil.t2) if stencil.trans[a][2] > 1]
        if across:
            status += (f", and 3 bins of {stencil.trans[across[0]][1]:.3g} c/ωpe across it in "
                       + ' and '.join(across) + ' for the transverse derivatives')
        else:
            status += ', averaged across it'
        if self.needs_time_neighbours():
            have = [n for n in ('before', 'after') if n in self.neighbours]
            status += ('; d/dt ' + ('centered' if len(have) == 2 else f'one-sided ({have[0]} only)')
                       if have else '; d/dt unavailable: no neighbouring output')
        self.status_text = status + '.'
        self.compute()

        if self.settings_window is not None:
            try:
                self.settings_window.show_status()
            except Tk.TclError:
                pass

    def momentum_rate(self, species, weight=1.0):
        '''d/dt of the momentum sums of `species` on the slice, per 1/omega_pe,
        or None. `species` is a list of indices, `weight` their masses.'''
        before, after = self.neighbours.get('before'), self.neighbours.get('after')
        if before is None and after is None:
            return None
        weights = np.broadcast_to(np.asarray(weight, dtype=np.float64), (len(species),))

        def momentum(sums):
            return sum(w * self.stencil.center(sums[s][[se.UX, se.UY, se.UZ]])
                       for s, w in zip(species, weights))
        return fb.time_derivative(
            (self.t_now, momentum(self.sums)),
            None if before is None else (before['t'], momentum(before['sums'])),
            None if after is None else (after['t'], momentum(after['sums'])))

    def em_momentum_rate(self):
        '''d/dt of E x B on the slice, per 1/omega_pe, or None.'''
        before, after = self.neighbours.get('before'), self.neighbours.get('after')
        if before is None and after is None:
            return None

        def exb(E, B):
            return np.cross(self.stencil.center(E), self.stencil.center(B), axis=0)
        return fb.time_derivative(
            (self.t_now, exb(self.E, self.B)),
            None if before is None else (before['t'], exb(before['E'], before['B'])),
            None if after is None else (after['t'], exb(after['E'], after['B'])))

    def compute(self):
        '''Fill self.series with (key, label, values, style) for every line.'''
        raise NotImplementedError

    ####
    #
    # Drawing
    #
    ####

    def draw(self):
        self.gs = gridspec.GridSpecFromSubplotSpec(100, 100, subplot_spec=self.parent.gs0[self.FigWrap.pos])
        extent = self.gs[self.parent.axes_extent[0]:self.parent.axes_extent[1],
                         self.parent.axes_extent[2]:self.parent.axes_extent[3]]
        share_x_ax, _ = self.parent.GetSharedAxes(self.FigWrap.pos)
        self.axes = self.figure.add_subplot(extent, sharex=share_x_ax)
        self.axes.tick_params(labelsize=self.parent.MainParamDict['NumFontSize'], color='black')
        self.zero_line = self.axes.axhline(0.0, color='0.6', lw=0.6, zorder=0)
        self.lines = {}
        self.legend = None
        self.update_1d()

    def refresh(self):
        if self.axes is None:
            # not drawn yet; the redraw that follows a change of chart type will
            return
        self.update_1d()

    def update_1d(self):
        wanted = {s[0] for s in self.series}
        for key in list(self.lines):
            if key not in wanted:
                self.lines.pop(key).remove()
        x = self.stencil.h_centers
        handles, labels = [], []
        ymin, ymax = np.inf, -np.inf
        for key, label, values, style in self.series:
            if key not in self.lines:
                line, = self.axes.plot([], [])
                self.lines[key] = line
            line = self.lines[key]
            line.set_color(style.get('color', 'k'))
            line.set_linewidth(style.get('lw', 1.2))
            if style.get('dashes') is None:
                line.set_linestyle('-')
            else:
                line.set_dashes(style['dashes'])
            vals = np.asarray(values, dtype=np.float64)
            line.set_data(x, vals)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                ymin, ymax = min(ymin, finite.min()), max(ymax, finite.max())
            handles.append(line)
            labels.append(label)

        if np.isfinite(ymin):
            pad = 0.04 * (ymax - ymin) if ymax > ymin else max(abs(ymax), 1e-30) * 0.04
            ylims = [ymin - pad, ymax + pad]
            if self.GetPlotParam('symmetric'):
                top = max(abs(ylims[0]), abs(ylims[1]))
                ylims = [-top, top]
            self.axes.set_ylim(ylims)
        if self.GetPlotParam('set_v_min'):
            self.axes.set_ylim(bottom=self.GetPlotParam('v_min'))
        if self.GetPlotParam('set_v_max'):
            self.axes.set_ylim(top=self.GetPlotParam('v_max'))

        self.make_legend(handles, labels)
        plot_axes.apply_limits(self, (self.stencil.h_edges[0], self.stencil.h_edges[-1]))
        main = self.parent.MainParamDict
        self.axes.set_xlabel(plot_axes.AXIS_LABELS[self.stencil.axis], labelpad=main['xLabelPad'],
                             color='black', size=main['AxLabelSize'])
        self.axes.set_ylabel(self.ylabel(), labelpad=main['yLabelPad'], color='black', size=main['AxLabelSize'])

    def legend_outside(self):
        return bool(self.GetPlotParam('legend_outside'))

    def make_legend(self, handles, labels):
        if self.legend is not None:
            if not getattr(self, '_legend_was_outside', False) and self.legend._get_loc() != 1:
                self.SetPlotParam('legend_loc', ' '.join(str(x) for x in self.legend._get_loc()), update_plot=False)
            self.legend.remove()
        kwargs = dict(framealpha=.05, fontsize=self.parent.MainParamDict['legendLabelSize'])
        self._legend_was_outside = self.legend_outside()
        if self._legend_was_outside:
            # In the slot a 2D panel keeps for its colorbar: to the right of
            # the axes, or above them when the colorbars are horizontal.
            if self.parent.MainParamDict['HorizontalCbars']:
                kwargs.update(loc='lower left', bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0,
                              ncol=min(len(handles), 4) or 1)
            else:
                ext = self.parent.cbar_extent
                slot = self.gs[ext[0]:ext[1], ext[2]:ext[3]].get_position(self.figure)
                kwargs.update(loc='upper left', bbox_to_anchor=(slot.x0, 1.0), borderaxespad=0.0, ncol=1,
                              bbox_transform=transforms.blended_transform_factory(self.figure.transFigure,
                                                                                  self.axes.transAxes))
        else:
            kwargs.update(loc=1, ncol=1 if len(handles) < 6 else 2)
        self.legend = self.axes.legend(handles, labels, **kwargs)
        self.legend.get_frame().set_facecolor('k')
        self.legend.get_frame().set_linewidth(0.0)
        self.legend.set_visible(bool(self.GetPlotParam('show_legend')) and bool(handles))
        if self._legend_was_outside:
            return
        self.legend.set_draggable(True, update='loc')
        if self.GetPlotParam('legend_loc') != 'N/A':
            loc = self.GetPlotParam('legend_loc').split()
            self.legend._set_loc((float(loc[0]), float(loc[1])))

    ####
    #
    # Plumbing
    #
    ####

    def ChangePlotType(self, str_arg):
        self.FigWrap.ChangeGraph(str_arg)

    def GetPlotParam(self, keyname):
        return self.FigWrap.GetPlotParam(keyname)

    def SetPlotParam(self, keyname, value, update_plot=True, NeedsRedraw=False):
        self.FigWrap.SetPlotParam(keyname, value, update_plot=update_plot, NeedsRedraw=NeedsRedraw)

    def OpenSettings(self):
        if self.settings_window is not None:
            self.settings_window.destroy()
        self.settings_window = self.SETTINGS(self)


####
#
# Ohm's law
#
####

class OhmsLawPanel(BalancePanel):
    plot_param_dict = dict(BalancePanel.plot_param_dict)
    plot_param_dict.update({'species': 1,   # 0 = ions, 1 = electrons
                            'terms': ''})    # comma separated, see fluid_balance.OHM_TERMS; '' = defaults

    TERM_NAMES = {'E': 'E  (the field)',
                  'vxb': '−V×B  (convection)',
                  'ideal': 'E + V×B  (non-ideal part of E)',
                  'pressure': '∇·P / qn  (pressure)',
                  'inertia': '∇·(εUU) / qn  (bulk inertia)',
                  'heat': '∇·(qU+Uq) / qn  (heat flux)',
                  'dpdt': '∂T^0i/∂t / qn  (momentum change)',
                  'rhs': 'Sum of the right-hand side',
                  'residual': 'E − sum  (residual)'}

    def species(self):
        s = self.GetPlotParam('species')
        return s if s in (0, 1) else 1

    def available_terms(self):
        return [t for t in fb.OHM_TERMS if t != 'dpdt' or self.GetPlotParam('time_deriv')]

    def default_terms(self):
        return ['E', 'vxb', 'pressure', 'inertia', 'heat', 'dpdt', 'residual']

    def compute(self):
        s = self.species()
        i = self.component()
        m_over_q = self.masses[s] / self.charges[s] * self.c ** 2 / self.c_omp
        dpdt = self.momentum_rate([s]) if self.GetPlotParam('time_deriv') else None
        terms = fb.ohm_terms(self.sums[s], self.stencil, self.E, self.B, i, m_over_q, dpdt=dpdt)
        ci, sp = COMPONENTS[i], SPECIES_TEX[s]
        qn = r'q_%sn_%s' % (sp, sp)
        labels = {'E': r'$E_%s$' % ci,
                  'vxb': r'$-(\mathbf{V}_%s\times\mathbf{B})_%s$' % (sp, ci),
                  'ideal': r'$E_%s+(\mathbf{V}_%s\times\mathbf{B})_%s$' % (ci, sp, ci),
                  'pressure': r'$(\nabla\cdot\mathsf{P}_%s)_%s/%s$' % (sp, ci, qn),
                  'inertia': r'$\nabla\cdot(\varepsilon\mathbf{UU})_{%s,%s}/%s$' % (sp, ci, qn),
                  'heat': r'$\nabla\cdot(\mathbf{qU}+\mathbf{Uq})_{%s,%s}/%s$' % (sp, ci, qn),
                  'dpdt': r'$\partial_t T^{0%s}_%s/%s$' % (ci, sp, qn),
                  'rhs': r'sum of RHS',
                  'residual': r'$E_%s-$ sum' % ci}
        self.series = []
        for key in self.selected_terms():
            values = terms.get(key)
            if values is None:
                continue
            style = {'color': TERM_COLORS[key]}
            if key == 'rhs':
                style['dashes'] = (5, 2)
            if key in ('E', 'residual'):
                style['lw'] = 1.6
            self.series.append((key, labels[key], values, style))
            if key == 'pressure' and self.GetPlotParam('split_pressure'):
                for a in fb.AXES:
                    self.series.append(('pressure_' + a,
                                        r'$\partial_%s P_{%s,%s%s}/%s$' % (a, sp, ci, a, qn),
                                        terms['pressure_' + a],
                                        {'color': TERM_COLORS['pressure'], 'dashes': SPLIT_DASHES[a], 'lw': 0.9}))

    def ylabel(self):
        return r'$E_%s$ terms  [code units]' % COMPONENTS[self.component()]


####
#
# Pressure balance
#
####

class PressureBalancePanel(BalancePanel):
    plot_param_dict = dict(BalancePanel.plot_param_dict)
    plot_param_dict.update({'mode': 0,           # 0 = force densities (the divergence form), 1 = stresses
                            'integrate': False,  # force densities integrated along the slice, int f dx
                            'species': 0,        # particle terms of: 0 = all species together, 1 = ions,
                                                 # 2 = electrons, 3 = ions and electrons separately
                            'force_terms': '',   # comma separated, see fluid_balance.PB_FORCE_TERMS
                            'stress_terms': ''}) # comma separated, see fluid_balance.PB_STRESS_TERMS

    TERM_NAMES = {'mag_pressure': 'Magnetic pressure  B²/2',
                  'mag_tension': 'Magnetic tension  BB',
                  'elec_pressure': 'Electric pressure  E²/2',
                  'elec_tension': 'Electric tension  EE',
                  'em_momentum': 'Field momentum  ∂(E×B)/∂t / c',
                  'pressure': 'Particle pressure  P',
                  'inertia': 'Bulk inertia  εUU',
                  'heat': 'Heat flux  qU+Uq',
                  'dpdt': 'Particle momentum  ∂T^0i/∂t',
                  'residual': 'Residual  (sum of all)',
                  'total': 'Total  T^ij (particles + fields)',
                  'p_xx': 'Pxx', 'p_yy': 'Pyy', 'p_zz': 'Pzz'}
    PARTICLE_TERMS = ('pressure', 'inertia', 'heat', 'dpdt', 'p_xx', 'p_yy', 'p_zz')

    def mode(self):
        return 1 if self.GetPlotParam('mode') == 1 else 0

    def integrated(self):
        '''Whether the force densities are shown integrated along the slice.'''
        return self.mode() == 0 and bool(self.GetPlotParam('integrate'))

    def term_param(self):
        return 'stress_terms' if self.mode() else 'force_terms'

    def available_terms(self):
        if self.mode():
            return list(fb.PB_STRESS_TERMS) + ['p_xx', 'p_yy', 'p_zz']
        dt = self.GetPlotParam('time_deriv')
        return [t for t in fb.PB_FORCE_TERMS if dt or t not in ('dpdt', 'em_momentum')]

    def default_terms(self):
        if self.mode():
            return ['mag_pressure', 'mag_tension', 'pressure', 'inertia', 'total']
        return ['mag_pressure', 'mag_tension', 'pressure', 'inertia', 'dpdt', 'residual']

    def shown_species(self):
        '''The particle groups whose terms are drawn: 'total', 0 (ions) or 1 (electrons).'''
        s = self.GetPlotParam('species')
        return {0: ['total'], 1: [0], 2: [1], 3: [0, 1]}.get(s, ['total'])

    def unit_factors(self):
        '''(particle, field) factors putting the sums and the fields in units of n0 m c^2.'''
        stride = _scalar(self.FigWrap.LoadKey('stride'), 1.0)
        stride = stride if np.isfinite(stride) and stride > 0 else 1.0
        ppc0 = _scalar(self.FigWrap.LoadKey('ppc0'))
        self.has_ppc0 = np.isfinite(ppc0) and ppc0 > 0
        n0 = ppc0 if self.has_ppc0 else 1.0
        per_particle = stride * self.parent.MainParamDict['PrtlStride']
        particle = per_particle / (self.bin_volume_cells(self.stencil) * n0)
        # The field pressure is F^2/2 in code units; divide it by n0 m c^2 with
        # m the code-unit mass, which Tristan v2 scales by its unit_ch.
        if self.version == 1:
            unit_ms = 1.0
        else:
            unit_ms = self.c ** 2 / ((ppc0 if self.has_ppc0 else 1.0) * self.c_omp ** 2)
        field = 1.0 / (unit_ms * self.m_ref * self.c ** 2 * n0)
        return particle, field

    def compute(self):
        i = self.component()
        j = fb.AXIS_INDEX[self.stencil.axis]
        p_factor, f_factor = self.unit_factors()
        rel_masses = [m / self.m_ref for m in self.masses]
        species_idx = [0, 1] if self.has_ions else [1]
        splits, total = fb.particle_splits([self.sums[s] for s in species_idx],
                                           [rel_masses[s] for s in species_idx])
        by_group = {'total': total}
        for s, split in zip(species_idx, splits):
            by_group[s] = split
        dt = bool(self.GetPlotParam('time_deriv'))
        ci, cj = COMPONENTS[i], COMPONENTS[j]

        if self.mode() == 0:
            fields = fb.field_force_terms(self.stencil, self.E, self.B, i, f_factor,
                                          em_momentum_dt=self.em_momentum_rate()[i] if dt and self.neighbours else None)
            particle = {}
            for group, split in by_group.items():
                terms = {k: v * p_factor for k, v in fb.force_terms(split, self.stencil, i).items()}
                members = species_idx if group == 'total' else [group]
                rate = self.momentum_rate(members, [rel_masses[s] for s in members]) if dt else None
                terms['dpdt'] = -rate[i] * p_factor if rate is not None else None
                particle[group] = terms
            residual = sum(fields[k] for k in ('mag_pressure', 'mag_tension', 'elec_pressure', 'elec_tension'))
            residual = residual + particle['total']['pressure'] + particle['total']['inertia'] + particle['total']['heat']
            for extra in (fields['em_momentum'], particle['total']['dpdt']):
                if extra is not None:
                    residual = residual + extra
            labels = {'mag_pressure': r'$-\partial_%s B^2/2$' % ci,
                      'mag_tension': r'$\partial_j(B_%sB_j)$' % ci,
                      'elec_pressure': r'$-\partial_%s E^2/2$' % ci,
                      'elec_tension': r'$\partial_j(E_%sE_j)$' % ci,
                      'em_momentum': r'$-\partial_t(\mathbf{E}\times\mathbf{B})_%s/c$' % ci,
                      'residual': 'residual'}
            values = dict(fields, residual=residual)
            if self.integrated():
                values, labels, particle = self.integrate(values, particle, by_group, i, j, f_factor, p_factor)
        else:
            fields = fb.field_stress_terms(self.stencil, self.E, self.B, i, j, f_factor)
            particle = {}
            for group, split in by_group.items():
                terms = {k: v * p_factor for k, v in fb.stress_terms(split, self.stencil, i, j).items()}
                for a in fb.AXES:
                    k = fb.AXIS_INDEX[a]
                    terms['p_' + a * 2] = self.stencil.center(split['pressure'][k, k]) * p_factor
                particle[group] = terms
            tot = sum(fields.values())
            tot = tot + particle['total']['pressure'] + particle['total']['inertia'] + particle['total']['heat']
            labels = {'mag_pressure': r'$B^2/2$',
                      'mag_tension': r'$-B_%sB_%s$' % (ci, cj),
                      'elec_pressure': r'$E^2/2$',
                      'elec_tension': r'$-E_%sE_%s$' % (ci, cj),
                      'total': r'$T^{%s%s}_{\rm tot}$' % (ci, cj)}
            values = dict(fields, total=tot)
            if i != j:
                values.pop('mag_pressure')
                values.pop('elec_pressure')

        self.series = []
        groups = [g for g in self.shown_species() if g in by_group]
        for key in self.selected_terms():
            if key in self.PARTICLE_TERMS:
                for group in groups:
                    vals = particle[group].get(key)
                    if vals is None:
                        continue
                    style = {'color': TERM_COLORS[key], 'dashes': SPECIES_DASHES[group]}
                    self.series.append((f'{key}|{group}', self.particle_label(key, group, ci, cj), vals, style))
                    if key == 'pressure' and self.mode() == 0 and self.GetPlotParam('split_pressure'):
                        for a in fb.AXES:
                            self.series.append(
                                (f'pressure_{a}|{group}',
                                 self.split_label(a, group, ci, cj),
                                 particle[group]['pressure_' + a],
                                 {'color': TERM_COLORS['pressure'], 'dashes': SPLIT_DASHES[a], 'lw': 0.9}))
            else:
                vals = values.get(key)
                if vals is None:
                    continue
                style = {'color': TERM_COLORS[key]}
                if key in ('residual', 'total'):
                    style['lw'] = 1.6
                self.series.append((key, labels[key], vals, style))

    def integrate(self, values, particle, by_group, i, j, f_factor, p_factor):
        '''Replace each force density f with int f d(slice axis), and relabel.

        Each integral's constant makes its mean that of minus the stress the force
        is -d_j of, so a term whose force is purely along the slice is minus that
        stress: the magnetic pressure term of the i = j component is -B^2/2.'''
        dh = self.stencil.dh
        ci, cj = COMPONENTS[i], COMPONENTS[j]
        stresses = fb.field_stress_terms(self.stencil, self.E, self.B, i, j, f_factor)
        values = {k: None if v is None else fb.integrate_force(v, dh, stresses.get(k))
                  for k, v in values.items()}
        new_particle = {}
        for group, terms in particle.items():
            stress = {k: v * p_factor for k, v in fb.stress_terms(by_group[group], self.stencil, i, j).items()}
            stress['pressure_' + cj] = stress['pressure']
            new_particle[group] = {k: None if v is None else fb.integrate_force(v, dh, stress.get(k))
                                   for k, v in terms.items()}
        d = r'\,d%s' % cj
        labels = {'mag_pressure': (r'$-B^2/2$' if i == j else r'$-\int\partial_%s B^2/2%s$' % (ci, d)),
                  'mag_tension': r'$\int\partial_k(B_%sB_k)%s$' % (ci, d),
                  'elec_pressure': (r'$-E^2/2$' if i == j else r'$-\int\partial_%s E^2/2%s$' % (ci, d)),
                  'elec_tension': r'$\int\partial_k(E_%sE_k)%s$' % (ci, d),
                  'em_momentum': r'$-\int\partial_t(\mathbf{E}\times\mathbf{B})_%s/c%s$' % (ci, d),
                  'residual': r'$\int$residual$%s$' % d}
        return values, labels, new_particle

    def split_label(self, a, group, ci, cj):
        sp = self.species_sup(group)
        if not self.integrated():
            return r'$-\partial_%s P%s_{%s%s}$' % (a, sp, ci, a)
        if a == cj:
            return r'$-P%s_{%s%s}$' % (sp, ci, a)
        return r'$-\int\partial_%s P%s_{%s%s}\,d%s$' % (a, sp, ci, a, cj)

    @staticmethod
    def species_sup(group):
        return '' if group == 'total' else '^{%s}' % SPECIES_TEX[group]

    def particle_label(self, key, group, ci, cj):
        '''The legend label of a particle term, marked with its species.'''
        sp = self.species_sup(group)
        if key in ('p_xx', 'p_yy', 'p_zz'):
            return r'$P%s_{%s}$' % (sp, key[2:])
        if self.integrated():
            d = r'\,d%s' % cj
            return {'pressure': r'$-\int(\nabla\cdot\mathsf{P}%s)_%s%s$' % (sp, ci, d),
                    'inertia': r'$-\int[\nabla\cdot(\varepsilon\mathbf{UU})%s]_%s%s$' % (sp, ci, d),
                    'heat': r'$-\int[\nabla\cdot(\mathbf{qU}+\mathbf{Uq})%s]_%s%s$' % (sp, ci, d),
                    'dpdt': r'$-\int\partial_t T^{0%s}%s%s$' % (ci, '' if group == 'total' else '_{%s}' % SPECIES_TEX[group], d)}[key]
        if self.mode() == 0:
            return {'pressure': r'$-(\nabla\cdot\mathsf{P}%s)_%s$' % (sp, ci),
                    'inertia': r'$-[\nabla\cdot(\varepsilon\mathbf{UU})%s]_%s$' % (sp, ci),
                    'heat': r'$-[\nabla\cdot(\mathbf{qU}+\mathbf{Uq})%s]_%s$' % (sp, ci),
                    'dpdt': r'$-\partial_t T^{0%s}%s$' % (ci, '' if group == 'total' else '_{%s}' % SPECIES_TEX[group])}[key]
        return {'pressure': r'$P%s_{%s%s}$' % (sp, ci, cj),
                'inertia': r'$(\varepsilon U_%sU_%s)%s$' % (ci, cj, sp),
                'heat': r'$(q_%sU_%s+U_%sq_%s)%s$' % (ci, cj, ci, cj, sp)}[key]

    def ylabel(self):
        unit = r'n_0 m_%s c^2' % self.mass_unit if getattr(self, 'has_ppc0', True) else r'm_%s c^2/{\rm cell}' % self.mass_unit
        if self.mode():
            return r'Stress $T^{%s%s}$  [$%s$]' % (COMPONENTS[self.component()], self.stencil.axis, unit)
        if self.integrated():
            return r'$\int f_%s\,d%s$  [$%s$]' % (COMPONENTS[self.component()], self.stencil.axis, unit)
        return r'Force density, $%s$  [$%s\,\omega_{\rm pe}/c$]' % (COMPONENTS[self.component()], unit)


####
#
# Settings windows
#
####

class BalanceSettings(Tk.Toplevel):
    '''The settings shared by both panels, laid out top to bottom: what is
    shown, where the slice is, how it is binned, and how it is drawn.'''

    TITLE = 'Balance'

    def __init__(self, parent):
        self.parent = parent
        Tk.Toplevel.__init__(self)
        self.wm_title('%s (%d,%d) Settings' % ((self.TITLE,) + tuple(self.parent.FigWrap.pos)))
        self.protocol('WM_DELETE_WINDOW', self.OnClosing)
        self.bind('<Return>', self.TxtEnter)
        self.frm = ttk.Frame(self, padding=6)
        self.frm.pack(fill=Tk.BOTH, expand=True)

        head = ttk.Frame(self.frm)
        head.grid(row=0, column=0, sticky=Tk.W + Tk.E, pady=(0, 4))
        self.ctypevar = Tk.StringVar(self)
        self.ctypevar.set(self.parent.chartType)
        self.ctypevar.trace('w', self.ctypeChanged)
        ttk.Label(head, text='Chart Type:').pack(side=Tk.LEFT)
        ttk.OptionMenu(head, self.ctypevar, self.parent.chartType,
                       *tuple(self.parent.ChartTypes)).pack(side=Tk.LEFT, padx=(2, 12))

        self.body = None
        self.build_body()

        self.status = ttk.Label(self.frm, text='', foreground='gray35', wraplength=460, justify=Tk.LEFT)
        self.status.grid(row=2, column=0, sticky=Tk.W, pady=(4, 0))
        self.show_status()

    def show_status(self):
        self.status.config(text=self.parent.status_text)

    def build_body(self):
        if self.body is not None:
            self.body.destroy()
        self.body = ttk.Frame(self.frm)
        self.body.grid(row=1, column=0, sticky=Tk.W + Tk.E)
        self.body.columnconfigure(0, weight=1)
        row = self.build_quantity(self.body, 0)
        self.build_terms(self.body, row)
        self.build_slice(self.body, row + 1)
        self.build_display(self.body, row + 2)

    def section(self, master, row, title):
        box = ttk.LabelFrame(master, text=title, padding=4)
        box.grid(row=row, column=0, sticky=Tk.W + Tk.E, pady=2)
        return box

    def radio_row(self, master, label, key, names, rebuild=False):
        line = ttk.Frame(master)
        ttk.Label(line, text=label).pack(side=Tk.LEFT)
        var = Tk.IntVar(self)
        var.set(self.parent.GetPlotParam(key))

        def handler():
            if var.get() != self.parent.GetPlotParam(key):
                self.parent.SetPlotParam(key, var.get(), update_plot=not rebuild)
                if rebuild:
                    self.build_body()
                    self.parent.SetPlotParam(key, var.get())
        for i, name in enumerate(names):
            ttk.Radiobutton(line, text=name, variable=var, value=i, command=handler).pack(side=Tk.LEFT, padx=(4, 2))
        setattr(self, key + '_var', var)
        return line

    def check(self, master, text, key, rebuild=False):
        var = Tk.IntVar(self)
        var.set(self.parent.GetPlotParam(key))

        def handler():
            if bool(var.get()) != bool(self.parent.GetPlotParam(key)):
                self.parent.SetPlotParam(key, bool(var.get()), update_plot=not rebuild)
                if rebuild:
                    self.build_body()
                    self.parent.SetPlotParam(key, bool(var.get()))
        setattr(self, key + '_var', var)
        return ttk.Checkbutton(master, text=text, variable=var, command=handler)

    def entry(self, master, var, width=7):
        return ttk.Entry(master, textvariable=var, width=width)

    def build_quantity(self, master, row):
        raise NotImplementedError

    # Which terms

    def build_terms(self, master, row):
        box = self.section(master, row, 'Terms')
        selected = self.parent.selected_terms()
        self.TermVars = {}
        for n, term in enumerate(self.parent.available_terms()):
            var = Tk.IntVar(self)
            var.set(term in selected)
            self.TermVars[term] = var
            ttk.Checkbutton(box, text=self.parent.TERM_NAMES[term], variable=var,
                            command=self.TermsChanged).grid(row=n // 2, column=n % 2, sticky=Tk.W, padx=(0, 12))
        extra = ttk.Frame(box)
        extra.grid(row=100, column=0, columnspan=2, sticky=Tk.W, pady=(4, 0))
        if self.show_split_option():
            self.check(extra, 'Split ∇·P into ∂_j P_ij for each j', 'split_pressure').pack(side=Tk.LEFT)
        ttk.Button(extra, text='Defaults', command=self.SelectDefaults).pack(side=Tk.LEFT, padx=(8, 0))

    def show_split_option(self):
        return True

    def TermsChanged(self):
        chosen = [t for t, var in self.TermVars.items() if var.get()]
        self.parent.SetPlotParam(self.parent.term_param(), ','.join(chosen) if chosen else 'none')

    def SelectDefaults(self):
        defaults = self.parent.default_terms()
        for t, var in self.TermVars.items():
            var.set(t in defaults)
        self.parent.SetPlotParam(self.parent.term_param(), '')

    # Where the slice is and how it is binned

    def build_slice(self, master, row):
        box = self.section(master, row, 'Slice & binning')
        plot_axes.add_axis_buttons(box, self, self.parent, row=0, column=0, two_d=False,
                                   on_change=self.build_body)
        axis = plot_axes.plot_axis_name(self.parent)
        loc = ttk.Frame(box)
        loc.grid(row=1, column=0, sticky=Tk.W, pady=(2, 0))
        how = 'Averaged over' if plot_axes.averages_lineouts(self.parent) else 'Through the centre of'
        ttk.Label(loc, text=how + ' the 2D view (set in the main window\'s "1D lineouts").',
                  foreground='gray35').pack(side=Tk.LEFT)

        line = ttk.Frame(box)
        line.grid(row=2, column=0, sticky=Tk.W, pady=(2, 0))
        ttk.Label(line, text='# bins along (0 = auto)').pack(side=Tk.LEFT)
        self.BinsVar = Tk.StringVar(self)
        self.BinsVar.set(str(self.parent.GetPlotParam('nbins')))
        self.entry(line, self.BinsVar, 6).pack(side=Tk.LEFT, padx=(2, 10))
        ttk.Label(line, text='width across [c/ωpe] (0 = auto)').pack(side=Tk.LEFT)
        self.WidthVar = Tk.StringVar(self)
        self.WidthVar.set(str(self.parent.GetPlotParam('trans_width')))
        self.entry(line, self.WidthVar, 6).pack(side=Tk.LEFT, padx=2)

        line2 = ttk.Frame(box)
        line2.grid(row=3, column=0, sticky=Tk.W, pady=(2, 0))
        ttk.Label(line2, text='Smoothing (boxcar, bins)').pack(side=Tk.LEFT)
        self.SmoothVar = Tk.StringVar(self)
        self.SmoothVar.set(str(self.parent.GetPlotParam('smooth')))
        self.entry(line2, self.SmoothVar, 4).pack(side=Tk.LEFT, padx=(2, 10))
        self.check(line2, 'Time derivatives (reads the neighbouring outputs)', 'time_deriv',
                   rebuild=True).pack(side=Tk.LEFT)

    # How it is drawn

    def build_display(self, master, row):
        box = self.section(master, row, 'Display')
        flags = ttk.Frame(box)
        flags.grid(row=0, column=0, sticky=Tk.W)
        self.check(flags, 'Symmetric about 0', 'symmetric').pack(side=Tk.LEFT)
        self.check(flags, 'Legend', 'show_legend').pack(side=Tk.LEFT, padx=6)
        self.check(flags, 'in the colorbar slot', 'legend_outside').pack(side=Tk.LEFT)
        lims = ttk.Frame(box)
        lims.grid(row=1, column=0, sticky=Tk.W, pady=(2, 0))
        self.Vmin = Tk.StringVar(self)
        self.Vmin.set(str(self.parent.GetPlotParam('v_min')))
        self.Vmax = Tk.StringVar(self)
        self.Vmax.set(str(self.parent.GetPlotParam('v_max')))
        self.check(lims, 'Set y min', 'set_v_min').pack(side=Tk.LEFT)
        self.entry(lims, self.Vmin).pack(side=Tk.LEFT, padx=(2, 10))
        self.check(lims, 'Set y max', 'set_v_max').pack(side=Tk.LEFT)
        self.entry(lims, self.Vmax).pack(side=Tk.LEFT, padx=2)

    # Text entries

    def TxtEnter(self, e):
        self.FieldsCallback()

    def FieldsCallback(self):
        changed = False
        for var, key, cast, low in ((self.Vmin, 'v_min', float, None), (self.Vmax, 'v_max', float, None),
                                    (self.BinsVar, 'nbins', int, 0), (self.WidthVar, 'trans_width', float, 0.0),
                                    (self.SmoothVar, 'smooth', int, 1)):
            try:
                value = cast(var.get())
                if low is not None and value < low:
                    raise ValueError
            except ValueError:
                var.set(str(self.parent.GetPlotParam(key)))
                continue
            if value != self.parent.GetPlotParam(key):
                self.parent.SetPlotParam(key, value, update_plot=False)
                if key in ('v_min', 'v_max'):
                    changed |= bool(self.parent.GetPlotParam('set_' + key))
                else:
                    changed = True

        if changed:
            self.parent.parent.RenewCanvas()

    def ctypeChanged(self, *args):
        if self.ctypevar.get() != self.parent.chartType:
            self.parent.ChangePlotType(self.ctypevar.get())
            self.destroy()

    def OnClosing(self):
        self.parent.settings_window = None
        self.destroy()


class OhmsLawSettings(BalanceSettings):
    TITLE = "Ohm's law"

    def build_quantity(self, master, row):
        box = self.section(master, row, "Momentum equation of one species, solved for E")
        self.radio_row(box, 'Species:', 'species', SPECIES_NAMES).grid(row=0, column=0, sticky=Tk.W)
        self.radio_row(box, 'Component:', 'component', COMPONENTS).grid(row=1, column=0, sticky=Tk.W)
        ttk.Label(box, text='E = −V×B + [∂T^0i/∂t + ∇·(εUU) + ∇·P + ∇·(qU+Uq)] / qn,  split in the frame of the '
                            'species\' particle flux.  In the units of the E field output.',
                  foreground='gray35', wraplength=460, justify=Tk.LEFT).grid(row=2, column=0, sticky=Tk.W, pady=(2, 0))
        return row + 1


class PressureBalanceSettings(BalanceSettings):
    TITLE = 'Pressure balance'

    def build_quantity(self, master, row):
        box = self.section(master, row, 'Total momentum balance')
        self.radio_row(box, 'Show:', 'mode', ('Force densities  (−∂_j T^ij)', 'Stresses  T^ij along the slice'),
                       rebuild=True).grid(row=0, column=0, sticky=Tk.W)
        if not self.parent.mode():
            self.check(box, 'Integrate along the slice:  −∫ f dx, e.g. B²/2 rather than −∂B²/2',
                       'integrate').grid(row=4, column=0, sticky=Tk.W, pady=(2, 0))
        self.radio_row(box, 'Component i:', 'component', COMPONENTS).grid(row=1, column=0, sticky=Tk.W)
        self.radio_row(box, 'Particle terms of:', 'species',
                       ('All', 'Ions', 'Electrons', 'Each species')).grid(row=2, column=0, sticky=Tk.W)
        if self.parent.mode():
            text = ('The (i, j) components of each momentum flux, j being the axis the slice runs along. '
                    'In a steady 1D state their total is constant along the slice.')
        else:
            text = ('Each force density; with every term included they add up to zero. The residual always '
                    'includes every term, with the particle terms of all species together. Integrated, each '
                    'term\'s constant makes its mean that of its stress T^ij along the slice (0 if none).')
        ttk.Label(box, text=text + '  Code units (magnetic pressure B²/2), per n0 m c², n0 = ppc0 per cell.', foreground='gray35',
                  wraplength=460, justify=Tk.LEFT).grid(row=3, column=0, sticky=Tk.W, pady=(2, 0))
        return row + 1

    def show_split_option(self):
        return not self.parent.mode()


OhmsLawPanel.SETTINGS = OhmsLawSettings
PressureBalancePanel.SETTINGS = PressureBalanceSettings
