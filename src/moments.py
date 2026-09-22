
#!/usr/bin/env python
import time
import tkinter as Tk
from tkinter import ttk
import matplotlib
import numpy as np
import new_cmaps
from new_cnorms import PowerNormWithNeg, PowerNormFunc
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib.ticker import FuncFormatter
from NumbaMoments import stepify
import plot_axes
import stress_energy as se

# The species a moment can be taken over. The first two are the prtl_type
# indices used everywhere else in Iseult; 'total' adds the two up.
SPECIES = ('ions', 'electrons', 'total')
SPECIES_TEX = ('i', 'e', r'\rm tot')

# Line dash patterns used to tell components apart when the colour is used for
# the species. None is a solid line.
DASHES = [None, (1, 1), (5, 1), (3, 1, 1, 1), (6, 2), (2, 2), (8, 2, 2, 2),
          (1, 3), (4, 1, 1, 1, 1, 1), (10, 3)]

# The out-of-plane axis of each pair of in-plane axes
THIRD_AXIS = {frozenset('xy'): 'z', frozenset('xz'): 'y', frozenset('yz'): 'x'}


class  MomentsPanel:
    # A dictionary of all of the parameters for this plot with the default parameters

    plot_param_dict = {'twoD': 0,
                       'm_type': 0, # Which quantity; an index into stress_energy.FAMILIES:
                                    # 0 = 3-velocity, 1 = 4-velocity, 2 = energy, 3 = stress-energy tensor
                       'basis': 0, # 0 = lab x, y, z; 1 = parallel/perpendicular to the local B
                       'components': '', # Comma separated component names, see stress_energy.components.
                                         # Empty means use the old show_x/y/z flags; 'none' means none.
                       'normalization': 0, # Stress-energy only: 0 = per unit volume, 1 = per particle
                       'mass_weight': False, # 4-velocity only: show the momentum m u instead of u
                       'species_2d': 2, # The species a 2D map shows: 0 = ions, 1 = electrons, 2 = total
                       'v_min': 0,
                       'v_max' : 10,
                       'set_v_min': False,
                       'set_v_max': False,
                       'show_x': True, # Only read from configs saved before 'components' existed
                       'show_y': False,
                       'show_z': False,
                       'show_ions': True,
                       'show_electrons': True,
                       'show_total': False,
                       'UpstreamFrame': False,
                       'weighted': False,
                       'xbins': 100, # The bins along the profile, or along the horizontal axis in 2D
                       'ybins': 100, # The bins along the vertical axis in 2D
                       'slab_half_width': 0.0, # 2D only: use particles within this distance [c/omega_pe]
                                               # of the slice, 0 = the whole depth
                       'spatial_x': True,
                       'show_legend': True,
                       'spatial_y': False,
                       'symmetric': False,
                       'logy': False,
                       'legend_loc': 'N/A',
                       'filter_by_viewport': True,
                       'cnorm_type': 'Linear', # 2D colour norm: Linear, Pow or Log
                       'cpow_num': 0.6, # The gamma of the Pow norm
                       'UseDivCmap': False,
                       'cmap': 'None', # If cmap is none, the plot will inherit the parent's cmap
                       'show_cbar': True,
                       'show_labels': True,
                       'OutlineText': True,
                       'interpolation': 'none',
                       'face_color': 'gainsboro'
                       } # legend_loc is a string that stores the
                                         # location of the legend in figure pixels.
                                         # Unfortunately it is not always up to date.
                                         # N/A plots it at the 'best' location.

    # A 1D moment is a profile along one spatial axis, a 2D one a map of the
    # slice plane, which can be rotated like the other 2D panels.
    plot_axes.add_axis_params(plot_param_dict, two_d=True)

    gradient =  np.linspace(0, 1, 256)# A way to make the colorbar display better
    gradient = np.vstack((gradient, gradient))

    def __init__(self, parent, figwrapper):
        self.settings_window = None
        self.FigWrap = figwrapper
        self.parent = parent
        self.ChartTypes = self.FigWrap.PlotTypeDict.keys()
        self.chartType = self.FigWrap.chartType
        self.figure = self.FigWrap.figure
        self.SetPlotParam('spatial_y', self.GetPlotParam('twoD'), update_plot = False)
        self.status_text = ''
        self.InterpolationMethods = ['none','nearest', 'bilinear', 'bicubic', 'spline16',
            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

    ####
    #
    # What is being shown
    #
    ####

    def family(self):
        '''The quantity being shown: 'beta', 'u', 'energy' or 'T'.'''
        m_type = self.GetPlotParam('m_type')
        return se.FAMILIES[m_type] if 0 <= m_type < len(se.FAMILIES) else se.FAMILIES[0]

    def basis(self):
        '''The basis vectors and tensors are shown in: 'lab' or 'fa'.'''
        if self.family() == 'energy':
            return 'lab'
        return se.BASES[1] if self.GetPlotParam('basis') else se.BASES[0]

    def selected_components(self):
        '''The components to show, in the order the settings window lists them.'''
        family, basis = self.family(), self.basis()
        allowed = se.components(family, basis)
        stored = self.GetPlotParam('components')
        if stored == 'none':
            chosen = []
        elif stored:
            chosen = [c for c in stored.split(',') if c in allowed]
        else:
            chosen = self.legacy_components(family, basis)
        if not chosen and stored != 'none':
            chosen = se.default_components(family, basis)
        chosen = [c for c in allowed if c in chosen]
        if self.GetPlotParam('twoD'):
            # A map can only show one thing
            return chosen[:1] or se.default_components(family, basis)
        return chosen

    def legacy_components(self, family, basis):
        '''The components a config saved before the stress-energy tensor existed asked for.'''
        if family == 'energy':
            return [c for c, key in (('thermal', 'show_x'), ('ke', 'show_y')) if self.GetPlotParam(key)]
        if basis == 'lab' and family in ('beta', 'u'):
            return [c for c in 'xyz' if self.GetPlotParam('show_' + c)]
        return []

    def shown_species(self):
        '''The indices into SPECIES that are drawn.'''
        if self.GetPlotParam('twoD'):
            species = self.GetPlotParam('species_2d')
            return [species if species in (0, 1, 2) else 2]
        keys = ('show_ions', 'show_electrons', 'show_total')
        return [i for i, key in enumerate(keys) if self.GetPlotParam(key)]

    def needs_field(self):
        family = self.family()
        return any(se.needs_field(family, c) for c in self.selected_components())

    def spatial_plot_axes(self):
        '''The physical axes of this panel. The moment axis of a profile is not one.'''
        if self.GetPlotParam('twoD'):
            return plot_axes.two_d_axes(self)
        return plot_axes.plot_axis_name(self), None

    ####
    #
    # Loading and binning the particles
    #
    ####

    def binning_spec(self):
        '''Everything that decides which particles land in which bin.

        The binned sums are cached under a key made from this, so anything that
        changes the sums must be in it, and nothing else should be: changing
        which component is shown must not trigger a new pass over the particles.'''
        two_d = bool(self.GetPlotParam('twoD'))
        viewport = None
        if not two_d and self.GetPlotParam('filter_by_viewport') and self.parent.is_viewport_zoomed():
            # A 2D map covers the whole plane, so it is never restricted to the
            # viewport; being the viewport itself, it would otherwise chase its
            # own zoom.
            viewport = self.parent.get_active_viewport()
        if two_d:
            h_axis, v_axis = plot_axes.two_d_axes(self)
            nh, nv = int(self.GetPlotParam('xbins')), int(self.GetPlotParam('ybins'))
            slab = float(self.GetPlotParam('slab_half_width'))
        else:
            h_axis, v_axis = plot_axes.plot_axis_name(self), None
            nh, nv = int(self.GetPlotParam('xbins')), 1
            slab = 0.0
        third = THIRD_AXIS[frozenset(h_axis + v_axis)] if two_d else None
        slab_key = ''
        if slab > 0:
            slab_key = f'_slab{slab:g}@{self._slice_index(third)}'
        base = '|'.join(['moments', '2D' if two_d else '1D', h_axis, str(v_axis),
                         str(max(nh, 1)), str(max(nv, 1)),
                         'W' if self.GetPlotParam('weighted') else '',
                         str(self.parent.MainParamDict['PrtlStride']),
                         plot_axes.viewport_key(viewport), slab_key])
        return {'two_d': two_d, 'h_axis': h_axis, 'v_axis': v_axis, 'third': third,
                'nh': max(nh, 1), 'nv': max(nv, 1), 'viewport': viewport, 'slab': slab,
                'weighted': bool(self.GetPlotParam('weighted')),
                'field_aligned': self.needs_field(),
                'key_lab': base + '|lab', 'key_fa': base + '|fa'}

    @staticmethod
    def cached(data_dict, spec):
        '''The binned sums for `spec` if they have been worked out already.'''
        if data_dict is None:
            return None
        if spec['field_aligned']:
            return data_dict.get(spec['key_fa'])
        # The field-aligned pass holds everything the lab one does
        return data_dict.get(spec['key_lab'], data_dict.get(spec['key_fa']))

    def target_data_dict(self):
        '''The DataDict of the timestep about to be shown, if it has been visited.

        set_plot_keys is asked for its keys before MainApp swaps in the
        DataDict of the new timestep, so the parent's DataDict may still be
        the previous one.'''
        p = self.parent
        try:
            if not p.NewDirectory and p.TimeStep.value in p.timestep_visited:
                return p.ListOfDataDict[p.timestep_visited.index(p.TimeStep.value)]
        except (AttributeError, ValueError, IndexError):
            pass
        return None

    def set_plot_keys(self):
        '''A helper function that will insure that each hdf5 file will only be
        opened once per time step'''
        self.arrs_needed = ['c_omp', 'istep', 'me', 'mi']
        if self.family() == 'T' and self.GetPlotParam('normalization') == 0:
            self.arrs_needed += ['stride', 'ppc0']

        spec = self.binning_spec()
        if self.cached(self.target_data_dict(), spec) is not None:
            # The particles have already been binned for this timestep, and
            # the raw particle data is dropped after every draw, so don't ask
            # for it to be read back in from disk.
            return self.arrs_needed

        self.arrs_needed += ['ui', 'vi', 'wi', 'ue', 've', 'we']
        if spec['weighted']:
            self.arrs_needed += ['chi', 'che']

        # The coordinates the bins run along, every coordinate a selected
        # region constrains, and, to find the field at each particle, all of them.
        needed_axes = {spec['h_axis']}
        if spec['v_axis'] is not None:
            needed_axes.add(spec['v_axis'])
        if spec['viewport'] is not None:
            needed_axes |= {axis for axis, low, high in spec['viewport'] if axis is not None}
        if spec['slab'] > 0:
            needed_axes.add(spec['third'])
        if spec['field_aligned']:
            needed_axes |= {'x', 'y', 'z'}
            self.arrs_needed += ['bx', 'by', 'bz']
        if spec['h_axis'] != 'x':
            needed_axes.add('x') # the fallback for data without that coordinate

        for prtl_type in (0, 1):
            for key in plot_axes.available_position_keys(self, prtl_type, sorted(needed_axes)):
                if key not in self.arrs_needed:
                    self.arrs_needed.append(key)

        return self.arrs_needed

    def _slice_index(self, axis):
        # Not known until MainApp has read the field shape of the first timestep
        return getattr(self.parent, axis + 'Slice', 0)

    def _max_index(self, axis):
        return {'x': self.parent.MaxXInd, 'y': self.parent.MaxYInd, 'z': self.parent.MaxZInd}[axis]

    def extent_cells(self, axis):
        '''The size of the domain along `axis` in cells. A dimension the
        simulation does not have is one cell thick.'''
        max_index = self._max_index(axis)
        return 1.0 if max_index == 0 else (max_index + 1) * self.istep

    def raw_positions(self, prtl_type, axis, n):
        '''Particle positions along `axis` in cells, or None if the data does not have them.'''
        try:
            coord = self.FigWrap.LoadKey(plot_axes.PRTL_POS_KEYS[prtl_type][axis])
        except KeyError:
            return None
        coord = np.asanyarray(coord)
        if coord.ndim != 1 or coord.shape[0] != n:
            return None
        return coord

    def LoadData(self):
        ''' A helper function that checks if the moments have
        already been calculated and if they haven't, it calculates
        them then stores them.'''
        self.c_omp = self.FigWrap.LoadKey('c_omp')
        self.istep = self.FigWrap.LoadKey('istep')
        me, mi = self.FigWrap.LoadKey('me'), self.FigWrap.LoadKey('mi')
        self.totalcolor = new_cmaps.cmaps[self.parent.MainParamDict['ColorMap']](0.0)

        # Masses in units of the ion mass, as the old moments were. A run
        # without ions is shown in units of the electron mass instead.
        with np.errstate(divide='ignore', invalid='ignore'):
            memi = np.float64(me) / np.float64(mi)
        if np.isfinite(memi) and memi > 0:
            self.masses = (1.0, float(memi))
            self.mass_unit = 'i'
        else:
            self.masses = (1.0, 1.0)
            self.mass_unit = 'e'
            self.SetPlotParam('show_ions', False, update_plot = False)

        if self.GetPlotParam('cmap') == 'None':
            if self.GetPlotParam('UseDivCmap'):
                self.cmap = self.parent.MainParamDict['DivColorMap']
            else:
                self.cmap = self.parent.MainParamDict['ColorMap']
        else:
            self.cmap = self.GetPlotParam('cmap')

        spec = self.binning_spec()
        self.viewport = spec['viewport']
        if self.viewport is not None:
            self.parent.last_phase_viewport = self.viewport

        entry = self.cached(self.parent.DataDict, spec)
        if entry is None:
            entry = self.integrate(spec)
            self.parent.DataDict[entry['key']] = entry
            self.status_text = (f"Binned {entry['n_particles']:,} particles into "
                                f"{entry['n_bins']:,} bins in {entry['seconds']:.2f} s"
                                + (', with B at each particle' if spec['field_aligned'] else ''))
        else:
            self.status_text = (f"Using the moments already binned for this timestep "
                                f"({entry['n_particles']:,} particles, {entry['n_bins']:,} bins)")
        self.moments = entry
        self.prof_axis = entry['h_axis']
        self.dens_factor, self.has_ppc0 = self.density_factor(entry)

        if self.settings_window is not None:
            try:
                self.settings_window.show_status()
            except Tk.TclError:
                pass

    def density_factor(self, entry):
        '''What turns a per-bin sum into a density, in units of ppc0 per cell,
        the same n_0 the density panel normalizes to. Also whether ppc0 was known.'''
        if not (self.family() == 'T' and self.GetPlotParam('normalization') == 0):
            return 1.0, True
        try:
            stride = float(np.squeeze(self.FigWrap.LoadKey('stride')))
        except (KeyError, TypeError, ValueError):
            stride = 1.0
        stride = stride if np.isfinite(stride) and stride > 0 else 1.0
        try:
            ppc0 = float(np.squeeze(self.FigWrap.LoadKey('ppc0')))
        except (KeyError, TypeError, ValueError):
            ppc0 = np.nan
        has_ppc0 = np.isfinite(ppc0) and ppc0 > 0
        per_particle = stride * self.parent.MainParamDict['PrtlStride']
        return per_particle / (entry['bin_volume'] * (ppc0 if has_ppc0 else 1.0)), has_ppc0

    def integrate(self, spec):
        '''Bin the moments of both species. This is the only place the particles are touched.'''
        start = time.perf_counter()
        h_axis, v_axis = spec['h_axis'], spec['v_axis']

        # The ranges the bins cover, in c/omega_pe
        h_range = (0.0, plot_axes.domain_extent(self, h_axis))
        v_range = (0.0, plot_axes.domain_extent(self, v_axis)) if spec['two_d'] else None
        # How far the bin extends along the axes not binned, in cells
        other_axes = [a for a in 'xyz' if a not in (h_axis, v_axis)]
        other_extent = {a: self.extent_cells(a) for a in other_axes}
        if spec['viewport'] is not None:
            for axis, low, high in spec['viewport']:
                if axis == h_axis:
                    h_range = (low, high)
                elif axis in other_extent:
                    top = plot_axes.domain_extent(self, axis)
                    width = (min(high, top) - max(low, 0.0)) * self.c_omp
                    other_extent[axis] = max(width, 0.0) if self._max_index(axis) > 0 else 1.0
        slab_range = None
        if spec['slab'] > 0 and self._max_index(spec['third']) > 0:
            center = self._slice_index(spec['third']) * self.istep / self.c_omp
            slab_range = (center - spec['slab'], center + spec['slab'])
            top = plot_axes.domain_extent(self, spec['third'])
            other_extent[spec['third']] = (min(slab_range[1], top) - max(slab_range[0], 0.0)) * self.c_omp
        if h_range[1] <= h_range[0]:
            h_range = (h_range[0], h_range[0] + 1.0)

        bfield = None
        if spec['field_aligned']:
            bfield = tuple(np.ascontiguousarray(self.FigWrap.LoadKey(k), dtype = np.float32)
                           for k in ('bx', 'by', 'bz'))

        sums = []
        n_particles = 0
        for prtl_type in (0, 1):
            names = ('ui', 'vi', 'wi') if prtl_type == 0 else ('ue', 've', 'we')
            u, v, w = (self.FigWrap.LoadKey(k) for k in names)
            n = len(u)
            n_particles += n

            h = self.raw_positions(prtl_type, h_axis, n)
            if h is None and h_axis != 'x' and not spec['two_d']:
                # e.g. a 1D run has no y, so fall back to plotting against x
                self.SetPlotParam('plot_axis', 0, update_plot = False)
                return self.integrate(self.binning_spec())
            if h is None:
                raise ValueError(f'The particle data has no {h_axis} positions to bin along.')
            vpos = None
            if spec['two_d']:
                vpos = self.raw_positions(prtl_type, v_axis, n)
                if vpos is None:
                    raise ValueError(f'The particle data has no {v_axis} positions, '
                                     'which a 2D map of the moments needs.')

            mask = None
            if spec['viewport'] is not None or slab_range is not None:
                mask = np.ones(n, dtype=bool)
                if spec['viewport'] is not None:
                    plot_axes.filter_by_viewport(self, spec['viewport'], prtl_type, mask)
                if slab_range is not None:
                    depth = plot_axes.load_positions(self, prtl_type, spec['third'], n_expected=n)
                    if depth is not None:
                        mask &= (depth >= slab_range[0]) & (depth <= slab_range[1])

            weights = None
            if spec['weighted']:
                weights = np.asanyarray(self.FigWrap.LoadKey('chi' if prtl_type == 0 else 'che'))
                if weights.ndim != 1 or len(weights) != n:
                    print('Moments: the data has no per-particle charges to weight by; ignoring the weighting.')
                    weights = None
                else:
                    weights = np.abs(weights)

            positions = None
            fallback = (0.0, 0.0, 0.0)
            if bfield is not None:
                positions = tuple(self.raw_positions(prtl_type, a, n) for a in 'xyz')
                # A coordinate the particles lack is taken to be at the slice
                fallback = tuple(self._slice_index(a) * self.istep for a in 'xyz')

            sums.append(se.bin_moments(
                u, v, w, h, (h_range[0] * self.c_omp, h_range[1] * self.c_omp), spec['nh'],
                vpos = vpos,
                v_range = None if v_range is None else (v_range[0] * self.c_omp, v_range[1] * self.c_omp),
                nv = spec['nv'], weights = weights, mask = mask,
                bfield = bfield, positions = positions, istep = self.istep,
                fallback_position = fallback))

        h_edges = np.linspace(h_range[0], h_range[1], spec['nh'] + 1)
        v_edges = np.linspace(v_range[0], v_range[1], spec['nv'] + 1) if spec['two_d'] else None
        bin_volume = (h_range[1] - h_range[0]) * self.c_omp / spec['nh']
        if spec['two_d']:
            bin_volume *= (v_range[1] - v_range[0]) * self.c_omp / spec['nv']
        for axis in other_axes:
            bin_volume *= other_extent[axis]
        return {'sums': sums, 'h_edges': h_edges, 'v_edges': v_edges, 'h_axis': h_axis,
                'key': spec['key_fa'] if spec['field_aligned'] else spec['key_lab'],
                'bin_volume': bin_volume, 'n_particles': n_particles,
                'n_bins': spec['nh'] * spec['nv'], 'seconds': time.perf_counter() - start}

    ####
    #
    # Evaluating the components
    #
    ####

    def values(self, species, comp):
        '''One component, for one species (an index into SPECIES), in every bin.'''
        sums = self.moments['sums']
        if species == 2:
            # A run without ions has only its electrons to add up
            parts = [0, 1] if self.mass_unit == 'i' else [1]
        else:
            parts = [species]
        num, mass = se.combine([sums[p] for p in parts], [self.masses[p] for p in parts])
        return se.evaluate(num, mass, self.family(), comp,
                           normalization = 'particle' if self.GetPlotParam('normalization') else 'density',
                           dens_factor = self.dens_factor,
                           mass_weight = bool(self.GetPlotParam('mass_weight')))

    def series(self):
        '''(species, component) pairs for every line of a profile.'''
        return [(s, c) for s in self.shown_species() for c in self.selected_components()]

    def comp_tex(self, comp):
        return se.tex_label(self.family(), comp, bool(self.GetPlotParam('mass_weight')))

    def units_tex(self):
        '''The units of the quantity shown, or '' for a dimensionless one.'''
        m = r'm_%s' % self.mass_unit
        family = self.family()
        if family == 'u' and self.GetPlotParam('mass_weight'):
            return r'%s c' % m
        if family == 'energy':
            comps = self.selected_components()
            return '' if comps == ['gamma_bulk'] else r'%s c^2' % m
        if family == 'T':
            if self.GetPlotParam('normalization'):
                return r'%s c^2\ /\ {\rm particle}' % m
            if self.has_ppc0:
                return r'n_0 %s c^2' % m
            return r'%s c^2\ /\ {\rm cell}' % m
        return ''

    def ylabel(self):
        comps = self.selected_components()
        family = self.family()
        if len(comps) == 1:
            body = self.comp_tex(comps[0])
        else:
            body = {'beta': r'\langle\beta\rangle',
                    'u': r'\langle p^\mu\rangle' if self.GetPlotParam('mass_weight') else r'\langle u^\mu\rangle',
                    'energy': r'E',
                    'T': r'T^{\mu\nu}'}[family]
        units = self.units_tex()
        return '$' + body + (r'\ [' + units + ']' if units else '') + '$'

    ####
    #
    # Drawing
    #
    ####

    def draw(self):

        ''' A function that draws the data. In the interest in speeding up the
        code, draw should only be called when you want to recreate the whole
        figure, i.e. it  will be slow. Most times you will only want to update
        what has changed in the figure. This will be done in a function called
        refresh, that should be much much faster.'''

        # Set the tick color
        tick_color = 'black'

        # Create a gridspec to handle spacing better
        self.gs = gridspec.GridSpecFromSubplotSpec(100,100, subplot_spec = self.parent.gs0[self.FigWrap.pos])
        extent = self.gs[self.parent.axes_extent[0]:self.parent.axes_extent[1], self.parent.axes_extent[2]:self.parent.axes_extent[3]]

        share_x_ax, share_y_ax = self.parent.GetSharedAxes(self.FigWrap.pos)
        if self.GetPlotParam('twoD'):
            self.axes = self.figure.add_subplot(extent, sharex = share_x_ax, sharey = share_y_ax)
        else:
            self.axes = self.figure.add_subplot(extent, sharex = share_x_ax)

        if int(matplotlib.__version__[0]) < 2:
            self.axes.set_axis_bgcolor(self.GetPlotParam('face_color'))
        else:
            self.axes.set_facecolor(self.GetPlotParam('face_color'))
        self.axes.tick_params(labelsize = self.parent.MainParamDict['NumFontSize'], color=tick_color)

        if self.GetPlotParam('twoD'):
            self.draw_2d()
        else:
            self.lines = {}
            self.legend = None
            self.update_1d()

    def refresh(self):

        '''This is a function that will be called only if self.axes already
        holds a moments type plot. We only update things that have changed & are
        shown. The plot will be redrawn after all subplots are refreshed. '''
        if self.GetPlotParam('twoD'):
            self.refresh_2d()
        else:
            self.update_1d()

    # 1D

    def line_style(self, species, comp, n_species, comp_index):
        '''Colour by species and dashes by component, unless only one species
        is shown, when the components are easier to tell apart by colour.'''
        if n_species == 1 and len(self.selected_components()) > 1:
            return {'color': matplotlib.cm.tab10(comp_index % 10), 'dashes': None}
        color = (self.parent.ion_color, self.parent.electron_color, self.totalcolor)[species]
        return {'color': color, 'dashes': DASHES[comp_index % len(DASHES)]}

    def update_1d(self):
        '''Make the lines match the selected species and components, and fill them in.

        Lines are added and removed as the selection changes, so changing what
        is shown needs only a refresh, not a redraw of the whole figure.'''
        series = self.series()
        wanted = set(series)
        for key in list(self.lines):
            if key not in wanted:
                self.lines.pop(key).remove()

        comps = self.selected_components()
        n_species = len(self.shown_species())
        x_edges = self.moments['h_edges']
        handles, labels = [], []
        ymin, ymax = np.inf, -np.inf
        for species, comp in series:
            style = self.line_style(species, comp, n_species, comps.index(comp))
            if (species, comp) not in self.lines:
                line, = self.axes.plot([], [], ls = '-', color = style['color'])
                self.lines[(species, comp)] = line
            line = self.lines[(species, comp)]
            line.set_color(style['color'])
            if style['dashes'] is None:
                line.set_linestyle('-')
            else:
                line.set_dashes(style['dashes'])
            vals = np.asarray(self.values(species, comp), dtype = np.float64)
            line.set_data(*stepify(x_edges, vals))
            finite = vals[np.isfinite(vals)]
            if self.GetPlotParam('logy'):
                finite = finite[finite > 0]
            if finite.size:
                ymin, ymax = min(ymin, finite.min()), max(ymax, finite.max())
            handles.append(line)
            labels.append('${' + self.comp_tex(comp) + '}_{' + SPECIES_TEX[species] + '}$')

        self.axes.set_yscale('log' if self.GetPlotParam('logy') else 'linear')
        if np.isfinite(ymin):
            if self.GetPlotParam('logy'):
                lo, hi = np.log10(ymin), np.log10(ymax)
                pad = 0.04 * (hi - lo) if hi > lo else 0.5
                ylims = [10**(lo - pad), 10**(hi + pad)]
            else:
                pad = 0.04 * (ymax - ymin) if ymax > ymin else max(abs(ymax), 1.0) * 0.04
                ylims = [ymin - pad, ymax + pad]
                if self.GetPlotParam('symmetric'):
                    top = max(abs(ylims[0]), abs(ylims[1]))
                    ylims = [-top, top]
            self.axes.set_ylim(ylims)
        if self.GetPlotParam('set_v_min'):
            self.axes.set_ylim(bottom = self.GetPlotParam('v_min'))
        if self.GetPlotParam('set_v_max'):
            self.axes.set_ylim(top = self.GetPlotParam('v_max'))

        self.make_legend(handles, labels)
        plot_axes.apply_limits(self)
        self.axes.set_xlabel(plot_axes.AXIS_LABELS[self.prof_axis], labelpad = self.parent.MainParamDict['xLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])
        self.axes.set_ylabel(self.ylabel(), labelpad = self.parent.MainParamDict['yLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])

    def make_legend(self, handles, labels):
        if self.legend is not None:
            # Keep wherever the user dragged the old one to
            if self.legend._get_loc() != 1:
                self.SetPlotParam('legend_loc', ' '.join(str(x) for x in self.legend._get_loc()), update_plot = False)
            self.legend.remove()
        self.legend = self.axes.legend(handles, labels,
                                       framealpha = .05, fontsize = self.parent.MainParamDict['legendLabelSize'],
                                       loc = 1, ncol = max(1, len(self.shown_species())))
        self.legend.get_frame().set_facecolor('k')
        self.legend.get_frame().set_linewidth(0.0)
        self.legend.set_visible(bool(self.GetPlotParam('show_legend')) and bool(handles))
        self.legend.set_draggable(True, update = 'loc')
        if self.GetPlotParam('legend_loc') != 'N/A':
            tmp_tup = float(self.GetPlotParam('legend_loc').split()[0]),float(self.GetPlotParam('legend_loc').split()[1])
            self.legend._set_loc(tmp_tup)

    # 2D

    def image(self):
        return np.asarray(self.values(self.shown_species()[0], self.selected_components()[0]), dtype = np.float64)

    def two_d_label(self):
        comp = self.selected_components()[0]
        species = self.shown_species()[0]
        label = '${' + self.comp_tex(comp) + '}_{' + SPECIES_TEX[species] + '}$'
        units = self.units_tex()
        if units:
            label += '  $[' + units + ']$'
        return label

    def norm(self, vmin=None, vmax=None):
        if self.GetPlotParam('cnorm_type') == 'Log':
            return mcolors.LogNorm(vmin, vmax)
        if self.GetPlotParam('cnorm_type') == 'Pow':
            return PowerNormWithNeg(self.GetPlotParam('cpow_num'), vmin, vmax,
                                    div_cmap = self.GetPlotParam('UseDivCmap'), midpoint = 0.0,
                                    stretch_colors = self.GetPlotParam('symmetric'))
        if self.GetPlotParam('UseDivCmap'):
            return PowerNormWithNeg(1.0, vmin, vmax, midpoint = 0.0,
                                    stretch_colors = self.GetPlotParam('symmetric'))
        return mcolors.Normalize(vmin, vmax)

    def image_extent(self):
        h, v = self.moments['h_edges'], self.moments['v_edges']
        return [h[0], h[-1], v[0], v[-1]]

    def set_color_limits(self, img):
        finite = img[np.isfinite(img)]
        if self.GetPlotParam('cnorm_type') == 'Log':
            finite = finite[finite > 0]
        vmin, vmax = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        if self.GetPlotParam('set_v_min'):
            vmin = self.GetPlotParam('v_min')
        if self.GetPlotParam('set_v_max'):
            vmax = self.GetPlotParam('v_max')
        if self.GetPlotParam('symmetric') and self.GetPlotParam('cnorm_type') != 'Log':
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        if self.GetPlotParam('cnorm_type') == 'Log':
            vmin = vmin if vmin > 0 else (vmax * 1e-3 if vmax > 0 else 1e-3)
            vmax = vmax if vmax > vmin else vmin * 10
        elif vmax <= vmin:
            pad = abs(vmin) * 0.05 if vmin != 0 else 1.0
            vmin, vmax = vmin - pad, vmax + pad
        self.cax.norm.vmin = vmin
        self.cax.norm.vmax = vmax

    def masked(self, img):
        if self.GetPlotParam('cnorm_type') == 'Log':
            return np.ma.masked_where(~(img > 0), img)
        return np.ma.masked_invalid(img)

    def draw_2d(self):
        if self.GetPlotParam('OutlineText'):
            annotate_kwargs = {'horizontalalignment': 'right', 'verticalalignment': 'top',
                               'size' : self.parent.MainParamDict['annotateTextSize'],
                               'path_effects' : [PathEffects.withStroke(linewidth=1.5,foreground="k")]}
        else:
            annotate_kwargs = {'horizontalalignment' : 'right', 'verticalalignment' : 'top',
                               'size' : self.parent.MainParamDict['annotateTextSize']}

        img = self.image()
        self.cax = self.axes.imshow(self.masked(img), norm = self.norm(), origin = 'lower',
                                    extent = self.image_extent(),
                                    interpolation = self.GetPlotParam('interpolation'),
                                    **plot_axes.image_kwargs(self))
        self.cax.set_cmap(new_cmaps.cmaps[self.cmap])
        self.set_color_limits(img)

        self.an_2d = self.axes.annotate(self.two_d_label(),
                                        xy = (0.9,.9),
                                        xycoords= 'axes fraction',
                                        color = 'white',
                                        **annotate_kwargs)
        self.an_2d.set_visible(self.GetPlotParam('show_labels'))

        self.axC = self.figure.add_subplot(self.gs[self.parent.cbar_extent[0]:self.parent.cbar_extent[1], self.parent.cbar_extent[2]:self.parent.cbar_extent[3]])
        self.parent.cbarList.append(self.axC)
        if self.parent.MainParamDict['HorizontalCbars']:
            self.cbar = self.axC.imshow(self.gradient, aspect='auto', cmap=new_cmaps.cmaps[self.cmap])
            self.cbar.set_extent([0, 1.0, 0, 1.0])
            self.axC.tick_params(axis='x', which = 'both', top = False,
                                 labelsize=self.parent.MainParamDict['NumFontSize'])
            self.axC.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        else:
            self.cbar = self.axC.imshow(np.transpose(self.gradient)[::-1], aspect='auto', cmap=new_cmaps.cmaps[self.cmap])
            self.cbar.set_extent([0, 1.0, 0, 1.0])
            self.axC.tick_params(axis='x', which = 'both', top = False, bottom = False,
                                 labelbottom = False, labelsize=self.parent.MainParamDict['NumFontSize'])
            self.axC.tick_params(axis='y', which='both', left=False, right=True, labelleft=False,
                                 labelright = True, labelsize = self.parent.MainParamDict['NumFontSize'])
        if not self.GetPlotParam('show_cbar'):
            self.axC.set_visible(False)
        else:
            self.CbarTickFormatter()

        self.set_2d_limits_and_labels()

    def refresh_2d(self):
        img = self.image()
        self.cax.set_data(self.masked(img))
        self.cax.set_extent(self.image_extent())
        self.set_color_limits(img)
        self.an_2d.set_text(self.two_d_label())
        if self.GetPlotParam('show_cbar'):
            self.CbarTickFormatter()
        self.set_2d_limits_and_labels()

    def set_2d_limits_and_labels(self):
        h0, h1, v0, v1 = self.image_extent()
        plot_axes.apply_limits(self, (h0, h1), (v0, v1))
        horiz_axis, vert_axis = plot_axes.two_d_axes(self)
        self.axes.set_xlabel(plot_axes.AXIS_LABELS[horiz_axis], labelpad = self.parent.MainParamDict['xLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])
        self.axes.set_ylabel(plot_axes.AXIS_LABELS[vert_axis], labelpad = self.parent.MainParamDict['yLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])

    def CbarTickFormatter(self):
        ''' A helper function that sets the cbar ticks & labels, in the same
        way as the density panel: the colorbar is an image of the gradient
        whose extent is set to the data range.'''
        clim = np.copy(self.cax.get_clim())
        horizontal = self.parent.MainParamDict['HorizontalCbars']
        if self.GetPlotParam('cnorm_type') == 'Log':
            lo, hi = np.log10(clim[0]), np.log10(clim[1])
            formatter = FuncFormatter(lambda val, pos: r'$10^{%g}$' % round(val, 2))
            if horizontal:
                self.cbar.set_extent([lo, hi, 0, 1])
                self.axC.set_xlim(lo, hi)
                self.axC.xaxis.set_major_formatter(formatter)
            else:
                self.cbar.set_extent([0, 1, lo, hi])
                self.axC.set_ylim(lo, hi)
                self.axC.yaxis.set_major_formatter(formatter)
            return

        if self.GetPlotParam('cnorm_type') == 'Pow' or self.GetPlotParam('UseDivCmap'):
            gamma = self.GetPlotParam('cpow_num') if self.GetPlotParam('cnorm_type') == 'Pow' else 1.0
            data_range = np.linspace(clim[0], clim[1], 512)
            cbardata = PowerNormFunc(data_range, vmin = data_range[0], vmax = data_range[-1], gamma = gamma,
                                     midpoint = 0.0, div_cmap = self.GetPlotParam('UseDivCmap'),
                                     stretch_colors = self.GetPlotParam('symmetric'))
            cbardata = np.vstack((cbardata, cbardata))
            self.cbar.set_data(cbardata if horizontal else np.transpose(cbardata)[::-1])

        if horizontal:
            self.cbar.set_extent([clim[0], clim[1], 0, 1])
            self.axC.set_xlim(clim[0], clim[1])
        else:
            self.cbar.set_extent([0, 1, clim[0], clim[1]])
            self.axC.set_ylim(clim[0], clim[1])
            self.axC.locator_params(axis='y', nbins=6)

    ####
    #
    # Plumbing
    #
    ####

    def ChangePlotType(self, str_arg):
        self.FigWrap.ChangeGraph(str_arg)

    def GetPlotParam(self, keyname):
        return self.FigWrap.GetPlotParam(keyname)

    def SetPlotParam(self, keyname, value, update_plot = True, NeedsRedraw = False):
        self.FigWrap.SetPlotParam(keyname, value, update_plot = update_plot, NeedsRedraw = NeedsRedraw)

    def OpenSettings(self):
        if self.settings_window is None:
            self.settings_window = MomentsSettings(self)
        else:
            self.settings_window.destroy()
            self.settings_window = MomentsSettings(self)


class MomentsSettings(Tk.Toplevel):
    '''The settings window, laid out top to bottom in the order one tends to
    decide things: what quantity, which components of it, for which species,
    how it is binned, and how it is drawn.

    The body of the window depends on the quantity, basis and 1D/2D choice,
    so it is rebuilt whenever one of those changes.'''

    def __init__(self, parent):
        self.parent = parent
        Tk.Toplevel.__init__(self)

        self.wm_title('Moments (%d,%d) Settings' % self.parent.FigWrap.pos)
        self.protocol('WM_DELETE_WINDOW', self.OnClosing)
        self.bind('<Return>', self.TxtEnter)

        self.frm = ttk.Frame(self, padding = 6)
        self.frm.pack(fill=Tk.BOTH, expand=True)

        # Create the OptionMenu to chooses the Chart Type:
        head = ttk.Frame(self.frm)
        head.grid(row = 0, column = 0, sticky = Tk.W + Tk.E, pady = (0, 4))
        self.ctypevar = Tk.StringVar(self)
        self.ctypevar.set(self.parent.chartType) # default value
        self.ctypevar.trace('w', self.ctypeChanged)
        ttk.Label(head, text="Chart Type:").pack(side = Tk.LEFT)
        ttk.OptionMenu(head, self.ctypevar, self.parent.chartType, *tuple(self.parent.ChartTypes)).pack(side = Tk.LEFT, padx = (2, 12))

        self.TwoDVar = Tk.IntVar(self)
        self.TwoDVar.set(self.parent.GetPlotParam('twoD'))
        ttk.Checkbutton(head, text = 'Show as 2-D map', variable = self.TwoDVar,
                        command = self.Change2d).pack(side = Tk.LEFT)

        self.body = None
        self.build_body()

        self.status = ttk.Label(self.frm, text = '', foreground = 'gray35', wraplength = 440, justify = Tk.LEFT)
        self.status.grid(row = 2, column = 0, sticky = Tk.W, pady = (4, 0))
        self.show_status()

    def show_status(self):
        self.status.config(text = self.parent.status_text)

    def build_body(self):
        if self.body is not None:
            self.body.destroy()
        self.body = ttk.Frame(self.frm)
        self.body.grid(row = 1, column = 0, sticky = Tk.W + Tk.E)
        self.body.columnconfigure(0, weight = 1)
        two_d = bool(self.parent.GetPlotParam('twoD'))
        self.build_quantity(self.body, 0)
        self.build_components(self.body, 1, two_d)
        self.build_species(self.body, 2, two_d)
        self.build_integration(self.body, 3, two_d)
        self.build_display(self.body, 4, two_d)

    def section(self, master, row, title):
        box = ttk.LabelFrame(master, text = title, padding = 4)
        box.grid(row = row, column = 0, sticky = Tk.W + Tk.E, pady = 2)
        return box

    # What quantity

    def build_quantity(self, master, row):
        box = self.section(master, row, 'Quantity')
        self.FamilyVar = Tk.IntVar(self)
        self.FamilyVar.set(self.parent.GetPlotParam('m_type'))
        # Laid out in the order that reads naturally, keeping the legacy m_type values
        for pos, family in enumerate(('beta', 'u', 'T', 'energy')):
            ttk.Radiobutton(box, text = se.FAMILY_NAMES[family], variable = self.FamilyVar,
                            value = se.FAMILIES.index(family),
                            command = self.FamilyChanged).grid(row = pos // 2, column = pos % 2, sticky = Tk.W, padx = (0, 12))

        basis_row = ttk.Frame(box)
        basis_row.grid(row = 2, column = 0, columnspan = 2, sticky = Tk.W, pady = (4, 0))
        ttk.Label(basis_row, text = 'Basis:').pack(side = Tk.LEFT)
        self.BasisVar = Tk.IntVar(self)
        self.BasisVar.set(self.parent.GetPlotParam('basis'))
        state = ['disabled'] if self.parent.family() == 'energy' else ['!disabled']
        for i, basis in enumerate(se.BASES):
            rb = ttk.Radiobutton(basis_row, text = se.BASIS_NAMES[basis], variable = self.BasisVar,
                                 value = i, command = self.BasisChanged)
            rb.pack(side = Tk.LEFT, padx = (4, 4))
            rb.state(state)

    # Which components

    def build_components(self, master, row, two_d):
        family, basis = self.parent.family(), self.parent.basis()
        title = 'Component (a map shows one)' if two_d else 'Components'
        box = self.section(master, row, title)
        selected = self.parent.selected_components()
        self.CompVars = {}
        self.CompVar = Tk.StringVar(self)
        self.CompVar.set(selected[0] if selected else '')

        def widget(parent_frame, comp, text=None):
            text = se.ui_label(family, comp) if text is None else text
            if two_d:
                return ttk.Radiobutton(parent_frame, text = text, variable = self.CompVar,
                                       value = comp, command = self.ComponentsChanged)
            var = Tk.IntVar(self)
            var.set(comp in selected)
            self.CompVars[comp] = var
            return ttk.Checkbutton(parent_frame, text = text, variable = var,
                                   command = self.ComponentsChanged)

        if family == 'T':
            grid = ttk.Frame(box)
            grid.grid(row = 0, column = 0, sticky = Tk.W)
            idx = ('0',) + se.SPATIAL[basis]
            names = {'0': 't', 'x': 'x', 'y': 'y', 'z': 'z', 'par': '∥', 'perp': '⊥'}
            ttk.Label(grid, text = 'Tᵘᵛ', foreground = 'gray35').grid(row = 0, column = 0, padx = 4)
            for j, b in enumerate(idx):
                ttk.Label(grid, text = 'ν=' + names[b]).grid(row = 0, column = j + 1)
            for i, a in enumerate(idx):
                ttk.Label(grid, text = 'μ=' + names[a]).grid(row = i + 1, column = 0, sticky = Tk.E, padx = 4)
                for j, b in enumerate(idx):
                    if j < i:
                        # T is symmetric, so the lower triangle repeats the upper one
                        ttk.Label(grid, text = '·', foreground = 'gray60').grid(row = i + 1, column = j + 1)
                        continue
                    comp = se.tensor_key(a, b, basis)
                    widget(grid, comp).grid(row = i + 1, column = j + 1, sticky = Tk.W, padx = 2)
            if basis == 'fa':
                ttk.Label(box, text = '∥∥ = P∥,  ⊥⊥ = P⊥ (per perpendicular direction);  '
                                      '0⊥ and ∥⊥ are magnitudes', foreground = 'gray35').grid(row = 1, column = 0, sticky = Tk.W)
            inv = ttk.Frame(box)
            inv.grid(row = 2, column = 0, sticky = Tk.W, pady = (4, 0))
            ttk.Label(inv, text = 'Scalars:').pack(side = Tk.LEFT)
            for comp in se.T_INVARIANTS:
                widget(inv, comp).pack(side = Tk.LEFT, padx = 2)
        else:
            line = ttk.Frame(box)
            line.grid(row = 0, column = 0, sticky = Tk.W)
            for comp in se.components(family, basis):
                widget(line, comp).pack(side = Tk.LEFT, padx = (0, 8))
            if basis == 'fa' and family != 'energy':
                ttk.Label(box, text = '|⊥| is the magnitude of the mean perpendicular vector, '
                                      'e.g. the E×B drift', foreground = 'gray35').grid(row = 1, column = 0, sticky = Tk.W)

        if not two_d:
            quick = ttk.Frame(box)
            quick.grid(row = 3, column = 0, sticky = Tk.W, pady = (4, 0))
            if family == 'T':
                ttk.Button(quick, text = 'Diagonal', command = self.SelectDiagonal).pack(side = Tk.LEFT)
                ttk.Button(quick, text = 'Energy & momentum', command = self.SelectEnergyFlux).pack(side = Tk.LEFT, padx = 4)
            else:
                ttk.Button(quick, text = 'All', command = self.SelectAll).pack(side = Tk.LEFT)
            ttk.Button(quick, text = 'Clear', command = self.SelectNone).pack(side = Tk.LEFT, padx = 4)

    def set_components(self, comps):
        self.parent.SetPlotParam('components', ','.join(comps) if comps else 'none')

    def ComponentsChanged(self):
        if self.parent.GetPlotParam('twoD'):
            comps = [self.CompVar.get()]
        else:
            comps = [c for c, var in self.CompVars.items() if var.get()]
        self.set_components(comps)

    def select(self, comps):
        for c, var in self.CompVars.items():
            var.set(c in comps)
        self.ComponentsChanged()

    def SelectAll(self):
        self.select(list(self.CompVars))

    def SelectNone(self):
        self.select([])

    def SelectDiagonal(self):
        idx = ('0',) + se.SPATIAL[self.parent.basis()]
        self.select([se.tensor_key(a, a, self.parent.basis()) for a in idx])

    def SelectEnergyFlux(self):
        basis = self.parent.basis()
        self.select([se.tensor_key('0', a, basis) for a in ('0',) + se.SPATIAL[basis]])

    # Which species

    def build_species(self, master, row, two_d):
        box = self.section(master, row, 'Species')
        if two_d:
            self.SpeciesVar = Tk.IntVar(self)
            self.SpeciesVar.set(self.parent.shown_species()[0])
            for i, name in enumerate(('Ions', 'Electrons', 'Total')):
                ttk.Radiobutton(box, text = name, variable = self.SpeciesVar, value = i,
                                command = self.Species2DChanged).pack(side = Tk.LEFT, padx = (0, 8))
            return
        self.SpeciesVars = {}
        for key, name in (('show_ions', 'Ions'), ('show_electrons', 'Electrons'), ('show_total', 'Total')):
            var = Tk.IntVar(self)
            var.set(self.parent.GetPlotParam(key))
            self.SpeciesVars[key] = var
            ttk.Checkbutton(box, text = name, variable = var,
                            command = lambda k=key: self.parent.SetPlotParam(k, self.SpeciesVars[k].get())).pack(side = Tk.LEFT, padx = (0, 8))

    def Species2DChanged(self):
        if self.SpeciesVar.get() != self.parent.GetPlotParam('species_2d'):
            self.parent.SetPlotParam('species_2d', self.SpeciesVar.get())

    # How the particles are binned

    def entry(self, master, var, width = 7):
        return ttk.Entry(master, textvariable = var, width = width)

    def build_integration(self, master, row, two_d):
        box = self.section(master, row, 'Binning & normalization')
        family = self.parent.family()

        line = ttk.Frame(box)
        line.grid(row = 0, column = 0, sticky = Tk.W)
        self.xBins = Tk.StringVar(self)
        self.xBins.set(str(self.parent.GetPlotParam('xbins')))
        self.yBins = Tk.StringVar(self)
        self.yBins.set(str(self.parent.GetPlotParam('ybins')))
        if two_d:
            ttk.Label(line, text = '# bins  horizontal').pack(side = Tk.LEFT)
            self.entry(line, self.xBins, 6).pack(side = Tk.LEFT, padx = 2)
            ttk.Label(line, text = 'vertical').pack(side = Tk.LEFT)
            self.entry(line, self.yBins, 6).pack(side = Tk.LEFT, padx = 2)
            self.RotateVar = Tk.IntVar(self)
            self.RotateVar.set(bool(self.parent.GetPlotParam('rotate_90')))
            ttk.Checkbutton(line, text = 'Rotate 90°', variable = self.RotateVar,
                            command = self.RotateChanged).pack(side = Tk.LEFT, padx = (12, 0))
        else:
            ttk.Label(line, text = '# bins').pack(side = Tk.LEFT)
            self.entry(line, self.xBins, 6).pack(side = Tk.LEFT, padx = (2, 12))
            plot_axes.add_axis_buttons(box, self, self.parent, row = 1, column = 0, two_d = False)

        opts = ttk.Frame(box)
        opts.grid(row = 2, column = 0, sticky = Tk.W, pady = (2, 0))
        self.WeightVar = Tk.IntVar(self)
        self.WeightVar.set(self.parent.GetPlotParam('weighted'))
        ttk.Checkbutton(opts, text = 'Weight by charge', variable = self.WeightVar,
                        command = lambda: self.parent.SetPlotParam('weighted', self.WeightVar.get())).pack(side = Tk.LEFT, padx = (0, 8))
        if two_d:
            ttk.Label(opts, text = 'Slab half-width [c/ωp] (0 = whole depth)').pack(side = Tk.LEFT)
            self.SlabVar = Tk.StringVar(self)
            self.SlabVar.set(str(self.parent.GetPlotParam('slab_half_width')))
            self.entry(opts, self.SlabVar, 6).pack(side = Tk.LEFT, padx = 2)
        else:
            self.FilterVPVar = Tk.IntVar(self)
            self.FilterVPVar.set(self.parent.GetPlotParam('filter_by_viewport'))
            ttk.Checkbutton(opts, text = 'Restrict to the 2D viewport', variable = self.FilterVPVar,
                            command = lambda: self.parent.SetPlotParam('filter_by_viewport', self.FilterVPVar.get())).pack(side = Tk.LEFT)

        if family == 'T':
            norm = ttk.Frame(box)
            norm.grid(row = 3, column = 0, sticky = Tk.W, pady = (2, 0))
            ttk.Label(norm, text = 'Tᵘᵛ per:').pack(side = Tk.LEFT)
            self.NormVar = Tk.IntVar(self)
            self.NormVar.set(self.parent.GetPlotParam('normalization'))
            for i, name in enumerate(('unit volume  [n0 m c², n0 = ppc0 per cell]', 'particle  [m c²]')):
                ttk.Radiobutton(norm, text = name, variable = self.NormVar, value = i,
                                command = self.NormChanged).pack(side = Tk.LEFT, padx = 4)
        if family == 'u':
            self.MassVar = Tk.IntVar(self)
            self.MassVar.set(self.parent.GetPlotParam('mass_weight'))
            ttk.Checkbutton(box, text = 'Show momentum p = m u  (in units of the ion mass)', variable = self.MassVar,
                            command = lambda: self.parent.SetPlotParam('mass_weight', self.MassVar.get())).grid(row = 3, column = 0, sticky = Tk.W, pady = (2, 0))

    def RotateChanged(self):
        if bool(self.RotateVar.get()) != bool(self.parent.GetPlotParam('rotate_90')):
            self.parent.SetPlotParam('rotate_90', bool(self.RotateVar.get()), NeedsRedraw = True)

    def NormChanged(self):
        if self.NormVar.get() != self.parent.GetPlotParam('normalization'):
            self.parent.SetPlotParam('normalization', self.NormVar.get())

    # How it is drawn

    def check(self, master, text, key, redraw = False):
        var = Tk.IntVar(self)
        var.set(self.parent.GetPlotParam(key))
        def handler():
            if var.get() != self.parent.GetPlotParam(key):
                if redraw:
                    self.parent.SetPlotParam(key, var.get(), NeedsRedraw = True)
                else:
                    self.parent.SetPlotParam(key, var.get())
        setattr(self, key + '_var', var)
        return ttk.Checkbutton(master, text = text, variable = var, command = handler)

    def build_display(self, master, row, two_d):
        box = self.section(master, row, 'Display')

        lims = ttk.Frame(box)
        lims.grid(row = 1, column = 0, sticky = Tk.W, pady = (2, 0))
        self.Vmin = Tk.StringVar(self)
        self.Vmin.set(str(self.parent.GetPlotParam('v_min')))
        self.Vmax = Tk.StringVar(self)
        self.Vmax.set(str(self.parent.GetPlotParam('v_max')))
        what = 'color' if two_d else 'y'
        self.check(lims, f'Set {what} min', 'set_v_min').pack(side = Tk.LEFT)
        self.entry(lims, self.Vmin).pack(side = Tk.LEFT, padx = (2, 10))
        self.check(lims, f'Set {what} max', 'set_v_max').pack(side = Tk.LEFT)
        self.entry(lims, self.Vmax).pack(side = Tk.LEFT, padx = 2)

        flags = ttk.Frame(box)
        flags.grid(row = 0, column = 0, sticky = Tk.W)
        self.powGamma = Tk.StringVar(self)
        self.powGamma.set(str(self.parent.GetPlotParam('cpow_num')))
        if two_d:
            ttk.Label(flags, text = 'Color norm:').pack(side = Tk.LEFT)
            self.cnormvar = Tk.StringVar(self)
            self.cnormvar.set(self.parent.GetPlotParam('cnorm_type'))
            self.cnormvar.trace('w', self.cnormChanged)
            ttk.OptionMenu(flags, self.cnormvar, self.parent.GetPlotParam('cnorm_type'),
                           'Linear', 'Pow', 'Log').pack(side = Tk.LEFT, padx = 2)
            ttk.Label(flags, text = 'Pow γ').pack(side = Tk.LEFT)
            self.entry(flags, self.powGamma, 5).pack(side = Tk.LEFT, padx = (2, 10))
            self.check(flags, 'Diverging cmap', 'UseDivCmap', redraw = True).pack(side = Tk.LEFT)
            self.check(flags, 'Symmetric about 0', 'symmetric', redraw = True).pack(side = Tk.LEFT, padx = 6)

            more = ttk.Frame(box)
            more.grid(row = 2, column = 0, sticky = Tk.W, pady = (2, 0))
            self.check(more, 'Colorbar', 'show_cbar', redraw = True).pack(side = Tk.LEFT)
            self.check(more, 'Label', 'show_labels', redraw = True).pack(side = Tk.LEFT, padx = 6)
            ttk.Label(more, text = 'Interpolation:').pack(side = Tk.LEFT)
            self.InterpolVar = Tk.StringVar(self)
            self.InterpolVar.set(self.parent.GetPlotParam('interpolation'))
            self.InterpolVar.trace('w', self.InterpolChanged)
            ttk.OptionMenu(more, self.InterpolVar, self.parent.GetPlotParam('interpolation'),
                           *tuple(self.parent.InterpolationMethods)).pack(side = Tk.LEFT, padx = 2)
        else:
            self.check(flags, 'Log y', 'logy').pack(side = Tk.LEFT)
            self.check(flags, 'Symmetric about 0', 'symmetric').pack(side = Tk.LEFT, padx = 6)
            self.check(flags, 'Legend', 'show_legend').pack(side = Tk.LEFT)

    def cnormChanged(self, *args):
        if self.parent.GetPlotParam('cnorm_type') != self.cnormvar.get():
            self.parent.SetPlotParam('cnorm_type', self.cnormvar.get(), NeedsRedraw = True)

    def InterpolChanged(self, *args):
        if self.InterpolVar.get() != self.parent.GetPlotParam('interpolation'):
            self.parent.cax.set_interpolation(self.InterpolVar.get())
            self.parent.SetPlotParam('interpolation', self.InterpolVar.get())

    # Changes that alter the layout of the window

    def FamilyChanged(self):
        m_type = self.FamilyVar.get()
        if m_type == self.parent.GetPlotParam('m_type'):
            return
        self.parent.SetPlotParam('m_type', m_type, update_plot = False)
        family, basis = self.parent.family(), self.parent.basis()
        self.parent.SetPlotParam('components', ','.join(se.default_components(family, basis)), update_plot = False)
        self.build_body()
        self.parent.SetPlotParam('m_type', m_type)

    def BasisChanged(self):
        basis_index = self.BasisVar.get()
        if basis_index == self.parent.GetPlotParam('basis'):
            return
        family = self.parent.family()
        old = self.parent.selected_components()
        self.parent.SetPlotParam('basis', basis_index, update_plot = False)
        basis = self.parent.basis()
        # Keep what makes sense in both bases, e.g. T^00 or the scalars
        allowed = se.components(family, basis)
        kept = [c for c in old if c in allowed] or se.default_components(family, basis)
        self.parent.SetPlotParam('components', ','.join(kept), update_plot = False)
        self.build_body()
        self.parent.SetPlotParam('basis', basis_index)

    def Change2d(self):
        if self.TwoDVar.get() == self.parent.GetPlotParam('twoD'):
            return
        self.parent.SetPlotParam('spatial_y', self.TwoDVar.get(), update_plot = False)
        self.parent.SetPlotParam('twoD', self.TwoDVar.get())
        self.build_body()

    def ctypeChanged(self, *args):
        if self.ctypevar.get() == self.parent.chartType:
            pass
        else:
            self.parent.ChangePlotType(self.ctypevar.get())
            self.destroy()

    # Text entries

    def TxtEnter(self, e):
        self.FieldsCallback()

    def FieldsCallback(self):
        two_d = bool(self.parent.GetPlotParam('twoD'))
        changed = False
        redraw = False
        fields = [(self.Vmin, 'v_min', float), (self.Vmax, 'v_max', float),
                  (self.xBins, 'xbins', int)]
        if two_d:
            fields += [(self.yBins, 'ybins', int), (self.SlabVar, 'slab_half_width', float),
                       (self.powGamma, 'cpow_num', float)]
        for var, key, cast in fields:
            try:
                value = cast(var.get())
                if cast is int and value < 1:
                    raise ValueError
                if key == 'slab_half_width' and value < 0:
                    raise ValueError
            except ValueError:
                #if they type in random stuff, just set it ot the param value
                var.set(str(self.parent.GetPlotParam(key)))
                continue
            if abs(value - self.parent.GetPlotParam(key)) > 1E-6:
                self.parent.SetPlotParam(key, value, update_plot = False)
                if key == 'cpow_num':
                    redraw = self.parent.GetPlotParam('cnorm_type') == 'Pow'
                elif key in ('v_min', 'v_max'):
                    changed |= bool(self.parent.GetPlotParam('set_' + key))
                else:
                    changed = True
        if redraw:
            self.parent.SetPlotParam('cpow_num', self.parent.GetPlotParam('cpow_num'), NeedsRedraw = True)
        elif changed:
            self.parent.SetPlotParam('v_min', self.parent.GetPlotParam('v_min'))

    def OnClosing(self):
        self.parent.settings_window = None
        self.destroy()
