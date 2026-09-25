#!/usr/bin/env pythonw
import tkinter as Tk
from tkinter import ttk
import matplotlib
import numpy as np
import new_cmaps
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import plot_axes
import phase_space

class PhasePanel:
    # A dictionary of all of the parameters for this plot with the default parameters

    plot_param_dict = {'twoD' : 1,
                       'mom_dim': 0,
                       'masked': 1,
                       'cnorm_type': 'Log', #Colormap normalization. Opts are Log or Linear
                       'prtl_type': 0,
                       'cpow_num': 0.6,
                       'show_cbar': True,
                       'weighted': False,
                       'show_shock': False,
                       'show_int_region': True,
                       'xbins' : 200,
                       'pbins' : 200,
                       'v_min': -2.0,
                       'v_max' : 0,
                       'set_v_min': False,
                       'set_v_max': False,
                       'p_min': -2.0,
                       'p_max' : 2,
                       'set_E_min' : False,
                       'E_min': 1.0,
                       'set_E_max': False,
                       'E_max': 200.0,
                       'set_p_min': False,
                       'set_p_max': False,
                       # The limits of the horizontal axis
                       'h_min': 0.0,
                       'h_max': 1.0,
                       'set_h_min': False,
                       'set_h_max': False,
                       'equal_aspect': False,
                       'spatial_x': True,
                       'spatial_y': False,
                       'symmetric': False,
                       'interpolation': 'nearest',
                       'filter_by_viewport': True,
                       'face_color': 'gainsboro',
                       # The quantities on the horizontal and vertical axes,
                       # from phase_space.QUANTITIES. None falls back to the
                       # legacy 'plot_axis' and 'mom_dim' params.
                       'phase_x': None,
                       'phase_y': None}

    # 'plot_axis' is only kept so that views saved before 'phase_x' existed
    # still load; a phase plot has no rotation toggle.
    plot_axes.add_axis_params(plot_param_dict, two_d=False)


    prtl_opts = ['proton_p', 'electron_p']

    gradient =  np.linspace(0, 1, 256)# A way to make the colorbar display better
    gradient = np.vstack((gradient, gradient))

    def __init__(self, parent, figwrapper):

        self.settings_window = None
        self.FigWrap = figwrapper
        self.parent = parent
        self.ChartTypes = self.FigWrap.PlotTypeDict.keys()
        self.chartType = self.FigWrap.chartType
        self.figure = self.FigWrap.figure
        self.InterpolationMethods = ['none','nearest', 'bilinear', 'bicubic', 'spline16',
            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

        # A variable that controls whether the energy integration region
        # is shown
        self.IntRegVar = Tk.IntVar()
        self.IntRegVar.set(self.GetPlotParam('show_int_region'))
        self.IntRegVar.trace('w', self.IntVarHandler)
        # Figure out the energy color the intergration region
        if self.GetPlotParam('prtl_type') == 1: #electons
            self.energy_color = self.parent.electron_color
        else:
            self.energy_color = self.parent.ion_color
        # A list that will hold any lines for the integration region


    def IntVarHandler(self, *args):
        # This should only be called by the user-interaction when all the plots already exist...
        # so we can take some shortcut  s and assume a lot of things are already created.
        self.SetPlotParam('show_int_region', self.IntRegVar.get(), update_plot = False)
        # The spectral integration region is a range in x, drawn as vertical
        # lines, so it only means something when x runs horizontally.
        if self.IntRegVar.get() == True and plot_axes.marker_orientation(self, 'x') == 'v':
            # We need to show the integration region.

            # Look for all the spectra plots and plot the lines.
            for i in range(self.parent.MainParamDict['NumOfRows']):
                for j in range(self.parent.MainParamDict['NumOfCols']):
                    if self.parent.SubPlotList[i][j].chartType == 'SpectraPlot':
                        k = min(self.parent.SubPlotList[i][j].graph.spect_num, len(self.parent.dashes_options)-1)
                        # figure out if we are as ion phase diagram or an electron one
                        if self.GetPlotParam('prtl_type') == 0:
                            # Append the left line to the list
                            self.IntRegionLines.append(self.axes.axvline(
                            max(self.parent.SubPlotList[i][j].graph.i_left_loc, self.xmin+1),
                            linewidth = 1.5, linestyle = '-', color = self.energy_color))
                            # Choose the right dashes pattern
                            self.IntRegionLines[-1].set_dashes(self.parent.dashes_options[k])
                            # Append the left line to the list
                            self.IntRegionLines.append(self.axes.axvline(
                            min(self.parent.SubPlotList[i][j].graph.i_right_loc, self.xmax-1),
                            linewidth = 1.5, linestyle = '-', color = self.energy_color))
                            # Choose the right dashes pattern
                            self.IntRegionLines[-1].set_dashes(self.parent.dashes_options[k])
                        else:
                            # Append the left line to the list
                            self.IntRegionLines.append(self.axes.axvline(
                            max(self.parent.SubPlotList[i][j].graph.e_left_loc, self.xmin+1),
                            linewidth = 1.5, linestyle = '-', color = self.energy_color))
                            # Choose the right dashes pattern
                            self.IntRegionLines[-1].set_dashes(self.parent.dashes_options[k])
                            # Append the left line to the list
                            self.IntRegionLines.append(self.axes.axvline(
                            min(self.parent.SubPlotList[i][j].graph.e_right_loc, self.xmax-1),
                            linewidth = 1.5, linestyle = '-', color = self.energy_color))
                            # Choose the right dashes pattern
                            self.IntRegionLines[-1].set_dashes(self.parent.dashes_options[k])

        # CLOSES IF. NOW IF WE TURN OFF THE INTEGRATION REGIONS, we have to delete all the lines.
        else:
            for i in range(len(self.IntRegionLines)):
                self.IntRegionLines.pop(0).remove()
        # Update the canvas
        self.parent.canvas.draw_idle()

    def phase_axes(self):
        '''The (horizontal, vertical) quantities the user has chosen.'''
        return phase_space.phase_axes(self.GetPlotParam)

    def shown_axes(self):
        '''The (horizontal, vertical) quantities actually plotted, which differ
        from phase_axes only when the data lacks a chosen coordinate.'''
        return getattr(self, 'horiz', None) or self.phase_axes()[0], \
               getattr(self, 'vert', None) or self.phase_axes()[1]

    def spatial_plot_axes(self):
        '''The physical axes of this panel: None for an axis that is not a position.'''
        return tuple(q if phase_space.is_spatial(q) else None for q in self.shown_axes())

    def shares_both_axes(self):
        '''Whether both axes are linked to other panels, in which case
        matplotlib can only keep the aspect by changing the limits.'''
        axes = getattr(self, 'axes', None)
        if axes is None:
            return False
        return len(axes.get_shared_x_axes().get_siblings(axes)) > 1 \
            and len(axes.get_shared_y_axes().get_siblings(axes)) > 1

    def boost(self):
        '''The (Gamma, beta) of the Lorentz boost set in the main window, or None.'''
        if not self.parent.MainParamDict['DoLorentzBoost']:
            return None
        return phase_space.boost_factors(self.parent.MainParamDict['GammaBoost'])

    def ChangePlotType(self, str_arg):
        self.FigWrap.ChangeGraph(str_arg)

    def norm(self, vmin=None, vmax=None):
        if self.GetPlotParam('cnorm_type') == "Log":
            return  mcolors.LogNorm(vmin, vmax)

        else:
            return mcolors.Normalize(vmin, vmax)


    def set_plot_keys(self):
        '''A helper function that will insure that each hdf5 file will only be
        opened once per time step'''
        self.arrs_needed = ['c_omp', 'bx', 'istep', 'me', 'mi']
        prtl_type = self.GetPlotParam('prtl_type')
        horiz, vert = self.phase_axes()

        # Energy cuts and boosts need all three components of the momentum.
        all_components = self.boost() is not None \
            or self.GetPlotParam('set_E_min') or self.GetPlotParam('set_E_max')
        self.arrs_needed += phase_space.momentum_keys_needed(prtl_type, (horiz, vert), all_components)

        if self.GetPlotParam('weighted'):
            self.arrs_needed.append(phase_space.WEIGHT_KEYS[prtl_type])

        # The positions plotted, plus x, which counts the particles and places
        # the shock and the spectral integration region.
        needed_axes = {'x'} | {q for q in (horiz, vert) if phase_space.is_spatial(q)}
        if self.GetPlotParam('filter_by_viewport') and self.parent.is_viewport_zoomed():
            # The region selected in a 2D panel can constrain any coordinate.
            needed_axes |= {'y', 'z'}

        for key in plot_axes.available_position_keys(self, prtl_type, sorted(needed_axes)):
            if key not in self.arrs_needed:
                self.arrs_needed.append(key)

        return self.arrs_needed

    def LoadData(self):
        ''' A helper function that checks if the histogram has
        already been calculated and if it hasn't, it calculates
        it then stores it.'''
        self.viewport = None
        if self.GetPlotParam('filter_by_viewport') and self.parent.is_viewport_zoomed():
            self.viewport = self.parent.get_active_viewport()
            if self.viewport is not None:
                self.parent.last_phase_viewport = self.viewport

        prtl_type = self.GetPlotParam('prtl_type')
        horiz, vert = self.phase_axes()
        boost = self.boost()

        self.key_name = 'phase_' + horiz + '_' + vert + '_'
        self.key_name += str(self.GetPlotParam('pbins')) + 'x' + str(self.GetPlotParam('xbins'))
        if self.GetPlotParam('masked'):
            self.key_name += 'masked_'
        if self.GetPlotParam('weighted'):
            self.key_name += 'weighted_'
        if self.GetPlotParam('set_E_min'):
            self.key_name += 'Emin_'+str(self.GetPlotParam('E_min')) + '_'
        if self.GetPlotParam('set_E_max'):
            self.key_name += 'Emax_'+str(self.GetPlotParam('E_max')) + '_'
        if boost is not None:
            self.key_name += 'boosted_'+ str(self.parent.MainParamDict['GammaBoost'])+'_'
        self.key_name += self.prtl_opts[prtl_type]
        self.key_name += str(int(self.parent.MainParamDict['PrtlStride']))
        self.key_name += plot_axes.viewport_key(self.viewport)
        self.key_name += phase_space.limits_key(self.GetPlotParam)

        self.c_omp = self.FigWrap.LoadKey('c_omp')
        self.istep = self.FigWrap.LoadKey('istep')

        if self.key_name in self.parent.DataDict.keys():
            self.hist2d, (self.horiz, self.vert) = self.parent.DataDict[self.key_name]
            return

        # A stride that leaves a single particle hands back a scalar.
        def load(key):
            return np.atleast_1d(self.FigWrap.LoadKey(key))

        # Every particle has an x, so it fixes how many particles there are,
        # which is how a coordinate missing from the data is recognised.
        n_prtls = len(load(plot_axes.PRTL_POS_KEYS[prtl_type]['x']))

        def positions(axis):
            try:
                coord = load(plot_axes.PRTL_POS_KEYS[prtl_type][axis])
            except KeyError:
                return None
            if coord.ndim != 1 or coord.shape[0] != n_prtls:
                return None
            return coord / self.c_omp

        quantities = phase_space.ParticleQuantities(load, positions, prtl_type, boost)

        # A 1D or 2D run may not hold the chosen coordinate; show x instead.
        self.horiz, self.vert = [q if quantities(q) is not None else 'x' for q in (horiz, vert)]
        h_values = quantities(self.horiz)
        v_values = quantities(self.vert)

        in_range = np.isfinite(h_values) & np.isfinite(v_values)

        if self.GetPlotParam('set_E_min') or self.GetPlotParam('set_E_max'):
            # The energy of each particle in units of m_e c^2
            energy = quantities.lab_gamma()
            if prtl_type == 0:
                energy = energy*self.FigWrap.LoadKey('mi')/self.FigWrap.LoadKey('me')
            if self.GetPlotParam('set_E_min'):
                in_range &= energy >= self.GetPlotParam('E_min')
            if self.GetPlotParam('set_E_max'):
                in_range &= energy <= self.GetPlotParam('E_max')

        # Keep only the particles inside the region picked out in a 2D panel,
        # and bin any position the region constrains over that range only.
        ranges = {}
        for axis, low, high in (self.viewport or ()):
            coord = quantities(axis) if axis is not None else None
            if coord is None:
                continue
            in_range &= (coord >= low) & (coord <= high)
            ranges[axis] = (low, high)

        def axis_range(quantity, values, axis):
            '''The range to bin over, which the limits set in the settings
            window override so that all of the bins land inside the plot.'''
            if phase_space.is_spatial(quantity):
                if quantity in ranges:
                    default = ranges[quantity]
                else:
                    high = plot_axes.domain_extent(self, quantity)
                    default = (0.0, high if high != 0 else 1.0)
            else:
                default = phase_space.data_range(values[in_range])
            return phase_space.limited_range(default, self.GetPlotParam, axis)

        weights = None
        if self.GetPlotParam('weighted'):
            weights = load(phase_space.WEIGHT_KEYS[prtl_type])[in_range]

        self.hist2d = phase_space.histogram(h_values[in_range], v_values[in_range],
                                            axis_range(self.horiz, h_values, 'h'),
                                            axis_range(self.vert, v_values, 'p'),
                                            self.GetPlotParam('xbins'), self.GetPlotParam('pbins'),
                                            weights = weights,
                                            masked = self.GetPlotParam('masked'))
        self.parent.DataDict[self.key_name] = self.hist2d, (self.horiz, self.vert)

    def UpdateLabelsandColors(self):
        # set the colors
        if self.GetPlotParam('prtl_type') == 0: #protons
            self.energy_color = self.parent.ion_color
        else: #electons
            self.energy_color = self.parent.electron_color

        for line in self.IntRegionLines:
            line.set_color(self.energy_color)
        boosted = self.boost() is not None
        horiz, vert = self.shown_axes()
        self.x_label = phase_space.axis_label(horiz, self.GetPlotParam('prtl_type'), boosted)
        self.y_label = phase_space.axis_label(vert, self.GetPlotParam('prtl_type'), boosted)

    def sampling_description(self):
        '''A sentence saying which particles the histogram is made from.'''
        if not self.GetPlotParam('filter_by_viewport'):
            return 'Using every particle in the domain.'
        viewport = getattr(self, 'viewport', None)
        if not viewport:
            return 'Using every particle. Zoom a 2D panel to use only the particles it shows.'
        parts = [f'{axis} ∈ [{low:.4g}, {high:.4g}]' for axis, low, high in viewport if axis is not None]
        return 'Using the particles in the zoomed view: ' + ', '.join(parts)

    def draw(self):
        # In order to speed up the plotting, we only recalculate everything
        # if necessary.
        self.IntRegionLines = []
        # Figure out the color and ylabel
        # Choose the particle type and px, py, or pz
        self.UpdateLabelsandColors()

        self.xmin = self.hist2d[2][0]
        self.xmax = self.hist2d[2][-1]

        self.ymin = self.hist2d[1][0]
        self.ymax = self.hist2d[1][-1]


        if self.GetPlotParam('masked'):
            self.tick_color = 'k'
        else:
            self.tick_color = 'white'


        self.clim = list(self.hist2d[3])

        if self.GetPlotParam('set_v_min'):
            self.clim[0] = 10**self.GetPlotParam('v_min')
        if self.GetPlotParam('set_v_max'):
            self.clim[1] = 10**self.GetPlotParam('v_max')


        self.gs = gridspec.GridSpecFromSubplotSpec(100,100, subplot_spec = self.parent.gs0[self.FigWrap.pos])#, bottom=0.2,left=0.1,right=0.95, top = 0.95)

        # A phase plot only shares the axes that are positions.
        share_x_ax, share_y_ax = self.parent.GetSharedAxes(self.FigWrap.pos)
        self.axes = self.figure.add_subplot(self.gs[self.parent.axes_extent[0]:self.parent.axes_extent[1], self.parent.axes_extent[2]:self.parent.axes_extent[3]],
                                           sharex = share_x_ax, sharey = share_y_ax)

        self.cax = self.axes.imshow(self.hist2d[0],
                                    cmap = new_cmaps.cmaps[self.parent.MainParamDict['ColorMap']],
                                    norm = self.norm(), origin = 'lower',
                                    aspect = 'auto',
                                    interpolation=self.GetPlotParam('interpolation'))

        self.cax.set_extent([self.xmin, self.xmax, self.ymin, self.ymax])

        self.cax.set_clim(self.clim)

        self.shock_line = plot_axes.add_marker_line(self, 'x', self.parent.shock_loc, linewidth = 1.5, linestyle = '--', color = self.parent.shock_color, path_effects=[PathEffects.Stroke(linewidth=2, foreground='k'),
                   PathEffects.Normal()])
        if not (self.GetPlotParam('show_shock') and plot_axes.shows_axis(self, 'x')):
            self.shock_line.set_visible(False)




        self.axC = self.figure.add_subplot(self.gs[self.parent.cbar_extent[0]:self.parent.cbar_extent[1], self.parent.cbar_extent[2]:self.parent.cbar_extent[3]])
        self.parent.cbarList.append(self.axC)

        # Technically I should use the colorbar class here,
        # but I found it annoying in some of it's limitations.
        if self.parent.MainParamDict['HorizontalCbars']:
            self.cbar = self.axC.imshow(self.gradient, aspect='auto',
                                    cmap=new_cmaps.cmaps[self.parent.MainParamDict['ColorMap']])
            # Make the colobar axis more like the real colorbar
            self.axC.tick_params(axis='x',
                                which = 'both', # bothe major and minor ticks
                                top = False, # turn off top ticks
                                labelsize=self.parent.MainParamDict['NumFontSize'])

            self.axC.tick_params(axis='y',          # changes apply to the y-axis
                                which='both',      # both major and minor ticks are affected
                                left=False,      # ticks along the bottom edge are off
                                right=False,         # ticks along the top edge are off
                                labelleft=False)

        else:
            self.cbar = self.axC.imshow(np.transpose(self.gradient)[::-1], aspect='auto',
                                    cmap=new_cmaps.cmaps[self.parent.MainParamDict['ColorMap']])
            # Make the colobar axis more like the real colorbar
            self.axC.tick_params(axis='x',
                                which = 'both', # bothe major and minor ticks
                                top = False, # turn off top ticks
                                bottom = False,
                                labelbottom = False,
                                labelsize=self.parent.MainParamDict['NumFontSize'])

            self.axC.tick_params(axis='y',          # changes apply to the y-axis
                                which='both',      # both major and minor ticks are affected
                                left=False,      # ticks along the bottom edge are off
                                right=True,         # ticks along the top edge are off
                                labelleft=False,
                                labelright=True,
                                labelsize=self.parent.MainParamDict['NumFontSize'])

        self.cbar.set_extent([0, 1.0, 0, 1.0])

        if not self.GetPlotParam('show_cbar'):
            self.axC.set_visible(False)

        if int(matplotlib.__version__[0]) < 2:
            self.axes.set_axis_bgcolor(self.GetPlotParam('face_color'))
        else:
            self.axes.set_facecolor(self.GetPlotParam('face_color'))
        self.axes.tick_params(labelsize = self.parent.MainParamDict['NumFontSize'], color=self.tick_color)
        self.axes.set_xlabel(self.x_label, labelpad = self.parent.MainParamDict['xLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])
        self.axes.set_ylabel(self.y_label, labelpad = self.parent.MainParamDict['yLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])

        self.refresh()

    def refresh(self):
        '''This is a function that will be called only if self.axes already
        holds a density type plot. We only update things that have shown. If
        hasn't changed, or isn't viewed, don't touch it. The difference between this and last
        time, is that we won't actually do any drawing in the plot. The plot
        will be redrawn after all subplots data is changed. '''


        # Main goal, only change what is showing..
        self.xmin = self.hist2d[2][0]
        self.xmax = self.hist2d[2][-1]
        self.ymin = self.hist2d[1][0]
        self.ymax = self.hist2d[1][-1]
        self.clim = list(self.hist2d[3])

        self.cax.set_data(self.hist2d[0])

        self.cax.set_extent([self.xmin,self.xmax, self.ymin, self.ymax])


        if self.GetPlotParam('set_v_min'):
            self.clim[0] =  10**self.GetPlotParam('v_min')
        if self.GetPlotParam('set_v_max'):
            self.clim[1] =  10**self.GetPlotParam('v_max')

        self.cax.set_clim(self.clim)
        if self.GetPlotParam('show_cbar'):
            self.CbarTickFormatter()


        if self.GetPlotParam('show_shock'):
            plot_axes.move_marker_line(self, self.shock_line, 'x', self.parent.shock_loc)

        self.UpdateLabelsandColors()
        self.axes.set_xlabel(self.x_label, labelpad = self.parent.MainParamDict['xLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])
        self.axes.set_ylabel(self.y_label, labelpad = self.parent.MainParamDict['yLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])

        # A position follows the limits set in the main window when the
        # spatial axes are linked. The vertical limits set in this panel's
        # settings win over those.
        horiz, vert = self.spatial_plot_axes()
        linked = self.parent.MainParamDict['LinkSpatial'] == 1
        if linked and vert is not None:
            self.ymin, self.ymax = plot_axes.limits_for_axis(self, vert) or (self.ymin, self.ymax)
        if self.GetPlotParam('set_p_min'):
            self.ymin = self.GetPlotParam('p_min')
        if self.GetPlotParam('set_p_max'):
            self.ymax = self.GetPlotParam('p_max')
        if self.GetPlotParam('symmetric'):
            self.ymin = -max(abs(self.ymin), abs(self.ymax))
            self.ymax = abs(self.ymin)
        self.axes.set_ylim(self.ymin, self.ymax)

        if linked and horiz is not None:
            self.xmin, self.xmax = plot_axes.limits_for_axis(self, horiz) or (self.xmin, self.xmax)
        if self.GetPlotParam('set_h_min'):
            self.xmin = self.GetPlotParam('h_min')
        if self.GetPlotParam('set_h_max'):
            self.xmax = self.GetPlotParam('h_max')
        self.axes.set_xlim(self.xmin, self.xmax)

        # One unit is drawn the same length along both axes, which is what
        # makes e.g. an x-y or a ux-uy plot look like the real thing.
        self.axes.set_aspect('equal' if self.GetPlotParam('equal_aspect') else 'auto',
                             adjustable = 'datalim' if self.shares_both_axes() else 'box')

        if self.settings_window is not None:
            self.settings_window.UpdateSampling()

    def CbarTickFormatter(self):
        ''' A helper function that sets the cbar ticks & labels. This used to be
        easier, but because I am no longer using the colorbar class i have to do
        stuff manually.'''
        clim = np.copy(self.cax.get_clim())
        if self.GetPlotParam('show_cbar'):
            if self.GetPlotParam('cnorm_type') == "Log":
                if self.parent.MainParamDict['HorizontalCbars']:
                    self.cbar.set_extent([np.log10(clim[0]),np.log10(clim[1]),0,1])
                    self.axC.set_xlim(np.log10(clim[0]),np.log10(clim[1]))
                    self.axC.xaxis.set_label_position("top")
                    if self.GetPlotParam('prtl_type') ==0:
                        self.axC.set_xlabel(r'$\log{\ \ f_i(p)}$', size = self.parent.MainParamDict['AxLabelSize'])#, labelpad =15, rotation = -90)
                    else:
                        self.axC.set_xlabel(r'$\log{\ \ f_e(p)}$', size = self.parent.MainParamDict['AxLabelSize'])#, size = 12,labelpad =15, rotation = -90)

                else:
                    self.cbar.set_extent([0,1,np.log10(clim[0]),np.log10(clim[1])])
                    self.axC.set_ylim(np.log10(clim[0]),np.log10(clim[1]))
                    self.axC.locator_params(axis='y',nbins=6)
                    self.axC.yaxis.set_label_position("right")
                    if self.GetPlotParam('prtl_type') ==0:
                        self.axC.set_ylabel(r'$\log{\ \ f_i(p)}$', labelpad = self.parent.MainParamDict['cbarLabelPad'], rotation = -90, size = self.parent.MainParamDict['AxLabelSize'])
                    else:
                        self.axC.set_ylabel(r'$\log{\ \ f_e(p)}$', labelpad = self.parent.MainParamDict['cbarLabelPad'], rotation = -90, size = self.parent.MainParamDict['AxLabelSize'])

            else:# self.GetPlotParam('cnorm_type') == "Linear":
                if self.parent.MainParamDict['HorizontalCbars']:
                    self.cbar.set_extent([clim[0], clim[1], 0, 1])
                    self.axC.set_xlim(clim[0], clim[1])
                    if self.GetPlotParam('prtl_type') ==0:
                        self.axC.set_xlabel(r'$f_i(p)$', size = self.parent.MainParamDict['AxLabelSize'])
                    else:
                        self.axC.set_xlabel(r'$f_e(p)$', size = self.parent.MainParamDict['AxLabelSize'])

                else:
                    self.cbar.set_extent([0, 1, clim[0], clim[1]])
                    self.axC.set_ylim(clim[0], clim[1])
                    self.axC.locator_params(axis='y', nbins=6)
                    self.axC.yaxis.set_label_position("right")
                    if self.GetPlotParam('prtl_type') ==0:
                        self.axC.set_ylabel(r'$f_i(p)$', labelpad = self.parent.MainParamDict['cbarLabelPad'], rotation = -90, size = self.parent.MainParamDict['AxLabelSize'])
                    else:
                        self.axC.set_ylabel(r'$f_e(p)$', labelpad = self.parent.MainParamDict['cbarLabelPad'], rotation = -90, size = self.parent.MainParamDict['AxLabelSize'])



    def GetPlotParam(self, keyname):
        return self.FigWrap.GetPlotParam(keyname)

    def SetPlotParam(self, keyname, value,  update_plot = True, NeedsRedraw = False):
        self.FigWrap.SetPlotParam(keyname, value,  update_plot = update_plot, NeedsRedraw = NeedsRedraw)

    def OpenSettings(self):
        if self.settings_window is None:
            self.settings_window = PhaseSettings(self)
        else:
            self.settings_window.destroy()
            self.settings_window = PhaseSettings(self)


class PhaseSettings(Tk.Toplevel):
    '''The settings window of a phase plot.

    It is laid out in three groups: what to plot (the species and the
    quantity on each axis), which particles go into the histogram, and how
    the histogram is displayed. Typed values are applied on <Return>.'''

    PAD = {'padx': 4, 'pady': 2}

    def __init__(self, parent):
        self.parent = parent
        Tk.Toplevel.__init__(self)

        self.wm_title('Phase Plot (%d,%d) Settings' % self.parent.FigWrap.pos)
        frm = ttk.Frame(self, padding = 6)
        frm.pack(fill=Tk.BOTH, expand=True)
        frm.columnconfigure(0, weight = 1)
        self.protocol('WM_DELETE_WINDOW', self.OnClosing)
        self.bind('<Return>', self.TxtEnter)

        # The chart type
        top = ttk.Frame(frm)
        top.grid(row = 0, column = 0, sticky = Tk.EW, **self.PAD)
        ttk.Label(top, text="Chart type:").pack(side = Tk.LEFT)
        self.ctypevar = Tk.StringVar(self)
        self.ctypevar.set(self.parent.chartType) # default value
        self.ctypevar.trace('w', self.ctypeChanged)
        ttk.OptionMenu(top, self.ctypevar, self.parent.chartType,
                       *tuple(self.parent.ChartTypes)).pack(side = Tk.LEFT, padx = 4)

        self.BuildProjection(frm).grid(row = 1, column = 0, sticky = Tk.EW, **self.PAD)
        self.BuildSelection(frm).grid(row = 2, column = 0, sticky = Tk.EW, **self.PAD)
        self.BuildDisplay(frm).grid(row = 3, column = 0, sticky = Tk.EW, **self.PAD)

        ttk.Label(frm, text = 'Press Enter to apply typed values.',
                  foreground = 'gray40').grid(row = 4, column = 0, sticky = Tk.W, **self.PAD)

        self.UpdateAxisDependentControls()
        self.UpdateSampling()

    ####
    #
    # Layout
    #
    ####

    def BuildProjection(self, frm):
        '''The species, and the quantity and number of bins along each axis.'''
        box = ttk.LabelFrame(frm, text = 'Projection', padding = 6)

        ttk.Label(box, text = 'Species:').grid(row = 0, column = 0, sticky = Tk.W, **self.PAD)
        species = ttk.Frame(box)
        species.grid(row = 0, column = 1, columnspan = 3, sticky = Tk.W)
        self.pvar = Tk.IntVar()
        self.pvar.set(self.parent.GetPlotParam('prtl_type'))
        for i, name in enumerate(['ions', 'electrons']):
            ttk.Radiobutton(species, text = name, variable = self.pvar, value = i,
                            command = self.RadioPrtl).pack(side = Tk.LEFT, padx = (0, 8))

        ttk.Label(box, text = 'Bins').grid(row = 1, column = 2, sticky = Tk.W, **self.PAD)

        choices = [phase_space.DISPLAY_NAMES[q] for q in self.AvailableQuantities()]
        horiz, vert = self.parent.phase_axes()

        self.HorizVar = Tk.StringVar(self, phase_space.DISPLAY_NAMES[horiz])
        self.VertVar = Tk.StringVar(self, phase_space.DISPLAY_NAMES[vert])
        self.xBins = Tk.StringVar(self, str(self.parent.GetPlotParam('xbins')))
        self.pBins = Tk.StringVar(self, str(self.parent.GetPlotParam('pbins')))

        for row, (label, var, bins) in enumerate([('Horizontal axis:', self.HorizVar, self.xBins),
                                                  ('Vertical axis:', self.VertVar, self.pBins)], start = 2):
            ttk.Label(box, text = label).grid(row = row, column = 0, sticky = Tk.W, **self.PAD)
            chooser = ttk.Combobox(box, textvariable = var, values = choices,
                                   state = 'readonly', width = 12)
            chooser.grid(row = row, column = 1, sticky = Tk.W, **self.PAD)
            chooser.bind('<<ComboboxSelected>>', self.AxesChanged)
            ttk.Entry(box, textvariable = bins, width = 6).grid(row = row, column = 2, sticky = Tk.W, **self.PAD)

        ttk.Button(box, text = 'Swap axes ⇅', command = self.SwapAxes).grid(
            row = 2, column = 3, rowspan = 2, sticky = Tk.NS, **self.PAD)
        return box

    def BuildSelection(self, frm):
        '''Which particles go into the histogram.'''
        box = ttk.LabelFrame(frm, text = 'Particles', padding = 6)

        self.FilterVPVar = Tk.IntVar()
        self.FilterVPVar.set(self.parent.GetPlotParam('filter_by_viewport'))
        ttk.Checkbutton(box, text = 'Only the particles in the zoomed 2D view',
                        variable = self.FilterVPVar,
                        command = self.FilterVPHandler).grid(row = 0, column = 0, columnspan = 4, sticky = Tk.W, **self.PAD)
        self.SamplingLabel = ttk.Label(box, foreground = 'gray30', wraplength = 380, justify = Tk.LEFT)
        self.SamplingLabel.grid(row = 1, column = 0, columnspan = 4, sticky = Tk.W, padx = (24, 4))

        self.WeightVar = Tk.IntVar()
        self.WeightVar.set(self.parent.GetPlotParam('weighted'))
        ttk.Checkbutton(box, text = 'Weight by charge', variable = self.WeightVar,
                        command = lambda: self.parent.SetPlotParam('weighted', self.WeightVar.get())
                        ).grid(row = 2, column = 0, columnspan = 4, sticky = Tk.W, **self.PAD)

        self.setEminVar, self.Emin = self.LimitRow(box, 3, 0, 'Min energy (m_e c²)', 'set_E_min', 'E_min')
        self.setEmaxVar, self.Emax = self.LimitRow(box, 3, 2, 'Max energy (m_e c²)', 'set_E_max', 'E_max')
        return box

    def BuildDisplay(self, frm):
        '''How the histogram is shown.'''
        box = ttk.LabelFrame(frm, text = 'Display', padding = 6)

        self.setHminVar, self.Hmin = self.LimitRow(box, 0, 0, 'Horizontal min', 'set_h_min', 'h_min')
        self.setHmaxVar, self.Hmax = self.LimitRow(box, 0, 2, 'Horizontal max', 'set_h_max', 'h_max')
        self.setPminVar, self.Pmin = self.LimitRow(box, 1, 0, 'Vertical min', 'set_p_min', 'p_min')
        self.setPmaxVar, self.Pmax = self.LimitRow(box, 1, 2, 'Vertical max', 'set_p_max', 'p_max')

        self.SymVar = Tk.IntVar()
        self.SymVar.set(self.parent.GetPlotParam('symmetric'))
        ttk.Checkbutton(box, text = 'Vertical symmetric about zero', variable = self.SymVar,
                        command = self.SymmetricHandler).grid(row = 2, column = 0, columnspan = 2, sticky = Tk.W, **self.PAD)

        self.AspectVar = Tk.IntVar()
        self.AspectVar.set(self.parent.GetPlotParam('equal_aspect'))
        ttk.Checkbutton(box, text = 'Equal aspect ratio', variable = self.AspectVar,
                        command = lambda: self.SetToggle('equal_aspect', self.AspectVar)
                        ).grid(row = 2, column = 2, columnspan = 2, sticky = Tk.W, **self.PAD)

        self.setVminVar, self.Vmin = self.LimitRow(box, 3, 0, 'log f min', 'set_v_min', 'v_min')
        self.setVmaxVar, self.Vmax = self.LimitRow(box, 3, 2, 'log f max', 'set_v_max', 'v_max')

        self.MaskVar = Tk.IntVar()
        self.MaskVar.set(self.parent.GetPlotParam('masked'))
        ttk.Checkbutton(box, text = 'Mask empty bins', variable = self.MaskVar,
                        command = lambda: self.parent.SetPlotParam('masked', self.MaskVar.get())
                        ).grid(row = 4, column = 0, columnspan = 2, sticky = Tk.W, **self.PAD)

        self.CbarVar = Tk.IntVar()
        self.CbarVar.set(self.parent.GetPlotParam('show_cbar'))
        ttk.Checkbutton(box, text = 'Show color bar', variable = self.CbarVar,
                        command = self.CbarHandler).grid(row = 4, column = 2, columnspan = 2, sticky = Tk.W, **self.PAD)

        interp = ttk.Frame(box)
        interp.grid(row = 5, column = 0, columnspan = 4, sticky = Tk.W, **self.PAD)
        ttk.Label(interp, text = 'Interpolation:').pack(side = Tk.LEFT)
        self.InterpolVar = Tk.StringVar(self)
        self.InterpolVar.set(self.parent.GetPlotParam('interpolation')) # default value
        self.InterpolVar.trace('w', self.InterpolChanged)
        ttk.OptionMenu(interp, self.InterpolVar, self.parent.GetPlotParam('interpolation'),
                       *tuple(self.parent.InterpolationMethods)).pack(side = Tk.LEFT, padx = 4)

        # These mark positions in x, so they only apply when x is plotted.
        self.ShockVar = Tk.IntVar()
        self.ShockVar.set(self.parent.GetPlotParam('show_shock'))
        self.ShockButton = ttk.Checkbutton(box, text = 'Show shock', variable = self.ShockVar,
                                           command = self.ShockVarHandler)
        self.ShockButton.grid(row = 6, column = 0, columnspan = 2, sticky = Tk.W, **self.PAD)

        self.IntRegButton = ttk.Checkbutton(box, text = 'Show spectra energy region',
                                            variable = self.parent.IntRegVar)
        self.IntRegButton.grid(row = 6, column = 2, columnspan = 2, sticky = Tk.W, **self.PAD)
        return box

    def LimitRow(self, box, row, column, text, set_param, value_param):
        '''A checkbox that turns a limit on, next to the entry that sets it.'''
        set_var = Tk.IntVar()
        set_var.set(self.parent.GetPlotParam(set_param))
        set_var.trace('w', lambda *args: self.SetToggle(set_param, set_var))
        value_var = Tk.StringVar(self, str(self.parent.GetPlotParam(value_param)))
        ttk.Checkbutton(box, text = text, variable = set_var).grid(row = row, column = column, sticky = Tk.W, **self.PAD)
        ttk.Entry(box, textvariable = value_var, width = 7).grid(row = row, column = column + 1, sticky = Tk.W, **self.PAD)
        return set_var, value_var

    ####
    #
    # Projection
    #
    ####

    def AvailableQuantities(self):
        '''The quantities that can be plotted: every momentum quantity, and the
        positions the particle data holds.'''
        prtl_type = self.parent.GetPlotParam('prtl_type')
        keys = plot_axes.available_position_keys(self.parent, prtl_type, phase_space.SPATIAL)
        present = {axis for axis in phase_space.SPATIAL
                   if plot_axes.PRTL_POS_KEYS[prtl_type][axis] in keys}
        if not present:
            # The file could not be checked, so offer them all.
            present = set(phase_space.SPATIAL)
        # Always offer whatever is already chosen.
        present |= {q for q in self.parent.phase_axes() if phase_space.is_spatial(q)}
        return [q for q in phase_space.QUANTITIES if q in present or not phase_space.is_spatial(q)]

    def SelectedAxes(self):
        by_name = {name: q for q, name in phase_space.DISPLAY_NAMES.items()}
        return by_name[self.HorizVar.get()], by_name[self.VertVar.get()]

    def AxesChanged(self, *args):
        horiz, vert = self.SelectedAxes()
        if (horiz, vert) == self.parent.phase_axes():
            return
        # A zoom into the old quantities means nothing for the new ones, so
        # the panel starts again from its full view.
        try:
            i, j = self.parent.FigWrap.pos
            self.parent.parent.prev_ctype_list[i][j] = None
        except (AttributeError, IndexError):
            pass
        self.CarryLimits(self.parent.phase_axes(), (horiz, vert))
        # Both are set so a view saved from here no longer needs the legacy params.
        self.parent.SetPlotParam('phase_x', horiz, update_plot = False)
        self.parent.SetPlotParam('phase_y', vert, update_plot = False)
        # Only an x axis follows the shock when the x limits are shock-relative.
        self.parent.SetPlotParam('spatial_x', horiz == 'x', update_plot = False)
        self.parent.SetPlotParam('spatial_y', phase_space.is_spatial(vert), update_plot = False)
        self.UpdateAxisDependentControls(horiz)
        self.parent.SetPlotParam('phase_y', vert, NeedsRedraw = True)

    def CarryLimits(self, old, new):
        '''Keep the limits of an axis with its quantity: a swap swaps them, and
        an axis given a new quantity has its limits turned off.'''
        rows = {'h': (self.setHminVar, self.Hmin, self.setHmaxVar, self.Hmax),
                'p': (self.setPminVar, self.Pmin, self.setPmaxVar, self.Pmax)}
        names = ('set_{}_min', '{}_min', 'set_{}_max', '{}_max')
        saved = {axis: [self.parent.GetPlotParam(n.format(axis)) for n in names] for axis in 'hp'}
        if new == old[::-1] and new != old:
            target = {'h': saved['p'], 'p': saved['h']}
        else:
            target = {}
            for axis, before, after in zip('hp', old, new):
                values = list(saved[axis])
                if before != after:
                    values[0] = values[2] = False
                target[axis] = values
        for axis, values in target.items():
            for name, value, var in zip(names, values, rows[axis]):
                self.parent.SetPlotParam(name.format(axis), value, update_plot = False)
                var.set(value if name.startswith('set') else str(value))

    def SwapAxes(self):
        horiz, vert = self.HorizVar.get(), self.VertVar.get()
        self.HorizVar.set(vert)
        self.VertVar.set(horiz)
        self.AxesChanged()

    def UpdateAxisDependentControls(self, horiz = None):
        '''Grey out the markers of x positions unless x runs horizontally.'''
        if horiz is None:
            horiz = self.parent.phase_axes()[0]
        state = ['!disabled'] if horiz == 'x' else ['disabled']
        self.ShockButton.state(state)
        self.IntRegButton.state(state)

    def UpdateSampling(self):
        '''Say which particles the histogram was made from.'''
        try:
            self.SamplingLabel.config(text = self.parent.sampling_description())
        except Tk.TclError:
            # The window is already gone.
            pass

    ####
    #
    # Handlers
    #
    ####

    def FilterVPHandler(self):
        self.parent.SetPlotParam('filter_by_viewport', self.FilterVPVar.get())
        self.UpdateSampling()

    def ShockVarHandler(self, *args):
        if self.parent.GetPlotParam('show_shock')== self.ShockVar.get():
            pass
        else:
            self.parent.shock_line.set_visible(self.ShockVar.get() and plot_axes.shows_axis(self.parent, 'x'))
            self.parent.SetPlotParam('show_shock', self.ShockVar.get())


    def CbarHandler(self, *args):
        if self.parent.GetPlotParam('show_cbar')== self.CbarVar.get():
            pass
        else:
            self.parent.axC.set_visible(self.CbarVar.get())
            self.parent.SetPlotParam('show_cbar', self.CbarVar.get(), update_plot =self.parent.GetPlotParam('twoD'))


    def ctypeChanged(self, *args):
        if self.ctypevar.get() == self.parent.chartType:
            pass
        else:
            self.parent.ChangePlotType(self.ctypevar.get())
            self.destroy()

    def InterpolChanged(self, *args):
        if self.InterpolVar.get() == self.parent.GetPlotParam('interpolation'):
            pass
        else:
            self.parent.cax.set_interpolation(self.InterpolVar.get())
            self.parent.SetPlotParam('interpolation', self.InterpolVar.get())

    def RadioPrtl(self):
        if self.pvar.get() == self.parent.GetPlotParam('prtl_type'):
            pass
        else:
            self.parent.SetPlotParam('prtl_type', self.pvar.get(), update_plot =  False)
            self.parent.UpdateLabelsandColors()
            self.parent.axes.set_ylabel(self.parent.y_label, labelpad = self.parent.parent.MainParamDict['yLabelPad'], color = 'black', size = self.parent.parent.MainParamDict['AxLabelSize'])
            self.parent.SetPlotParam('prtl_type', self.pvar.get())

    def SymmetricHandler(self, *args):
        if self.parent.GetPlotParam('symmetric') == self.SymVar.get():
            pass
        else:
            self.parent.SetPlotParam('symmetric', self.SymVar.get(), update_plot = True)

    def SetToggle(self, param, var):
        if var.get() != self.parent.GetPlotParam(param):
            self.parent.SetPlotParam(param, var.get())

    def TxtEnter(self, e):
        self.FieldsCallback()

    def FieldsCallback(self):
        #### First set the Float Values
        tkvarLimList = [self.Vmin, self.Vmax, self.Hmin, self.Hmax, self.Pmin, self.Pmax, self.Emin, self.Emax]
        plot_param_List = ['v_min', 'v_max', 'h_min', 'h_max', 'p_min', 'p_max', 'E_min', 'E_max']
        tkvarSetList = [self.setVminVar, self.setVmaxVar, self.setHminVar, self.setHmaxVar, self.setPminVar, self.setPmaxVar, self.setEminVar, self.setEmaxVar]
        to_reload = False
        for j in range(len(tkvarLimList)):
            try:
            #make sure the user types in a float
                if np.abs(float(tkvarLimList[j].get()) - self.parent.GetPlotParam(plot_param_List[j])) > 1E-4:
                    self.parent.SetPlotParam(plot_param_List[j], float(tkvarLimList[j].get()), update_plot = False)
                    to_reload += True*tkvarSetList[j].get()

            except ValueError:
                #if they type in random stuff, just set it ot the param value
                tkvarLimList[j].set(str(self.parent.GetPlotParam(plot_param_List[j])))

        intVarList = [self.pBins, self.xBins]
        intParamList = ['pbins', 'xbins']
        for j in range(len(intVarList)):
            try:
            #make sure the user types in a positive integer
                n_bins = int(float(intVarList[j].get()))
                if n_bins < 1:
                    raise ValueError
                intVarList[j].set(str(n_bins))
                if n_bins != int(self.parent.GetPlotParam(intParamList[j])):
                    self.parent.SetPlotParam(intParamList[j], n_bins, update_plot = False)
                    to_reload += True

            except ValueError:
                #if they type in random stuff, just set it ot the param value
                intVarList[j].set(str(self.parent.GetPlotParam(intParamList[j])))


        if to_reload:
            self.parent.SetPlotParam('v_min', self.parent.GetPlotParam('v_min'))

    def OnClosing(self):
        self.parent.settings_window = None
        self.destroy()
