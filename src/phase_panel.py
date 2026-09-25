#!/usr/bin/env pythonw
import matplotlib, sys
sys.path.append('../')

import numpy as np
import new_cmaps
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import phase_space

from plot_axes import SLICE_PLANE_AXES

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
                       'h_min': 0.0,
                       'h_max': 1.0,
                       'set_h_min': False,
                       'set_h_max': False,
                       'equal_aspect': False,
                       'spatial_x': True,
                       'symmetric': False,
                       'spatial_y': False,
                       'interpolation': 'nearest',
                       'filter_by_viewport': True,
                       'face_color': 'gainsboro',
                       'plot_axis': 0,
                       # The quantities on the horizontal and vertical axes,
                       # see phase_space.phase_axes.
                       'phase_x': None,
                       'phase_y': None}


    gradient =  np.linspace(0, 1, 256)# A way to make the colorbar display better
    gradient = np.vstack((gradient, gradient))
    def __init__(self, parent, pos, param_dict):
        self.param_dict = {}
        for key, val in self.plot_param_dict.items():
            self.param_dict[key] = val
        for key, val in param_dict.items():
            self.param_dict[key] = val
        self.pos = pos
        self.parent = parent
        self.chartType = 'PhasePlot'
        self.figure = self.parent.figure
        self.InterpolationMethods = ['none','nearest', 'bilinear', 'bicubic', 'spline16',
            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']


        if self.GetPlotParam('prtl_type') == 1: #electons
            self.energy_color = self.parent.electron_color
        else:
            self.energy_color = self.parent.ion_color
        # A list that will hold any lines for the integration region


    def norm(self, vmin=None, vmax=None):
        if self.GetPlotParam('cnorm_type') == "Log":
            return  mcolors.LogNorm(vmin, vmax)

        else:
            return mcolors.Normalize(vmin, vmax)


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

    def shown_axes(self):
        '''The (horizontal, vertical) quantities plotted.'''
        return getattr(self, 'horiz', None) or phase_space.phase_axes(self.GetPlotParam)[0], \
               getattr(self, 'vert', None) or phase_space.phase_axes(self.GetPlotParam)[1]

    def boost(self):
        if not self.parent.MainParamDict['DoLorentzBoost']:
            return None
        return phase_space.boost_factors(self.parent.MainParamDict['GammaBoost'])

    def get_active_viewport_headless(self, output):
        if not hasattr(self.parent, 'SubPlotList') or self.parent.SubPlotList is None:
            return None
        for i in range(self.parent.MainParamDict['NumOfRows']):
            if i >= len(self.parent.SubPlotList):
                continue
            for j in range(self.parent.MainParamDict['NumOfCols']):
                if j >= len(self.parent.SubPlotList[i]):
                    continue
                subplot = self.parent.SubPlotList[i][j]
                chartType = getattr(subplot, 'chartType', '')
                if chartType in ['FieldsPlot', 'DensityPlot', 'MagPlots', 'Moments']:
                    twoD = False
                    if hasattr(subplot, 'GetPlotParam'):
                        twoD = subplot.GetPlotParam('twoD')
                    elif hasattr(subplot, 'param_dict'):
                        twoD = subplot.param_dict.get('twoD', False)
                    if twoD:
                        if hasattr(subplot, 'axes') and subplot.axes is not None:
                            xlim = subplot.axes.get_xlim()
                            ylim = subplot.axes.get_ylim()
                        else:
                            c_omp = getattr(output, 'c_omp', 1.0)
                            istep = getattr(output, 'istep', 1.0)
                            xlim_min = 0.0
                            xlim_max = getattr(output, 'bx').shape[2] / c_omp * istep
                            if self.parent.MainParamDict['SetxLim']:
                                if self.parent.MainParamDict['xLimsRelative']:
                                    shock_loc = getattr(self.parent, 'shock_loc', 0.0)
                                    xlim_min = self.parent.MainParamDict['xLeft'] + shock_loc
                                    xlim_max = self.parent.MainParamDict['xRight'] + shock_loc
                                else:
                                    xlim_min = self.parent.MainParamDict['xLeft']
                                    xlim_max = self.parent.MainParamDict['xRight']
                            
                            ylim_min = 0.0
                            plane = self.parent.MainParamDict['2DSlicePlane']
                            if plane == 0:
                                shape_val = getattr(output, 'bx').shape[1]
                            elif plane == 1:
                                shape_val = getattr(output, 'bx').shape[0]
                            else:
                                shape_val = getattr(output, 'bx').shape[0]
                            ylim_max = shape_val / c_omp * istep
                            
                            if self.parent.MainParamDict['SetyLim']:
                                ylim_min = self.parent.MainParamDict['yBottom']
                                ylim_max = self.parent.MainParamDict['yTop']
                            
                            xlim = (xlim_min, xlim_max)
                            ylim = (ylim_min, ylim_max)
                            
                        plane = self.parent.MainParamDict['2DSlicePlane']
                        xlim_min, xlim_max = min(xlim), max(xlim)
                        ylim_min, ylim_max = min(ylim), max(ylim)
                        return (xlim_min, xlim_max, ylim_min, ylim_max, plane)
        return None

    def update_data(self, output):
        self.viewport = None
        if self.GetPlotParam('filter_by_viewport'):
            self.viewport = self.get_active_viewport_headless(output)
            if self.viewport is not None:
                xlim_0, xlim_1, ylim_0, ylim_1, plane = self.viewport
                xlim_min, xlim_max = min(xlim_0, xlim_1), max(xlim_0, xlim_1)
                ylim_min, ylim_max = min(ylim_0, ylim_1), max(ylim_0, ylim_1)
                bx = getattr(output, 'bx', None)
                if bx is not None:
                    c_omp = getattr(output, 'c_omp', 1.0)
                    istep = getattr(output, 'istep', 1.0)
                    full_x_span = bx.shape[-1] / c_omp * istep
                    if plane == 0:
                        full_y_span = bx.shape[1] / c_omp * istep
                    else:
                        full_y_span = bx.shape[0] / c_omp * istep
                    
                    if (xlim_max - xlim_min) >= 0.98 * full_x_span and (ylim_max - ylim_min) >= 0.98 * full_y_span:
                        self.viewport = None

        prtl_type = self.GetPlotParam('prtl_type')
        self.c_omp = getattr(output, 'c_omp')
        self.istep = getattr(output, 'istep')
        n_prtls = len(getattr(output, 'xi' if prtl_type == 0 else 'xe'))

        def positions(axis):
            key = {0: {'x': 'xi', 'y': 'yi', 'z': 'zi'},
                   1: {'x': 'xe', 'y': 'ye', 'z': 'ze'}}[prtl_type][axis]
            try:
                coord = np.asanyarray(getattr(output, key))
            except Exception:
                return None
            if coord.ndim != 1 or coord.shape[0] != n_prtls:
                return None
            return coord / self.c_omp

        quantities = phase_space.ParticleQuantities(lambda key: getattr(output, key),
                                                    positions, prtl_type, self.boost())
        horiz, vert = phase_space.phase_axes(self.GetPlotParam)
        # A 1D or 2D run may not hold the chosen coordinate; show x instead.
        self.horiz, self.vert = [q if quantities(q) is not None else 'x' for q in (horiz, vert)]
        h_values = quantities(self.horiz)
        v_values = quantities(self.vert)

        in_range = np.isfinite(h_values) & np.isfinite(v_values)

        if self.GetPlotParam('set_E_min') or self.GetPlotParam('set_E_max'):
            # The energy of each particle in units of m_e c^2
            energy = quantities.lab_gamma()
            if prtl_type == 0:
                energy = energy*getattr(output, 'mi')/getattr(output, 'me')
            if self.GetPlotParam('set_E_min'):
                in_range &= energy >= self.GetPlotParam('E_min')
            if self.GetPlotParam('set_E_max'):
                in_range &= energy <= self.GetPlotParam('E_max')

        # Keep only the particles inside the viewport of the 2D panel.
        ranges = {}
        if self.viewport is not None:
            xlim_0, xlim_1, ylim_0, ylim_1, plane = self.viewport
            h_axis, v_axis = SLICE_PLANE_AXES[plane]
            for axis, low, high in ((h_axis, min(xlim_0, xlim_1), max(xlim_0, xlim_1)),
                                    (v_axis, min(ylim_0, ylim_1), max(ylim_0, ylim_1))):
                coord = quantities(axis)
                if coord is None:
                    continue
                in_range &= (coord >= low) & (coord <= high)
                ranges[axis] = (low, high)

        field_shape = getattr(output, 'bx').shape

        def axis_range(quantity, values, axis):
            if phase_space.is_spatial(quantity):
                if quantity in ranges:
                    default = ranges[quantity]
                else:
                    n_cells = field_shape[{'z': 0, 'y': 1, 'x': 2}[quantity]]
                    high = n_cells/self.c_omp*self.istep
                    default = (0.0, high if high != 0 else 1.0)
            else:
                default = phase_space.data_range(values[in_range])
            return phase_space.limited_range(default, self.GetPlotParam, axis)

        weights = None
        if self.GetPlotParam('weighted'):
            weights = getattr(output, phase_space.WEIGHT_KEYS[prtl_type])[in_range]

        self.hist2d = phase_space.histogram(h_values[in_range], v_values[in_range],
                                            axis_range(self.horiz, h_values, 'h'),
                                            axis_range(self.vert, v_values, 'p'),
                                            self.GetPlotParam('xbins'), self.GetPlotParam('pbins'),
                                            weights = weights,
                                            masked = self.GetPlotParam('masked'))

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


        self.gs = gridspec.GridSpecFromSubplotSpec(100,100, subplot_spec = self.parent.gs0[self.pos])#, bottom=0.2,left=0.1,right=0.95, top = 0.95)

        self.axes = self.figure.add_subplot(self.gs[self.parent.axes_extent[0]:self.parent.axes_extent[1], self.parent.axes_extent[2]:self.parent.axes_extent[3]])

        self.cax = self.axes.imshow(self.hist2d[0],
                                    cmap = new_cmaps.cmaps[self.parent.MainParamDict['ColorMap']],
                                    norm = self.norm(), origin = 'lower',
                                    aspect = 'auto',
                                    interpolation=self.GetPlotParam('interpolation'))

        self.cax.set_extent([self.xmin, self.xmax, self.ymin, self.ymax])

        self.cax.set_clim(self.clim)

        self.shock_line = self.axes.axvline(self.parent.shock_loc, linewidth = 1.5, linestyle = '--', color = self.parent.shock_color, path_effects=[PathEffects.Stroke(linewidth=2, foreground='k'),
                   PathEffects.Normal()])
        if not (self.GetPlotParam('show_shock') and self.shown_axes()[0] == 'x'):
            self.shock_line.set_visible(False)




        self.axC = self.figure.add_subplot(self.gs[self.parent.cbar_extent[0]:self.parent.cbar_extent[1], self.parent.cbar_extent[2]:self.parent.cbar_extent[3]])


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
            self.cbar = self.axC.imshow(np.transpose(self.gradient)[::-1], aspect='auto', origin='upper',
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
            self.shock_line.set_xdata([self.parent.shock_loc,self.parent.shock_loc])

        self.UpdateLabelsandColors()
        self.axes.set_xlabel(self.x_label, labelpad = self.parent.MainParamDict['xLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])
        self.axes.set_ylabel(self.y_label, labelpad = self.parent.MainParamDict['yLabelPad'], color = 'black', size = self.parent.MainParamDict['AxLabelSize'])

        if self.GetPlotParam('set_p_min'):
            self.ymin = self.GetPlotParam('p_min')
        if self.GetPlotParam('set_p_max'):
            self.ymax = self.GetPlotParam('p_max')
        if self.GetPlotParam('symmetric'):
            self.ymin = -max(abs(self.ymin), abs(self.ymax))
            self.ymax = abs(self.ymin)

        self.axes.set_ylim(self.ymin, self.ymax)

        if self.shown_axes()[0] == 'x' and self.parent.MainParamDict['SetxLim'] and self.parent.MainParamDict['LinkSpatial'] == 1:
            if self.parent.MainParamDict['xLimsRelative']:
                self.axes.set_xlim(self.parent.MainParamDict['xLeft'] + self.parent.shock_loc,
                                   self.parent.MainParamDict['xRight'] + self.parent.shock_loc)
            else:
                self.axes.set_xlim(self.parent.MainParamDict['xLeft'], self.parent.MainParamDict['xRight'])

        else:
            self.axes.set_xlim(self.xmin,self.xmax)
        if self.GetPlotParam('set_h_min'):
            self.axes.set_xlim(left = self.GetPlotParam('h_min'))
        if self.GetPlotParam('set_h_max'):
            self.axes.set_xlim(right = self.GetPlotParam('h_max'))

        self.axes.set_aspect('equal' if self.GetPlotParam('equal_aspect') else 'auto')

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
        return self.param_dict[keyname]
