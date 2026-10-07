#! /usr/bin/env python
import re # regular expressions
import os, sys # Used to make the code portable

# Iseult is mostly used over VNC. There, OpenGL integration only slows Qt's
# start up, and a fractional scale factor would have matplotlib render (and
# VNC ship) more pixels than the screen shows.
os.environ.setdefault('QT_XCB_GL_INTEGRATION', 'none')
os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '0')

import data_loading # Allows us the read the data files
import time, string, io
import traceback # so one broken panel can be reported instead of crashing Iseult
from PIL import Image
import matplotlib
matplotlib.use('QtAgg')
import new_cmaps
import numpy as np
from collections import deque
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from phase_plots import PhasePanel
from fields_plots import FieldsPanel
from density_plots import DensPanel
from spectra import SpectralPanel
from mag_plots import BPanel
from energy_plots import EnergyPanel
from fft_plots import FFTPanel
from total_energy_plots import TotEnergyPanel
from moments import MomentsPanel
from balance_panels import OhmsLawPanel, PressureBalancePanel
import plot_axes
import phase_space
from view_state import view_limits, view_from_limits
import movie_writer
import preset_views
from functools import partial
import subprocess, yaml
import pathlib

# The settings panes are written against Tk's API, which qt_compat provides on Qt.
import qt_compat as Tk
from qt_compat import ttk, filedialog, messagebox, simpledialog

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['image.resample'] = False
matplotlib.rcParams['image.origin'] = 'upper'
# Iseult never titles its axes. Giving titles a fixed height stops matplotlib
# measuring every panel on every draw to find room for them.
matplotlib.rcParams['axes.titley'] = 1.0

import argparse


class IseultCanvas(FigureCanvasQTAgg):
    '''The figure's canvas. Redrawing the whole figure is slow, so while the
    window is being resized only the last size it settles on is drawn.'''

    RESIZE_DELAY_MS = 150

    def __init__(self, figure):
        FigureCanvasQTAgg.__init__(self, figure)
        # Clicking the figure takes focus from any entry being typed in, as in Tk
        self.setFocusPolicy(Qt.ClickFocus)
        self._pending_resize = None
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._apply_resize)
        self._sized = False

    def resizeEvent(self, event):
        if not self._sized:
            # The first size is the one everything is laid out with; use it now.
            self._sized = True
            FigureCanvasQTAgg.resizeEvent(self, event)
            return
        self._pending_resize = QtGui.QResizeEvent(event.size(), event.oldSize())
        self._resize_timer.start(self.RESIZE_DELAY_MS)

    def _apply_resize(self):
        event, self._pending_resize = self._pending_resize, None
        if event is not None:
            FigureCanvasQTAgg.resizeEvent(self, event)


class MyCustomToolbar(NavigationToolbar2QT):
    '''matplotlib's navigation toolbar, which also has Iseult check whether a
    change of view needs the viewport-filtered plots to be remade.'''

    def __init__(self, plotCanvas, app, parent):
        # plotCanvas is the canvas we want to link to the toolbar,
        # app is the iseult main app
        NavigationToolbar2QT.__init__(self, plotCanvas, parent, coordinates=True)
        self.app = app
        self.setMovable(False)
        self.setIconSize(QtCore.QSize(20, 20))

    def home(self, *args, **kwargs):
        NavigationToolbar2QT.home(self, *args, **kwargs)
        self.app.after(100, self.app.check_limits_and_renew)

    def back(self, *args, **kwargs):
        NavigationToolbar2QT.back(self, *args, **kwargs)
        self.app.after(100, self.app.check_limits_and_renew)

    def forward(self, *args, **kwargs):
        NavigationToolbar2QT.forward(self, *args, **kwargs)
        self.app.after(100, self.app.check_limits_and_renew)

class SubPlotWrapper:
    """A simple class that will eventually hold all of the information
    about each sub_plot in the Figure"""

    def __init__(self, parent, figure=None, pos = None, subplot_spec = None, ctype=None, graph = None):
        # What happens when a SubPlotWrapper is initiated
        self.parent = parent # Define the parent. MUST BE MainApp OF ISEULT
        self.chartType = 'PhasePlot' # The default chartype
        # A dictionary that contains all of the plot types.
        # The panels must follow certain conventions. TODO: Make a class and
        # Have the panels extend this class.
        self.PlotTypeDict = {'PhasePlot': PhasePanel,
                             'EnergyPlot': EnergyPanel,
                             'FieldsPlot': FieldsPanel,
                             'DensityPlot': DensPanel,
                             'SpectraPlot': SpectralPanel,
                             'MagPlots': BPanel,
                             'FFTPlots': FFTPanel,
                             'TotalEnergyPlot': TotEnergyPanel,
                             'Moments': MomentsPanel,
                             'OhmsLaw': OhmsLawPanel,
                             'PressureBalance': PressureBalancePanel
                             }
        #####
        #
        # First we create dictionary of dictionarys that will store all of the
        # plot params values at self.PlotParamsDict['ctype']['param_name'],
        #
        # We also need a dictionary that returns a list of all of the params of
        # a certain type, used in loading the config files.
        # self.ParamsTypeDict['ctype']['BoolList'] returns all of the Booleans
        # stored in the self.PlotParamsDict['ctype'] dictionary.
        # BoolList, IntList, FloatList and StrList are the options, and the only
        # types that are allowed self.PlotParamsDict['ctype'] dictionary.
        #
        ####
        self.GenParamDict()
        self.figure = figure
        self.subplot_spec = subplot_spec
        self.pos = pos
        self.graph = graph # The panel class-- e.g. PhasesPanel, FieldsPanel...etc
        self.Changedto1D = False # needed to keep track of color bars and views
        self.Changedto2D = False # needed to keep track of color bars and views
        # True while this panel could not be drawn. Its artists are then either
        # missing or left over from a cleared figure, so the rest of Iseult has
        # to leave it alone until it is successfully drawn again.
        self.draw_failed = False
        #
    def GetKeys(self):
        ''' A function that returns a list of all of the keys required to plot
        the subplot contained within SubPlotWrapper. the set_plot_keys function
        must be defined in each of the subplot panel classes.'''
        return self.graph.set_plot_keys()

    def LoadKey(self, h5key):
        '''This is a function the graph should call to load a particular key
        stored in the Tristan outputfiles'''
        return self.parent.DataDict[h5key]
    def LoadData(self):

        ''' LoadData is called by MainApp, it is defined by the subplot panel
        class, but basically it is a function that should load all of the raw
        output data of the current time slice required to make the plot and
        then perform all the necessary calculations to calculate the quantity
        of interest.'''
        self.graph.LoadData()
    def ChangeGraph(self, str_arg):
        '''ChangeGraph changes the plotted graph to the one given by str_arg.
        str_arg must be a key in self.PlotTypeDict'''

        # First check if the current plot has a color bar
        tmpIs2D = self.PlotParamsDict[self.chartType]['twoD']

        # The settings tab is open if the change came from it, and should stay so
        had_settings = getattr(self.graph, 'settings_window', None) is not None

        # Change the graph type
        self.chartType = str_arg
        # put a list of the previous chart types in iseult

        self.graph = self.PlotTypeDict[self.chartType](self.parent, self)
        if had_settings:
            # The caller closes the old graph's tab right after this returns
            self.graph.OpenSettings()

        # If the graph changes from 1D or 2D, we need to save this
        if tmpIs2D != self.PlotParamsDict[self.chartType]['twoD']:
            self.Changedto1D = not self.PlotParamsDict[self.chartType]['twoD']
            self.Changedto2D = self.PlotParamsDict[self.chartType]['twoD']

        self.parent.after(100, lambda: self.parent.RenewCanvas(ForceRedraw = True))

    def GenParamDict(self):
        '''First we create dictionary of dictionarys that will store all of the
        plot params values at self.PlotParamsDict['ctype']['param_name'],

        We also need a dictionary that returns a list of all of the params of
        a certain type, used in loading the config files.
        self.ParamsTypeDict['ctype']['BoolList'] returns all of the Booleans
        stored in the self.PlotParamsDict['ctype'] dictionary.
        BoolList, IntList, FloatList and StrList are the options, and the only
        types that are allowed self.PlotParamsDict['ctype'] dictionary.
        Generate a dictionary that will store all of the params at dict['ctype']['param_name']
        '''

        self.PlotParamsDict = {plot_type: '' for plot_type in self.PlotTypeDict.keys()}
        for elm in self.PlotTypeDict.keys():

            self.PlotParamsDict[elm] = {key: self.PlotTypeDict[elm].plot_param_dict[key] for key in self.PlotTypeDict[elm].plot_param_dict.keys()}




    def RestoreDefaultPlotParams(self, ctype = None, RestoreAll = False):
        ''' Restore the PlotParamsDictionary to the default values contained in
        the class definition.'''
        if ctype is None: # restore the currently shown plot
            ctype = self.chartType
        if RestoreAll: # Restore all of the plot types for this subplot
            self.GenParamDict()
        else: # Only restore the values of the ctype listed.
            self.PlotParamsDict[ctype] = {key: self.PlotTypeDict[ctype].plot_param_dict[key] for key in self.PlotTypeDict[ctype].plot_param_dict.keys()}

    def SetPlotParam(self, pname, val, ctype = None, update_plot = True, NeedsRedraw = False):
        ''' A function that changes plot param 'pname' for the ctype chart to val
        in the dictionary held by SubPlotWrapper. If update_plot is true,
        the figure will be refreshed. If NeedsRedraw is true, the figure will
        be cleared then re-drawn.'''


        if ctype is None:
            ctype = self.chartType
        # Check to see if a plot is changed from 1d to 2d
        if pname =='twoD':
            if self.PlotParamsDict[ctype][pname] == 1 and val == 0:
                self.Changedto1D = True
                NeedsRedraw = True
            if self.PlotParamsDict[ctype][pname] == 0 and val == 1:
                self.Changedto2D = True
                NeedsRedraw = True

        self.PlotParamsDict[ctype][pname] = val
        if update_plot or NeedsRedraw:
            self.parent.RenewCanvas(ForceRedraw = NeedsRedraw)



    def GetPlotParam(self, pname, ctype = None):
        ''' A function that returns the value of the plot param 'pname' for the
        ctype chart. If ctype is None, the currently shown subplot is used'''

        if ctype is None:
            ctype = self.chartType
        return self.PlotParamsDict[ctype][pname]

    def SetGraph(self, ctype = None):
        ''' SetGraph is useful if you want to change the plot type without redrawing the figure
        (e.g. if you want to set many graphs at once, like when loading in a config file.)'''
        if ctype:
            self.chartType = ctype
        self.graph = self.PlotTypeDict[self.chartType](self.parent, self)

    def DrawGraph(self):
        ''' This function calls a function that must be defined in the subplotpanel
         class, e.g. FieldsPanel.... It creates all of the axes used in the panel,
         and writes the data. It should be called after clearing a figure, the
         chartype is changed or when first initializing the figure. It will be
         called if RenewCanvas(ForceRedraw = True)'''

        self.graph.draw()

    def DrawGraphSafely(self):
        '''Draw this panel, keeping a failure inside it.

        A panel is drawn into a figure that has just been cleared, so an
        exception escaping here would leave every panel after it holding
        artists that belong to the cleared figure, and the whole session would
        then raise on any later refresh. Instead the failure is reported in the
        panel itself and the rest of the figure is drawn as usual. Returns
        whether the panel drew.'''
        self.draw_failed = False
        try:
            self.DrawGraph()
        except Exception:
            self.HandlePanelFailure('draw')
        return not self.draw_failed

    def RefreshGraph(self):
        ''' This function calls a function that must be defined in the subplotpanel
         class, e.g. FieldsPanel.... It only updates things held by  the panel to the data in output files with the current timestep.
         It should be called when stepping through the times, or possibly when a plot param changes.
         It will be called if RenewCanvas(ForceRedraw = False)'''
        self.graph.refresh()

    def RefreshGraphSafely(self):
        '''Refresh this panel, keeping a failure inside it.

        A panel that did not draw has nothing to refresh, and one that raises
        while refreshing is left showing why rather than being allowed to take
        the other panels down with it. Returns whether the panel refreshed.'''
        if self.draw_failed:
            return False
        try:
            self.RefreshGraph()
        except Exception:
            self.HandlePanelFailure('refresh')
        return not self.draw_failed

    def HandlePanelFailure(self, action):
        '''Report the exception being handled inside the panel's own cell.

        The traceback still goes to the terminal, where it is the only record
        of what actually went wrong, and the panel is marked as failed so that
        nothing else in Iseult touches its half-built artists. The next full
        redraw, e.g. after the offending setting is changed, tries the panel
        again.'''
        self.draw_failed = True
        message = f'{self.chartType} failed to {action}:\n' + traceback.format_exc()
        print(message, file = sys.stderr)
        # Only the last line of the traceback is short enough to be readable in
        # a panel; the terminal has the rest.
        summary = traceback.format_exc().strip().split('\n')[-1]
        verb = 'drawn' if action == 'draw' else 'updated'
        try:
            self.ShowPanelError(f'{self.chartType} could not be {verb}\n\n{summary}\n\nSee the terminal for the traceback.')
        except Exception:
            # Drawing the message is a courtesy; never let it raise in turn.
            print('Could not show the error in the panel itself:\n' + traceback.format_exc(), file = sys.stderr)

    def ShowPanelError(self, message):
        '''Replace this panel with `message` written in its grid cell.'''
        figure = self.parent.f
        # Any axes the panel managed to make holds a partly updated plot, which
        # would otherwise be left on the figure underneath the message.
        for attr in ('axes', 'axC'):
            axes = getattr(self.graph, attr, None)
            if axes is not None and axes in figure.axes:
                axes.remove()
        axes = figure.add_subplot(self.parent.gs0[self.pos])
        axes.set_xticks([])
        axes.set_yticks([])
        axes.text(0.5, 0.5, message,
                  transform = axes.transAxes, ha = 'center', va = 'center',
                  wrap = True, color = 'firebrick',
                  size = self.parent.MainParamDict['NumFontSize'])
        # Handing the panel these axes keeps the parts of Iseult that only ask
        # a panel for its axes, e.g. restoring the view, working on an axes
        # that is really in the current figure.
        self.graph.axes = axes

    def OpenSubplotSettings(self):

        ''' A function that that must be defined in the subplotpanel class, e.g.
        FieldsPanel.... Opens up the pop-up that allows one to change parameters
        of the plot, change chart type, etc.'''

        self.graph.OpenSettings()

    def CpuDomainLocs(self):
        '''The CPU boundary locations along the horizontal and vertical axes of
        this subplot. Which of them is x and which is y depends on the physical
        axes the panel is plotted against.'''
        locs = {'x': self.parent.cpu_x_locs, 'y': self.parent.cpu_y_locs,
                'z': self.parent.cpu_y_locs}
        horiz, vert = plot_axes.plot_axes_of(self.graph)
        return (locs.get(horiz, []), locs.get(vert, []))

    def SetCpuDomainLines(self):
        '''This function sets the Cpu lines up. It should only be called when
        redrawing the axes and after the axes is creates as it creates all of
        the line objects.'''

        self.cpu_x_lines = []
        self.cpu_y_lines = []
        horiz_locs, vert_locs = self.CpuDomainLocs()
        for i in range(len(horiz_locs)):
            self.cpu_x_lines.append(self.graph.axes.axvline(horiz_locs[i], linewidth = 1, linestyle = ':',color = 'w') )
        for i in range(len(vert_locs)):
            self.cpu_y_lines.append(self.graph.axes.axhline(vert_locs[i], linewidth = 1, linestyle = ':',color = 'w'))

    def UpdateCpuDomainLines(self):
        '''This updates the location of the Cpu lines. It should only be called
        when refreshing the axes as it requires the line objects to already be
        created.'''
        horiz_locs, vert_locs = self.CpuDomainLocs()
        # The number of boundaries changes with the plane shown, so the lines
        # made for the last one may not fit.
        if len(self.cpu_x_lines) != len(horiz_locs) or len(self.cpu_y_lines) != len(vert_locs):
            self.RemoveCpuDomainLines()
            self.SetCpuDomainLines()
            return
        for i in range(len(self.cpu_x_lines)):
            self.cpu_x_lines[i].set_xdata([horiz_locs[i],horiz_locs[i]])

        for i in range(len(self.cpu_y_lines)):
            self.cpu_y_lines[i].set_ydata([vert_locs[i],vert_locs[i]])


    def RemoveCpuDomainLines(self):
        '''This removes the Cpu lines. It should only be called
        when the user unselects show CPU domains.'''
        # iterate over the line list and destroy the objects
        for i in range(len(self.cpu_x_lines)):
            self.cpu_x_lines.pop().remove()
        for i in range(len(self.cpu_y_lines)):
            self.cpu_y_lines.pop().remove()
class Knob:
    """
    ---- Taken from the Matplotlib gallery
    Knob - simple class with a "setKnob" method.
    A Knob instance is attached to a Param instance, e.g., param.attach(knob)
    Base class is for documentation purposes.
    """
    def setKnob(self, value):
        pass

class Param:
    """
    ---- Taken from the Matplotlib gallery
    The idea of the "Param" class is that some parameter in the GUI may have
    several knobs that both control it and reflect the parameter's state, e.g.
    a slider, text, and dragging can all change the value of the frequency in
    the waveform of this example.
    The class allows a cleaner way to update/"feedback" to the other knobs when
    one is being changed.  Also, this class handles min/max constraints for all
    the knobs.
    Idea - knob list - in "set" method, knob object is passed as well
      - the other knobs in the knob list have a "set" method which gets
        called for the others.
    """
    def __init__(self, initialValue=None, minimum=0., maximum=1.):
        self.minimum = minimum
        self.maximum = maximum
        if initialValue != self.constrain(initialValue):
            raise ValueError('illegal initial value')
        self.value = initialValue
        self.knobs = []

    def attach(self, knob):
        self.knobs += [knob]

    def set(self, value, knob=None):
        if self.value != self.constrain(value):
            self.value = self.constrain(value)
            for feedbackKnob in self.knobs:
                if feedbackKnob != knob:
                    feedbackKnob.setKnob(self.value)
        # Adding a new feature that allows one to loop backwards or forwards:
        elif self.maximum != self.minimum:

            if self.value == self.maximum:
                self.value = self.minimum
                for feedbackKnob in self.knobs:
                    if feedbackKnob != knob:
                        feedbackKnob.setKnob(self.value)

            elif self.value == self.minimum:
                self.value = self.maximum
                for feedbackKnob in self.knobs:
                    if feedbackKnob != knob:
                        feedbackKnob.setKnob(self.value)
        return self.value

    def setMax(self, max_arg, knob=None):
        self.maximum = max_arg
        self.value = self.constrain(self.value)
        for feedbackKnob in self.knobs:
            if feedbackKnob != knob:
                feedbackKnob.setKnob(self.value)
        return self.value

    def constrain(self, value):
        if value <= self.minimum:
            value = self.minimum
        if value >= self.maximum:
            value = self.maximum
        return value



class PlaybackBar(QtWidgets.QToolBar):

    """
    The bar that handles the time-stepping in Iseult: step left, play/pause,
    step right, the time step, a slider through the simulation, loop and
    record toggles, and buttons for the measurement and general settings
    and for reloading or refreshing the data.
    """

    def __init__(self, app, param):
        QtWidgets.QToolBar.__init__(self, 'Playback', app.window)
        self.app = app
        self.playPressed = False
        self.setMovable(False)
        self.setIconSize(QtCore.QSize(20, 20))
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        style = self.style()

        # This param should be the time-step of the simulation
        self.param = param

        self.skipLB = self.addAction(style.standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward), '')
        self.skipLB.setToolTip('Step back (Left arrow)')
        self.skipLB.triggered.connect(self.SkipLeft)

        self._play_icon = style.standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        self._pause_icon = style.standardIcon(QtWidgets.QStyle.SP_MediaPause)
        self.playB = self.addAction(self._play_icon, 'Play')
        self.playB.setToolTip('Play / pause (Space)')
        self.playB.triggered.connect(self.PlayHandler)

        self.skipRB = self.addAction(style.standardIcon(QtWidgets.QStyle.SP_MediaSeekForward), '')
        self.skipRB.setToolTip('Step forward (Right arrow)')
        self.skipRB.triggered.connect(self.SkipRight)

        self.addSeparator()
        self.addWidget(QtWidgets.QLabel(' n = '))

        # The box to type a time step into. Enter (or leaving the box) goes there.
        self.tstep = QtWidgets.QSpinBox()
        self.tstep.setKeyboardTracking(False)
        self.tstep.setRange(self.param.minimum, self.param.maximum)
        self.tstep.setValue(self.param.value)
        self.tstep.setMinimumWidth(70)
        self.tstep.valueChanged.connect(self.TextCallback)
        self.addWidget(self.tstep)
        self.maxLabel = QtWidgets.QLabel()
        self.addWidget(self.maxLabel)

        # A slider that shows the progress through the simulation and selects
        # a time. Dragging it only changes the number shown; the time step is
        # drawn once it is let go of.
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(self.param.minimum, self.param.maximum)
        self.slider.setValue(self.param.value)
        self.slider.setMinimumWidth(120)
        self.slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged.connect(self.ScaleHandler)
        self.slider.sliderReleased.connect(self.UpdateValue)
        self.addWidget(self.slider)
        self._show_max()

        self.addSeparator()
        self.LoopB = QtWidgets.QCheckBox('Loop')
        self.LoopB.setChecked(bool(self.app.MainParamDict['LoopPlayback']))
        self.LoopB.toggled.connect(self.LoopChanged)
        self.addWidget(self.LoopB)
        self.RecB = QtWidgets.QCheckBox('Record')
        self.RecB.setToolTip('Save a PNG of every time step drawn')
        self.RecB.setChecked(bool(self.app.MainParamDict['Recording']))
        self.RecB.toggled.connect(self.RecChanged)
        self.addWidget(self.RecB)

        self.addSeparator()
        self.MeasuresB = self.addAction('FFT')
        self.MeasuresB.setToolTip('FFT measurement region')
        self.MeasuresB.triggered.connect(self.OpenMeasures)
        self.SettingsB = self.addAction('Settings')
        self.SettingsB.setToolTip('General settings (S)')
        self.SettingsB.triggered.connect(self.app.OpenSettings)
        reload_action = self.addAction(style.standardIcon(QtWidgets.QStyle.SP_BrowserReload), 'Reload')
        reload_action.setToolTip('Look for new output files (R)')
        reload_action.triggered.connect(self.OnReload)
        refresh_action = self.addAction('Refresh')
        refresh_action.setToolTip('Reload the current time step from disk')
        refresh_action.triggered.connect(self.OnRefresh)

        # The play loop
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.blink)

        #attach the parameter to the Playbackbar
        self.param.attach(self)

    ####
    #
    # What MainApp tells the bar
    #
    ####

    def show_step(self, value):
        '''Show `value` as the current time step without acting on it.'''
        for widget in (self.tstep, self.slider):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)

    def set_max(self, maximum):
        maximum = max(int(maximum), self.param.minimum)
        for widget in (self.tstep, self.slider):
            widget.blockSignals(True)
            widget.setMaximum(maximum)
            widget.blockSignals(False)
        self._show_max()

    def _show_max(self):
        self.maxLabel.setText(' / %d ' % self.slider.maximum())
        self.slider.setPageStep(max(1, self.slider.maximum() // 20))

    def set_recording(self, value):
        self.RecB.blockSignals(True)
        self.RecB.setChecked(bool(value))
        self.RecB.blockSignals(False)

    def set_loop(self, value):
        self.LoopB.blockSignals(True)
        self.LoopB.setChecked(bool(value))
        self.LoopB.blockSignals(False)

    ####
    #
    # Handlers
    #
    ####

    def OnReload(self, *args):
        _ = self.app.checkAndFindFilePaths(reload_mode = True)
        self.app.RenewCanvas()

    def OnRefresh(self, *args):
        self.app.RefreshTimeStep()
        self.app.RenewCanvas()

    def RecChanged(self, checked):
        value = int(checked)
        if value != self.app.MainParamDict['Recording']:
            self.app.MainParamDict['Recording'] = value
            if value == 1:
                self.app.PrintFig()

    def LoopChanged(self, checked):
        self.app.MainParamDict['LoopPlayback'] = int(checked)

    def SkipLeft(self, *args):
        self.app.StepInteractively(lambda: self.param.set(self.param.value - self.app.MainParamDict['SkipSize']))

    def SkipRight(self, *args):
        self.app.StepInteractively(lambda: self.param.set(self.param.value + self.app.MainParamDict['SkipSize']))

    def PlayHandler(self, *args):
        if not self.playPressed:
            self.playPressed = True
            self.app.RenewCanvas()
            self.playB.setText('Pause')
            self.playB.setIcon(self._pause_icon)
            self.timer.start(int(self.app.MainParamDict['WaitTime']*1E3))
        else:
            # pause the play loop and set the button back to play
            self.playPressed = False
            self.timer.stop()
            self.app.RenewCanvas()
            self.playB.setText('Play')
            self.playB.setIcon(self._play_icon)

    def OpenMeasures(self, *args):
        if self.app.measure_window is not None:
            self.app.measure_window.destroy()
        self.app.measure_window = MeasureFrame(self.app)

    def blink(self):
        if self.playPressed:
            # First check to see if the timestep can get larger
            if self.param.value == self.param.maximum and not self.app.MainParamDict['LoopPlayback']:
                # push pause button
                self.PlayHandler()
                return
            # otherwise skip right by size skip size
            self.param.set(self.param.value + self.app.MainParamDict['SkipSize'])
            # Wait from the end of this frame's draw, so a slow frame is still seen
            self.timer.start(int(self.app.MainParamDict['WaitTime']*1E3))

    def TextCallback(self, *args):
        value = self.tstep.value()
        if value != self.param.value:
            self.app.StepInteractively(lambda: self.param.set(value))

    def ScaleHandler(self, value):
        # Follow the slider in the box. Only a released drag, a click on the
        # groove or a key changes the time step itself.
        self.tstep.blockSignals(True)
        self.tstep.setValue(value)
        self.tstep.blockSignals(False)
        if not self.slider.isSliderDown():
            self.UpdateValue()

    def UpdateValue(self, *args):
        value = self.slider.value()
        if value != self.param.value:
            self.app.StepInteractively(lambda: self.param.set(value))

    def setKnob(self, value):
        pass


class _FormDialog(QtWidgets.QDialog):
    '''A modal dialog of labelled entries with OK/Cancel buttons. Subclasses
    fill in the form and implement validate and apply.'''

    def __init__(self, app, title, ok_text = 'OK', cancel = True):
        QtWidgets.QDialog.__init__(self, app.window)
        self.app = app
        self.setWindowTitle(title)
        self.form = QtWidgets.QFormLayout()
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(self.form)
        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton(ok_text, QtWidgets.QDialogButtonBox.AcceptRole)
        if cancel:
            buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def entry(self, label, text = ''):
        e = QtWidgets.QLineEdit(str(text))
        self.form.addRow(label, e)
        return e

    def ok(self):
        if self.validate():
            self.accept()
            self.apply()

    def warn(self, message):
        QtWidgets.QMessageBox.warning(self, 'Bad input', message)

    def validate(self):
        return True

    def apply(self):
        pass


class SaveDialog(_FormDialog):
    '''Saves the current state of Iseult as a preset view.'''

    def __init__(self, parent):
        _FormDialog.__init__(self, parent, 'Save Current State', ok_text = 'Save')
        self.e1 = self.entry('Name of view:')
        self.exec()

    def validate(self):
        if self.e1.text().strip() == '':
            self.warn('Field must contain a name, please try again')
            return False
        return True

    def apply(self):
        ''' Save the config file'''
        name = self.e1.text().strip()
        self.app.SaveIseultState(os.path.join(self.app.IseultDir, '.iseult_configs', name.replace(' ', '_') +'.yml'), name)


class MaxNDialog(_FormDialog):
    '''Asks for the largest output file number to consider.'''

    def __init__(self, parent):
        _FormDialog.__init__(self, parent, 'Max Frame', cancel = False)
        self.e1 = self.entry('Max frame (-1 for last frame):', self.app.cmd_args.n)
        self.exec()

    def validate(self):
        try:
            self.N = int(self.e1.text())
        except ValueError:
            self.warn('Max N must contain an int, please try again')
            return False
        return True

    def apply(self):
        '''Update the -n option'''
        self.app.cmd_args.n = self.N


class MovieDialog(_FormDialog):
    '''Asks what frames to make a movie of, and where to save it.'''

    def __init__(self, parent):
        _FormDialog.__init__(self, parent, 'Make a Movie', ok_text = 'Save')
        self.e1 = self.entry('Name of movie:')
        self.e2 = self.entry('First frame:', 1)
        self.e3 = self.entry('Last frame (-1 for final frame):', -1)
        self.e4 = self.entry('Step size:', 1)
        self.e5 = self.entry('Frames per second:', self.app.cmd_args.framerate)
        # The resolution of the movie. The figure keeps its size in inches, so
        # a higher dpi gives more pixels (and larger text in pixels), not a
        # different layout. Defaults to the dpi of the figure on screen.
        self.e7 = self.entry('DPI:', f'{self.app.f.dpi:g}')

        row = QtWidgets.QHBoxLayout()
        self.e6 = QtWidgets.QLineEdit(os.path.abspath(os.path.join(self.app.dirname, '..')))
        self.e6.setMinimumWidth(320)
        browse = QtWidgets.QPushButton('Browse…')
        browse.setAutoDefault(False)
        browse.clicked.connect(self.browse)
        row.addWidget(self.e6)
        row.addWidget(browse)
        self.form.addRow('Movie directory:', row)
        self.exec()

    def browse(self):
        path = filedialog.askdirectory(title = 'Movie directory', initialdir = self.e6.text(), parent = self)
        if path:
            self.e6.setText(path)

    def validate(self):
        ''' Check to make sure the Movie will work'''
        name = self.e1.text().strip()
        self.outdir = self.e6.text().strip()
        if name == '':
            self.warn('Field must contain a name, please try again')
            return False
        self.Name = name.replace(' ', '_') + '.mov'
        try:
            self.StartFrame = int(self.e2.text())
            self.EndFrame = int(self.e3.text())
        except ValueError:
            self.warn('The first and last frames must be integers, please try again')
            return False
        try:
            self.Step = int(self.e4.text())
            assert self.Step > 0
        except (ValueError, AssertionError):
            self.warn('Step must be an integer >0, please try again')
            return False
        try:
            self.FPS = int(self.e5.text())
            assert self.FPS > 0
        except (ValueError, AssertionError):
            self.warn('FPS must be an integer >0, please try again')
            return False
        try:
            self.DPI = float(self.e7.text())
            assert self.DPI > 0
        except (ValueError, AssertionError):
            self.warn('DPI must be a number >0, please try again')
            return False

        n_frames = len(self.app.PathDict['Param'])
        if self.StartFrame < 0:
            self.StartFrame = n_frames + self.StartFrame + 1
        if self.EndFrame < 0:
            self.EndFrame = n_frames + self.EndFrame + 1
        if self.StartFrame == 0:
            self.warn('Starting frame cannot be zero')
            return False
        if self.EndFrame == 0:
            self.warn('Ending frame cannot be zero')
            return False

        if not os.path.isdir(self.outdir):
            self.warn(f'{self.outdir} is not a directory')
            return False
        filepath = os.path.join(self.outdir, self.Name)
        try:
            with open(filepath, 'w'):
                pass
            os.remove(filepath)
        except IOError:
            self.warn(f'You do not have write access to {self.outdir}')
            return False
        return True

    def apply(self):
        ''' Save the Movie'''
        self.app.MakeAMovie(fname = self.Name,
                                start = self.StartFrame,
                                stop = self.EndFrame,
                                step = self.Step,
                                FPS = self.FPS,
                                outdir = self.outdir,
                                dpi = self.DPI)


class _PresetList(QtWidgets.QListWidget):
    def __init__(self, on_drop):
        QtWidgets.QListWidget.__init__(self)
        self.on_drop = on_drop

    def dropEvent(self, event):
        QtWidgets.QListWidget.dropEvent(self, event)
        self.on_drop()


class PresetManager(QtWidgets.QDialog):
    '''Delete, rename and reorder the views in the Preset Views menu.
    Every change is written to .iseult_configs right away.'''

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent.window)
        self.setWindowTitle('Manage Preset Views')
        self.app = parent
        self.config_dir = os.path.join(parent.IseultDir, '.iseult_configs')
        self.presets = []

        self.listbox = _PresetList(self.dropped)
        self.listbox.setMinimumSize(260, 320)
        self.listbox.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.listbox.itemDoubleClicked.connect(lambda item: self.rename())

        buttons = QtWidgets.QVBoxLayout()
        for text, command in [('Move to Top', self.move_top),
                              ('Move Up', partial(self.move, -1)),
                              ('Move Down', partial(self.move, 1)),
                              ('Move to Bottom', self.move_bottom),
                              (None, None),
                              ('Rename…', self.rename),
                              ('Delete', self.delete),
                              (None, None),
                              ('Load', self.load),
                              ('Close', self.close)]:
            if text is None:
                buttons.addSpacing(10)
            else:
                b = QtWidgets.QPushButton(text)
                b.setAutoDefault(False)
                b.clicked.connect(command)
                buttons.addWidget(b)
        buttons.addStretch(1)

        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.listbox, 1)
        body.addLayout(buttons)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(body)
        hint = QtWidgets.QLabel('Drag or use Alt+Up/Down to reorder. Default cannot be renamed or deleted.')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        for key, command in [('Alt+Up', partial(self.move, -1)),
                             ('Alt+Down', partial(self.move, 1)),
                             ('Delete', self.delete),
                             ('F2', self.rename)]:
            QtGui.QShortcut(QtGui.QKeySequence(key), self.listbox, command)

        self.refresh()
        if self.presets:
            self.select(0)
        self.show()

    def refresh(self, select_file=None):
        self.presets = preset_views.list_presets(self.config_dir)
        self.listbox.blockSignals(True)
        self.listbox.clear()
        for name, fname in self.presets:
            item = QtWidgets.QListWidgetItem(name)
            item.setData(Qt.UserRole, fname)
            if fname in preset_views.PROTECTED:
                item.setForeground(QtGui.QColor('gray'))
            self.listbox.addItem(item)
        self.listbox.blockSignals(False)
        if select_file is not None:
            for i, (_, fname) in enumerate(self.presets):
                if fname == select_file:
                    self.select(i)

    def select(self, i):
        self.listbox.setCurrentRow(i)

    def current(self):
        row = self.listbox.currentRow()
        return row if row >= 0 else None

    def reorder(self, src, dst):
        dst = max(0, min(dst, len(self.presets)-1))
        if src is None or src == dst:
            return
        order = [fname for _, fname in self.presets]
        order.insert(dst, order.pop(src))
        preset_views.save_order(self.config_dir, order)
        self.refresh(select_file=order[dst])

    def dropped(self):
        '''Save the order the list was dragged into.'''
        order = [self.listbox.item(i).data(Qt.UserRole) for i in range(self.listbox.count())]
        moved = self.listbox.currentItem().data(Qt.UserRole) if self.listbox.currentItem() else None
        preset_views.save_order(self.config_dir, order)
        self.refresh(select_file=moved)

    def move(self, step):
        i = self.current()
        if i is not None:
            self.reorder(i, i+step)

    def move_top(self):
        self.reorder(self.current(), 0)

    def move_bottom(self):
        self.reorder(self.current(), len(self.presets)-1)

    def rename(self):
        i = self.current()
        if i is None:
            return
        name, fname = self.presets[i]
        new_name = simpledialog.askstring('Rename Preset', 'New name:', initialvalue=name, parent=self)
        if new_name is None or new_name.strip() == name:
            return
        try:
            new_file = preset_views.rename_preset(self.config_dir, fname, new_name)
        except (ValueError, OSError) as e:
            messagebox.showwarning('Cannot rename', str(e), parent=self)
            return
        self.refresh(select_file=new_file)

    def delete(self):
        i = self.current()
        if i is None:
            return
        name, fname = self.presets[i]
        if not messagebox.askyesno('Delete Preset', f'Delete the preset view "{name}"?\nThis removes {fname}.',
                                   parent=self):
            return
        try:
            preset_views.delete_preset(self.config_dir, fname)
        except (ValueError, OSError) as e:
            messagebox.showwarning('Cannot delete', str(e), parent=self)
            return
        self.refresh()
        if self.presets:
            self.select(min(i, len(self.presets)-1))

    def load(self):
        i = self.current()
        if i is not None:
            self.app.LoadConfig(os.path.join(self.config_dir, self.presets[i][1]))

    def winfo_exists(self):
        try:
            return self.isVisible()
        except RuntimeError:
            return False

    def lift(self):
        self.raise_()
        self.activateWindow()


class SettingsFrame(Tk.Toplevel):
    def __init__(self, parent):

        Tk.Toplevel.__init__(self)
        self.wm_title('General Settings')
        self.protocol('WM_DELETE_WINDOW', self.OnClosing)

        self.bind('<Return>', self.SettingsCallback)

        self.parent = parent
        frm = ttk.Frame(self)
        frm.pack(fill=Tk.BOTH, expand=True)

        # Make an entry to change the skip size
        self.skipSize = Tk.StringVar(self)
        self.skipSize.set(self.parent.MainParamDict['SkipSize']) # default value
        self.skipSize.trace('w', self.SkipSizeChanged)
        ttk.Label(frm, text="Skip Size:").grid(row=0)
        self.skipEnter = ttk.Entry(frm, textvariable=self.skipSize, width = 6)
        self.skipEnter.grid(row =0, column = 1, sticky = Tk.W + Tk.E)

        # Make an button to change the wait time
        self.waitTime = Tk.StringVar(self)
        self.waitTime.set(self.parent.MainParamDict['WaitTime']) # default value
        self.waitTime.trace('w', self.WaitTimeChanged)
        ttk.Label(frm, text="Playback Wait Time:").grid(row=1)
        self.waitEnter = ttk.Entry(frm, textvariable=self.waitTime, width = 6)
        self.waitEnter.grid(row =1, column = 1, sticky = Tk.W + Tk.E)

        # Have a list of the color maps
        self.cmapvar = Tk.StringVar(self)
        self.cmapvar.set(self.parent.MainParamDict['ColorMap']) # default value
        self.cmapvar.trace('w', self.CmapChanged)

        ttk.Label(frm, text="Color map:").grid(row=2)
        cmapChooser = ttk.OptionMenu(frm, self.cmapvar, self.parent.MainParamDict['ColorMap'], *tuple(new_cmaps.sequential))
        cmapChooser.grid(row =2, column = 1, sticky = Tk.W + Tk.E)

        # Have a list of the color maps
        self.divcmapList = new_cmaps.cmaps.keys()
        self.div_cmapvar = Tk.StringVar(self)
        self.div_cmapvar.set(self.parent.MainParamDict['DivColorMap']) # default value
        self.div_cmapvar.trace('w', self.DivCmapChanged)

        ttk.Label(frm, text="Diverging Cmap:").grid(row=3)
        cmapChooser = ttk.OptionMenu(frm, self.div_cmapvar, self.parent.MainParamDict['DivColorMap'], *tuple(new_cmaps.diverging))
        cmapChooser.grid(row =3, column = 1, sticky = Tk.W + Tk.E)


        # Make an entry to change the number of columns
        self.columnNum = Tk.StringVar(self)
        self.columnNum.set(self.parent.MainParamDict['NumOfCols']) # default value
        self.columnNum.trace('w', self.ColumnNumChanged)
        ttk.Label(frm, text="# of columns:").grid(row=4)
        self.ColumnSpin = ttk.Spinbox(frm,  from_=1, to=self.parent.MainParamDict['MaxCols'], textvariable=self.columnNum, width = 6)
        self.ColumnSpin.grid(row =4, column = 1, sticky = Tk.W + Tk.E)

        # Make an entry to change the number of columns
        self.rowNum = Tk.StringVar(self)
        self.rowNum.set(self.parent.MainParamDict['NumOfRows']) # default value
        self.rowNum.trace('w', self.RowNumChanged)
        ttk.Label(frm, text="# of rows:").grid(row=5)
        self.RowSpin = ttk.Spinbox(frm, from_=1, to=self.parent.MainParamDict['MaxRows'], textvariable=self.rowNum, width = 6)
        self.RowSpin.grid(row =5, column = 1, sticky = Tk.W + Tk.E)

        self.PrtlStrideVar = Tk.StringVar()
        self.PrtlStrideVar.set(str(self.parent.MainParamDict['PrtlStride']))
        ttk.Entry(frm, textvariable = self.PrtlStrideVar, width =6).grid(row =6, column =1, sticky = Tk.W +Tk.E)
        ttk.Label(frm, text='Particle stride').grid(row= 6,column =0)

        # Control whether or not Title is shown
        self.TitleVar = Tk.IntVar()
        self.TitleVar.set(self.parent.MainParamDict['ShowTitle'])
        self.TitleVar.trace('w', self.TitleChanged)

        self.LimVar = Tk.IntVar()
        self.LimVar.set(self.parent.MainParamDict['SetxLim'])
        self.LimVar.trace('w', self.LimChanged)

        self.xleft = Tk.StringVar()
        self.xleft.set(str(self.parent.MainParamDict['xLeft']))
        self.xright = Tk.StringVar()
        self.xright.set(str(self.parent.MainParamDict['xRight']))


        ttk.Label(frm, text = 'min').grid(row= 7, column = 1, sticky = Tk.N)
        ttk.Label(frm, text = 'max').grid(row= 7, column = 2, sticky = Tk.N)
        cb = ttk.Checkbutton(frm, text ='Set xlim',
                        variable = self.LimVar)
        cb.grid(row = 8, sticky = Tk.N)
        ttk.Entry(frm, textvariable=self.xleft, width = 8).grid(row = 8, column =1, sticky = Tk.N)
        ttk.Entry(frm, textvariable=self.xright, width = 8).grid(row = 8, column =2, sticky = Tk.N)



        self.yLimVar = Tk.IntVar()
        self.yLimVar.set(self.parent.MainParamDict['SetyLim'])
        self.yLimVar.trace('w', self.yLimChanged)



        self.yleft = Tk.StringVar()
        self.yleft.set(str(self.parent.MainParamDict['yBottom']))
        self.yright = Tk.StringVar()
        self.yright.set(str(self.parent.MainParamDict['yTop']))


        ttk.Checkbutton(frm, text ='Set ylim',
                        variable = self.yLimVar).grid(row = 9, sticky = Tk.N)
        ttk.Entry(frm, textvariable=self.yleft, width = 8 ).grid(row = 9, column =1, sticky = Tk.N)
        ttk.Entry(frm, textvariable=self.yright, width =8 ).grid(row = 9, column =2, sticky = Tk.N)

        self.kLimVar = Tk.IntVar()
        self.kLimVar.set(self.parent.MainParamDict['SetkLim'])
        self.kLimVar.trace('w', self.kLimChanged)



        self.kleft = Tk.StringVar()
        self.kleft.set(str(self.parent.MainParamDict['kLeft']))
        self.kright = Tk.StringVar()
        self.kright.set(str(self.parent.MainParamDict['kRight']))


        ttk.Checkbutton(frm, text ='Set klim', variable = self.kLimVar).grid(row = 10, sticky = Tk.N)
        ttk.Entry(frm, textvariable=self.kleft, width = 8 ).grid(row = 10, column =1, sticky = Tk.N)
        ttk.Entry(frm, textvariable=self.kright, width =8 ).grid(row = 10, column =2, sticky = Tk.N)

        self.xRelVar = Tk.IntVar()
        self.xRelVar.set(self.parent.MainParamDict['xLimsRelative'])
        self.xRelVar.trace('w', self.xRelChanged)
        ttk.Checkbutton(frm, text = "x limits & zooms relative to shock",
                        variable = self.xRelVar).grid(row = 11, columnspan = 3, sticky = Tk.W)

        framecb = ttk.Frame(frm)

        ttk.Label(framecb, text='Choose 2D plane:').pack(side = Tk.LEFT, expand = 0)
        self.PlaneVar = Tk.IntVar()
        self.PlaneVar.set(self.parent.MainParamDict['2DSlicePlane'])
        self.xybutton = ttk.Radiobutton(framecb,
                            text='x-y',
                            variable=self.PlaneVar,
                            command = self.RadioPlane,
                            value=0)
        self.xybutton.pack(side = Tk.LEFT, expand = 0)
        self.xzbutton = ttk.Radiobutton(framecb,
                            text='x-z',
                            variable=self.PlaneVar,
                            command = self.RadioPlane,
                            value=1)
        self.xzbutton.pack(side = Tk.LEFT, expand = 0)
        self.yzbutton = ttk.Radiobutton(framecb,
                            text='y-z',
                            variable=self.PlaneVar,
                            command = self.RadioPlane,
                            value=2)
        self.yzbutton.pack(side = Tk.LEFT, expand = 0)
        framecb.grid(row = 12, columnspan = 4)

        framex = ttk.Frame(frm)
        self.xSliceVar = Tk.IntVar()
        self.xSliceVar.set(self.parent.xSlice)
        self.units_listx = []
        for i in range(self.parent.MaxXInd+1):
            self.units_listx.append(str(i*self.parent.istep/self.parent.c_omp))

        self.xSliceVarC_omp = Tk.StringVar()
        self.xSliceVarC_omp.set(self.units_listx[self.xSliceVar.get()])

        # The x-slice is where the 2D y-z plane is cut. Lineouts are taken
        # across the 2D viewport instead, see plot_axes.lineout_window.
        labelx = ttk.Label(framex, text='x-slice')#
        labelx.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=0)


        # A slider that will select the 2D slice in the simulation
        self.sliderx = ttk.Scale(framex, from_=0, to=self.parent.MaxXInd, command = self.xScaleHandler)
        self.sliderx.set(self.xSliceVar.get())
        self.sliderx.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)


        self.txtEnterx = ttk.Entry(framex, textvariable=self.xSliceVarC_omp, width=6)
        self.txtEnterx.pack(side=Tk.LEFT, fill = Tk.BOTH, expand = 0)
        if self.parent.MaxXInd ==0:
            self.txtEnterx.state(['disabled'])
            self.sliderx.state(['disabled'])
        ttk.Label(framex, text='[c_omp]').pack(side=Tk.LEFT, fill=Tk.BOTH, expand=0)
        # bind releasing the moust button to updating the plots.
        self.sliderx.bind("<ButtonRelease-1>", self.xUpdateValue)


        self.framex = framex

        framey = ttk.Frame(frm)
        self.ySliceVar = Tk.IntVar()
        self.ySliceVar.set(self.parent.ySlice)
        self.units_listy = []
        for i in range(self.parent.MaxYInd+1):
            self.units_listy.append(str(i*self.parent.istep/self.parent.c_omp))

        self.ySliceVarC_omp = Tk.StringVar()
        self.ySliceVarC_omp.set(self.units_listy[self.ySliceVar.get()])

        # The y-slice is where the 2D x-z plane is cut.
        labely = ttk.Label(framey, text='y-slice')#
        labely.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=0)


        # A slider that will select the 2D slice in the simulation
        self.slidery = ttk.Scale(framey, from_=0, to=self.parent.MaxYInd, command = self.yScaleHandler)
        self.slidery.set(self.ySliceVar.get())
        self.slidery.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)


        self.txtEntery = ttk.Entry(framey, textvariable=self.ySliceVarC_omp, width=6)
        self.txtEntery.pack(side=Tk.LEFT, fill = Tk.BOTH, expand = 0)
        if self.parent.MaxYInd ==0:
            self.txtEntery.state(['disabled'])
            self.slidery.state(['disabled'])
        ttk.Label(framey, text='[c_omp]').pack(side=Tk.LEFT, fill=Tk.BOTH, expand=0)
        # bind releasing the moust button to updating the plots.
        self.slidery.bind("<ButtonRelease-1>", self.yUpdateValue)


        self.framey = framey

        framez = ttk.Frame(frm)
        self.zSliceVar = Tk.IntVar()
        self.zSliceVar.set(int(np.around(self.parent.MainParamDict['zSlice']*self.parent.MaxZInd)))

        self.units_listz = []
        for i in range(self.parent.MaxZInd+1):
            self.units_listz.append(str(i*self.parent.istep/self.parent.c_omp))

        self.zSliceVarC_omp = Tk.StringVar()
        self.zSliceVarC_omp.set(self.units_listz[self.zSliceVar.get()])

        # An entry box that will let us choose the time-step
        ttk.Label(framez, text='z-slice').pack(side=Tk.LEFT, fill=Tk.BOTH, expand=0)

        # A slider that will select the 2D slice in the simulation
        self.sliderz = ttk.Scale(framez, from_=0, to=self.parent.MaxZInd, command = self.zScaleHandler)
        self.sliderz.set(self.zSliceVar.get())
        self.sliderz.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)

        self.txtEnterz = ttk.Entry(framez, textvariable=self.zSliceVarC_omp, width=6)
        self.txtEnterz.pack(side=Tk.LEFT, fill = Tk.BOTH, expand = 0)
        ttk.Label(framez, text='[c_omp]').pack(side=Tk.LEFT, fill=Tk.BOTH, expand=0)
        # bind releasing the moust button to updating the plots.
        self.sliderz.bind("<ButtonRelease-1>", self.zUpdateValue)
        if self.parent.MaxZInd ==0:
            self.xzbutton.state(['disabled'])
            self.txtEnterz.state(['disabled'])
            self.sliderz.state(['disabled'])


        self.framez = framez
        # Only the slider for the axis across the 2D plane is shown.
        self.GridPlaneSlider()

        cb = ttk.Checkbutton(frm, text = "Show Title",
                        variable = self.TitleVar)
        cb.grid(row = 16, sticky = Tk.W)
        # Control whether or not axes are shared with a radio box:
        self.toLinkList = ['None', 'All spatial', 'All non p-x', 'All 2-D spatial']
        self.LinkedVar = Tk.IntVar()
        self.LinkedVar.set(self.parent.MainParamDict['LinkSpatial'])

        ttk.Label(frm, text='Share spatial axes:').grid(row = 0, column = 2, sticky = Tk.W)

        for i in range(len(self.toLinkList)):
            ttk.Radiobutton(frm,
                    text=self.toLinkList[i],
                    variable=self.LinkedVar,
                    command = self.RadioLinked,
                    value=i).grid(row = 1+i, column = 2, sticky =Tk.N)

        self.AspectVar = Tk.IntVar()
        self.AspectVar.set(self.parent.MainParamDict['ImageAspect'])
        self.AspectVar.trace('w', self.AspectVarChanged)

        cb = ttk.Checkbutton(frm, text = "Aspect = 1",
                                variable = self.AspectVar)
        cb.grid(row = 16, column = 1, sticky = Tk.W)

        self.ConstantShockVar = Tk.IntVar()
        self.ConstantShockVar.set(self.parent.MainParamDict['ConstantShockVel'])
        self.ConstantShockVar.trace('w', self.ShockSpeedVarChanged)

        cb = ttk.Checkbutton(frm, text = "Constant Shock v",
                                variable = self.ConstantShockVar)
        cb.grid(row = 16, column = 2, sticky = Tk.W)

        # 1D lineouts are taken across the region the 2D panels show:
        # through its centre, or averaged over all of it.
        self.Average1DVar = Tk.IntVar()
        self.Average1DVar.set(self.parent.MainParamDict['Average1D'])
        self.Average1DVar.trace('w', self.AverageChanged)
        frame1d = ttk.Frame(frm)
        ttk.Label(frame1d, text='1D lineouts across the 2D view:').pack(side=Tk.LEFT, expand=0)
        for value, text in ((0, 'center'), (1, 'average')):
            ttk.Radiobutton(frame1d, text=text, variable=self.Average1DVar,
                            value=value).pack(side=Tk.LEFT, expand=0)
        frame1d.grid(row = 14, columnspan = 4)

        self.CbarOrientation = Tk.IntVar()
        self.CbarOrientation.set(self.parent.MainParamDict['HorizontalCbars'])
        self.CbarOrientation.trace('w', self.OrientationChanged)

        cb = ttk.Checkbutton(frm, text = "Horizontal Cbars",
                                variable = self.CbarOrientation)
        cb.grid(row = 17, sticky = Tk.W)


        self.LinkKVar = Tk.IntVar()
        self.LinkKVar.set(self.parent.MainParamDict['LinkK'])
        self.LinkKVar.trace('w', self.LinkKChanged)

        cb = ttk.Checkbutton(frm, text = "Share k-axes",
                                variable = self.LinkKVar)
        cb.grid(row = 17, column =1, sticky = Tk.W)



        self.LorentzBoostVar = Tk.IntVar()
        self.LorentzBoostVar.set(self.parent.MainParamDict['DoLorentzBoost'])
        self.LorentzBoostVar.trace('w', self.LorentzBoostChanged)
        cb = ttk.Checkbutton(frm, text='Boost phase plots along x', variable =  self.LorentzBoostVar).grid(row = 18, sticky = Tk.W)
        # One number sets the speed of the frame: a value below 1 in size is
        # read as beta = v/c, anything else as the Lorentz factor Gamma.
        ttk.Label(frm, text='Frame speed (β or Γ):').grid(row= 18, column =1, sticky = Tk.E)
        self.GammaVar = Tk.StringVar()
        self.GammaVar.set(str(self.parent.MainParamDict['GammaBoost']))
        ttk.Entry(frm, textvariable=self.GammaVar, width = 7).grid(row = 18, column = 2, sticky = Tk.W)
        self.BoostReadout = ttk.Label(frm, foreground = 'gray30')
        self.BoostReadout.grid(row = 19, column = 0, columnspan = 3, sticky = Tk.W)
        self.GammaVar.trace('w', self.UpdateBoostReadout)
        self.UpdateBoostReadout()

    def xScaleHandler(self, e):
        # if changing the scale will change the value of the parameter, do so
        if self.xSliceVar.get() != int(self.sliderx.get()):
            self.xSliceVar.set(int(self.sliderx.get()))
            self.xSliceVarC_omp.set(self.units_listx[self.xSliceVar.get()])

    def yScaleHandler(self, e):
        # if changing the scale will change the value of the parameter, do so
        if self.ySliceVar.get() != int(self.slidery.get()):
            self.ySliceVar.set(int(self.slidery.get()))
            self.ySliceVarC_omp.set(self.units_listy[self.ySliceVar.get()])

    def zScaleHandler(self, e):
        # if changing the scale will change the value of the parameter, do so
        if self.zSliceVar.get() != int(self.sliderz.get()):
            self.zSliceVar.set(int(self.sliderz.get()))
            self.zSliceVarC_omp.set(self.units_listz[self.zSliceVar.get()])

    def zUpdateValue(self, e):
        if self.zSliceVar.get() == self.parent.zSlice:
            pass

        else:
            self.parent.MainParamDict['zSlice'] = float(self.zSliceVar.get())/self.parent.MaxZInd
            self.zSliceVarC_omp.set(self.units_listz[self.zSliceVar.get()])
            self.parent.RenewCanvas()

    def xUpdateValue(self, e):
        if self.xSliceVar.get() == self.parent.xSlice:
            pass

        else:
            self.parent.MainParamDict['xSlice'] = float(self.xSliceVar.get())/self.parent.MaxXInd
            self.xSliceVarC_omp.set(self.units_listx[self.xSliceVar.get()])
            self.parent.RenewCanvas()

    def yUpdateValue(self, e):
        if self.ySliceVar.get() == self.parent.ySlice:
            pass

        else:
            self.parent.MainParamDict['ySlice'] = float(self.ySliceVar.get())/self.parent.MaxYInd
            self.ySliceVarC_omp.set(self.units_listy[self.ySliceVar.get()])
            self.parent.RenewCanvas()


    def AspectVarChanged(self, *args):
        if self.AspectVar.get() == self.parent.MainParamDict['ImageAspect']:
            pass

        else:
            self.parent.MainParamDict['ImageAspect'] = self.AspectVar.get()
            self.parent.RenewCanvas(ForceRedraw = True)


    def ShockSpeedVarChanged(self, *args):
        if self.parent.MainParamDict['ConstantShockVel'] != self.ConstantShockVar.get():
            self.parent.MainParamDict['ConstantShockVel'] = self.ConstantShockVar.get()
            self.parent.RenewCanvas(ForceRedraw = True)
    def AverageChanged(self, *args):
        if self.parent.MainParamDict['Average1D'] != self.Average1DVar.get():
            self.parent.MainParamDict['Average1D'] = self.Average1DVar.get()
            self.parent.RenewCanvas()

    def OrientationChanged(self, *args):
        if self.CbarOrientation.get() == self.parent.MainParamDict['HorizontalCbars']:
            pass

        else:
            if self.CbarOrientation.get():
                self.parent.axes_extent = self.parent.MainParamDict['HAxesExtent']
                self.parent.cbar_extent = self.parent.MainParamDict['HCbarExtent']
                self.parent.SubPlotParams = self.parent.MainParamDict['HSubPlotParams']

            else:
                self.parent.axes_extent = self.parent.MainParamDict['VAxesExtent']
                self.parent.cbar_extent = self.parent.MainParamDict['VCbarExtent']
                self.parent.SubPlotParams = self.parent.MainParamDict['VSubPlotParams']
            self.parent.MainParamDict['HorizontalCbars'] = self.CbarOrientation.get()
            self.parent.f.subplots_adjust( **self.parent.SubPlotParams)
            self.parent.RenewCanvas(ForceRedraw=True)

    def UpdateBoostReadout(self, *args):
        '''Spell out the boost the frame-speed entry describes.'''
        try:
            value = float(self.GammaVar.get())
        except ValueError:
            self.BoostReadout.config(text = '    Enter β = v/c (|β| < 1) or Γ (≥ 1); negative boosts toward −x.')
            return
        boost = phase_space.boost_factors(value)
        if boost is None or boost[1] == 0:
            text = 'no boost'
        else:
            big_gamma, beta = boost
            text = f'β = {abs(beta):.4g}, Γ = {big_gamma:.4g}, frame moving toward {"+" if beta > 0 else "−"}x'
        self.BoostReadout.config(text = '    ' + text + '   (|value| < 1 is β, else Γ; − for −x)')

    def LorentzBoostChanged(self, *args):
        if self.LorentzBoostVar.get() == self.parent.MainParamDict['DoLorentzBoost']:
            pass

        else:
            self.parent.MainParamDict['DoLorentzBoost'] = self.LorentzBoostVar.get()
            self.parent.RenewCanvas()

    def TitleChanged(self, *args):
        if self.TitleVar.get()==self.parent.MainParamDict['ShowTitle']:
            pass
        else:
            self.parent.MainParamDict['ShowTitle'] = self.TitleVar.get()
            if self.TitleVar.get() == False:
                self.parent.f.suptitle('')

            self.parent.RenewCanvas()

    def RadioLinked(self, *args):
        # If the shared axes are changed, the whole plot must be redrawn
        if self.LinkedVar.get() == self.parent.MainParamDict['LinkSpatial']:
            pass
        else:
            self.parent.MainParamDict['LinkSpatial'] = self.LinkedVar.get()
            self.parent.RenewCanvas(ForceRedraw = True)
    def RadioPlane(self, *args):
        # If the shared axes are changed, the whole plot must be redrawn
        if self.PlaneVar.get() == self.parent.MainParamDict['2DSlicePlane']:
            pass
        else:
            self.parent.MainParamDict['2DSlicePlane'] = self.PlaneVar.get()
            self.GridPlaneSlider()
            # Which panels share an axis depends on the plane, and a redraw
            # also retries any panel the last plane could not show.
            self.parent.RenewCanvas(ForceRedraw = True)


    def GridPlaneSlider(self):
        '''Show the slider that moves the 2D plane along the axis across it.'''
        frames = (self.framez, self.framey, self.framex) # indexed by '2DSlicePlane'
        # All are taken out first, so the one shown never shares its cell.
        for frame in frames:
            frame.grid_remove()
        frames[self.PlaneVar.get()].grid(row = 13, columnspan = 4)

    def LinkKChanged(self, *args):
        # If the shared axes are changed, the whole plot must be redrawn
        if self.LinkKVar.get() == self.parent.MainParamDict['LinkK']:
            pass
        else:
            self.parent.MainParamDict['LinkK'] = self.LinkKVar.get()
            self.parent.RenewCanvas(ForceRedraw = True)

    def xRelChanged(self, *args):
        # If the shared axes are changed, the whole plot must be redrawn
        if self.xRelVar.get() == self.parent.MainParamDict['xLimsRelative']:
            pass
        else:
            self.parent.MainParamDict['xLimsRelative'] = self.xRelVar.get()
            self.parent.RenewCanvas()


    def CmapChanged(self, *args):
    # Note here that Tkinter passes an event object to onselect()
        if self.cmapvar.get() == self.parent.MainParamDict['ColorMap']:
            pass
        else:
            self.parent.MainParamDict['ColorMap'] = self.cmapvar.get()
            if self.parent.MainParamDict['ColorMap'] in self.parent.cmaps_with_green:
                self.parent.ion_color = "#{0:02x}{1:02x}{2:02x}".format(int(np.round(new_cmaps.cmaps['plasma'](0.55)[0]*255)), int(np.round(new_cmaps.cmaps['plasma'](0.55)[1]*255)), int(np.round(new_cmaps.cmaps['plasma'](0.55)[2]*255)))
                self.parent.electron_color ="#{0:02x}{1:02x}{2:02x}".format(int(np.round(new_cmaps.cmaps['plasma'](0.8)[0]*255)), int(np.round(new_cmaps.cmaps['plasma'](0.8)[1]*255)), int(np.round(new_cmaps.cmaps['plasma'](0.8)[2]*255)))

                self.parent.ion_fit_color = 'r'
                self.parent.electron_fit_color = 'yellow'

            else:
                self.parent.ion_color = "#{0:02x}{1:02x}{2:02x}".format(int(np.round(new_cmaps.cmaps['viridis'](0.45)[0]*255)), int(np.round(new_cmaps.cmaps['viridis'](0.45)[1]*255)), int(np.round(new_cmaps.cmaps['viridis'](0.45)[2]*255)))
                self.parent.electron_color ="#{0:02x}{1:02x}{2:02x}".format(int(np.round(new_cmaps.cmaps['viridis'](0.75)[0]*255)), int(np.round(new_cmaps.cmaps['viridis'](0.75)[1]*255)), int(np.round(new_cmaps.cmaps['viridis'](0.75)[2]*255)))

                self.parent.ion_fit_color = 'mediumturquoise'
                self.parent.electron_fit_color = 'lime'


            self.parent.RenewCanvas(ForceRedraw = True)

    def DivCmapChanged(self, *args):
    # Note here that Tkinter passes an event object to onselect()
        if self.div_cmapvar.get() == self.parent.MainParamDict['DivColorMap']:
            pass
        else:
            self.parent.MainParamDict['DivColorMap'] = self.div_cmapvar.get()
            self.parent.RenewCanvas(ForceRedraw = True)


    def SkipSizeChanged(self, *args):
    # Note here that Tkinter passes an event object to SkipSizeChange()
        try:
            if self.skipSize.get() == '':
                pass
            else:
                self.parent.MainParamDict['SkipSize'] = int(self.skipSize.get())
        except ValueError:
            self.skipSize.set(self.parent.MainParamDict['SkipSize'])

    def RowNumChanged(self, *args):
        try:
            if self.rowNum.get() == '':
                pass
            if int(self.rowNum.get())<1:
                self.rowNum.set(1)
            if int(self.rowNum.get())>self.parent.MainParamDict['MaxRows']:
                self.rowNum.set(self.parent.MainParamDict['MaxRows'])
            if int(self.rowNum.get()) != self.parent.MainParamDict['NumOfRows']:
                self.parent.MainParamDict['NumOfRows'] = int(self.rowNum.get())
                self.parent.UpdateGridSpec()
        except ValueError:
            self.rowNum.set(self.parent.MainParamDict['NumOfRows'])

    def ColumnNumChanged(self, *args):
        try:
            if self.columnNum.get() == '':
                pass
            if int(self.columnNum.get())<1:
                self.columnNum.set(1)
            if int(self.columnNum.get())>self.parent.MainParamDict['MaxCols']:
                self.columnNum.set(self.parent.MainParamDict['MaxCols'])
            if int(self.columnNum.get()) != self.parent.MainParamDict['NumOfCols']:
                self.parent.MainParamDict['NumOfCols'] = int(self.columnNum.get())
                self.parent.UpdateGridSpec()

        except ValueError:
            self.columnNum.set(self.parent.MainParamDict['NumOfCols'])

    def WaitTimeChanged(self, *args):
    # Note here that Tkinter passes an event object to onselect()
        try:
            if self.waitTime.get() == '':
                pass
            else:
                self.parent.MainParamDict['WaitTime'] = float(self.waitTime.get())
        except ValueError:
            self.waitTime.set(self.parent.MainParamDict['WaitTime'])

    def CheckIfLimsChanged(self):
        to_reload = False
        tmplist = [self.xleft, self.xright, self.yleft, self.yright, self.kleft, self.kright]
        limkeys = ['xLeft', 'xRight', 'yBottom', 'yTop', 'kLeft', 'kRight']
        setKeys = ['SetxLim', 'SetyLim', 'SetkLim']
        for j in range(6):
            setlims = self.parent.MainParamDict[setKeys[j//2]]
            tmpkey = limkeys[j]

            try:
            #make sure the user types in a a number and that it has changed.
                if np.abs(float(tmplist[j].get()) - self.parent.MainParamDict[tmpkey]) > 1E-4:
                    self.parent.MainParamDict[tmpkey] = float(tmplist[j].get())
                    to_reload += setlims

            except ValueError:
                #if they type in random stuff, just set it ot the param value
                tmplist[j].set(str(self.parent.MainParamDict[tmpkey]))
        return to_reload

    def CheckIfGammaChanged(self):
        to_reload = False
        try:
        #make sure the user types in a float
            if np.abs(float(self.GammaVar.get()) - self.parent.MainParamDict['GammaBoost']) > 1E-8:
                self.parent.MainParamDict['GammaBoost'] = float(self.GammaVar.get())
                to_reload += True

        except ValueError:
            #if they type in random stuff, just set it to the param value
            self.GammaVar.set(str(self.parent.MainParamDict['GammaBoost']))
        return to_reload*self.parent.MainParamDict['DoLorentzBoost']

    def CheckIfStrideChanged(self):
        to_reload = False

        try:
            #make sure the user types in a int
            if int(self.PrtlStrideVar.get()) <= 0:
                self.PrtlStrideVar.set(str(self.parent.MainParamDict['PrtlStride']))
            if int(self.PrtlStrideVar.get()) != self.parent.MainParamDict['PrtlStride']:
                self.parent.MainParamDict['PrtlStride'] = int(self.PrtlStrideVar.get())
                self.parent.stride = self.parent.MainParamDict['PrtlStride']
                self.parent.StrideChanged()
                to_reload += True

        except ValueError:
            #if they type in random stuff, just set it to the param value
            self.PrtlStrideVar.set(str(self.parent.MainParamDict['PrtlStride']))
        return to_reload

    def CheckIfSliceChanged(self):
        to_reload = False
        try:
            #make sure the user types in a float
            self.xSliceVar.set(int(np.around(float(self.xSliceVarC_omp.get())*self.parent.c_omp/self.parent.istep)))
            if int(self.xSliceVar.get()) < 0:
                self.xSliceVar.set(0)

            elif int(self.xSliceVar.get()) > self.parent.MaxXInd:
                self.xSliceVar.set(self.parent.MaxXInd)
            self.xSliceVarC_omp.set(self.units_listx[self.xSliceVar.get()])
            if self.xSliceVar.get() != int(np.around(self.parent.MainParamDict['xSlice']*self.parent.MaxXInd)):
                self.parent.MainParamDict['xSlice'] = float(self.xSliceVar.get())/self.parent.MaxXInd
                self.sliderx.set(self.xSliceVar.get())
                to_reload += True
        except ValueError:
            #if they type in random stuff, just set it to the param value
            self.xSliceVarC_omp.set(self.units_listx[self.xSliceVar.get()])


        try:
            #make sure the user types in a float
            self.ySliceVar.set(int(np.around(float(self.ySliceVarC_omp.get())*self.parent.c_omp/self.parent.istep)))
            if int(self.ySliceVar.get()) < 0:
                self.ySliceVar.set(0)

            elif int(self.ySliceVar.get()) > self.parent.MaxYInd:
                self.ySliceVar.set(self.parent.MaxYInd)
            self.ySliceVarC_omp.set(self.units_listy[self.ySliceVar.get()])
            if self.ySliceVar.get() != int(np.around(self.parent.MainParamDict['ySlice']*self.parent.MaxYInd)):
                self.parent.MainParamDict['ySlice'] = float(self.ySliceVar.get())/self.parent.MaxYInd
                self.slidery.set(self.ySliceVar.get())
                to_reload += True
        except ValueError:
            #if they type in random stuff, just set it to the param value
            self.ySliceVarC_omp.set(self.units_listy[self.ySliceVar.get()])

        try:
            #make sure the user types in a float
            self.zSliceVar.set(int(np.around(float(self.zSliceVarC_omp.get())*self.parent.c_omp/self.parent.istep)))
            if int(self.zSliceVar.get()) < 0:
                self.zSliceVar.set(0)

            elif int(self.zSliceVar.get()) > self.parent.MaxZInd:
                self.zSliceVar.set(self.parent.MaxZInd)
            self.zSliceVarC_omp.set(self.units_listz[self.zSliceVar.get()])
            if self.zSliceVar.get() != int(np.around(self.parent.MainParamDict['zSlice']*self.parent.MaxZInd)):
                self.parent.MainParamDict['zSlice'] = float(self.zSliceVar.get())/self.parent.MaxZInd
                self.sliderz.set(self.zSliceVar.get())
                to_reload += True

        except ValueError:
            #if they type in random stuff, just set it to the param value
            self.zSliceVarC_omp.set(self.units_listz[self.zSliceVar.get()])
        return to_reload


    def LimChanged(self, *args):
        if self.LimVar.get()==self.parent.MainParamDict['SetxLim']:
            pass
        else:
            self.parent.MainParamDict['SetxLim'] = self.LimVar.get()
            self.parent.RenewCanvas()

    def yLimChanged(self, *args):
        if self.yLimVar.get()==self.parent.MainParamDict['SetyLim']:
            pass
        else:
            self.parent.MainParamDict['SetyLim'] = self.yLimVar.get()
            self.parent.RenewCanvas()

    def kLimChanged(self, *args):
        if self.kLimVar.get()==self.parent.MainParamDict['SetkLim']:
            pass
        else:
            self.parent.MainParamDict['SetkLim'] = self.kLimVar.get()
            self.parent.RenewCanvas()


    def SettingsCallback(self, e):
        to_reload = self.CheckIfLimsChanged()
        to_reload += self.CheckIfGammaChanged()
        to_reload += self.CheckIfStrideChanged()
        to_reload += self.CheckIfSliceChanged()
        if to_reload:
            self.parent.RenewCanvas()



    def OnClosing(self):
        self.parent.settings_window = None
        self.destroy()

class MeasureFrame(Tk.Toplevel):
    def __init__(self, parent):

        Tk.Toplevel.__init__(self)
        self.wm_title('Take Measurements')
        self.protocol('WM_DELETE_WINDOW', self.OnClosing)


        self.parent = parent

        self.bind('<Return>', self.TxtEnter)
        frm = ttk.Frame(self)
        frm.pack(fill=Tk.BOTH, expand=True)


        ttk.Label(frm, text='NOTE: Spectral Measurements have been moved ' +'\r' + 'to the spectral subplot settings window.').grid(row = 0, rowspan = 2,columnspan = 3, sticky = Tk.W)



        # Make an entry to change the integration region
        # A StringVar for a box to type in a value for the left ion region
        self.FFTLVar = Tk.StringVar()
        # set it to the left value
        self.FFTLVar.set(str(self.parent.MainParamDict['FFTLeft']))

        # A StringVar for a box to type in a value for the right ion region
        self.FFTRVar = Tk.StringVar()
        # set it to the right value
        self.FFTRVar.set(str(self.parent.MainParamDict['FFTRight']))

        ttk.Label(frm, text='left').grid(row = 2, column = 1, sticky = Tk.N)
        ttk.Label(frm, text='right').grid(row = 2, column = 2, sticky = Tk.N)

        ttk.Label(frm, text='FFT region:').grid(row = 3, sticky = Tk.W)
        ttk.Entry(frm, textvariable=self.FFTLVar, width=7).grid(row =3, column = 1, sticky = Tk.W + Tk.E)

        ttk.Entry(frm, textvariable=self.FFTRVar, width=7).grid(row = 3, column =2, sticky = Tk.W + Tk.E)

        self.FFTRelVar = Tk.IntVar()
        self.FFTRelVar.set(self.parent.MainParamDict['FFTRelative'])
        self.FFTRelVar.trace('w', self.FFTRelChanged)
        cb = ttk.Checkbutton(frm, text = "FFT Region relative to shock?",
                        variable = self.FFTRelVar)
        cb.grid(row = 4, columnspan = 3, sticky = Tk.W)


    def CheckIfFloatChanged(self, tkVar, paramKey):
        to_reload = False
        try:
            #make sure the user types in a int
            if np.abs(float(tkVar.get())- self.parent.MainParamDict[paramKey])>1E-6:
                self.parent.MainParamDict[paramKey] = float(tkVar.get())
                to_reload = True
            return to_reload

        except ValueError:
            #if they type in random stuff, just set it to the param value
            tkVar.set(str(self.parent.MainParamDict[paramKey]))
            return to_reload

    def TxtEnter(self, e):
        self.MeasuresCallback()


    def FFTRelChanged(self, *args):
        if self.FFTRelVar.get()==self.parent.MainParamDict['FFTRelative']:
            pass
        else:
            self.parent.MainParamDict['FFTRelative'] = self.FFTRelVar.get()
            self.parent.RenewCanvas()



    def MeasuresCallback(self):
        tkvarIntList = [self.FFTLVar, self.FFTRVar]
        IntValList = ['FFTLeft', 'FFTRight']

        to_reload = False

        for j in range(len(tkvarIntList)):
            to_reload += self.CheckIfFloatChanged(tkvarIntList[j], IntValList[j])

        if to_reload:
            self.parent.RenewCanvas()

    def OnClosing(self):
        self.parent.measure_window = None
        self.destroy()



class SettingsDock(QtWidgets.QWidget):
    '''The window beside the figure that the settings panes open in, one tab
    each. Opening a pane that is already open brings its tab to the front.
    It is always its own top-level window, so opening or closing it never
    resizes (and redraws) the figure.'''

    HINT = ('Right-click a panel to change its settings.\n\n'
            'S opens the general settings.')

    def __init__(self, window):
        QtWidgets.QWidget.__init__(self, window, Qt.Window)
        self.setObjectName('SettingsDock')
        self.setWindowTitle('Settings')
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.setElideMode(Qt.ElideRight)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.hint = QtWidgets.QLabel(self.HINT)
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet('color: gray; padding: 24px;')
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.hint)
        self.stack.addWidget(self.tabs)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)
        self._pages = {}
        self._placed = False
        self._toggle_action = QtGui.QAction('Settings Panel', self)
        self._toggle_action.setCheckable(True)
        self._toggle_action.toggled.connect(self.setVisible)

    def toggleViewAction(self):
        return self._toggle_action

    def showEvent(self, event):
        if not self._placed:
            # First time up: open it just to the right of the main window
            self._placed = True
            window = self.parent()
            geom = window.frameGeometry()
            self.resize(max(self.width(), 360), window.height())
            self.move(geom.right() + 1, geom.top())
        self._toggle_action.setChecked(True)
        QtWidgets.QWidget.showEvent(self, event)

    def hideEvent(self, event):
        self._toggle_action.setChecked(False)
        QtWidgets.QWidget.hideEvent(self, event)

    def _page_of(self, top):
        return self._pages.get(id(top))

    def add(self, top):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(top.qw)
        scroll._top = top
        self._pages[id(top)] = scroll
        index = self.tabs.addTab(scroll, 'Settings')
        self.tabs.setCurrentIndex(index)
        self.stack.setCurrentWidget(self.tabs)
        was_hidden = not self.isVisible()
        self.show()
        self.raise_()
        # Fit the dock to the pane once it is laid out
        QtCore.QTimer.singleShot(0, lambda: self._fit(scroll, grow_only = not was_hidden))

    def _fit(self, scroll, grow_only):
        try:
            want = scroll.widget().sizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 8
        except RuntimeError:
            return
        if grow_only and want <= self.width():
            return
        self.resize(want, self.height())

    def set_title(self, top, title):
        scroll = self._page_of(top)
        if scroll is not None:
            index = self.tabs.indexOf(scroll)
            self.tabs.setTabText(index, title.replace(' Settings', '').replace('&', '&&'))
            self.tabs.setTabToolTip(index, title)

    def show_tab(self, top):
        scroll = self._page_of(top)
        if scroll is not None:
            self.tabs.setCurrentWidget(scroll)
            self.show()

    def remove(self, top):
        scroll = self._pages.pop(id(top), None)
        if scroll is None:
            return
        index = self.tabs.indexOf(scroll)
        if index >= 0:
            self.tabs.removeTab(index)
        scroll.takeWidget()
        scroll.deleteLater()
        if self.tabs.count() == 0:
            self.stack.setCurrentWidget(self.hint)

    def _close_tab(self, index):
        scroll = self.tabs.widget(index)
        top = getattr(scroll, '_top', None)
        if top is not None:
            top._request_close()


class MainWindow(QtWidgets.QMainWindow):
    '''The Qt window that MainApp drives.'''

    def __init__(self, app):
        QtWidgets.QMainWindow.__init__(self)
        self.app = app

    def closeEvent(self, event):
        event.accept()
        QtWidgets.QApplication.instance().quit()


class MainApp:
    """ The main app of Iseult. It drives the Qt window in self.window, and
    provides the few Tk-style calls (after, winfo_width, ...) that the
    panels and older code use."""

    def __init__(self, name, cmd_args):
        self.window = MainWindow(self)
        self.window.setWindowTitle(name)
        self.settings_window = None
        self.measure_window = None
        self.preset_window = None

        # Time steps taken from the playback bar are drawn once Qt is idle;
        # see setKnob and StepInteractively.
        self._defer_step_render = False
        self._pending_step_render = None

        self.cmd_args = cmd_args
        # Maps ('horiz'|'vert', physical axis) onto the axes that every later
        # panel with that combination shares its limits with. Rebuilt on every
        # redraw by ReDrawCanvas.
        self.shared_axes = {}

        # An int that stores the current stride
        self.stride = 0

        self.IseultDir = os.path.join(os.path.dirname(__file__),'..')

        # a list of cmaps with orange prtl colors
        self.cmaps_with_green = ['viridis', 'Rainbow + White', 'Blue/Green/Red/Yellow', 'Cube YF', 'Linear_L']

        # The settings panes open as tabs of their own window beside the figure
        self.settings_dock = SettingsDock(self.window)
        Tk.set_window_host(self.settings_dock)

        # A list that will keep track of whether a given axes is a colorbar or not:
        self.cbarList = []

        # The dictionary that holdsd the paths
        self.PathDict = {'Flds': [], 'Prtl': [], 'Param': [], 'Spect': []}

        # A dictionary that allows use to see in what HDF5 file each key is stored.
        # i.e. {'ui': 'Prtl', 'ue': 'Flds', etc...},  Originally I generated the
        # key dictionary automatically, but I don't think that is safe anymore.

        self.H5KeyDict = {u'mx0': 'Param',
                          u'teststarti': 'Param',
                          u'teststartl': 'Param',
                          u'sizex': 'Param',
                          u'sizey': 'Param',
                          u'c_omp': 'Param',
                          u'qi': 'Param',
                          u'istep1': 'Param',
                          u'my0': 'Param',
                          u'dlapion': 'Param',
                          u'testendion': 'Param',
                          u'caseinit': 'Param',
                          u'pltstart': 'Param',
                          u'stride': 'Param',
                          u'ntimes': 'Param',
                          u'cooling': 'Param',
                          u'btheta': 'Param',
                          u'c': 'Param',
                          u'acool': 'Param',
                          u'istep': 'Param',
                          u'delgam': 'Param',
                          u'me': 'Param',
                          u'dlaplec': 'Param',
                          u'mi': 'Param',
                          u'torqint': 'Param',
                          u'mx': 'Param',
                          u'mz0': 'Param',
                          u'yi': 'Prtl',
                          u'proci': 'Prtl',
                          u'proce': 'Prtl',
                          u'ye': 'Prtl',
                          u'zi': 'Prtl',
                          u'ze': 'Prtl',
                          u'xsl': 'Spect',
                          u'umean': 'Spect',
                          u'spece': 'Spect',
                          u'v3xi': 'Flds',
                          u'ey': 'Flds',
                          u'ex': 'Flds',
                          u'ez': 'Flds',
                          u'specp': 'Spect',
                          u'densi': 'Flds',
                          u'specprest': 'Spect',
                          u'we': 'Prtl',
                          u'jx': 'Flds',
                          u'jy': 'Flds',
                          u'jz': 'Flds',
                          u'gmax': 'Spect',
                          u'gmin': 'Spect',
                          'spect_dens': 'Spect',
                          u'wi': 'Prtl',
                          u'bx': 'Flds',
                          u'by': 'Flds',
                          u'bz': 'Flds',
                          u'dgam': 'Spect',
                          u'gamma': 'Spect',
                          u'xi': 'Prtl',
                          u'xe': 'Prtl',
                          u'che': 'Prtl',
                          u'chi': 'Prtl',
                          u'ui': 'Prtl',
                          u'ue': 'Prtl',
                          u've': 'Prtl',
                          u'gamma0': 'Param',
                          u'vi': 'Prtl',
                          u'my': 'Param',
                          u'specerest': 'Spect',
                          u'v3yi': 'Flds',
                          u'walloc': 'Param',
                          u'testendlec': 'Param',
                          u'v3x': 'Flds',
                          u'v3y': 'Flds',
                          u'v3z': 'Flds',
                          u'xinject2': 'Param',
                          u'gammae': 'Prtl',
                          u'bphi': 'Param',
                          u'gammai': 'Prtl',
                          u'dummy': 'Param',
                          u'dens': 'Flds',
                          u'sigma': 'Param',
                          u'interval': 'Param',
                          u'inde': 'Prtl',
                          u'v3zi': 'Flds',
                          u'time': 'Param',
                          u'splitratio': 'Param',
                          u'indi': 'Prtl',
                          u'divE': 'Flds',
                          u'ppc0': 'Param'}
        self.prtl_keys = []
        for k, v in self.H5KeyDict.items():
            if v =='Prtl':
                self.prtl_keys.append(k)

        # Create the figure
        self.f = Figure(figsize = (2,2), dpi = 100, edgecolor = 'none', facecolor = 'w')
        self.canvas = IseultCanvas(self.f)
        self.window.setCentralWidget(self.canvas)

        self.GenMainParamDict()
        self.geometry(self.MainParamDict['WindowSize'])

        if self.MainParamDict['HorizontalCbars']:
            self.axes_extent = self.MainParamDict['HAxesExtent']
            self.cbar_extent = self.MainParamDict['HCbarExtent']
            self.SubPlotParams = self.MainParamDict['HSubPlotParams']

        else:
            self.axes_extent = self.MainParamDict['VAxesExtent']
            self.cbar_extent = self.MainParamDict['VCbarExtent']
            self.SubPlotParams = self.MainParamDict['VSubPlotParams']
        self.f.subplots_adjust( **self.SubPlotParams)

        # Make the object hold the timestep info
        self.TimeStep = Param(1, minimum=1, maximum=1000)
        self.playbackbar = PlaybackBar(self, self.TimeStep)
        self.window.addToolBar(Qt.BottomToolBarArea, self.playbackbar)

        # Add the toolbar
        self.toolbar = MyCustomToolbar(self.canvas, self, self.window)
        self.window.addToolBar(Qt.TopToolBarArea, self.toolbar)

        self.BuildMenus()
        self.BuildShortcuts()

        # Some options to set the way the spectral lines are dashed
        self.dashes_options = [[],[3,1],[5,1],[1,1]]
        # Look for the tristan output files and load the file paths into
        # previous objects
        self.dirname = os.curdir
        if len(self.cmd_args.O[0])>0:
            self.dirname = os.path.join(self.dirname, self.cmd_args.O[0])

        self.findDir()

        self.TimeStep.attach(self)
        self.window.show()
        # Lay the window out before the first draw, so it is drawn at its real size
        QtWidgets.QApplication.processEvents()
        self.InitializeCanvas()

        if self.cmd_args.b :
            self.MakeAMovie('out.mov', 1, -1, 1, 10)
            sys.exit(0)

    ####
    #
    # Menus and keys
    #
    ####

    def BuildMenus(self):
        menubar = self.window.menuBar()
        def add(menu, text, slot, shortcut=None):
            action = menu.addAction(text)
            action.triggered.connect(lambda checked=False: slot())
            if shortcut is not None:
                action.setShortcut(QtGui.QKeySequence(shortcut))
            return action

        fileMenu = menubar.addMenu('&File')
        add(fileMenu, 'Open Directory…', self.OnOpen, 'Ctrl+O')
        add(fileMenu, 'Save Current State…', self.OpenSaveDialog, 'Ctrl+S')
        add(fileMenu, 'Make a Movie…', self.OpenMovieDialog, 'Ctrl+M')
        add(fileMenu, 'Reset Session', self.ResetSession)
        fileMenu.addSeparator()
        add(fileMenu, 'Exit', self.quit, 'Ctrl+Q')

        viewMenu = menubar.addMenu('&View')
        add(viewMenu, 'General Settings', self.OpenSettings)
        add(viewMenu, 'FFT Region', self.playbackbar.OpenMeasures)
        dock_action = self.settings_dock.toggleViewAction()
        dock_action.setText('Settings Panel')
        dock_action.setShortcut(QtGui.QKeySequence('Ctrl+E'))
        viewMenu.addAction(dock_action)

        self.presetMenu = menubar.addMenu('&Preset Views')
        self.presetMenu.aboutToShow.connect(self.ViewUpdate)

    def BuildShortcuts(self):
        # Window-wide keys. A text box that uses the key itself (e.g. the
        # arrows while typing) keeps it; Qt only fires these otherwise.
        for key, command in [(Qt.Key_Left, self.playbackbar.SkipLeft),
                             (Qt.Key_Right, self.playbackbar.SkipRight),
                             (Qt.Key_Space, self.playbackbar.PlayHandler),
                             (Qt.Key_R, self.playbackbar.OnReload),
                             (Qt.Key_S, self.OpenSettings)]:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.window)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.setAutoRepeat(key in (Qt.Key_Left, Qt.Key_Right))
            shortcut.activated.connect(command)

    ####
    #
    # The Tk-style calls used around Iseult
    #
    ####

    def after(self, ms, func=None, *args):
        if func is None:
            return None
        return Tk.after(ms, func, *args)

    def after_idle(self, func, *args):
        return Tk.after_idle(func, *args)

    def after_cancel(self, timer_id):
        Tk.after_cancel(timer_id)

    def update(self):
        QtWidgets.QApplication.processEvents()

    update_idletasks = update

    def winfo_width(self):
        return self.window.width()

    def winfo_height(self):
        return self.window.height()

    def geometry(self, size):
        m = re.match(r'\s*(\d+)x(\d+)', str(size))
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            screen = self.window.screen().availableGeometry()
            self.window.resize(min(w, screen.width()), min(h, screen.height()))

    def focus_set(self):
        self.canvas.setFocus()

    def mainloop(self):
        QtWidgets.QApplication.instance().exec()

    def ViewUpdate(self):
        # Rebuilt every time the menu opens so it follows renames, deletions
        # and reordering done in the PresetManager.
        config_dir = os.path.join(self.IseultDir, '.iseult_configs')
        self.presetMenu.clear()
        for name, cfile in preset_views.list_presets(config_dir):
            action = self.presetMenu.addAction(name)
            action.triggered.connect(partial(lambda path, checked=False: self.LoadConfig(path), os.path.join(config_dir, cfile)))
        self.presetMenu.addSeparator()
        self.presetMenu.addAction('Manage Presets…').triggered.connect(lambda checked=False: self.OpenPresetManager())
    def StrideChanged(self):
        # first we have to remove the calculated energy time steps
        self.TotalEnergyTimeSteps = []
        self.TotalEnergyTimes = np.array([])
        self.TotalIonEnergy = np.array([])
        self.TotalElectronEnergy = np.array([])

        self.TotalMagEnergy = np.array([])
        self.TotalBxEnergy = np.array([])
        self.TotalByEnergy = np.array([])
        self.TotalBzEnergy = np.array([])

        self.TotalExEnergy = np.array([])
        self.TotalEyEnergy = np.array([])
        self.TotalEzEnergy = np.array([])

        self.TotalElectricEnergy = np.array([])

        # figure out all keys that have 'Prtl'
        # now we have to go through the data dictionary and remove the particle info
        for DataDict in self.ListOfDataDict:
            for k in self.prtl_keys:
                DataDict.pop(k, None)


    def quit(self, *args):
        print("quitting...")
        QtWidgets.QApplication.instance().quit()

    def GenMainParamDict(self, config_file = None):
        ''' The function that reads in a config file and then makes MainParamDict to hold all of the main iseult parameters.
            It also sets all of the plots parameters.'''

        #config = configparser.RawConfigParser()

        if config_file is None:
            try:
                with open(os.path.join(self.IseultDir, '.iseult_configs', self.cmd_args.p.strip().replace(' ', '_') +'.yml')) as f:
                    cfgDict = yaml.safe_load(f)
            except:
                print('Cannot find/load ' +  self.cmd_args.p.strip().replace(' ', '_') +'.yml in .iseult_configs. If the name of view contains whitespace,')
                print('either it must be enclosed in quotation marks or given with whitespace replaced by _.')
                print('Name is case sensitive. Reverting to Default view')
                with open(os.path.join(self.IseultDir, '.iseult_configs', 'Default.yml')) as f:
                    cfgDict = yaml.safe_load(f)
        else:
            with open(config_file) as f:
                cfgDict = yaml.safe_load(f)

        # Since configparser reads in strings we have to format the data.
        # First create MainParamDict with the default parameters,
        # the dictionary that will hold the parameters for the program.
        # See ./iseult_configs/Default.cfg for a description of what each parameter does.
        self.MainParamDict = {'zSlice': 0.0, # THIS IS A float WHICH IS THE RELATIVE POSITION OF THE 2D SLICE 0->1
                              '2DSlicePlane': 0, # 0 = x-y plane, 1 == x-z plane, 2 == y-z plane
                              'Average1D': 0,
                              'ySlice': 0.5, # THIS IS A FLOAT WHICH IS THE RELATIVE POSITION OF THE 1D SLICE 0->1
                              'xSlice': 0.5, # THIS IS A FLOAT WHICH IS THE RELATIVE POSITION OF THE 1D SLICE 0->1
                              'WindowSize': '1200x700',
                              'yTop': 100.0,
                              'yBottom': 0.0,
                              'Reload2End': True,
                              'ColorMap': 'viridis',
                              'FFTLeft': 0.0,
                              'ShowTitle': True,
                              'ImageAspect': 0,
                              'WaitTime': 0.01,
                              'MaxCols': 8,
                              'VAxesExtent': [4, 90, 0, 92],
                              'kRight': 1.0,
                              'DoLorentzBoost': False,
                              'NumOfRows': 3,
                              'MaxRows': 8,
                              'SetkLim': False,
                              'VCbarExtent': [4, 90, 94, 97],
                              'SkipSize': 5,
                              'xLeft': 0.0,
                              'NumFontSize': 11,
                              'AxLabelSize': 11,
                              'FFTRelative': True,
                              'NumOfCols': 2,
                              'VSubPlotParams': {'right': 0.95,
                                                 'bottom': 0.06,
                                                 'top': 0.93,
                                                 'wspace': 0.23,
                                                 'hspace': 0.15,
                                                 'left': 0.06},
                              'HAxesExtent': [18, 92, 0, -1],
                              'SetyLim': False,
                              'HSubPlotParams': {'right': 0.95,
                                                 'bottom': 0.06,
                                                 'top': 0.91,
                                                 'wspace': 0.15,
                                                 'hspace': 0.3,
                                                 'left': 0.06},
                              'yLabelPad': 0,
                              'cbarLabelPad': 15,
                              'SetxLim': False,
                              'xLimsRelative': False,
                              'ConstantShockVel': True,
                              'xRight': 100.0,
                              'LinkSpatial': 1,
                              'HCbarExtent': [0, 4, 0, -1],
                              'Recording': False,
                              'xLabelPad': 0,
                              'annotateTextSize': 18,
                              'FFTRight': 200.0,
                              'ClearFig': True,
                              'HorizontalCbars': False,
                              'DivColorMap': 'BuYlRd',
                              'LinkK': True,
                              'GammaBoost': 0.0,
                              'kLeft': 0.1,
                              'LoopPlayback': True,
                              'PrtlStride': 5,
                              'electron_color': '#fca636',
                              'electron_fit_color': 'yellow',
                              'ion_color': '#d6556d',
                              'ion_fit_color': 'r',
                              'shock_color': 'w',
                              'FFT_color': 'k',
                              'legendLabelSize':11}
        for key, val in cfgDict['MainParamDict'].items():
            self.MainParamDict[key] = val
        self.electron_color = self.MainParamDict['electron_color']
        self.ion_color = self.MainParamDict['ion_color']
        self.shock_color = self.MainParamDict['shock_color']
        self.ion_fit_color = self.MainParamDict['ion_fit_color']
        self.electron_fit_color = self.MainParamDict['electron_fit_color']
        self.FFT_color = self.MainParamDict['FFT_color']

        # if stride is 0 that means it has only been initialized... set to default
        if self.stride == 0:
            self.stride = self.MainParamDict['PrtlStride']

    def SaveIseultState(self, cfgfile, cfgname):
        #config = configparser.RawConfigParser()

        # When adding sections or items, add them in the reverse order of
        # how you want them to be displayed in the actual file.
        # In addition, please note that using RawConfigParser's and the raw
        # mode of ConfigParser's respective set functions, you can assign
        # non-string values to keys internally, but will receive an error
        # when attempting to write to a file or when you get it in non-raw
        # mode. SafeConfigParser does not allow such assignments to take place.
        #config.add_section('general')
        cfgDict = {}
        #config.set('general', 'ConfigName', cfgname)

        #config.add_section('main')
        cfgDict['general'] = {'ConfigName': cfgname}
        #DictList = ['HSubPlotParams', 'VSubPlotParams']
        #IntListsList = ['HAxesExtent', 'HCbarExtent', 'VAxesExtent', 'VCbarExtent']

        # Update the 'WindowSize' attribute to the current window size
        self.MainParamDict['WindowSize'] = str(self.winfo_width())+'x'+str(self.winfo_height())
        # Get figsize and dpi

        self.MainParamDict['FigSize'] = [float(self.f.get_size_inches()[0]),float(self.f.get_size_inches()[1])]

        #print(self.MainParamDict['FigSize'], type(self.MainParamDict['FigSize']))
        self.MainParamDict['dpi'] = self.f.dpi
        #print(self.MainParamDict['FigSize'], self.MainParamDict['dpi'] )
        # Update the current subplot params


        tmp_param_str = 'HSubPlotParams' if self.MainParamDict['HorizontalCbars'] else 'VSubPlotParams'
        try:
            self.MainParamDict[tmp_param_str]['left']=float(self.f.subplotpars.left)
            self.MainParamDict[tmp_param_str]['right']=float(self.f.subplotpars.right)
            self.MainParamDict[tmp_param_str]['top']=float(self.f.subplotpars.top)
            self.MainParamDict[tmp_param_str]['bottom']=float(self.f.subplotpars.bottom)
            self.MainParamDict[tmp_param_str]['wspace']=float(self.f.subplotpars.wspace)
            self.MainParamDict[tmp_param_str]['hspace']=float(self.f.subplotpars.hspace)

        except:
            pass
        cfgDict['MainParamDict'] = self.MainParamDict
        cfgDict['MainParamDict']['electron_color'] = self.electron_color
        cfgDict['MainParamDict']['ion_color'] = self.ion_color
        cfgDict['MainParamDict']['shock_color'] = self.shock_color
        cfgDict['MainParamDict']['ion_fit_color'] = self.ion_fit_color
        cfgDict['MainParamDict']['electron_fit_color'] = self.electron_fit_color
        cfgDict['MainParamDict']['FFT_color'] = self.FFT_color
        self.SaveLLoc()
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                tmp_str = 'Chart' + str(i) + '_' + str(j)
                #config.add_section(tmp_str)
                tmp_ctype = self.SubPlotList[i][j].chartType
                #$config.set(tmp_str, 'ChartType', tmp_ctype)
                #for key in self.SubPlotList[i][j].PlotParamsDict[tmp_ctype].keys():
                    #config.set(tmp_str, key, str(s[key]))
                cfgDict[tmp_str] = self.SubPlotList[i][j].PlotParamsDict[tmp_ctype]
                cfgDict[tmp_str]['ChartType'] = tmp_ctype

        #print(yaml.dump(cfgDict))
        # Writing our configuration file to 'example.cfg'

        with open(cfgfile, 'w') as cfgFile:
            cfgFile.write(yaml.safe_dump(cfgDict))

    def CheckMaxNPopUp(self):
        MaxNDialog(self)

    def __findFilesInSubdir(self, file_glob: str, subdir: str) -> list[pathlib.Path]:
        file_list = list(self.dirname.glob(file_glob))
        if len(file_list) == 0:
            file_list = list((self.dirname/subdir).glob(file_glob))

        return file_list

    def checkAndFindFilePaths(self, reload_mode: bool = False) -> bool:
        # Convert to pathlib.Path
        self.dirname = pathlib.Path(self.dirname)

        # Check for "output" directory and update path if it's found
        if (self.dirname/'output').is_dir():
            self.dirname /= 'output'

        # Check if data is from Tristan v1 or v2 and set file names accordingly
        if len(list(self.dirname.glob("param.*"))) > 0:
            # This is Tristan v1 data
            param_name = 'param'
            spectra_name = 'spect'
        elif len(list(self.dirname.glob("params.*"))) > 0:
            # This is Tristan v2 data
            param_name = 'params'
            spectra_name = 'spec'
        else:
            # This directory does not contain any Tristan files
            return False

        # Load each list of files, searching in subdirectories if needed
        param_files    = list(self.dirname.glob(f"{param_name}.*"))
        flds_files     = self.__findFilesInSubdir('flds.tot.*', 'flds')
        prtl_files     = self.__findFilesInSubdir('prtl.tot.*', 'prtl')
        spectra_files  = self.__findFilesInSubdir(f'{spectra_name}.*', 'spec')

        # Strip out .xdmf files
        flds_files = [path for path in flds_files if not path.suffix == '.xdmf']

        # Find which files are in all four lists
        intersection = set.intersection(set([filename.suffix for filename in param_files]),
                                        set([filename.suffix for filename in flds_files]),
                                        set([filename.suffix for filename in prtl_files]),
                                        set([filename.suffix for filename in spectra_files]))

        # Check that there is at least one complete set of data. If not then return early
        if len(intersection) == 0:
            return False

        # Limit us to the first N files if the relevant CLI flag is set
        if self.cmd_args.n != -1:
            intersection = sorted(intersection)[:self.cmd_args.n]

        # Reduce file path lists to just the complete datasets, sort, and assign to member variables
        self.PathDict['Param'] = sorted([path for path in param_files   if path.suffix in intersection])
        self.PathDict['Flds']  = sorted([path for path in flds_files    if path.suffix in intersection])
        self.PathDict['Prtl']  = sorted([path for path in prtl_files    if path.suffix in intersection])
        self.PathDict['Spect'] = sorted([path for path in spectra_files if path.suffix in intersection])

        if reload_mode:
            if int(self.cmd_args.n)!=-1:
                self.CheckMaxNPopUp()
        else:
            self.NewDirectory = True
            self.movie_dir = ''

        self.TimeStep.setMax(len(self.PathDict['Flds']))
        self.playbackbar.set_max(len(self.PathDict['Flds']))
        if self.MainParamDict['Reload2End']:
            self.TimeStep.value = len(self.PathDict['Flds'])
        self.playbackbar.show_step(self.TimeStep.value)
        self.shock_finder()

        return True

    def OnOpen(self, e = None):
        """open a file"""

        if self.cmd_args.n != -1:
            self.CheckMaxNPopUp()

        tmpdir = filedialog.askdirectory(title = 'Choose the directory of the output files', **self.dir_opt)
        if tmpdir == '':
            self.findDir()

        else:
            self.dirname = tmpdir
        if not self.checkAndFindFilePaths():
#            p = MyDalog(self, 'Directory must contain either the output directory or all of the following: \n flds.tot.*, ptrl.tot.*, params.*, spect.*', title = 'Cannot find output files')
#            self.wait_window(p.top)
            self.findDir()
        else:
            self.ReDrawCanvas()

    def ResetSession(self, e = None):
        """open a file"""
        if int(self.cmd_args.n) != -1:
            self.CheckMaxNPopUp()
        self.checkAndFindFilePaths()
        self.ReDrawCanvas()

    def findDir(self, dlgstr = 'Choose the directory of the output files.'):
        """Look for /ouput folder, where the simulation results are
        stored. If output files are already in the path, they are
        automatically loaded"""
        # defining options for opening a directory
        self.dir_opt = {}
        self.dir_opt['initialdir'] = os.curdir
        self.dir_opt['mustexist'] = True
        self.dir_opt['parent'] = self

        if not self.checkAndFindFilePaths():
            tmpdir = filedialog.askdirectory(title = dlgstr, **self.dir_opt)
            if tmpdir != '':
                self.dirname = tmpdir
            if not self.checkAndFindFilePaths():
#                p = MyDialog(self, 'Directory must contain either the output directory or all of the following: \n flds.tot.*, ptrl.tot.*, params.*, spect.*', title = 'Cannot find output files')
#                self.wait_window(p.top)
                self.findDir()


    def InitializeCanvas(self, config_file = None):
        '''Initializes the figure, and then packs it into the main window.
        Should only be called once.'''
        if config_file is None:
            try:
                with open(os.path.join(self.IseultDir, '.iseult_configs', self.cmd_args.p.strip().replace(' ', '_') +'.yml')) as f:
                    cfgDict = yaml.safe_load(f)
            except:
                print('Cannot find/load ' +  self.cmd_args.p.strip().replace(' ', '_') +'.yml in .iseult_configs. If the name of view contains whitespace,')
                print('either it must be enclosed in quotation marks or given with whitespace removed.')
                print('Name is case sensitive. Reverting to Default view')
                with open(os.path.join(self.IseultDir, '.iseult_configs', 'Default.yml')) as f:
                    cfgDict = yaml.safe_load(f)
        else:
            with open(config_file) as f:
                cfgDict = yaml.safe_load(f)


        # divy up the figure into a bunch of subplots using GridSpec.
        self.gs0 = gridspec.GridSpec(self.MainParamDict['NumOfRows'],self.MainParamDict['NumOfCols'])

        # Create the list of all of subplot wrappers
        self.SubPlotList = []
        for i in range(self.MainParamDict['MaxRows']):
            tmplist = [SubPlotWrapper(self, figure = self.f, pos=(i,j)) for j in range(self.MainParamDict['MaxCols'])]
            self.SubPlotList.append(tmplist)
        for i in range(self.MainParamDict['MaxRows']):
            for j in range(self.MainParamDict['MaxCols']):
                tmp_str = f"Chart{i}_{j}"
                if tmp_str in cfgDict.keys():
                    tmpchart_type = cfgDict[tmp_str]['ChartType']
                    self.SubPlotList[i][j].SetGraph(tmpchart_type)
                    for key, val in cfgDict[tmp_str].items():
                        self.SubPlotList[i][j].PlotParamsDict[tmpchart_type][key] = val

                else:
                    # The graph isn't specifiedin the config file, just set it equal to phase plots
                    self.SubPlotList[i][j].SetGraph('PhasePlot')


        # Make a list that will hold the previous ctype
        self.MakePrevCtypeList()
        self.ReDrawCanvas()
        self.f.canvas.mpl_connect('button_press_event', self.onclick)
        self.f.canvas.mpl_connect('button_release_event', self.on_release)
        self.f.canvas.mpl_connect('draw_event', self.on_draw)

    def LoadConfig(self, config_file):
        # First get rid of any & all pop up windows:
        if self.settings_window is not None:
            self.settings_window.destroy()
        if self.measure_window is not None:
            self.measure_window.destroy()
        # Go through each sub-plot destroying any pop-up and
        # restoring to default params
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                self.SubPlotList[i][j].RestoreDefaultPlotParams()
                try:
                    self.SubPlotList[i][j].graph.settings_window.destroy()
                except:
                    pass
        # Read in the config file
        #config = configparser.RawConfigParser()
        #config.read(config_file)
        cfgDict = {}
        with open(config_file, 'r') as f:
            cfgDict = yaml.safe_load(f)
        # Generate the Main Param Dict
        self.GenMainParamDict(config_file)

        #Loading a config file may change the stride... watch out!
        if self.stride != self.MainParamDict['PrtlStride']:
            self.stride = self.MainParamDict['PrtlStride']
            self.StrideChanged()
        # Load in all the subplot params
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                tmp_str = 'Chart' + str(i) + '_' + str(j)

                if tmp_str in cfgDict.keys():

                    tmpchart_type = cfgDict[tmp_str]['ChartType']
                    self.SubPlotList[i][j].SetGraph(tmpchart_type)
                    for key, val in cfgDict[tmp_str].items():
                        self.SubPlotList[i][j].PlotParamsDict[tmpchart_type][key] = val
                else:
                    # The graph isn't specified in the config file, just set it equal to a phase plot
                    self.SubPlotList[i][j].SetGraph('PhasePlot')
        # There are a few parameters that need to be loaded separately, mainly in the playbackbar.
        self.playbackbar.set_recording(self.MainParamDict['Recording'])
        self.playbackbar.set_loop(self.MainParamDict['LoopPlayback'])

        # refresh the geometry
        print(self.MainParamDict['WindowSize'])
        self.geometry(self.MainParamDict['WindowSize'])
        if self.MainParamDict['HorizontalCbars']:
            self.axes_extent = self.MainParamDict['HAxesExtent']
            self.cbar_extent = self.MainParamDict['HCbarExtent']
            self.SubPlotParams = self.MainParamDict['HSubPlotParams']

        else:
            self.axes_extent = self.MainParamDict['VAxesExtent']
            self.cbar_extent = self.MainParamDict['VCbarExtent']
            self.SubPlotParams = self.MainParamDict['VSubPlotParams']
        self.f.subplots_adjust( **self.SubPlotParams)
        # refresh the gridspec and re-draw all of the subplots
        self.UpdateGridSpec()

    def UpdateGridSpec(self, *args):
        '''A function that handles updates the gridspec that divides up of the
        plot into X x Y subplots'''
        # To prevent orphaned windows, we have to kill all of the windows of the
        # subplots that are no longer shown.

        for i in range(self.MainParamDict['MaxRows']):
            for j in range(self.MainParamDict['MaxCols']):
                if i < self.MainParamDict['NumOfRows'] and j < self.MainParamDict['NumOfCols']:
                    pass
                elif self.SubPlotList[i][j].graph.settings_window is not None:
                    self.SubPlotList[i][j].graph.settings_window.destroy()

        self.gs0 = gridspec.GridSpec(self.MainParamDict['NumOfRows'],self.MainParamDict['NumOfCols'])
        self.RenewCanvas(keep_view = False, ForceRedraw = True)

    def LoadAllKeys(self):
        ''' A function that will find out will arrays need to be loaded for
        to draw the graphs. Then it will save all the data necessary to
        If the time hasn't changed, it will only load new keys.'''
        # Make a dictionary that stores all of the keys we will need to load
        # to draw the graphs.
        self.ToLoad = {'Flds': [], 'Prtl': [], 'Param': [], 'Spect': []}
        # we always load time because it is needed to calculate the shock location
        self.ToLoad[self.H5KeyDict['time']].append('time')
        # We always load enough to calculate xmin, xmax, ymin, ymax & the cpu domains:
        self.ToLoad[self.H5KeyDict['c_omp']].append('c_omp')
        self.ToLoad[self.H5KeyDict['istep']].append('istep')
        self.ToLoad[self.H5KeyDict['dens']].append('dens')
        self.ToLoad[self.H5KeyDict['mx']].append('mx')
        self.ToLoad[self.H5KeyDict['my']].append('my')
        # look at each subplot and see what is needed
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                # for each subplot, see what keys are needed
                tmpList = self.SubPlotList[i][j].GetKeys()

                for elm in tmpList:
                    # find out what type of file the key is stored in
                    ftype = self.H5KeyDict[elm]
                    # add the key to the list of that file type
                    self.ToLoad[ftype].append(elm)

        # Check to make sure the 2DSlice is OK...
        # Grab c_omp & istep
        filepath = self.PathDict['Param'][self.TimeStep.value-1]
        self.c_omp = data_loading.load_dataset(filepath, 'c_omp', slice(0,1))
        self.istep = data_loading.load_dataset(filepath, 'istep', slice(0,1))

        # FIND THE SLICE
        filepath = self.PathDict['Flds'][self.TimeStep.value-1]
        bx_shape = data_loading.dataset_shape(filepath, 'bx')
        self.MaxZInd, self.MaxYInd, self.MaxXInd  = np.array(bx_shape) - 1
        self.flds_shape = tuple(bx_shape)

        self.ySlice = int(np.around(self.MainParamDict['ySlice']*self.MaxYInd))
        self.zSlice = int(np.around(self.MainParamDict['zSlice']*self.MaxZInd))
        self.xSlice = int(np.around(self.MainParamDict['xSlice']*self.MaxXInd))

        # See if we are in a new Directory
        if self.NewDirectory:
            # Create a new Dictionary that will have StateHashes of visited steps
            self.SavedHashes = {}
            self.SavedImgStr = {}
            self.SavedImgSize = {}
            self.diff_from_home = []
            self.saved_views = {}
            self.spatial_view = {}


            self.TotalEnergyTimeSteps = []
            self.TotalEnergyTimes = np.array([])
            self.TotalIonEnergy = np.array([])
            self.TotalElectronEnergy = np.array([])

            self.TotalMagEnergy = np.array([])
            self.TotalBxEnergy = np.array([])
            self.TotalByEnergy = np.array([])
            self.TotalBzEnergy = np.array([])

            self.TotalExEnergy = np.array([])
            self.TotalEyEnergy = np.array([])
            self.TotalEzEnergy = np.array([])
            self.TotalElectricEnergy = np.array([])


            # Make a list of timesteps we have already loaded.
            self.timestep_visited = []

            # Timestep queue that ensures that we delete the oldest viewed
            # timestep if memory gets too large
            self.timestep_queue = deque()

            # For each timestep we visit, we will load a dictionary and place it in a list
            self.ListOfDataDict = []

            # Keys found missing in this directory, so each is warned about once
            self.missing_keys_warned = set()

            self.NewDirectory = False
        # see if one of the plots is the total energy panel
        self.showing_total_energy_plt = False
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                self.showing_total_energy_plt += str(self.SubPlotList[i][j].chartType) == 'TotalEnergyPlot'

        if not self.TimeStep.value in self.TotalEnergyTimeSteps:
            if self.showing_total_energy_plt:
                tmp_L = ['ui', 'vi', 'wi', 'ue', 've', 'we', 'mi', 'me', 'stride', 'bx', 'by', 'bz', 'ex', 'ey', 'ez','qi','c']
                for elm in tmp_L:
                    self.ToLoad[self.H5KeyDict[elm]].append(elm)

        if self.TimeStep.value in self.timestep_visited:
            cur_ind = self.timestep_visited.index(self.TimeStep.value)
            # Now the most recently viewed step. Moved in one go, so that a key
            # that fails to load below cannot leave the step visited but out of
            # the queue.
            self.timestep_queue.remove(self.TimeStep.value)
            self.timestep_queue.append(self.TimeStep.value)
            self.DataDict = self.ListOfDataDict[cur_ind]
            for pkey in self.ToLoad.keys():
                tmplist = list(set(self.ToLoad[pkey])) # get rid of duplicate keys
                tmplist2 = np.copy(tmplist)

                # get rid of keys that are already loaded
                for i in range(len(tmplist2)):
                    if tmplist2[i] in self.DataDict.keys():
                        tmplist.remove(tmplist2[i])
                # Now iterate over each path key and create a datadictionary
                filepath = self.PathDict[pkey][self.TimeStep.value-1]
                if len(tmplist)> 0:
                    if pkey =='Prtl': # we load particle arrays with a stride because they are expensive
                        for elm in tmplist:
                            self.DataDict[elm] = data_loading.load_dataset(filepath, elm, slice(None, None, self.MainParamDict['PrtlStride']))
                    else:
                        for elm in tmplist:
                            try:
                                if elm == 'spect_dens':
                                    self.DataDict[elm] = data_loading.load_dataset(filepath, 'dens')
                                else:
                                    self.DataDict[elm] = data_loading.load_dataset(filepath, elm)
                            except KeyError:
                                self.DataDict[elm] = self.MissingKey(filepath, elm)

        else:
            # The time has not already been visited so we have to reload everything
            self.DataDict = {}
            for pkey in self.ToLoad.keys():
                tmplist = list(set(self.ToLoad[pkey])) # get rid of duplicate keys
                # Load the file
                filepath = self.PathDict[pkey][self.TimeStep.value-1]
                if len(tmplist)> 0:
                    if pkey =='Prtl': # we load particle arrays with a stride because they are expensive
                        for elm in tmplist:
                            self.DataDict[elm] = data_loading.load_dataset(filepath, elm, slice(None, None, self.MainParamDict['PrtlStride']))
                    else:
                        for elm in tmplist:
                            try:
                                if elm == 'spect_dens':
                                    self.DataDict[elm] = data_loading.load_dataset(filepath, 'dens')
                                else:
                                    self.DataDict[elm] = data_loading.load_dataset(filepath, elm, cli_args=self.cmd_args)
                            except KeyError:
                                self.DataDict[elm] = self.MissingKey(filepath, elm)

            # don't keep more than 30 time steps in memory because of RAM issues
            if len(self.timestep_visited)>30:
                oldest_time = self.timestep_queue.popleft()
                oldest_ind = self.timestep_visited.index(oldest_time)
                self.timestep_visited.remove(oldest_time)
                self.ListOfDataDict.pop(oldest_ind)
            self.timestep_visited.append(self.TimeStep.value)
            self.ListOfDataDict.append(self.DataDict)
            self.timestep_queue.append(self.TimeStep.value)

        if not self.TimeStep.value in self.TotalEnergyTimeSteps:
            if self.showing_total_energy_plt:
                self.TotalEnergyTimeSteps.append(self.TimeStep.value)
                self.TotalEnergyTimeSteps.sort()
                ind = self.TotalEnergyTimes.searchsorted(self.DataDict['time'])
                self.TotalEnergyTimes = np.append(np.append(self.TotalEnergyTimes[0:ind],self.DataDict['time']),self.TotalEnergyTimes[ind:])

                TotalElectronKE = self.DataDict['ue']*self.DataDict['ue']
                TotalElectronKE += self.DataDict['ve']*self.DataDict['ve']
                TotalElectronKE += self.DataDict['we']*self.DataDict['we']+1
                TotalElectronKE = np.sum(np.sqrt(TotalElectronKE)-1)
                #TotalElectronKE += -len(self.DataDict['we'])

                TotalElectronKE *= self.DataDict['stride']*self.MainParamDict['PrtlStride'] # multiply by the stride.
                TotalElectronKE *= np.abs(self.DataDict['qi'])*self.DataDict['c']**2 # * m_e c^2, mass of particle is its charge, qe/me=1



                TotalIonKE = self.DataDict['ui']*self.DataDict['ui']
                TotalIonKE += self.DataDict['vi']*self.DataDict['vi']
                TotalIonKE += self.DataDict['wi']*self.DataDict['wi']+1
                TotalIonKE = np.sum(np.sqrt(TotalIonKE)-1)
                #TotalIonKE += -len(self.DataDict['we'])

                TotalIonKE *= self.DataDict['stride']*self.MainParamDict['PrtlStride'] # multiply by the stride
                TotalIonKE *= self.DataDict['mi']/self.DataDict['me']*np.abs(self.DataDict['qi'])*self.DataDict['c']**2 #mass of particle is its charge, qe/me=1

                TotalKE = (TotalElectronKE +TotalIonKE)
                # Divide by x size
#                TotalKEDensity *= (self.DataDict['dens'][0,:,:].shape[1]/self.DataDict['c_omp'][0]*self.DataDict['istep'][0])**-1
                # Divide by y size
#                TotalKEDensity *= (self.DataDict['dens'][0,:,:].shape[0]/self.DataDict['c_omp'][0]*self.DataDict['istep'][0])**-1

                self.TotalElectronEnergy = np.append(np.append(self.TotalElectronEnergy[0:ind],TotalElectronKE),self.TotalElectronEnergy[ind:])
                self.TotalIonEnergy = np.append(np.append(self.TotalIonEnergy[0:ind],TotalIonKE),self.TotalIonEnergy[ind:])


                BxEnergy = np.sum(self.DataDict['bx'][:,:,:]*self.DataDict['bx'][:,:,:]) * self.DataDict['istep']**2*.5
                ByEnergy = np.sum(self.DataDict['by'][:,:,:]*self.DataDict['by'][:,:,:]) * self.DataDict['istep']**2*.5
                BzEnergy = np.sum(self.DataDict['bz'][:,:,:]*self.DataDict['bz'][:,:,:]) * self.DataDict['istep']**2*.5
                #TotalBEnergy = BxEnergy + ByEnergy + BzEnergy

                ExEnergy = np.sum(self.DataDict['ex'][:,:,:]*self.DataDict['ex'][:,:,:]) * self.DataDict['istep']**2*.5
                EyEnergy = np.sum(self.DataDict['ey'][:,:,:]*self.DataDict['ey'][:,:,:]) * self.DataDict['istep']**2*.5
                EzEnergy = np.sum(self.DataDict['ez'][:,:,:]*self.DataDict['ez'][:,:,:]) * self.DataDict['istep']**2*.5

                #TotalEEnergy = ExEnergy + EyEnergy + EzEnergy

                # sum over the array and then divide by the number of points len(x)*len(y)
                self.TotalBxEnergy = np.append(np.append(self.TotalBxEnergy[0:ind],BxEnergy), self.TotalBxEnergy[ind:])
                self.TotalByEnergy = np.append(np.append(self.TotalByEnergy[0:ind],ByEnergy), self.TotalByEnergy[ind:])
                self.TotalBzEnergy = np.append(np.append(self.TotalBzEnergy[0:ind],BzEnergy), self.TotalBzEnergy[ind:])
                self.TotalMagEnergy = self.TotalBxEnergy + self.TotalByEnergy + self.TotalBzEnergy

                self.TotalExEnergy = np.append(np.append(self.TotalExEnergy[0:ind],ExEnergy), self.TotalExEnergy[ind:])
                self.TotalEyEnergy = np.append(np.append(self.TotalEyEnergy[0:ind],EyEnergy), self.TotalEyEnergy[ind:])
                self.TotalEzEnergy = np.append(np.append(self.TotalEzEnergy[0:ind],EzEnergy), self.TotalEzEnergy[ind:])
                self.TotalElectricEnergy = self.TotalExEnergy + self.TotalEyEnergy + self.TotalEzEnergy

        if self.MainParamDict['ConstantShockVel']:
            # We can just calculate the time * self.shock_speed
            if np.isnan(self.prev_shock_loc):
                # If self.prev_shock_loc is NaN, that means this is the first time
                # the shock has been found, and the previous and current shock_loc
                # should be the same.

                # First calculate the new shock location
                self.shock_loc = self.DataDict['time']*self.shock_speed
                # Set previous shock loc to current location
                self.prev_shock_loc = np.copy(self.shock_loc)
            else:
                # First save the previous shock location,
                self.prev_shock_loc = np.copy(self.shock_loc)
                # Now calculate the new shock location
                self.shock_loc = self.DataDict['time']*self.shock_speed

        else:
            # Let's see if the shock_loc is in the DataDict
            if not 'shock_loc' in self.DataDict.keys():
                # Have to figure out where the shock is

                # Leave at least one column to look in, however narrow the grid
                jstart = int(min(10*self.DataDict['c_omp']/self.DataDict['istep'], self.DataDict['dens'][0,:,:].shape[1]-1))
                jstart = max(jstart, 0)
                cur_xaxis = np.arange(self.DataDict['dens'][0,:,:].shape[1])/self.DataDict['c_omp']*self.DataDict['istep']
                # Find the shock by seeing where the density is 1/2 of it's
                # max value.

                dens_half_max = max(self.DataDict['dens'][0,:,:][self.DataDict['dens'][0,:,:].shape[0]//2,jstart:])*.5

                # Find the farthest location where the average density is greater
                # than half max
                ishock = np.where(self.DataDict['dens'][0,:,:][self.DataDict['dens'][0,:,:].shape[0]//2,jstart:]>=dens_half_max)[0][-1]
                self.DataDict['shock_loc'] = cur_xaxis[ishock]

            if np.isnan(self.prev_shock_loc):
                # If self.prev_shock_loc is NaN, that means this is the first time
                # the shock has been found, and the previous and current shock_loc
                # should be the same.

                # First calculate the new shock location
                self.shock_loc = self.DataDict['shock_loc']
                # Set previous shock loc to current location
                self.prev_shock_loc = np.copy(self.shock_loc)
            else:
                # First save the previous shock location,
                self.prev_shock_loc = np.copy(self.shock_loc)
                # Now calculate the new shock location
                self.shock_loc = self.DataDict['shock_loc']

        self.cpu_x_locs = np.cumsum(self.DataDict['mx']-5)/self.DataDict['c_omp']
        self.cpu_y_locs = np.cumsum(self.DataDict['my']-5)/self.DataDict['c_omp']
        # Now that the DataDict is created, iterate over all the subplots and
        # load the data into them:
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                self.SubPlotList[i][j].LoadData()

    def MissingKey(self, filepath, elm):
        '''A stand-in for `elm`, which is not in the file at `filepath`.
        Raises the KeyError again if there is no sensible stand-in.'''
        if elm == 'sizex':
            return 1
        elif elm == 'c':
            return 0.45
        elif elm == 'ppc0':
            return np.nan
        elif elm == 'my':
            istep = data_loading.load_dataset(filepath, 'istep', slice(0,1))
            my0   = data_loading.load_dataset(filepath, 'my0', slice(0,1))
            tmpSize = ((self.MaxYInd+1)*istep)//(my0-5)
            return np.ones(tmpSize)*my0
        elif elm == 'mx':
            istep = data_loading.load_dataset(filepath, 'istep', slice(0,1))
            mx0   = data_loading.load_dataset(filepath, 'mx0', slice(0,1))
            tmpSize = ((self.MaxXInd+1)*istep)//(mx0-5)
            return np.ones(tmpSize)*mx0
        elif elm in ['v3x', 'v3y', 'v3z', 'v3xi', 'v3yi', 'v3zi', 'divE']:
            # Shown as zeros, so the panels asking for it can still be drawn
            what = 'divE' if elm == 'divE' else 'velocity'
            if what not in self.missing_keys_warned:
                self.missing_keys_warned.add(what)
                name = 'The divergence of E (divE)' if what == 'divE' else f'The velocity array \'{elm}\''
                message = f'{name} is not in the fields files, so it is shown as zero.'
                print('Warning: ' + message)
                self.after(0, lambda: messagebox.showwarning('Missing Data', message))
            return np.zeros(self.flds_shape)
        raise KeyError(elm)

    def RefreshTimeStep(self):
        ''' A function that will find out will arrays need to be loaded for
        to draw the graphs. Then it will save all the data necessaru to
        If the time hasn't changed, it will only load new keys.'''
        if self.TimeStep.value in self.timestep_visited:
            cur_ind = self.timestep_visited.index(self.TimeStep.value)
            self.timestep_visited.pop(cur_ind)
            self.ListOfDataDict.pop(cur_ind)
            self.timestep_queue.remove(self.TimeStep.value)

        if self.TimeStep.value in self.TotalEnergyTimeSteps:
            self.TotalEnergyTimeSteps.remove(self.TimeStep.value)
            ind = self.TotalEnergyTimes.searchsorted(self.DataDict['time'])
            if ind < len(self.TotalEnergyTimes)-1:
                self.TotalEnergyTimes = np.append(self.TotalEnergyTimes[0:ind],self.TotalEnergyTimes[ind+1:])
                self.TotalElectronEnergy = np.append(self.TotalElectronEnergy[0:ind],self.TotalElectronEnergy[ind+1:])
                self.TotalIonEnergy = np.append(self.TotalIonEnergy[0:ind],self.TotalIonEnergy[ind+1:])
                self.TotalMagEnergy = np.append(self.TotalMagEnergy[0:ind], self.TotalMagEnergy[ind+1:])
                self.TotalBzEnergy = np.append(self.TotalBzEnergy[0:ind], self.TotalBzEnergy[ind+1:])
                self.TotalByEnergy = np.append(self.TotalByEnergy[0:ind], self.TotalByEnergy[ind+1:])
                self.TotalBxEnergy = np.append(self.TotalBxEnergy[0:ind], self.TotalBxEnergy[ind+1:])
                self.TotalEzEnergy = np.append(self.TotalEzEnergy[0:ind], self.TotalEzEnergy[ind+1:])
                self.TotalEyEnergy = np.append(self.TotalEyEnergy[0:ind], self.TotalEyEnergy[ind+1:])
                self.TotalExEnergy = np.append(self.TotalExEnergy[0:ind], self.TotalExEnergy[ind+1:])
                self.TotalElectricEnergy = np.append(self.TotalElectricEnergy[0:ind], self.TotalElectricEnergy[ind+1:])
            else:
                self.TotalEnergyTimes = self.TotalEnergyTimes[0:ind]
                self.TotalElectronEnergy = self.TotalElectronEnergy[0:ind]
                self.TotalIonEnergy = self.TotalIonEnergy[0:ind]
                self.TotalMagEnergy = self.TotalMagEnergy[0:ind]
                self.TotalBxEnergy = self.TotalBxEnergy[0:ind]
                self.TotalByEnergy = self.TotalByEnergy[0:ind]
                self.TotalBzEnergy = self.TotalBzEnergy[0:ind]
                self.TotalExEnergy = self.TotalExEnergy[0:ind]
                self.TotalEyEnergy = self.TotalEyEnergy[0:ind]
                self.TotalEzEnergy = self.TotalEzEnergy[0:ind]
                self.TotalElectricEnergy = self.TotalElectricEnergy[0:ind]

    def MakePrevCtypeList(self):
        self.prev_ctype_list = []
        # The panels as they were just drawn, so the next SaveView can read
        # each one's zoom even if its wrapper has since been given a new graph.
        self.drawn_panels = []
        for i in range(self.MainParamDict['NumOfRows']):
            tmp_ctype_l = []
            for j in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][j]
                tmp_ctype_l.append(str(subplot.chartType))
                axes = getattr(subplot.graph, 'axes', None)
                if axes is None or subplot.draw_failed:
                    continue
                horiz, vert = plot_axes.plot_axes_of(subplot.graph)
                self.drawn_panels.append({'pos': (i, j), 'axes': axes,
                                          'ctype': str(subplot.chartType),
                                          'twoD': bool(subplot.GetPlotParam('twoD')),
                                          'axis_names': (horiz, vert),
                                          'linked': self.ShouldLinkSpatial(subplot)})
            self.prev_ctype_list.append(tmp_ctype_l)

    def SaveLLoc(self):
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][j]
                if subplot.chartType in ('Moments', 'TotalEnergyPlot', 'OhmsLaw', 'PressureBalance'):
                    try:
                        if subplot.graph.legend._get_loc() != 1:
                            subplot.SetPlotParam('legend_loc', ' '.join(str(x) for x in subplot.graph.legend._get_loc()), update_plot = False)
                    except:
                        pass
                if subplot.chartType == 'SpectraPlot':
                    try:
                        if subplot.graph.legDelta._get_loc() != 1:
                            subplot.SetPlotParam('PL_legend_loc', ' '.join(str(x) for x in subplot.graph.legDelta._get_loc()), update_plot = False)
                    except:
                        pass
                    try:
                        if subplot.graph.legT._get_loc() != 2:
                            subplot.SetPlotParam('T_legend_loc', ' '.join(str(x) for x in subplot.graph.legT._get_loc()), update_plot = False)
                    except:
                        pass
    def SetLLoc(self):
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][j]
                if subplot.draw_failed:
                    # It has no legend to place, only an error message.
                    continue
                if subplot.chartType in ('Moments', 'TotalEnergyPlot', 'OhmsLaw', 'PressureBalance'):
                    if subplot.GetPlotParam('legend_loc') != 'N/A':
                        tmp_tup = float(subplot.GetPlotParam('legend_loc').split()[0]),float(subplot.GetPlotParam('legend_loc').split()[1])
                        try:
                            subplot.graph.legend._set_loc(tmp_tup)
                        except AttributeError:
                            pass
                if subplot.chartType == 'SpectraPlot':
                    if subplot.GetPlotParam('T_legend_loc') != 'N/A':
                        tmp_tup = float(subplot.GetPlotParam('T_legend_loc').split()[0]),float(subplot.GetPlotParam('T_legend_loc').split()[1])
                        try:
                            subplot.graph.legT._set_loc(tmp_tup)
                        except:
                            pass
                    if subplot.GetPlotParam('PL_legend_loc') != 'N/A':
                        tmp_tup = float(subplot.GetPlotParam('PL_legend_loc').split()[0]),float(subplot.GetPlotParam('PL_legend_loc').split()[1])
                        try:
                            subplot.graph.legDelta._set_loc(tmp_tup)
                        except:
                            pass

    """
    def FindCbars(self, prev = False):
        ''' A function that will find where all the cbars are in the current view '''
        self.cbarList = []
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                self.IsCbarList.append(False)
                if prev ==True:
                    if self.SubPlotList[i][j].GetPlotParam('twoD') == 1 and not self.SubPlotList[i][j].Changedto2D:
                    # Note the axes still show up in the view if they are set to zero so we have to do it this way.
                        self.IsCbarList.append(True)
                    elif self.SubPlotList[i][j].Changedto1D:
                        self.IsCbarList.append(True)
                elif self.SubPlotList[i][j].GetPlotParam('twoD') == 1:
                    self.IsCbarList.append(True)

    """
    def SaveView(self):
        '''Read out what the user has zoomed into before the figure is renewed.

        Zooms are remembered two ways. self.saved_views holds each panel's
        own limits by its place in the grid, for the axes that are not spatial
        (the value axis of a lineout, the momentum axis of a phase plot) and
        for panels that do not share their spatial axes. self.spatial_view
        holds the zoom in each physical coordinate, taken from the panels
        that do share theirs, so that it carries over to whichever panels
        show that coordinate after the redraw, however the grid or the chart
        types have changed in the meantime.'''
        cur_view = {ax: view for ax, (view, _pos) in self.toolbar._nav_stack().items()}
        self.toolbar._nav_stack.home()
        home_view = {ax: view for ax, (view, _pos) in self.toolbar._nav_stack().items()}

        self.saved_views = {}
        self.spatial_view = {}
        self.diff_from_home = []
        for panel in getattr(self, 'drawn_panels', []):
            ax = panel['axes']
            if ax not in cur_view or ax not in home_view:
                continue
            cur_lims = view_limits(cur_view[ax])
            home_lims = view_limits(home_view[ax])
            is_changed = [home_lims[n] - cur_lims[n] != 0.0 for n in range(4)]
            self.saved_views[panel['pos']] = dict(panel, lims = cur_lims, is_changed = is_changed)

            diff_list = []
            for n in range(4):
                if not is_changed[n]:
                    # A string, so that a zoom whose edge sits exactly on the
                    # shock still hashes differently from no zoom at all.
                    diff_list.append('n/a')
                elif self.MainParamDict['xLimsRelative'] and n < 2:
                    diff_list.append(cur_lims[n]-self.shock_loc)
                else:
                    diff_list.append(cur_lims[n])
            self.diff_from_home.append((panel['pos'], tuple(diff_list)))

            if panel['linked']:
                for slot, axis in zip((0, 2), panel['axis_names']):
                    if axis is not None and (is_changed[slot] or is_changed[slot+1]):
                        self.spatial_view.setdefault(axis, (cur_lims[slot], cur_lims[slot+1]))

    def LoadView(self):
        '''Put the zoom SaveView read out back onto the renewed panels.'''
        self.toolbar.push_current()
        cur_view = {ax: view for ax, (view, _pos) in self.toolbar._nav_stack().items()}
        saved_views = getattr(self, 'saved_views', {})
        spatial_view = getattr(self, 'spatial_view', {})
        # How far a shock-relative x zoom has to move to follow the shock.
        shock_shift = self.MainParamDict['xLimsRelative']*(self.shock_loc-self.prev_shock_loc)

        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][j]
                ax = getattr(subplot.graph, 'axes', None)
                if subplot.draw_failed or ax is None or ax not in cur_view:
                    subplot.Changedto1D = subplot.Changedto2D = False
                    continue
                lims = list(view_limits(cur_view[ax]))
                # Which of the four limits come from the user's own zoom
                # rather than from the panel that was just drawn.
                kept = [False, False, False, False]
                axis_names = plot_axes.plot_axes_of(subplot.graph)

                saved = saved_views.get((i, j))
                try:
                    prev_ctype = self.prev_ctype_list[i][j]
                except (AttributeError, IndexError):
                    prev_ctype = None
                if saved is not None and prev_ctype == subplot.chartType \
                        and saved['ctype'] == subplot.chartType \
                        and saved['twoD'] == bool(subplot.GetPlotParam('twoD')) \
                        and saved['axis_names'] == axis_names:
                    # The same plot as before, so all of its zoom still applies.
                    for n in range(4):
                        if saved['is_changed'][n]:
                            kept[n] = True
                            lims[n] = saved['lims'][n]

                if self.ShouldLinkSpatial(subplot):
                    # A spatial axis shows the viewport the user framed in that
                    # coordinate, whichever panel they framed it in.
                    for slot, axis in zip((0, 2), axis_names):
                        if axis in spatial_view:
                            lims[slot], lims[slot+1] = spatial_view[axis]
                            kept[slot] = kept[slot+1] = True

                for slot, axis in zip((0, 2), axis_names):
                    if axis == 'x':
                        for n in (slot, slot+1):
                            if kept[n]:
                                lims[n] += shock_shift

                if any(kept):
                    ax._set_view(view_from_limits(cur_view[ax], lims, kept))
                subplot.Changedto1D = False
                subplot.Changedto2D = False
        self.toolbar.push_current()

    def RenewCanvas(self, keep_view = True, ForceRedraw = False):

        '''We have two way of updated the graphs: 1) by refreshing them using
        self.RefreshCanvas, we don't recreate all of the artists that matplotlib
        needs to make the plot work. self.RefreshCanvas should be fast. Two we
        can ReDraw the canvas using self.ReDrawCanvas. This recreates all the
        artists and will be slow. Sometimes the graph must be redrawn however,
        if the GridSpec changed, more plots are added, the chartype changed, if
        the plot went from 2d to 1D, etc.. If any change occurs that requires a
        redraw, renewcanvas must be called with ForceRedraw = True. '''

        # This renews the figure at the current time step, which is all a
        # step still waiting to be drawn would have done.
        if self._pending_step_render is not None:
            self.after_cancel(self._pending_step_render)
            self._pending_step_render = None

        self.SaveLLoc()
        if ForceRedraw:
            self.ReDrawCanvas(keep_view = keep_view)
        else:
            self.RefreshCanvas(keep_view = keep_view)
        # Record the current ctypes for later
        self.MakePrevCtypeList()


        # remove some unnecessary data
        tmp_list = ['ui', 'vi', 'wi', 'ue', 've', 'we', 'che', 'chi']
        for elm in tmp_list:
            self.DataDict.pop(elm, None)
        self.SetLLoc()
        # Save the image for quick playback later
        #self.SaveTmpFig()




    def HashIseultState(self):
        ''' A function that saves a hash of the current state of Iseult. Used to
        determine if we can just show a saved image of a previous timeslice,
        or if we must reload it.'''

        #First update the main param dict to save the current window size:
        self.MainParamDict['WindowSize'] = str(self.winfo_width())+'x'+str(self.winfo_height())
        # a tuple that will eventually be hashed.
        state_tuple = self.freeze(self.diff_from_home)
        # keys we should skip over when making the hash.
        SkipList = ['Reload2End', 'WaitTime', 'MaxCols', 'MaxRows', 'SkipSize', 'Recording', 'ClearFig']
        for key in self.MainParamDict.keys():
            if key in SkipList:
                pass
            else:
                state_tuple += key, self.freeze(self.MainParamDict[key])
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']): #add every chart's param dictionary
                tmp_str = 'Chart' + str(i) + ',' + str(j)
                tmp_ctype = self.SubPlotList[i][j].chartType
                state_tuple += tmp_str, tmp_ctype, self.freeze(self.SubPlotList[i][j].PlotParamsDict[tmp_ctype])

        # Now save the difference of the zoom from the home view of the current plot


        if self.showing_total_energy_plt:
            state_tuple += self.freeze(self.TotalEnergyTimeSteps)
        # add to the state_tuple the last modification time of all the output files:
        for key in self.PathDict.keys():
            state_tuple += os.path.getmtime(self.PathDict[key][self.TimeStep.value-1]),
#        fname = 'iseult_img_'+ str(self.TimeStep.value).zfill(3)+'.png'
        self.StateHash = hash(state_tuple)
#        print self.freeze(self.MainParamDict)

    def SaveTmpFig(self):
        self.HashIseultState()
        already_saved = False
        if self.TimeStep.value in self.SavedHashes.keys(): # we have already saved an image for this TimeStep
            # is the current state of Iseult equal to the state when we saved said image?
            already_saved = self.SavedHashes[self.TimeStep.value] ==  self.StateHash


        if not already_saved: # nope, better save it again!
            self.SavedHashes[self.TimeStep.value] =  self.StateHash
            self.SavedImgSize[self.TimeStep.value] =int(self.f.get_size_inches()[0]*self.f.dpi), int(self.f.get_size_inches()[1]*self.f.dpi)

            ram = io.BytesIO()
            self.f.savefig(ram, format='raw', dpi=self.f.dpi, facecolor=self.f.get_facecolor())
            ram.seek(0)
            self.SavedImgStr[self.TimeStep.value] = ram.read() # Save the image into SavedImgs
            ram.close()

    def freeze(self, d):
        ''' This function takes in a dictionary or list, which are not hashable,
        and returns a frozen set, which can be used to hash the dictionary'''
        if isinstance(d, dict):
            return frozenset((key, self.freeze(value)) for key, value in d.items())
        elif isinstance(d, list):
            return tuple(self.freeze(value) for value in d)
        return d

    def ReDrawCanvas(self, keep_view = True):
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][j]
                if subplot is not None and hasattr(subplot, 'graph') and subplot.graph is not None:
                    subplot.graph._in_refresh = True

        #  We need to see if the user has moved around the zoom level in python.
        # First we see if there are any views in the toolbar
        cur_view =  self.toolbar._nav_stack.__call__()
        if cur_view is None:
            keep_view = False
        if self.NewDirectory:
            keep_view = False
        if keep_view:
            self.SaveView()
        self.RecordLineoutViewport()

        # The toolbar stacks up views keyed by the axes they belong to, using
        # weak references, so every view it is still holding empties out as
        # soon as the axes below are thrown away. Left in place, the emptied
        # home view is what the next SaveView would try to measure the user's
        # zoom against, and the zoom would be silently lost. The view being
        # kept has already been read out above, so the stack is dropped here
        # and rebuilt from the new axes by LoadView.
        self.toolbar._nav_stack.clear()

        # Figure.clf() would first clear every axes, rebuilding all its ticks,
        # only to throw it away straight after; removing the axes first skips
        # that, which is a good part of the cost of a redraw. The empty figure
        # is not drawn either: the full one replaces it before Tk is next idle,
        # and drawing it would only flash a blank frame over a remote display.
        for ax in list(self.f.axes):
            self.f.delaxes(ax)
        self.f.clf()
        # Which axes are colorbars is worked out as the panels are drawn, and
        # the axes recorded last time no longer exist.
        self.cbarList = []

        self.LoadAllKeys()


        # Calculate the new xmin, and xmax

        # Work out which panels share their limits with which. A panel no
        # longer necessarily has x on its horizontal axis, so panels are
        # matched up by which physical axis sits where rather than by assuming
        # the horizontal axis is always x. self.shared_axes holds the first
        # axes drawn for each ('horiz'|'vert', axis name) combination; every
        # later panel with the same combination shares its limits with it.
        self.shared_axes = {}
        self.first_k = None
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):

                # First handle the axes sharing
                if self.SubPlotList[i][j].chartType == 'FFTPlots':
                    # The plot type is a spectral plot, which has no spatial dim
                    if self.first_k is None:
                        self.first_k = (i,j)

                # Now... We can draw the graph. A panel that fails shows why in
                # its own cell; the others are drawn as if nothing happened.
                self.SubPlotList[i][j].DrawGraphSafely()

                # ... and let the panels drawn after it share its limits.
                self.RegisterSharedAxes((i,j))

        if self.MainParamDict['ShowTitle']:
            tmpstr = self.PathDict['Prtl'][self.TimeStep.value-1].suffix
            self.f.suptitle(os.path.abspath(self.dirname)+ '/*'+tmpstr+r' at time t = %d $\omega_{pe}^{-1}$'  % round(self.DataDict['time']), size = 15)
        if keep_view:
            self.LoadView()


        ####
        #
        # Write the lines to the phase plots
        #
        ####

        # first find all the phase plots that need writing to
        self.phase_plot_list = []
        self.spectral_plot_list = []

        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                if self.SubPlotList[i][j].draw_failed:
                    # A panel that did not draw has no lines to write to.
                    continue
                if self.SubPlotList[i][j].chartType =='PhasePlot' or self.SubPlotList[i][j].chartType =='EnergyPlot':
                    # The integration region is a range in x drawn as vertical
                    # lines, so it only goes on panels with x running horizontally.
                    if self.SubPlotList[i][j].GetPlotParam('show_int_region') \
                            and plot_axes.marker_orientation(self.SubPlotList[i][j].graph, 'x') == 'v':
                        self.phase_plot_list.append([i,j])
                if self.SubPlotList[i][j].chartType =='SpectraPlot':
                    self.spectral_plot_list.append([i,j])

        for pos in self.phase_plot_list:
            if self.SubPlotList[pos[0]][pos[1]].draw_failed:
                continue
            if self.SubPlotList[pos[0]][pos[1]].GetPlotParam('prtl_type') == 0:
                for spos in self.spectral_plot_list:
                    if self.SubPlotList[spos[0]][spos[1]].draw_failed:
                        continue
                    if self.SubPlotList[spos[0]][spos[1]].GetPlotParam('show_ions'):
                        k = min(self.SubPlotList[spos[0]][spos[1]].graph.spect_num, len(self.dashes_options)-1)
                        # Append the left line to the list
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines.append(self.SubPlotList[pos[0]][pos[1]].graph.axes.axvline(
                        max(self.SubPlotList[spos[0]][spos[1]].graph.i_left_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmin+1),
                        linewidth = 1.5, linestyle = '-', color = self.ion_color))
                        # Choose the left dashes pattern
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines[-1].set_dashes(self.dashes_options[k])

                        # Append the right line to the list
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines.append(self.SubPlotList[pos[0]][pos[1]].graph.axes.axvline(
                        min(self.SubPlotList[spos[0]][spos[1]].graph.i_right_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmax+1),
                        linewidth = 1.5, linestyle = '-', color = self.ion_color))
                        # Choose the right dashes pattern
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines[-1].set_dashes(self.dashes_options[k])
            else:
                for spos in self.spectral_plot_list:
                    if self.SubPlotList[spos[0]][spos[1]].draw_failed:
                        continue
                    if self.SubPlotList[spos[0]][spos[1]].GetPlotParam('show_electrons'):
                        k = min(self.SubPlotList[spos[0]][spos[1]].graph.spect_num, len(self.dashes_options)-1)
                        # Append the left line to the list
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines.append(self.SubPlotList[pos[0]][pos[1]].graph.axes.axvline(
                        max(self.SubPlotList[spos[0]][spos[1]].graph.e_left_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmin+1),
                        linewidth = 1.5, linestyle = '-', color = self.electron_color))
                        # Choose the left dashes pattern
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines[-1].set_dashes(self.dashes_options[k])

                        # Append the right line to the list
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines.append(self.SubPlotList[pos[0]][pos[1]].graph.axes.axvline(
                        min(self.SubPlotList[spos[0]][spos[1]].graph.e_right_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmax+1),
                        linewidth = 1.5, linestyle = '-', color = self.electron_color))
                        # Choose the right dashes pattern
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines[-1].set_dashes(self.dashes_options[k])

        # Synchronously refresh vectors on all active subplots
        import vector_arrows
        for i in range(self.MainParamDict['NumOfRows']):
            for col in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][col]
                if subplot is not None and hasattr(subplot, 'graph') and subplot.graph is not None \
                        and not subplot.draw_failed:
                    g = subplot.graph
                    if subplot.chartType in ['FieldsPlot', 'DensityPlot']:
                        if hasattr(g, 'GetPlotParam') and g.GetPlotParam("show_vectors"):
                            if hasattr(g, 'c_omp') and hasattr(g, 'istep'):
                                vector_arrows.refresh_vectors(g)

        # Reset _in_refresh flag to False
        for i in range(self.MainParamDict['NumOfRows']):
            for col in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][col]
                if subplot is not None and hasattr(subplot, 'graph') and subplot.graph is not None:
                    subplot.graph._in_refresh = False

        self.AlignSharedAxes()
        # Drawn once Tk is idle, so that any other changes made in response to
        # the same events are drawn along with these rather than each costing
        # a full render. Anything that needs the pixels now (saving a frame,
        # the movie writer) renders the figure itself.
        self.canvas.draw_idle()


        if self.MainParamDict['Recording']:
            self.PrintFig()


    def RefreshCanvas(self, keep_view = True):
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][j]
                if subplot is not None and hasattr(subplot, 'graph') and subplot.graph is not None:
                    subplot.graph._in_refresh = True

        #  We need to see if the user has moved around the zoom level in python.
        # First we see if there are any views in the toolbar
        cur_view =  self.toolbar._nav_stack.__call__()
        if cur_view is None:

            keep_view = False

            self.diff_from_home = []
            self.saved_views = {}
            self.spatial_view = {}


        if self.NewDirectory:
            keep_view = False
        if keep_view:
            self.SaveView()
        self.RecordLineoutViewport()


        self.toolbar._nav_stack.clear()

        self.LoadAllKeys()


        # By design, which panel owns each shared axis cannot change if the
        # graph is being refreshed. Any call that would require this needs a redraw
        # Now we refresh the graph.
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):

                self.SubPlotList[i][j].RefreshGraphSafely()

        if self.MainParamDict['ShowTitle']:
            tmpstr = self.PathDict['Prtl'][self.TimeStep.value-1].suffix
            self.f.suptitle(os.path.abspath(self.dirname)+ '/*.'+tmpstr+r' at time t = %d $\omega_{pe}^{-1}$'  % round(self.DataDict['time']), size = 15)

        if keep_view:
            self.LoadView()


        for pos in self.phase_plot_list:
            i = 0
            if self.SubPlotList[pos[0]][pos[1]].draw_failed:
                continue
            if self.SubPlotList[pos[0]][pos[1]].GetPlotParam('prtl_type') == 0:
                for spos in self.spectral_plot_list:
                    if self.SubPlotList[spos[0]][spos[1]].draw_failed:
                        continue
                    if self.SubPlotList[spos[0]][spos[1]].GetPlotParam('show_ions'):
                        # Update the left line to the list
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines[i].set_xdata(
                        [max(self.SubPlotList[spos[0]][spos[1]].graph.i_left_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmin+1),
                        max(self.SubPlotList[spos[0]][spos[1]].graph.i_left_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmin+1)])
                        i+=1
                        # Append the right line of the list
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines[i].set_xdata(
                        [min(self.SubPlotList[spos[0]][spos[1]].graph.i_right_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmax-1),
                        min(self.SubPlotList[spos[0]][spos[1]].graph.i_right_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmax-1)])
                        i+=1
            else:
                for spos in self.spectral_plot_list:
                    if self.SubPlotList[spos[0]][spos[1]].draw_failed:
                        continue
                    if self.SubPlotList[spos[0]][spos[1]].GetPlotParam('show_electrons'):
                        # Update the left line to the list
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines[i].set_xdata(
                        [max(self.SubPlotList[spos[0]][spos[1]].graph.e_left_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmin+1),
                        max(self.SubPlotList[spos[0]][spos[1]].graph.e_left_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmin-1)])
                        i+=1
                        # Append the right line of the list
                        self.SubPlotList[pos[0]][pos[1]].graph.IntRegionLines[i].set_xdata(
                        [min(self.SubPlotList[spos[0]][spos[1]].graph.e_right_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmax+1),
                        min(self.SubPlotList[spos[0]][spos[1]].graph.e_right_loc, self.SubPlotList[pos[0]][pos[1]].graph.xmax-1)])
                        i+=1
        # Synchronously refresh vectors on all active subplots
        import vector_arrows
        for i in range(self.MainParamDict['NumOfRows']):
            for col in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][col]
                if subplot is not None and hasattr(subplot, 'graph') and subplot.graph is not None \
                        and not subplot.draw_failed:
                    g = subplot.graph
                    if subplot.chartType in ['FieldsPlot', 'DensityPlot']:
                        if hasattr(g, 'GetPlotParam') and g.GetPlotParam("show_vectors"):
                            if hasattr(g, 'c_omp') and hasattr(g, 'istep'):
                                vector_arrows.refresh_vectors(g)

        # Reset _in_refresh flag to False
        for i in range(self.MainParamDict['NumOfRows']):
            for col in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][col]
                if subplot is not None and hasattr(subplot, 'graph') and subplot.graph is not None:
                    subplot.graph._in_refresh = False

        self.AlignSharedAxes()
        # Drawn once Tk is idle; see ReDrawCanvas.
        self.canvas.draw_idle()

        if self.MainParamDict['Recording']:
            self.PrintFig()

    def PrintFig(self, MakingMovie = False, Flag = True):
        if self.movie_dir == '':
            self.movie_dir = os.path.abspath(os.path.join(self.dirname, '..', 'Movie'))
        if Flag:
            if self.MainParamDict['Recording'] or MakingMovie:
                try:
                    os.makedirs(self.movie_dir)

                except (OSError, IOError):
                    if not os.path.isdir(self.movie_dir):
                        self.PrintFig(MakingMovie = MakingMovie, Flag = self.recordProblemsPrompt())

            if MakingMovie and os.path.isdir(self.movie_dir):
                try:
                    os.makedirs(os.path.join(self.movie_dir, '../tmp_erase'))

                except (OSError, IOError):
                    if not os.path.isdir(os.path.join(self.movie_dir, '../tmp_erase')):
                        self.PrintFig(MakingMovie = MakingMovie, Flag = self.recordProblemsPrompt())

            fname = 'iseult_img_'+ str(self.TimeStep.value).zfill(self.length_of_outfiles)+'.png'
            if self.MainParamDict['Recording'] :
                try:
                    self.f.savefig(os.path.join(self.movie_dir, fname))#, dpi = 300)#, facecolor=self.f.get_facecolor())#, edgecolor='none')
                except (OSError, IOError):
                    self.PrintFig(MakingMovie = MakingMovie, Flag = self.recordProblemsPrompt())
            if MakingMovie:
                try:
                    self.f.savefig(os.path.join(self.movie_dir, '../tmp_erase', fname))#, dpi = 300)#, facecolor=self.f.get_facecolor())#, edgecolor='none')
                except (OSError, IOError):
                    self.PrintFig(MakingMovie = MakingMovie, Flag = self.recordProblemsPrompt())


    def recordProblemsPrompt(self):
        if messagebox.askyesno("Recording Problems", "You do not have write access to " +self.movie_dir + ". Would you like record frames to a different directory?"):
            mvdir_opt = {}
            mvdir_opt['initialdir'] = self.dirname
            mvdir_opt['mustexist'] = True
            mvdir_opt['parent'] = self
            self.movie_dir = filedialog.askdirectory(title = 'Please choose a different directory where you have write access to save images.', **self.dir_opt)
            return True
        else:
            self.MainParamDict['Recording'] = False
            self.playbackbar.set_recording(False)
            return False

    def MakeAMovie(self, fname, start, stop, step, FPS, outdir = os.curdir, dpi = None):
        '''Record a movie of frames start to stop (inclusive), every step-th
        one, at FPS frames per second. dpi sets the resolution; None uses the
        dpi of the figure on screen.'''
        # First find the last frame is stop is -1:

        if stop == -1:
            stop = len(self.PathDict['Param'])

        # Now build all the frames we have to visit
        frame_arr = np.arange(start, stop, step)
        if frame_arr[-1] != stop:
            frame_arr = np.append(frame_arr, stop)

        # If total energy plot is showing, we have to loop through everything twice.

        if self.showing_total_energy_plt:
            for k in frame_arr:
                self.TimeStep.set(k)

        outpath = os.path.join(outdir, fname)
        print(f'Writing {len(frame_arr)} frames to {outpath}')
        progress = QtWidgets.QProgressDialog(f'Writing {outpath}', 'Stop', 0, len(frame_arr), self.window)
        progress.setWindowTitle('Making a Movie')
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        try:
            with movie_writer.MovieWriter(outpath, FPS) as movie:
                for n, i in enumerate(frame_arr):
                    if progress.wasCanceled():
                        print('Movie stopped early')
                        break
                    self.TimeStep.set(i)
                    movie.write(movie_writer.render_frame(self.f, dpi))
                    print(f"saved frame {i}")
                    progress.setValue(n + 1)
                    QtWidgets.QApplication.processEvents()
        finally:
            progress.close()
            self.playbackbar.show_step(self.TimeStep.value)

    def OpenSaveDialog(self):
        SaveDialog(self)
    def OpenPresetManager(self):
        if self.preset_window is not None and self.preset_window.winfo_exists():
            self.preset_window.lift()
        else:
            self.preset_window = PresetManager(self)
    def OpenMovieDialog(self):
        MovieDialog(self)

    def ShouldLinkSpatial(self, subplot):
        '''Whether `subplot` takes part in the sharing of spatial axes, given
        the current 'Share spatial axes' setting.'''
        if subplot.draw_failed:
            # It is showing an error message, not a coordinate.
            return False
        mode = self.MainParamDict['LinkSpatial']
        if mode == 0: # 'None'
            return False
        if mode != 1 and subplot.chartType in ['PhasePlot', 'EnergyPlot']:
            # 'All non p-x' and 'All 2-D spatial' leave the phase-space panels
            # with limits of their own
            return False
        if mode == 3 and not subplot.GetPlotParam('twoD'):
            # 'All 2-D spatial'
            return False
        if subplot.chartType in ['SpectraPlot', 'FFTPlots', 'TotalEnergyPlot']:
            # These have no spatial axis at all
            return False
        return True

    def GetSharedAxes(self, pos):
        '''The (sharex, sharey) axes the subplot at `pos` should be created
        with, or None where it has nothing to share limits with.

        A panel only shares an axis with a panel that has the same physical
        coordinate in the same place, so a panel plotted against y does not
        inherit the limits of one plotted against x.'''
        subplot = self.SubPlotList[pos[0]][pos[1]]
        if not self.ShouldLinkSpatial(subplot):
            return None, None
        horiz, vert = plot_axes.plot_axes_of(subplot.graph)
        share_x = self.shared_axes.get(('horiz', horiz)) if horiz is not None else None
        share_y = self.shared_axes.get(('vert', vert)) if vert is not None else None
        return share_x, share_y

    def RegisterSharedAxes(self, pos):
        '''Record the axes of the subplot at `pos` as the one that later
        subplots with the same physical axes share their limits with.'''
        subplot = self.SubPlotList[pos[0]][pos[1]]
        if not self.ShouldLinkSpatial(subplot):
            return
        axes = getattr(subplot.graph, 'axes', None)
        if axes is None:
            return
        horiz, vert = plot_axes.plot_axes_of(subplot.graph)
        if horiz is not None:
            self.shared_axes.setdefault(('horiz', horiz), axes)
        if vert is not None:
            self.shared_axes.setdefault(('vert', vert), axes)

    def SpatialAxisGroups(self):
        '''The panels grouped by which physical coordinate sits on which plot axis.

        Returns {('horiz'|'vert', axis name): [graph, ...]}, using the same
        rules as the limit sharing, so a group is exactly the set of panels
        that are meant to line up with one another.'''
        groups = {}
        for i in range(self.MainParamDict['NumOfRows']):
            for j in range(self.MainParamDict['NumOfCols']):
                subplot = self.SubPlotList[i][j]
                if not self.ShouldLinkSpatial(subplot):
                    continue
                graph = subplot.graph
                if getattr(graph, 'axes', None) is None:
                    continue
                horiz, vert = plot_axes.plot_axes_of(graph)
                if horiz is not None:
                    groups.setdefault(('horiz', horiz), []).append(graph)
                if vert is not None:
                    groups.setdefault(('vert', vert), []).append(graph)
        return groups

    def AlignSharedAxes(self):
        '''Give panels that show the same physical coordinate the same scale.

        Matching limits is not enough on its own: with 'Aspect = 1' matplotlib
        shrinks a 2D panel's box to keep its pixels square, so the same range
        of, say, y would be drawn across a different width than in a lineout
        and the two would not line up. Every panel in a group is therefore
        squeezed to the narrowest box in that group, so that equal limits
        really do mean equal scale.

        Panels whose aspect matplotlib controls set the target and are left
        alone; the rest are resized around the middle of their own grid cell,
        which is where matplotlib anchors an aspect-constrained panel too.'''
        groups = self.SpatialAxisGroups()
        if not groups:
            return

        all_graphs = {id(g): g for members in groups.values() for g in members}.values()
        if not any(g.axes.get_aspect() != 'auto' for g in all_graphs):
            # Every box is the full grid cell, so the scales already match.
            return

        # Start from the untouched grid cells so that repeated calls do not
        # shrink the panels a little further each time.
        for graph in all_graphs:
            spec = graph.axes.get_subplotspec()
            if spec is not None:
                graph.axes.set_position(spec.get_position(self.f))
        # Work out the aspect-locked boxes the way drawing the figure would,
        # without paying for laying out every tick and label to do it.
        for graph in all_graphs:
            if graph.axes.get_aspect() != 'auto':
                graph.axes.apply_aspect()

        for (side, _axis), members in groups.items():
            free = [g for g in members if g.axes.get_aspect() == 'auto']
            if len(free) == len(members):
                continue # nothing is constraining this group
            if side == 'horiz':
                target = min(g.axes.get_position().width for g in members)
            else:
                target = min(g.axes.get_position().height for g in members)
            for graph in free:
                box = graph.axes.get_position()
                if side == 'horiz':
                    graph.axes.set_position([box.x0 + (box.width - target)/2, box.y0,
                                             target, box.height])
                else:
                    graph.axes.set_position([box.x0, box.y0 + (box.height - target)/2,
                                             box.width, target])

    def get_active_viewport(self):
        '''The spatial region shown by the first 2D spatial panel, as a tuple
        of (physical axis, low, high) triples. Panels that select particles by
        region use this, so it is given in physical coordinates rather than as
        the plot's own x and y limits, which may be swapped by a rotation.'''
        if not hasattr(self, 'SubPlotList') or self.SubPlotList is None:
            return None
        for i in range(self.MainParamDict['NumOfRows']):
            if i >= len(self.SubPlotList):
                continue
            for j in range(self.MainParamDict['NumOfCols']):
                if j >= len(self.SubPlotList[i]):
                    continue
                subplot = self.SubPlotList[i][j]
                if subplot.draw_failed:
                    # Its axes hold an error message, not a region of the domain.
                    continue
                if subplot.chartType in ['FieldsPlot', 'DensityPlot', 'MagPlots', 'Moments']:
                    if subplot.graph and subplot.graph.GetPlotParam('twoD'):
                        if hasattr(subplot.graph, 'axes') and subplot.graph.axes is not None:
                            horiz, vert = plot_axes.plot_axes_of(subplot.graph)
                            xlim = subplot.graph.axes.get_xlim()
                            ylim = subplot.graph.axes.get_ylim()
                            return ((horiz, min(xlim), max(xlim)),
                                    (vert, min(ylim), max(ylim)))
        return None

    def RecordLineoutViewport(self):
        '''Remember the region 1D lineouts are taken across, see plot_axes.lineout_window.

        It is read before any panel is drawn, while the 2D panels still show
        the region the user was looking at: once they are redrawn their limits
        may be the whole domain again until the view is restored.'''
        self.lineout_viewport = None if self.NewDirectory else self.get_active_viewport()

    def has_lineouts(self):
        '''Whether any panel takes a 1D lineout across the 2D viewport.'''
        for i in range(self.MainParamDict['NumOfRows']):
            if i >= len(self.SubPlotList):
                continue
            for j in range(self.MainParamDict['NumOfCols']):
                if j >= len(self.SubPlotList[i]):
                    continue
                subplot = self.SubPlotList[i][j]
                if subplot.draw_failed or not subplot.graph:
                    continue
                if subplot.chartType in ('OhmsLaw', 'PressureBalance', 'FFTPlots'):
                    return True
                if subplot.chartType in ('FieldsPlot', 'DensityPlot', 'MagPlots') \
                        and not subplot.graph.GetPlotParam('twoD'):
                    return True
        return False

    def is_viewport_zoomed(self):
        '''Whether the user has zoomed into any spatial coordinate.'''
        return bool(getattr(self, 'spatial_view', None)) or any(
            any(saved['is_changed']) for saved in getattr(self, 'saved_views', {}).values()
            if saved['twoD'] and saved['ctype'] in ('FieldsPlot', 'DensityPlot', 'MagPlots', 'Moments'))

    def on_draw(self, event):
        '''Keep the shared axes lined up after an interactive zoom or pan.

        Zooming an aspect-locked 2D panel changes the size of its box as soon
        as it is drawn, but the lineouts sharing its limits would only be
        resized to match at the next refresh. So after every draw the boxes
        are aligned again, and the canvas redrawn if that moved any of them.'''
        if getattr(self, '_aligning', False):
            return
        self._aligning = True
        try:
            before = [ax.get_position().bounds for ax in self.f.axes]
            self.AlignSharedAxes()
            after = [ax.get_position().bounds for ax in self.f.axes]
            if not np.allclose(before, after, atol=1e-6):
                self.canvas.draw_idle()
        except Exception:
            pass
        finally:
            self._aligning = False

    def on_release(self, event):
        pending = getattr(self, '_pending_click', None)
        self._pending_click = None
        if pending is not None and event.button == pending[3]:
            x, y, t, _ = pending
            if time.time() - t < 0.4 and abs(event.x - x) < 5 and abs(event.y - y) < 5:
                self._open_clicked_settings(event)
        # Defer limit check slightly to let toolbar updates complete
        self.after(100, self.check_limits_and_renew)

    def check_limits_and_renew(self):
        # Check if there is any PhasePlot with filter_by_viewport enabled
        has_viewport_phase_plot = False
        if hasattr(self, 'SubPlotList') and self.SubPlotList is not None:
            for i in range(self.MainParamDict['NumOfRows']):
                if i >= len(self.SubPlotList):
                    continue
                for j in range(self.MainParamDict['NumOfCols']):
                    if j >= len(self.SubPlotList[i]):
                        continue
                    subplot = self.SubPlotList[i][j]
                    if subplot.chartType in ['PhasePlot', 'Moments'] and subplot.graph:
                        if subplot.graph.GetPlotParam('filter_by_viewport'):
                            has_viewport_phase_plot = True
                            break
                if has_viewport_phase_plot:
                    break

        # Get current active viewport
        viewport = self.get_active_viewport()
        if viewport is None:
            return

        # Compare with the last used viewport for phase plots, and with the
        # one the lineouts were cut across
        if has_viewport_phase_plot and \
                (not hasattr(self, 'last_phase_viewport') or self.last_phase_viewport != viewport):
            self.RenewCanvas(keep_view=True)
        elif getattr(self, 'lineout_viewport', None) != viewport and self.has_lineouts():
            self.RenewCanvas(keep_view=True)

    def onclick(self, event):
        '''After being clicked, we should use the x and y of the cursor to
        determine what subplot was clicked'''

        # Since the location of the cursor is returned in pixels and gs0 is
        # given as a relative value, we must first convert the value into a
        # relative x and y
        if not event.inaxes:
            pass
        if event.button == 1:
            return
        # In pan/zoom mode a right-drag zooms, so only a brief click without
        # movement (judged on release) may open the settings
        mode = getattr(self.toolbar, 'mode', '')
        if getattr(mode, 'value', mode):
            self._pending_click = (event.x, event.y, time.time(), event.button)
            return
        self._open_clicked_settings(event)

    def _open_clicked_settings(self, event):
        fig_size = self.f.get_size_inches()*self.f.dpi # Fig size in px

        x_loc = event.x/fig_size[0] # The relative x position of the mouse in the figure
        y_loc = event.y/fig_size[1] # The relative y position of the mouse in the figure

        sub_plots = self.gs0.get_grid_positions(self.f)
        row_array = np.sort(np.append(sub_plots[0], sub_plots[1]))
        col_array = np.sort(np.append(sub_plots[2], sub_plots[3]))
        i = int((len(row_array)-row_array.searchsorted(y_loc))/2)
        j = int(col_array.searchsorted(x_loc)//2)

        self.SubPlotList[i][j].OpenSubplotSettings()

    def shock_finder(self):
        '''The main idea of the shock finder, is we go to the last timestep
        in the simulation and find where the density is half it's max value.
        We then calculate the speed of the of the shock assuming it is
        traveling at constant velocity. We also calculate the initial B & E fields.'''

        # First load the first field file to find the initial size of the
        # box in the x direction, and find the initial field values

        # Find out what sigma is
        try:
            ''' Obviously the most correct way to do this to to calculate b0 from sigma.
            This is proving more difficult that I thought it would be, so I am calculating it
            as Jaehong did.

            sigma = f['sigma'][0]
            gamma0 = f['gamma0'][0]
            c = f['c'][0]
            btheta = f['btheta'][0]
            bphi = f['bphi'][0]

            ppc0 = f['ppc0'][0]
            mi = f['mi'][0]
            me = f['me'][0]
            print mi, c, ppc0
            if gamma0 <1:
                beta0 = gamma0
                gamma0 = 1/np.sqrt(1-gamma0**2)
            else:
                beta0=np.sqrt(1-gamma0**(-2))


            if sigma <= 1E-10:
                self.b0 = 1.0
                self.e0 = 1.0
            else:
                # b0 in the upstream frame
                self.b0 = np.sqrt(gamma0*ppc0*.5*c**2*(mi+me)*sigma)
                # Translate to the downstream frame
                b_x = self.b0*np.cos(btheta)*np.cos(bphi)
                b_y = self.b0*np.sin(btheta)*np.cos(bphi)
                b_z = self.b0*np.sin(btheta)*np.cos(bphi)
                print 'sigma b0', self.b0
                '''
            # Normalize by b0
            filepath = self.PathDict['Param'][0]
            abs_sigma = np.abs(data_loading.load_dataset(filepath, 'sigma'))

            if abs_sigma == 0:
                self.btheta = np.nan
            else:
                self.btheta = self.c_omp = data_loading.load_dataset(filepath, 'btheta', slice(0,1))
        except KeyError:
            self.btheta = np.nan

        filepath = self.PathDict['Flds'][0]
        nxf0 = data_loading.dataset_shape(filepath, 'by')[1]
        if np.isnan(self.btheta):
            # No known background field (an unmagnetized run, or data that
            # does not record btheta): fields are shown unnormalized, and
            # changes in them are measured from zero.
            self.b0 = 1.0
            self.e0 = 1.0
            self.bx0 = self.by0 = self.bz0 = 0.0
            self.ex0 = self.ey0 = self.ez0 = 0.0
        else:
            # Normalize by b0
            b_slice = (slice(0,1),slice(-1,None),slice(-10,-9))
            self.bx0 = data_loading.load_dataset(filepath, 'bx', b_slice)
            self.by0 = data_loading.load_dataset(filepath, 'by', b_slice)
            self.bz0 = data_loading.load_dataset(filepath, 'bz', b_slice)
            self.b0 = np.sqrt(self.bx0**2+self.by0**2+self.bz0**2)
            e_slice = (slice(0,1), slice(-1,None), slice(-2,-1))
            self.ex0 = data_loading.load_dataset(filepath, 'ex', e_slice)
            self.ey0 = data_loading.load_dataset(filepath, 'ey', e_slice)
            self.ez0 = data_loading.load_dataset(filepath, 'ez', e_slice)
            self.e0 = np.sqrt(self.ex0**2+self.ey0**2+self.ez0**2)

        # Load the final time step to find the shock's location at the end.
        filepath = self.PathDict['Flds'][-1]
        dens_slice = (slice(0,1))
        dens_arr = data_loading.load_dataset(filepath, 'dens', dens_slice)[0,:,:]

        # I use this file to get the final time, the istep, interval, and c_omp
        filepath = self.PathDict['Param'][-1]
        final_time = data_loading.load_dataset(filepath, 'time')
        istep      = data_loading.load_dataset(filepath, 'istep')
        interval   = data_loading.load_dataset(filepath, 'interval')
        c_omp      = data_loading.load_dataset(filepath, 'c_omp')

        # Find out where the shock is at the last time step.
        # Clamp against dens_arr's own width (not nxf0, which comes from the
        # first Flds file and can be >= the last file's width), leaving at
        # least one column so the slice below is never empty.
        jstart = int(min(10*c_omp/istep, nxf0, dens_arr.shape[1]-1))
        jstart = max(jstart, 0)
        # build the final x_axis of the plot

        xaxis_final = np.arange(dens_arr.shape[1])/c_omp*istep
        # Find the shock by seeing where the density is 1/2 of it's
        # max value.

        dens_half_max = max(dens_arr[dens_arr.shape[0]//2,jstart:])*.5

        # Find the farthest location where the average density is greater
        # than half max
        ishock_final = np.where(dens_arr[dens_arr.shape[0]//2,jstart:]>=dens_half_max)[0][-1]
        xshock_final = xaxis_final[ishock_final]
        # Avoid inf/NaN when the last output is at t=0 (e.g. only one dump)
        self.shock_speed = xshock_final/final_time if final_time > 0 else 0.0
        self.prev_shock_loc = np.nan

    def setKnob(self, value):
        # If the time parameter changes update the plots
        if self._defer_step_render and not self.MainParamDict['Recording']:
            # Stepped from the playback bar: draw once Qt is idle, by which
            # time any further steps already queued (e.g. from a held arrow
            # key) have been taken, so only the step the user ends up on is
            # loaded and drawn. While recording, every step is drawn so that
            # every step is saved.
            if self._pending_step_render is None:
                self._pending_step_render = self.after_idle(self._RenderPendingStep)
            # Show the step now; it is drawn once Qt is idle
            self.playbackbar.show_step(value)
        else:
            self.playbackbar.show_step(value)
            self.RenewCanvas()

    def StepInteractively(self, change_step):
        '''Call change_step, which sets self.TimeStep, deferring the redraw
        it causes until Tk is idle. Everything else that sets the time step,
        e.g. MakeAMovie, still has the figure redrawn before set returns.'''
        self._defer_step_render = True
        try:
            change_step()
        finally:
            self._defer_step_render = False

    def _RenderPendingStep(self):
        self._pending_step_render = None
        self.RenewCanvas()

    def OpenSettings(self, *args):
        if self.settings_window is None:
            self.settings_window = SettingsFrame(self)
        else:
            self.settings_window.destroy()
            self.settings_window = SettingsFrame(self)


def runMe(cmd_args):
    qapp = Tk.ensure_app()
    qapp.setApplicationName('Iseult')
    # A flat style with no animations: quick to draw, and cheap to send over VNC
    qapp.setStyle('Fusion')
    for effect in (Qt.UI_AnimateMenu, Qt.UI_FadeMenu, Qt.UI_AnimateCombo,
                   Qt.UI_AnimateTooltip, Qt.UI_FadeTooltip, Qt.UI_AnimateToolBox):
        qapp.setEffectEnabled(effect, False)
    app = MainApp('Iseult', cmd_args)
    app.mainloop()
