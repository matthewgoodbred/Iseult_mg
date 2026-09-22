#!/usr/bin/env python
"""This file contains the functions to control and generate the arrow vector overlays.
"""

import tkinter as Tk
import numpy as np
import plot_axes
import matplotlib


# The vector fields that can be drawn, as arrows here or as streamlines in
# streamlines.py. The index into this list is what 'vector_type' stores.
VECTOR_FIELDS = ['B Field', 'E field', 'J [current]', 'Vi (ion vel)',
                 'Ve (electron vel)', 'S (Poynting)']

B_FIELD, E_FIELD, J_FIELD, VI_FIELD, VE_FIELD, S_FIELD = range(len(VECTOR_FIELDS))


def add_vector_params(param_dictionary):
    """Add data to the parameter dictionary for controlling the vectors.
    """
    param_dictionary["show_vectors"] = False
    param_dictionary["vector_type"] = B_FIELD


def field_keys(vtype):
    """The datasets a vector field needs in order to give any component."""
    if vtype == B_FIELD:
        return ['bx', 'by', 'bz']
    if vtype == E_FIELD:
        return ['ex', 'ey', 'ez']
    if vtype == J_FIELD:
        return ['jx', 'jy', 'jz']
    if vtype == VI_FIELD:
        return ['v3xi', 'v3yi', 'v3zi']
    if vtype == VE_FIELD:
        return ['v3x', 'v3y', 'v3z', 'v3xi', 'v3yi', 'v3zi', 'dens', 'densi']
    if vtype == S_FIELD:
        return plot_axes.poynting_keys()
    return []


def field_component(panel, vtype, axis):
    """The whole array of one component of vector field `vtype`.

    `axis` is a physical axis name, so the caller can ask for whichever
    component points along the direction it is about to draw.
    """
    load = panel.FigWrap.LoadKey
    if vtype == B_FIELD:
        return load('b' + axis)
    if vtype == E_FIELD:
        return load('e' + axis)
    if vtype == J_FIELD:
        return load('j' + axis)
    if vtype == VI_FIELD:
        return load('v3' + axis + 'i')
    if vtype == VE_FIELD:
        # The electron fluid velocity is what is left of the total once the
        # ions are taken out of it.
        dens = np.asanyarray(load('dens'))
        densi = np.asanyarray(load('densi'))
        dense = np.maximum(dens - densi, 1e-5)
        return (dens * load('v3' + axis) - densi * load('v3' + axis + 'i')) / dense
    if vtype == S_FIELD:
        return plot_axes.poynting_component(load, axis)
    raise KeyError(f'unknown vector field {vtype}')


def in_plane_field(panel, vtype):
    """The (horizontal, vertical) parts of vector field `vtype` on this panel."""
    return plot_axes.in_plane_slices(panel, lambda axis: field_component(panel, vtype, axis))


def add_vector_plot_keys(panel):
    """Add components of the selected vector type to arrs_needed.
    """
    panel.arrs_needed.extend(field_keys(panel.GetPlotParam('vector_type')))


def add_vector_buttons(settings, panel, starting_row):
    """Add the vectors checkbox and dropdown selection to the settings window next to streamlines.
    """
    settings.VectorList = list(VECTOR_FIELDS)

    settings.show_vectors = Tk.BooleanVar()
    settings.show_vectors.set(settings.parent.GetPlotParam("show_vectors"))
    Tk.ttk.Checkbutton(
        settings.frm,
        text="Display Vectors",
        variable=settings.show_vectors,
        command=lambda: __show_vector_handler(settings, panel),
    ).grid(row=starting_row + 1, column=1, sticky=Tk.W)

    settings.vector_type = Tk.StringVar()
    vtype_val = settings.parent.GetPlotParam("vector_type")
    # Clamp in case it's out of range of the list
    if vtype_val >= len(settings.VectorList):
        vtype_val = 0
    settings.vector_type.set(settings.VectorList[vtype_val])

    vector_chooser = Tk.ttk.OptionMenu(
        settings.frm,
        settings.vector_type,
        settings.VectorList[vtype_val],
        *tuple(settings.VectorList),
        command=lambda val: __change_vector_type(settings, panel)
    )
    vector_chooser.grid(row=starting_row + 1, column=2, sticky=Tk.W + Tk.E)


def __show_vector_handler(settings, panel):
    """Handle what happens when the `show_vectors` button is toggled.
    """
    settings.parent.SetPlotParam(
        "show_vectors", settings.show_vectors.get(), update_plot=False, NeedsRedraw=True
    )

    if not settings.parent.GetPlotParam("show_vectors"):
        remove_vectors(panel)
    else:
        # Reload keys first to ensure we have the correct vector data loaded
        settings.parent.parent.LoadAllKeys()
        register_zoom_callback(panel)

    settings.parent.parent.canvas.draw_idle()


def __change_vector_type(settings, panel):
    """Handle when the vector selection dropdown is changed.
    """
    selected_val = settings.vector_type.get()
    try:
        vtype_idx = settings.VectorList.index(selected_val)
    except ValueError:
        vtype_idx = 0

    settings.parent.SetPlotParam("vector_type", vtype_idx, update_plot=False, NeedsRedraw=True)

    # Reload keys since we might need new datasets (e.g. if we switched from B to Vi)
    settings.parent.parent.LoadAllKeys()

    if settings.parent.GetPlotParam("show_vectors"):
        refresh_vectors(panel)

    settings.parent.parent.canvas.draw_idle()


def remove_vectors(panel):
    """Remove quiver vector arrows if they exist.
    """
    if hasattr(panel, 'vector_quiver') and panel.vector_quiver is not None:
        try:
            panel.vector_quiver.remove()
        except Exception:
            pass
        panel.vector_quiver = None


def register_zoom_callback(panel):
    """Connect zoom/pan events on axis limits to redrawing vectors.
    """
    if not hasattr(panel, 'vector_cid_x') or panel.vector_cid_x is None:
        panel.vector_cid_x = panel.axes.callbacks.connect('xlim_changed', lambda ax: on_limits_changed(panel))
    if not hasattr(panel, 'vector_cid_y') or panel.vector_cid_y is None:
        panel.vector_cid_y = panel.axes.callbacks.connect('ylim_changed', lambda ax: on_limits_changed(panel))


def on_limits_changed(panel):
    """Triggered on axes zoom/pan. Schedules a deferred vector update to run after limits settle.
    """
    if hasattr(panel, '_in_refresh') and panel._in_refresh:
        return
    main_app = panel.parent
    if hasattr(main_app, '_zoom_timer_id') and main_app._zoom_timer_id is not None:
        try:
            main_app.after_cancel(main_app._zoom_timer_id)
        except Exception:
            pass
        main_app._zoom_timer_id = None

    def execute_update():
        main_app._zoom_timer_id = None
        for row in range(main_app.MainParamDict['NumOfRows']):
            for col in range(main_app.MainParamDict['NumOfCols']):
                subplot = main_app.SubPlotList[row][col]
                if subplot is not None and hasattr(subplot, 'graph') and subplot.graph is not None:
                    g = subplot.graph
                    if hasattr(g, 'GetPlotParam') and g.GetPlotParam("show_vectors"):
                        if hasattr(g, 'c_omp') and hasattr(g, 'istep'):
                            refresh_vectors(g)
        main_app.canvas.draw_idle()

    # Schedule the update to run after 50ms (after limits-changed propagation has fully completed)
    main_app._zoom_timer_id = main_app.after(50, execute_update)


def draw_vectors(panel):
    """Draw vectors on the panel's axes.
    """
    remove_vectors(panel)

    try:
        # in_plane_field picks the components that really do point along this
        # panel's horizontal and vertical axes, rotation included.
        U_full, V_full = in_plane_field(panel, panel.GetPlotParam('vector_type'))
    except (AttributeError, KeyError, TypeError, IndexError):
        return

    if U_full.ndim != 2 or V_full.ndim != 2:
        return

    register_zoom_callback(panel)

    xlim = panel.axes.get_xlim()
    ylim = panel.axes.get_ylim()

    c_omp = panel.c_omp
    istep = panel.istep

    Ny_grid, Nx_grid = U_full.shape

    i_min = int(np.floor(xlim[0] * c_omp / istep))
    i_max = int(np.ceil(xlim[1] * c_omp / istep))
    j_min = int(np.floor(ylim[0] * c_omp / istep))
    j_max = int(np.ceil(ylim[1] * c_omp / istep))

    # Clamp indices
    i_min = max(0, min(i_min, Nx_grid - 1))
    i_max = max(0, min(i_max, Nx_grid - 1))
    j_min = max(0, min(j_min, Ny_grid - 1))
    j_max = max(0, min(j_max, Ny_grid - 1))

    if i_max <= i_min:
        i_max = i_min + 1
    if j_max <= j_min:
        j_max = j_min + 1

    # Target 50 vector points in each direction
    stride_x = max(1, (i_max - i_min) // 50)
    stride_y = max(1, (j_max - j_min) // 50)

    ix = np.arange(i_min, i_max + 1, stride_x)
    iy = np.arange(j_min, j_max + 1, stride_y)

    ix = ix[ix < Nx_grid]
    iy = iy[iy < Ny_grid]

    if len(ix) == 0 or len(iy) == 0:
        return

    IX, IY = np.meshgrid(ix, iy)

    X = IX * (istep / c_omp)
    Y = IY * (istep / c_omp)

    U = U_full[IY, IX]
    V = V_full[IY, IX]

    # Temporarily turn off autoscale so quiver doesn't alter axes limits
    autoscale_on = panel.axes.get_autoscale_on()
    panel.axes.set_autoscale_on(False)
    try:
        panel.vector_quiver = panel.axes.quiver(X, Y, U, V, pivot='middle', color='black')
    finally:
        panel.axes.set_autoscale_on(autoscale_on)


def refresh_vectors(panel):
    """Refresh the vector overlay by removing the old quiver and drawing a new one.
    """
    remove_vectors(panel)
    draw_vectors(panel)
