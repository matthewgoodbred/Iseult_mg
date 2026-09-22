'''Reading and rebuilding the view objects matplotlib's navigation toolbar keeps.

A 'view' is whatever `Axes._get_view()` returns. The toolbar stacks them up so
it can step back through them, and Iseult reads them so that a frame the user
has zoomed into survives the plot being redrawn underneath it. matplotlib 3.9
changed a view from a plain 4 element tuple of limits to a dict that also
carries the autoscale state, so the two functions here are the only place that
has to know which form is in use; everywhere else deals in the four numbers.
'''

def view_limits(view):
    '''The (x0, x1, y0, y1) limits held by `view`.'''
    if isinstance(view, dict):
        return tuple(view['xlim']) + tuple(view['ylim'])
    return tuple(view)

def view_from_limits(view, lims, kept):
    '''A copy of `view` showing the limits `lims`.

    `kept` says which of the four limits were taken from the user's own zoom
    rather than from the panel that was just drawn. An axis the user has framed
    themselves must not be autoscaled back open by the next draw, so its
    autoscale is turned off; an axis they left alone keeps whatever the panel
    asked for.'''
    if not isinstance(view, dict):
        return tuple(lims)
    new_view = dict(view)
    new_view['xlim'] = (lims[0], lims[1])
    new_view['ylim'] = (lims[2], lims[3])
    if kept[0] or kept[1]:
        new_view['autoscalex_on'] = False
    if kept[2] or kept[3]:
        new_view['autoscaley_on'] = False
    return new_view
