'''Tk-style variables and widgets backed by Qt.

Every panel's settings pane was written against tkinter: Tk variables with
get/set/trace, ttk widgets bound to them, and laid out with grid and pack.
This module gives those panes the same names and semantics on top of
PySide6, so that their handlers, which hold most of the logic, stay as they
were. Import it in place of tkinter:

    import qt_compat as Tk
    from qt_compat import ttk

A Toplevel is not a window of its own: it is handed to the window host that
MainApp registers with set_window_host, which shows it as a tab of the
settings dock. With no host (e.g. in tests) it is shown as a plain window.
'''
import os
import re
import sys
import traceback

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

ttk = sys.modules[__name__]

# Tk's constants
N, S, E, W = 'n', 's', 'e', 'w'
NS, EW, NSEW = 'ns', 'ew', 'nsew'
NE, NW, SE, SW = 'ne', 'nw', 'se', 'sw'
LEFT, RIGHT, TOP, BOTTOM = 'left', 'right', 'top', 'bottom'
BOTH, X, Y, NONE = 'both', 'x', 'y', 'none'
HORIZONTAL, VERTICAL = 'horizontal', 'vertical'
END, INSERT = 'end', 'insert'
ACTIVE, NORMAL, DISABLED = 'active', 'normal', 'disabled'
CENTER = 'center'


def ensure_app():
    """The QApplication, made if there is none yet."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        # Load plugins from PySide6's own Qt. Otherwise a qt.conf left beside
        # python by a conda Qt5 (e.g. installed for matplotlib) points Qt at
        # plugins it cannot load, and Qt cannot start.
        import PySide6
        plugins = os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins')
        if os.path.isdir(plugins):
            QtCore.QCoreApplication.setLibraryPaths([plugins])
        app = QtWidgets.QApplication(sys.argv[:1])
    return app


class TclError(ValueError, RuntimeError):
    '''Raised where tkinter would raise TclError: a variable that cannot be
    read as its type, or a widget that has been destroyed.'''


def _report():
    # Tk prints errors raised in callbacks and carries on; so do we.
    traceback.print_exc()


####
#
# Timers
#
####

_timers = {}
_next_timer_id = [0]


def after(ms, func, *args):
    '''Call func(*args) after ms milliseconds. Returns an id for after_cancel.'''
    _next_timer_id[0] += 1
    timer_id = 'after#%d' % _next_timer_id[0]
    timer = QtCore.QTimer()
    timer.setSingleShot(True)

    def fire():
        _timers.pop(timer_id, None)
        try:
            func(*args)
        except Exception:
            _report()

    timer.timeout.connect(fire)
    _timers[timer_id] = timer
    timer.start(max(int(ms), 0))
    return timer_id


def after_idle(func, *args):
    return after(0, func, *args)


def after_cancel(timer_id):
    timer = _timers.pop(timer_id, None)
    if timer is not None:
        timer.stop()


####
#
# Variables
#
####

def _same(a, b):
    '''Whether two variable values are equal the way Tk compares them, as
    strings, with True == 1 and 1 == 1.0.'''
    if isinstance(a, bool):
        a = int(a)
    if isinstance(b, bool):
        b = int(b)
    if str(a) == str(b):
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


class Variable:
    _default = ''

    def __init__(self, master=None, value=None, name=None):
        self._value = self._default if value is None else value
        self._traces = {}
        self._next_trace = 0
        # Callbacks that push a new value into the widgets bound to this variable
        self._syncs = []

    def set(self, value):
        self._value = value
        for sync in list(self._syncs):
            try:
                sync(value)
            except RuntimeError:
                # The widget has been deleted
                self._syncs.remove(sync)
        for callback in list(self._traces.values()):
            try:
                callback('', '', 'write')
            except Exception:
                _report()

    def get(self):
        return self._value

    def trace(self, mode, callback):
        self._next_trace += 1
        name = 'trace%d' % self._next_trace
        self._traces[name] = callback
        return name

    trace_variable = trace
    trace_add = trace

    def trace_remove(self, mode, name):
        self._traces.pop(name, None)

    def trace_vdelete(self, mode, name):
        self._traces.pop(name, None)

    def _bind(self, sync):
        self._syncs.append(sync)
        sync(self._value)


class StringVar(Variable):
    def get(self):
        return str(self._value)


class IntVar(Variable):
    _default = 0

    def get(self):
        value = self._value
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                raise TclError('expected integer but got "%s"' % value)


class DoubleVar(Variable):
    _default = 0.0

    def get(self):
        try:
            return float(self._value)
        except (TypeError, ValueError):
            raise TclError('expected floating-point number but got "%s"' % self._value)


class BooleanVar(Variable):
    _default = False

    def get(self):
        value = self._value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ('1', 'true', 'yes', 'on'):
                return True
            if low in ('0', 'false', 'no', 'off', ''):
                return False
            raise TclError('expected boolean value but got "%s"' % value)
        return bool(value)


####
#
# Helpers
#
####

class _Event:
    '''Stands in for the Tk event handed to bound callbacks.'''

    def __init__(self, widget=None, **kw):
        self.widget = widget
        self.x = self.y = 0
        self.keysym = self.char = ''
        self.__dict__.update(kw)


def _color(name):
    '''A Tk colour name as a Qt one. Tk's grayNN is NN% of white.'''
    m = re.fullmatch(r'gr[ae]y(\d{1,3})', str(name).strip().lower())
    if m:
        v = round(255 * int(m.group(1)) / 100)
        return 'rgb(%d,%d,%d)' % (v, v, v)
    return str(name)


def _pad(padding):
    if padding is None:
        return (0, 0, 0, 0)
    if isinstance(padding, (int, float)):
        p = int(padding)
        return (p, p, p, p)
    padding = [int(p) for p in padding]
    if len(padding) == 2:
        return (padding[0], padding[1], padding[0], padding[1])
    if len(padding) == 4:
        return tuple(padding)
    return (padding[0],) * 4


def _grid_alignment(sticky):
    sticky = (sticky or '').lower()
    align = Qt.AlignmentFlag(0)
    if 'w' in sticky and 'e' in sticky:
        pass
    elif 'e' in sticky:
        align |= Qt.AlignRight
    else:
        # Tk centres an unsticky widget in its cell. Left aligned reads more
        # cleanly in a form, so that is what we do.
        align |= Qt.AlignLeft
    if 'n' in sticky and 's' in sticky:
        pass
    elif 'n' in sticky:
        align |= Qt.AlignTop
    elif 's' in sticky:
        align |= Qt.AlignBottom
    else:
        align |= Qt.AlignVCenter
    return align


def _mnemonic_safe(text):
    # Qt reads '&' in button and tab text as marking a keyboard shortcut
    return str(text).replace('&', '&&')


def _char_width(widget, chars):
    return widget.fontMetrics().horizontalAdvance('0') * chars + 12


####
#
# Widgets
#
####

class Widget:
    '''A Tk-like handle on a Qt widget, `self.qw`.'''

    def __init__(self, master, qw):
        self.master = master
        self.qw = qw
        # Keeps this handle alive for as long as the Qt widget is, since the
        # widget's signals are connected to its methods.
        qw._tk_handle = self
        self._bindings = {}
        if master is not None and master._container() is not None:
            qw.setParent(master._container())
        # Tk widgets are not shown until they are gridded or packed.
        qw.hide()

    # Geometry management

    def grid(self, row=None, column=0, rowspan=1, columnspan=1, sticky='', **ignored):
        self.master._grid_add(self, row, column, rowspan, columnspan, sticky)

    def pack(self, side=TOP, fill=NONE, expand=False, anchor=None, **ignored):
        self.master._pack_add(self, side, fill, expand)

    def place(self, **ignored):
        self.qw.show()

    def grid_forget(self):
        self.master._forget(self)
        self.qw.hide()

    grid_remove = pack_forget = grid_forget

    # Configuration

    def config(self, **kw):
        try:
            self._configure(kw)
        except RuntimeError as e:
            raise TclError(str(e))

    configure = config

    def _configure(self, kw):
        if 'state' in kw:
            self.qw.setEnabled(kw['state'] != DISABLED)
        if 'text' in kw and hasattr(self.qw, 'setText'):
            self.qw.setText(_mnemonic_safe(kw['text']))
        if 'foreground' in kw:
            self.qw.setStyleSheet('color: %s;' % _color(kw['foreground']))

    def state(self, statespec=None):
        if statespec:
            for s in statespec:
                if s == 'disabled':
                    self.qw.setEnabled(False)
                elif s == '!disabled':
                    self.qw.setEnabled(True)
        return () if self.qw.isEnabled() else ('disabled',)

    def instate(self, statespec):
        enabled = self.qw.isEnabled()
        return all((s == 'disabled') != enabled if s in ('disabled', '!disabled') else True
                   for s in statespec) if statespec else True

    # Events

    def bind(self, sequence, callback, add=None):
        self._bindings[sequence] = callback

    def _fire(self, sequence, **kw):
        callback = self._bindings.get(sequence)
        if callback is not None:
            try:
                callback(_Event(widget=self, **kw))
            except Exception:
                _report()
            return True
        return False

    # Everything else

    def destroy(self):
        if self.master is not None:
            self.master._forget(self)
        try:
            self.qw.setParent(None)
            self.qw.deleteLater()
        except RuntimeError:
            pass

    def winfo_exists(self):
        try:
            self.qw.objectName()
            return 1
        except RuntimeError:
            return 0

    def focus_set(self):
        self.qw.setFocus()

    focus = focus_set

    def after(self, ms, func, *args):
        return after(ms, func, *args)

    def after_idle(self, func, *args):
        return after_idle(func, *args)

    def after_cancel(self, timer_id):
        after_cancel(timer_id)

    def update(self):
        pass

    update_idletasks = update

    def toplevel(self):
        w = self
        while w is not None and not isinstance(w, Toplevel):
            w = w.master
        return w

    def _container(self):
        return None


class _Container(Widget):
    '''A widget that other widgets are gridded or packed into.'''

    HSPACING, VSPACING = 8, 3

    def __init__(self, master, qw, padding=None):
        Widget.__init__(self, master, qw)
        self._padding = _pad(padding)
        self._grid_layout = None
        self._box = None
        self._n_front = 0
        self._next_row = 0
        # (row, column) -> the child (or the holder of the children) in it
        self._cells = {}

    def _container(self):
        return self.qw

    def _ensure_grid(self):
        if self._grid_layout is None:
            self._grid_layout = QtWidgets.QGridLayout(self._container())
            self._grid_layout.setContentsMargins(*self._padding)
            self._grid_layout.setHorizontalSpacing(self.HSPACING)
            self._grid_layout.setVerticalSpacing(self.VSPACING)
            # Rows keep their natural height, at the top, rather than being
            # spread over the whole height of the settings panel.
            self._grid_layout.setAlignment(Qt.AlignTop)
        return self._grid_layout

    def _grid_add(self, child, row, column, rowspan, columnspan, sticky):
        layout = self._ensure_grid()
        self._forget(child)
        if row is None:
            row = self._next_row
        self._next_row = max(self._next_row, row + rowspan)
        key = (row, column)
        existing = self._cells.get(key)
        if existing is not None:
            # Tk lets several widgets share a cell, e.g. a label stuck to its
            # west side and an entry to its east. Put them side by side.
            if not isinstance(existing, _CellHolder):
                holder = _CellHolder(self)
                layout.removeWidget(existing.qw)
                holder.add(existing, existing._sticky)
                layout.addWidget(holder.qw, row, column, existing._span[0], existing._span[1])
                holder.qw.show()
                self._cells[key] = existing = holder
            existing.add(child, sticky)
            child._grid_key = key
            return
        child._sticky = sticky
        child._span = (rowspan, columnspan)
        child._grid_key = key
        self._cells[key] = child
        layout.addWidget(child.qw, row, column, rowspan, columnspan, _grid_alignment(sticky))
        child.qw.show()

    def _pack_add(self, child, side, fill, expand):
        self._forget(child)
        horizontal = side in (LEFT, RIGHT)
        if self._box is None:
            direction = QtWidgets.QBoxLayout.LeftToRight if horizontal else QtWidgets.QBoxLayout.TopToBottom
            self._box = QtWidgets.QBoxLayout(direction, self._container())
            self._box.setContentsMargins(*self._padding)
            self._box.setSpacing(self.HSPACING if horizontal else self.VSPACING)
            # Packed widgets gather at the start (and end) of the box, as in Tk
            self._box.addStretch(0)
        box_horizontal = self._box.direction() == QtWidgets.QBoxLayout.LeftToRight
        if box_horizontal:
            cross_fill = fill in (Y, BOTH)
            align = Qt.AlignmentFlag(0) if cross_fill else Qt.AlignVCenter
        else:
            cross_fill = fill in (X, BOTH)
            align = Qt.AlignmentFlag(0) if cross_fill else Qt.AlignLeft
        stretch = 1 if expand else 0
        if side in (RIGHT, BOTTOM):
            self._box.insertWidget(self._n_front + 1, child.qw, stretch, align)
        else:
            self._box.insertWidget(self._n_front, child.qw, stretch, align)
            self._n_front += 1
        child._packed = side not in (RIGHT, BOTTOM)
        child.qw.show()

    def _forget(self, child):
        key = getattr(child, '_grid_key', None)
        if key is not None:
            child._grid_key = None
            if self._cells.get(key) is child:
                del self._cells[key]
            if self._grid_layout is not None:
                self._grid_layout.removeWidget(child.qw)
        if getattr(child, '_packed', None) is not None and self._box is not None:
            if child._packed:
                self._n_front -= 1
            child._packed = None
            self._box.removeWidget(child.qw)

    def columnconfigure(self, index, weight=0, minsize=None, **ignored):
        self._ensure_grid().setColumnStretch(index, weight)
        if minsize:
            self._grid_layout.setColumnMinimumWidth(index, minsize)

    def rowconfigure(self, index, weight=0, minsize=None, **ignored):
        self._ensure_grid().setRowStretch(index, weight)
        if minsize:
            self._grid_layout.setRowMinimumHeight(index, minsize)

    grid_columnconfigure = columnconfigure
    grid_rowconfigure = rowconfigure

    def winfo_children(self):
        return [w._tk_handle for w in self._container().children()
                if isinstance(w, QtWidgets.QWidget) and hasattr(w, '_tk_handle')]


class _CellHolder:
    '''Holds the widgets gridded into the same cell, west ones to the left
    and east ones to the right.'''

    def __init__(self, container):
        self.qw = QtWidgets.QWidget(container._container())
        self._row = QtWidgets.QHBoxLayout(self.qw)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(container.HSPACING)
        self._row.addStretch(1)
        self._n_west = 0

    def add(self, child, sticky):
        sticky = (sticky or '').lower()
        if 'e' in sticky and 'w' not in sticky:
            self._row.addWidget(child.qw)
        else:
            self._row.insertWidget(self._n_west, child.qw)
            self._n_west += 1
        child._grid_key = None
        child.qw.show()


class Frame(_Container):
    def __init__(self, master=None, padding=None, **ignored):
        _Container.__init__(self, master, QtWidgets.QWidget(), padding)


class LabelFrame(_Container):
    def __init__(self, master=None, text='', padding=None, **ignored):
        box = QtWidgets.QGroupBox(_mnemonic_safe(text))
        pad = _pad(padding)
        # Leave room for the title
        _Container.__init__(self, master, box, (pad[0] + 4, pad[1] + 2, pad[2] + 4, pad[3] + 4))

    def _configure(self, kw):
        if 'text' in kw:
            self.qw.setTitle(_mnemonic_safe(kw.pop('text')))
        Widget._configure(self, kw)


class Label(Widget):
    def __init__(self, master=None, text='', textvariable=None, foreground=None,
                 wraplength=None, justify=None, width=None, font=None, **ignored):
        Widget.__init__(self, master, QtWidgets.QLabel(self._text(text)))
        self.qw.setTextInteractionFlags(Qt.NoTextInteraction)
        if foreground is not None:
            self.qw.setStyleSheet('color: %s;' % _color(foreground))
        self._wrap = None
        if wraplength:
            self._set_wrap(int(wraplength))
        if width:
            self.qw.setMinimumWidth(_char_width(self.qw, int(width)))
        if textvariable is not None:
            textvariable._bind(lambda v: self._set_text(v))

    @staticmethod
    def _text(text):
        return str(text).replace('\r', '\n')

    def _set_wrap(self, width):
        self._wrap = width
        self.qw.setWordWrap(True)
        self.qw.setMaximumWidth(width)
        self._fit()

    def _fit(self):
        # A wrapped label in a grid cell is not given the height its wrapped
        # text needs, so ask for it.
        if self._wrap:
            metrics = self.qw.fontMetrics()
            text = self.qw.text()
            natural = max([metrics.horizontalAdvance(line) for line in text.split('\n')] + [0]) + 4
            width = min(self._wrap, natural)
            rect = metrics.boundingRect(QtCore.QRect(0, 0, width, 100000), int(Qt.TextWordWrap), text)
            self.qw.setMinimumSize(width, rect.height() + 2)

    def _set_text(self, text):
        self.qw.setText(self._text(text))
        self._fit()

    def _configure(self, kw):
        if 'text' in kw:
            self._set_text(kw.pop('text'))
        if 'wraplength' in kw:
            self._set_wrap(int(kw.pop('wraplength')))
        Widget._configure(self, kw)


class Button(Widget):
    def __init__(self, master=None, text='', command=None, width=None, state=None, **ignored):
        Widget.__init__(self, master, QtWidgets.QPushButton(_mnemonic_safe(text)))
        # Return applies typed values; it should never also press a button.
        self.qw.setAutoDefault(False)
        self.qw.setDefault(False)
        self._command = command
        self.qw.clicked.connect(self.invoke)
        if state == DISABLED:
            self.qw.setEnabled(False)

    def invoke(self, *args):
        if self._command is not None:
            try:
                self._command()
            except Exception:
                _report()

    def _configure(self, kw):
        if 'command' in kw:
            self._command = kw.pop('command')
        Widget._configure(self, kw)


class Checkbutton(Widget):
    def __init__(self, master=None, text='', variable=None, command=None,
                 onvalue=1, offvalue=0, state=None, **ignored):
        Widget.__init__(self, master, QtWidgets.QCheckBox(_mnemonic_safe(text)))
        self._var = variable if variable is not None else IntVar()
        self._on, self._off = onvalue, offvalue
        self._command = command
        self._var._bind(self._sync)
        self.qw.clicked.connect(self._clicked)
        if state == DISABLED:
            self.qw.setEnabled(False)

    def _sync(self, value):
        self.qw.blockSignals(True)
        self.qw.setChecked(_same(value, self._on))
        self.qw.blockSignals(False)

    def _clicked(self, checked):
        self._var.set(self._on if checked else self._off)
        self.invoke_command()

    def invoke_command(self):
        if self._command is not None:
            try:
                self._command()
            except Exception:
                _report()

    def invoke(self):
        self.qw.click()


class Radiobutton(Widget):
    def __init__(self, master=None, text='', variable=None, value=None, command=None,
                 state=None, **ignored):
        Widget.__init__(self, master, QtWidgets.QRadioButton(_mnemonic_safe(text)))
        # Radio buttons in the same frame may belong to different variables,
        # so the variable, not Qt, decides which of them is checked.
        self.qw.setAutoExclusive(False)
        self._var = variable if variable is not None else StringVar()
        self._value = value
        self._command = command
        self._var._bind(self._sync)
        self.qw.clicked.connect(self._clicked)
        if state == DISABLED:
            self.qw.setEnabled(False)

    def _sync(self, value):
        self.qw.blockSignals(True)
        self.qw.setChecked(_same(value, self._value))
        self.qw.blockSignals(False)

    def _clicked(self, *args):
        self.qw.blockSignals(True)
        self.qw.setChecked(True)
        self.qw.blockSignals(False)
        self._var.set(self._value)
        if self._command is not None:
            try:
                self._command()
            except Exception:
                _report()

    def invoke(self):
        self._clicked()


class _LineEdit(QtWidgets.QLineEdit):
    '''A QLineEdit that asks for as much room as Tk's `width` chars.'''

    chars = 10

    def sizeHint(self):
        hint = QtWidgets.QLineEdit.sizeHint(self)
        return QtCore.QSize(_char_width(self, self.chars), hint.height())

    minimumSizeHint = sizeHint


class Entry(Widget):
    '''A text entry. As in Tk, its variable follows every keystroke. Typed
    values are applied (the settings pane's <Return> binding is called) on
    Return, and also when the entry loses focus with unapplied changes.'''

    def __init__(self, master=None, textvariable=None, width=None, state=None,
                 justify=None, show=None, **ignored):
        Widget.__init__(self, master, _LineEdit())
        self.qw.chars = int(width) if width else 10
        if show:
            self.qw.setEchoMode(QtWidgets.QLineEdit.Password)
        self._var = textvariable
        self._dirty = False
        if textvariable is not None:
            textvariable._bind(self._sync)
        self.qw.textEdited.connect(self._edited)
        self.qw.returnPressed.connect(self._commit)
        self.qw.editingFinished.connect(self._finished)
        if state == DISABLED:
            self.qw.setEnabled(False)

    def _sync(self, value):
        text = str(value)
        if self.qw.text() != text:
            self.qw.setText(text)

    def _edited(self, text):
        self._dirty = True
        if self._var is not None:
            self._var.set(text)

    def _commit(self):
        self._dirty = False
        if self._fire('<Return>'):
            return
        top = self.toplevel()
        if top is not None:
            top._fire('<Return>')

    def _finished(self):
        # Also sent after returnPressed, by which point nothing is left to apply
        if self._dirty:
            self._commit()

    def get(self):
        return self.qw.text()

    def insert(self, index, text):
        current = self.qw.text()
        pos = len(current) if index in (END, INSERT) else int(index)
        self.qw.setText(current[:pos] + str(text) + current[pos:])
        if self._var is not None:
            self._var.set(self.qw.text())

    def delete(self, first, last=None):
        current = self.qw.text()
        first = len(current) if first == END else int(first)
        if last is None:
            last = first + 1
        last = len(current) if last == END else int(last)
        self.qw.setText(current[:first] + current[last:])
        if self._var is not None:
            self._var.set(self.qw.text())


class Spinbox(Widget):
    def __init__(self, master=None, from_=0, to=100, textvariable=None, width=None,
                 increment=1, command=None, **ignored):
        Widget.__init__(self, master, QtWidgets.QSpinBox())
        self.qw.setRange(int(from_), int(to))
        self.qw.setSingleStep(int(increment))
        if width:
            self.qw.setMinimumWidth(_char_width(self.qw, int(width)) + 16)
        self._var = textvariable
        self._command = command
        if textvariable is not None:
            textvariable._bind(self._sync)
        self.qw.valueChanged.connect(self._changed)

    def _sync(self, value):
        try:
            value = int(float(value))
        except (TypeError, ValueError):
            return
        self.qw.blockSignals(True)
        self.qw.setValue(value)
        self.qw.blockSignals(False)

    def _changed(self, value):
        if self._var is not None:
            self._var.set(str(value))
        if self._command is not None:
            try:
                self._command()
            except Exception:
                _report()

    def get(self):
        return str(self.qw.value())

    def set(self, value):
        self.qw.setValue(int(float(value)))

    def _configure(self, kw):
        if 'to' in kw:
            self.qw.setMaximum(int(kw.pop('to')))
        if 'from_' in kw:
            self.qw.setMinimum(int(kw.pop('from_')))
        Widget._configure(self, kw)


class _Choice(Widget):
    '''The common part of OptionMenu and Combobox.'''

    def __init__(self, master, var, values):
        Widget.__init__(self, master, QtWidgets.QComboBox())
        self.qw.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        # A wheel over a closed menu should scroll the settings, not change the choice
        self.qw.setFocusPolicy(Qt.StrongFocus)
        self.qw.installEventFilter(_WHEEL_GUARD)
        self._var = var
        self._set_values(values)
        var._bind(self._sync)
        self.qw.activated.connect(self._activated)

    def _set_values(self, values):
        self._values = list(values)
        self.qw.blockSignals(True)
        self.qw.clear()
        self.qw.addItems([str(v) for v in self._values])
        self.qw.blockSignals(False)
        self._sync(self._var._value)

    def _sync(self, value):
        index = -1
        for i, v in enumerate(self._values):
            if _same(v, value):
                index = i
                break
        self.qw.blockSignals(True)
        self.qw.setCurrentIndex(index)
        self.qw.blockSignals(False)

    def _activated(self, index):
        if 0 <= index < len(self._values):
            self._var.set(self._values[index])
            self._selected(self._values[index])

    def _selected(self, value):
        pass

    def get(self):
        return self._var.get()

    def set(self, value):
        self._var.set(value)


class OptionMenu(_Choice):
    def __init__(self, master, variable, default=None, *values, command=None, **ignored):
        self._command = command
        _Choice.__init__(self, master, variable, values)
        if default:
            variable.set(default)

    def set_menu(self, default=None, *values):
        self._set_values(values)
        if default:
            self._var.set(default)

    def _selected(self, value):
        if self._command is not None:
            try:
                self._command(value)
            except Exception:
                _report()


class Combobox(_Choice):
    def __init__(self, master=None, textvariable=None, values=(), state=None, width=None, **ignored):
        _Choice.__init__(self, master, textvariable if textvariable is not None else StringVar(), values)
        if width:
            self.qw.setMinimumContentsLength(int(width))

    def _selected(self, value):
        self._fire('<<ComboboxSelected>>')

    def current(self, index=None):
        if index is None:
            return self.qw.currentIndex()
        self._var.set(self._values[index])

    def _configure(self, kw):
        if 'values' in kw:
            self._set_values(kw.pop('values'))
        Widget._configure(self, kw)


class _WheelGuard(QtCore.QObject):
    '''Lets a wheel event over an unfocused combobox or slider scroll the
    settings dock instead of changing the value under the cursor.'''

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return False


_WHEEL_GUARD = _WheelGuard()


class Scale(Widget):
    '''A continuous slider from `from_` to `to`. `command` is called with the
    new value whenever it moves; a <ButtonRelease-1> binding is called when
    the user lets go of it, or after a click or key moves it.'''

    def __init__(self, master=None, from_=0, to=1, command=None, orient=HORIZONTAL,
                 variable=None, value=None, length=None, **ignored):
        slider = QtWidgets.QSlider(Qt.Horizontal if orient == HORIZONTAL else Qt.Vertical)
        Widget.__init__(self, master, slider)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.installEventFilter(_WHEEL_GUARD)
        if length:
            slider.setMinimumWidth(int(length))
        self._command = command
        self._var = variable
        self._setting = False
        self._set_range(from_, to)
        slider.valueChanged.connect(self._moved)
        slider.sliderReleased.connect(self._released)
        if variable is not None:
            variable._bind(lambda v: self.set(v))
        elif value is not None:
            self.set(value)

    def _set_range(self, from_, to):
        self._from, self._to = float(from_), float(to)
        span = abs(self._to - self._from)
        # Step through whole units of a wide integer range, else finely
        if span >= 1000 and float(from_).is_integer() and float(to).is_integer():
            self._steps = int(round(span))
        else:
            self._steps = 1000 if span > 0 else 1
        self.qw.blockSignals(True)
        self.qw.setRange(0, self._steps)
        self.qw.setPageStep(max(self._steps // 10, 1))
        self.qw.blockSignals(False)

    def _to_value(self, pos):
        return self._from + (self._to - self._from) * pos / self._steps

    def get(self):
        return self._to_value(self.qw.value())

    def set(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if self._to != self._from:
            pos = round((value - self._from) / (self._to - self._from) * self._steps)
        else:
            pos = 0
        self._setting = True
        try:
            self.qw.setValue(int(min(max(pos, 0), self._steps)))
        finally:
            self._setting = False

    def _moved(self, pos):
        value = self._to_value(pos)
        if self._var is not None and not self._setting:
            self._var.set(value)
        if self._command is not None:
            try:
                self._command(str(value))
            except Exception:
                _report()
        if not self._setting and not self.qw.isSliderDown():
            # Moved by a click on the groove or a key, not by dragging
            self._released()

    def _released(self):
        self._fire('<ButtonRelease-1>')

    def _configure(self, kw):
        if 'to' in kw or 'from_' in kw:
            value = self.get()
            self._set_range(kw.pop('from_', self._from), kw.pop('to', self._to))
            self.set(value)
        Widget._configure(self, kw)


class Separator(Widget):
    def __init__(self, master=None, orient=HORIZONTAL, **ignored):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine if orient == HORIZONTAL else QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        Widget.__init__(self, master, line)


class Notebook(Widget):
    def __init__(self, master=None, **ignored):
        Widget.__init__(self, master, QtWidgets.QTabWidget())
        self.qw.setDocumentMode(True)

    def _container(self):
        return self.qw

    def add(self, child, text='', **ignored):
        index = self.qw.addTab(child.qw, _mnemonic_safe(text))
        # The tab widget shows only the current page
        child.qw.setVisible(index == self.qw.currentIndex())

    def select(self, tab_id=None):
        if tab_id is None:
            return self.qw.currentIndex()
        index = tab_id if isinstance(tab_id, int) else self.qw.indexOf(tab_id.qw)
        self.qw.setCurrentIndex(index)

    def _forget(self, child):
        pass


class Text(Widget):
    '''A multi-line text box. Indices are only honoured as 'start' ("1.0")
    and 'end'.'''

    def __init__(self, master=None, height=None, width=None, **ignored):
        Widget.__init__(self, master, QtWidgets.QPlainTextEdit())
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.qw.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        if width:
            self.qw.setMinimumWidth(metrics.horizontalAdvance('0') * int(width) + 24)
        if height:
            self.qw.setMinimumHeight(metrics.lineSpacing() * int(height) + 12)
        self.yview = lambda *args: None

    def insert(self, index, text):
        cursor = self.qw.textCursor()
        cursor.movePosition(QtGui.QTextCursor.Start if str(index) == '1.0' else QtGui.QTextCursor.End)
        cursor.insertText(str(text))

    def get(self, start='1.0', end=END):
        # Tk ends the text with a newline
        return self.qw.toPlainText() + '\n'

    def delete(self, start='1.0', end=END):
        self.qw.clear()

    def _configure(self, kw):
        kw.pop('yscrollcommand', None)
        Widget._configure(self, kw)


class Scrollbar:
    '''Qt widgets scroll themselves, so this does nothing.'''

    def __init__(self, master=None, **kw):
        self.master = master

    def pack(self, **kw):
        pass

    grid = place = pack

    def config(self, **kw):
        pass

    configure = config

    def set(self, *args):
        pass


####
#
# Windows
#
####

_window_host = [None]


def set_window_host(host):
    '''Make Toplevels tabs of `host`, which has add(top), remove(top) and
    set_title(top, title) methods.'''
    _window_host[0] = host


class _Window(QtWidgets.QWidget):
    '''The Qt side of a Toplevel shown as a window of its own.'''

    def closeEvent(self, event):
        handle = getattr(self, '_tk_handle', None)
        if handle is not None and handle._alive and handle._close_callback is not None:
            event.ignore()
            handle._request_close()
            return
        if handle is not None:
            handle._alive = False
        event.accept()


class Toplevel(_Container):
    '''A settings pane. It becomes a tab of the window host if there is one,
    otherwise a window of its own.'''

    def __init__(self, master=None, padding=6, **ignored):
        _Container.__init__(self, None, _Window(), padding)
        self._alive = True
        self._title = ''
        self._close_callback = None
        self._host = _window_host[0]
        if self._host is not None:
            self._host.add(self)
        else:
            self.qw.setWindowFlag(Qt.Window, True)
            self.qw.show()

    def _container(self):
        return self.qw

    def wm_title(self, title=None):
        if title is None:
            return self._title
        self._title = str(title)
        if self._host is not None:
            self._host.set_title(self, self._title)
        else:
            self.qw.setWindowTitle(self._title)

    title = wm_title

    def protocol(self, name, callback=None):
        if name == 'WM_DELETE_WINDOW':
            self._close_callback = callback

    def _request_close(self):
        '''The user closed the tab or window.'''
        if self._close_callback is not None:
            try:
                self._close_callback()
            except Exception:
                _report()
            if self._alive:
                # The callback did not destroy us; close anyway.
                self.destroy()
        else:
            self.destroy()

    def destroy(self):
        if not self._alive:
            return
        self._alive = False
        if self._host is not None:
            self._host.remove(self)
        try:
            self.qw.hide()
            self.qw.setParent(None)
            self.qw.deleteLater()
        except RuntimeError:
            pass

    def winfo_exists(self):
        return 1 if self._alive else 0

    def lift(self):
        if self._host is not None:
            self._host.show_tab(self)
        else:
            self.qw.raise_()
            self.qw.activateWindow()

    # Window-manager calls that have no meaning in a tab

    def _noop(self, *args, **kw):
        pass

    transient = grab_set = geometry = resizable = minsize = maxsize = _noop
    withdraw = deiconify = iconify = attributes = wait_window = focus_force = _noop


####
#
# Standard dialogs
#
####

def _dialog_parent(parent=None):
    if parent is not None:
        qw = getattr(parent, 'qw', parent)
        if isinstance(qw, QtWidgets.QWidget):
            return qw
    return QtWidgets.QApplication.activeWindow()


class messagebox:
    @staticmethod
    def showwarning(title='', message='', parent=None, **ignored):
        QtWidgets.QMessageBox.warning(_dialog_parent(parent), str(title), str(message))
        return 'ok'

    @staticmethod
    def showerror(title='', message='', parent=None, **ignored):
        QtWidgets.QMessageBox.critical(_dialog_parent(parent), str(title), str(message))
        return 'ok'

    @staticmethod
    def showinfo(title='', message='', parent=None, **ignored):
        QtWidgets.QMessageBox.information(_dialog_parent(parent), str(title), str(message))
        return 'ok'

    @staticmethod
    def askyesno(title='', message='', parent=None, **ignored):
        answer = QtWidgets.QMessageBox.question(_dialog_parent(parent), str(title), str(message),
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        return answer == QtWidgets.QMessageBox.Yes


class filedialog:
    @staticmethod
    def askdirectory(title='', initialdir='', parent=None, **ignored):
        # Qt's own dialog: quicker over VNC than a desktop portal, and the
        # same wherever Iseult runs.
        path = QtWidgets.QFileDialog.getExistingDirectory(
            _dialog_parent(parent), str(title), str(initialdir),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontUseNativeDialog)
        return path or ''


class simpledialog:
    @staticmethod
    def askstring(title='', prompt='', initialvalue='', parent=None, **ignored):
        text, ok = QtWidgets.QInputDialog.getText(_dialog_parent(parent), str(title), str(prompt),
                                                  text=str(initialvalue or ''))
        return text if ok else None
