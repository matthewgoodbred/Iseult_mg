#!/usr/bin/env python
"""The quantities a phase plot can put on either of its axes.

A phase plot is a 2D histogram of particles in any projection of phase space.
Each axis is one of QUANTITIES: a position, a component of the four-velocity
u = gamma*beta, the Lorentz factor, or a component of the three-velocity. The
panel records its choice in two params,

    'phase_x' -- the quantity along the horizontal axis, and
    'phase_y' -- the quantity along the vertical axis.

Views saved before these params existed only knew the old x-vs-momentum plot,
chosen with 'plot_axis' (which position) and 'mom_dim' (which momentum). Those
are still honoured while 'phase_x' / 'phase_y' are unset (None).

Nothing here depends on Tk, so the interactive panel and the headless one
used for movies share it.
"""
import numpy as np
import numpy.ma as ma

from Numba2DHist import Fast2DHist, Fast2DWeightedHist

QUANTITIES = ('x', 'y', 'z', 'ux', 'uy', 'uz', 'gamma', 'betax', 'betay', 'betaz')

SPATIAL = ('x', 'y', 'z')

# What each quantity is called in the settings window.
DISPLAY_NAMES = {'x': 'x',
                 'y': 'y',
                 'z': 'z',
                 'ux': 'ux = γβx',
                 'uy': 'uy = γβy',
                 'uz': 'uz = γβz',
                 'gamma': 'γ',
                 'betax': 'βx = vx/c',
                 'betay': 'βy = vy/c',
                 'betaz': 'βz = vz/c'}

# The momentum keys, indexed by [prtl_type][component]. prtl_type 0 is ions.
MOMENTUM_KEYS = {0: ('ui', 'vi', 'wi'),
                 1: ('ue', 've', 'we')}

WEIGHT_KEYS = {0: 'chi', 1: 'che'}

_COMPONENT = {'x': 0, 'y': 1, 'z': 2}

# The legacy params: 'plot_axis' indexed ('x', 'y'), 'mom_dim' indexed ux, uy, uz.
_LEGACY_POSITIONS = ('x', 'y')
_LEGACY_MOMENTA = ('ux', 'uy', 'uz')


def is_spatial(quantity):
    return quantity in SPATIAL


def _param(get_param, name, default=None):
    try:
        return get_param(name)
    except KeyError:
        return default


def phase_axes(get_param):
    """The (horizontal, vertical) quantities of a phase plot.

    `get_param` looks up one of the panel's params by name.
    """
    horiz = _param(get_param, 'phase_x')
    vert = _param(get_param, 'phase_y')
    if horiz not in QUANTITIES:
        try:
            horiz = _LEGACY_POSITIONS[_param(get_param, 'plot_axis', 0)]
        except (IndexError, TypeError):
            horiz = 'x'
    if vert not in QUANTITIES:
        try:
            vert = _LEGACY_MOMENTA[_param(get_param, 'mom_dim', 0)]
        except (IndexError, TypeError):
            vert = 'ux'
    return horiz, vert


def momentum_keys_needed(prtl_type, quantities, all_components=False):
    """The momentum data keys needed to work out `quantities`.

    A single four-velocity component needs only its own key, but the Lorentz
    factor, the three-velocity, a boost or an energy cut need all three, which
    the caller asks for with `all_components`.
    """
    keys = MOMENTUM_KEYS[prtl_type]
    momenta = [q for q in quantities if not is_spatial(q)]
    if not momenta:
        return list(keys) if all_components else []
    if all_components or any(q not in ('ux', 'uy', 'uz') for q in momenta):
        return list(keys)
    return [keys[_COMPONENT[q[-1]]] for q in dict.fromkeys(momenta)]


####
#
# Lorentz boost along x
#
####

def boost_factors(gamma_boost):
    """The (Gamma, beta) of the boost set by the main 'GammaBoost' setting.

    The setting is read as a Lorentz factor when |value| >= 1 and as a velocity
    otherwise; a negative value boosts in the -x direction. Returns None when
    there is no boost.
    """
    if np.abs(gamma_boost) <= 1E-8:
        return None
    if gamma_boost >= 1:
        big_gamma = gamma_boost
        beta = np.sqrt(1 - 1 / gamma_boost**2)
    elif gamma_boost > -1:
        beta = gamma_boost
        big_gamma = 1 / np.sqrt(1 - beta**2)
    else:
        big_gamma = -gamma_boost
        beta = -np.sqrt(1 - 1 / gamma_boost**2)
    return big_gamma, beta


def four_velocity(u, v, w, boost=None):
    """(ux, uy, uz, gamma) of each particle, boosted along x if `boost` is given.

    `boost` is the (Gamma, beta) pair from boost_factors.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    gamma = np.sqrt(1 + u**2 + v**2 + w**2)
    if boost is not None:
        big_gamma, beta = boost
        u, gamma = big_gamma * (u - beta * gamma), big_gamma * (gamma - beta * u)
    return u, v, w, gamma


class ParticleQuantities:
    """Lazily works out the phase-space quantities of one species.

    `load` takes a data key, e.g. 'ue', and gives back the whole array.
    `positions` takes a physical axis name and gives back the particle
    positions along it in c/omega_pe, or None when the data does not hold them.
    """

    def __init__(self, load, positions, prtl_type, boost=None):
        self._load = load
        self._positions = positions
        self.prtl_type = prtl_type
        self.boost = boost
        self._four_velocity = None
        self._lab_gamma = None
        self._cache = {}

    def _momenta(self):
        if self._four_velocity is None:
            u, v, w = (self._load(key) for key in MOMENTUM_KEYS[self.prtl_type])
            self._four_velocity = four_velocity(u, v, w, self.boost)
        return self._four_velocity

    def lab_gamma(self):
        """The Lorentz factor in the simulation frame, used for energy cuts."""
        if self.boost is None:
            return self._momenta()[3]
        if self._lab_gamma is None:
            u, v, w = (self._load(key) for key in MOMENTUM_KEYS[self.prtl_type])
            self._lab_gamma = four_velocity(u, v, w)[3]
        return self._lab_gamma

    def __call__(self, quantity):
        """The values of `quantity`, or None for a position the data lacks."""
        if quantity not in self._cache:
            self._cache[quantity] = self._compute(quantity)
        return self._cache[quantity]

    def _compute(self, quantity):
        if is_spatial(quantity):
            return self._positions(quantity)
        if self.boost is None and quantity in ('ux', 'uy', 'uz') and self._four_velocity is None:
            # Unboosted, a four-velocity component is stored as is, so there is
            # no need to load the other two.
            return self._load(MOMENTUM_KEYS[self.prtl_type][_COMPONENT[quantity[-1]]])
        ux, uy, uz, gamma = self._momenta()
        if quantity == 'gamma':
            return gamma
        u = (ux, uy, uz)[_COMPONENT[quantity[-1]]]
        if quantity.startswith('beta'):
            return u / gamma
        return u


####
#
# Histogramming
#
####

def data_range(values):
    """The (low, high) of `values`, widened if every value is the same."""
    if len(values) == 0:
        return 0.0, 1.0
    low, high = float(np.min(values)), float(np.max(values))
    if high == low:
        high = low + 1
    return low, high


def limited_range(default, get_param, axis):
    """`default` narrowed or widened by the limits set on one axis of a panel.

    `axis` is 'h' or 'p', the prefix of the panel's 'set_h_min', 'h_min', ...
    or 'set_p_min', 'p_min', ... params. A pair of limits that would leave an
    empty range is ignored.
    """
    low, high = default
    if _param(get_param, 'set_' + axis + '_min', False):
        low = float(get_param(axis + '_min'))
    if _param(get_param, 'set_' + axis + '_max', False):
        high = float(get_param(axis + '_max'))
    if not high > low:
        return default
    return (low, high)


def limits_key(get_param):
    """A string identifying the axis limits that change the binning, for caching."""
    key = ''
    for name in ('h_min', 'h_max', 'p_min', 'p_max'):
        if _param(get_param, 'set_' + name, False):
            key += '_' + name + '_' + str(get_param(name))
    return key


def histogram(horiz_values, vert_values, horiz_range, vert_range,
              horiz_bins, vert_bins, weights=None, masked=True):
    """The normalised 2D histogram of a phase plot.

    Returns (image, vert_range, horiz_range, clim), where the image has the
    vertical quantity along its first axis, as imshow expects.
    """
    horiz_values = np.asarray(horiz_values, dtype=float)
    vert_values = np.asarray(vert_values, dtype=float)
    if weights is not None:
        hist = Fast2DWeightedHist(vert_values, horiz_values, np.asarray(weights, dtype=float),
                                  vert_range[0], vert_range[1], vert_bins,
                                  horiz_range[0], horiz_range[1], horiz_bins)
    else:
        hist = Fast2DHist(vert_values, horiz_values,
                          vert_range[0], vert_range[1], vert_bins,
                          horiz_range[0], horiz_range[1], horiz_bins)
    if not np.any(hist > 0):
        # No particles were binned.
        zval = ma.masked_all(hist.shape) if masked else np.ones(hist.shape)
        return zval, list(vert_range), list(horiz_range), [0.1, 1]
    if masked:
        zval = ma.masked_array(hist)
        zval[zval <= 0] = ma.masked
        zval *= float(zval.max())**(-1)
        clim = [zval[np.logical_not(zval.mask)].min(), zval.max()]
    else:
        zval = np.copy(hist)
        zval[zval == 0] = 0.5
        zval *= float(zval.max())**(-1)
        clim = [zval.min(), zval.max()]
    return zval, list(vert_range), list(horiz_range), clim


####
#
# Labels
#
####

_POSITION_LABELS = {'x': r'$x\ [c/\omega_{\rm pe}]$',
                    'y': r'$y\ [c/\omega_{\rm pe}]$',
                    'z': r'$z\ [c/\omega_{\rm pe}]$'}


def axis_label(quantity, prtl_type, boosted=False):
    """The matplotlib label for an axis showing `quantity`."""
    if is_spatial(quantity):
        # The boost is along x, so only x is a primed coordinate.
        if boosted and quantity == 'x':
            return r'$x\prime\ [c/\omega_{\rm pe}]$'
        return _POSITION_LABELS[quantity]
    s = 'i' if prtl_type == 0 else 'e'
    p = r'\prime' if boosted else ''
    if quantity == 'gamma':
        return rf'$\gamma{p}_{s}$'
    comp = quantity[-1]
    if quantity.startswith('beta'):
        return rf'$\beta{p}_{{{comp},{s}}}$'
    return rf'$\gamma{p}_{s}\beta{p}_{{{comp},{s}}}$'
