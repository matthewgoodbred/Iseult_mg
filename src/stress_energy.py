#!/usr/bin/env python
"""Binned velocity, 4-velocity and stress-energy moments of the particles.

Everything the Moments panel can show is a linear function of a handful of
per-bin particle sums (or, for the rest-frame quantities, a simple function of
those sums). All of them are therefore accumulated together in a single,
multi-threaded pass over the particles by `bin_moments`, and the result is
cached by the panel. Picking a different component, species, basis or
normalization afterwards only re-evaluates `evaluate` on the cached sums, which
costs nothing next to the particle pass.

Units are c = 1. For every particle, with 4-velocity u^mu = gamma (1, beta),
the sums accumulated in each bin are (w is the particle weight)

    N          = sum w                       number
    B_i        = sum w beta_i                number flux  (N^i)
    G          = sum w gamma                 -> T^00 / m
    U_i        = sum w u_i                   -> T^0i / m
    T_ij       = sum w u_i u_j / gamma       -> T^ij / m

and, when a field-aligned basis is asked for, the same quantities projected
along and across the *local* magnetic field direction b = B/|B|, with B
interpolated to each particle's position:

    Bpar       = sum w beta . b
    Bperp_i    = sum w (beta - (beta . b) b)_i
    Upar       = sum w u . b
    Uperp_i    = sum w (u - (u . b) b)_i
    Tparpar    = sum w (u . b)^2 / gamma
    Tperpperp  = sum w |u_perp|^2 / gamma     (both perpendicular directions)
    Tparperp_i = sum w (u . b) u_perp_i / gamma

The perpendicular vectors are summed in the lab basis, so e.g. |<beta_perp>|
is the magnitude of the mean perpendicular drift (the E x B drift for a
magnetized plasma), not the mean of the magnitudes.

The field-aligned pass also sums the fields at the particles, sum w B_i and
sum w E_i, so that the mean field in each bin is known. The rest-frame pressure
tensor needs it: that tensor is only defined once the bin has been averaged
over, so it is projected onto the bin's mean field as seen in the rest frame.

The rest-frame pressure tensor P^ij and temperature P^ij / n are worked out
from the binned lab-frame N^mu and T^munu (see `rest_frame_pressure`), in one
of two rest frames:

    Eckart   U^mu = N^mu / n,  n = sqrt(-N.N)    no particle flux
    Landau   T^mu_nu U^nu = -e U^mu               no energy flux

The two differ when there is heat flux, e.g. for two species streaming
through each other, or a beam through a background.
"""
import math
import os

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads

####
#
# Accumulator layout
#
####

N_, BX, BY, BZ, G_, UX, UY, UZ, TXX, TYY, TZZ, TXY, TXZ, TYZ = range(14)
N_LAB = 14
(BPAR, BPERPX, BPERPY, BPERPZ, UPAR, UPERPX, UPERPY, UPERPZ,
 TPARPAR, TPERPPERP, TPARPERPX, TPARPERPY, TPARPERPZ,
 FBX, FBY, FBZ, FEX, FEY, FEZ) = range(14, 33)
N_FIELD_ALIGNED = 33

# The largest per-thread scratch space the particle pass may use, in bytes.
# Each thread bins its share of the particles into a private copy of the
# histogram so that no atomics are needed; for very fine 2D binnings fewer
# threads are used rather than allocating a copy each.
_SCRATCH_BYTES = 256 * 1024**2
# Below this many particles per thread the threading overhead is not worth it.
_MIN_PER_THREAD = 20000
# Each thread should also bin several particles per bin of its private
# histogram, or zeroing and summing the copies costs more than it saves.
_MIN_PER_BIN = 10


####
#
# The particle pass
#
####

@njit(cache=True, inline='always')
def _lerp_index(f, n):
    '''The two grid points either side of fractional index f, and the weight
    of the upper one. Clamped to the grid, and degenerate for a collapsed axis.'''
    if n == 1 or f <= 0.0:
        return 0, 0, 0.0
    if f >= n - 1:
        return n - 1, n - 1, 0.0
    i0 = int(f)
    return i0, i0 + 1, f - i0


@njit(cache=True)
def _trilinear(arr, k0, k1, tk, j0, j1, tj, i0, i1, ti):
    '''Trilinear interpolation of a (z, y, x) array.'''
    c00 = arr[k0, j0, i0] * (1.0 - ti) + arr[k0, j0, i1] * ti
    c01 = arr[k0, j1, i0] * (1.0 - ti) + arr[k0, j1, i1] * ti
    c10 = arr[k1, j0, i0] * (1.0 - ti) + arr[k1, j0, i1] * ti
    c11 = arr[k1, j1, i0] * (1.0 - ti) + arr[k1, j1, i1] * ti
    c0 = c00 * (1.0 - tj) + c01 * tj
    c1 = c10 * (1.0 - tj) + c11 * tj
    return c0 * (1.0 - tk) + c1 * tk


@njit(parallel=True, cache=True)
def _accumulate(h, v, h0, inv_dh, nh, v0, inv_dv, nv, two_d,
                u, vv, w, wts, mask,
                field_aligned, px, py, pz, pconst, bxg, byg, bzg,
                has_e, exg, eyg, ezg, inv_istep, n_chunks):
    n = u.shape[0]
    nacc = N_FIELD_ALIGNED if field_aligned else N_LAB
    nbins = nh * nv
    scratch = np.zeros((n_chunks, nbins, nacc))
    use_w = wts.shape[0] == n
    use_mask = mask.shape[0] == n
    has_px = px.shape[0] == n
    has_py = py.shape[0] == n
    has_pz = pz.shape[0] == n
    nzg, nyg, nxg = bxg.shape
    chunk = (n + n_chunks - 1) // n_chunks

    for c in prange(n_chunks):
        acc = scratch[c]
        for i in range(c * chunk, min(n, (c + 1) * chunk)):
            if use_mask and not mask[i]:
                continue
            fh = (h[i] - h0) * inv_dh
            if fh < 0.0 or fh >= nh:
                continue
            b = int(fh)
            if two_d:
                fv = (v[i] - v0) * inv_dv
                if fv < 0.0 or fv >= nv:
                    continue
                b += int(fv) * nh

            wt = wts[i] if use_w else 1.0
            ux = float(u[i])
            uy = float(vv[i])
            uz = float(w[i])
            g = math.sqrt(1.0 + ux * ux + uy * uy + uz * uz)
            ig = 1.0 / g
            wg = wt * ig

            acc[b, N_] += wt
            acc[b, BX] += wg * ux
            acc[b, BY] += wg * uy
            acc[b, BZ] += wg * uz
            acc[b, G_] += wt * g
            acc[b, UX] += wt * ux
            acc[b, UY] += wt * uy
            acc[b, UZ] += wt * uz
            acc[b, TXX] += wg * ux * ux
            acc[b, TYY] += wg * uy * uy
            acc[b, TZZ] += wg * uz * uz
            acc[b, TXY] += wg * ux * uy
            acc[b, TXZ] += wg * ux * uz
            acc[b, TYZ] += wg * uy * uz

            if field_aligned:
                # The magnetic field at the particle, from the output grid
                fx = (px[i] if has_px else pconst[0]) * inv_istep
                fy = (py[i] if has_py else pconst[1]) * inv_istep
                fz = (pz[i] if has_pz else pconst[2]) * inv_istep
                i0, i1, ti = _lerp_index(fx, nxg)
                j0, j1, tj = _lerp_index(fy, nyg)
                k0, k1, tk = _lerp_index(fz, nzg)
                bfx = _trilinear(bxg, k0, k1, tk, j0, j1, tj, i0, i1, ti)
                bfy = _trilinear(byg, k0, k1, tk, j0, j1, tj, i0, i1, ti)
                bfz = _trilinear(bzg, k0, k1, tk, j0, j1, tj, i0, i1, ti)
                acc[b, FBX] += wt * bfx
                acc[b, FBY] += wt * bfy
                acc[b, FBZ] += wt * bfz
                if has_e:
                    acc[b, FEX] += wt * _trilinear(exg, k0, k1, tk, j0, j1, tj, i0, i1, ti)
                    acc[b, FEY] += wt * _trilinear(eyg, k0, k1, tk, j0, j1, tj, i0, i1, ti)
                    acc[b, FEZ] += wt * _trilinear(ezg, k0, k1, tk, j0, j1, tj, i0, i1, ti)
                bmag = math.sqrt(bfx * bfx + bfy * bfy + bfz * bfz)
                if bmag > 0.0:
                    hx = bfx / bmag
                    hy = bfy / bmag
                    hz = bfz / bmag
                else:
                    # No field, no preferred direction: all of u is 'perp'
                    hx = 0.0
                    hy = 0.0
                    hz = 0.0
                upar = ux * hx + uy * hy + uz * hz
                upx = ux - upar * hx
                upy = uy - upar * hy
                upz = uz - upar * hz
                acc[b, BPAR] += wg * upar
                acc[b, BPERPX] += wg * upx
                acc[b, BPERPY] += wg * upy
                acc[b, BPERPZ] += wg * upz
                acc[b, UPAR] += wt * upar
                acc[b, UPERPX] += wt * upx
                acc[b, UPERPY] += wt * upy
                acc[b, UPERPZ] += wt * upz
                acc[b, TPARPAR] += wg * upar * upar
                acc[b, TPERPPERP] += wg * (upx * upx + upy * upy + upz * upz)
                acc[b, TPARPERPX] += wg * upar * upx
                acc[b, TPARPERPY] += wg * upar * upy
                acc[b, TPARPERPZ] += wg * upar * upz

    out = np.zeros((nbins, nacc))
    for b in prange(nbins):
        for c in range(n_chunks):
            for k in range(nacc):
                out[b, k] += scratch[c, b, k]
    return out


_EMPTY_F = np.zeros(0)
_EMPTY_B = np.zeros(0, dtype=np.bool_)
_EMPTY_GRID = np.zeros((1, 1, 1), dtype=np.float32)


def _free_cores():
    '''How many cores are not already busy, e.g. on a shared login node.

    Asking for more threads than there are idle cores makes the whole pass
    wait on whichever thread got descheduled, which is far slower than using
    fewer threads to begin with.'''
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    try:
        busy = int(os.getloadavg()[0])
    except (AttributeError, OSError):
        busy = 0
    return max(1, ncpu - busy)


def _n_chunks(n, nbins, nacc):
    '''How many threads, each with its own copy of the histogram, to bin with.'''
    by_threads = min(get_num_threads(), _free_cores())
    by_work = max(1, min(n // _MIN_PER_THREAD, n // (_MIN_PER_BIN * nbins)))
    by_memory = max(1, _SCRATCH_BYTES // max(1, nbins * nacc * 8))
    return int(max(1, min(by_threads, by_work, by_memory)))


def bin_moments(u, v, w, h, h_range, nh, vpos=None, v_range=None, nv=1,
                weights=None, mask=None, bfield=None, positions=None,
                istep=1.0, fallback_position=(0.0, 0.0, 0.0), efield=None):
    '''Bin every moment of one particle species in a single pass.

    Parameters
    ----------
    u, v, w : arrays
        The spatial 4-velocity components gamma*beta_{x,y,z}.
    h : array
        The coordinate binned along the first (horizontal) axis.
    h_range : (low, high)
        The range of `h` covered by the `nh` bins, in the same units as `h`.
    vpos, v_range, nv :
        The same for the second axis of a 2D binning. Leave `vpos` as None
        for a 1D profile.
    weights : array, optional
        A weight for each particle.
    mask : bool array, optional
        Only the particles where `mask` is True are binned.
    bfield : (bx, by, bz), optional
        The magnetic field on the (z, y, x) output grid. When given, the
        field-aligned sums are accumulated as well, with the field
        interpolated to each particle.
    efield : (ex, ey, ez), optional
        The electric field on the same grid as `bfield`, in the same units.
        Only its mean over each bin is used, to find the magnetic field in
        the rest frame. Taken to be zero if not given.
    positions : (x, y, z), optional
        Particle positions in cells, used to interpolate `bfield`. An entry may
        be None for a coordinate the data does not hold, in which case the
        matching entry of `fallback_position` (also in cells) is used.
    istep : float
        The spacing of the field grid in cells.

    Returns
    -------
    array of shape (nacc, nh) for 1D, or (nacc, nv, nh) for 2D, where nacc is
    N_LAB, or N_FIELD_ALIGNED when `bfield` was given.
    '''
    n = len(u)
    two_d = vpos is not None
    if not two_d:
        vpos, v_range, nv = h, (0.0, 1.0), 1
    nh, nv = int(nh), int(nv)
    dh = (h_range[1] - h_range[0]) / nh
    dv = (v_range[1] - v_range[0]) / nv
    inv_dh = 1.0 / dh if dh > 0 else 0.0
    inv_dv = 1.0 / dv if dv > 0 else 0.0

    wts = _EMPTY_F if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
    msk = _EMPTY_B if mask is None else np.ascontiguousarray(mask, dtype=np.bool_)

    field_aligned = bfield is not None
    if field_aligned:
        bxg, byg, bzg = (np.ascontiguousarray(b, dtype=np.float32) for b in bfield)
        positions = positions if positions is not None else (None, None, None)
        px, py, pz = (_EMPTY_F if p is None else p for p in positions)
    else:
        bxg = byg = bzg = _EMPTY_GRID
        px = py = pz = _EMPTY_F
    has_e = field_aligned and efield is not None
    if has_e:
        exg, eyg, ezg = (np.ascontiguousarray(e, dtype=np.float32) for e in efield)
        if any(e.shape != bxg.shape for e in (exg, eyg, ezg)):
            raise ValueError('The electric and magnetic fields must be on the same grid.')
    else:
        exg = eyg = ezg = _EMPTY_GRID
    pconst = np.asarray(fallback_position, dtype=np.float64)

    nacc = N_FIELD_ALIGNED if field_aligned else N_LAB
    n_chunks = _n_chunks(n, nh * nv, nacc)
    # Only start as many threads as there are chunks to work on.
    threads_before = get_num_threads()
    set_num_threads(n_chunks)
    try:
        sums = _accumulate(h, vpos, float(h_range[0]), inv_dh, nh,
                           float(v_range[0]), inv_dv, nv, two_d,
                           u, v, w, wts, msk,
                           field_aligned, px, py, pz, pconst, bxg, byg, bzg,
                           has_e, exg, eyg, ezg, 1.0 / float(istep), n_chunks)
    finally:
        set_num_threads(threads_before)
    sums = sums.T  # (nacc, nbins)
    if two_d:
        return sums.reshape(nacc, nv, nh)
    return sums.reshape(nacc, nh)


####
#
# Which components can be shown
#
####

# The index of each family is the panel's legacy 'm_type', so that configs
# saved before the stress-energy tensor existed still show the same thing.
FAMILIES = ('beta', 'u', 'energy', 'T', 'P', 'Theta', 'U')

FAMILY_NAMES = {'beta': '3-velocity  <β>',
                'u': '4-velocity  <u^μ> = <γ(1, β)>',
                'energy': 'Energy',
                'T': 'Stress-energy tensor  T^μν',
                'P': "Rest-frame pressure  P'^ij",
                'Theta': "Rest-frame temperature  Θ'^ij = P'^ij / n'",
                'U': 'Rest-frame 4-velocity  U^μ'}

# The families that are worked out in the plasma rest frame, and so depend on
# which rest frame is picked
REST_FRAME_FAMILIES = ('P', 'Theta')
# Every family that depends on which rest frame is picked: the rest-frame
# tensors, and the 4-velocity of the rest frame itself
FRAME_FAMILIES = REST_FRAME_FAMILIES + ('U',)

FRAMES = ('eckart', 'landau')
FRAME_NAMES = {'eckart': 'Eckart  (no number flux)',
               'landau': 'Landau  (no energy flux)'}

BASES = ('lab', 'fa')
BASIS_NAMES = {'lab': 'Lab  (x, y, z)',
               'fa': 'Field-aligned  (∥, ⊥ to local B)'}

# The spatial indices of a vector or tensor in each basis
SPATIAL = {'lab': ('x', 'y', 'z'), 'fa': ('par', 'perp')}

# Scalars that do not depend on the basis, shown under the tensor grid
T_INVARIANTS = ('trace', 'e_rest', 'p_rest')
# n_rest is the proper density, the number density in the chosen rest frame
P_INVARIANTS = ('scalar', 'n_rest')
# The temperature anisotropy of the field-aligned basis. The ratio is the same
# for the pressure and the temperature, as the density cancels.
ANISOTROPY = ('par_over_perp', 'perp_over_par')


def tensor_key(a, b, basis):
    '''The component name of T^{ab}, with the indices in canonical order.'''
    order = ('0',) + SPATIAL[basis]
    a, b = sorted((a, b), key=order.index)
    return a + b


def components(family, basis):
    '''Every component of `family` that can be shown in `basis`, in order.'''
    if family == 'beta':
        return list(SPATIAL[basis]) + ['mag']
    if family in ('u', 'U'):
        return ['t'] + list(SPATIAL[basis])
    if family == 'energy':
        return ['ke', 'thermal', 'gamma_bulk']
    if family in REST_FRAME_FAMILIES:
        idx = SPATIAL[basis]
        invariants = P_INVARIANTS
    else:
        idx = ('0',) + SPATIAL[basis]
        invariants = T_INVARIANTS
    comps = [tensor_key(idx[i], idx[j], basis)
             for i in range(len(idx)) for j in range(i, len(idx))]
    if family in REST_FRAME_FAMILIES and basis == 'fa':
        comps += list(ANISOTROPY)
    return comps + list(invariants)


def default_components(family, basis):
    '''What is shown when a family is first picked.'''
    return {'beta': ['x'] if basis == 'lab' else ['par'],
            'u': ['x'] if basis == 'lab' else ['par'],
            'energy': ['ke'],
            'T': ['00'],
            'P': ['scalar'],
            'Theta': ['scalar'],
            'U': ['x'] if basis == 'lab' else ['par']}[family]


def needs_field(family, comp):
    '''Whether a component needs the field-aligned sums.'''
    return family != 'energy' and ('par' in comp or 'perp' in comp)


_UI_INDEX = {'0': '0', 'x': 'x', 'y': 'y', 'z': 'z', 'par': '∥', 'perp': '⊥', 't': '0'}


def _split_tensor(comp):
    '''The two indices of a tensor component name, e.g. '0par' -> ('0', 'par').'''
    for first in ('0', 'x', 'y', 'z', 'par', 'perp'):
        if comp.startswith(first) and comp[len(first):] in ('0', 'x', 'y', 'z', 'par', 'perp'):
            return first, comp[len(first):]
    raise KeyError(comp)


def ui_label(family, comp):
    '''A short plain-text name for a settings-window widget.'''
    special = {'mag': '|β|' if family == 'beta' else '|u|',
               't': 'γ = u^t',
               'ke': 'Kinetic  <γ-1>',
               'thermal': "Thermal, rest frame  <γ'-1>",
               'gamma_bulk': 'Bulk Lorentz factor  Γ',
               'trace': 'Tr(Tij)/3  (mean pressure)',
               'e_rest': "e'  (rest-frame energy density)",
               'p_rest': "P'  (rest-frame pressure)"}
    if family == 'U':
        return {'t': 'U^0 = Γ', 'x': 'U^x', 'y': 'U^y', 'z': 'U^z',
                'par': 'U^∥', 'perp': '|U^⊥|'}[comp]
    if comp in ANISOTROPY:
        return {'par_over_perp': 'T∥ / T⊥', 'perp_over_par': 'T⊥ / T∥'}[comp]
    if family in REST_FRAME_FAMILIES and comp == 'scalar':
        return "Tr/3  (scalar P')" if family == 'P' else "Tr/3  (scalar Θ')"
    if family in REST_FRAME_FAMILIES and comp == 'n_rest':
        return "n'  (proper density)"
    if comp in special:
        return special[comp]
    if family in ('T',) + REST_FRAME_FAMILIES:
        a, b = _split_tensor(comp)
        return _UI_INDEX[a] + _UI_INDEX[b]
    return {'par': '∥', 'perp': '|⊥|'}.get(comp, comp)


_TEX_INDEX = {'0': '0', 'x': 'x', 'y': 'y', 'z': 'z', 'par': r'\parallel', 'perp': r'\perp'}


def tex_label(family, comp, mass_weight=False):
    '''A mathtext label for a legend or 2D annotation, without the $ signs.'''
    if family == 'beta':
        return {'x': r'\langle\beta_x\rangle', 'y': r'\langle\beta_y\rangle',
                'z': r'\langle\beta_z\rangle', 'mag': r'|\langle\beta\rangle|',
                'par': r'\langle\beta_\parallel\rangle',
                'perp': r'|\langle\beta_\perp\rangle|'}[comp]
    if family == 'u':
        sym = 'p' if mass_weight else 'u'
        if comp == 't':
            return r'\langle %s^0\rangle' % sym if mass_weight else r'\langle\gamma\rangle'
        if comp == 'perp':
            return r'|\langle %s_\perp\rangle|' % sym
        return r'\langle %s_{%s}\rangle' % (sym, _TEX_INDEX[comp])
    if family == 'U':
        if comp == 'perp':
            return r'|U^\perp|'
        return r'U^{%s}' % _TEX_INDEX['0' if comp == 't' else comp]
    if family == 'energy':
        return {'ke': r'\langle\gamma-1\rangle',
                'thermal': r"\langle\gamma'-1\rangle",
                'gamma_bulk': r'\Gamma'}[comp]
    if comp == 'par_over_perp':
        return r'T_\parallel/T_\perp'
    if comp == 'perp_over_par':
        return r'T_\perp/T_\parallel'
    if family in REST_FRAME_FAMILIES:
        sym = 'P' if family == 'P' else r'\Theta'
        if comp == 'scalar':
            return sym
        if comp == 'n_rest':
            return "n'"
        if comp == 'parpar':
            return sym + r'_\parallel'
        if comp == 'perpperp':
            return sym + r'_\perp'
        if comp == 'parperp':
            return '|' + sym + r'_{\parallel\perp}|'
        a, b = _split_tensor(comp)
        return sym + '_{%s%s}' % (a, b)
    if comp == 'trace':
        return r'T^{i}_{\ i}/3'
    if comp == 'e_rest':
        return r"e'"
    if comp == 'p_rest':
        return r"P'"
    a, b = _split_tensor(comp)
    body = r'T^{%s%s}' % (_TEX_INDEX[a], _TEX_INDEX[b])
    if 'perp' in comp and comp != 'perpperp':
        # these are the magnitude of a perpendicular vector
        return '|' + body + '|'
    return body


####
#
# Turning the sums into the components
#
####

def combine(sums_list, masses):
    '''Add up the sums of several species.

    Returns (num, mass): the plain sums, and the sums weighted by each
    species' mass. Number-like quantities (<beta>, <u>, counts) come from the
    first, energy and momentum densities from the second.'''
    num = None
    mass = None
    for sums, m in zip(sums_list, masses):
        num = sums.copy() if num is None else num + sums
        mass = m * sums if mass is None else mass + m * sums
    return num, mass


def _safe_div(a, b):
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.true_divide(a, b)
    return np.where(b > 0, out, np.nan)


def _norm3(arr, i):
    return np.sqrt(arr[i] ** 2 + arr[i + 1] ** 2 + arr[i + 2] ** 2)


_T_LAB = {'00': G_, '0x': UX, '0y': UY, '0z': UZ,
          'xx': TXX, 'yy': TYY, 'zz': TZZ, 'xy': TXY, 'xz': TXZ, 'yz': TYZ,
          '0par': UPAR, 'parpar': TPARPAR}


def _rest_frame(num, mass):
    '''The proper number density n' and energy density e' (as sums, i.e. per
    bin rather than per volume), in the Eckart frame, where the particle flux
    vanishes: U^mu = N^mu / n'.'''
    n0 = num[N_]
    nx, ny, nz = num[BX], num[BY], num[BZ]
    n_rest = np.sqrt(np.maximum(n0 * n0 - (nx * nx + ny * ny + nz * nz), 0.0))
    t = mass
    uu = (n0 * n0 * t[G_]
          - 2.0 * n0 * (nx * t[UX] + ny * t[UY] + nz * t[UZ])
          + nx * nx * t[TXX] + ny * ny * t[TYY] + nz * nz * t[TZZ]
          + 2.0 * (nx * ny * t[TXY] + nx * nz * t[TXZ] + ny * nz * t[TYZ]))
    e_rest = _safe_div(uu, n_rest * n_rest)
    return n_rest, e_rest


# The metric, diag(-1, 1, 1, 1)
_ETA = np.array([-1.0, 1.0, 1.0, 1.0])
_LAB_INDEX = {'x': 0, 'y': 1, 'z': 2}


def _lab_tensor(mass):
    '''T^munu in every bin, as an array of shape (bins..., 4, 4).'''
    t = mass
    rows = ((G_, UX, UY, UZ),
            (UX, TXX, TXY, TXZ),
            (UY, TXY, TYY, TYZ),
            (UZ, TXZ, TYZ, TZZ))
    return np.stack([np.stack([t[k] for k in row], axis=-1) for row in rows], axis=-2)


def _number_flux(num):
    '''N^mu in every bin, as an array of shape (bins..., 4).'''
    return np.stack([num[N_], num[BX], num[BY], num[BZ]], axis=-1)


def frame_velocity(num, mass, frame):
    '''The 4-velocity U^mu of the plasma rest frame in every bin, shape
    (bins..., 4), and the number of particles in the bin as counted in that
    frame, n = -N.U (so n / volume is the rest-frame density).

    frame 'eckart' is the frame with no particle flux, U = N / sqrt(-N.N).
    frame 'landau' is the frame with no energy flux, U the timelike
    eigenvector of T^mu_nu. For a physical T^munu (a sum over particles of
    m u^mu u^nu / gamma) that is the eigenvector with the only negative
    eigenvalue, -e.

    Bins where the frame is not defined, e.g. empty ones, are NaN.'''
    N = _number_flux(num)
    if frame == 'eckart':
        n = np.sqrt(np.maximum(N[..., 0] ** 2 - np.sum(N[..., 1:] ** 2, axis=-1), 0.0))
        with np.errstate(invalid='ignore', divide='ignore'):
            U = N / n[..., None]
        U[~(n > 0)] = np.nan
        return U, np.where(n > 0, n, np.nan)
    if frame != 'landau':
        raise KeyError(frame)

    M = _lab_tensor(mass) * _ETA  # T^mu_nu
    valid = np.all(np.isfinite(M), axis=(-2, -1)) & (num[N_] > 0) & (mass[G_] > 0)
    # Something harmless to diagonalize in the bins that are not valid
    M = np.where(valid[..., None, None], M, np.diag(-_ETA))
    lam, vec = np.linalg.eig(M)
    k = np.argmin(lam.real, axis=-1)
    v = np.take_along_axis(vec.real, k[..., None, None], axis=-1)[..., 0]
    norm2 = v[..., 0] ** 2 - np.sum(v[..., 1:] ** 2, axis=-1)
    valid &= norm2 > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        U = v * (np.sign(v[..., 0]) / np.sqrt(np.where(valid, norm2, np.nan)))[..., None]
    U[~valid] = np.nan
    n = N[..., 0] * U[..., 0] - np.sum(N[..., 1:] * U[..., 1:], axis=-1)
    return U, np.where(valid & (n > 0), n, np.nan)


def frame_velocity_component(num, mass, comp, frame):
    '''One component of the 4-velocity U^mu of each bin's own rest frame,
    relative to the lab: 't' (its Lorentz factor), a lab axis, or, along the
    bin's mean lab-frame magnetic field, 'par' and the magnitude 'perp'.
    The field-aligned ones need the field sums of the field-aligned pass.
    Bins where the frame is not defined, e.g. empty ones, are NaN.'''
    U, _ = frame_velocity(num, mass, frame)
    if comp == 't':
        return U[..., 0]
    if comp in _LAB_INDEX:
        return U[..., 1 + _LAB_INDEX[comp]]
    if num.shape[0] < N_FIELD_ALIGNED:
        raise ValueError('The field-aligned basis needs the field-aligned particle pass.')
    B = np.stack([num[FBX], num[FBY], num[FBZ]], axis=-1)
    mag = np.sqrt(np.sum(B ** 2, axis=-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        bhat = B / mag[..., None]
    bhat[~(mag > 0)] = np.nan
    u = U[..., 1:]
    upar = np.sum(u * bhat, axis=-1)
    if comp == 'par':
        return upar
    if comp == 'perp':
        return np.sqrt(np.sum((u - upar[..., None] * bhat) ** 2, axis=-1))
    raise KeyError(comp)


def boost_to_rest(U):
    '''The pure boost Lambda^mu_nu that takes the lab frame to the frame
    moving with 4-velocity U, shape (bins..., 4, 4). Its spatial axes are
    the lab x, y and z, boosted without a rotation.'''
    g = U[..., 0]
    u = U[..., 1:]
    L = np.empty(U.shape + (4,))
    L[..., 0, 0] = g
    L[..., 0, 1:] = -u
    L[..., 1:, 0] = -u
    with np.errstate(invalid='ignore', divide='ignore'):
        L[..., 1:, 1:] = np.eye(3) + u[..., :, None] * u[..., None, :] / (g + 1.0)[..., None, None]
    return L


def rest_frame_pressure(num, mass, frame):
    '''The pressure tensor in the rest frame, as a sum over the bin.

    Returns (P, n, U, L): P^ij in the rest frame, shape (bins..., 3, 3); the
    rest-frame particle count n; the frame's 4-velocity U; and the boost L into it.

    In the rest frame T'^munu = e U'U' + P' + q'U' + U'q' with U' = (1, 0), so
    P'^ij is simply the spatial part of the boosted T'^munu. In the Landau frame
    the energy flux q' vanishes, in the Eckart frame it is the heat flux.'''
    U, n = frame_velocity(num, mass, frame)
    L = boost_to_rest(U)
    T = np.einsum('...ma,...ab,...nb->...mn', L, _lab_tensor(mass), L)
    return T[..., 1:, 1:], n, U, L


def rest_frame_field_direction(num, U, L):
    '''The direction of the mean magnetic field of each bin as seen in the
    rest frame, as a unit 3-vector in the rest frame's (boosted x, y, z) axes.

    The field the plasma sees is the 4-vector b^mu = -*F^munu U_nu, which in
    terms of the lab fields and U = (gamma, u) is
        b^mu = (u . B,  gamma B - u x E),
    and is orthogonal to U, so it is purely spatial in the rest frame.
    It needs the field sums of the field-aligned pass. Bins with no field are NaN.'''
    if num.shape[0] < N_FIELD_ALIGNED:
        raise ValueError('The field-aligned basis needs the field-aligned particle pass.')
    count = num[N_]
    with np.errstate(invalid='ignore', divide='ignore'):
        B = np.stack([num[FBX], num[FBY], num[FBZ]], axis=-1) / count[..., None]
        E = np.stack([num[FEX], num[FEY], num[FEZ]], axis=-1) / count[..., None]
    g = U[..., 0]
    u = U[..., 1:]
    b4 = np.concatenate([np.sum(u * B, axis=-1)[..., None],
                         g[..., None] * B - np.cross(u, E)], axis=-1)
    b_rest = np.einsum('...ma,...a->...m', L, b4)[..., 1:]
    mag = np.sqrt(np.sum(b_rest ** 2, axis=-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        bhat = b_rest / mag[..., None]
    bhat[~(mag > 0)] = np.nan
    return bhat


def pressure_component(P, comp, bhat=None):
    '''One component of a (bins..., 3, 3) tensor: a lab-axis one such as 'xy',
    'scalar' (a third of the trace), or, given the field direction `bhat`,
    'parpar', 'perpperp' (per perpendicular direction) or 'parperp' (the
    magnitude of the parallel-perpendicular part).'''
    trace = P[..., 0, 0] + P[..., 1, 1] + P[..., 2, 2]
    if comp == 'scalar':
        return trace / 3.0
    if comp in ANISOTROPY:
        ppar = pressure_component(P, 'parpar', bhat)
        pperp = pressure_component(P, 'perpperp', bhat)
        if comp == 'par_over_perp':
            return _safe_div(ppar, pperp)
        return _safe_div(pperp, ppar)
    if comp in ('parpar', 'perpperp', 'parperp'):
        Pb = np.einsum('...ij,...j->...i', P, bhat)
        ppar = np.sum(bhat * Pb, axis=-1)
        if comp == 'parpar':
            return ppar
        if comp == 'perpperp':
            return (trace - ppar) / 2.0
        return np.sqrt(np.sum((Pb - ppar[..., None] * bhat) ** 2, axis=-1))
    a, b = _split_tensor(comp)
    return P[..., _LAB_INDEX[a], _LAB_INDEX[b]]


def evaluate(num, mass, family, comp, normalization='density',
             dens_factor=1.0, mass_weight=False, rest_frame='eckart'):
    '''The value of one component in every bin.

    Parameters
    ----------
    num, mass : arrays
        As returned by `combine`.
    family, comp :
        Which quantity, e.g. ('T', '0x') or ('beta', 'perp').
    normalization : 'density' or 'particle'
        For the stress-energy tensor only: per unit volume (the sums times
        `dens_factor`), or per particle (the sums divided by the number of
        particles in the bin).
    mass_weight : bool
        For the 4-velocity only: show the momentum m u rather than u.
    rest_frame : 'eckart' or 'landau'
        For the rest-frame pressure, temperature and 4-velocity only: which rest frame.
        The pressure and the proper density n_rest are always per unit volume,
        the temperature P / n per particle counted in the rest frame.

    Bins holding no particles come back as NaN, except for a density, which is
    then genuinely zero.
    '''
    count = num[N_]
    if family == 'beta':
        if comp == 'mag':
            return _safe_div(_norm3(num, BX), count)
        if comp == 'perp':
            return _safe_div(_norm3(num, BPERPX), count)
        return _safe_div(num[{'x': BX, 'y': BY, 'z': BZ, 'par': BPAR}[comp]], count)

    if family == 'u':
        src = mass if mass_weight else num
        if comp == 'mag':
            return _safe_div(_norm3(src, UX), count)
        if comp == 'perp':
            return _safe_div(_norm3(src, UPERPX), count)
        return _safe_div(src[{'t': G_, 'x': UX, 'y': UY, 'z': UZ, 'par': UPAR}[comp]], count)

    if family == 'energy':
        if comp == 'ke':
            return _safe_div(mass[G_] - mass[N_], count)
        n_rest, e_rest = _rest_frame(num, mass)
        if comp == 'gamma_bulk':
            return _safe_div(count, n_rest)
        # thermal: rest-frame energy per particle less its rest mass
        return _safe_div(e_rest, n_rest) - _safe_div(mass[N_], count)

    if family == 'U':
        return frame_velocity_component(num, mass, comp, rest_frame)

    if family in REST_FRAME_FAMILIES and comp == 'n_rest':
        # The particles the rest frame counts, n = -N.U, per unit volume
        _, n = frame_velocity(num, mass, rest_frame)
        return np.where(count > 0, n, 0.0) * dens_factor

    if family in REST_FRAME_FAMILIES:
        P, n, U, L = rest_frame_pressure(num, mass, rest_frame)
        bhat = rest_frame_field_direction(num, U, L) if needs_field(family, comp) else None
        raw = pressure_component(P, comp, bhat)
        if comp in ANISOTROPY:
            # a dimensionless ratio, NaN where the bin is empty
            return raw
        if family == 'Theta':
            return _safe_div(raw, n)
        return np.where(count > 0, raw, 0.0) * dens_factor

    # The stress-energy tensor, as a sum over the bin
    if comp in ('e_rest', 'p_rest'):
        n_rest, raw = _rest_frame(num, mass)
        raw = np.where(n_rest > 0, raw, 0.0)
        if comp == 'p_rest':
            trace4 = -mass[G_] + mass[TXX] + mass[TYY] + mass[TZZ]
            raw = (trace4 + raw) / 3.0
        per = n_rest
    else:
        if comp in _T_LAB:
            raw = mass[_T_LAB[comp]]
        elif comp == 'trace':
            raw = (mass[TXX] + mass[TYY] + mass[TZZ]) / 3.0
        elif comp == '0perp':
            raw = _norm3(mass, UPERPX)
        elif comp == 'parperp':
            raw = _norm3(mass, TPARPERPX)
        elif comp == 'perpperp':
            # per perpendicular direction, i.e. the gyrotropic P_perp
            raw = mass[TPERPPERP] / 2.0
        else:
            raise KeyError(comp)
        per = count

    if normalization == 'particle':
        return _safe_div(raw, per)
    raw = np.where(count > 0, raw, 0.0)
    return raw * dens_factor
