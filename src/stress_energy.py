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
 TPARPAR, TPERPPERP, TPARPERPX, TPARPERPY, TPARPERPZ) = range(14, 27)
N_FIELD_ALIGNED = 27

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
                field_aligned, px, py, pz, pconst, bxg, byg, bzg, inv_istep,
                n_chunks):
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
                istep=1.0, fallback_position=(0.0, 0.0, 0.0)):
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
                           1.0 / float(istep), n_chunks)
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
FAMILIES = ('beta', 'u', 'energy', 'T')

FAMILY_NAMES = {'beta': '3-velocity  <β>',
                'u': '4-velocity  <uᵘ> = <γ(1, β)>',
                'energy': 'Energy',
                'T': 'Stress-energy tensor  Tᵘᵛ'}

BASES = ('lab', 'fa')
BASIS_NAMES = {'lab': 'Lab  (x, y, z)',
               'fa': 'Field-aligned  (∥, ⊥ to local B)'}

# The spatial indices of a vector or tensor in each basis
SPATIAL = {'lab': ('x', 'y', 'z'), 'fa': ('par', 'perp')}

# Scalars that do not depend on the basis, shown under the tensor grid
T_INVARIANTS = ('trace', 'e_rest', 'p_rest')


def tensor_key(a, b, basis):
    '''The component name of T^{ab}, with the indices in canonical order.'''
    order = ('0',) + SPATIAL[basis]
    a, b = sorted((a, b), key=order.index)
    return a + b


def components(family, basis):
    '''Every component of `family` that can be shown in `basis`, in order.'''
    if family == 'beta':
        return list(SPATIAL[basis]) + ['mag']
    if family == 'u':
        return ['t'] + list(SPATIAL[basis])
    if family == 'energy':
        return ['ke', 'thermal', 'gamma_bulk']
    idx = ('0',) + SPATIAL[basis]
    comps = [tensor_key(idx[i], idx[j], basis)
             for i in range(len(idx)) for j in range(i, len(idx))]
    return comps + list(T_INVARIANTS)


def default_components(family, basis):
    '''What is shown when a family is first picked.'''
    return {'beta': ['x'] if basis == 'lab' else ['par'],
            'u': ['x'] if basis == 'lab' else ['par'],
            'energy': ['ke'],
            'T': ['00']}[family]


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
               't': 'γ = uᵗ',
               'ke': 'Kinetic  <γ-1>',
               'thermal': "Thermal, rest frame  <γ'-1>",
               'gamma_bulk': 'Bulk Lorentz factor  Γ',
               'trace': 'Tr(Tij)/3  (mean pressure)',
               'e_rest': "e'  (rest-frame energy density)",
               'p_rest': "P'  (rest-frame pressure)"}
    if comp in special:
        return special[comp]
    if family == 'T':
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
    if family == 'energy':
        return {'ke': r'\langle\gamma-1\rangle',
                'thermal': r"\langle\gamma'-1\rangle",
                'gamma_bulk': r'\Gamma'}[comp]
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


def evaluate(num, mass, family, comp, normalization='density',
             dens_factor=1.0, mass_weight=False):
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
