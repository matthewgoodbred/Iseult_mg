#!/usr/bin/env python
"""The terms of each species' momentum equation (Ohm's law) and of the total
momentum balance (pressure balance), from the particle stress-energy tensor.

Everything is built from the per-bin particle sums of `stress_energy`: the
number 4-flux N^mu and the stress-energy tensor T^{mu nu} of each species.
These are split covariantly in the Eckart frame, the frame moving with the
particles, U^mu = N^mu / n' with n' = sqrt(-N_mu N^mu) (metric -+++, c = 1):

    T^{mu nu} = e U^mu U^nu + q^mu U^nu + U^mu q^nu + P^{mu nu}

    e          = U_mu U_nu T^{mu nu}                rest-frame energy density
    q^mu       = -Delta^mu_a T^{a b} U_b             heat flux
    P^{mu nu}  = Delta^mu_a Delta^nu_b T^{a b}       pressure (stress) tensor
    Delta^{mu nu} = g^{mu nu} + U^mu U^nu            projector orthogonal to U

and the lab-frame spatial parts of the three pieces are the bulk inertia
e U^i U^j, the heat-flux stress q^i U^j + U^i q^j, and the pressure P^{ij}.
Non-relativistically these reduce to rho V^i V^j, ~0, and the usual thermal
pressure tensor. As P^{mu nu} is the projected tensor, its lab components
include the P Gamma^2 V V part of the relativistic enthalpy flux.

The momentum equation of species s, in conservative form, is

    d_t T^{0i}_s + d_j T^{ij}_s = q_s (N^0_s E^i + (N_s x B)^i)

and dividing by q_s N^0_s gives the species' generalized Ohm's law

    E = -V_s x B + [d_t T^{0i} + d_j(e U U) + d_j P + d_j(q U + U q)] / (q_s n_s)

with V_s = N_s / N^0_s its mean 3-velocity and n_s = N^0_s its lab-frame
density. Summed over species and combined with Maxwell's equations, the lorentz
force becomes the divergence of the Maxwell stress less the rate of change of
the field momentum, which is the pressure (momentum) balance

    0 = -d_t T^{0i} - d_j T^{ij}                       particles
        - d_i B^2/8pi + d_j(B^i B^j)/4pi              magnetic pressure, tension
        - d_i E^2/8pi + d_j(E^i E^j)/4pi              electric pressure, tension
        - d_t (E x B)^i / 4pi c                       field momentum

Units. Both Tristan versions push particles with d(c u)/dt = (q/m)(E + beta x B)
per time step, with E and B in the units they are written out in, and q/m =
+-qi/m for ions/electrons in terms of Iseult's 'qi', 'mi' and 'me'. So the
Ohm's-law terms are electric fields in the same units as the E the Fields
panel shows. The fields are Gaussian with 4 pi replaced by
c^2 / (ppc0 c_omp^2) in Tristan v2 (its `unit_ch`) and by 1 in Tristan v1, with
charges +-qi, so the pressure-balance terms can all be put in units of
n0 m c^2 per c/omega_pe, n0 = ppc0 particles per cell.

The particles are binned in a thin stencil around the slice: along the slice,
and in three bins across it in each transverse direction that the simulation
has, so that the transverse derivatives are central differences across the
slice. Where the main window asks for 1D averages, the whole transverse extent
is one bin and the transverse derivatives are dropped, as they average to zero
in a periodic box.
"""
import numpy as np

import stress_energy as se

# The metric, -+++
ETA = np.array([-1.0, 1.0, 1.0, 1.0])

AXES = ('x', 'y', 'z')
AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


####
#
# The covariant split of the stress-energy tensor
#
####

def four_flux(num):
    '''N^mu from the plain (not mass weighted) sums, shape (4, ...).'''
    return np.stack([num[se.N_], num[se.BX], num[se.BY], num[se.BZ]])


def stress_energy_tensor(mass):
    '''T^{mu nu} from the mass-weighted sums, shape (4, 4, ...).'''
    t = mass
    rows = [[t[se.G_], t[se.UX], t[se.UY], t[se.UZ]],
            [t[se.UX], t[se.TXX], t[se.TXY], t[se.TXZ]],
            [t[se.UY], t[se.TXY], t[se.TYY], t[se.TYZ]],
            [t[se.UZ], t[se.TXZ], t[se.TYZ], t[se.TZZ]]]
    return np.array([np.stack(r) for r in rows])


def eckart_split(N, T):
    '''Split T^{mu nu} in the frame moving with the particle flux N^mu.

    Returns a dict with the rest-frame density 'n_rest', the 4-velocity 'U',
    the rest-frame energy density 'e', the heat flux 'q' and the pressure
    tensor 'P' (all contravariant), plus the lab-frame spatial 3x3 blocks
    'inertia' = e U^i U^j, 'heat' = q^i U^j + U^i q^j and 'pressure' = P^{ij},
    which add up to T^{ij}. Where a bin holds no particles everything is NaN.
    '''
    shape = N.shape[1:]
    eta = ETA.reshape((4,) + (1,) * len(shape))
    n2 = N[0] ** 2 - np.sum(N[1:] ** 2, axis=0)
    n_rest = np.sqrt(np.maximum(n2, 0.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        U = np.where(n_rest > 0, N / n_rest, np.nan)
    U_low = eta * U
    TU = np.einsum('ab...,b...->a...', T, U_low)      # T^{a b} U_b
    e = np.einsum('a...,a...->...', U_low, TU)        # U_a T^{a b} U_b
    q = -(TU + U * e)
    P = (T + np.einsum('a...,b...->ab...', U, TU) + np.einsum('a...,b...->ab...', TU, U)
         + e * np.einsum('a...,b...->ab...', U, U))
    s = slice(1, 4)
    return {'n_rest': n_rest, 'U': U, 'e': e, 'q': q, 'P': P,
            'inertia': e * np.einsum('i...,j...->ij...', U[s], U[s]),
            'heat': (np.einsum('i...,j...->ij...', q[s], U[s])
                     + np.einsum('i...,j...->ij...', U[s], q[s])),
            'pressure': P[s, s]}


####
#
# The stencil around the slice
#
####

class Stencil:
    '''The bins a 1D balance is worked out on.

    Parameters
    ----------
    axis : 'x' or 'y'
        The axis the slice runs along.
    h_edges : array
        The bin edges along the slice, in c/omega_pe.
    trans : dict
        For each of the other two axes, (center, width, n): the slice location
        and the width of each bin across it, in c/omega_pe, and whether there
        are three bins across the slice (n = 3, so the derivative across it
        can be taken) or one (n = 1, no derivative).

    Arrays on the stencil have the shape (..., n2, n1, nh), where 1 is the
    in-plane transverse axis and 2 is z (or y when the slice runs along z).
    '''

    def __init__(self, axis, h_edges, trans):
        self.axis = axis
        self.h_edges = np.asarray(h_edges, dtype=np.float64)
        self.nh = len(self.h_edges) - 1
        self.dh = (self.h_edges[-1] - self.h_edges[0]) / self.nh
        self.h_centers = 0.5 * (self.h_edges[1:] + self.h_edges[:-1])
        others = [a for a in AXES if a != axis]
        # the in-plane transverse axis first
        self.t1, self.t2 = (others if others[0] != 'z' else others[::-1])
        self.trans = trans
        self.n1 = trans[self.t1][2]
        self.n2 = trans[self.t2][2]
        self.c1 = self.n1 // 2
        self.c2 = self.n2 // 2

    @property
    def nv(self):
        return self.n1 * self.n2

    def center(self, arr):
        '''The row of the stencil on the slice itself.'''
        return arr[..., self.c2, self.c1, :]

    def derivative(self, arr, axis):
        '''d arr / d axis on the slice, per c/omega_pe.'''
        if axis == self.axis:
            row = self.center(arr)
            if self.nh < 2:
                return np.zeros_like(row)
            return np.gradient(row, self.dh, axis=-1)
        width = self.trans[axis][1]
        if axis == self.t1:
            if self.n1 < 3:
                return np.zeros_like(self.center(arr))
            return (arr[..., self.c2, 2, :] - arr[..., self.c2, 0, :]) / (2.0 * width)
        if self.n2 < 3:
            return np.zeros_like(self.center(arr))
        return (arr[..., 2, self.c1, :] - arr[..., 0, self.c1, :]) / (2.0 * width)

    def divergence(self, tensor, i):
        '''sum_j d_j tensor[i, j] on the slice, and the three terms separately.

        `tensor` is a 3x3 block of stencil arrays, shape (3, 3, n2, n1, nh).'''
        parts = {a: self.derivative(tensor[i, AXIS_INDEX[a]], a) for a in AXES}
        return parts['x'] + parts['y'] + parts['z'], parts

    def gradient(self, scalar, i):
        '''d_i scalar on the slice.'''
        return self.derivative(scalar, AXES[i])

    def transverse_index(self, positions):
        '''The stencil row of each particle and whether it is in the stencil.

        `positions` maps each transverse axis to the particle positions in
        c/omega_pe, or None if the data does not hold them. Returns (v, mask):
        v is the row, as a float in [0, nv) for binning, and mask says which
        particles are in the stencil, or is None when every one is.'''
        v = None
        mask = None
        for axis, stride in ((self.t1, 1), (self.t2, self.n1)):
            center, width, n = self.trans[axis]
            if n == 1:
                continue
            pos = positions.get(axis)
            if pos is None:
                raise ValueError(f'The particle data has no {axis} positions, '
                                 'which the derivative across the slice needs.')
            f = (np.asarray(pos, dtype=np.float64) - (center - 0.5 * n * width)) / width
            idx = np.floor(f)
            inside = (idx >= 0) & (idx < n)
            mask = inside if mask is None else mask & inside
            idx = np.clip(idx, 0, n - 1)
            v = idx * stride if v is None else v + idx * stride
        if v is None:
            return None, None
        return v + 0.5, mask


def _bin_weights(coord, edges, spacing):
    '''The matrix W such that W @ values is the mean of the grid's linear
    interpolant over each bin between `edges`. Grid values sit on nodes at
    `coord`, and are held constant beyond the first and last node.'''
    n_bins = len(edges) - 1
    size = len(coord)
    widths = np.diff(edges)
    m = int(np.clip(np.ceil(2.0 * np.max(widths) / spacing), 4, 64))
    frac = (np.arange(m) + 0.5) / m
    samples = edges[:-1, None] + widths[:, None] * frac[None, :]
    f = np.clip((samples - coord[0]) / spacing, 0.0, size - 1.0)
    i0 = np.minimum(np.floor(f).astype(int), size - 1)
    i1 = np.minimum(i0 + 1, size - 1)
    t = f - i0
    W = np.zeros((n_bins, size))
    rows = np.repeat(np.arange(n_bins), m)
    np.add.at(W, (rows, i0.ravel()), (1.0 - t).ravel() / m)
    np.add.at(W, (rows, i1.ravel()), t.ravel() / m)
    return W


def bin_grid(arr, stencil, offsets, spacing):
    '''Average a (z, y, x) grid quantity over each bin of the stencil.

    Parameters
    ----------
    arr : array
        The grid values, possibly only a slab of the whole grid.
    offsets : dict
        The grid index of arr[0] along each axis, for a slab.
    spacing : float
        The grid spacing in c/omega_pe (istep / c_omp).

    The grid values sit on the nodes, and each bin gets the mean of their
    linear interpolant over it, so a bin the size of a grid cell gets the value
    at its centre, where the particles in it are centred too. A transverse axis
    with a single bin of no width is averaged over entirely.
    '''
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError('Grid quantities must be 3D (z, y, x) arrays.')
    field_axis = {'z': 0, 'y': 1, 'x': 2}
    # (t2, t1, along the slice)
    work = np.transpose(arr, (field_axis[stencil.t2], field_axis[stencil.t1], field_axis[stencil.axis]))

    def reduce(values, axis_name, dim, edges=None):
        size = values.shape[dim]
        if edges is None:
            center, width, n = stencil.trans[axis_name]
            if width is None:
                return np.mean(values, axis=dim, keepdims=True)
            edges = center - 0.5 * n * width + width * np.arange(n + 1)
        coord = (offsets.get(axis_name, 0) + np.arange(size)) * spacing
        if size == 1:
            return np.repeat(values, len(edges) - 1, axis=dim)
        W = _bin_weights(coord, np.asarray(edges, dtype=np.float64), spacing)
        return np.moveaxis(np.tensordot(W, values, axes=([1], [dim])), 0, dim)

    work = reduce(work, stencil.t2, 0)
    work = reduce(work, stencil.t1, 1)
    return reduce(work, stencil.axis, 2, stencil.h_edges)


def smooth(arr, width):
    '''A boxcar of `width` bins along the last axis, normalized at the edges.'''
    width = min(int(width), arr.shape[-1])
    if width <= 1:
        return arr
    kernel = np.ones(width)
    ones = np.convolve(np.ones(arr.shape[-1]), kernel, mode='same')
    flat = arr.reshape(-1, arr.shape[-1])
    out = np.empty_like(flat, dtype=np.float64)
    for k in range(flat.shape[0]):
        out[k] = np.convolve(flat[k], kernel, mode='same') / ones
    return out.reshape(arr.shape)


####
#
# Time derivatives from the neighbouring outputs
#
####

def time_derivative(center, before=None, after=None):
    '''d/dt from the values at neighbouring times.

    `center` is (t, value); `before` and `after` are (t, value) or None. A
    central difference is used when both neighbours exist, one-sided otherwise.
    Returns None when there is no neighbour.'''
    t0, f0 = center
    if before is not None and after is not None and after[0] > before[0]:
        return (after[1] - before[1]) / (after[0] - before[0])
    if after is not None and after[0] > t0:
        return (after[1] - f0) / (after[0] - t0)
    if before is not None and t0 > before[0]:
        return (f0 - before[1]) / (t0 - before[0])
    return None


####
#
# Ohm's law
#
####

OHM_TERMS = ('E', 'vxb', 'ideal', 'pressure', 'inertia', 'heat', 'dpdt', 'rhs', 'residual')


def ohm_terms(num, stencil, E, B, i, m_over_q, dpdt=None):
    '''Every term of one species' generalized Ohm's law, component i.

    Parameters
    ----------
    num : array (nacc, n2, n1, nh)
        The species' plain sums on the stencil. Only the sums' ratios matter,
        so neither the bin volume nor the particle stride is needed.
    E, B : arrays (3, n2, n1, nh)
        The fields averaged over each bin, in output units.
    i : int
        The component, 0, 1 or 2 for x, y, z.
    m_over_q : float
        (m / q) c^2 / c_omp for the species, which turns the momentum-flux
        divergence per particle into an electric field.
    dpdt : array (3, nh), optional
        d/dt of the momentum sums on the slice, per omega_pe^-1.

    Returns a dict of 1D arrays along the slice, including the pressure
    divergence split by derivative ('pressure_x', ...).
    '''
    N = four_flux(num)
    T = stress_energy_tensor(num)
    split = eckart_split(N, T)
    count = stencil.center(num[se.N_])
    with np.errstate(invalid='ignore', divide='ignore'):
        per = np.where(count > 0, m_over_q / count, np.nan)

    Ec, Bc = stencil.center(E), stencil.center(B)
    with np.errstate(invalid='ignore', divide='ignore'):
        V = np.where(count > 0, stencil.center(N[1:]) / count, np.nan)
    vxb = np.cross(V, Bc, axis=0)[i]

    # E + V x B: what is left of E once the ideal convective part is taken out
    out = {'E': Ec[i], 'vxb': -vxb, 'ideal': Ec[i] + vxb}
    for key in ('pressure', 'inertia', 'heat'):
        total, parts = stencil.divergence(split[key], i)
        out[key] = total * per
        if key == 'pressure':
            for a in AXES:
                out['pressure_' + a] = parts[a] * per
    out['dpdt'] = dpdt[i] * per if dpdt is not None else None
    rhs = out['vxb'] + out['pressure'] + out['inertia'] + out['heat']
    if out['dpdt'] is not None:
        rhs = rhs + out['dpdt']
    out['rhs'] = rhs
    out['residual'] = out['E'] - rhs
    return out


####
#
# Pressure balance
#
####

PB_FORCE_TERMS = ('mag_pressure', 'mag_tension', 'elec_pressure', 'elec_tension', 'em_momentum',
                  'pressure', 'inertia', 'heat', 'dpdt', 'residual')
PB_STRESS_TERMS = ('mag_pressure', 'mag_tension', 'elec_pressure', 'elec_tension',
                   'pressure', 'inertia', 'heat', 'total')


def maxwell_stress(F):
    '''The magnetic (or electric) part of the Maxwell stress, split into its
    isotropic pressure F^2/2 and its tension F^i F^j, in units where 4 pi = 1.'''
    return 0.5 * np.sum(F ** 2, axis=0), np.einsum('i...,j...->ij...', F, F)


def particle_splits(species_sums, masses):
    '''The Eckart split of each species and of all of them together.

    `species_sums` is a list of plain sums, `masses` their masses in the
    reference unit. The total is split in the frame of the total particle
    flux, so it is not the sum of the species' splits, though its T^{ij} is.'''
    splits = []
    for sums, m in zip(species_sums, masses):
        splits.append(eckart_split(four_flux(sums), m * stress_energy_tensor(sums)))
    num, mass = se.combine(species_sums, masses)
    total = eckart_split(four_flux(num), stress_energy_tensor(mass))
    return splits, total


def force_terms(split, stencil, i):
    '''-div of each piece of a species' momentum flux, component i.'''
    out = {}
    for key in ('pressure', 'inertia', 'heat'):
        total, parts = stencil.divergence(split[key], i)
        out[key] = -total
        if key == 'pressure':
            for a in AXES:
                out['pressure_' + a] = -parts[a]
    return out


def field_force_terms(stencil, E, B, i, em_factor, em_momentum_dt=None):
    '''The Maxwell-stress force density, component i.

    `em_factor` turns F^2 into a pressure in the chosen units, i.e.
    1 / (4 pi m c^2 n0). `em_momentum_dt` is d/dt of (E x B)^i on the slice.'''
    out = {}
    for name, F in (('mag', B), ('elec', E)):
        pressure, tension = maxwell_stress(F)
        out[name + '_pressure'] = -stencil.gradient(pressure, i) * em_factor
        out[name + '_tension'] = stencil.divergence(tension, i)[0] * em_factor
    out['em_momentum'] = -em_momentum_dt * em_factor if em_momentum_dt is not None else None
    return out


def stress_terms(split, stencil, i, j):
    '''The (i, j) component of each piece of a species' momentum flux, on the slice.'''
    return {key: stencil.center(split[key][i, j]) for key in ('pressure', 'inertia', 'heat')}


def field_stress_terms(stencil, E, B, i, j, em_factor):
    '''The (i, j) component of the Maxwell momentum flux -sigma^{ij}, split into
    its pressure and tension parts, on the slice. With these signs the
    particle and field fluxes add up to the total, which is conserved.'''
    out = {}
    for name, F in (('mag', B), ('elec', E)):
        pressure, tension = maxwell_stress(stencil.center(F))
        out[name + '_pressure'] = (pressure if i == j else np.zeros_like(pressure)) * em_factor
        out[name + '_tension'] = -tension[i, j] * em_factor
    return out
