import numpy as np
import pytest

import fluid_balance as fb
import stress_energy as se


def one_d_stencil(length, nh):
    return fb.Stencil('x', np.linspace(0.0, length, nh + 1), {'y': (None, None, 1), 'z': (None, None, 1)})


def bin_1d(stencil, x, u, v, w, mask=None):
    sums = se.bin_moments(u, v, w, x, (stencil.h_edges[0], stencil.h_edges[-1]), stencil.nh,
                          vpos=np.full(len(u), 0.5), v_range=(0.0, 1.0), nv=1, mask=mask)
    return sums.reshape(se.N_LAB, 1, 1, stencil.nh)


def boosted_maxwellian(n, theta, gamma_boost, seed=0):
    '''Particles isotropic in their rest frame, boosted along x.

    A particle's share of a fixed lab-frame volume goes as gamma / gamma', as
    d^3p / gamma is invariant, so each boosted particle carries that weight.'''
    rng = np.random.default_rng(seed)
    up = rng.normal(0.0, np.sqrt(theta), (3, n))
    gp = np.sqrt(1 + np.sum(up ** 2, axis=0))
    bb = np.sqrt(1 - 1 / gamma_boost ** 2)
    ux = gamma_boost * (up[0] + bb * gp)
    g = np.sqrt(1 + ux ** 2 + up[1] ** 2 + up[2] ** 2)
    return ux, up[1], up[2], g / gp


def test_eckart_split_adds_up_and_is_orthogonal():
    rng = np.random.default_rng(1)
    u, v, w = rng.normal(0.4, 1.0, (3, 5000))
    x = rng.uniform(0, 10, 5000)
    sums = se.bin_moments(u, v, w, x, (0, 10), 7)
    N, T = fb.four_flux(sums), fb.stress_energy_tensor(sums)
    split = fb.eckart_split(N, T)
    np.testing.assert_allclose(split['inertia'] + split['heat'] + split['pressure'], T[1:, 1:], rtol=1e-9, atol=1e-9)
    U_low = fb.ETA[:, None] * split['U']
    # U is a unit timelike vector, q and P are orthogonal to it
    np.testing.assert_allclose(np.sum(U_low * split['U'], axis=0), -1.0, rtol=1e-12)
    np.testing.assert_allclose(np.sum(U_low * split['q'], axis=0), 0.0, atol=1e-8 * np.abs(split['e']).max())
    np.testing.assert_allclose(np.einsum('ab...,b...->a...', split['P'], U_low), 0.0,
                               atol=1e-8 * np.abs(split['e']).max())


def test_eckart_split_of_a_boosted_plasma():
    '''A boosted isotropic plasma: the frame is the boost, there is no heat
    flux, and the rest-frame pressure is isotropic.'''
    theta, gamma_boost = 0.5, 2.0
    ux, uy, uz, weights = boosted_maxwellian(400000, theta, gamma_boost)
    x = np.zeros_like(ux) + 0.5
    sums = se.bin_moments(ux, uy, uz, x, (0, 1), 1, weights=weights)
    split = fb.eckart_split(fb.four_flux(sums), fb.stress_energy_tensor(sums))
    U = split['U'][:, 0]
    assert U[0] == pytest.approx(gamma_boost, rel=2e-3)
    assert abs(U[2]) < 5e-3 and abs(U[3]) < 5e-3
    e = split['e'][0]
    # heat flux is small next to the energy flux
    assert np.max(np.abs(split['q'][:, 0])) < 5e-3 * e
    # boost P back to the rest frame: P' = diag(p, p, p)
    b = np.sqrt(1 - 1 / gamma_boost ** 2)
    L = np.array([[gamma_boost, -gamma_boost * b, 0, 0], [-gamma_boost * b, gamma_boost, 0, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]])
    P_rest = L @ split['P'][:, :, 0] @ L.T
    p = np.trace(P_rest[1:, 1:]) / 3
    np.testing.assert_allclose(np.diag(P_rest)[1:], p, rtol=1e-2)
    assert np.max(np.abs(P_rest[0])) < 1e-2 * p
    # and it is the pressure of the rest-frame sample, <u'_x^2 / gamma'> per particle
    rng = np.random.default_rng(0)
    up = rng.normal(0.0, np.sqrt(theta), (3, 400000))
    p_expected = np.mean(up[0] ** 2 / np.sqrt(1 + np.sum(up ** 2, axis=0)))
    assert p / split['n_rest'][0] == pytest.approx(p_expected, rel=1e-2)


def test_stencil_derivatives_of_a_linear_field():
    '''bin_grid then the stencil derivatives recover the gradient exactly.'''
    nz, ny, nx = 6, 20, 40
    spacing = 0.5
    z, y, x = np.meshgrid(*(np.arange(n) * spacing for n in (nz, ny, nx)), indexing='ij')
    f = 3.0 * x - 2.0 * y + 0.7 * z
    trans = {'y': (4.0, 1.0, 3), 'z': (1.5, 0.5, 3)}
    stencil = fb.Stencil('x', np.linspace(0, nx * spacing, 21), trans)
    binned = fb.bin_grid(f, stencil, {}, spacing)
    assert binned.shape == (3, 3, 20)
    interior = slice(1, -2)
    np.testing.assert_allclose(stencil.derivative(binned, 'x')[interior], 3.0, rtol=1e-12)
    np.testing.assert_allclose(stencil.derivative(binned, 'y'), -2.0, rtol=1e-12)
    np.testing.assert_allclose(stencil.derivative(binned, 'z'), 0.7, rtol=1e-12)


def test_bin_grid_on_a_slab_matches_the_whole_grid():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(8, 30, 50))
    spacing = 0.25
    stencil = fb.Stencil('y', np.linspace(0, 30 * spacing, 16), {'x': (6.0, 0.5, 3), 'z': (1.0, 0.5, 3)})
    whole = fb.bin_grid(f, stencil, {}, spacing)
    slab = fb.bin_grid(f[1:8, :, 15:35], stencil, {'z': 1, 'x': 15}, spacing)
    np.testing.assert_allclose(slab, whole)


def test_transverse_index_selects_the_stencil_rows():
    stencil = fb.Stencil('x', np.linspace(0, 10, 11), {'y': (5.0, 1.0, 3), 'z': (None, None, 1)})
    y = np.array([3.4, 3.6, 4.4, 4.6, 5.0, 6.49, 6.51])
    v, mask = stencil.transverse_index({'y': y})
    np.testing.assert_array_equal(mask, [False, True, True, True, True, True, False])
    np.testing.assert_array_equal(np.floor(v[mask]), [0, 0, 1, 1, 2])


def test_pressure_gradient_force():
    '''A static plasma with a density gradient and a uniform temperature:
    -d_x P_xx = -theta dn/dx, and the bulk inertia vanishes.'''
    rng = np.random.default_rng(3)
    L, n = 100.0, 3_000_000
    k = 2 * np.pi / L
    # density 1 + 0.5 sin(kx), by rejection sampling
    x = rng.uniform(0, L, 2 * n)
    x = x[rng.uniform(0, 1.5, 2 * n) < 1 + 0.5 * np.sin(k * x)][:n]
    theta = 0.01
    u, v, w = rng.normal(0.0, np.sqrt(theta), (3, len(x)))
    stencil = one_d_stencil(L, 50)
    sums = bin_1d(stencil, x, u, v, w)
    splits, total = fb.particle_splits([sums], [1.0])
    terms = fb.force_terms(total, stencil, 0)
    n_per_bin = len(x) / 50 / (1.0)                 # mean particles per bin
    expected = -theta * n_per_bin * 0.5 * k * np.cos(k * stencil.h_centers)
    interior = slice(2, -2)
    err = terms['pressure'][interior] - expected[interior]
    assert np.sqrt(np.mean(err ** 2)) < 0.1 * np.max(np.abs(expected))
    assert np.max(np.abs(terms['inertia'][interior])) < 0.05 * np.max(np.abs(expected))


def boris_push(u, v, w, E, B, q_over_m, c):
    '''One Tristan step, fields in output units: d(c u) = (q/m)(E + beta x B).'''
    ex0, ey0, ez0 = (0.5 * q_over_m * e for e in E)
    bx0, by0, bz0 = (0.5 * q_over_m * b / c for b in B)
    u0, v0, w0 = c * u + ex0, c * v + ey0, c * w + ez0
    g = c / np.sqrt(c ** 2 + u0 ** 2 + v0 ** 2 + w0 ** 2)
    bx0, by0, bz0 = g * bx0, g * by0, g * bz0
    f = 2.0 / (1.0 + bx0 ** 2 + by0 ** 2 + bz0 ** 2)
    u1 = (u0 + v0 * bz0 - w0 * by0) * f
    v1 = (v0 + w0 * bx0 - u0 * bz0) * f
    w1 = (w0 + u0 * by0 - v0 * bx0) * f
    u0 = (u0 + v1 * bz0 - w1 * by0 + ex0) / c
    v0 = (v0 + w1 * bx0 - u1 * bz0 + ey0) / c
    w0 = (w0 + u1 * by0 - v1 * bx0 + ez0) / c
    return u0, v0, w0


@pytest.mark.parametrize('q_over_m', [-1.0, 1.0 / 25.0])
def test_ohms_law_closes_with_tristan_units(q_over_m):
    '''Push a uniform plasma in uniform fields that are not in E x B balance,
    so it accelerates. With the time derivative from the neighbouring
    "outputs", the Ohm's-law residual must vanish in Tristan's units.'''
    rng = np.random.default_rng(4)
    c, c_omp = 0.45, 10.0
    E = np.array([0.002, 0.004, -0.001])
    B = np.array([0.001, -0.002, 0.01])
    n = 200000
    L = 200.0  # cells
    x = rng.uniform(0, L, n)
    u, v, w = rng.normal(0.05, 0.2, (3, n))
    interval = 5
    snapshots = []
    for step in range(3 * interval + 1):
        if step % interval == 0:
            snapshots.append((step * c / c_omp, x / c_omp, u.copy(), v.copy(), w.copy()))
        u, v, w = boris_push(u, v, w, E, B, q_over_m, c)
        g = np.sqrt(1 + u ** 2 + v ** 2 + w ** 2)
        x = (x + c * u / g) % L
    stencil = one_d_stencil(L / c_omp, 4)
    sums = [bin_1d(stencil, sx, su, sv, sw) for _, sx, su, sv, sw in snapshots]
    times = [s[0] for s in snapshots]

    def mom(s):
        return stencil.center(s[[se.UX, se.UY, se.UZ]])
    dpdt = fb.time_derivative((times[1], mom(sums[1])), (times[0], mom(sums[0])), (times[2], mom(sums[2])))
    fields_E = np.broadcast_to(E[:, None, None, None], (3, 1, 1, stencil.nh))
    fields_B = np.broadcast_to(B[:, None, None, None], (3, 1, 1, stencil.nh))
    m_over_q = 1.0 / q_over_m * c ** 2 / c_omp
    for i in range(3):
        terms = fb.ohm_terms(sums[1], stencil, fields_E, fields_B, i, m_over_q, dpdt=dpdt)
        # On average over the box the balance is exact up to the push's own
        # discretization. Bin by bin, particles crossing between bins add
        # noise to d/dt that the flux divergence only balances statistically,
        # and m/q amplifies it for the ions.
        assert abs(np.mean(terms['residual'])) < 1e-2 * np.max(np.abs(E))
        np.testing.assert_allclose(terms['residual'], 0.0, atol=0.12 * np.max(np.abs(E)))
        assert np.max(np.abs(terms['pressure'])) < 0.2 * np.max(np.abs(E))
        # and the time derivative matters: without it the balance fails
        assert np.max(np.abs(terms['dpdt'])) > 0.1 * np.max(np.abs(E))
        # E + V x B is E with the convective term taken out
        np.testing.assert_allclose(terms['ideal'], terms['E'] - terms['vxb'])


def test_magnetic_pressure_and_tension_of_a_sheared_field():
    '''B = (1, tanh x, 0): -d_x B^2/2 = -tanh sech^2, and d_j(B_x B_j) = d_x(B_x^2) = 0
    while d_j(B_y B_j) = d_x(B_y B_x) = sech^2.'''
    nx = 400
    spacing = 0.05
    xg = (np.arange(nx) - nx / 2) * spacing
    B = np.zeros((3, 1, 1, nx))
    B[0] = 1.0
    B[1] = np.tanh(xg)
    E = np.zeros_like(B)
    stencil = fb.Stencil('x', np.linspace(0, nx * spacing, nx + 1), {'y': (None, None, 1), 'z': (None, None, 1)})
    bins = [fb.bin_grid(B[k], stencil, {}, spacing) for k in range(3)]
    Bb = np.stack(bins)
    fx = fb.field_force_terms(stencil, np.zeros_like(Bb), Bb, 0, 1.0)
    fy = fb.field_force_terms(stencil, np.zeros_like(Bb), Bb, 1, 1.0)
    interior = slice(2, -2)
    # the bins are centred half a cell past the grid nodes
    xc = stencil.h_centers - nx / 2 * spacing
    sech2 = 1 / np.cosh(xc) ** 2
    np.testing.assert_allclose(fx['mag_pressure'][interior], -(np.tanh(xc) * sech2)[interior], atol=2e-3)
    np.testing.assert_allclose(fx['mag_tension'][interior], 0.0, atol=1e-12)
    np.testing.assert_allclose(fy['mag_tension'][interior], sech2[interior], atol=2e-3)
    np.testing.assert_allclose(fy['mag_pressure'], 0.0)


def test_integrated_magnetic_pressure_is_b_squared_over_two():
    '''int (-d_x B^2/2) dx, with the constant matched to the stress, is -B^2/2.'''
    nx = 400
    spacing = 0.05
    xg = (np.arange(nx) - nx / 2) * spacing
    B = np.zeros((3, 1, 1, nx))
    B[0] = 1.0
    B[1] = np.tanh(xg)
    stencil = fb.Stencil('x', np.linspace(0, nx * spacing, nx + 1), {'y': (None, None, 1), 'z': (None, None, 1)})
    Bb = np.stack([fb.bin_grid(B[k], stencil, {}, spacing) for k in range(3)])
    E = np.zeros_like(Bb)
    force = fb.field_force_terms(stencil, E, Bb, 0, 1.0)
    stress = fb.field_stress_terms(stencil, E, Bb, 0, 0, 1.0)
    integral = fb.integrate_force(force['mag_pressure'], stencil.dh, stress['mag_pressure'])
    np.testing.assert_allclose(integral, -stress['mag_pressure'], atol=2e-3)
    # the along-slice part of the tension integrates to +B_x B_x
    integral = fb.integrate_force(force['mag_tension'], stencil.dh, stress['mag_tension'])
    np.testing.assert_allclose(integral, 1.0, atol=1e-12)


def test_integrate_force_skips_empty_bins():
    force = np.array([1.0, np.nan, 1.0, 1.0])
    out = fb.integrate_force(force, 1.0)
    assert np.isnan(out[1])
    assert np.all(np.isfinite(out[[0, 2, 3]]))
    assert abs(np.nanmean(out)) < 1e-12
    np.testing.assert_allclose(np.diff(out[2:]), 1.0)


def test_time_derivative_falls_back_to_one_side():
    assert fb.time_derivative((1.0, 2.0), None, (2.0, 5.0)) == 3.0
    assert fb.time_derivative((1.0, 2.0), (0.0, 1.0), None) == 1.0
    assert fb.time_derivative((1.0, 2.0), (0.0, 1.0), (2.0, 5.0)) == 2.0
    assert fb.time_derivative((1.0, 2.0)) is None


def test_smooth_preserves_constants():
    arr = np.ones((2, 3, 17)) * 4.0
    np.testing.assert_allclose(fb.smooth(arr, 5), arr)


def test_smooth_wider_than_the_array():
    arr = np.arange(4.0).reshape(2, 2)
    assert fb.smooth(arr, 5).shape == arr.shape
