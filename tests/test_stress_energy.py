import numpy as np
import pytest

import stress_energy as se


def particles(n=20000, seed=0, drift=(0.0, 0.0, 0.0), spread=1.0):
    rng = np.random.default_rng(seed)
    u = rng.normal(drift[0], spread, n)
    v = rng.normal(drift[1], spread, n)
    w = rng.normal(drift[2], spread, n)
    return u, v, w


def reference_sums(x, u, v, w, lo, hi, nbins, weights=None):
    '''The lab-frame sums worked out the slow, obvious way.'''
    weights = np.ones_like(u) if weights is None else weights
    g = np.sqrt(1 + u**2 + v**2 + w**2)
    idx = np.floor((x - lo) / (hi - lo) * nbins).astype(int)
    keep = (idx >= 0) & (idx < nbins)
    idx, u, v, w, g, weights = idx[keep], u[keep], v[keep], w[keep], g[keep], weights[keep]
    per_particle = {se.N_: 1.0 + 0 * g, se.BX: u / g, se.BY: v / g, se.BZ: w / g,
                    se.G_: g, se.UX: u, se.UY: v, se.UZ: w,
                    se.TXX: u * u / g, se.TYY: v * v / g, se.TZZ: w * w / g,
                    se.TXY: u * v / g, se.TXZ: u * w / g, se.TYZ: v * w / g}
    out = np.zeros((se.N_LAB, nbins))
    for k, vals in per_particle.items():
        out[k] = np.bincount(idx, weights=weights * vals, minlength=nbins)
    return out


def test_lab_sums_match_reference():
    u, v, w = particles()
    x = np.random.default_rng(1).uniform(-5, 105, len(u))
    weights = np.random.default_rng(2).uniform(0.5, 2, len(u))
    sums = se.bin_moments(u, v, w, x, (0.0, 100.0), 50, weights=weights)
    assert sums.shape == (se.N_LAB, 50)
    np.testing.assert_allclose(sums, reference_sums(x, u, v, w, 0.0, 100.0, 50, weights), rtol=1e-10, atol=1e-9)


def test_float32_input_and_mask():
    u, v, w = (a.astype(np.float32) for a in particles())
    x = np.random.default_rng(1).uniform(0, 100, len(u)).astype(np.float32)
    mask = x < 50
    sums = se.bin_moments(u, v, w, x, (0.0, 100.0), 10, mask=mask)
    assert sums[se.N_, :5].sum() == mask.sum()
    assert np.all(sums[:, 5:] == 0)


def test_two_d_binning_matches_one_d_marginal():
    u, v, w = particles()
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 10, len(u)), rng.uniform(0, 4, len(u))
    sums2d = se.bin_moments(u, v, w, x, (0, 10), 20, vpos=y, v_range=(0, 4), nv=8)
    sums1d = se.bin_moments(u, v, w, x, (0, 10), 20)
    assert sums2d.shape == (se.N_LAB, 8, 20)
    np.testing.assert_allclose(sums2d.sum(axis=1), sums1d, rtol=1e-12)


def test_field_aligned_uniform_field_is_a_rotation_of_the_lab_tensor():
    '''With a uniform field, projecting each particle is the same as projecting
    the binned tensor, so the two routes must agree.'''
    u, v, w = particles(drift=(0.3, -0.2, 0.5))
    x = np.random.default_rng(4).uniform(0, 10, len(u))
    b = np.array([1.0, 2.0, -0.5])
    bhat = b / np.linalg.norm(b)
    grid = [np.full((1, 3, 6), c, dtype=np.float32) for c in b]
    sums = se.bin_moments(u, v, w, x, (0, 10), 4, bfield=grid, positions=(x, None, None), istep=2.0)
    assert sums.shape == (se.N_FIELD_ALIGNED, 4)
    num, mass = se.combine([sums], [1.0])

    T = np.array([[mass[se.TXX], mass[se.TXY], mass[se.TXZ]],
                  [mass[se.TXY], mass[se.TYY], mass[se.TYZ]],
                  [mass[se.TXZ], mass[se.TYZ], mass[se.TZZ]]])
    T0 = np.array([mass[se.UX], mass[se.UY], mass[se.UZ]])
    t_parpar = np.einsum('i,ijk,j->k', bhat, T, bhat)
    t_0par = np.einsum('i,ik->k', bhat, T0)
    trace = T[0, 0] + T[1, 1] + T[2, 2]
    kw = dict(normalization='density', dens_factor=1.0)
    np.testing.assert_allclose(se.evaluate(num, mass, 'T', 'parpar', **kw), t_parpar, rtol=1e-6)
    np.testing.assert_allclose(se.evaluate(num, mass, 'T', '0par', **kw), t_0par, rtol=1e-6)
    np.testing.assert_allclose(se.evaluate(num, mass, 'T', 'perpperp', **kw), (trace - t_parpar) / 2, rtol=1e-6)
    # The trace does not depend on the basis
    np.testing.assert_allclose(3 * se.evaluate(num, mass, 'T', 'trace', **kw),
                               se.evaluate(num, mass, 'T', 'parpar', **kw)
                               + 2 * se.evaluate(num, mass, 'T', 'perpperp', **kw), rtol=1e-6)

    beta = np.array([se.evaluate(num, mass, 'beta', c) for c in 'xyz'])
    beta_par = se.evaluate(num, mass, 'beta', 'par')
    np.testing.assert_allclose(beta_par, np.einsum('i,ik->k', bhat, beta), rtol=1e-6)
    perp = beta - beta_par * bhat[:, None]
    np.testing.assert_allclose(se.evaluate(num, mass, 'beta', 'perp'), np.linalg.norm(perp, axis=0), rtol=1e-6)


def test_field_is_interpolated_to_each_particle():
    '''A field along x in the left half of the grid and along y in the right:
    particles moving along x are all parallel on the left, all perpendicular
    on the right.'''
    n = 1000
    u, v, w = np.full(n, 2.0), np.zeros(n), np.zeros(n)
    x = np.concatenate([np.full(n // 2, 2.0), np.full(n // 2, 18.0)])  # cells
    bx = np.zeros((1, 1, 11), dtype=np.float32)
    by = np.zeros_like(bx)
    bx[..., :5] = 1.0
    by[..., 6:] = 1.0
    sums = se.bin_moments(u, v, w, x, (0, 20), 2, bfield=(bx, by, np.zeros_like(bx)),
                          positions=(x, None, None), istep=2.0)
    num, mass = se.combine([sums], [1.0])
    np.testing.assert_allclose(se.evaluate(num, mass, 'u', 'par'), [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(se.evaluate(num, mass, 'u', 'perp'), [0.0, 2.0], atol=1e-12)


def boosted(u, v, w, beta):
    '''Boost 4-velocities along x, returning them with the weights that make
    a fixed-volume sample of the moving plasma.'''
    g_rest = np.sqrt(1 + u**2 + v**2 + w**2)
    gb = 1 / np.sqrt(1 - beta**2)
    u_lab = gb * (u + beta * g_rest)
    g_lab = np.sqrt(1 + u_lab**2 + v**2 + w**2)
    return u_lab, v, w, g_lab / g_rest


def test_rest_frame_quantities_are_frame_independent():
    u, v, w = particles(n=50000, spread=0.7)
    x = np.full(len(u), 0.5)
    rest = se.bin_moments(u, v, w, x, (0, 1), 1)
    ub, vb, wb, weights = boosted(u, v, w, 0.6)
    lab = se.bin_moments(ub, vb, wb, x, (0, 1), 1, weights=weights)
    for sums in (rest, lab):
        num, mass = se.combine([sums], [1.0])
        sums_values = {c: se.evaluate(num, mass, 'energy', c) for c in ('thermal', 'gamma_bulk')}
        sums_values['e'] = se.evaluate(num, mass, 'T', 'e_rest', normalization='particle')
        sums_values['p'] = se.evaluate(num, mass, 'T', 'p_rest', normalization='particle')
        if sums is rest:
            expected = sums_values
        else:
            for key, val in sums_values.items():
                if key == 'gamma_bulk':
                    continue
                np.testing.assert_allclose(val, expected[key], rtol=1e-9, err_msg=key)
            np.testing.assert_allclose(sums_values['gamma_bulk'], 1 / np.sqrt(1 - 0.6**2), rtol=2e-2)


def test_cold_beam_has_no_thermal_energy():
    n = 100
    u, v, w = np.full(n, 3.0), np.full(n, -1.0), np.zeros(n)
    sums = se.bin_moments(u, v, w, np.zeros(n), (-1, 1), 1)
    num, mass = se.combine([sums], [1.0])
    gamma = np.sqrt(1 + 9 + 1)
    np.testing.assert_allclose(se.evaluate(num, mass, 'energy', 'thermal'), 0.0, atol=1e-12)
    np.testing.assert_allclose(se.evaluate(num, mass, 'energy', 'gamma_bulk'), gamma)
    np.testing.assert_allclose(se.evaluate(num, mass, 'energy', 'ke'), gamma - 1)
    np.testing.assert_allclose(se.evaluate(num, mass, 'T', 'p_rest', normalization='particle'), 0.0, atol=1e-12)
    np.testing.assert_allclose(se.evaluate(num, mass, 'T', 'e_rest', normalization='particle'), 1.0)


def test_species_are_combined_by_mass():
    u, v, w = particles(n=1000)
    x = np.zeros(len(u))
    s = se.bin_moments(u, v, w, x, (-1, 1), 1)
    num, mass = se.combine([s, s], [1.0, 0.25])
    one_num, one_mass = se.combine([s], [1.0])
    # A density adds up
    np.testing.assert_allclose(se.evaluate(num, mass, 'T', '00'), 1.25 * se.evaluate(one_num, one_mass, 'T', '00'))
    # An average does not care that the second species is lighter
    np.testing.assert_allclose(se.evaluate(num, mass, 'beta', 'x'), se.evaluate(one_num, one_mass, 'beta', 'x'))


def test_empty_bins():
    sums = se.bin_moments(np.ones(3), np.zeros(3), np.zeros(3), np.array([0.5, 0.5, 0.5]), (0, 2), 2)
    num, mass = se.combine([sums], [1.0])
    assert np.isnan(se.evaluate(num, mass, 'beta', 'x')[1])
    assert se.evaluate(num, mass, 'T', '00')[1] == 0.0
    assert np.isnan(se.evaluate(num, mass, 'T', '00', normalization='particle')[1])


@pytest.mark.parametrize('family', se.FAMILIES)
@pytest.mark.parametrize('basis', se.BASES)
def test_every_component_evaluates_and_has_labels(family, basis):
    u, v, w = particles(n=500)
    x = np.random.default_rng(5).uniform(0, 1, len(u))
    grid = [np.ones((1, 1, 1), dtype=np.float32)] * 3
    sums = se.bin_moments(u, v, w, x, (0, 1), 3, bfield=grid, positions=(x, None, None))
    num, mass = se.combine([sums], [1.0])
    for comp in se.components(family, basis):
        vals = se.evaluate(num, mass, family, comp)
        assert vals.shape == (3,)
        assert np.all(np.isfinite(vals))
        assert se.ui_label(family, comp)
        assert se.tex_label(family, comp)


def boost_particles(u, v, w, beta):
    '''Boost the 4-velocities of a sample at rest to a lab frame in which it
    moves with 3-velocity `beta`, with the weights of a fixed-volume sample.'''
    beta = np.asarray(beta, dtype=float)
    gb = 1 / np.sqrt(1 - beta @ beta)
    ub = gb * beta
    U = np.stack([u, v, w], axis=1)
    g = np.sqrt(1 + np.sum(U**2, axis=1))
    lab = U @ (np.eye(3) + np.outer(ub, ub) / (gb + 1)).T + np.outer(g, ub)
    g_lab = np.sqrt(1 + np.sum(lab**2, axis=1))
    return lab[:, 0], lab[:, 1], lab[:, 2], g_lab / g


def mirrored_anisotropic(n=20000, seed=0):
    '''A sample with P_zz > P_xx = P_yy and, because every particle has its
    mirror image, no particle or energy flux: it is exactly at rest.'''
    rng = np.random.default_rng(seed)
    u, v, w = rng.normal(0, 0.5, n), rng.normal(0, 0.5, n), rng.normal(0, 1.2, n)
    return [np.concatenate([a, -a]) for a in (u, v, w)]


def rest_pressure(u, v, w):
    g = np.sqrt(1 + u**2 + v**2 + w**2)
    return np.array([[np.sum(a * b / g) for b in (u, v, w)] for a in (u, v, w)])


def uniform_grid(vec):
    return [np.full((1, 1, 2), c, dtype=np.float32) for c in vec]


@pytest.mark.parametrize('frame', se.FRAMES)
def test_rest_frame_pressure_survives_a_boost(frame):
    '''Boost a plasma at rest, with its magnetic field B' and no electric
    field, into a lab where it drifts obliquely to B: the rest-frame pressure
    tensor, its field-aligned components and the temperature must come back.'''
    u, v, w = mirrored_anisotropic()
    P_rest = rest_pressure(u, v, w)
    B_rest = np.array([0.3, 0.0, 1.0])
    bhat = B_rest / np.linalg.norm(B_rest)
    beta = np.array([0.6, 0.3, 0.0])
    gb = 1 / np.sqrt(1 - beta @ beta)
    # The lab fields of a plasma with E' = 0 in its rest frame
    E = -gb * np.cross(beta, B_rest)
    B = gb * B_rest - gb**2 / (gb + 1) * beta * (beta @ B_rest)

    ub, vb, wb, weights = boost_particles(u, v, w, beta)
    x = np.zeros(len(ub))
    sums = se.bin_moments(ub, vb, wb, x, (-1, 1), 1, weights=weights,
                          bfield=uniform_grid(B), efield=uniform_grid(E), positions=(x, None, None))
    num, mass = se.combine([sums], [1.0])

    P, n, U, L = se.rest_frame_pressure(num, mass, frame)
    np.testing.assert_allclose(P[0], P_rest, rtol=1e-10, atol=1e-8 * np.abs(P_rest).max())
    np.testing.assert_allclose(n, len(u), rtol=1e-12)
    np.testing.assert_allclose(U[0, 1:] / U[0, 0], beta, atol=1e-12)

    kw = dict(rest_frame=frame, dens_factor=1.0)
    p_par = bhat @ P_rest @ bhat
    p_perp = (np.trace(P_rest) - p_par) / 2
    np.testing.assert_allclose(se.evaluate(num, mass, 'P', 'parpar', **kw), p_par, rtol=1e-6)
    np.testing.assert_allclose(se.evaluate(num, mass, 'P', 'perpperp', **kw), p_perp, rtol=1e-6)
    off = P_rest @ bhat - p_par * bhat
    np.testing.assert_allclose(se.evaluate(num, mass, 'P', 'parperp', **kw), np.linalg.norm(off),
                               rtol=1e-5, atol=1e-6 * p_par)
    np.testing.assert_allclose(se.evaluate(num, mass, 'P', 'xz', **kw), P_rest[0, 2],
                               atol=1e-8 * np.abs(P_rest).max())
    np.testing.assert_allclose(se.evaluate(num, mass, 'Theta', 'scalar', **kw),
                               np.trace(P_rest) / 3 / len(u), rtol=1e-10)
    np.testing.assert_allclose(se.evaluate(num, mass, 'Theta', 'parpar', **kw), p_par / len(u), rtol=1e-6)
    for family in se.REST_FRAME_FAMILIES:
        np.testing.assert_allclose(se.evaluate(num, mass, family, 'par_over_perp', **kw),
                                   p_par / p_perp, rtol=1e-6)
        np.testing.assert_allclose(se.evaluate(num, mass, family, 'perp_over_par', **kw),
                                   p_perp / p_par, rtol=1e-6)


def test_without_an_electric_field_the_lab_field_is_boosted():
    '''A plasma drifting along B sees the same B, whatever E is taken to be.'''
    u, v, w = mirrored_anisotropic(n=5000)
    P_rest = rest_pressure(u, v, w)
    ub, vb, wb, weights = boost_particles(u, v, w, (0.0, 0.0, 0.8))
    x = np.zeros(len(ub))
    sums = se.bin_moments(ub, vb, wb, x, (-1, 1), 1, weights=weights,
                          bfield=uniform_grid((0.0, 0.0, 2.0)), positions=(x, None, None))
    num, mass = se.combine([sums], [1.0])
    for frame in se.FRAMES:
        np.testing.assert_allclose(se.evaluate(num, mass, 'P', 'parpar', rest_frame=frame),
                                   P_rest[2, 2], rtol=1e-9)
        np.testing.assert_allclose(se.evaluate(num, mass, 'P', 'perpperp', rest_frame=frame),
                                   (P_rest[0, 0] + P_rest[1, 1]) / 2, rtol=1e-9)


def test_eckart_and_landau_frames_differ_with_heat_flux():
    '''A dense, light, cold beam through a sparse, heavy one carries a heat
    flux: the frame with no particle flux is not the one with no energy flux.'''
    n1, n2 = 3000, 1000
    light = (np.full(n1, 0.5), np.zeros(n1), np.zeros(n1))
    heavy = (np.full(n2, -0.2), np.zeros(n2), np.zeros(n2))
    x1, x2 = np.zeros(n1), np.zeros(n2)
    sums = [se.bin_moments(*light, x1, (-1, 1), 1), se.bin_moments(*heavy, x2, (-1, 1), 1)]
    num, mass = se.combine(sums, [0.1, 1.0])

    U_e, n_e = se.frame_velocity(num, mass, 'eckart')
    U_l, n_l = se.frame_velocity(num, mass, 'landau')
    assert U_e[0, 1] > 0 > U_l[0, 1]
    np.testing.assert_allclose(U_e[0, 0]**2 - np.sum(U_e[0, 1:]**2), 1.0)
    np.testing.assert_allclose(U_l[0, 0]**2 - np.sum(U_l[0, 1:]**2), 1.0)

    N = se._number_flux(num)
    T = se._lab_tensor(mass)
    L_e, L_l = se.boost_to_rest(U_e), se.boost_to_rest(U_l)
    # No particle flux in the Eckart frame, no energy flux in the Landau one
    np.testing.assert_allclose(np.einsum('...ma,...a->...m', L_e, N)[0, 1:], 0.0, atol=1e-9 * n1)
    T_l = np.einsum('...ma,...ab,...nb->...mn', L_l, T, L_l)
    np.testing.assert_allclose(T_l[0, 0, 1:], 0.0, atol=1e-9 * T_l[0, 0, 0])
    # Each frame counts the particles it sees
    np.testing.assert_allclose(n_l, N[..., 0] * U_l[..., 0] - np.sum(N[..., 1:] * U_l[..., 1:], axis=-1))
    np.testing.assert_allclose(n_e, np.sqrt(N[0, 0]**2 - np.sum(N[0, 1:]**2)))
    # The heat flux shows up as different pressures
    p_e = se.evaluate(num, mass, 'P', 'xx', rest_frame='eckart')
    p_l = se.evaluate(num, mass, 'P', 'xx', rest_frame='landau')
    assert np.all(p_e > 0) and np.all(p_l > 0)
    assert not np.allclose(p_e, p_l, rtol=1e-3)


def test_proper_density_is_the_count_in_the_chosen_frame():
    '''n' = -N.U per volume: sqrt(-N.N) in the Eckart frame, less than that in the Landau one.'''
    n1, n2 = 3000, 1000
    light = (np.full(n1, 0.5), np.zeros(n1), np.zeros(n1))
    heavy = (np.full(n2, -0.2), np.zeros(n2), np.zeros(n2))
    sums = [se.bin_moments(*light, np.zeros(n1), (-1, 1), 2),
            se.bin_moments(*heavy, np.zeros(n2), (-1, 1), 2)]
    num, mass = se.combine(sums, [0.1, 1.0])
    N = se._number_flux(num)
    n_eckart = np.sqrt(N[..., 0]**2 - np.sum(N[..., 1:]**2, axis=-1))
    for family in se.REST_FRAME_FAMILIES:
        assert 'n_rest' in se.components(family, 'lab')
        assert 'n_rest' in se.components(family, 'fa')
        assert not se.needs_field(family, 'n_rest')
        n_e = se.evaluate(num, mass, family, 'n_rest', dens_factor=2.0, rest_frame='eckart')
        n_l = se.evaluate(num, mass, family, 'n_rest', dens_factor=2.0, rest_frame='landau')
        # The bin holding the particles, and the empty one
        np.testing.assert_allclose(n_e[1], 2.0 * n_eckart[1])
        assert n_e[0] == 0.0 and n_l[0] == 0.0
        # -N.U >= sqrt(-N.N) for any unit timelike U, so the Eckart frame,
        # moving with N, counts the fewest particles
        assert 0 < n_e[1] < n_l[1]
    # One cold particle species: every frame is its rest frame, n' = N / gamma
    u = np.full(100, 0.75)
    sums = se.bin_moments(u, np.zeros(100), np.zeros(100), np.zeros(100), (-1, 1), 1)
    num, mass = se.combine([sums], [1.0])
    for frame in se.FRAMES:
        np.testing.assert_allclose(se.evaluate(num, mass, 'P', 'n_rest', rest_frame=frame),
                                   100 / np.sqrt(1 + 0.75**2), rtol=1e-9)


def test_rest_frame_pressure_matches_the_stress_energy_scalar():
    '''The Eckart scalar pressure is the P' the stress-energy tensor already showed.'''
    u, v, w = particles(n=5000, drift=(0.4, -0.3, 0.2), spread=0.8)
    x = np.random.default_rng(6).uniform(0, 1, len(u))
    num, mass = se.combine([se.bin_moments(u, v, w, x, (0, 1), 4)], [1.0])
    np.testing.assert_allclose(se.evaluate(num, mass, 'P', 'scalar', rest_frame='eckart'),
                               se.evaluate(num, mass, 'T', 'p_rest'), rtol=1e-10)


def test_cold_beam_has_no_rest_frame_pressure_in_either_frame():
    n = 100
    u, v, w = np.full(n, 3.0), np.full(n, -1.0), np.zeros(n)
    num, mass = se.combine([se.bin_moments(u, v, w, np.zeros(n), (-1, 1), 1)], [1.0])
    for frame in se.FRAMES:
        for comp in se.components('P', 'lab'):
            if comp == 'n_rest':
                # not a pressure: the beam's own density, n / gamma
                np.testing.assert_allclose(se.evaluate(num, mass, 'P', comp, rest_frame=frame), n / np.sqrt(11))
                continue
            np.testing.assert_allclose(se.evaluate(num, mass, 'P', comp, rest_frame=frame), 0.0, atol=1e-9)
            np.testing.assert_allclose(se.evaluate(num, mass, 'Theta', comp, rest_frame=frame), 0.0, atol=1e-9)


@pytest.mark.parametrize('frame', se.FRAMES)
def test_rest_frame_empty_bins(frame):
    sums = se.bin_moments(np.ones(3), np.zeros(3), np.zeros(3), np.array([0.5, 0.5, 0.5]), (0, 2), 2)
    num, mass = se.combine([sums], [1.0])
    assert se.evaluate(num, mass, 'P', 'xx', rest_frame=frame)[1] == 0.0
    assert np.isnan(se.evaluate(num, mass, 'Theta', 'xx', rest_frame=frame)[1])
    assert np.isfinite(se.evaluate(num, mass, 'Theta', 'xx', rest_frame=frame)[0])


def test_anisotropy_is_only_offered_field_aligned():
    for family in se.REST_FRAME_FAMILIES:
        assert set(se.ANISOTROPY) <= set(se.components(family, 'fa'))
        assert not set(se.ANISOTROPY) & set(se.components(family, 'lab'))
        for comp in se.ANISOTROPY:
            assert se.needs_field(family, comp)
            se.ui_label(family, comp), se.tex_label(family, comp)


def test_frame_velocity_components_are_each_bins_own_frame():
    '''Two cells drifting differently each get their own U^mu, in either frame.'''
    u = np.r_[np.full(100, 3.0), np.full(100, -1.0)]
    v = np.r_[np.full(100, 0.5), np.zeros(100)]
    x = np.r_[np.full(100, -0.5), np.full(100, 0.5)]
    z = np.zeros(200)
    num, mass = se.combine([se.bin_moments(u, v, z, x, (-1, 1), 2)], [1.0])
    expected = np.array([[np.sqrt(1 + 9 + 0.25), 3.0, 0.5, 0.0],
                         [np.sqrt(2), -1.0, 0.0, 0.0]])
    for frame in se.FRAMES:
        for i, comp in enumerate(se.components('U', 'lab')):
            np.testing.assert_allclose(se.evaluate(num, mass, 'U', comp, rest_frame=frame),
                                       expected[:, i], atol=1e-9)


def test_frame_velocity_differs_between_eckart_and_landau_with_heat_flux():
    n1, n2 = 3000, 1000
    light = (np.full(n1, 0.5), np.zeros(n1), np.zeros(n1))
    heavy = (np.full(n2, -0.2), np.zeros(n2), np.zeros(n2))
    sums = [se.bin_moments(*light, np.zeros(n1), (-1, 1), 1),
            se.bin_moments(*heavy, np.zeros(n2), (-1, 1), 1)]
    num, mass = se.combine(sums, [0.1, 1.0])
    U_e = se.evaluate(num, mass, 'U', 'x', rest_frame='eckart')
    U_l = se.evaluate(num, mass, 'U', 'x', rest_frame='landau')
    assert U_e[0] > 0 > U_l[0]
    np.testing.assert_allclose(U_l, se.frame_velocity(num, mass, 'landau')[0][..., 1])
    # Empty bins have no frame
    empty = se.bin_moments(np.ones(3), np.zeros(3), np.zeros(3), np.full(3, 0.5), (0, 2), 2)
    num, mass = se.combine([empty], [1.0])
    for comp in se.components('U', 'lab'):
        assert np.isnan(se.evaluate(num, mass, 'U', comp)[1])


def test_frame_velocity_along_and_across_the_mean_field():
    '''A beam drifting at 45 degrees to B splits evenly into U_par and |U_perp|.'''
    n = 500
    u, v, w = np.full(n, 2.0), np.zeros(n), np.full(n, 2.0)
    x = np.zeros(n)
    sums = se.bin_moments(u, v, w, x, (-1, 1), 1,
                          bfield=uniform_grid((0.0, 0.0, 3.0)), positions=(x, None, None))
    num, mass = se.combine([sums], [1.0])
    assert se.components('U', 'fa') == ['t', 'par', 'perp']
    assert se.needs_field('U', 'par') and not se.needs_field('U', 'x')
    for frame in se.FRAMES:
        np.testing.assert_allclose(se.evaluate(num, mass, 'U', 'par', rest_frame=frame), 2.0)
        np.testing.assert_allclose(se.evaluate(num, mass, 'U', 'perp', rest_frame=frame), 2.0)
        np.testing.assert_allclose(se.evaluate(num, mass, 'U', 't', rest_frame=frame), 3.0)
    for comp in se.components('U', 'lab') + se.components('U', 'fa'):
        se.ui_label('U', comp), se.tex_label('U', comp)
