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
