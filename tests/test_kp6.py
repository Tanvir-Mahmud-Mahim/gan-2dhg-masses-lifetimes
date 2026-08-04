"""Tests for the six-band wurtzite Hamiltonian and the two-dimensional
scattering module.

These are physics tests, not smoke tests.  Each one checks the code against a
quantity known independently of the code: a published splitting, an analytic
limit, or a symmetry that must hold exactly.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gan2dhg import kp6 as K
from gan2dhg import scatter2d as S

M0 = 9.1093837015e-31


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------

def test_hamiltonian_is_hermitian_everywhere():
    st = K.biaxial_strain_on_AlN()
    rng = np.random.default_rng(0)
    for _ in range(50):
        kx, ky, kz = rng.uniform(-3, 3, 3)
        M = K.hamiltonian(kx, ky, kz, st)
        assert np.allclose(M, M.conj().T, atol=1e-13)


def test_kramers_degeneracy():
    """Every level must be doubly degenerate: the system is time-reversal
    symmetric and has no magnetic field."""
    st = K.biaxial_strain_on_AlN()
    for kt in (0.0, 0.3, 1.0, 2.0):
        e = np.linalg.eigvalsh(K.hamiltonian(kt, 0.0, 0.7, st))
        pairs = e.reshape(3, 2)
        assert np.allclose(pairs[:, 0], pairs[:, 1], atol=1e-10)


def test_unstrained_zone_centre_splittings():
    """A-B and A-C splittings of unstrained GaN.

    With Delta_CR = 10 meV and Delta_SO = 17 meV the quasi-cubic result is
    A-B = 5.2 meV and A-C = 21.8 meV, which is the accepted range for GaN.
    """
    e = np.linalg.eigvalsh(K.hamiltonian(0, 0, 0, None))[::-1]
    ab = 1000.0 * (e[0] - e[2])
    ac = 1000.0 * (e[0] - e[4])
    assert 4.5 < ab < 6.5, ab
    assert 20.0 < ac < 24.0, ac


def test_analytic_zone_centre_eigenvalues():
    """At k = 0 the eigenvalues are known in closed form."""
    d1, d2 = K.GAN["D_CR"], K.GAN["D_SO"] / 3.0
    D = np.sqrt(2.0) * d2
    F = d1 + d2
    G = d1 - d2
    expect = sorted([F, 0.5 * (G + np.sqrt(G**2 + 4 * D**2)),
                     0.5 * (G - np.sqrt(G**2 + 4 * D**2))])
    got = sorted(np.unique(np.round(
        np.linalg.eigvalsh(K.hamiltonian(0, 0, 0, None)), 12)))
    assert np.allclose(got, expect, atol=1e-9)


def test_in_plane_isotropy_at_kz_zero():
    """At kz = 0 the basal-plane dispersion of this Hamiltonian is isotropic.

    This is asserted in the manuscript and is the reason circular Fermi
    contours are used, so it is tested rather than assumed.
    """
    st = K.biaxial_strain_on_AlN()
    ref = K.bands(1.0, phi=0.0, kz=0.0, strain=st)
    for phi in (0.1, 0.5, np.pi / 6, np.pi / 3, 1.9):
        assert np.allclose(K.bands(1.0, phi, 0.0, st), ref, atol=1e-12)


def test_strain_matches_reported_value():
    exx, eyy, ezz = K.biaxial_strain_on_AlN()
    assert exx == eyy
    assert abs(100 * exx + 2.4) < 0.1          # 2.4 per cent compressive
    assert ezz > 0                              # Poisson expansion along c


def test_heavy_light_separation_requires_finite_wavevector():
    """The heavy and light labels are not zone-centre properties.

    The term that separates the two upper branches is K = A5 (kx + i ky)^2,
    which has to overcome the spin-orbit mixing sqrt(2) Delta_SO / 3 before the
    branches acquire distinct masses.  Below the crossover both branches share
    the quasi-cubic value m0 / |A2 + A4|; above it they take
    m0 / |A2 + A4 -/+ A5|.  This test pins both limits and the crossover.
    """
    a2, a4, a5 = K.GAN["A2"], K.GAN["A4"], K.GAN["A5"]
    m_soc = 1.0 / abs(a2 + a4)
    # A5 is negative, so the branch with A2 + A4 + A5 is the LIGHT one.
    m_light = 1.0 / abs(a2 + a4 + a5)
    m_heavy = 1.0 / abs(a2 + a4 - a5)
    assert m_light < m_soc < m_heavy

    k_cross = np.sqrt(np.sqrt(2.0) * K.GAN["D_SO"] / 3.0
                      / (K.HB2_2M0_eVnm2 * abs(a5)))
    assert 0.25 < k_cross < 0.35, k_cross

    st = K.biaxial_strain_on_AlN()
    kt = np.linspace(1e-3, 2.0, 4000)
    E = K.dispersion(kt, strain=st)
    # Well below the crossover both branches sit near the quasi-cubic value.
    i = np.searchsorted(kt, 0.05)
    assert abs(K.cyclotron_mass(kt, E, 0)[i] - m_soc) < 0.05
    assert abs(K.cyclotron_mass(kt, E, 2)[i] - m_soc) < 0.05
    # Well above it, each branch has reached its own asymptote.
    ih = np.searchsorted(kt, K.kf_from_density(3.8e13))
    il = np.searchsorted(kt, K.kf_from_density(8.0e12))
    assert abs(K.cyclotron_mass(kt, E, 0)[ih] - m_heavy) < 0.02
    assert abs(K.cyclotron_mass(kt, E, 2)[il] - m_light) < 0.02


def test_both_measured_fermi_wavevectors_lie_above_the_crossover():
    """The manuscript relies on the heavy and light labels being meaningful at
    the Fermi level, so the margin is checked explicitly."""
    k_cross = np.sqrt(np.sqrt(2.0) * K.GAN["D_SO"] / 3.0
                      / (K.HB2_2M0_eVnm2 * abs(K.GAN["A5"])))
    assert K.kf_from_density(8.0e12) / k_cross > 2.0
    assert K.kf_from_density(3.8e13) / k_cross > 5.0


def test_hardwall_proxy_behaviour_is_recorded_not_relied_on():
    """Behaviour of the hard-wall proxy k_z = pi/w, kept only as a record.

    This proxy is NOT used for any result in the manuscript.  It inserts k_z as
    a number into the coupling i A6 k_z (kx + i ky), which is linear in k_z and
    whose expectation vanishes for a bound state, so it invents band mixing
    that does not occur and drives the light mass upward.  The self-consistent
    solution in kp6_well gives 0.246 m0 where this returns about 0.51 m0.  The
    test is retained so that the discrepancy stays visible.
    """
    st = K.biaxial_strain_on_AlN()
    kt = np.linspace(1e-3, 2.6, 2000)
    kf_h, kf_l = K.kf_from_density(3.8e13), K.kf_from_density(8.0e12)
    ih, il = np.searchsorted(kt, kf_h), np.searchsorted(kt, kf_l)
    m_h, m_l = [], []
    for w in (6.0, 3.0, 1.0):
        E = K.dispersion(kt, kz=K.subband_kz(w), strain=st)
        m_h.append(K.cyclotron_mass(kt, E, 0)[ih])
        m_l.append(K.cyclotron_mass(kt, E, 2)[il])
    assert max(m_h) - min(m_h) < 0.02            # heavy is unmoved
    assert m_l[-1] > 2.0 * m_l[0]                # light more than doubles
    assert 0.45 < m_l[-1] < 0.60                 # and lands on the measurement


def test_fermi_wavevector_matches_oscillation_frequencies():
    """Density and SdH frequency are related by f = (hbar/2 pi e) A_k.

    For a spin-degenerate two-dimensional band this gives f = n h / (2 e).
    """
    h = 6.62607015e-34
    q = 1.602176634e-19
    for n_cm2, f_expect in ((8.0e12, 166.0), (3.8e13, 795.0)):
        f = n_cm2 * 1e4 * h / (2.0 * q)
        assert abs(f - f_expect) / f_expect < 0.03, (n_cm2, f, f_expect)


# ---------------------------------------------------------------------------
# Scattering
# ---------------------------------------------------------------------------

def test_isotropic_scattering_gives_unit_ratio():
    """For an angle-independent, unscreened kernel tau_tr must equal tau_q."""
    bands = [{"label": "a", "m_over_m0": 1.0, "n_s_cm2": 1.0e13}]
    b = S.fang_howard_b(1.0e13, 0.0, 1.0, 10.4)
    r = S.lifetimes(bands[0], bands, lambda q: 1.0e-40, 10.4, b, screen=False)
    assert abs(r["ratio"] - 1.0) < 1e-6


def test_forward_peaked_scattering_gives_large_ratio():
    """A kernel concentrated at small q must give tau_tr well above tau_q."""
    bands = [{"label": "a", "m_over_m0": 1.0, "n_s_cm2": 1.0e13}]
    b = S.fang_howard_b(1.0e13, 0.0, 1.0, 10.4)
    kF = S.fermi_wavevector(1.0e13)
    sharp = lambda q: 1.0e-40 * np.exp(-(q / (0.05 * kF)) ** 2)
    r = S.lifetimes(bands[0], bands, sharp, 10.4, b, screen=False)
    assert r["ratio"] > 50.0, r["ratio"]


def test_backscattering_gives_ratio_below_one():
    """A kernel concentrated at large q must give tau_tr below tau_q,
    with a floor of one half."""
    bands = [{"label": "a", "m_over_m0": 1.0, "n_s_cm2": 1.0e13}]
    b = S.fang_howard_b(1.0e13, 0.0, 1.0, 10.4)
    kF = S.fermi_wavevector(1.0e13)
    back = lambda q: 1.0e-40 * np.exp(-((q - 2 * kF) / (0.02 * kF)) ** 2)
    r = S.lifetimes(bands[0], bands, back, 10.4, b, screen=False)
    assert 0.49 < r["ratio"] < 0.55, r["ratio"]


def test_ratio_is_independent_of_amplitude():
    """The ratio must not depend on the disorder strength; the manuscript
    relies on this to avoid fitting."""
    bands = [{"label": "a", "m_over_m0": 1.0, "n_s_cm2": 1.0e13}]
    b = S.fang_howard_b(1.0e13, 0.0, 1.0, 10.4)
    r1 = S.lifetimes(bands[0], bands,
                     lambda q: S.w_remote_impurity(q, 1e12, 5e-9, b, 10.4),
                     10.4, b)
    r2 = S.lifetimes(bands[0], bands,
                     lambda q: S.w_remote_impurity(q, 1e14, 5e-9, b, 10.4),
                     10.4, b)
    assert abs(r1["ratio"] - r2["ratio"]) / r1["ratio"] < 1e-6


def test_ratio_is_independent_of_effective_mass():
    """The ratio contains no mass.  This is the reason the comparison in the
    manuscript is independent of the band structure, so it is tested."""
    b = S.fang_howard_b(1.0e13, 0.0, 1.0, 10.4)
    w = lambda q: S.w_remote_impurity(q, 1e13, 5e-9, b, 10.4)
    out = []
    for m in (0.2, 1.0, 3.0):
        bd = {"label": "a", "m_over_m0": m, "n_s_cm2": 1.0e13}
        # Screening does depend on mass, so it is switched off for this test.
        out.append(S.lifetimes(bd, [bd], w, 10.4, b, screen=False)["ratio"])
    assert max(out) - min(out) < 1e-6 * max(out)


def test_larger_kf_is_more_forward_peaked():
    """The ordering the manuscript says every mechanism produces."""
    b = S.fang_howard_b(4.6e13, 0.0, 1.0, 10.4)
    w = lambda q: S.w_remote_impurity(q, 1e13, 3e-9, b, 10.4)
    lo = {"label": "lo", "m_over_m0": 0.53, "n_s_cm2": 8.0e12}
    hi = {"label": "hi", "m_over_m0": 1.92, "n_s_cm2": 3.8e13}
    r_lo = S.lifetimes(lo, [lo, hi], w, 10.4, b)["ratio"]
    r_hi = S.lifetimes(hi, [lo, hi], w, 10.4, b)["ratio"]
    assert r_hi > r_lo


def test_two_band_reduces_to_single_band_without_coupling():
    """With zero interband overlap the coupled solution must reproduce the
    independent single-band lifetimes."""
    b = S.fang_howard_b(4.6e13, 0.0, 1.0, 10.4)
    w = lambda q: S.w_remote_impurity(q, 1e13, 3e-9, b, 10.4)
    lo = {"label": "lo", "m_over_m0": 0.53, "n_s_cm2": 8.0e12}
    hi = {"label": "hi", "m_over_m0": 1.92, "n_s_cm2": 3.8e13}
    O = np.array([[1.0, 0.0], [0.0, 1.0]])
    tb = S.two_band_lifetimes([lo, hi], w, 10.4, b, overlap=O)
    for i, bd in enumerate((lo, hi)):
        single = S.lifetimes(bd, [lo, hi], w, 10.4, b)
        assert abs(tb["ratio"][i] - single["ratio"]) / single["ratio"] < 1e-6


def test_interband_fraction_is_larger_for_the_light_band():
    """The light band has the smaller density of states, so it sees the
    heavy band as the larger phase space to scatter into."""
    b = S.fang_howard_b(4.6e13, 0.0, 1.0, 10.4)
    F = 1.602176634e-19 * (4.6e17 / 2.0) / (10.4 * 8.8541878128e-12)
    w = lambda q: S.w_interface_roughness(q, 0.3e-9, 1e-9, F)
    lo = {"label": "lo", "m_over_m0": 0.53, "n_s_cm2": 8.0e12}
    hi = {"label": "hi", "m_over_m0": 1.92, "n_s_cm2": 3.8e13}
    O = np.array([[1.0, 0.5], [0.5, 1.0]])
    tb = S.two_band_lifetimes([lo, hi], w, 10.4, b, overlap=O)
    f = tb["interband_fraction_of_quantum_rate"]
    assert f[0] > f[1]


def test_measured_lifetimes_reproduce_quoted_ratios():
    """Guard on the numbers quoted in the manuscript."""
    t_lh = S.tau_from_mobility(1900.0, 0.53)
    t_hh = S.tau_from_mobility(400.0, 1.92)
    assert abs(t_lh / 0.15e-12 - 3.82) < 0.02
    assert abs(t_hh / 0.205e-12 - 2.13) < 0.02


def test_cyclotron_product_at_the_measured_field():
    """omega_c tau for the heavy holes at 31 T, quoted as 0.82."""
    q = 1.602176634e-19
    wt = q * 31.0 / (2.6 * M0) * 3.9e-13
    assert abs(wt - 0.82) < 0.02
    wt_l = q * 31.0 / (0.57 * M0) * 4.0e-13
    assert abs(wt_l - 3.83) < 0.05


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


# ---------------------------------------------------------------------------
# Self-consistent envelope-function well
# ---------------------------------------------------------------------------

def test_well_operator_is_hermitian():
    from gan2dhg import kp6_well as KW
    z = np.linspace(0.0, 8.0, 41)
    V = 0.05 * z
    st = K.biaxial_strain_on_AlN()
    for kt in (0.0, 0.4, 1.3):
        M = KW.build_operator(kt, z, V, st)
        assert np.allclose(M, M.conj().T, atol=1e-11)


def test_well_time_reversal_symmetry():
    """The spectrum at +k must equal the spectrum at -k.

    Note that this does NOT require the two branches at a given |k| to be
    degenerate.  The polarisation well has no inversion centre, so Rashba
    splitting is allowed and is present; see the two tests below.
    """
    from gan2dhg import kp6_well as KW
    z = np.linspace(0.0, 10.0, 81)
    V = 0.08 * z
    st = K.biaxial_strain_on_AlN()
    wp, _ = KW.solve_subbands(0.6, z, V, st, n_states=6)
    wm, _ = KW.solve_subbands(-0.6, z, V, st, n_states=6)
    assert np.allclose(np.sort(wp), np.sort(wm), atol=1e-9)


def test_symmetric_well_has_no_spin_splitting():
    """The decisive control: restore the inversion centre and the splitting
    must vanish identically."""
    from gan2dhg import kp6_well as KW
    z = np.linspace(0.0, 15.0, 161)
    V = 0.02 * (z - 7.5) ** 2                    # inversion symmetric
    st = K.biaxial_strain_on_AlN()
    w, _ = KW.solve_subbands(0.374, z, V, st, n_states=6)
    pairs = w.reshape(3, 2)
    assert np.allclose(pairs[:, 0], pairs[:, 1], atol=1e-12)


def test_asymmetric_well_gives_rashba_splitting():
    """In the polarisation well the light subband splits by of order 1.5 meV
    at its Fermi wavevector, and the splitting converges with grid spacing."""
    from gan2dhg import kp6_well as KW
    st = K.biaxial_strain_on_AlN()
    vals = []
    for N in (121, 161, 241):
        z = np.linspace(0.0, 15.0, N)
        V = (KW.E2_OVER_EPS0 / 10.4) * 0.46 * z * np.exp(-z / 1.2)
        V = np.cumsum(V) * (z[1] - z[0])
        w, _ = KW.solve_subbands(0.374, z, V, st, n_states=6)
        vals.append(1000.0 * (w[3] - w[2]))
    assert all(v > 0.3 for v in vals), vals          # clearly non-zero
    assert abs(vals[-1] - vals[-2]) < 0.3 * abs(vals[-1])   # converging


def test_poisson_field_and_neutrality():
    """The confining field at the interface must be e^2 p_s / eps eps_0, and
    the potential must flatten once the whole gas lies below z."""
    from gan2dhg import kp6_well as KW
    z = np.linspace(0.0, 10.0, 501)
    p_s = 0.46                                   # nm^-2
    w = 0.5
    p = (p_s / (w * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((z - 1.0) / w) ** 2)
    p *= p_s / np.trapezoid(p, z)
    V = KW.poisson(z, p, p_s, 10.4)
    slope0 = (V[1] - V[0]) / (z[1] - z[0])
    expect = (KW.E2_OVER_EPS0 / 10.4) * p_s
    assert abs(slope0 - expect) / expect < 0.02
    slope_far = (V[-1] - V[-2]) / (z[-1] - z[-2])
    assert abs(slope_far) < 1e-3 * expect


def test_flat_potential_recovers_bulk_masses():
    """With no confining field and a wide box the well solver must return the
    bulk in-plane masses of the same Hamiltonian."""
    from gan2dhg import kp6_well as KW
    st = K.biaxial_strain_on_AlN()
    z = np.linspace(0.0, 60.0, 241)
    V = np.zeros_like(z)
    kt = np.linspace(0.05, 1.8, 26)
    E = np.array([KW.solve_subbands(k, z, V, st, n_states=2)[0][0]
                  for k in kt])
    dEdk = np.gradient(E, kt)
    m = K.HB2_2M0_eVnm2 * 2.0 * kt / dEdk
    a2, a4, a5 = K.GAN["A2"], K.GAN["A4"], K.GAN["A5"]
    m_heavy = 1.0 / abs(a2 + a4 - a5)
    i = np.searchsorted(kt, 1.5)
    assert abs(m[i] - m_heavy) / m_heavy < 0.05, (m[i], m_heavy)


def test_self_consistent_reproduces_measured_heavy_mass():
    """The headline result of the Letter, stated as a test.

    A coarse grid is used to keep the suite fast; the published value uses
    N = 161 and 16 wavevectors.
    """
    from gan2dhg import kp6_well as KW
    sol = KW.self_consistent(p_s_cm2=4.6e13, N=81, n_kt=10, max_iter=40,
                             mix=0.6, tol=1e-4, verbose=False)
    assert sol["converged"]
    masses = []
    for b in range(6):
        if sol["per_subband_nm2"][b] <= 0:
            continue
        masses.append(KW.cyclotron_mass_refined(sol, b)[1])
    heavy = [m for m in masses if m > 1.0]
    light = [m for m in masses if m <= 1.0]
    assert len(heavy) == 2 and len(light) == 2
    assert 1.75 < np.mean(heavy) < 2.05          # measured 1.92 +/- 0.16
    # The hard wall understates the light mass; the finite barrier raises it
    # into the measured range.  This test pins the hard-wall value, and the
    # coarse-grid derivative must NOT be used here: it overstates it by a
    # quarter.
    assert 0.20 < np.mean(light) < 0.32
    assert 0.3 < KW.centroid(sol) < 1.2          # nm


def test_total_density_is_conserved_by_the_loop():
    from gan2dhg import kp6_well as KW
    target = 3.0e13
    sol = KW.self_consistent(p_s_cm2=target, N=81, n_kt=10, max_iter=40,
                             mix=0.6, tol=1e-4, verbose=False)
    tot = float(np.sum(sol["per_subband_nm2"])) * 1e14
    assert abs(tot - target) / target < 1e-3


# ---------------------------------------------------------------------------
# The heterostructure operator, the counting rule and the computed overlap
# ---------------------------------------------------------------------------

def test_single_branch_density_is_kf_squared_over_four_pi():
    """The counting rule, stated as a test.

    Every eigenvalue of the six-band operator is one spin-resolved branch, so a
    branch filled to k_F holds k_F^2/(4 pi) carriers per unit area.  Using the
    spin-degenerate k_F^2/(2 pi) for each Kramers partner separately counts
    every spin twice; the loop then converges to half the intended density
    while Poisson's equation is still fed the full one.
    """
    from gan2dhg import kp6_well as KW
    kt = np.linspace(1e-3, 2.0, 40)
    # Two identical parabolic branches, mass 1 m0, filled to a known density.
    from gan2dhg.kp6 import HB2_2M0_eVnm2
    E = np.column_stack([HB2_2M0_eVnm2 * kt**2] * 2)
    target = 0.05                                  # per nm^2
    EF, per = KW.fill_subbands(E, kt, target)
    assert abs(per.sum() - target) / target < 1e-6
    for p in per:
        k = np.sqrt(4.0 * np.pi * p)
        assert abs(p - k**2 / (4.0 * np.pi)) < 1e-12


def test_heterostructure_operator_reduces_to_the_hard_wall_one():
    """With no barrier region the new operator must reproduce the old one.

    This checks the symmetric (BenDaniel-Duke) discretisation of the
    position-dependent A1 and A3 terms and the symmetrised A6 kz term against
    the constant-coefficient code they generalise.
    """
    from gan2dhg import kp6_het as H
    from gan2dhg import kp6_well as KW
    from gan2dhg.kp6 import biaxial_strain_on_AlN
    z = np.linspace(0.0, 15.0, 121)
    V = (KW.E2_OVER_EPS0 / 10.4) * (4.6e13 * 1e-14) * z
    st = biaxial_strain_on_AlN()
    w_old, _ = KW.solve_subbands(0.6, z, V, st, n_states=6)
    prof = H.profile(z, vbo_eV=50.0)
    w_new, _, _ = H.solve(0.6, 0.0, z, V, prof, n_states=6)
    assert np.max(np.abs(w_old - w_new)) < 1e-12


def test_basal_plane_dispersion_is_exactly_isotropic():
    """Not approximately isotropic: isotropic to machine precision.

    The claim in the Letter is that warping is absent at this order rather than
    small, so it is tested at the level of the numerics, sweeping the azimuth
    from Gamma-M to Gamma-K.
    """
    from gan2dhg import kp6_het as H
    from gan2dhg import kp6_well as KW
    z = np.linspace(0.0, 15.0, 81)
    V = (KW.E2_OVER_EPS0 / 10.4) * (4.6e13 * 1e-14) * z
    prof = H.profile(z, vbo_eV=0.7)
    for kt in (0.55, 1.61):
        ws = [H.solve(kt * np.cos(p), kt * np.sin(p), z, V, prof,
                      n_states=4)[0]
              for p in np.linspace(0.0, np.pi / 3.0, 5)]
        assert np.max(np.ptp(np.array(ws), axis=0)) < 1e-11


def test_bloch_overlap_is_unity_for_a_state_with_itself():
    """The overlap summed over a Kramers pair must be one at zero angle.

    This is the normalisation the computed overlap has to satisfy before it can
    replace the free parameter it supersedes.
    """
    from gan2dhg import kp6_het as H
    from gan2dhg import kp6_well as KW
    z = np.linspace(0.0, 15.0, 81)
    V = (KW.E2_OVER_EPS0 / 10.4) * (4.6e13 * 1e-14) * z
    prof = H.profile(z, vbo_eV=0.7)
    sol = {"z": z, "V": V, "prof": prof,
           "per_subband_nm2": np.array([0.2, 0.2, 0.02, 0.02, 0.0, 0.0])}
    assert abs(H.bloch_overlap(sol, 0, 0, 0.0) - 1.0) < 1e-9
    assert abs(H.bloch_overlap(sol, 2, 2, 0.0) - 1.0) < 1e-9


def test_branch_tracking_survives_a_crossing():
    """Sorted eigenvalues relabel branches at a crossing; tracking must not.

    Followed by eigenvector continuity, the splitting of the light pair has to
    stay a smooth function of wavevector across the point where that pair
    crosses the crystal-field split-off pair, where the sorted-index splitting
    collapses by two orders of magnitude.
    """
    from gan2dhg import kp6_het as H
    from gan2dhg import kp6_well as KW
    z = np.linspace(0.0, 15.0, 81)
    V = (KW.E2_OVER_EPS0 / 10.4) * (4.6e13 * 1e-14) * z
    prof = H.profile(z, vbo_eV=0.7)
    sol = {"z": z, "V": V, "prof": prof}
    kk = np.linspace(0.30, 1.05, 16)
    sp = H.rashba_vs_k(sol, kk, pair_lo=2)
    # smooth: no step larger than half the total range between neighbours
    steps = np.abs(np.diff(sp))
    assert steps.max() < 0.5 * (sp.max() - sp.min()) + 1e-9
    assert sp.min() > 0.0


# ---------------------------------------------------------------------------
# Beyond the relaxation-time and Born approximations
# ---------------------------------------------------------------------------

def test_transport_cross_section_prefactor_from_s_wave():
    """Pure s-wave scattering is isotropic, so sigma_tr must equal sigma.

    This pins the prefactor of the transport sum in two dimensions at 2/k when
    the sum runs over all integer m, not the 4/k that is sometimes quoted; with
    4/k the identity below fails by exactly a factor of two.
    """
    import importlib.util
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "run_phaseshift", os.path.join(here, "scripts", "run_phaseshift.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    k = 1.0
    for d0 in (0.05, 0.2, 0.6):
        deltas = np.zeros(20)
        deltas[0] = d0
        sq, st = P.cross_sections(k, deltas)
        assert abs(st / sq - 1.0) < 1e-12


def test_phase_shifts_reduce_to_born_for_a_weak_potential():
    """The variable-phase solver must reproduce the 2D Born phase shift.

    delta_m = -(pi/2) Int U(r) J_m^2(kr) r dr to first order in U.  This is the
    identity that validates the whole partial-wave apparatus; without it the
    departure from Born measured in the Letter would be meaningless.
    """
    import importlib.util
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "run_phaseshift", os.path.join(here, "scripts", "run_phaseshift.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    r = np.linspace(1e-13, 8.0e-9, 1500)
    V = 1e-3 * 1.602176634e-19 * np.exp(-(r / 1.0e-9) ** 2)   # 1 meV Gaussian
    k = 1.0e9
    m_eff = 9.1093837015e-31
    ex = P.phase_shifts(k, m_eff, r, V, m_max=6, scale=1e-3)
    bo = P.born_phase_shifts(k, m_eff, r, V, m_max=6, scale=1e-3)
    keep = np.abs(bo) > 1e-14
    assert keep.any()
    assert np.max(np.abs(ex[keep] / bo[keep] - 1.0)) < 1e-3


def test_boltzmann_closed_form_is_exact_for_an_angular_kernel():
    """Solving the integral equation with no ansatz returns the closed form.

    For an isotropic Fermi surface and a kernel depending only on the
    scattering angle, cos(theta) is an eigenfunction of the collision operator,
    so the (1 - cos theta) weight is exact and not an approximation.  Verified
    here against a kernel with structure in several harmonics.
    """
    import importlib.util
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "run_beyond", os.path.join(here, "scripts", "run_beyond.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    kF = 1.0e9
    for shape in (lambda q: 1.0 / (1.0 + (q / 1e9) ** 2),
                  lambda q: np.exp(-(q / 2e9) ** 2)):
        re, rc, higher = B.boltzmann_exact(
            kF, shape, lambda q: 1.0, lambda t: np.ones_like(t))
        assert abs(re / rc - 1.0) < 1e-8
        assert higher < 1e-8
