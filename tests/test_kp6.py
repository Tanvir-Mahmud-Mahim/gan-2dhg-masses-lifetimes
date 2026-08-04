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

from gpfet import kp6 as K
from gpfet import scatter2d as S

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


def test_confinement_raises_light_mass_not_heavy():
    """The central claim of the mass section, stated as a test."""
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
