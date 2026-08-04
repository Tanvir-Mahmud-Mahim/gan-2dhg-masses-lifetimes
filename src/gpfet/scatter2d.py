"""Band-resolved elastic scattering in a two-dimensional hole gas.

PURPOSE
-------
Three experiments have now been performed on the same polarisation-induced
GaN/AlN two-dimensional hole gas, and between them they report, for each of the
two occupied valence subbands, a transport lifetime, a quantum (single
particle) lifetime, and a cyclotron lifetime.  The state of the art in
modelling this system, Bader et al., Appl. Phys. Lett. 114, 253501 (2019),
deliberately does not resolve the elastic scattering into mechanisms, on the
stated grounds that the candidates "are temperature-independent and have
similar dependence on effective masses".

This module tests that premise.  In two dimensions the candidate mechanisms do
NOT share a dependence on mass, sheet density or standoff distance, and the
quantity that separates them most sharply is the ratio

    tau_tr / tau_q

because it measures the angular character of the scattering and nothing else.
A ratio near unity means isotropic, genuinely short-range scattering.  A ratio
much greater than unity means forward-peaked, long-range scattering.  The ratio
is dimensionless and independent of the disorder strength, so it can be
compared with experiment without fitting any amplitude.

FORMALISM
---------
For elastic scattering on a circular Fermi surface of a single two-dimensional
parabolic subband, with q = 2 k_F sin(theta/2),

    1/tau_q  = (m / (2 pi hbar^3)) * Int_0^{2pi} dtheta  W(q)
    1/tau_tr = (m / (2 pi hbar^3)) * Int_0^{2pi} dtheta  W(q) (1 - cos theta)

where W(q) is the squared, screened, form-factor-weighted matrix element.  See
Ando, Fowler and Stern, Rev. Mod. Phys. 54, 437 (1982).

Screening is treated in the static random phase approximation in the long
wavelength limit, generalised to two occupied subbands:

    epsilon(q) = 1 + (q_s / q) F_sc(q),
    q_s = (e^2 / (2 eps eps0)) * sum_i m_i / (pi hbar^2)

Both subbands contribute to the polarisability, which is the two-band feature
that a single-band treatment misses.  A subband only contributes for q below
its own 2 k_F, which is enforced.

WHAT IS AND IS NOT CLAIMED
--------------------------
This is a relaxation-time treatment with parabolic subbands and a variational
envelope function.  It is not an ab initio calculation and it does not compete
with one for absolute mobilities.  What it does compute, and what is compared
with experiment, is the RATIO tau_tr/tau_q for each band, and the ratio between
bands, in which the disorder amplitude cancels.
"""

import numpy as np
from scipy.integrate import quad

from .constants import HBAR, M0, PI, Q

EPS0 = 8.8541878128e-12          # F/m


# ---------------------------------------------------------------------------
# Envelope function
# ---------------------------------------------------------------------------

def fang_howard_b(n_s_cm2, n_depl_cm2, m_z_over_m0, eps_r):
    """Fang-Howard variational parameter b (1/m) for a triangular well.

    b = [ 33 pi m_z e^2 (n_depl + 11 n_s / 32) / (2 eps eps0 hbar^2) ]^{1/3}

    Fang and Howard, Phys. Rev. Lett. 16, 797 (1966); see also Ando, Fowler and
    Stern, Rev. Mod. Phys. 54, 437 (1982), Eq. (3.19).
    """
    n_s = n_s_cm2 * 1.0e4          # m^-2
    n_d = n_depl_cm2 * 1.0e4
    mz = m_z_over_m0 * M0
    num = 33.0 * PI * mz * Q**2 * (n_d + 11.0 * n_s / 32.0)
    den = 2.0 * eps_r * EPS0 * HBAR**2
    return (num / den) ** (1.0 / 3.0)


def form_factor_carrier(q, b):
    """Subband form factor F(q) for the Fang-Howard envelope.

    F(q) = (1 + 9 (q/b) / 8 + 3 (q/b)^2 / 8) / (1 + q/b)^3

    This is the self-consistent carrier form factor; the standard result for
    charge located OUTSIDE the well is F_rem below.
    """
    x = q / b
    return (1.0 + 1.125 * x + 0.375 * x * x) / (1.0 + x) ** 3


def form_factor_remote(q, b):
    """Form factor for a charge sheet outside the well, F_rem(q) = (1+q/b)^-3."""
    return (1.0 + q / b) ** -3


# ---------------------------------------------------------------------------
# Screening, generalised to several occupied subbands
# ---------------------------------------------------------------------------

def screening_wavevector(bands, eps_r):
    """q_s (1/m) from the sum of subband densities of states.

    bands: list of dicts with keys m_over_m0 and n_s_cm2.
    """
    tot = sum(b["m_over_m0"] * M0 / (PI * HBAR**2) for b in bands)
    return Q**2 * tot / (2.0 * eps_r * EPS0)


def dielectric(q, bands, eps_r, b_fh):
    """Static dielectric function, two-band, with the 2 k_F cut-off per band."""
    pol = 0.0
    for bd in bands:
        kF = fermi_wavevector(bd["n_s_cm2"])
        # A subband screens only for q <= 2 k_F; beyond that its
        # polarisability falls off (Stern 1967).
        if q <= 2.0 * kF:
            pol += bd["m_over_m0"] * M0 / (PI * HBAR**2)
        else:
            m = bd["m_over_m0"] * M0
            pol += (m / (PI * HBAR**2)) * (
                1.0 - np.sqrt(1.0 - (2.0 * kF / q) ** 2))
    qs = Q**2 * pol / (2.0 * eps_r * EPS0)
    return 1.0 + (qs / q) * form_factor_carrier(q, b_fh)


def fermi_wavevector(n_s_cm2, spin_degenerate=True):
    """k_F (1/m) for a two-dimensional subband.

    With spin degeneracy, n = k_F^2 / (2 pi), so k_F = sqrt(2 pi n).
    """
    n = n_s_cm2 * 1.0e4
    return np.sqrt(2.0 * PI * n) if spin_degenerate else np.sqrt(4.0 * PI * n)


# ---------------------------------------------------------------------------
# Matrix elements, squared and angle resolved
# ---------------------------------------------------------------------------

def w_remote_impurity(q, N_i_cm2, d_m, b_fh, eps_r):
    """Remote ionised sheet of areal density N_i at standoff d.

    In this heterostructure the obvious candidate is the Mg-doped top 5 nm of
    the 15 nm GaN layer, whose ionised acceptors stand off the interface by
    about 10 to 15 nm.
    """
    N_i = N_i_cm2 * 1.0e4
    v = Q**2 / (2.0 * eps_r * EPS0 * q)
    return N_i * (v * np.exp(-q * d_m) * form_factor_remote(q, b_fh)) ** 2


def w_background_impurity(q, N_b_cm3, b_fh, eps_r):
    """Background ionised impurities distributed through the channel."""
    N_b = N_b_cm3 * 1.0e6
    v = Q**2 / (2.0 * eps_r * EPS0 * q)
    # Integrating a uniform distribution over the envelope gives, to the
    # accuracy of this treatment, an effective sheet density N_b / b.
    return (N_b / b_fh) * (v * form_factor_carrier(q, b_fh)) ** 2


def w_interface_roughness(q, delta_m, lam_m, F_eff):
    """Gaussian-correlated interface roughness.

    <|U(q)|^2> = pi delta^2 lambda^2 exp(-q^2 lambda^2 / 4) (e F_eff)^2

    F_eff is the effective electric field pressing the gas against the
    interface.  Ando, Fowler and Stern, Rev. Mod. Phys. 54, 437 (1982).
    """
    return (PI * delta_m**2 * lam_m**2
            * np.exp(-(q * lam_m) ** 2 / 4.0) * (Q * F_eff) ** 2)


def w_dislocation(q, N_dis_cm2, c_lat_m, f_occ, eps_r, b_fh):
    """Charged threading dislocations, modelled as lines of charge.

    Each dislocation carries a linear charge density f_occ * e / c along the
    c axis.  Only the in-plane scattering enters.
    """
    N = N_dis_cm2 * 1.0e4
    lam_lin = f_occ * Q / c_lat_m
    v = lam_lin * Q / (2.0 * eps_r * EPS0 * q**2)
    return N * (v * form_factor_carrier(q, b_fh)) ** 2


# ---------------------------------------------------------------------------
# Lifetimes
# ---------------------------------------------------------------------------

def _rates(band, bands, w_fn, eps_r, b_fh, screen=True):
    """Return (1/tau_q, 1/tau_tr) in s^-1 for one subband."""
    m = band["m_over_m0"] * M0
    kF = fermi_wavevector(band["n_s_cm2"])
    pref = m / (2.0 * PI * HBAR**3)

    def integrand(theta, transport):
        q = 2.0 * kF * np.sin(theta / 2.0)
        if q <= 0.0:
            return 0.0
        w = w_fn(q)
        if screen:
            w = w / dielectric(q, bands, eps_r, b_fh) ** 2
        return w * ((1.0 - np.cos(theta)) if transport else 1.0)

    iq = quad(integrand, 1e-6, 2.0 * PI - 1e-6, args=(False,), limit=200)[0]
    it = quad(integrand, 1e-6, 2.0 * PI - 1e-6, args=(True,), limit=200)[0]
    return pref * iq, pref * it


def lifetimes(band, bands, w_fn, eps_r, b_fh, screen=True):
    """Quantum and transport lifetimes (s) and their ratio, for one subband."""
    rq, rt = _rates(band, bands, w_fn, eps_r, b_fh, screen)
    tq = 1.0 / rq if rq > 0 else np.inf
    tt = 1.0 / rt if rt > 0 else np.inf
    return {"tau_q_s": tq, "tau_tr_s": tt,
            "ratio": tt / tq if np.isfinite(tt / tq) else np.nan,
            "mu_tr_cm2Vs": Q * tt / (band["m_over_m0"] * M0) * 1.0e4,
            "mu_q_cm2Vs": Q * tq / (band["m_over_m0"] * M0) * 1.0e4}


# ---------------------------------------------------------------------------
# Two-band treatment with interband scattering
# ---------------------------------------------------------------------------
#
# Both subbands of this gas are cut by the same Fermi level, so a hole on the
# light-hole Fermi circle can scatter elastically onto the heavy-hole Fermi
# circle and back.  The heavy band carries the larger density of states, so for
# the light band this channel need not be small.  Single-band relaxation-time
# treatments omit it entirely, and it is the one term whose angular character
# is not controlled by a single Fermi wavevector: the momentum transfer for an
# interband event runs from |k_Fi - k_Fj| to k_Fi + k_Fj, never reaching zero.
#
# The transport lifetimes are then coupled and follow from the linearised
# Boltzmann equation rather than from a single integral.  Writing
#
#     A_ij = (m_j / 2 pi hbar^3) Int dtheta W_ij(q(theta))
#     B_ij = (m_j / 2 pi hbar^3) Int dtheta W_ij(q(theta)) cos theta
#
# with q^2 = k_Fi^2 + k_Fj^2 - 2 k_Fi k_Fj cos theta, the quantum rate of band
# i is the total out-scattering rate, sum_j A_ij, while the transport lifetimes
# solve the coupled system
#
#     sum_j [ delta_ij sum_k A_ik - B_ij (v_j / v_i) ] tau_tr,j = 1
#
# which reduces to the familiar (1 - cos theta) average when the off-diagonal
# terms vanish.


def _q_interband(theta, kFi, kFj):
    return np.sqrt(np.maximum(
        kFi**2 + kFj**2 - 2.0 * kFi * kFj * np.cos(theta), 0.0))


def coupling_matrices(bands, w_fn, eps_r, b_fh, overlap=None, screen=True):
    """Return the A and B matrices defined above, in s^-1.

    overlap[i][j] is the squared Bloch overlap between subbands i and j; it is
    unity on the diagonal.  For the interband term it is a number below unity
    whose value is not known independently, so it is carried as an explicit
    parameter and its influence is reported rather than hidden.
    """
    n = len(bands)
    kF = [fermi_wavevector(b["n_s_cm2"]) for b in bands]
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    if overlap is None:
        overlap = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            mj = bands[j]["m_over_m0"] * M0
            pref = mj / (2.0 * PI * HBAR**3) * overlap[i][j]

            def integ(theta, weight):
                q = _q_interband(theta, kF[i], kF[j])
                if q <= 0.0:
                    return 0.0
                w = w_fn(q)
                if screen:
                    w = w / dielectric(q, bands, eps_r, b_fh) ** 2
                return w * (np.cos(theta) if weight else 1.0)

            A[i, j] = pref * quad(integ, 1e-6, 2.0 * PI - 1e-6,
                                  args=(False,), limit=200)[0]
            B[i, j] = pref * quad(integ, 1e-6, 2.0 * PI - 1e-6,
                                  args=(True,), limit=200)[0]
    return A, B


def two_band_lifetimes(bands, w_fn, eps_r, b_fh, overlap=None, screen=True):
    """Quantum and transport lifetimes for every subband, coupled."""
    A, B = coupling_matrices(bands, w_fn, eps_r, b_fh, overlap, screen)
    n = len(bands)
    kF = np.array([fermi_wavevector(b["n_s_cm2"]) for b in bands])
    m = np.array([b["m_over_m0"] * M0 for b in bands])
    v = HBAR * kF / m

    tau_q = 1.0 / A.sum(axis=1)

    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] += A[i, :].sum()
        for j in range(n):
            M[i, j] -= B[i, j] * v[j] / v[i]
    tau_tr = np.linalg.solve(M, np.ones(n))

    return {
        "tau_q_s": tau_q,
        "tau_tr_s": tau_tr,
        "ratio": tau_tr / tau_q,
        "mu_cm2Vs": np.array([Q * tau_tr[i] / m[i] * 1.0e4 for i in range(n)]),
        "A": A, "B": B,
        "interband_fraction_of_quantum_rate": np.array(
            [1.0 - A[i, i] / A[i, :].sum() for i in range(n)]),
    }


def mobility_from_tau(tau_s, m_over_m0):
    """mu in cm^2 V^-1 s^-1."""
    return Q * tau_s / (m_over_m0 * M0) * 1.0e4


def tau_from_mobility(mu_cm2Vs, m_over_m0):
    """tau in s."""
    return mu_cm2Vs * 1.0e-4 * m_over_m0 * M0 / Q
