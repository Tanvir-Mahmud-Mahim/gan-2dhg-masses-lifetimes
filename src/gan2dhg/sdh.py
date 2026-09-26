"""How strongly each subband's density-of-states oscillation appears in rho_xx.

In a Shubnikov-de Haas measurement the oscillation of the density of states at
the Fermi level, delta_j for subband j, enters the conductivity in two ways
(Dmitriev, Mirlin, Polyakov and Zudov, Rev. Mod. Phys. 84, 1709 (2012),
Sec. II.C.2; Eqs. 38 to 40 in the numbering of arXiv:1111.2176): it scales
every scattering rate into subband j, so it renormalizes the transport times,
and it multiplies the dissipative conductivity sigma_xx of the carriers of
subband j by nu_j / nu_j0.  For a single carrier in a strong field the two
together give Delta rho_xx / rho_xx = 2 Delta nu / nu_0, which with
Delta nu / nu_0 = -2 delta cos(2 pi E_F / hbar omega_c) (their Eq. 31) and the
thermal factor X / sinh X (their Eq. 40) is the standard amplitude
4 delta X / sinh X.  In a two-carrier gas the two carriers contribute very unequally to
rho_xx: in strong fields the carrier with the lower mobility (here the heavy
holes) carries most of sigma_xx, so a relative oscillation of equal size moves
rho_xx by different amounts for the two subbands, and the field at which an
oscillation first becomes visible is not a measure of its Dingle factor alone.

First-order model
-----------------
* Each carrier c is a Drude channel with density n_c and mobility mu_c.
* The scattering rates follow the coupled two-subband Boltzmann equation
  (scripts/run_tension.py).  Every rate from subband i into subband j is
  proportional to the density of states of j, so delta_j scales column j of the
  rate matrices A and B by (1 + delta_j).  W[c, j] = -d ln tau_tr,c / d delta_j
  follows from the coupled equation (dos_weights).
* sigma_xx of carrier c carries the factor (1 + delta_c) (dos_in_sxx=True, the
  form of Dmitriev et al.); sigma_xy does not, so that the Hall conductivity in
  a strong field stays n e / B.  dos_in_sxx=False keeps only the rates.
* The relative oscillation of rho_xx at the frequency of subband j is then
  K_j delta_j (response_weights).

delta_j carries the Lifshitz-Kosevich factors of subband j; for the fundamental
harmonic its prefactor does not depend on the mass, so the ratio of the two
oscillation amplitudes in rho_xx is
    A_H / A_L = K_H R_T,H R_D,H R_s,H / (K_L R_T,L R_D,L R_s,L).

Non-perturbative model
----------------------
rho_xx_quantum evaluates rho_xx(B) without expanding in delta: Lorentzian
Landau levels summed to all harmonics in closed form, spin splitting as two
shifted level combs, rates from the coupled equation with the energy-dependent
densities of states, sigma averaged over -df/de and inverted.  It contains the
harmonics, the products of the two oscillations (magneto-intersubband terms)
and the thermal average, and is used to test the first-order model and the
analysis of the measured oscillations (scripts/run_forward_calibration.py).
"""

import numpy as np
from scipy.optimize import brentq

KB = 8.617333262e-5              # eV/K
HBAR_E_OVER_M0 = 1.1576764e-4    # eV/T, hbar e / m0


# ---------------------------------------------------------------------------
# first-order response
# ---------------------------------------------------------------------------

def rho_xx(B, n, mu, rate_scale=(1.0, 1.0), dos=None):
    """Two-carrier Drude rho_xx (arbitrary units) at field B (T).

    n: densities (any common unit); mu: mobilities (m^2/Vs); rate_scale:
    factor multiplying each carrier's scattering rate; dos: optional factor
    nu_c / nu_c0 multiplying sigma_xx of each carrier.
    """
    sxx = sxy = 0.0
    for c, (nc, muc, s) in enumerate(zip(n, mu, rate_scale)):
        m = muc / s
        w = m * B
        s0 = nc * m
        sxx += s0 / (1.0 + w * w) * (1.0 if dos is None else dos[c])
        sxy += s0 * w / (1.0 + w * w)
    return sxx / (sxx ** 2 + sxy ** 2)


def sensitivities(B, n, mu, h=1e-6):
    """S_c = d ln rho_xx / d ln(1/tau_c) for each carrier."""
    out = []
    for c in range(len(n)):
        up = [1.0] * len(n)
        dn = [1.0] * len(n)
        up[c] += h
        dn[c] -= h
        out.append((np.log(rho_xx(B, n, mu, up)) - np.log(rho_xx(B, n, mu, dn)))
                   / (2 * h))
    return np.array(out)


def response_weights(B, n, mu, W, dos_in_sxx=True, h=1e-6):
    """K_j = d ln rho_xx / d delta_j.

    W[c, j] = -d ln tau_tr,c / d delta_j (dos_weights).  With dos_in_sxx=False
    this equals sensitivities(B, n, mu) @ W.
    """
    W = np.asarray(W, float)
    nc = len(n)
    out = []
    for j in range(nc):
        vals = []
        for sgn in (1.0, -1.0):
            d = np.zeros(nc)
            d[j] = sgn * h
            scale = 1.0 + W @ d
            dos = (1.0 + d) if dos_in_sxx else None
            vals.append(np.log(rho_xx(B, n, mu, scale, dos)))
        out.append((vals[0] - vals[1]) / (2 * h))
    return np.array(out)


def coupled_tau(A, Bk, v):
    """Transport lifetimes of the coupled equation (units of 1/A)."""
    M = np.diag(A.sum(axis=1)) - Bk * (v[None, :] / v[:, None])
    return np.linalg.solve(M, np.ones(len(v)))


def dos_weights(A, Bk, v, h=1e-6):
    """W[c, j] = -d ln tau_tr,c / d delta_j, from the coupled equation.

    The rows sum to one: scaling every rate by (1 + delta) scales every
    lifetime by 1/(1 + delta).
    """
    n = len(v)
    W = np.zeros((n, n))
    for j in range(n):
        s = np.ones(n)
        s[j] += h
        t1 = coupled_tau(A * s[None, :], Bk * s[None, :], v)
        s = np.ones(n)
        s[j] -= h
        t2 = coupled_tau(A * s[None, :], Bk * s[None, :], v)
        W[:, j] = -(np.log(t1) - np.log(t2)) / (2 * h)
    return W


def thermal_factor(m, B, T):
    """Lifshitz-Kosevich X / sinh X for mass m (units of m0)."""
    X = 2 * np.pi ** 2 * KB * T * m / (HBAR_E_OVER_M0 * B)
    return X / np.sinh(X)


def amplitude_ratio(B, T, K, mu_q, m, spin=(1.0, 1.0)):
    """A_H / A_L predicted for quantum mobilities mu_q = (L, H) (m^2/Vs).

    spin: spin reduction factors |R_s| of the two subbands.
    """
    aL = spin[0] * thermal_factor(m[0], B, T) * np.exp(-np.pi / (mu_q[0] * B))
    aH = spin[1] * thermal_factor(m[1], B, T) * np.exp(-np.pi / (mu_q[1] * B))
    return (K[1] * aH) / (K[0] * aL)


def heavy_quantum_mobility(ratio, B, T, K, mu_q_L, m, spin=(1.0, 1.0)):
    """Heavy-hole quantum mobility (m^2/Vs) that reproduces a measured A_H / A_L."""
    f = lambda mu: amplitude_ratio(B, T, K, (mu_q_L, mu), m, spin) - ratio
    return brentq(f, 1e-4, 1.0, xtol=1e-9)


def spin_factor_from_harmonics(r21):
    """|cos(pi S)| from the measured ratio R_s(2)/R_s(1) = cos(2 pi S)/cos(pi S).

    The positive root of 2c^2 - r21 c - 1 = 0.
    """
    return (r21 + np.sqrt(r21 ** 2 + 8.0)) / 4.0


# ---------------------------------------------------------------------------
# non-perturbative model
# ---------------------------------------------------------------------------

def lorentz_dos(a, theta):
    """nu / nu_0 for Lorentzian Landau levels, summed to all harmonics.

    1 + 2 sum_r (-1)^r exp(-r a) cos(r theta) = sinh a / (cosh a + cos theta),
    with a = pi / (mu_q B) and theta = 2 pi (E - E_edge) / (hbar omega_c).
    """
    return np.sinh(a) / (np.cosh(a) + np.cos(theta))


def _tau_batch(A, Bk, v, s):
    """Coupled 2x2 transport times for rows of density-of-states ratios s."""
    As = A[None, :, :] * s[:, None, :]
    Bs = Bk[None, :, :] * s[:, None, :]
    m00 = As[:, 0].sum(-1) - Bs[:, 0, 0]
    m11 = As[:, 1].sum(-1) - Bs[:, 1, 1]
    m01 = -Bs[:, 0, 1] * v[1] / v[0]
    m10 = -Bs[:, 1, 0] * v[0] / v[1]
    det = m00 * m11 - m01 * m10
    return np.stack([(m11 - m01) / det, (m00 - m10) / det], -1)


def rho_xx_quantum(inv_B, T, A, Bk, v, n, mu_hall, masses, freqs, mu_q,
                   dos_in_sxx=True, spin=(0.0, 0.0), ne=161):
    """Non-perturbative rho_xx (arbitrary units) of the two-subband gas.

    inv_B: array of 1/B (1/T).  A, Bk, v: coupled-equation matrices and
    velocities of the disorder (only their ratios matter); n, mu_hall: densities
    and Hall mobilities (m^2/Vs), to which the zero-oscillation mobilities are
    normalized; masses (m0), freqs (T), mu_q (m^2/Vs) of the two subbands.
    spin: S = g* m* / 2 m0 of each subband; the two spin combs are shifted by
    +-pi S, which gives the spin reduction factor cos(pi r S) of harmonic r.
    """
    inv_B = np.atleast_1d(np.asarray(inv_B, float))
    n = np.asarray(n, float)
    mu_hall = np.asarray(mu_hall, float)
    masses = np.asarray(masses, float)
    freqs = np.asarray(freqs, float)
    mu_q = np.asarray(mu_q, float)
    spin = np.asarray(spin, float)
    tau0 = _tau_batch(A, Bk, v, np.ones((1, 2)))[0]
    kT = KB * max(T, 1e-3)
    e = np.linspace(-9 * kT, 9 * kT, ne)
    wgt = np.exp(e / kT) / (1 + np.exp(e / kT)) ** 2
    wgt /= wgt.sum()
    out = np.empty(len(inv_B))
    for i, x in enumerate(inv_B):
        B = 1.0 / x
        hw = HBAR_E_OVER_M0 * B / masses
        a = np.pi / (mu_q * B)
        th = 2 * np.pi * (freqs[None, :] * x + e[:, None] / hw[None, :])
        s = 0.5 * (lorentz_dos(a[None, :], th + np.pi * spin[None, :])
                   + lorentz_dos(a[None, :], th - np.pi * spin[None, :]))
        tau = _tau_batch(A, Bk, v, s)
        mu = mu_hall[None, :] * tau / tau0[None, :]
        wc = mu * B
        pref = n[None, :] * (s if dos_in_sxx else 1.0)
        sxx = (wgt * (pref * mu / (1 + wc ** 2)).sum(-1)).sum()
        sxy = (wgt * (n[None, :] * mu * wc / (1 + wc ** 2)).sum(-1)).sum()
        out[i] = sxx / (sxx ** 2 + sxy ** 2)
    return out
