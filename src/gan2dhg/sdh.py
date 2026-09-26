"""How strongly each subband's density-of-states oscillation appears in rho_xx.

In a Shubnikov-de Haas measurement the oscillation of the density of states at
the Fermi level, delta_j for subband pair j, modulates the scattering rates and
hence the resistance.  In a two-carrier gas the two carriers contribute very
unequally to rho_xx: in strong fields the carrier with the lower mobility (here
the heavy holes) carries most of sigma_xx.  A relative oscillation delta_j of
equal size therefore moves rho_xx by different amounts for the two subbands,
and the field at which an oscillation first becomes visible is not a measure of
its Dingle factor alone.

Model (first order in delta):

* Each carrier c is a Drude channel with density n_c and mobility mu_c.
  S_c = d ln rho_xx / d ln(1/tau_c) is the sensitivity of rho_xx to its
  scattering rate.
* The scattering rates follow from the coupled two-subband Boltzmann equation
  (scripts/run_tension.py).  Every rate from subband i into subband j is
  proportional to the density of states of j, so an oscillation delta_j scales
  column j of the rate matrices A and B by (1 + delta_j).  The response of the
  transport lifetimes, W[c, j] = -d ln tau_tr,c / d delta_j, is obtained by
  solving the coupled equation with the scaled matrices.
* The relative oscillation of rho_xx at the frequency of subband j is then
  K_j delta_j with K_j = sum_c S_c W[c, j].

delta_j carries the Lifshitz-Kosevich factors of subband j; for the fundamental
harmonic its prefactor does not depend on the mass, so the ratio of the two
oscillation amplitudes in rho_xx is
    A_H / A_L = K_H R_T,H R_D,H R_s,H / (K_L R_T,L R_D,L R_s,L).
The spin factors R_s are unknown and are taken equal.
"""

import numpy as np
from scipy.optimize import brentq

KB = 8.617333262e-5              # eV/K
HBAR_E_OVER_M0 = 1.1576764e-4    # eV/T, hbar e / m0


def rho_xx(B, n, mu, rate_scale=(1.0, 1.0)):
    """Two-carrier Drude rho_xx (arbitrary units) at field B (T).

    n: densities (any common unit); mu: mobilities (m^2/Vs); rate_scale:
    factor multiplying each carrier's scattering rate.
    """
    sxx = sxy = 0.0
    for nc, muc, s in zip(n, mu, rate_scale):
        m = muc / s
        w = m * B
        s0 = nc * m
        sxx += s0 / (1.0 + w * w)
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


def amplitude_ratio(B, T, K, mu_q, m):
    """A_H / A_L predicted for quantum mobilities mu_q = (L, H) (m^2/Vs)."""
    aL = thermal_factor(m[0], B, T) * np.exp(-np.pi / (mu_q[0] * B))
    aH = thermal_factor(m[1], B, T) * np.exp(-np.pi / (mu_q[1] * B))
    return (K[1] * aH) / (K[0] * aL)


def heavy_quantum_mobility(ratio, B, T, K, mu_q_L, m):
    """Heavy-hole quantum mobility (m^2/Vs) that reproduces a measured A_H / A_L."""
    f = lambda mu: amplitude_ratio(B, T, K, (mu_q_L, mu), m) - ratio
    return brentq(f, 1e-4, 1.0, xtol=1e-9)
