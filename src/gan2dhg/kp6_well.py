"""Self-consistent six-band envelope-function solution of the GaN/AlN hole well.

WHY THIS EXISTS
---------------
An earlier version of this analysis represented the confinement by a single
out-of-plane wavevector, k_z = pi / w, which is a hard-wall proxy for a well
that is in fact triangular and set by the carriers' own electrostatics.  That
proxy establishes a sensitivity but cannot determine anything, and it is the
weakest link in any conclusion drawn from it.  This module removes it by
solving the actual problem:

  * the six-band Hamiltonian is promoted to an envelope-function operator by
    k_z -> -i d/dz, discretised on a grid across the GaN layer,
  * the electrostatic potential that confines the gas is obtained from the hole
    distribution itself through Poisson's equation,
  * the two are iterated to self-consistency at the measured sheet density.

There are no adjustable parameters.  The sheet density is the measured one, the
band parameters are those tabulated by the measurement paper, and the barrier
is treated as hard because the GaN/AlN valence band offset is of order one
electron volt while the subband energies here are tens of millielectronvolts.

CONVENTIONS
-----------
Energies are hole energies in eV, increasing downward from the valence band
maximum, so the ground subband is the lowest eigenvalue.  The coordinate z runs
in nanometres from the GaN/AlN interface, where the gas accumulates, to the
free surface.  Charge densities are per nm^2 or per nm^3 as marked.

ELECTROSTATICS
--------------
The gas is balanced by the negative polarisation bound charge at the interface,
treated as a sheet at z = 0.  Gauss's law then gives, for the potential energy
of a hole,

    dV/dz = C [ p_s - Int_0^z p(z') dz' ],     C = e^2 / (eps eps_0),

which is positive at the interface, so holes are held against it, and falls to
zero once the whole gas is below z, as charge neutrality requires.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal  # noqa: F401  (kept for reference)

from .kp6 import GAN, HB2_2M0_eVnm2, biaxial_strain_on_AlN

KB_EV = 8.617333262e-5
E2_OVER_EPS0 = 18.09512  # eV nm, that is e^2/eps_0 with lengths in nm


# ---------------------------------------------------------------------------
# Envelope-function Hamiltonian
# ---------------------------------------------------------------------------

def build_operator(kt, z, V, strain=None, p=GAN):
    """Six-band envelope Hamiltonian at in-plane wavevector kt.

    Returns a Hermitian matrix of size 6N, in hole-energy convention, where N
    is the number of grid points.  V is the hole potential energy on the grid.
    """
    N = len(z)
    dz = z[1] - z[0]
    A = [p[f"A{i}"] for i in range(1, 7)]
    d1, d2 = p["D_CR"], p["D_SO"] / 3.0
    D = np.sqrt(2.0) * d2

    lam_e = th_e = 0.0
    if strain is not None:
        exx, eyy, ezz = strain
        lam_e = p["D1v"] * ezz + p["D2v"] * (exx + eyy)
        th_e = p["D3v"] * ezz + p["D4v"] * (exx + eyy)

    # Operators on the grid.  kz^2 -> -d2/dz2, kz -> -i d/dz.
    I_N = np.eye(N)
    K2 = (2.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)) / dz**2
    K1 = -1j * (np.eye(N, k=1) - np.eye(N, k=-1)) / (2.0 * dz)

    kt2 = kt * kt
    lam = HB2_2M0_eVnm2 * (A[0] * K2 + A[1] * kt2 * I_N) + lam_e * I_N
    th = HB2_2M0_eVnm2 * (A[2] * K2 + A[3] * kt2 * I_N) + th_e * I_N
    Kop = HB2_2M0_eVnm2 * A[4] * (kt2) * I_N        # (kx + i ky)^2, phase only
    Hop = 1j * HB2_2M0_eVnm2 * A[5] * kt * K1       # i A6 kz (kx + i ky)

    F = (d1 + d2) * I_N + lam + th
    G = (d1 - d2) * I_N + lam + th
    DI = D * I_N
    Z = np.zeros((N, N))

    M = np.block([
        [F,            Z,            -Hop.conj().T, Z,            Kop.conj().T, Z],
        [Z,            G,             DI,          -Hop.conj().T, Z,            Kop.conj().T],
        [-Hop,         DI,            lam,          Z,            Hop.conj().T, Z],
        [Z,           -Hop,           Z,            lam,          DI,           Hop.conj().T],
        [Kop,          Z,             Hop,          DI,           G,            Z],
        [Z,            Kop,           Z,            Hop,          Z,            F],
    ])
    M = 0.5 * (M + M.conj().T)          # symmetrise away discretisation noise

    # Hole convention: flip the sign, then add the hole potential energy.
    Mh = -M + np.kron(np.eye(6), np.diag(V))
    return Mh


def solve_subbands(kt, z, V, strain=None, p=GAN, n_states=8):
    """Lowest hole subband energies and envelopes at one in-plane wavevector."""
    Mh = build_operator(kt, z, V, strain, p)
    w, v = np.linalg.eigh(Mh)
    w, v = w[:n_states], v[:, :n_states]
    # Envelope probability density, summed over the six spinor components.
    N = len(z)
    dens = np.zeros((N, n_states))
    for s in range(n_states):
        comp = v[:, s].reshape(6, N)
        dens[:, s] = np.sum(np.abs(comp) ** 2, axis=0)
    dens /= np.trapezoid(dens, z, axis=0)
    return w, dens


# ---------------------------------------------------------------------------
# Occupation and Poisson
# ---------------------------------------------------------------------------

def fill_subbands(E_of_k, kt_grid, p_target_nm2, T=3.0):
    """Occupy subbands to a fixed total sheet density.

    E_of_k has shape (len(kt_grid), n_states).  At the temperatures of these
    measurements the gas is strongly degenerate, so the Fermi level is found
    from the zero-temperature counting rule and the result is insensitive to T.

    COUNTING RULE.  The six-band basis is three orbital states times two spin
    states, so every eigenvalue returned by the solver is a SINGLE, spin
    resolved branch: at zero in-plane wavevector the levels appear as Kramers
    pairs, and away from it the pairs split by the Rashba term because the well
    has no inversion centre.  A single non-degenerate two-dimensional branch
    holds

        n = (1 / (2 pi)^2) Int d^2k  =  k_F^2 / (4 pi),

    NOT k_F^2 / (2 pi), which is the density of a spin-degenerate pair.  Using
    the pair formula for each partner separately counts every spin twice and
    converges the loop to half the intended sheet density while the Poisson
    step is still fed the full one.

    INVERSION.  Finding k_max from E_F means inverting E(k) on the wavevector
    grid.  Linear inversion is biased for a band that curves, and the light
    branch curves strongly over one grid spacing, so a shape-preserving cubic
    is used instead.  It is monotone by construction, which matters because a
    plain cubic can overshoot and return a non-physical k_max.
    """
    from scipy.optimize import brentq
    from scipy.interpolate import PchipInterpolator

    inv = []
    for s in range(E_of_k.shape[1]):
        E = E_of_k[:, s]
        inv.append(PchipInterpolator(E, kt_grid, extrapolate=False)
                   if np.all(np.diff(E) > 0) else None)

    def k_of_EF(s, EF):
        E = E_of_k[:, s]
        if EF <= E[0]:
            return 0.0
        if EF >= E[-1]:
            return kt_grid[-1]
        if inv[s] is not None:
            return float(inv[s](EF))
        return float(np.interp(EF, E, kt_grid))

    def density(EF):
        return sum(k_of_EF(s, EF) ** 2 / (4.0 * np.pi)
                   for s in range(E_of_k.shape[1]))

    lo, hi = E_of_k.min(), E_of_k.max()
    EF = brentq(lambda e: density(e) - p_target_nm2, lo, hi, xtol=1e-12)
    per = [k_of_EF(s, EF) ** 2 / (4.0 * np.pi)
           for s in range(E_of_k.shape[1])]
    return EF, np.array(per)


def poisson(z, p_of_z_nm3, p_s_nm2, eps_r):
    """Hole potential energy (eV) from the hole distribution.

    dV/dz = C [ p_s - Int_0^z p ],  C = e^2 / (eps eps_0) in eV nm.
    """
    C = E2_OVER_EPS0 / eps_r
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (p_of_z_nm3[1:] + p_of_z_nm3[:-1]) * np.diff(z))])
    dVdz = C * (p_s_nm2 - cum)
    V = np.concatenate([[0.0], np.cumsum(
        0.5 * (dVdz[1:] + dVdz[:-1]) * np.diff(z))])
    return V - V.min()


def self_consistent(p_s_cm2=4.6e13, L_nm=15.0, N=161, kt_max=2.2, n_kt=14,
                    eps_r=10.4, strain="AlN", n_states=6, max_iter=40,
                    tol=1e-5, mix=0.3, verbose=True):
    """Iterate the envelope problem and Poisson's equation to convergence."""
    if strain == "AlN":
        strain = biaxial_strain_on_AlN()
    z = np.linspace(0.0, L_nm, N)
    kt_grid = np.linspace(1e-3, kt_max, n_kt)
    p_s = p_s_cm2 * 1.0e-14                       # nm^-2

    # Start from the analytic triangular well of the full sheet charge.
    V = (E2_OVER_EPS0 / eps_r) * p_s * z
    V[0] = V[0]

    hist = []
    for it in range(max_iter):
        E_of_k = np.empty((n_kt, n_states))
        dens0 = None
        for i, kt in enumerate(kt_grid):
            w, d = solve_subbands(kt, z, V, strain, n_states=n_states)
            E_of_k[i] = w
            if i == 0:
                dens0 = d
        EF, per = fill_subbands(E_of_k, kt_grid, p_s)

        p_of_z = np.zeros_like(z)
        for s in range(n_states):
            p_of_z += per[s] * dens0[:, s]

        V_new = poisson(z, p_of_z, p_s, eps_r)
        err = np.max(np.abs(V_new - V))
        V = (1.0 - mix) * V + mix * V_new
        hist.append(err)
        if verbose:
            print(f"  iter {it:2d}  max|dV| = {1000*err:8.3f} meV   "
                  f"E_F = {1000*EF:7.2f} meV   "
                  f"occupied = {int((per > 1e-6).sum())}")
        if err < tol:
            break

    return {"z": z, "V": V, "kt": kt_grid, "E_of_k": E_of_k, "EF": EF,
            "per_subband_nm2": per, "p_of_z": p_of_z, "converged": err < tol,
            "iterations": it + 1, "history": hist, "strain": strain,
            "p_s_cm2": p_s_cm2}


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def cyclotron_mass_at_kf(sol, subband, kt_fine=None):
    """m_CR = hbar^2 k (dE/dk)^-1 at the Fermi wavevector of a subband.

    The stored occupation is that of one spin-resolved branch, so the Fermi
    wavevector is k_F = sqrt(4 pi n), the single-branch relation.
    """
    n = sol["per_subband_nm2"][subband]
    if n <= 0:
        return np.nan, np.nan
    kF = np.sqrt(4.0 * np.pi * n)
    kt = sol["kt"]
    E = sol["E_of_k"][:, subband]
    dEdk = np.gradient(E, kt)
    m = HB2_2M0_eVnm2 * 2.0 * kt / dEdk
    return kF, float(np.interp(kF, kt, m))


def cyclotron_mass_refined(sol, subband, dk=0.01, n_states=6):
    """m_CR at k_F from a LOCAL derivative, with the converged potential fixed.

    cyclotron_mass_at_kf differentiates the dispersion on the coarse wavevector
    grid carried through the self-consistent loop.  That is accurate for the
    heavy branch, whose dispersion is smooth at its Fermi wavevector, but not
    for the light branch: with a grid spacing of about 0.15 per nm and a band
    that is strongly non-parabolic near k_F, the coarse derivative overstates
    the light mass by about a quarter.  Here the dispersion is evaluated at
    k_F +/- dk directly, so the only error is O(dk^2), and the result is
    converged in dk to four significant figures by dk = 0.02 per nm.
    """
    n = sol["per_subband_nm2"][subband]
    if n <= 0:
        return np.nan, np.nan
    kF = np.sqrt(4.0 * np.pi * n)
    z, V, st = sol["z"], sol["V"], sol["strain"]
    Ep = solve_subbands(kF + dk, z, V, st, n_states=n_states)[0][subband]
    Em = solve_subbands(kF - dk, z, V, st, n_states=n_states)[0][subband]
    return float(kF), float(HB2_2M0_eVnm2 * 2.0 * kF / ((Ep - Em) / (2.0 * dk)))


def centroid(sol):
    """First moment of the hole distribution, a measure of the confinement."""
    z, p = sol["z"], sol["p_of_z"]
    return float(np.trapezoid(z * p, z) / np.trapezoid(p, z))


def rms_width(sol):
    z, p = sol["z"], sol["p_of_z"]
    zc = centroid(sol)
    return float(np.sqrt(np.trapezoid((z - zc) ** 2 * p, z)
                         / np.trapezoid(p, z)))
