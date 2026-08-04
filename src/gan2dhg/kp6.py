"""Six-band wurtzite k.p valence band structure for the strained GaN 2DHG.

The Hamiltonian is the one of Chuang and Chang, Phys. Rev. B 54, 2491 (1996),
in the basis

  |1> = |(X + iY)/sqrt2, up>   |4> = |(X - iY)/sqrt2, down>
  |2> = |(X - iY)/sqrt2, up>   |5> = |(X + iY)/sqrt2, down>
  |3> = |Z, up>                |6> = |Z, down>

with

  F = D1 + D2 + lam + th,        G = D1 - D2 + lam + th
  lam = (hbar^2 / 2 m0)(A1 kz^2 + A2 kt^2) + lam_eps
  th  = (hbar^2 / 2 m0)(A3 kz^2 + A4 kt^2) + th_eps
  K   = (hbar^2 / 2 m0) A5 (kx + i ky)^2
  H   = (hbar^2 / 2 m0) A6 (kx + i ky) kz
  Dlt = sqrt(2) D3

and D1 = Delta_CR, D2 = D3 = Delta_SO / 3.  Strain enters through the
Bir-Pikus terms lam_eps = D1v ezz + D2v (exx + eyy) and
th_eps = D3v ezz + D4v (exx + eyy); the shear potentials D5v and D6v vanish for
the biaxial case treated here.

PARAMETERS
----------
Every number is taken from Extended Data Table 1 of Chang et al.,
Nature Electronics 9, 346 (2026), doi 10.1038/s41928-026-01590-8, read from
the arXiv full text of arXiv:2501.16213.  That table in turn attributes the
Rashba-Sheka-Pikus parameters to Rinke et al., Phys. Rev. B 77, 075202 (2008)
and the deformation potentials to Yan et al., Phys. Rev. B 90, 125118 (2014).
Using the same inputs as the measurement paper means any difference in the
conclusions comes from the analysis and not from the parameter choice.

A NOTE ON WHAT IS ISOTROPIC AND WHAT IS NOT
-------------------------------------------
At kz = 0 the off-diagonal coupling H vanishes and K depends on the in-plane
angle only through a phase, so the basal-plane dispersion of this Hamiltonian
is isotropic.  Circular Fermi contours are therefore justified at this order,
and the feature that matters for the present problem is not warping but
non-parabolicity, which is strong: the three valence bands sit within a few
tens of meV of one another and repel as the in-plane wavevector grows.
"""

import numpy as np

HBAR = 1.054571817e-34
M0 = 9.1093837015e-31
QE = 1.602176634e-19
HB2_2M0_eVnm2 = HBAR**2 / (2.0 * M0) / QE * 1.0e18   # eV nm^2

# --- GaN, Extended Data Table 1 of Chang et al. ----------------------------
GAN = {
    "A1": -5.947, "A2": -0.528, "A3": 5.414,
    "A4": -2.512, "A5": -2.510, "A6": -3.202,
    "D_CR": 0.010, "D_SO": 0.017,
    "a_A": 3.189, "c_A": 5.185,
    "C11": 390.0, "C12": 145.0, "C13": 106.0, "C33": 398.0, "C44": 105.0,
    # Valence deformation potentials.  The table lists the combinations
    # (acz - D1) and (act - D2) together with acz and act, from which D1 and
    # D2 follow; D3 and D4 are listed directly.
    "acz": -11.3, "act": -4.9, "acz_m_D1": -6.07, "act_m_D2": -8.88,
    "D3v": 5.38, "D4v": -2.69, "D5v": -2.56, "D6v": -3.88,
}
ALN = {"a_A": 3.112, "c_A": 4.982}

GAN["D1v"] = GAN["acz"] - GAN["acz_m_D1"]     # eV
GAN["D2v"] = GAN["act"] - GAN["act_m_D2"]     # eV


def biaxial_strain_on_AlN(p=GAN, sub=ALN):
    """Pseudomorphic strain of GaN grown on bulk AlN."""
    exx = eyy = (sub["a_A"] - p["a_A"]) / p["a_A"]
    ezz = -2.0 * p["C13"] / p["C33"] * exx
    return exx, eyy, ezz


def hamiltonian(kx, ky, kz, strain=None, p=GAN):
    """6x6 Hamiltonian in eV; k in 1/nm.

    Hole energies are returned by the caller as the NEGATIVE of the eigenvalues
    of this matrix, since the convention here is the electron valence-band
    energy measured upward.
    """
    A = [p[f"A{i}"] for i in range(1, 7)]
    d1, d2 = p["D_CR"], p["D_SO"] / 3.0
    d3 = d2
    kt2 = kx * kx + ky * ky
    ktp = kx + 1j * ky

    lam_e = th_e = 0.0
    if strain is not None:
        exx, eyy, ezz = strain
        lam_e = p["D1v"] * ezz + p["D2v"] * (exx + eyy)
        th_e = p["D3v"] * ezz + p["D4v"] * (exx + eyy)

    lam = HB2_2M0_eVnm2 * (A[0] * kz * kz + A[1] * kt2) + lam_e
    th = HB2_2M0_eVnm2 * (A[2] * kz * kz + A[3] * kt2) + th_e
    # A7, the linear-in-k term, is taken as zero, which is the standard
    # treatment and makes H and I equal.  The shear deformation terms D5 and
    # D6 vanish for the biaxial strain considered here.
    K = HB2_2M0_eVnm2 * A[4] * ktp * ktp
    Hh = 1j * HB2_2M0_eVnm2 * A[5] * kz * ktp
    Ii = Hh
    D = np.sqrt(2.0) * d3

    F = d1 + d2 + lam + th
    G = d1 - d2 + lam + th

    # Block form as written out in arXiv:2607.03753, Sec. II, which reproduces
    # the Hamiltonian of Chuang and Chang.  Hermiticity is asserted below.
    M = np.array([
        [F,            0.0,          -np.conj(Hh), 0.0,          np.conj(K),   0.0],
        [0.0,          G,             D,          -np.conj(Hh),  0.0,          np.conj(K)],
        [-Hh,          D,             lam,         0.0,          np.conj(Ii),  0.0],
        [0.0,         -Hh,            0.0,         lam,          D,            np.conj(Ii)],
        [K,            0.0,           Ii,          D,            G,            0.0],
        [0.0,          K,             0.0,         Ii,           0.0,          F],
    ], dtype=complex)
    assert np.allclose(M, M.conj().T, atol=1e-12), "Hamiltonian not Hermitian"
    return M


def bands(kt, phi=0.0, kz=0.0, strain=None, p=GAN):
    """Hole energies (eV, measured downward from the valence band maximum
    at k = 0) for the six states, sorted with the topmost band first."""
    kx, ky = kt * np.cos(phi), kt * np.sin(phi)
    e = np.linalg.eigvalsh(hamiltonian(kx, ky, kz, strain, p))
    return np.sort(e)[::-1]


def dispersion(kt_grid, phi=0.0, kz=0.0, strain=None, p=GAN, n_bands=6):
    """Hole energy of each band on a grid of in-plane wavevector.

    Returns an array of shape (len(kt_grid), n_bands), zeroed at the valence
    band maximum and positive downward, which is the natural hole convention.

    Branches are followed by eigenvector continuity rather than by sorting on
    energy.  Sorting fails wherever two branches cross, which happens in this
    material because the light branch rises quickly and passes the crystal
    field split-off band within the range of wavevector occupied by the gas.
    """
    top = bands(0.0, phi, kz, strain, p)[0]
    out = np.empty((len(kt_grid), n_bands))
    prev_vec = None
    for i, kt in enumerate(kt_grid):
        kx, ky = kt * np.cos(phi), kt * np.sin(phi)
        w, v = np.linalg.eigh(hamiltonian(kx, ky, kz, strain, p))
        order = np.argsort(w)[::-1]
        w, v = w[order], v[:, order]
        if prev_vec is not None:
            # Optimal assignment on the overlap matrix.  A greedy match is not
            # sufficient: the light branch crosses the crystal field split-off
            # branch within the occupied range of wavevector, and a greedy rule
            # swaps the two there.
            from scipy.optimize import linear_sum_assignment
            ov = np.abs(prev_vec.conj().T @ v) ** 2
            _, perm = linear_sum_assignment(-ov)
            w, v = w[perm], v[:, perm]
        prev_vec = v
        out[i] = (top - w)[:n_bands]
    return out


def subband_kz(width_nm):
    """Effective out-of-plane wavevector for the lowest state of a well.

    The polarisation-induced gas is not in a square well, so this is used only
    to show how sensitive the in-plane masses are to confinement, never as a
    quantitative model of the real triangular potential.
    """
    return np.pi / width_nm


def cyclotron_mass(kt_grid, E, band_index):
    """m_CR(k) = hbar^2 k (dE/dk)^-1, in units of m0.

    This is the definition used by Chang et al., and it is the mass that both
    the density of states and the Lifshitz-Kosevich thermal damping respond to:
    DOS(E) = m_CR(E) / (pi hbar^2).
    """
    e = E[:, band_index]
    dEdk = np.gradient(e, kt_grid)                    # eV nm
    with np.errstate(divide="ignore", invalid="ignore"):
        m = HB2_2M0_eVnm2 * 2.0 * kt_grid / dEdk
    return m


def curvature_mass(kt_grid, E, band_index):
    """m_curv(k) = hbar^2 (d2E/dk2)^-1, in units of m0."""
    e = E[:, band_index]
    d2 = np.gradient(np.gradient(e, kt_grid), kt_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * HB2_2M0_eVnm2 / d2


def kf_from_density(n_s_cm2, spin_degenerate=True):
    """Fermi wavevector in 1/nm for a two-dimensional band."""
    n_nm2 = n_s_cm2 * 1.0e-14
    return np.sqrt(2.0 * np.pi * n_nm2) if spin_degenerate \
        else np.sqrt(4.0 * np.pi * n_nm2)
