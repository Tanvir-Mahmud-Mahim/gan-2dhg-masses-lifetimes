"""Landau levels of the six-band envelope-function Hamiltonian.

A magnetic field B along the growth axis (the c axis) quantises the in-plane
motion.  With the kinetic wavevector k = (p + eA)/hbar the in-plane components
no longer commute, [k_x, k_y] = -i / l^2 with l^2 = hbar / (e B), and the
combinations k_+ = k_x + i k_y and k_- = k_x - i k_y become ladder operators:

    k_+ = (sqrt 2 / l) a^dagger,     k_- = (sqrt 2 / l) a,     [a, a^dagger] = 1.

The Hamiltonian of Chuang and Chang is axially symmetric at this order, so
every term changes the Landau index of the component it acts on by a fixed
amount.  Reading the offsets off the matrix used in kp6 and kp6_het, the six
spinor components of one eigenstate carry Landau indices

    (n, n + 1, n + 1, n + 2, n + 2, n + 3)

for an integer n >= -3, components with a negative index being absent.  The
problem therefore separates into independent blocks labelled by n, each of
dimension at most 6 N_z, and within a block the operator is obtained from the
zero-field one by the replacements

    k_perp^2            -> (2 / l^2)(m + 1/2)            on a component of index m
    k_+  (index m -> m+1) -> (sqrt 2 / l) sqrt(m + 1)
    k_+^2 (index m -> m+2) -> (2 / l^2) sqrt((m + 1)(m + 2)).

The in-plane kinetic term A_2 k_perp^2 is taken in the symmetrised order
(k_+ k_- + k_- k_+)/2, which is the only choice that adds no magnetic moment
of its own.

SPIN AND ORBITAL ZEEMAN TERMS
-----------------------------
The components carry total angular momentum J_z = 3/2, 1/2, 1/2, -1/2, -1/2,
-3/2, which identifies them as

    c0 = |X+iY, up>   c1 = |X+iY, down>   c2 = |Z, up>
    c3 = |Z, down>    c4 = |X-iY, up>     c5 = |X-iY, down>,

with orbital angular momentum L_z = +1, +1, 0, 0, -1, -1.  The free-electron
spin term g0 mu_B B S_z, with g0 = 2, is included.  The magnetic moment that a
valence band acquires from remote bands (the parameter kappa of Luttinger's
cubic Hamiltonian) has no counterpart in the parameter set A_1 to A_6 that is
used throughout this work, and no measured value exists for GaN holes.  It is
therefore represented by a term kappa_L mu_B B L_z whose coefficient is an
explicit, scanned input and never a fitted one.

SIGN OF B
---------
For B < 0 the roles of a and a^dagger are exchanged, the Landau offsets change
sign and the Zeeman terms change sign.  Time reversal requires the two spectra
to coincide; that equality is used as a test of the index bookkeeping.
"""

import numpy as np
from scipy.linalg import eigh

from .kp6 import HB2_2M0_eVnm2
from .kp6_het import _op_a_kz, _op_a_kz2

HBAR_OVER_E = 6.582119569e-16      # V s, so l^2 = HBAR_OVER_E / B in m^2
MU_B = 5.7883818060e-5             # eV / T
OFFSETS = np.array([0, 1, 1, 2, 2, 3])
L_Z = np.array([1, 1, 0, 0, -1, -1])
S_Z = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])


def magnetic_length_nm(B):
    """l = sqrt(hbar / e|B|) in nm."""
    return np.sqrt(HBAR_OVER_E / abs(B)) * 1e9


def block_operator(n, B, z, V, prof, kappa_L=0.0, g0=2.0):
    """Hole-convention operator of Landau block n at field B (tesla).

    Returns (M, comps) where comps lists the spinor components present in the
    block, in order, each occupying len(z) consecutive rows.
    """
    s = 1 if B > 0 else -1
    idx = n + s * OFFSETS
    comps = [c for c in range(6) if idx[c] >= 0]
    if not comps:
        return None, comps
    Nz = len(z)
    dz = z[1] - z[0]
    l2 = magnetic_length_nm(B) ** 2
    c = HB2_2M0_eVnm2

    def kperp2(m):
        return (2.0 / l2) * (m + 0.5)

    def kplus(m_from):
        # matrix element of k_+ from index m_from to m_from + s:
        # a^dagger gives sqrt(m + 1) for B > 0, a gives sqrt(m) for B < 0
        m = max(m_from, 0)
        return np.sqrt(2.0 / l2) * (np.sqrt(m + 1.0) if s > 0 else np.sqrt(m))

    def kplus2(m_from):
        if s > 0:
            m = max(m_from, 0)
            return (2.0 / l2) * np.sqrt((m + 1.0) * (m + 2.0))
        m = max(m_from, 1)
        return (2.0 / l2) * np.sqrt(m * (m - 1.0))

    Kz2_A1 = _op_a_kz2(prof["A1"], dz)
    Kz2_A3 = _op_a_kz2(prof["A3"], dz)
    Kz_A6 = _op_a_kz(prof["A6"], dz)
    d1 = np.diag(prof["d1"])
    d2 = np.diag(prof["d2"])
    DI = np.sqrt(2.0) * d2
    sh = np.diag(prof["shift"])

    def lam(m):
        return c * (Kz2_A1 + np.diag(prof["A2"]) * kperp2(m)) + np.diag(prof["lam_e"])

    def th(m):
        return c * (Kz2_A3 + np.diag(prof["A4"]) * kperp2(m)) + np.diag(prof["th_e"])

    diag = {}
    for q in range(6):
        m = idx[q]
        if m < 0:
            continue
        if q in (0, 5):
            D = d1 + d2 + lam(m) + th(m) + sh
        elif q in (1, 4):
            D = d1 - d2 + lam(m) + th(m) + sh
        else:
            D = lam(m) + sh
        zee = MU_B * B * (kappa_L * L_Z[q] + g0 * S_Z[q])
        diag[q] = D + zee * np.eye(Nz)

    # Lower-triangle couplings (i <- j), as in kp6_het.build_operator, with the
    # scalar k_+ replaced by its ladder matrix element.
    Hraw = 1j * c * Kz_A6                       # H / k_+
    Kraw = c * np.diag(prof["A5"])              # K / k_+^2
    low = {}

    def add(i, j, mat):
        if idx[i] >= 0 and idx[j] >= 0:
            low[(i, j)] = mat

    add(2, 0, -Hraw * kplus(idx[0]))
    add(3, 1, -Hraw * kplus(idx[1]))
    add(4, 2, Hraw * kplus(idx[2]))
    add(5, 3, Hraw * kplus(idx[3]))
    add(4, 0, Kraw * kplus2(idx[0]))
    add(5, 1, Kraw * kplus2(idx[1]))
    add(2, 1, DI.astype(complex))
    add(4, 3, DI.astype(complex))

    pos = {q: k for k, q in enumerate(comps)}
    Mb = np.zeros((len(comps) * Nz, len(comps) * Nz), dtype=complex)
    for q in comps:
        a = pos[q] * Nz
        Mb[a:a + Nz, a:a + Nz] = diag[q]
    for (i, j), mat in low.items():
        a, b = pos[i] * Nz, pos[j] * Nz
        Mb[a:a + Nz, b:b + Nz] = mat
        Mb[b:b + Nz, a:a + Nz] = mat.conj().T
    if not np.allclose(Mb, Mb.conj().T, atol=1e-10):
        raise AssertionError("Landau block is not Hermitian")
    Vfull = np.concatenate([V] * len(comps))
    return -Mb + np.diag(Vfull), comps


def spectrum(B, z, V, prof, e_max, kappa_L=0.0, g0=2.0, n_max=None,
             return_blocks=False):
    """All Landau levels below the hole energy e_max (eV) at field B.

    Blocks are added in increasing n until a block has no level below e_max;
    the lowest level of successive blocks rises monotonically at large n, so
    this is a complete enumeration once two consecutive blocks are empty.
    """
    levels, labels = [], []
    empty = 0
    n = -3
    while True:
        M, comps = block_operator(n, B, z, V, prof, kappa_L=kappa_L, g0=g0)
        if M is not None:
            w = eigh(M, eigvals_only=True, subset_by_value=(-np.inf, e_max),
                     driver="evr")
            if len(w) == 0:
                empty += 1
            else:
                empty = 0
                levels.extend(w.tolist())
                labels.extend([n] * len(w))
        n += 1
        if (empty >= 2 and n > 0) or (n_max is not None and n > n_max):
            break
    levels = np.array(levels)
    order = np.argsort(levels)
    if return_blocks:
        return levels[order], np.array(labels)[order]
    return levels[order]


def spectrum_with_character(B, z, V, prof, e_max, kappa_L=0.0, g0=2.0,
                            z_bulk=3.0):
    """Levels below e_max with, for each, its block index, the weight on the
    heavy-hole components c0 and c5, and the weight beyond z_bulk (nm).

    The last quantity identifies states that are not bound to the interface:
    in the self-consistent potential the field vanishes beyond the gas, so
    states above the bulk band edge extend across the GaN layer.
    """
    out = []
    n, empty = -3, 0
    far = z > z_bulk
    Nz = len(z)
    while True:
        M, comps = block_operator(n, B, z, V, prof, kappa_L=kappa_L, g0=g0)
        if M is not None:
            w, v = eigh(M, subset_by_value=(-np.inf, e_max), driver="evr")
            if len(w) == 0:
                empty += 1
            else:
                empty = 0
                for k in range(len(w)):
                    comp = v[:, k].reshape(len(comps), Nz)
                    dens = np.abs(comp) ** 2
                    hh = sum(dens[i].sum() for i, q in enumerate(comps)
                             if q in (0, 5))
                    out.append((w[k], n, hh / dens.sum(),
                                dens[:, far].sum() / dens.sum()))
        n += 1
        if empty >= 2 and n > 0:
            break
    out.sort()
    return np.array(out)


def degeneracy_per_nm2(B):
    """Number of states per Landau level per nm^2, e|B|/h."""
    return 1.0 / (2.0 * np.pi * magnetic_length_nm(B) ** 2)
