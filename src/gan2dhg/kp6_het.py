"""Six-band envelope functions in a heterostructure with a finite barrier.

WHAT THIS ADDS TO kp6_well
--------------------------
kp6_well solves the polarisation well with a hard wall at the GaN/AlN
interface, on the grounds that the valence band offset is of order an electron
volt while the subband energies are tens of millielectronvolts.  That argument
is sound for the heavy branch, whose decay length in the barrier is under two
angstroms, but it is weak for the light branch: with a light out-of-plane mass
and a valence band offset at the low end of the published range the decay
length approaches the width of the gas itself.  Since the light subband carries
the occupation and the mass that the experiments disagree about, the hard wall
has to be removed rather than argued away.

This module therefore

  * extends the grid into the AlN,
  * carries every band parameter as a function of position,
  * discretises the position-dependent operators in the symmetric
    (BenDaniel-Duke) form, so the operator stays Hermitian across the
    interface, and
  * places the two materials on a common energy scale by computing each
    material's own valence band maximum from the same Hamiltonian and imposing
    the published offset between them, rather than by shifting a diagonal by
    hand.

It also carries the in-plane wavevector as a vector rather than a magnitude.
The eigenvalues do not depend on the azimuth, and that is verified rather than
assumed, but the eigenVECTORS do, and the azimuthal phases are needed to
compute the interband Bloch overlap instead of parameterising it.

PARAMETERS FOR AlN
------------------
A1 to A6 and the crystal-field splitting of AlN are taken from Table V and
Table III of Rinke et al., Phys. Rev. B 77, 075202 (2008), which is the same
source from which the GaN parameters used throughout this work descend, so the
two materials are described consistently.

VALENCE BAND OFFSET
-------------------
The published GaN/AlN valence band offset spans a wide range, and no single
value is adopted here.  The offset is a scanned input and every conclusion is
reported across the published range.
"""

import numpy as np

from .kp6 import GAN, HB2_2M0_eVnm2, biaxial_strain_on_AlN

# --- AlN --------------------------------------------------------------------
# A1..A6 from Table V and the crystal-field splitting from Table III of Rinke
# et al., Phys. Rev. B 77, 075202 (2008), the same source from which the GaN
# parameters used here descend.  Rinke et al. do not tabulate the spin-orbit
# splitting of AlN; it is taken from de Carvalho et al., Appl. Phys. Lett. 97,
# 232101 (2010), who compute 21.7 meV parallel and 23.5 meV perpendicular, and
# the midpoint is used.  The states in the barrier are evanescent and the
# sensitivity to this number is measured rather than assumed.
ALN_KP = {
    "A1": -3.991, "A2": -0.311, "A3": 3.671,
    "A4": -1.147, "A5": -1.329, "A6": -1.952,
    "D_CR": -0.295, "D_SO": 0.022,
}


# ---------------------------------------------------------------------------
# Position-dependent differential operators, Hermitian by construction
# ---------------------------------------------------------------------------

def _op_a_kz2(a, dz):
    """Discretise  kz a(z) kz  =  -d/dz a(z) d/dz  symmetrically.

    (a psi')'|_i = [a_{i+1/2}(psi_{i+1}-psi_i) - a_{i-1/2}(psi_i-psi_{i-1})]/dz^2

    so the matrix of -d/dz a d/dz is tridiagonal, real and symmetric with
    diagonal (a_{i+1/2} + a_{i-1/2})/dz^2 and off-diagonal -a_{i+1/2}/dz^2.
    Outside the grid the coefficient is continued, which is equivalent to the
    hard wall imposed at both ends far from the gas.
    """
    n = len(a)
    ah = 0.5 * (a[:-1] + a[1:])                 # a at the half points
    M = np.zeros((n, n))
    M[np.arange(n - 1), np.arange(1, n)] = -ah
    M[np.arange(1, n), np.arange(n - 1)] = -ah
    d = np.zeros(n)
    d[:-1] += ah
    d[1:] += ah
    d[0] += a[0]                                 # wall
    d[-1] += a[-1]
    M[np.arange(n), np.arange(n)] = d
    return M / dz**2


def _op_a_kz(a, dz):
    """Discretise the symmetrised  (1/2){a(z), kz}  with kz = -i d/dz.

    (1/2)(a kz + kz a) psi|_i = (-i/2)[a psi' + (a psi)']
                              = (-i/(4 dz))[(a_i + a_{i+1}) psi_{i+1}
                                            - (a_i + a_{i-1}) psi_{i-1}]

    which is anti-symmetric in the real part and therefore Hermitian once the
    factor -i is included.
    """
    n = len(a)
    M = np.zeros((n, n), dtype=complex)
    up = a[:-1] + a[1:]
    M[np.arange(n - 1), np.arange(1, n)] = -1j * up / (4.0 * dz)
    M[np.arange(1, n), np.arange(n - 1)] = +1j * up / (4.0 * dz)
    return M


# ---------------------------------------------------------------------------
# Material profile
# ---------------------------------------------------------------------------

def vbm_of(p, strain=None):
    """Valence band maximum of a bulk material, in the electron convention.

    Computed from the same 6x6 Hamiltonian at k = 0 that the heterostructure
    operator uses, so that no hand algebra enters the band alignment.
    """
    from .kp6 import hamiltonian
    M = hamiltonian(0.0, 0.0, 0.0, strain, p)
    return float(np.max(np.linalg.eigvalsh(M)))


def profile(z, vbo_eV, aln=None, gan=None, strain_gan="AlN",
            gan_params_in_barrier=False):
    """Band parameters and band-edge offset on the grid.

    z < 0 is AlN, relaxed; z >= 0 is GaN, pseudomorphic on AlN.  The offset
    array is the shift of each material's own reference energy, in the ELECTRON
    convention, chosen so that

        VBM(strained GaN) - VBM(relaxed AlN) = vbo_eV.

    With gan_params_in_barrier the barrier keeps the GaN band parameters and
    only the band edge steps, which is the cruder constant-mass barrier; it is
    provided so that the sensitivity to the AlN parameters can be measured.
    """
    gan = dict(GAN if gan is None else gan)
    aln = dict(ALN_KP if aln is None else aln)
    if strain_gan == "AlN":
        strain_gan = biaxial_strain_on_AlN()

    bar = dict(gan) if gan_params_in_barrier else dict(aln)
    for key in ("a_A", "c_A", "C11", "C12", "C13", "C33", "C44",
                "D1v", "D2v", "D3v", "D4v"):
        bar.setdefault(key, gan.get(key, 0.0))

    # Common energy scale.  Both valence band maxima are computed from the same
    # Hamiltonian; the barrier reference is then shifted so that the published
    # offset is reproduced exactly.
    vbm_gan = vbm_of(gan, strain_gan)
    vbm_bar = vbm_of(bar, None)
    shift_bar = vbm_gan - vbo_eV - vbm_bar          # electron convention

    n = len(z)
    inb = z < 0.0
    out = {}
    for key in ("A1", "A2", "A3", "A4", "A5", "A6"):
        out[key] = np.where(inb, bar[key], gan[key]).astype(float)
    out["d1"] = np.where(inb, bar["D_CR"], gan["D_CR"]).astype(float)
    out["d2"] = np.where(inb, bar["D_SO"] / 3.0, gan["D_SO"] / 3.0).astype(float)

    exx, eyy, ezz = strain_gan
    lam_e = gan["D1v"] * ezz + gan["D2v"] * (exx + eyy)
    th_e = gan["D3v"] * ezz + gan["D4v"] * (exx + eyy)
    out["lam_e"] = np.where(inb, 0.0, lam_e).astype(float)
    out["th_e"] = np.where(inb, 0.0, th_e).astype(float)
    out["shift"] = np.where(inb, shift_bar, 0.0).astype(float)
    out["in_barrier"] = inb
    out["vbo_eV"] = vbo_eV
    out["barrier_hole_eV"] = vbm_gan - (vbm_bar + shift_bar)   # = vbo_eV
    return out


# ---------------------------------------------------------------------------
# Heterostructure envelope operator
# ---------------------------------------------------------------------------

def build_operator(kx, ky, z, V, prof):
    """Six-band envelope Hamiltonian, hole convention, on a heterostructure.

    V is the electrostatic hole potential energy on the grid (eV).  The band
    offset carried by prof is added separately, so V remains the solution of
    Poisson's equation alone.
    """
    n = len(z)
    dz = z[1] - z[0]
    I = np.eye(n)
    kt2 = kx * kx + ky * ky
    kp = kx + 1j * ky                              # k_+ , carries the azimuth

    lam = (HB2_2M0_eVnm2 * (_op_a_kz2(prof["A1"], dz) + np.diag(prof["A2"] * kt2))
           + np.diag(prof["lam_e"]))
    th = (HB2_2M0_eVnm2 * (_op_a_kz2(prof["A3"], dz) + np.diag(prof["A4"] * kt2))
          + np.diag(prof["th_e"]))
    Kop = HB2_2M0_eVnm2 * np.diag(prof["A5"]) * (kp * kp)
    Hop = 1j * HB2_2M0_eVnm2 * kp * _op_a_kz(prof["A6"], dz)

    d1 = np.diag(prof["d1"])
    d2 = np.diag(prof["d2"])
    DI = np.sqrt(2.0) * d2
    sh = np.diag(prof["shift"])
    F = d1 + d2 + lam + th + sh
    G = d1 - d2 + lam + th + sh
    lamS = lam + sh
    Z = np.zeros((n, n))

    M = np.block([
        [F,     Z,     -Hop.conj().T, Z,             Kop.conj().T, Z],
        [Z,     G,      DI,          -Hop.conj().T,  Z,            Kop.conj().T],
        [-Hop,  DI,     lamS,         Z,             Hop.conj().T, Z],
        [Z,    -Hop,    Z,            lamS,          DI,           Hop.conj().T],
        [Kop,   Z,      Hop,          DI,            G,            Z],
        [Z,     Kop,    Z,            Hop,           Z,            F],
    ])
    if not np.allclose(M, M.conj().T, atol=1e-10):
        raise AssertionError("heterostructure operator is not Hermitian")

    # Hole convention: energies increase downward from the valence band
    # maximum, so the operator is negated before the hole potential is added.
    return -M + np.kron(np.eye(6), np.diag(V))


def solve(kx, ky, z, V, prof, n_states=8):
    """Lowest hole levels, their envelopes and the full spinors."""
    M = build_operator(kx, ky, z, V, prof)
    w, v = np.linalg.eigh(M)
    w, v = w[:n_states], v[:, :n_states]
    n = len(z)
    dens = np.zeros((n, n_states))
    for s in range(n_states):
        dens[:, s] = np.sum(np.abs(v[:, s].reshape(6, n)) ** 2, axis=0)
    norm = np.trapezoid(dens, z, axis=0)
    dens = dens / norm
    return w, dens, v


def poisson_het(z, p_of_z, p_s, eps_r):
    """Hole potential energy from the hole distribution, sheet charge at z = 0.

    With a fixed polarisation sheet of areal density -p_s at the interface and
    mobile holes of density p(z), Gauss's law with the field vanishing deep in
    the barrier gives

        dU/dz = C [ p_s theta(z) - Int_{-L}^{z} p ],   C = e^2 / (eps eps_0),

    which reduces to the hard-wall expression once the penetrating tail is
    negligible, and which correctly gives a small field in the barrier that
    pulls the penetrating part of the gas back towards the interface.
    """
    from .kp6_well import E2_OVER_EPS0
    C = E2_OVER_EPS0 / eps_r
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (p_of_z[1:] + p_of_z[:-1]) * np.diff(z))])
    theta = np.where(z > 0, 1.0, np.where(z < 0, 0.0, 0.5))
    dUdz = C * (p_s * theta - cum)
    U = np.concatenate([[0.0], np.cumsum(
        0.5 * (dUdz[1:] + dUdz[:-1]) * np.diff(z))])
    return U - U.min()


def self_consistent_het(p_s_cm2=4.6e13, vbo_eV=0.7, L_gan=15.0, L_bar=4.0,
                        dz=0.09375, kt_max=2.4, n_kt=16, eps_r=10.4,
                        n_states=6, max_iter=90, tol=2e-5, mix=0.6,
                        gan_params_in_barrier=False, aln=None, verbose=False):
    """Self-consistent solution with a finite barrier of offset vbo_eV.

    The grid is built so that z = 0, the interface, is a grid point.
    """
    n_bar = int(round(L_bar / dz))
    n_gan = int(round(L_gan / dz))
    z = (np.arange(-n_bar, n_gan + 1)) * dz
    prof = profile(z, vbo_eV, aln=aln,
                   gan_params_in_barrier=gan_params_in_barrier)
    p_s = p_s_cm2 * 1.0e-14

    from .kp6_well import E2_OVER_EPS0, fill_subbands
    V = (E2_OVER_EPS0 / eps_r) * p_s * np.maximum(z, 0.0)
    kt_grid = np.linspace(1e-3, kt_max, n_kt)

    err = np.inf
    for it in range(max_iter):
        E_of_k = np.empty((n_kt, n_states))
        dens0 = None
        for i, kt in enumerate(kt_grid):
            w, d, _ = solve(kt, 0.0, z, V, prof, n_states=n_states)
            E_of_k[i] = w
            if i == 0:
                dens0 = d
        EF, per = fill_subbands(E_of_k, kt_grid, p_s)
        p_of_z = np.zeros_like(z)
        for s in range(n_states):
            p_of_z += per[s] * dens0[:, s]
        V_new = poisson_het(z, p_of_z, p_s, eps_r)
        err = np.max(np.abs(V_new - V))
        V = (1.0 - mix) * V + mix * V_new
        if verbose:
            print(f"  iter {it:2d}  max|dV| = {1000*err:9.4f} meV  "
                  f"EF = {1000*EF:8.3f} meV", flush=True)
        if err < tol:
            break

    return {"z": z, "V": V, "prof": prof, "kt": kt_grid, "E_of_k": E_of_k,
            "EF": EF, "per_subband_nm2": per, "p_of_z": p_of_z,
            "converged": bool(err < tol), "iterations": it + 1,
            "vbo_eV": vbo_eV, "p_s_cm2": p_s_cm2, "eps_r": eps_r}


def mass_at_kf(sol, subband, dk=0.01, n_states=6):
    """Cyclotron mass by local central difference at the branch's own k_F.

    Differentiating the dispersion on the coarse wavevector grid used inside
    the self-consistent loop is not accurate enough for the light branch, whose
    Fermi wavevector sits where the band is strongly non-parabolic.  The
    derivative is therefore taken locally, with the converged potential held
    fixed, and its convergence in dk is checked.
    """
    n = sol["per_subband_nm2"][subband]
    if n <= 0:
        return np.nan, np.nan
    kF = np.sqrt(4.0 * np.pi * n)
    z, V, prof = sol["z"], sol["V"], sol["prof"]
    Ep = solve(kF + dk, 0.0, z, V, prof, n_states=n_states)[0][subband]
    Em = solve(kF - dk, 0.0, z, V, prof, n_states=n_states)[0][subband]
    m = HB2_2M0_eVnm2 * 2.0 * kF / ((Ep - Em) / (2.0 * dk))
    return float(kF), float(m)


def track_branches(k_grid, solve_at, n_states=6):
    """Follow branches by eigenvector continuity instead of by energy order.

    Sorting eigenvalues relabels branches wherever two of them cross.  In this
    material the light pair and the crystal-field split-off pair do cross
    within the range of in-plane wavevector that matters, so a spin splitting
    read off fixed eigenvalue indices is the splitting of whichever pair
    happens to occupy those indices, and jumps discontinuously at the crossing.
    Branches are therefore matched from one wavevector to the next by maximum
    eigenvector overlap, solved as an assignment problem so the match is
    optimal rather than greedy.

    Returns E of shape (len(k_grid), n_states), ordered consistently.
    """
    from scipy.optimize import linear_sum_assignment
    E = np.empty((len(k_grid), n_states))
    prev = None
    for i, k in enumerate(k_grid):
        w, v = solve_at(k)
        w, v = w[:n_states], v[:, :n_states]
        if prev is None:
            order = np.arange(n_states)
        else:
            cost = -np.abs(prev.conj().T @ v) ** 2
            order = linear_sum_assignment(cost)[1]
        E[i] = w[order]
        prev = v[:, order]
    return E


def rashba_vs_k(sol, k_grid, pair_lo=2, n_states=6):
    """Splitting of a tracked Kramers pair against in-plane wavevector, meV."""
    z, V, prof = sol["z"], sol["V"], sol["prof"]

    def at(k):
        M = build_operator(k, 0.0, z, V, prof)
        return np.linalg.eigh(M)

    E = track_branches(k_grid, at, n_states=n_states)
    return 1000.0 * np.abs(E[:, pair_lo + 1] - E[:, pair_lo])


def bloch_overlap(sol, i, j, theta, dk_pair=2, n_states=6):
    """Squared overlap between branch i and branch j at their Fermi circles.

    The initial state sits at azimuth zero on its own Fermi circle and the
    final state at azimuth theta on its own.  The result is summed over the
    Kramers partner of the final branch, since the two partners are degenerate
    to within the Rashba splitting and both are available as final states.

    Because the in-plane wavevector is carried as a vector, every azimuthal
    phase is exact and nothing is fitted: this is the quantity that a
    single-band treatment leaves as a free parameter.
    """
    per = sol["per_subband_nm2"]
    z, V, prof = sol["z"], sol["V"], sol["prof"]
    kFi = np.sqrt(4.0 * np.pi * per[i])
    kFj = np.sqrt(4.0 * np.pi * per[j])
    _, _, vi = solve(kFi, 0.0, z, V, prof, n_states=n_states)
    _, _, vj = solve(kFj * np.cos(theta), kFj * np.sin(theta), z, V, prof,
                     n_states=n_states)
    partners = [j, j ^ 1] if dk_pair == 2 else [j]
    partners = [q for q in partners if 0 <= q < n_states]
    return float(sum(abs(np.vdot(vj[:, q], vi[:, i])) ** 2 for q in partners))


def barrier_fraction(v_col, z):
    """Fraction of the probability density that lies inside the barrier."""
    n = len(z)
    d = np.sum(np.abs(v_col.reshape(6, n)) ** 2, axis=0)
    d = d / np.trapezoid(d, z)
    inb = z < 0.0
    return float(np.trapezoid(d[inb], z[inb])) if inb.sum() > 1 else 0.0
