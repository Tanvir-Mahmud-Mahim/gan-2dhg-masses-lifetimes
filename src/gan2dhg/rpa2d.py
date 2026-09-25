"""On-shell RPA (GW) quasiparticle mass of a multicomponent two-dimensional gas.

Each component is a parabolic, spin-degenerate band (mass m, sheet density n).
All components screen the Coulomb interaction through their 2D Lindhard
functions (random phase approximation); the self-energy of component a is

    Sigma_a(k, w) = Sigma_x(k) + Sigma_line(k, w) + Sigma_res(k, w)

with the correlation part w = W - V split into a line integral along the
imaginary frequency axis and a residue part at real frequency (the standard
decomposition; see Asgari et al., Phys. Rev. B 71, 045323, 2005).  The
on-shell effective mass is

    m*/m = 1 / (1 + (m / hbar^2 k_F) dSigma(k, xi_k)/dk |_{k_F}).

NUMERICAL TREATMENT.  Sigma_x and Sigma_res separately carry terms
(k - k_F) ln|k - k_F| whose derivatives diverge; they cancel in the sum.  The
sum is rearranged exactly as

    Sigma_x(k) + Sigma_res(k, xi_k) = F(k) + Sigma_res^W(k),
    F(k) = - int_{|p|<k} V(|k-p|) d^2p/(2 pi)^2,

where F is the exchange energy of a Fermi disc of radius k (smooth in k) and
Sigma_res^W carries the full screened W, which is finite at q -> 0; on the
shell between k_F and k the frequency never exceeds v_F q, so W is static
there and d Sigma_res^W / dk = (k_F / (2 pi)^2) int dphi W(2 k_F sin(phi/2), 0).
The line part is smooth and is differentiated numerically.  The code
reproduces the published on-shell RPA masses of the strictly 2D electron gas
(test suite).

Units: energies in meV, lengths in nm.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

HB2M0 = 38.09982   # hbar^2 / (2 m0), meV nm^2
E2 = 1439.9645     # e^2 / (4 pi eps0), meV nm


def gl(a, b, n):
    x, w = leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


class Species:
    """Parabolic, spin-degenerate band: mass m (m0), density n (nm^-2)."""

    def __init__(self, m, n_nm2):
        self.m = m
        self.n = n_nm2
        self.kF = np.sqrt(2 * np.pi * n_nm2)
        self.EF = HB2M0 * self.kF ** 2 / m
        self.N0 = m / (np.pi * 2 * HB2M0)

    def xi(self, p):
        return HB2M0 * p ** 2 / self.m - self.EF


def chi0(sp, q, z):
    """2D Lindhard function (g = 2) at complex frequency z (meV).

    chi0 = -N0 [1 - (s2 - s1)/(2A)] = -N0 [1 - 2z/(s1 + s2)], with
    s_j = c_j (1 - (beta k_F / c_j)^2)^(1/2), c1 = z - A, c2 = z + A,
    A = hbar^2 q^2 / 2m, beta = hbar^2 q / m.  The two forms are identical; the
    one with the larger denominator is used, which avoids cancellation both at
    small q and in the static limit above 2 k_F.
    """
    q = np.asarray(q, float)
    z = np.asarray(z, complex)
    A = HB2M0 * q ** 2 / sp.m
    bk = 2 * HB2M0 * q / sp.m * sp.kF
    c1 = z - A
    c2 = z + A
    s1 = c1 * np.sqrt(1 - (bk / c1) ** 2 + 0j)
    s2 = c2 * np.sqrt(1 - (bk / c2) ** 2 + 0j)
    ssum, sdif = s1 + s2, s2 - s1
    use_sum = np.abs(ssum) > np.abs(sdif)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(use_sum, 2 * z / ssum, sdif / (2 * A))
    return -sp.N0 * (1 - r)


class System:
    """Components sharing one screened interaction.

    eps is the background dielectric constant; ff(q) an optional form factor
    of the interaction (1 for a strictly two-dimensional gas).
    """

    def __init__(self, species, eps, ff=None):
        self.sp = species
        self.eps = eps
        self.ff = ff if ff is not None else (lambda q: np.ones_like(q))

    def V(self, q):
        return 2 * np.pi * E2 / (self.eps * q) * self.ff(q)

    def w(self, q, z):
        """Correlation part W - V of the screened interaction."""
        V = self.V(q)
        chi = sum(chi0(s, q, z) for s in self.sp)
        return V / (1 - V * chi) - V

    def W_static(self, q):
        return (self.w(q, 1e-12j) + self.V(q)).real


def fang_howard_form_factor(b):
    return lambda q: (1 + 9 * q / (8 * b) + 3 * q ** 2 / (8 * b ** 2)) \
        / (1 + q / b) ** 3


def _qgrid(bps, qmax, n_per=48, nlog=40, q0=1e-7):
    bps = sorted(set(b for b in bps if 0 < b < qmax))
    edges = [0.0] + bps + [qmax]
    t, wt = gl(np.log(q0 * edges[1]), np.log(edges[1]), nlog)
    Q, W = [np.exp(t)], [wt * np.exp(t)]
    for a, b in zip(edges[1:-1], edges[2:]):
        mid = 0.5 * (a + b)
        for aa, bb in ((a, mid), (mid, b)):
            x, wx = gl(aa, bb, n_per)
            Q.append(x); W.append(wx)
    return np.concatenate(Q), np.concatenate(W)


def F_moving(system, k, n=400):
    """-int_{|p|<k} V(|k-p|) d^2p / (2 pi)^2."""
    t, wt = gl(0.0, np.pi / 2, n)
    q = 2 * k * np.sin(t)
    dq = 2 * k * np.cos(t) * wt
    L = 2 * np.arccos(np.clip(q / (2 * k), -1, 1))
    return -np.sum(q * system.V(q) * L * dq) / (2 * np.pi) ** 2


def sigma_line(system, a, k, npsi=48, nth=40, qmax_fac=40.0):
    """Line part of the on-shell self-energy of component a at wavevector k."""
    sp = system.sp[a]
    kF = sp.kF
    om = sp.xi(k)
    p_om = np.sqrt(max((om + sp.EF) * sp.m / HB2M0, 0.0))
    bps = [abs(k - kF), k + kF, abs(k - p_om), k + p_om, 2 * k] + \
        [2 * s.kF for s in system.sp]
    q, wq = _qgrid(bps, qmax_fac * max(s.kF for s in system.sp))
    th, wth = gl(0.0, np.pi / 2, nth)
    S = 0.0
    for qi, wqi in zip(q, wq):
        C = (p_om ** 2 - k ** 2 - qi ** 2) / (2 * k * qi)
        cuts = [0.0, np.pi, 2 * np.pi]
        if -1 < C < 1:
            ps = np.arccos(C)
            cuts = sorted([0.0, ps, np.pi, 2 * np.pi - ps, 2 * np.pi])
        tot = 0.0
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            if c1 - c0 < 1e-14:
                continue
            psi, wpsi = gl(c0, c1, npsi)
            pp = np.sqrt(k ** 2 + qi ** 2 + 2 * k * qi * np.cos(psi))
            aa = om - sp.xi(pp)
            nu = np.abs(aa)[:, None] * np.tan(th)[None, :]
            ww = system.w(np.full(nu.shape, qi), 1j * nu).real
            tot += np.sum((np.sign(aa) / np.pi)
                          * np.sum(ww * wth[None, :], axis=1) * wpsi)
        S += -wqi * qi * tot / (2 * np.pi) ** 2
    return S


def mass_ratio(system, a, delta=0.005, nphi=400, parts=False, **kw):
    """On-shell m*/m of component a."""
    sp = system.sp[a]
    kF = sp.kF
    h = 1e-4 * kF
    dF = (F_moving(system, kF + h) - F_moving(system, kF - h)) / (2 * h)
    ph, wph = gl(0.0, np.pi, nphi)
    dres = 2 * kF * np.sum(system.W_static(2 * kF * np.sin(ph / 2)) * wph) \
        / (2 * np.pi) ** 2
    ks = kF * np.array([1 - delta, 1 + delta])
    L = [sigma_line(system, a, k, **kw) for k in ks]
    dline = (L[1] - L[0]) / (ks[1] - ks[0])
    norm = 2 * HB2M0 * kF / sp.m
    d = (dF + dres + dline) / norm
    r = 1.0 / (1.0 + d)
    return (r, (dF / norm, dres / norm, dline / norm)) if parts else r
