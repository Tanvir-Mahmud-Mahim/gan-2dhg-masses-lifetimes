"""Shared tools: fit disorder models to the four measured mobilities.

The rates of the mechanisms add; each mechanism has one scanned shape
parameter and one fitted amplitude.  The coupled two-subband Boltzmann
equation with the computed Bloch overlap gives the Hall (transport) and quantum
mobilities of both subbands (as in run_tension.py and run_mixtures.py).  A fit
minimizes the logarithmic misfit to (Hall L, Hall H, quantum L, quantum H);
"worst factor" is the largest ratio between a fitted and a measured mobility.
"""

import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import run_mixtures as RM                                  # noqa: E402
import run_tension as RT                                   # noqa: E402
from gan2dhg import measured as MS, scatter2d as S, sdh    # noqa: E402
from gan2dhg.constants import Q                            # noqa: E402

OV_FN, BANDS, B_FH, F_EFF, V, M_KG = RM.setup()
FAM = RT.families(B_FH, F_EFF, fine=True)
A_GAN = 3.189e-10                     # m, in-plane lattice constant of GaN
MISMATCH = 0.0242                     # compressive mismatch of GaN on AlN
MG_RANGE_NM = (10.0, 15.0)            # Methods of Chang et al.


def matrices(w):
    return S.coupling_matrices(BANDS, w, RT.EPS_R, B_FH, overlap_fn=OV_FN)


def solve(A, Bk):
    """(Hall, quantum) mobilities in cm^2/Vs for rate matrices A, Bk."""
    tau = sdh.coupled_tau(A, Bk, V)
    tq = 1.0 / A.sum(1)
    return Q * tau / M_KG * 1e4, Q * tq / M_KG * 1e4


def measured(mu_q_H):
    return np.array([MS.MU_HALL_L, MS.MU_HALL_H, MS.MU_Q_L, mu_q_H])


def library(roughness_max_nm=3.0):
    """Unit-amplitude matrices of every mechanism, keyed by name."""
    lib = {}
    lib["interface roughness"] = [(float(L), matrices(FAM["interface roughness"][2](L)))
                                  for L in np.linspace(0.2, 12, 60)]
    lib["remote ionised charge"] = [(float(d), matrices(FAM["remote ionised charge"][2](d)))
                                    for d in np.linspace(0.2, 30, 60)]
    lib["charged dislocations"] = [(1.0, matrices(FAM["charged dislocations"][2](1.0)))]
    lib["background impurities"] = [(1.0, matrices(FAM["background impurities"][2](1.0)))]
    ds = np.linspace(*MG_RANGE_NM, 12)
    mg = [matrices(lambda q, d=d: S.w_remote_impurity(q, 5e12 / len(ds), d * 1e-9, B_FH,
                                                        RT.EPS_R)) for d in ds]
    lib["Mg-doped layer"] = [(0.0, (sum(m[0] for m in mg), sum(m[1] for m in mg)))]
    lib["polarization fluctuations"] = [
        (float(xi), matrices(lambda q, xi=xi: S.w_polarization_fluctuation(
            q, 1e11, xi * 1e-9, B_FH, RT.EPS_R))) for xi in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)]
    lib["misfit lines"] = [(1.0, matrices(lambda q: S.w_misfit_lines(
        q, 1e5, 1.0, A_GAN, RT.EPS_R, B_FH)))]
    return lib


def power_library(ps):
    return [(float(round(p, 3)), matrices(lambda q, p=p: S.w_power_law(q, p, B_FH))) for p in ps]


# physical meaning of a fitted amplitude a (unit-amplitude libraries above)
def physical(name, a):
    if name == "interface roughness":
        return "rms height (nm)", 0.3 * np.sqrt(a)
    if name == "remote ionised charge":
        return "sheet density (cm^-2)", 5e12 * a
    if name == "charged dislocations":
        return "N f^2 (cm^-2)", 1e4 * a
    if name == "background impurities":
        return "volume density (cm^-3)", 1e17 * a
    if name == "Mg-doped layer":
        return "ionized acceptors (cm^-2)", 5e12 * a
    if name == "polarization fluctuations":
        return "rms interface charge (cm^-2)", 1e11 * np.sqrt(a)
    if name == "misfit lines":
        return "line length per area for f = 1 (cm^-1)", 1e5 * a
    return "amplitude", a


def relaxation_from_misfit(L_A_cm1):
    """Relaxed fraction of the mismatch for a hexagonal misfit network.

    Three sets of lines at 60 degrees, each relieving its normal strain b per
    spacing: relieved strain = L_A b / 2 with b = a (in-plane lattice constant).
    """
    return L_A_cm1 * 100.0 * A_GAN / (2.0 * MISMATCH)


def fit(entries, meas, starts=None):
    """Best amplitudes over a list of mechanism combinations.

    entries: list of combinations, each a list of (parameter, (A, Bk)).
    """
    best = None
    for combo in entries:
        mats = [c[1] for c in combo]

        def res(x):
            a = np.exp(x)
            A = sum(ai * m[0] for ai, m in zip(a, mats))
            Bk = sum(ai * m[1] for ai, m in zip(a, mats))
            mt, mq = solve(A, Bk)
            return np.log(np.r_[mt, mq] / meas)
        st = starts or ([(0,), (2,), (-2,), (4,), (-4,)] if len(mats) == 1 else
                        [(0, 0), (2, -2), (-2, 2), (4, 0), (0, 4)] if len(mats) == 2 else
                        [(-2.6, -2, 13), (-2.6, 3, 13.5), (-2.6, 4, 12), (-2.6, -2, -2),
                         (-2.6, 5, -2)])
        s_ = min((least_squares(res, np.array(x0, float)) for x0 in st), key=lambda z: z.cost)
        if best is None or s_.cost < best[0].cost:
            best = (s_, combo)
    s_, combo = best
    a = np.exp(s_.x)
    A = sum(ai * c[1][0] for ai, c in zip(a, combo))
    Bk = sum(ai * c[1][1] for ai, c in zip(a, combo))
    mt, mq = solve(A, Bk)
    return {"worst_factor": float(np.exp(np.max(np.abs(s_.fun)))),
            "parameters": [c[0] for c in combo], "amplitudes": a.tolist(),
            "mu_tr": mt.tolist(), "mu_q": mq.tolist(),
            "shares_of_quantum_rate": [((ai * c[1][0]).sum(1) / A.sum(1)).tolist()
                                       for ai, c in zip(a, combo)],
            "W": sdh.dos_weights(A, Bk, V).tolist(), "_A": A, "_B": Bk}


def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}
