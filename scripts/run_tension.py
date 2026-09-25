"""Can any elastic mechanism reproduce the measured lifetime ratios of both subbands?

Every calculation here uses the coupled two-subband Boltzmann equation, with
interband scattering and with the interband Bloch overlap computed from the
six-band spinors (results/barrier.json), inside the angular integrals.

  A. Each mechanism's shape parameter is scanned continuously and the closest
     simultaneous account of the two measured ratios is reported.  The
     amplitude is not scanned because the ratio does not contain it.
  B. The same, with the strength of screening scaled over two decades.
  C. The same, for seven assignments of the two sheet densities.
  D. The coupled equation is solved in a magnetic field, and the parameters
     that a four-parameter two-carrier Drude fit would return are computed
     exactly; the ratio an experiment would then report is compared with the
     measurement.
  E. Trajectories of the two ratios for Fig. 3(c) of the Letter.

THE COUPLED EQUATION IN A FIELD
-------------------------------
For elastic scattering on isotropic Fermi circles the deviation from
equilibrium in subband i is g_i = f0' e v_i k_hat . X_i.  With
X_i = X_ix + i X_iy the linearised Boltzmann equation in crossed E and B
(B along z) reduces exactly to

    sum_j [ M_ij + i omega_ci delta_ij ] X_j = -E,
    M_ij = delta_ij sum_k A_ik - B_ij v_j / v_i,      omega_ci = e B / m_i,

A and B being the kernels weighted by unity and by cos(theta)
(scatter2d.coupling_matrices).  At B = 0 this is the system solved for the
transport lifetimes, X_i = -tau_i E.  With S = diag(e / m_i),

    sigma_xx - i sigma_xy = e^2 sum_ij (n_i / m_i) [(M + i B S)^-1]_ij
                          = sum_c e n_c mu_c / (1 + i mu_c B),

where 1 / mu_c are the eigenvalues of S^-1 M and n_c follow from its
eigenvectors.  The coupled response therefore has EXACTLY the two-carrier
form, for any interband coupling, so a two-carrier fit reproduces it perfectly
and cannot reveal the coupling; what the coupling changes is the meaning of
the densities and mobilities that the fit returns.  The test suite checks
this decomposition against an explicit least-squares fit.

Outputs results/tension.json.  Requires results/barrier.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import measured as MS            # noqa: E402
from gan2dhg import scatter2d as S            # noqa: E402
from gan2dhg.constants import HBAR, M0, Q     # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
EPS_R = 10.4
EPS0 = 8.8541878128e-12


def overlap_function():
    bar = json.load(open(os.path.join(RES, "barrier.json")))
    ov = bar["overlap"]
    th = np.array(ov["theta_rad"])
    tabs = [np.array(ov["light_to_light"]), np.array(ov["light_to_heavy"]),
            np.array(ov["heavy_to_heavy"])]

    def sym(tab):
        def f(t):
            t = np.mod(t, 2 * np.pi)
            t = np.where(t > np.pi, 2 * np.pi - t, t)
            return np.interp(t, th, tab)
        return f
    f_ll, f_lh, f_hh = (sym(t) for t in tabs)

    def ov_fn(i, j, t):
        if i == j:
            return f_ll(t) if i == 0 else f_hh(t)
        return f_lh(t)
    return ov_fn


def coupled_matrix(A, Bk, n_cm2, m_over_m0):
    kF = np.array([S.fermi_wavevector(n) for n in n_cm2])
    v = HBAR * kF / (np.asarray(m_over_m0) * M0)
    return np.diag(A.sum(axis=1)) - Bk * (v[None, :] / v[:, None])


def two_carrier_equivalent(M, n_m2, m_kg):
    """Exact channel densities (m^-2) and mobilities (m^2/Vs).

    Channels are ordered as an experiment labels them: lower density first.
    """
    s = Q / m_kg
    kap, P = np.linalg.eig(M / s[:, None])
    kap, P = kap.real, P.real
    n_c = Q * ((n_m2 / m_kg) @ P) * np.linalg.solve(P, 1.0 / s)
    mu_c = 1.0 / kap
    order = np.argsort(n_c)
    return n_c[order], mu_c[order]


def response(M, n_m2, m_kg, B):
    """Complex sigma_xx - i sigma_xy (S) of the coupled system at field B."""
    Minv = np.linalg.inv(M + 1j * np.diag(Q * B / m_kg))
    return np.sum((Q ** 2 * n_m2 / m_kg)[:, None] * Minv)


def log_miss(rl, rh, targets):
    """Largest |ln| distance of the two ratios from a target point or box."""
    (l_lo, l_hi), (h_lo, h_hi) = targets

    def d(x, lo, hi):
        if not np.isfinite(x) or x <= 0:
            return np.inf
        if lo <= x <= hi:
            return 0.0
        return min(abs(np.log(x / lo)), abs(np.log(x / hi)))
    return max(d(rl, l_lo, l_hi), d(rh, h_lo, h_hi))


def families(b, F_EFF, fine=True):
    g = (lambda a, z, n: np.linspace(a, z, n if fine else max(n // 3, 8)))
    return {
        "interface roughness": (
            "correlation length (nm)", g(0.2, 12.0, 60),
            lambda L: (lambda q: S.w_interface_roughness(
                q, 0.3e-9, L * 1e-9, F_EFF))),
        "remote ionised charge": (
            "standoff (nm)", g(0.2, 30.0, 60),
            lambda d: (lambda q: S.w_remote_impurity(
                q, 5e12, d * 1e-9, b, EPS_R))),
        "charged dislocations": (
            "occupation fraction", g(0.05, 1.0, 20),
            lambda f: (lambda q: S.w_dislocation(
                q, 1e4, 4.982e-10, f, EPS_R, b))),
        "background impurities": (
            "none", np.array([1.0]),
            lambda _: (lambda q: S.w_background_impurity(
                q, 1e17, b, EPS_R))),
    }


def ratios(bands, w, b, ov_fn, eps_scale=None):
    """True ratios and the coupled matrices for one kernel."""
    if eps_scale is None:
        A, Bk = S.coupling_matrices(bands, w, EPS_R, b, overlap_fn=ov_fn)
    else:
        def ws(q, _w=w):
            e = 1.0 + eps_scale * (S.dielectric(q, bands, EPS_R, b) - 1.0)
            return _w(q) / e ** 2
        A, Bk = S.coupling_matrices(bands, ws, EPS_R, b, overlap_fn=ov_fn,
                                    screen=False)
    M = coupled_matrix(A, Bk, [bd["n_s_cm2"] for bd in bands],
                       [bd["m_over_m0"] for bd in bands])
    tau = np.linalg.solve(M, np.ones(len(bands)))
    tau_q = 1.0 / A.sum(axis=1)
    return tau / tau_q, A, M, tau, tau_q


def scan(bands, b, F_EFF, ov_fn, fine=True, eps_scale=None, apparent=False):
    out = []
    n_m2 = np.array([bd["n_s_cm2"] for bd in bands]) * 1e4
    m_kg = np.array([bd["m_over_m0"] for bd in bands]) * M0
    for name, (pname, grid, make) in families(b, F_EFF, fine).items():
        rows = []
        for p in grid:
            r, A, M, tau, tau_q = ratios(bands, make(p), b, ov_fn, eps_scale)
            rec = {"parameter": float(p), "ratio": r.tolist(),
                   "interband_share": [float(1 - A[i, i] / A[i].sum())
                                       for i in range(2)],
                   "miss_point": log_miss(*r, MS.TARGET_POINT),
                   "miss_box": log_miss(*r, MS.TARGET_BOX)}
            if apparent:
                n_c, mu_c = two_carrier_equivalent(M, n_m2, m_kg)
                mu_true = Q * tau / m_kg
                mu_q = Q * tau_q / m_kg
                app = mu_c / mu_q
                rec.update({
                    "apparent_ratio": app.tolist(),
                    "fit_n_over_true": (n_c / n_m2).tolist(),
                    "fit_mu_over_true": (mu_c / mu_true).tolist(),
                    "light_channel_more_mobile": bool(mu_c[0] > mu_c[1]),
                    "apparent_miss_point": log_miss(*app, MS.TARGET_POINT),
                    "apparent_miss_box": log_miss(*app, MS.TARGET_BOX)})
            rows.append(rec)
        fam = {"name": name, "parameter": pname, "scan": rows}
        for key in ("miss_point", "miss_box") + (
                ("apparent_miss_point", "apparent_miss_box") if apparent
                else ()):
            best = min(rows, key=lambda x: x[key])
            fam["best_" + key] = dict(best, factor=float(np.exp(best[key])))
        out.append(fam)
    return out


def closest(fams, key):
    return float(min(f["best_" + key]["factor"] for f in fams))


def main():
    ov_fn = overlap_function()
    NL, NH = MS.N_L_CM2, MS.N_H_CM2
    bands = [{"m_over_m0": MS.M_L, "n_s_cm2": NL},
             {"m_over_m0": MS.M_H, "n_s_cm2": NH}]
    b = S.fang_howard_b(NL + NH, 0.0, 1.9, EPS_R)
    F_EFF = Q * ((NL + NH) * 1e4 / 2.0) / (EPS_R * EPS0)

    out = {"measured": {"light": MS.R_L, "light_range": list(MS.R_L_RANGE),
                        "light_low_mass_assignment": MS.R_L_LOW,
                        "heavy_range": list(MS.R_H_RANGE), "heavy": MS.R_H}}
    print(f"measured: light {MS.R_L:.2f} ({MS.R_L_RANGE[0]:.2f}-"
          f"{MS.R_L_RANGE[1]:.2f}, down to {MS.R_L_LOW:.2f}), heavy "
          f"{MS.R_H_RANGE[0]:.2f}-{MS.R_H_RANGE[1]:.2f}")

    # ---- A and D: continuous scans, true and apparent ---------------------
    fams = scan(bands, b, F_EFF, ov_fn, fine=True, apparent=True)
    out["scan"] = fams
    for key in ("miss_point", "miss_box", "apparent_miss_point",
                "apparent_miss_box"):
        out["closest_" + key] = closest(fams, key)
    for f in fams:
        bp, bb = f["best_miss_point"], f["best_miss_box"]
        print(f"A {f['name']:24s} point: x{bp['factor']:.2f} at "
              f"{bp['parameter']:.3g} {np.round(bp['ratio'], 2)}   box: "
              f"x{bb['factor']:.2f} at {bb['parameter']:.3g} "
              f"{np.round(bb['ratio'], 2)}")
    print("A closest:", round(out["closest_miss_point"], 3),
          round(out["closest_miss_box"], 3))

    ok = [r for f in fams for r in f["scan"]
          if r["light_channel_more_mobile"]
          and all(0.6 <= x <= 1.4 for x in r["fit_n_over_true"])]
    out["two_carrier"] = {
        "n_cases": sum(len(f["scan"]) for f in fams),
        "n_cases_labelled_as_observed_and_densities_within_40pc": len(ok),
        "max_mobility_error_in_those_cases": float(
            max(max(abs(x - 1.0) for x in r["fit_mu_over_true"]) for r in ok)),
        "closest_apparent_point": out["closest_apparent_miss_point"],
        "closest_apparent_box": out["closest_apparent_miss_box"]}
    print("D", out["two_carrier"])

    # ---- B: screening strength ----------------------------------------------
    rows = []
    for sc in (0.1, 0.3, 3.0, 10.0):
        f = scan(bands, b, F_EFF, ov_fn, fine=True, eps_scale=sc)
        rows.append({"screening_scale": sc,
                     "closest_point": closest(f, "miss_point"),
                     "closest_box": closest(f, "miss_box")})
        print("B", rows[-1])
    out["screening_scan"] = rows

    # ---- C: density assignments ------------------------------------------
    cases = [("as published", NL, NH),
             ("both halved (spin resolved)", NL / 2, NH / 2),
             ("light halved only", NL / 2, NH),
             ("heavy halved only", NL, NH / 2),
             ("cyclotron-resonance densities", 0.65e13, 4.6e13),
             ("self-consistent occupations", 0.507e13, 4.093e13),
             ("equal densities", 2.3e13, 2.3e13)]
    rows = []
    for lab, nl, nh in cases:
        bd = [{"m_over_m0": MS.M_L, "n_s_cm2": nl},
              {"m_over_m0": MS.M_H, "n_s_cm2": nh}]
        bb = S.fang_howard_b(nl + nh, 0.0, 1.9, EPS_R)
        FF = Q * ((nl + nh) * 1e4 / 2.0) / (EPS_R * EPS0)
        f = scan(bd, bb, FF, ov_fn, fine=True)
        rows.append({"case": lab, "n_light_cm2": nl, "n_heavy_cm2": nh,
                     "closest_point": closest(f, "miss_point"),
                     "closest_box": closest(f, "miss_box")})
        print("C", rows[-1])
    out["density_cases"] = rows

    # ---- E: figure data ----------------------------------------------------
    out["figure"] = [{"name": f["name"],
                      "ratio_light": [r["ratio"][0] for r in f["scan"]],
                      "ratio_heavy": [r["ratio"][1] for r in f["scan"]],
                      "parameter": [r["parameter"] for r in f["scan"]]}
                     for f in fams]

    json.dump(out, open(os.path.join(RES, "tension.json"), "w"), indent=1)
    print("WROTE results/tension.json")


if __name__ == "__main__":
    main()
