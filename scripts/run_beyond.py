"""Three of the four remaining caveats, turned into calculations.

1. THE RELAXATION-TIME FORM IS EXACT, NOT AN APPROXIMATION.
   For elastic scattering on an isotropic Fermi surface with a kernel that
   depends only on the angle between initial and final wavevectors, the
   linearised Boltzmann equation

       Int dtheta' W(theta - theta') [phi(theta) - phi(theta')] = v_x(theta)

   is solved exactly by the first angular harmonic, because cos(theta) is an
   eigenfunction of the collision operator.  Here the integral equation is
   discretised and solved as a linear system with no assumption about the form
   of phi, and the solution is projected onto angular harmonics.  If the
   closed form is exact, the projection carries no weight beyond the first
   harmonic and the resulting lifetime matches Int W (1 - cos) exactly.

2. INELASTIC MECHANISMS ARE FROZEN OUT, AND BY HOW MUCH.
   The Bloch-Gruneisen temperature is computed from the sound velocity, which
   follows from the elastic constants and the mass density already implied by
   the lattice parameters in use, so no new input is needed.  Carrier-carrier
   scattering does not relax momentum and so cannot affect tau_tr, but it does
   shorten the single-particle lifetime tau_q and could therefore raise the
   ratio; its size is estimated from the standard two-dimensional scaling.

3. THE SCATTERING CENTRES NEED NOT BE UNCORRELATED.
   Correlation between the positions of the scatterers enters as a structure
   factor, <|U(q)|^2> = N |u(q)|^2 S(q).  Ionised acceptors that repel one
   another have S(q) -> 0 as q -> 0, which suppresses precisely the forward
   scattering that sets tau_tr/tau_q, and suppresses it differently in the two
   subbands because they sample different momentum transfer.  Clustering does
   the opposite.  Both are scanned, and the question asked is whether either
   can reproduce both measured ratios where uncorrelated disorder cannot.

Outputs results/beyond.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import scatter2d as S                 # noqa: E402
from gan2dhg.constants import HBAR, M0, PI, Q      # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
EPS_R = 10.4
EPS0 = 8.8541878128e-12
KB = 1.380649e-23

N_L, N_H = 0.80e13, 3.80e13
M_L, M_H = 0.53, 1.92
R_L, R_H = 3.82, 2.13

# Measured lifetimes, from mobility and quantum mobility.
TAU_TR_L = S.tau_from_mobility(1900.0, M_L)
TAU_TR_H = S.tau_from_mobility(400.0, M_H)
TAU_Q_L, TAU_Q_H = 0.150e-12, 0.205e-12


def load_overlap():
    bar = json.load(open(os.path.join(RES, "barrier.json")))
    ov = bar["overlap"]
    th = np.array(ov["theta_rad"])

    def sym(tab):
        tab = np.asarray(tab)

        def f(t):
            t = np.mod(t, 2 * PI)
            t = np.where(t > PI, 2 * PI - t, t)
            return np.interp(t, th, tab)
        return f
    return sym(ov["light_to_heavy"]), sym(ov["heavy_to_heavy"]), \
        sym(ov["light_to_light"])


# ---------------------------------------------------------------------------
# 1.  The exact solution of the linearised Boltzmann equation
# ---------------------------------------------------------------------------

def boltzmann_exact(kF, w_fn, eps_fn, ov_fn, n=720):
    """Solve the integral equation without assuming a form for phi.

    Returns the transport rate from the exact solution, the rate from the
    closed form, and the weight the exact solution carries in harmonics above
    the first.
    """
    th = np.linspace(0.0, 2 * PI, n, endpoint=False)
    dth = th[1] - th[0]
    q = 2.0 * kF * np.abs(np.sin(th / 2.0))
    q = np.maximum(q, 1e-4 * kF)
    K = np.array([w_fn(qq) / eps_fn(qq) ** 2 for qq in q]) * ov_fn(th)

    # Collision operator on the circle: (L phi)(i) = sum_j K(i-j)[phi_i - phi_j]
    Kmat = np.empty((n, n))
    for i in range(n):
        Kmat[i] = np.roll(K, i)
    L = np.diag(Kmat.sum(axis=1)) - Kmat
    L *= dth

    # Drive by v_x ~ cos(theta).  L annihilates the constant, so the constant
    # mode is lifted explicitly by a projector of comparable magnitude rather
    # than by a token regularisation; the drive is orthogonal to it, so this
    # cannot contaminate the answer.
    drive = np.cos(th)
    scale = float(np.mean(np.diag(L)))
    A = L + (scale / n) * np.ones((n, n))
    phi = np.linalg.solve(A, drive)
    phi -= phi.mean()

    # Harmonic content of the solution.
    c1 = 2.0 / n * np.sum(phi * np.cos(th))
    higher = 0.0
    for m in (2, 3, 4, 5, 6, 7):
        higher = max(higher, abs(2.0 / n * np.sum(phi * np.cos(m * th))))
        higher = max(higher, abs(2.0 / n * np.sum(phi * np.sin(m * th))))

    rate_exact = 1.0 / c1 if c1 != 0 else np.inf
    rate_closed = np.sum(K * (1.0 - np.cos(th))) * dth
    return rate_exact, rate_closed, higher / abs(c1)


# ---------------------------------------------------------------------------
# 2.  Inelastic bounds
# ---------------------------------------------------------------------------

def inelastic_bounds():
    # Mass density of GaN from the lattice parameters already in use.
    a, c = 3.189e-10, 5.185e-10
    V = (np.sqrt(3.0) / 2.0) * a * a * c              # wurtzite cell volume
    amu = 1.66053906660e-27
    rho = 2.0 * (69.723 + 14.007) * amu / V           # two formula units
    C11, C33 = 390e9, 398e9
    v_a, v_c = np.sqrt(C11 / rho), np.sqrt(C33 / rho)

    out = {"density_kg_m3": float(rho),
           "v_sound_inplane_m_s": float(v_a),
           "v_sound_c_axis_m_s": float(v_c), "bands": []}
    for name, n_cm2, m, tau_tr, tau_q in (
            ("light", N_L, M_L, TAU_TR_L, TAU_Q_L),
            ("heavy", N_H, M_H, TAU_TR_H, TAU_Q_H)):
        kF = S.fermi_wavevector(n_cm2)
        # Bloch-Gruneisen: the largest phonon a carrier on the Fermi circle can
        # absorb has wavevector 2 k_F.
        T_BG = HBAR * v_a * 2.0 * kF / KB
        EF = HBAR ** 2 * kF ** 2 / (2.0 * m * M0)
        rec = {"band": name, "kF_per_nm": float(kF * 1e-9),
               "T_BG_K": float(T_BG), "EF_meV": float(EF / Q * 1e3),
               "measured_rate_q_per_s": float(1.0 / tau_q),
               "measured_rate_tr_per_s": float(1.0 / tau_tr), "T": []}
        for T in (2.0, 3.0):
            # Deep in the Bloch-Gruneisen regime the acoustic rate is
            # suppressed by (T/T_BG)^5 relative to its equipartition value.
            supp = (T / T_BG) ** 5
            # Carrier-carrier: leading two-dimensional scaling, which relaxes
            # no momentum and so bears only on tau_q.
            x = KB * T / EF
            rate_ee = (EF / HBAR) * x ** 2 * np.log(1.0 / x) / (4.0 * PI)
            rec["T"].append({
                "T_K": T, "T_over_T_BG": float(T / T_BG),
                "acoustic_suppression": float(supp),
                "kT_over_EF": float(x),
                "rate_ee_per_s": float(rate_ee),
                "rate_ee_over_measured_q": float(rate_ee * tau_q)})
        out["bands"].append(rec)
    return out


# ---------------------------------------------------------------------------
# 3.  Correlated scattering centres
# ---------------------------------------------------------------------------

def structure_factor(q, xi, kind):
    """S(q) for correlated scatterer positions.

    'hole'    a correlation hole of radius xi, as mutually repelling ionised
              centres produce: S -> 0 at small q, S -> 1 at large q.
    'cluster' the opposite, excess long-wavelength density fluctuation.
    """
    g = np.exp(-(q * xi) ** 2 / 4.0)
    if kind == "hole":
        return 1.0 - g
    if kind == "cluster":
        return 1.0 + 3.0 * g
    return np.ones_like(q) if isinstance(q, np.ndarray) else 1.0


def correlated_scan(bands, b, ov_fn, F_EFF):
    """Best simultaneous account of both ratios when the centres are correlated.

    The structure factor multiplies the squared matrix element, so it enters
    exactly where the uncorrelated treatment put unity.  The scan is run
    through the same coupled two-subband solver used everywhere else, so the
    number it returns is directly comparable with the uncorrelated best fit.
    Both the correlation length and each mechanism's own shape parameter are
    varied, since correlation could in principle rescue a shape that alone
    fails.
    """
    fams = {
        "interface roughness": (np.linspace(0.4, 8.0, 12),
                                lambda L: (lambda q: S.w_interface_roughness(
                                    q, 0.3e-9, L * 1e-9, F_EFF))),
        "remote charge": (np.linspace(0.5, 20.0, 12),
                          lambda d: (lambda q: S.w_remote_impurity(
                              q, 5e12, d * 1e-9, b, EPS_R))),
        "background impurities": (np.array([1.0]),
                                  lambda _: (lambda q: S.w_background_impurity(
                                      q, 1e17, b, EPS_R))),
        "dislocations": (np.linspace(0.05, 1.0, 6),
                         lambda f: (lambda q: S.w_dislocation(
                             q, 1e4, 4.982e-10, f, EPS_R, b))),
    }
    # The winning family is refined on a fine joint grid, because the numbers
    # quoted are the best simultaneous account over both the correlation length
    # and the mechanism's own shape parameter, and a coarse grid does not
    # locate that minimum.
    fams["interface roughness"] = (np.linspace(0.3, 10.0, 30),
                                   fams["interface roughness"][1])
    rows, best = [], None
    for kind in ("none", "hole", "cluster"):
        xis = [0.0] if kind == "none" else list(np.linspace(0.3, 6.0, 20))
        for xi_nm in xis:
            xi = xi_nm * 1e-9
            for fam, (grid, make) in fams.items():
                for p in grid:
                    base = make(p)

                    def w(q, _b=base, _xi=xi, _k=kind):
                        return _b(q) * structure_factor(q, _xi, _k)

                    r = S.two_band_lifetimes(bands, w, EPS_R, b,
                                             overlap_fn=ov_fn)
                    rl, rh = float(r["ratio"][0]), float(r["ratio"][1])
                    miss = max(abs(np.log(rl / R_L)), abs(np.log(rh / R_H)))
                    rec = {"kind": kind, "xi_nm": xi_nm, "family": fam,
                           "parameter": float(p), "ratio_light": rl,
                           "ratio_heavy": rh, "light_over_heavy": rl / rh,
                           "log_miss": float(miss)}
                    rows.append(rec)
                    if best is None or miss < best["log_miss"]:
                        best = rec
    return rows, best


def main():
    f_lh, f_hh, f_ll = load_overlap()
    bands = [{"m_over_m0": M_L, "n_s_cm2": N_L},
             {"m_over_m0": M_H, "n_s_cm2": N_H}]
    b = S.fang_howard_b(N_L + N_H, 0.0, 1.9, EPS_R)
    F_EFF = Q * ((N_L + N_H) * 1e4 / 2.0) / (EPS_R * EPS0)
    mech = {
        "remote charge, d = 12 nm":
            lambda q: S.w_remote_impurity(q, 5e12, 12e-9, b, EPS_R),
        "roughness, L = 1 nm":
            lambda q: S.w_interface_roughness(q, 0.3e-9, 1.0e-9, F_EFF),
        "roughness, L = 3 nm":
            lambda q: S.w_interface_roughness(q, 0.3e-9, 3.0e-9, F_EFF),
        "background impurities":
            lambda q: S.w_background_impurity(q, 1e17, b, EPS_R),
        "dislocations":
            lambda q: S.w_dislocation(q, 1e4, 4.982e-10, 0.3, EPS_R, b),
    }
    out = {}

    # ---- 1 ---------------------------------------------------------------
    rows = []
    for name, w in mech.items():
        for lab, bd, ovf in (("light", bands[0], f_ll),
                             ("heavy", bands[1], f_hh)):
            kF = S.fermi_wavevector(bd["n_s_cm2"])
            re, rc, h = boltzmann_exact(
                kF, w, lambda q: S.dielectric(q, bands, EPS_R, b), ovf)
            rows.append({"mechanism": name, "band": lab,
                         "rate_exact": re, "rate_closed_form": rc,
                         "fractional_difference": abs(re / rc - 1.0),
                         "weight_above_first_harmonic": h})
    out["boltzmann"] = rows
    out["boltzmann_max_fractional_difference"] = max(
        r["fractional_difference"] for r in rows)
    out["boltzmann_max_higher_harmonic"] = max(
        r["weight_above_first_harmonic"] for r in rows)

    # ---- 2 ---------------------------------------------------------------
    out["inelastic"] = inelastic_bounds()

    # ---- 3 ---------------------------------------------------------------
    rows, best = correlated_scan(bands, b, lambda i, j, t:
                                 (f_ll(t) if i == 0 else f_hh(t))
                                 if i == j else f_lh(t), F_EFF)
    out["correlated"] = rows
    out["correlated_best"] = best
    out["correlated_best_factor"] = float(np.exp(best["log_miss"]))
    unc = [r for r in rows if r["kind"] == "none"]
    out["uncorrelated_best"] = min(unc, key=lambda r: r["log_miss"])
    out["uncorrelated_best_factor"] = float(
        np.exp(out["uncorrelated_best"]["log_miss"]))
    out["correlated_max_light_over_heavy"] = max(
        r["light_over_heavy"] for r in rows)
    out["best_by_kind"] = {
        k: min([r for r in rows if r["kind"] == k], key=lambda r: r["log_miss"])
        for k in ("none", "hole", "cluster")}

    json.dump(out, open(os.path.join(RES, "beyond.json"), "w"), indent=1)
    print("WROTE results/beyond.json")
    print(f"  Boltzmann: exact vs closed form differ by at most "
          f"{out['boltzmann_max_fractional_difference']:.2e}; "
          f"weight above the first harmonic at most "
          f"{out['boltzmann_max_higher_harmonic']:.2e}")
    for rec in out["inelastic"]["bands"]:
        t = rec["T"][1]
        print(f"  {rec['band']}: T_BG = {rec['T_BG_K']:.0f} K, "
              f"acoustic suppression at 3 K = {t['acoustic_suppression']:.2e}, "
              f"carrier-carrier rate / measured quantum rate = "
              f"{t['rate_ee_over_measured_q']:.2e}")
    print(f"  uncorrelated best: {out['uncorrelated_best']['family']}, "
          f"off by a factor {out['uncorrelated_best_factor']:.2f}")
    print(f"  correlated best:   {best['kind']} at xi = {best['xi_nm']} nm, "
          f"{best['family']}: L {best['ratio_light']:.3f} "
          f"H {best['ratio_heavy']:.3f}, off by a factor "
          f"{out['correlated_best_factor']:.2f}")
    print(f"  largest light/heavy reachable: "
          f"{out['correlated_max_light_over_heavy']:.3f} "
          f"(measured {R_L / R_H:.3f})")


if __name__ == "__main__":
    main()
