"""The heavy-hole quantum mobility from the measured oscillations.

Chang et al. obtain the light-hole quantum mobility from a Dingle analysis
(368 cm^2/Vs) but estimate the heavy-hole value, 167 to 200 cm^2/Vs, from the
onset of the heavy-hole oscillations at 50 to 60 T, because the small
signal-to-noise ratio precludes a Dingle analysis; the two ranges correspond
to omega_c tau_q = mu_q B = 1 at the onset.  That estimate assumes the two
oscillations appear in rho_xx with equal weight.  They do not
(src/gan2dhg/sdh.py): the heavy holes carry most of sigma_xx at these fields.

Data: data/chang2026_fig2_digitized.json, Fig. 2a (light-hole oscillation,
Delta R_xx) and Fig. 2c (heavy-hole oscillation left after subtracting the
light-hole fit) of Chang et al. read at the native resolution of the embedded
image, ten temperatures.

This script
  1. validates the digitization against what Chang et al. extract from the
     same figure: the light-hole Dingle mobility at each temperature, the
     light-hole Lifshitz-Kosevich (LK) mass at several fields, the heavy-hole LK
     mass and the normalized heavy-hole amplitudes of their Fig. 2f;
  2. measures the light-hole spin reduction factor from the second harmonic
     of the light-hole oscillation;
  3. fits the heavy-hole oscillation at seven temperatures jointly with a
     Dingle envelope (the "envelope route"), with a jackknife over temperatures,
     a moving-block bootstrap of the residuals and variations of the analysis;
  4. forms the heavy-to-light amplitude ratio at 66.7, 63.3 and 60.2 T and
     turns it into a heavy-hole quantum mobility (the "ratio route") for
       - no intersubband scattering (W = identity),
       - the rates-only response used previously and the full response with
         the density-of-states factor in sigma_xx,
       - the density-of-states weights of the disorder that fits the four
         mobilities, found self-consistently (interface roughness plus a
         long-range component with a power-law spectrum whose exponent is
         fitted), with the measured light-hole spin factor;
  5. tests whether a spread of the local Fermi level (density inhomogeneity)
     could replace long-range scattering.

The envelope route is calibrated with the non-perturbative model in
scripts/run_forward_calibration.py, which reads this output.

Outputs results/heavy_quantum_mobility.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares, brentq

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)

import disorder_fit as DF                                  # noqa: E402
from gan2dhg import measured as MS, sdh                    # noqa: E402

RES = os.path.join(HERE, '..', 'results')
DATA = os.path.join(HERE, '..', 'data', 'chang2026_fig2_digitized.json')
N = (MS.N_L_CM2, MS.N_H_CM2)
MU_HALL = (MS.MU_HALL_L * 1e-4, MS.MU_HALL_H * 1e-4)       # m^2/Vs
MU_Q_L = MS.MU_Q_L * 1e-4
MASS = (MS.M_L, MS.M_H)
RATIO_POINTS = (0.0150, 0.0158, 0.0166)                   # 1/B, T^-1
T_HEAVY = [1.8, 2.1, 2.5, 2.7, 4.0, 4.6, 6.0]
T_RATIO = [1.8, 2.1, 2.5, 2.7, 4.0, 4.6]


def RT(m, x, T):
    """LK thermal factor at 1/B = x (harmonic r: use r * x)."""
    return sdh.thermal_factor(m, 1.0 / np.asarray(x, float), T)


def load():
    D = json.load(open(DATA))

    def curves(key):
        c = D[key]
        return {float(T): (c["inv_B_start"] + c["inv_B_step"] * np.arange(len(v)), np.array(v))
                for T, v in c["curves"].items()}
    return D, curves("light_fig2a"), curves("heavy_fig2c")


def lk_fit(x, y, T, mass=MS.M_L, harmonic2=False):
    """Single-channel LK fit with a cubic background (light-hole oscillation).

    Unbounded least squares with Jacobian scaling, started from three Dingle
    mobilities and eight phases (and four phases of the second harmonic); the
    fit converges to the same minimum from every start.
    """
    def model(p, x):
        A, mu, F, ph, c0, c1, c2, c3 = p[:8]
        e = np.exp(-np.pi * x / mu)
        t = x - 0.023
        out = (A * e * RT(mass, x, T) * np.cos(2 * np.pi * F * x + ph)
               + c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3)
        if harmonic2:
            A2, ph2 = p[8:10]
            out = out + A2 * e ** 2 * RT(mass, 2 * x, T) * np.cos(4 * np.pi * F * x + ph2)
        return out
    # the data are normalized to unit spread for the fit and the linear
    # parameters are scaled back afterwards
    sc = float(np.std(y))
    yn = (y - np.mean(y)) / sc
    best = None
    for mu0 in (0.02, 0.037, 0.06):
        for ph0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            for ph2 in ((0.0, np.pi / 2, np.pi, 1.5 * np.pi) if harmonic2 else (None,)):
                p0 = [3.0 * np.exp(np.pi * 0.023 / mu0), mu0, 166.0, ph0, 0.0, 0.0, 0.0, 0.0] + \
                    ([0.3 * np.exp(2 * np.pi * 0.023 / mu0), ph2] if harmonic2 else [])
                r = least_squares(lambda p: model(p, x) - yn, p0, x_scale='jac')
                if r.x[1] > 0 and (best is None or r.cost < best.cost):
                    best = r
    p = best.x.copy()
    for k in (0, 4, 5, 6, 7) + ((8,) if harmonic2 else ()):
        p[k] *= sc
    p[4] += np.mean(y)
    best.cost *= sc ** 2
    if p[0] < 0:
        p[0], p[3] = -p[0], p[3] + np.pi
    if harmonic2 and p[8] < 0:
        p[8], p[9] = -p[8], p[9] + np.pi
    return p, model, best


def local_amp(x, y, x0, F, w):
    m = np.abs(x - x0) <= w / 2
    X = np.c_[np.cos(2 * np.pi * F * x[m]), np.sin(2 * np.pi * F * x[m]),
              np.ones(m.sum()), x[m] - x0]
    c = np.linalg.lstsq(X, y[m], rcond=None)[0]
    return float(np.hypot(c[0], c[1]))


def heavy_joint(data, p_init=None, xmin=0.0138, xmax=0.0200, bg=1, Ffix=None, mfix=None):
    """Joint LK + Dingle fit of the heavy-hole oscillation at several temperatures."""
    data = [(T, x[(x >= xmin - 1e-9) & (x <= xmax + 1e-9)], y[(x >= xmin - 1e-9) & (x <= xmax + 1e-9)])
            for T, x, y in data]
    n, nb = len(data), bg + 1

    def unpack(p):
        A, g, F, ph, m = p[:5]
        return A, g, (Ffix or F), ph, (mfix or m)

    def model(p):
        A, g, F, ph, m = unpack(p)
        out = []
        for k, (T, x, y) in enumerate(data):
            c = p[5 + nb * k:5 + nb * (k + 1)]
            out.append(A * np.exp(-np.pi * x / g) * RT(m, x, T) * np.cos(2 * np.pi * F * x + ph)
                       + sum(c[j] * (x - 0.0165) ** j for j in range(nb)))
        return out

    def resid(p):
        return np.concatenate([mo - y for mo, (T, x, y) in zip(model(p), data)])
    starts = [p_init] if p_init is not None else \
        [[20, g0, 795, ph0, 1.9] + [0] * (nb * n)
         for g0 in (0.006, 0.010, 0.016) for ph0 in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    best = None
    for p0 in starts:
        r = least_squares(resid, p0, x_scale='jac',
                          bounds=([0, 1e-4, 650, -50, 0.3] + [-1e5] * (nb * n),
                                  [1e7, 1, 950, 50, 6] + [1e5] * (nb * n)))
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    dof = len(best.fun) - len(best.x)
    cov = np.linalg.pinv(J.T @ J) * 2 * best.cost / dof
    A, g, F, ph, m = unpack(best.x)
    return ({"mu_q_H": g * 1e4, "sd_mu": float(np.sqrt(abs(cov[1, 1])) * 1e4), "F": F, "m_H": m,
             "sd_m": float(np.sqrt(abs(cov[4, 4]))), "rms": float(np.sqrt(2 * best.cost / dof))},
            best, model, data)


def ratio_route(ratios, W, dos_in_sxx, spin_L):
    vals = []
    for T, r in ratios.items():
        for x, ra in zip(RATIO_POINTS, r):
            K = sdh.response_weights(1 / x, N, MU_HALL, W, dos_in_sxx)
            vals.append(1e4 * sdh.heavy_quantum_mobility(ra, 1 / x, T, K, MU_Q_L, MASS,
                                                         spin=(spin_L, 1.0)))
    return {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals)),
            "values": vals}


def kratio(B, W, full):
    K = sdh.response_weights(B, N, MU_HALL, W, full)
    return float(K[1] / K[0])


def main():
    D, L, H = load()
    out = {"data": os.path.relpath(DATA, os.path.join(HERE, '..'))}

    # 1. validation ---------------------------------------------------------------
    light = {}
    for T, (x, y) in L.items():
        p, model, r = lk_fit(x, y, T)
        light[T] = {"A": p[0], "mu_q_cm2Vs": p[1] * 1e4, "F": p[2],
                    "rms_ohm": float(np.sqrt(2 * r.cost / (len(x) - 10)))}
    out["light_fits"] = {str(T): v for T, v in light.items()}
    lowT = sorted(T for T in light if T <= 6.0)
    out["light_dingle_1p8_to_6K"] = [min(light[T]["mu_q_cm2Vs"] for T in lowT),
                                     max(light[T]["mu_q_cm2Vs"] for T in lowT)]
    out["reported_light_dingle"] = [MS.MU_Q_L, 14.0]
    # LK mass from the amplitude of each light-hole oscillation at fixed field,
    # fitted locally (one period, linear background) at every temperature
    lkm = {}
    Ts = np.array(sorted(L))
    for xe in (0.0300, 0.0290, 0.0250, 0.0200, 0.0170, 0.0156, 0.0150):
        a = np.array([local_amp(L[t][0], L[t][1], xe, 166, 1 / 166) for t in Ts])
        f = least_squares(lambda q: q[0] * RT(q[1], xe, Ts) - a, [a[0], 0.5])
        lkm["%.1f T" % (1 / xe)] = float(f.x[1])
    out["light_LK_mass"] = lkm
    out["reported_light_LK_mass"] = {"32 T": 0.48, "72 T": 0.69}

    ratios = {}
    for T in T_RATIO:
        def AL(x0, T=T):
            return (light[T]["A"] * np.exp(-np.pi * x0 / (light[T]["mu_q_cm2Vs"] * 1e-4))
                    * RT(MS.M_L, x0, T))
        xh, yh = H[T]
        ratios[T] = [local_amp(xh, yh, x0, 795, 2 / 795) / AL(x0) for x0 in RATIO_POINTS]
    out["measured_ratios"] = {"inv_B": list(RATIO_POINTS),
                              "by_T": {str(T): r for T, r in ratios.items()}}
    heavy_local = {T: local_amp(H[T][0], H[T][1], 0.0150, 795, 2 / 795) for T in T_HEAVY}
    f2f = D["fig2f_heavy_normalized"]
    mine = [heavy_local[T] / heavy_local[1.8] for T in T_HEAVY]
    out["fig2f_comparison"] = {"T_K": T_HEAVY, "this_digitization": mine,
                               "chang_fig2f": f2f["value"],
                               "max_abs_difference": float(max(abs(a - b) for a, b in zip(mine, f2f["value"])))}

    # 2. light-hole spin factor from the second harmonic --------------------------
    harm = {}
    for T in lowT:
        # Dingle mobility and frequency from the fundamental-only fit; the two
        # harmonics (Dingle factors e and e^2, thermal factors at x and 2x) and a
        # cubic background are then fitted linearly
        x, y = L[T]
        mu, F = light[T]["mu_q_cm2Vs"] * 1e-4, light[T]["F"]
        e = np.exp(-np.pi * x / mu)
        t = x - 0.023
        g1 = e * RT(MS.M_L, x, T)
        g2 = e ** 2 * RT(MS.M_L, 2 * x, T)
        X = np.c_[g1 * np.cos(2 * np.pi * F * x), g1 * np.sin(2 * np.pi * F * x),
                  g2 * np.cos(4 * np.pi * F * x), g2 * np.sin(4 * np.pi * F * x),
                  np.ones_like(x), t, t ** 2, t ** 3]
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        A1, A2 = np.hypot(c[0], c[1]), np.hypot(c[2], c[3])
        a1 = A1 * np.exp(-np.pi * 0.015 / mu) * RT(MS.M_L, 0.015, T)
        a2 = A2 * np.exp(-2 * np.pi * 0.015 / mu) * RT(MS.M_L, 0.030, T)
        expected = np.exp(-np.pi * 0.015 / mu) * RT(MS.M_L, 0.030, T) / RT(MS.M_L, 0.015, T)
        harm[T] = {"A2_over_A1_66.7T": float(a2 / a1), "lorentzian_no_spin": float(expected)}
    r21 = np.array([v["A2_over_A1_66.7T"] / v["lorentzian_no_spin"] for v in harm.values()])
    spin_L = float(sdh.spin_factor_from_harmonics(np.mean(r21)))
    out["light_second_harmonic"] = {
        "by_T": {str(T): v for T, v in harm.items()},
        "Rs2_over_Rs1_mean": float(np.mean(r21)),
        "Rs2_over_Rs1_range": [float(r21.min()), float(r21.max())],
        "spin_factor_L": spin_L,
        "spin_factor_L_range": [float(sdh.spin_factor_from_harmonics(r21.min())),
                                float(sdh.spin_factor_from_harmonics(r21.max()))],
        "S_L": float(np.arccos(spin_L) / np.pi)}
    print("light spin factor %.3f" % spin_L, flush=True)

    # 3. envelope route -------------------------------------------------------------
    data = [(T, H[T][0], H[T][1]) for T in T_HEAVY]
    base, best, model, dsel = heavy_joint(data)
    out["envelope"] = {"window": "50 to 72 T, 1.8 to 6.0 K", "baseline": base}
    print("envelope: mu %.1f m %.2f F %.0f" % (base["mu_q_H"], base["m_H"], base["F"]), flush=True)
    out["envelope"]["jackknife"] = [heavy_joint(data[:i] + data[i + 1:])[0]["mu_q_H"]
                                    for i in range(len(data))]
    mod = model(best.x)
    res = [y - m for (T, x, y), m in zip(dsel, mod)]
    rng = np.random.default_rng(3)
    bs = []
    for it in range(200):
        nd = []
        for (T, x, y), m, r in zip(dsel, mod, res):
            Lr = len(r)
            idx = np.concatenate([np.arange(s, s + 25) % Lr
                                  for s in rng.integers(0, Lr, Lr // 25 + 1)])[:Lr]
            nd.append((T, x, m + r[idx]))
        bs.append(heavy_joint(nd, p_init=best.x)[0]["mu_q_H"])
    bs = np.array(bs)
    out["envelope"]["bootstrap"] = {"n": 200, "block": "25 samples (one heavy-hole period)",
                                    "median": float(np.median(bs)),
                                    "p16_p84": np.percentile(bs, [16, 84]).tolist(),
                                    "p2p5_p97p5": np.percentile(bs, [2.5, 97.5]).tolist()}
    var = {}
    for lbl, kw in (("m fixed 1.92", dict(mfix=1.92)), ("F fixed 795", dict(Ffix=795)),
                    ("background order 0", dict(bg=0)), ("background order 2", dict(bg=2)),
                    ("to 54 T", dict(xmax=0.0185)), ("to 52 T", dict(xmax=0.0192)),
                    ("from 70 T", dict(xmin=0.0142)), ("from 68.5 T", dict(xmin=0.0146))):
        var[lbl] = heavy_joint(data, **kw)[0]["mu_q_H"]
    for lbl, sub in (("drop 1.8 K", data[1:]), ("1.8 to 2.7 K", data[:4]),
                     ("4.0 to 6.0 K", data[4:])):
        var[lbl] = heavy_joint(sub)[0]["mu_q_H"]
    out["envelope"]["variations"] = var
    out["envelope"]["range_over_variations"] = [min(var.values()), max(var.values())]

    # 4. ratio route ----------------------------------------------------------------
    I = np.eye(2)
    out["ratio_route"] = {"no_intersubband_full_response": ratio_route(ratios, I, True, 1.0),
                          "no_intersubband_rates_only": ratio_route(ratios, I, False, 1.0)}
    lib = DF.library()
    rough = [e for e in lib["interface roughness"] if e[0] <= 2.01]
    plib = DF.power_library(np.arange(2.8, 5.21, 0.05))
    muH = 115.0
    hist = []
    for it in range(8):
        r = DF.fit([[e1, e2] for e1 in rough for e2 in plib], DF.measured(muH))
        W = np.array(r["W"])
        new = ratio_route(ratios, W, True, spin_L)["mean"]
        hist.append({"target": muH, "roughness_nm": r["parameters"][0],
                     "exponent": r["parameters"][1], "worst_factor": r["worst_factor"],
                     "W": W.tolist(), "result": new})
        print("self-consistency", it, "%.1f -> %.1f  p %.2f" % (muH, new, r["parameters"][1]),
              flush=True)
        if abs(new - muH) < 0.5:
            muH = new
            break
        muH = new
    Wsc = np.array(hist[-1]["W"])
    out["ratio_route"]["self_consistent"] = {
        "history": hist, "mu_q_H": muH, "W": Wsc.tolist(),
        "full_response_spin": ratio_route(ratios, Wsc, True, spin_L),
        "full_response_equal_spin": ratio_route(ratios, Wsc, True, 1.0),
        "rates_only_equal_spin": ratio_route(ratios, Wsc, False, 1.0),
        "K_H_over_K_L_63T": {"full": kratio(63.3, Wsc, True), "rates_only": kratio(63.3, Wsc, False)}}
    # the previous disorder model (roughness 0.8 nm + line charges), for comparison
    W_old = np.array([[0.1368629831688395, 0.8631370160827601],
                      [0.22577585134797573, 0.7742241496799807]])
    out["ratio_route"]["previous_model"] = {
        "W": W_old.tolist(),
        "rates_only_equal_spin": ratio_route(ratios, W_old, False, 1.0),
        "full_response_equal_spin": ratio_route(ratios, W_old, True, 1.0),
        "K_H_over_K_L_63T": {"full": kratio(63.3, W_old, True), "rates_only": kratio(63.3, W_old, False)}}
    conv = []
    for mu in MS.MU_Q_H_RANGE:
        for x, ra in zip(RATIO_POINTS, ratios[1.8]):
            K = sdh.response_weights(1 / x, N, MU_HALL, Wsc, True)
            pred = sdh.amplitude_ratio(1 / x, 1.8, K, (MU_Q_L, mu * 1e-4), MASS, (spin_L, 1.0))
            conv.append({"mu_q_H": mu, "B_T": 1 / x, "factor": float(pred / ra)})
    out["conventional_value_check"] = conv

    # 5. density inhomogeneity --------------------------------------------------------
    best_r = None
    for Lc, (A, Bk) in lib["interface roughness"]:
        s = least_squares(lambda z: np.log(DF.solve(np.exp(z[0]) * A, np.exp(z[0]) * Bk)[0]
                                           / np.array([MS.MU_HALL_L, MS.MU_HALL_H])), [0.0])
        if best_r is None or s.cost < best_r[0].cost:
            best_r = (s, Lc, A, Bk)
    s, Lc, A, Bk = best_r
    a = np.exp(s.x[0])
    mt, mq = DF.solve(a * A, a * Bk)
    Wr = sdh.dos_weights(a * A, a * Bk, DF.V)
    muL0, muH0 = mq * 1e-4
    xw = np.linspace(0.015, 0.030, 31)

    def envL(x, sg):
        return np.exp(-np.pi * x / muL0
                      - 2 * np.pi ** 2 * (MS.M_L * sg / sdh.HBAR_E_OVER_M0) ** 2 * x ** 2)

    def app(sg):
        return -np.pi / np.polyfit(xw, np.log(envL(xw, sg)), 1)[0]
    sig = brentq(lambda sg: app(sg) - MU_Q_L, 1e-6, 0.02)
    dFH = MS.M_H * sig / sdh.HBAR_E_OVER_M0
    pred = []
    for x in RATIO_POINTS:
        eH = np.exp(-np.pi * x / muH0 - 2 * np.pi ** 2 * dFH ** 2 * x ** 2)
        for lbl, full in (("rates only", False), ("full", True)):
            K = sdh.response_weights(1 / x, N, MU_HALL, Wr, full)
            pred.append({"B_T": 1 / x, "response": lbl,
                         "ratio": float(K[1] * RT(MS.M_H, x, 1.8) * eH
                                        / (K[0] * RT(MS.M_L, x, 1.8) * envL(x, sig)))})
    xh = np.linspace(0.0138, 0.020, 30)
    sH = np.polyfit(xh, np.log(np.exp(-np.pi * xh / muH0 - 2 * np.pi ** 2 * dFH ** 2 * xh ** 2)), 1)[0]
    out["inhomogeneity"] = {"roughness_nm": Lc, "mu_tr": mt.tolist(), "mu_q": mq.tolist(),
                            "sigma_meV": sig * 1e3,
                            "light_density_spread_percent":
                                100 * MS.M_L * sig / sdh.HBAR_E_OVER_M0 / 166.0,
                            "predicted_ratios": pred, "apparent_heavy_dingle": float(-np.pi / sH * 1e4)}

    json.dump(out, open(os.path.join(RES, "heavy_quantum_mobility.json"), "w"), indent=1,
              default=float)
    print("WROTE results/heavy_quantum_mobility.json")


if __name__ == "__main__":
    main()
