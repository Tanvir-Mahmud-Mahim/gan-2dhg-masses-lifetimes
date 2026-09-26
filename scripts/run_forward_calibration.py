"""Calibrate the two routes to the heavy-hole quantum mobility with a
non-perturbative model of rho_xx.

sdh.rho_xx_quantum computes rho_xx(B) of the two-subband gas without expanding
in the density-of-states oscillation: Lorentzian Landau levels to all
harmonics, spin splitting, rates from the coupled Boltzmann equation with the
energy-dependent densities of states, the density-of-states factor in
sigma_xx, and the thermal average of sigma.  The disorder is the
self-consistent one of results/heavy_quantum_mobility.json (interface roughness
plus a power-law long-range component), the light-hole spin parameter is the
measured one, the heavy-hole spin factor is taken as 1.

For assumed heavy-hole quantum mobilities the synthetic rho_xx at the seven
temperatures of Fig. 2c is scaled so that its light-hole amplitude at 66.7 T
equals the digitized one and then analyzed exactly as the data:
  - a single-channel LK fit to the light-hole oscillation over 30 to 72 T,
    subtracted to leave the heavy-hole oscillation (as in Fig. 2c);
  - the joint LK + Dingle envelope fit of the heavy-hole oscillation, 50 to
    72 T (the envelope route);
  - the heavy-to-light amplitude ratio at 66.7, 63.3 and 60.2 T (the ratio
    route, here without any first-order approximation).
The measured envelope value and the measured ratios are then converted to
heavy-hole quantum mobilities by interpolating these calibrations.

Also: the second-harmonic ratio of the light-hole oscillation with and without
the measured spin parameter, and a run with digitizing noise.

Outputs results/forward_calibration.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)

import disorder_fit as DF                                  # noqa: E402
import run_heavy_quantum_mobility as HQ                    # noqa: E402
from gan2dhg import measured as MS, sdh                    # noqa: E402

RES = os.path.join(HERE, '..', 'results')
XF = np.arange(0.0137, 0.0334, 0.00002)
XG = np.arange(0.0138, 0.03321, 0.00005)
T_SET = HQ.T_HEAVY
TRUE = (74.0, 90.0, 100.0, 115.0, 170.0)


def disorder(hq):
    """Rate matrices of the self-consistent disorder of heavy_quantum_mobility.json."""
    last = hq["ratio_route"]["self_consistent"]["history"][-1]
    lib = DF.library()
    rough = [e for e in lib["interface roughness"] if abs(e[0] - last["roughness_nm"]) < 1e-6]
    plib = DF.power_library([last["exponent"]])
    r = DF.fit([[rough[0], plib[0]]], DF.measured(hq["ratio_route"]["self_consistent"]["mu_q_H"]))
    return r["_A"], r["_B"], DF.strip(r)


MU_L_IN = [MS.MU_Q_L]          # light-hole quantum mobility put into the model (cm^2/Vs)


def synth(A, Bk, T, muH, S_L, full=True):
    return sdh.rho_xx_quantum(XF, T, A, Bk, DF.V, (MS.N_L_CM2, MS.N_H_CM2),
                              (MS.MU_HALL_L * 1e-4, MS.MU_HALL_H * 1e-4), (MS.M_L, MS.M_H),
                              (166.0, 795.0), (MU_L_IN[0] * 1e-4, muH * 1e-4),
                              dos_in_sxx=full, spin=(S_L, 0.0))


def calibrate_light(A, Bk, S_L, target, muH=95.0):
    """Light-hole input for which the analysis returns the measured Dingle mobility.

    In rho_xx the light-hole oscillation is not an exact Dingle exponential (both
    carriers respond to it, and the heavy-hole oscillation leaks into the fit),
    so the apparent Dingle mobility differs from the input; the input is scaled
    until the apparent value at 1.8 K equals the target.
    """
    hist = []
    for it in range(4):
        y = synth(A, Bk, 1.8, muH, S_L)
        p, model, _ = HQ.lk_fit(XF, y, 1.8)
        app = p[1] * 1e4
        hist.append((MU_L_IN[0], app))
        if abs(app - target) < 1.0:
            break
        MU_L_IN[0] *= target / app
    return hist


def analyze(A, Bk, muH, S_L, a_meas, full=True, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    scale = None
    heavy, ratios, muL = [], {}, []
    for T in T_SET:
        y = synth(A, Bk, T, muH, S_L, full)
        if scale is None:
            p, model, _ = HQ.lk_fit(XF, y, T)
            scale = a_meas / (p[0] * np.exp(-np.pi * 0.015 / p[1]) * HQ.RT(MS.M_L, 0.015, T))
        y = y * scale
        p, model, _ = HQ.lk_fit(XF, y, T)
        muL.append(p[1] * 1e4)
        r = np.interp(XG, XF, y - model(p, XF))
        if noise > 0:
            r = np.round((r + noise * rng.standard_normal(len(r))) / 0.24) * 0.24
        heavy.append((T, XG, r))
        if T in HQ.T_RATIO:
            AL = lambda x0: p[0] * np.exp(-np.pi * x0 / p[1]) * HQ.RT(MS.M_L, x0, T)
            ratios[T] = [HQ.local_amp(XG, r, x0, 795, 2 / 795) / AL(x0) for x0 in HQ.RATIO_POINTS]
    env = HQ.heavy_joint(heavy)[0]
    return {"mu_q_H_true": muH, "full_response": full, "noise_ohm": noise,
            "envelope": env["mu_q_H"], "envelope_mass": env["m_H"], "envelope_F": env["F"],
            "light_dingle": muL, "ratios": {str(T): v for T, v in ratios.items()}}


def interp_inverse(true, measured_values, target):
    """True value that reproduces a target, interpolating in 1/mu and ln(value)."""
    t = 1.0 / np.asarray(true, float)
    v = np.log(np.asarray(measured_values, float))
    o = np.argsort(v)
    return float(1.0 / np.interp(np.log(target), v[o], t[o]))


def main():
    hq = json.load(open(os.path.join(RES, "heavy_quantum_mobility.json")))
    A, Bk, dis = disorder(hq)
    S_L = hq["light_second_harmonic"]["S_L"]
    a_meas = hq["light_fits"]["1.8"]["A"] * np.exp(-np.pi * 0.015 / (hq["light_fits"]["1.8"]["mu_q_cm2Vs"] * 1e-4)) \
        * HQ.RT(MS.M_L, 0.015, 1.8)
    lowT = [v["mu_q_cm2Vs"] for T, v in hq["light_fits"].items() if float(T) <= 6.0]
    target = float(np.mean(lowT))
    hist = calibrate_light(A, Bk, S_L, target)
    print("light input", hist, flush=True)
    out = {"disorder": dis, "S_L": S_L, "light_amplitude_66.7T_ohm": a_meas,
           "light_input": {"target_apparent": target, "iterations": hist, "mu_q_L_in": MU_L_IN[0]},
           "runs": []}
    for muH in TRUE:
        r = analyze(A, Bk, muH, S_L, a_meas)
        out["runs"].append(r)
        print("true %.0f  envelope %.1f  light %.0f  ratios 1.8 K %s" %
              (muH, r["envelope"], r["light_dingle"][0], np.round(r["ratios"]["1.8"], 3)), flush=True)
    for kw in (dict(muH=90.0, noise=0.3), dict(muH=100.0, full=False)):
        r = analyze(A, Bk, S_L=S_L, a_meas=a_meas, **kw)
        out["runs"].append(r)
        print("check", kw, "envelope %.1f ratios 1.8 K %s" % (r["envelope"], np.round(r["ratios"]["1.8"], 3)),
              flush=True)

    main_runs = [r for r in out["runs"] if r["full_response"] and r["noise_ohm"] == 0]
    tr = [r["mu_q_H_true"] for r in main_runs]
    env = [r["envelope"] for r in main_runs]
    e_meas = hq["envelope"]["baseline"]["mu_q_H"]
    lo, hi = hq["envelope"]["bootstrap"]["p2p5_p97p5"]
    lin = lambda target: float(np.interp(target, env, tr))
    out["envelope_calibrated"] = {"central": lin(e_meas), "bootstrap_95": [lin(lo), lin(hi)],
                                  "recovered_over_true": [e / t for e, t in zip(env, tr)],
                                  "note": "linear interpolation of true against recovered; outside the"
                                          " calibrated range the nearest segment is extrapolated"}
    # extrapolate linearly outside the table
    if lo < env[0]:
        s = (tr[1] - tr[0]) / (env[1] - env[0])
        out["envelope_calibrated"]["bootstrap_95"][0] = float(tr[0] + s * (lo - env[0]))
    if hi > env[-1]:
        s = (tr[-1] - tr[-2]) / (env[-1] - env[-2])
        out["envelope_calibrated"]["bootstrap_95"][1] = float(tr[-1] + s * (hi - env[-1]))

    ratio_nonpert = {}
    for T in ("1.8", "2.1"):
        meas = hq["measured_ratios"]["by_T"][T]
        ratio_nonpert[T] = [interp_inverse(tr, [r["ratios"][T][k] for r in main_runs], meas[k])
                            for k in range(3)]
    allv = ratio_nonpert["1.8"] + ratio_nonpert["2.1"]
    out["ratio_nonperturbative"] = {"by_T": ratio_nonpert, "mean": float(np.mean(allv)),
                                    "range": [float(min(allv)), float(max(allv))]}
    # second harmonic of the synthetic light-hole oscillation
    harm = {}
    for S in (0.0, S_L):
        y = synth(A, Bk, 1.8, 90.0, S)
        p, model, _ = HQ.lk_fit(XF, y, 1.8, harmonic2=True)
        a1 = p[0] * np.exp(-np.pi * 0.015 / p[1]) * HQ.RT(MS.M_L, 0.015, 1.8)
        a2 = p[8] * np.exp(-2 * np.pi * 0.015 / p[1]) * HQ.RT(MS.M_L, 0.030, 1.8)
        harm["S_L=%.3f" % S] = float(a2 / a1)
    out["synthetic_second_harmonic_66.7T"] = harm
    json.dump(out, open(os.path.join(RES, "forward_calibration.json"), "w"), indent=1, default=float)
    print("envelope calibrated", out["envelope_calibrated"])
    print("ratio non-perturbative", out["ratio_nonperturbative"])
    print("second harmonic", harm)
    print("WROTE results/forward_calibration.json")


if __name__ == "__main__":
    main()
