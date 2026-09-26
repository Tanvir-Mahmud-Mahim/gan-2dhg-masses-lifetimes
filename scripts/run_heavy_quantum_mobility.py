"""The heavy-hole quantum mobility from the measured oscillation amplitudes.

Chang et al. obtain the light-hole quantum mobility from a Dingle analysis
(368 cm^2/Vs) but estimate the heavy-hole value, 167 to 200 cm^2/Vs, from the
onset of the heavy-hole oscillations at 50 to 60 T, because the small
signal-to-noise ratio precludes a Dingle analysis; the two ranges correspond
to omega_c tau_q = mu_q B = 1 at the onset.  That estimate assumes the two oscillations appear in
rho_xx with equal weight.  They do not (src/gan2dhg/sdh.py): the heavy holes
carry most of sigma_xx at these fields.

This script
  1. checks the digitized amplitudes of Fig. 2 of Chang et al.
     (data/chang2026_fig2_digitized.json) against the reported light-hole
     Dingle mobility;
  2. computes the sensitivities S_c of rho_xx with the measured densities and
     Hall mobilities;
  3. turns the measured ratio of the heavy- and light-hole amplitudes at 60 to
     67 T into a heavy-hole quantum mobility, for
       - no scattering between the subbands (W = identity),
       - the density-of-states weights W of every single mechanism of the
         scans (run_tension.families), which bracket the result, and
       - the weights of the disorder that fits the four mobilities, found
         self-consistently: the fit (roughness plus a line-charge component,
         the best pair of scripts/run_mixtures.py) depends on the heavy-hole
         value, and the weights depend on the fit;
  4. repeats the estimate with the equal-visibility criterion (light-hole
     onset near 25 T, heavy-hole onset at 50 to 60 T) as a cross-check;
  5. reports how large the heavy-hole oscillation would be if its quantum
     mobility were the conventional 167 to 200 cm^2/Vs.

Outputs results/heavy_quantum_mobility.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares, brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import run_mixtures as RM                                  # noqa: E402
import run_tension as RT                                   # noqa: E402
from gan2dhg import measured as MS, scatter2d as S, sdh    # noqa: E402
from gan2dhg.constants import Q                            # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, '..', 'results')
DATA = os.path.join(HERE, '..', 'data', 'chang2026_fig2_digitized.json')
N = (MS.N_L_CM2, MS.N_H_CM2)
MU_HALL = (MS.MU_HALL_L * 1e-4, MS.MU_HALL_H * 1e-4)       # m^2/Vs
MU_Q_L = MS.MU_Q_L * 1e-4
MASS = (MS.M_L, MS.M_H)
RATIO_POINTS = (0.0150, 0.0158, 0.0166)                   # 1/B, T^-1


def measured_ratios(D):
    out = {}
    for tag, T in (("1p8K", 1.8), ("2p1K", 2.1)):
        L = D["light_" + tag]
        H = D["heavy_" + tag]
        rows = []
        for x in RATIO_POINTS:
            aL = float(np.interp(x, L["inv_B_T"], L["amp_ohm"]))
            aH = float(np.interp(x, H["inv_B_T"], H["amp_ohm"]))
            rows.append({"B_T": 1.0 / x, "A_H_over_A_L": aH / aL})
        out[tag] = {"T_K": T, "points": rows}
    return out


def mu_H_for(W, ratios):
    """Heavy-hole quantum mobility (cm^2/Vs) at each field and temperature."""
    res = []
    for tag, r in ratios.items():
        for p in r["points"]:
            B = p["B_T"]
            K = sdh.sensitivities(B, N, MU_HALL) @ W
            res.append(1e4 * sdh.heavy_quantum_mobility(
                p["A_H_over_A_L"], B, r["T_K"], K, MU_Q_L, MASS))
    return res


def main():
    D = json.load(open(DATA))
    out = {"data": os.path.relpath(DATA, os.path.join(HERE, '..'))}

    # 1. validation of the digitized light-hole amplitudes
    x = np.array(D["light_1p8K"]["inv_B_T"])
    a = np.array(D["light_1p8K"]["amp_ohm"])
    s, _ = np.polyfit(x, np.log(a), 1)
    out["digitized_light_dingle_mobility_cm2Vs"] = float(-np.pi / s * 1e4)
    out["reported_light_dingle_mobility_cm2Vs"] = MS.MU_Q_L

    # 2. sensitivities
    out["sensitivities"] = {str(B): sdh.sensitivities(B, N, MU_HALL).tolist()
                            for B in (25.0, 40.0, 55.0, 60.2, 63.3, 66.7, 72.0)}

    # 3. heavy-hole quantum mobility
    ratios = measured_ratios(D)
    out["measured_ratios"] = ratios
    out["no_intersubband"] = {"W": np.eye(2).tolist(),
                              "mu_q_H_cm2Vs": mu_H_for(np.eye(2), ratios)}

    ov_fn, bands, b, F_EFF, v, m_kg = RM.setup()
    fam = RT.families(b, F_EFF, fine=True)
    singles = []
    for name, (pname, grid, make) in fam.items():
        for p in grid:
            A, Bk = S.coupling_matrices(bands, make(p), RT.EPS_R, b,
                                        overlap_fn=ov_fn)
            W = sdh.dos_weights(A, Bk, v)
            singles.append({"mechanism": name, "parameter": float(p),
                            "W": W.tolist(),
                            "mu_q_H_cm2Vs": mu_H_for(W, ratios)})
    allv = [m for r in singles for m in r["mu_q_H_cm2Vs"]] + \
        out["no_intersubband"]["mu_q_H_cm2Vs"]
    out["single_mechanism_weights"] = singles
    out["range_over_all_weights_cm2Vs"] = [float(min(allv)), float(max(allv))]
    print("range over no-intersubband and all single mechanisms:",
          [round(v_) for v_ in out["range_over_all_weights_cm2Vs"]], flush=True)

    # self-consistent weights: roughness (scanned correlation length) plus the
    # line-charge (dislocation-form) component, fitted to the four mobilities
    def solve(A, Bk):
        tau = sdh.coupled_tau(A, Bk, v)
        tq = 1.0 / A.sum(1)
        return Q * tau / m_kg * 1e4, Q * tq / m_kg * 1e4
    rough = [(L, S.coupling_matrices(bands, fam["interface roughness"][2](L),
                                     RT.EPS_R, b, overlap_fn=ov_fn))
             for L in np.linspace(0.2, 3.0, 29)]
    disl = S.coupling_matrices(bands, fam["charged dislocations"][2](1.0),
                               RT.EPS_R, b, overlap_fn=ov_fn)
    muH = float(np.mean(MS.MU_Q_H_RANGE))
    history = []
    for it in range(10):
        meas = np.array([MS.MU_HALL_L, MS.MU_HALL_H, MS.MU_Q_L, muH])
        best = None
        for L, er in rough:
            def res(xx, er=er):
                a1, a2 = np.exp(xx)
                mt, mq = solve(a1 * er[0] + a2 * disl[0], a1 * er[1] + a2 * disl[1])
                return np.log(np.r_[mt, mq] / meas)
            s_ = min((least_squares(res, np.array(x0, float))
                      for x0 in [(0, 0), (2, 2), (-2, 2), (0, 4), (2, -2)]),
                     key=lambda z: z.cost)
            if best is None or s_.cost < best[0].cost:
                best = (s_, L, er)
        s_, L, er = best
        a1, a2 = np.exp(s_.x)
        A = a1 * er[0] + a2 * disl[0]
        Bk = a1 * er[1] + a2 * disl[1]
        W = sdh.dos_weights(A, Bk, v)
        mt, mq = solve(A, Bk)
        vals = mu_H_for(W, ratios)
        new = float(np.mean(vals))
        history.append({"target_mu_q_H": muH, "roughness_correlation_nm": float(L),
                        "roughness_rms_nm": float(0.3 * np.sqrt(a1)),
                        "line_charge_Nf2_cm2": float(1e4 * a2),
                        "worst_factor": float(np.exp(np.max(np.abs(s_.fun)))),
                        "mu_tr": mt.tolist(), "mu_q": mq.tolist(), "W": W.tolist(),
                        "mu_q_H_from_amplitudes": vals, "mean": new})
        print("iteration", it, "target %.1f -> %.1f" % (muH, new),
              "Lambda %.2f  worst %.3f" % (L, history[-1]["worst_factor"]), flush=True)
        if abs(new - muH) < 0.2:
            muH = new
            break
        muH = new
    out["self_consistent"] = {"history": history, "mu_q_H_cm2Vs": muH,
                              "spread_cm2Vs": [float(min(history[-1]["mu_q_H_from_amplitudes"])),
                                               float(max(history[-1]["mu_q_H_from_amplitudes"]))]}
    Wsc = np.array(history[-1]["W"])

    # 4. equal-visibility cross-check: the heavy-hole oscillation at its onset
    #    (50 to 60 T) as large as the light-hole one at its onset (25 T)
    vis = []
    for label, W in (("no_intersubband", np.eye(2)), ("self_consistent", Wsc)):
        for BH in (50.0, 55.0, 60.0):
            KL = sdh.sensitivities(25.0, N, MU_HALL) @ W
            AL = KL[0] * sdh.thermal_factor(MASS[0], 25.0, 1.8) * \
                np.exp(-np.pi / (MU_Q_L * 25.0))
            KH = sdh.sensitivities(BH, N, MU_HALL) @ W
            f = lambda mu: KH[1] * sdh.thermal_factor(MASS[1], BH, 1.8) * \
                np.exp(-np.pi / (mu * BH)) - AL
            vis.append({"weights": label, "heavy_onset_T": BH,
                        "mu_q_H_cm2Vs": 1e4 * brentq(f, 1e-4, 1.0)})
    out["equal_visibility"] = vis

    # 5. amplitude ratio implied by the conventional value
    conv = []
    for mu in MS.MU_Q_H_RANGE:
        for p in ratios["1p8K"]["points"]:
            B = p["B_T"]
            K = sdh.sensitivities(B, N, MU_HALL) @ Wsc
            pred = sdh.amplitude_ratio(B, 1.8, K, (MU_Q_L, mu * 1e-4), MASS)
            conv.append({"mu_q_H": mu, "B_T": B, "predicted": float(pred),
                         "measured": p["A_H_over_A_L"],
                         "factor": float(pred / p["A_H_over_A_L"])})
    out["conventional_value_check"] = conv

    # resulting heavy-hole ratio of transport to quantum mobility
    muH_sc = out["self_consistent"]["mu_q_H_cm2Vs"]
    lo, hi = out["range_over_all_weights_cm2Vs"]
    out["heavy_ratio"] = {"nominal": MS.MU_HALL_H / muH_sc,
                          "range": [MS.MU_HALL_H_RANGE[0] / hi, MS.MU_HALL_H_RANGE[1] / lo]}
    out["quantum_mobility_ratio_L_over_H"] = {"nominal": MS.MU_Q_L / muH_sc,
                                              "range": [MS.MU_Q_L / hi, MS.MU_Q_L / lo]}
    json.dump(out, open(os.path.join(RES, "heavy_quantum_mobility.json"), "w"), indent=1)
    print("self-consistent heavy-hole quantum mobility %.1f cm2/Vs" % muH_sc)
    print("WROTE results/heavy_quantum_mobility.json")


if __name__ == "__main__":
    main()
