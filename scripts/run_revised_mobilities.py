"""The four mobilities with the heavy-hole quantum mobility from the amplitudes.

results/heavy_quantum_mobility.json gives the heavy-hole quantum mobility that
the measured amplitude ratio of the two oscillations implies.  The value
depends on the density-of-states weights W of the disorder, so each disorder
model is tested self-consistently: its two amplitudes are fitted to the four
mobilities (Hall 1900 and 400, quantum 368 and mu_H), its weights W are
computed (src/gan2dhg/sdh.py), mu_H is recomputed from the amplitudes with
these weights, and the loop is repeated until mu_H no longer changes.

Models: every single mechanism of run_tension.families; every pair of them;
and short-range roughness together with two candidate sources of long-range
potential in this structure,
  - the ionized acceptors of the Mg-doped top of the GaN layer, a uniform
    distribution 10 to 15 nm from the interface (Methods of Chang et al.: a
    15 nm GaN layer whose last 5 nm is heavily doped with Mg),
  - fluctuations of the interface polarization charge with a Gaussian
    correlation length xi (scatter2d.w_polarization_fluctuation).

Also reported: the ratio of the two quantum mobilities that each mechanism
gives on its own, the closest single mechanism to the revised ratios, and the
trajectory of the best pair across the plane of the two ratios (Fig. 3c).

Outputs results/revised_mobilities.json.
"""

import itertools
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import run_mixtures as RM                                  # noqa: E402
import run_tension as RT                                   # noqa: E402
import run_heavy_quantum_mobility as HQ                    # noqa: E402
from gan2dhg import measured as MS, scatter2d as S, sdh    # noqa: E402
from gan2dhg.constants import Q                            # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
BASE = {"interface roughness": ("rms height (nm)", 0.3),
        "remote ionised charge": ("sheet density (cm^-2)", 5e12),
        "charged dislocations": ("N f^2 (cm^-2)", 1e4),
        "background impurities": ("volume density (cm^-3)", 1e17),
        "Mg-doped layer": ("ionized acceptors (cm^-2)", 5e12),
        "polarization fluctuations": ("rms of the interface charge (cm^-2)", 1e11)}
MG_RANGE_NM = (10.0, 15.0)


def phys(name, a):
    label, base = BASE[name]
    if name in ("interface roughness", "polarization fluctuations"):
        return label, base * np.sqrt(a)
    return label, base * a


def main():
    ov_fn, bands, b, F_EFF, v, m_kg = RM.setup()
    fam = RT.families(b, F_EFF, fine=True)
    D = json.load(open(HQ.DATA))
    ratios = HQ.measured_ratios(D)
    hq = json.load(open(os.path.join(RES, "heavy_quantum_mobility.json")))
    muH0 = hq["self_consistent"]["mu_q_H_cm2Vs"]

    def solve(A, Bk):
        tau = sdh.coupled_tau(A, Bk, v)
        tq = 1.0 / A.sum(1)
        return tau / tq, Q * tau / m_kg * 1e4, Q * tq / m_kg * 1e4

    def cm(w):
        return S.coupling_matrices(bands, w, RT.EPS_R, b, overlap_fn=ov_fn)

    grids = {"interface roughness": np.linspace(0.2, 12, 60),
             "remote ionised charge": np.linspace(0.2, 30, 60),
             "charged dislocations": [1.0], "background impurities": [1.0]}
    lib = {k: [(float(p), cm(fam[k][2](p))) for p in g] for k, g in grids.items()}
    # Mg-doped layer: 12 sheets spread uniformly over the layer, total 5e12 cm^-2
    ds = np.linspace(*MG_RANGE_NM, 12)
    mg = [cm(lambda q, d=d: S.w_remote_impurity(q, 5e12 / len(ds), d * 1e-9, b,
                                                RT.EPS_R)) for d in ds]
    lib["Mg-doped layer"] = [(0.0, (sum(m[0] for m in mg), sum(m[1] for m in mg)))]
    lib["polarization fluctuations"] = [
        (float(xi), cm(lambda q, xi=xi: S.w_polarization_fluctuation(
            q, 1e11, xi * 1e-9, b, RT.EPS_R))) for xi in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)]

    out = {"mu_q_H_start": muH0}

    # quantum-mobility ratio of each mechanism on its own
    alone = []
    for name, L in lib.items():
        for p, (A, Bk) in L:
            r, mt, mq = solve(A, Bk)
            alone.append({"mechanism": name, "parameter": p, "ratios": r.tolist(),
                          "mu_q_L_over_mu_q_H": float(mq[0] / mq[1])})
    out["single_mechanism_quantum_ratio"] = alone
    base = [a["mu_q_L_over_mu_q_H"] for a in alone
            if a["mechanism"] in grids]
    out["quantum_ratio_range_over_scans"] = [float(min(base)), float(max(base))]

    def fit(entries, target):
        """Best fit of the amplitudes over the parameter grids."""
        meas = np.array([MS.MU_HALL_L, MS.MU_HALL_H, MS.MU_Q_L, target])
        best = None
        for combo in entries:
            mats = [c[1] for c in combo]

            def res(x):
                a = np.exp(x)
                A = sum(ai * m[0] for ai, m in zip(a, mats))
                Bk = sum(ai * m[1] for ai, m in zip(a, mats))
                _, mt, mq = solve(A, Bk)
                return np.log(np.r_[mt, mq] / meas)
            starts = [(0,), (2,), (-2,), (4,), (-4,)] if len(mats) == 1 else \
                [(0, 0), (2, -2), (-2, 2), (4, 0), (0, 4)]
            s_ = min((least_squares(res, np.array(x0, float)) for x0 in starts),
                     key=lambda z: z.cost)
            if best is None or s_.cost < best[0].cost:
                best = (s_, combo)
        return best

    def selfconsistent(names, entries):
        muH = muH0
        for it in range(8):
            s_, combo = fit(entries, muH)
            a = np.exp(s_.x)
            A = sum(ai * c[1][0] for ai, c in zip(a, combo))
            Bk = sum(ai * c[1][1] for ai, c in zip(a, combo))
            W = sdh.dos_weights(A, Bk, v)
            new = float(np.mean(HQ.mu_H_for(W, ratios)))
            if abs(new - muH) < 0.5:
                muH = new
                break
            muH = new
        s_, combo = fit(entries, muH)
        a = np.exp(s_.x)
        A = sum(ai * c[1][0] for ai, c in zip(a, combo))
        Bk = sum(ai * c[1][1] for ai, c in zip(a, combo))
        r, mt, mq = solve(A, Bk)
        parts = []
        for ai, c, n in zip(a, combo, names):
            _, mta, mqa = solve(ai * c[1][0], ai * c[1][1])
            parts.append({"mechanism": n, "parameter": c[0],
                          "amplitude": list(phys(n, ai)),
                          "mu_tr_alone": mta.tolist(), "mu_q_alone": mqa.tolist(),
                          "share_of_quantum_rate": ((ai * c[1][0]).sum(1) / A.sum(1)).tolist()})
        return {"mechanisms": list(names), "mu_q_H": muH,
                "worst_factor": float(np.exp(np.max(np.abs(s_.fun)))),
                "mu_tr": mt.tolist(), "mu_q": mq.tolist(), "ratios": r.tolist(),
                "W": sdh.dos_weights(A, Bk, v).tolist(), "components": parts}

    singles = []
    for n in grids:
        res_ = selfconsistent([n], [[e] for e in lib[n]])
        singles.append(res_)
        print("single", n, "mu_H %.1f worst %.3f" % (res_["mu_q_H"], res_["worst_factor"]),
              flush=True)
    out["singles"] = sorted(singles, key=lambda z: z["worst_factor"])

    pairs = []
    names = list(grids) + ["Mg-doped layer", "polarization fluctuations"]
    for n1, n2 in itertools.combinations_with_replacement(names, 2):
        if n1 == n2 and n1 not in ("interface roughness", "remote ionised charge"):
            continue
        if n1 in ("Mg-doped layer", "polarization fluctuations") or \
                n2 in ("Mg-doped layer", "polarization fluctuations"):
            if "interface roughness" not in (n1, n2):
                continue
        entries = [[e1, e2] for i, e1 in enumerate(lib[n1]) for j, e2 in enumerate(lib[n2])
                   if not (n1 == n2 and j <= i)]
        res_ = selfconsistent([n1, n2], entries)
        pairs.append(res_)
        print("pair", n1, "+", n2, "mu_H %.1f worst %.3f" % (res_["mu_q_H"], res_["worst_factor"]),
              [ (p["mechanism"], "%.3g" % p["amplitude"][1]) for p in res_["components"]], flush=True)
    out["pairs"] = sorted(pairs, key=lambda z: z["worst_factor"])

    # closest single mechanism to the revised ratios (light, heavy)
    rH = MS.MU_HALL_H / out["pairs"][0]["mu_q_H"]
    tgt = np.array([MS.R_L, rH])
    best = min(((float(np.max(np.abs(np.log(np.array(a["ratios"]) / tgt)))), a)
                for a in alone if a["mechanism"] in grids), key=lambda z: z[0])
    out["revised_ratio_target"] = tgt.tolist()
    out["closest_single_mechanism_ratio"] = {"factor": float(np.exp(best[0])),
                                             "mechanism": best[1]["mechanism"],
                                             "parameter": best[1]["parameter"],
                                             "ratios": best[1]["ratios"]}

    # trajectory of the best pair as its relative weight varies (Fig. 3c)
    top = out["pairs"][0]
    c1, c2 = top["components"]
    e1 = dict(lib[c1["mechanism"]])[c1["parameter"]]
    e2 = dict(lib[c2["mechanism"]])[c2["parameter"]]
    n1, n2 = e1[0].sum(1)[0], e2[0].sum(1)[0]
    traj = []
    for w in np.logspace(-4, 4, 161):
        r, _, _ = solve(e1[0] / n1 + w * e2[0] / n2, e1[1] / n1 + w * e2[1] / n2)
        traj.append({"weight": float(w), "ratios": r.tolist()})
    out["best_pair_trajectory"] = {"mechanisms": [c1["mechanism"], c2["mechanism"]],
                                   "parameters": [c1["parameter"], c2["parameter"]],
                                   "points": traj}
    json.dump(out, open(os.path.join(RES, "revised_mobilities.json"), "w"), indent=1)
    print("WROTE results/revised_mobilities.json")


if __name__ == "__main__":
    main()
