"""The four mobilities with the heavy-hole quantum mobility from the oscillations.

Reads results/heavy_quantum_mobility.json (ratio route, first order, and the
raw envelope route) and results/forward_calibration.json (envelope route
calibrated, ratio route without the first-order approximation).  The adopted
heavy-hole quantum mobility is the mean of the three central values; its range
is the spread of the three.

Then, for the four measured mobilities (Hall 1900 and 400, quantum 368 and the
heavy-hole value):
  1. interface roughness plus a long-range component with a power-law
     spectrum |V(q)|^2 ~ q^-p: the exponent p that fits, as a function of the
     heavy-hole quantum mobility (the four mobilities fix p once mu_q,H is
     known);
  2. roughness plus one named mechanism (remote charge, the Mg-doped layer,
     polarization-charge fluctuations, long-range roughness, background
     impurities, threading line charges, in-plane misfit lines) at the adopted
     value and across a grid of heavy-hole values;
  3. roughness plus two named long-range mechanisms at the adopted value and
     at the ends of its range; misfit-line amplitudes are converted to the
     relaxed fraction of the GaN mismatch;
  4. every single mechanism, the quantum-mobility ratio each gives on its
     own, and the closest single mechanism in the plane of the two lifetime
     ratios;
  5. the trajectory of roughness plus the best power-law component across the
     plane of the two ratios (Fig. 3c).

Outputs results/revised_mobilities.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)

import disorder_fit as DF                                  # noqa: E402
from gan2dhg import measured as MS                         # noqa: E402

RES = os.path.join(HERE, '..', 'results')


def main():
    hq = json.load(open(os.path.join(RES, "heavy_quantum_mobility.json")))
    fc = json.load(open(os.path.join(RES, "forward_calibration.json")))
    est = {"ratio route, first order": hq["ratio_route"]["self_consistent"]["full_response_spin"]["mean"],
           "envelope route, calibrated": fc["envelope_calibrated"]["central"],
           "ratio route, non-perturbative": fc["ratio_nonperturbative"]["mean"]}
    muA = float(np.mean(list(est.values())))
    rng = [float(min(est.values())), float(max(est.values()))]
    out = {"estimates": est, "adopted_mu_q_H": muA, "adopted_range": rng}
    print("adopted mu_q,H %.1f (%.1f to %.1f)" % (muA, *rng), flush=True)

    lib = DF.library()
    rough = [e for e in lib["interface roughness"] if e[0] <= 2.01]
    rough3 = [e for e in lib["interface roughness"] if e[0] <= 3.01]

    # 1. power-law exponent against the heavy-hole value
    P = np.arange(2.8, 5.21, 0.05)
    plib = DF.power_library(P)
    grid = sorted(set([float(v) for v in (60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 130, 150,
                                          167, 185, 200)] + [round(muA, 1)] + [round(v, 1) for v in rng]))
    scan = {}
    for tH in grid:
        meas = DF.measured(tH)
        rows = []
        for pe in plib:
            r = DF.fit([[e1, pe] for e1 in rough], meas)
            rows.append({"p": pe[0], "worst": r["worst_factor"], "roughness_nm": r["parameters"][0],
                         "shares": r["shares_of_quantum_rate"][1]})
        b = min(rows, key=lambda z: z["worst"])
        ok = [z["p"] for z in rows if z["worst"] <= 1.05]
        scan[str(tH)] = {"best": b, "p_within_5_percent": [min(ok), max(ok)] if ok else None,
                         "rows": rows}
        print("mu_q,H %6.1f  best p %.2f  worst %.3f  p(5%%) %s" % (tH, b["p"], b["worst"],
                                                                scan[str(tH)]["p_within_5_percent"]),
              flush=True)
    out["exponent_scan"] = scan

    # 2. roughness plus one named mechanism
    rough_long = [e for e in lib["interface roughness"] if e[0] >= 4.0][::2]
    named = {"threading line charges": lib["charged dislocations"],
             "remote ionised charge": lib["remote ionised charge"][::2],
             "Mg-doped layer": lib["Mg-doped layer"],
             "polarization fluctuations": lib["polarization fluctuations"],
             "long-range roughness": rough_long,
             "background impurities": lib["background impurities"],
             "misfit lines": lib["misfit lines"]}
    libname = {"threading line charges": "charged dislocations", "long-range roughness": "interface roughness"}
    pairs = {}
    for tH in sorted(set([float(v) for v in (60, 70, 80, 90, 100, 115, 130, 150, 170, 200)]
                         + [round(muA, 1)])):
        pairs[str(tH)] = {}
        for name, L2 in named.items():
            r = DF.fit([[e1, e2] for e1 in rough3 for e2 in L2], DF.measured(tH))
            nm = libname.get(name, name)
            r = DF.strip(r)
            r["amplitude_physical"] = [list(DF.physical("interface roughness", r["amplitudes"][0])),
                                       list(DF.physical(nm, r["amplitudes"][1]))]
            if name == "misfit lines":
                r["relaxed_fraction_f1"] = DF.relaxation_from_misfit(r["amplitude_physical"][1][1])
            pairs[str(tH)][name] = r
        print("pairs at %.1f:" % tH, {k: round(v["worst_factor"], 3) for k, v in pairs[str(tH)].items()},
              flush=True)
    out["pairs"] = pairs

    # 3. roughness plus two named long-range mechanisms
    pol40 = [e for e in lib["polarization fluctuations"] if e[0] == 40.0]
    combos = {"misfit lines + threading line charges": (lib["misfit lines"], lib["charged dislocations"],
                                                        "misfit lines", "charged dislocations"),
              "Mg-doped layer + threading line charges": (lib["Mg-doped layer"], lib["charged dislocations"],
                                                          "Mg-doped layer", "charged dislocations"),
              "polarization fluctuations (40 nm) + threading line charges": (pol40, lib["charged dislocations"],
                                                                             "polarization fluctuations",
                                                                             "charged dislocations"),
              "Mg-doped layer + misfit lines": (lib["Mg-doped layer"], lib["misfit lines"],
                                                "Mg-doped layer", "misfit lines"),
              "polarization fluctuations (40 nm) + misfit lines": (pol40, lib["misfit lines"],
                                                                   "polarization fluctuations", "misfit lines")}
    three = {}
    for tH in sorted(set([round(muA, 1)] + [round(v, 1) for v in rng])):
        three[str(tH)] = {}
        for name, (L2, L3, n2, n3) in combos.items():
            r = DF.strip(DF.fit([[e1, e2, e3] for e1 in rough for e2 in L2 for e3 in L3], DF.measured(tH)))
            r["amplitude_physical"] = [list(DF.physical("interface roughness", r["amplitudes"][0])),
                                       list(DF.physical(n2, r["amplitudes"][1])),
                                       list(DF.physical(n3, r["amplitudes"][2]))]
            if n2 == "misfit lines":
                r["relaxed_fraction_f1"] = DF.relaxation_from_misfit(r["amplitude_physical"][1][1])
            if n3 == "misfit lines":
                r["relaxed_fraction_f1"] = DF.relaxation_from_misfit(r["amplitude_physical"][2][1])
            three[str(tH)][name] = r
        print("three at %.1f:" % tH, {k: round(v["worst_factor"], 3) for k, v in three[str(tH)].items()},
              flush=True)
    out["three_components"] = three

    # 4. single mechanisms
    singles = {}
    for name in ("interface roughness", "remote ionised charge", "charged dislocations",
                 "background impurities"):
        singles[name] = DF.strip(DF.fit([[e] for e in lib[name]], DF.measured(muA)))
    out["singles"] = singles
    alone = []
    for name in ("interface roughness", "remote ionised charge", "charged dislocations",
                 "background impurities", "Mg-doped layer", "polarization fluctuations", "misfit lines"):
        for p, (A, Bk) in lib[name]:
            mt, mq = DF.solve(A, Bk)
            alone.append({"mechanism": name, "parameter": p, "ratios": (mt / mq).tolist(),
                          "mu_q_L_over_mu_q_H": float(mq[0] / mq[1])})
    out["single_mechanism_ratios"] = alone
    tgt = np.array([MS.R_L, MS.MU_HALL_H / muA])
    best = min(((float(np.max(np.abs(np.log(np.array(a["ratios"]) / tgt)))), a) for a in alone
                if a["mechanism"] in ("interface roughness", "remote ionised charge",
                                      "charged dislocations", "background impurities")),
               key=lambda z: z[0])
    out["revised_ratio_target"] = tgt.tolist()
    out["closest_single_mechanism_ratio"] = {"factor": float(np.exp(best[0])), **best[1]}

    # 5. trajectory of roughness plus the best power-law component (Fig. 3c)
    b = scan[str(round(muA, 1))]["best"]
    e1 = dict(lib["interface roughness"])[b["roughness_nm"]]
    e2 = dict(DF.power_library([b["p"]]))[b["p"]]
    n1, n2 = e1[0].sum(1)[0], e2[0].sum(1)[0]
    traj = []
    for w in np.logspace(-4, 4, 161):
        mt, mq = DF.solve(e1[0] / n1 + w * e2[0] / n2, e1[1] / n1 + w * e2[1] / n2)
        traj.append({"weight": float(w), "ratios": (mt / mq).tolist()})
    out["best_trajectory"] = {"roughness_nm": b["roughness_nm"], "exponent": b["p"], "points": traj}
    json.dump(out, open(os.path.join(RES, "revised_mobilities.json"), "w"), indent=1, default=float)
    print("WROTE results/revised_mobilities.json")


if __name__ == "__main__":
    main()
