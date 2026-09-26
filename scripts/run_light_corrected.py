"""The four-mobility fits with the light-hole quantum mobility corrected like the
heavy-hole one.

The heavy-hole value adopted in results/revised_mobilities.json is the quantum
mobility of the non-perturbative model that reproduces the measured
oscillations.  The light-hole value used there is the reported Dingle
mobility, 368 cm^2/Vs.  In the same model the analysis of the light-hole
oscillation returns 368 only if the light-hole quantum mobility is about 415
(results/forward_calibration.json, light_input).  This script repeats the
exponent scan and the fits with two and three components at the adopted
heavy-hole value with that corrected light-hole value, to show how much the
conclusions depend on it.

Outputs results/light_corrected_fits.json.
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
    fc = json.load(open(os.path.join(RES, "forward_calibration.json")))
    rv = json.load(open(os.path.join(RES, "revised_mobilities.json")))
    muL = fc["light_input"]["mu_q_L_in"]
    muH = rv["adopted_mu_q_H"]
    meas = np.array([MS.MU_HALL_L, MS.MU_HALL_H, muL, muH])
    out = {"mu_q_L": muL, "mu_q_H": muH}
    lib = DF.library()
    rough = [e for e in lib["interface roughness"] if e[0] <= 2.01]
    rough3 = [e for e in lib["interface roughness"] if e[0] <= 3.01]
    plib = DF.power_library(np.arange(2.8, 5.21, 0.05))
    rows = []
    for pe in plib:
        r = DF.fit([[e1, pe] for e1 in rough], meas)
        rows.append({"p": pe[0], "worst": r["worst_factor"], "roughness_nm": r["parameters"][0]})
    b = min(rows, key=lambda z: z["worst"])
    ok = [z["p"] for z in rows if z["worst"] <= 1.05]
    out["exponent"] = {"best": b, "p_within_5_percent": [min(ok), max(ok)] if ok else None}
    print("light %.0f heavy %.1f: best p %.2f worst %.3f  p(5%%) %s" % (muL, muH, b["p"], b["worst"],
                                                                    out["exponent"]["p_within_5_percent"]))
    pairs = {}
    for name in ("charged dislocations", "misfit lines", "Mg-doped layer", "polarization fluctuations",
                 "remote ionised charge"):
        L2 = lib[name] if name != "remote ionised charge" else lib[name][::2]
        r = DF.strip(DF.fit([[e1, e2] for e1 in rough3 for e2 in L2], meas))
        r["amplitude_physical"] = [list(DF.physical("interface roughness", r["amplitudes"][0])),
                                   list(DF.physical(name, r["amplitudes"][1]))]
        pairs[name] = r
    out["pairs"] = pairs
    print({k: round(v["worst_factor"], 3) for k, v in pairs.items()})
    pol40 = [e for e in lib["polarization fluctuations"] if e[0] == 40.0]
    three = {}
    for lbl, (n2, L2, n3, L3) in {
            "misfit lines + threading line charges": ("misfit lines", lib["misfit lines"],
                                                      "charged dislocations", lib["charged dislocations"]),
            "Mg-doped layer + threading line charges": ("Mg-doped layer", lib["Mg-doped layer"],
                                                        "charged dislocations", lib["charged dislocations"]),
            "polarization fluctuations (40 nm) + threading line charges": (
                "polarization fluctuations", pol40, "charged dislocations", lib["charged dislocations"]),
            "Mg-doped layer + misfit lines": ("Mg-doped layer", lib["Mg-doped layer"],
                                              "misfit lines", lib["misfit lines"])}.items():
        r = DF.strip(DF.fit([[e1, e2, e3] for e1 in rough for e2 in L2 for e3 in L3], meas))
        r["amplitude_physical"] = [list(DF.physical("interface roughness", r["amplitudes"][0])),
                                   list(DF.physical(n2, r["amplitudes"][1])),
                                   list(DF.physical(n3, r["amplitudes"][2]))]
        for k, n in ((1, n2), (2, n3)):
            if n == "misfit lines":
                r["relaxed_fraction_f1"] = DF.relaxation_from_misfit(r["amplitude_physical"][k][1])
        three[lbl] = r
    out["three_components"] = three
    print({k: round(v["worst_factor"], 3) for k, v in three.items()})
    json.dump(out, open(os.path.join(RES, "light_corrected_fits.json"), "w"), indent=1, default=float)
    print("WROTE results/light_corrected_fits.json")


if __name__ == "__main__":
    main()
