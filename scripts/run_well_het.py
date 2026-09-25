"""The finite-barrier self-consistent well at an offset of 0.7 eV, saved in full.

This is the solution shown in Figs. 2(a) and 2(b) of the Letter and the potential used by the
Landau-level, dispersion and overlap calculations.

Outputs results/well_het.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import kp6_het as H            # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    s = H.self_consistent_het(p_s_cm2=4.6e13, vbo_eV=0.7, L_bar=3.0,
                              dz=0.09375, n_kt=16, kt_max=2.4, max_iter=90,
                              tol=2e-5, mix=0.6, sparse=True)
    per = s["per_subband_nm2"]
    masses = []
    for b in range(len(per)):
        if per[b] <= 0:
            continue
        kF, m = H.mass_at_kf(s, b)
        masses.append({"subband": b, "n_cm2": float(per[b] * 1e14),
                       "kF_per_nm": kF, "m_CR": m,
                       "E_edge_meV": float(1000.0 * (s["E_of_k"][0, b]
                                                     - s["E_of_k"][0, 0]))})
    out = {"z": s["z"].tolist(), "V": s["V"].tolist(),
           "p_of_z": s["p_of_z"].tolist(), "kt": s["kt"].tolist(),
           "E_of_k": s["E_of_k"].tolist(), "EF": float(s["EF"]),
           "vbo_eV": 0.7, "converged": bool(s["converged"]),
           "offset_eV": s["prof"]["shift"].tolist(), "masses": masses}
    json.dump(out, open(os.path.join(RES, "well_het.json"), "w"))
    print("WROTE results/well_het.json", s["converged"],
          [round(m["m_CR"], 4) for m in masses])


if __name__ == "__main__":
    main()
