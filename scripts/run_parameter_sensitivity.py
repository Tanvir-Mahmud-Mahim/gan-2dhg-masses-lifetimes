"""Which band parameter controls the light-hole subband?

Each parameter of the GaN set (Extended Data Table 1 of Chang et al.) that
enters the unstrained or strained valence bands is scaled by 0.8 and by 1.2 in
turn, and the full self-consistent solution with the finite barrier is repeated
at the measured sheet density and an offset of 0.7 eV.  Reported for each case:
the heavy- and light-hole masses at their own Fermi wavevectors, the light-hole
occupation, the light-heavy separation and whether the light pair is occupied.

The question is whether any parameter moves the light subband (mass and
occupation) while leaving the heavy subband and the separation unchanged.

Outputs results/parameter_sensitivity.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import kp6_het as H             # noqa: E402
from gan2dhg.kp6 import GAN                  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
KW = dict(L_bar=3.0, dz=0.09375, n_kt=16, kt_max=2.4, max_iter=150, tol=2e-5,
          mix=0.4, sparse=True, n_states=8)
KEYS = ("A1", "A2", "A3", "A4", "A5", "A6", "D_CR", "D_SO", "D3v", "D4v")


def summarise(sol, label):
    per = sol["per_subband_nm2"]
    bands = []
    for b in range(len(per)):
        if per[b] > 0:
            kF, m = H.mass_at_kf(sol, b, n_states=8)
            bands.append({"index": b, "n_cm2": float(per[b] * 1e14), "m": m,
                          "edge_meV": float(1000 * (sol["E_of_k"][0, b]
                                                    - sol["E_of_k"][0, 0]))})
    rec = {"case": label, "converged": bool(sol["converged"]),
           "occupied_branches": len(bands), "bands": bands}
    if len(bands) == 4:
        rec.update(m_hh=0.5 * (bands[0]["m"] + bands[1]["m"]),
                   m_lh=0.5 * (bands[2]["m"] + bands[3]["m"]),
                   p_lh_1e13=(bands[2]["n_cm2"] + bands[3]["n_cm2"]) / 1e13,
                   Delta_meV=bands[2]["edge_meV"])
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in rec.items() if k != "bands"}), flush=True)
    return rec


def main():
    out = [summarise(H.self_consistent_het(p_s_cm2=4.6e13, vbo_eV=0.7, **KW),
                     "published set")]
    for key in KEYS:
        for f in (0.8, 1.2):
            gan = dict(GAN)
            gan[key] = GAN[key] * f
            sol = H.self_consistent_het(p_s_cm2=4.6e13, vbo_eV=0.7, gan=gan,
                                        **KW)
            out.append(summarise(sol, f"{key} x {f}"))
            json.dump(out, open(os.path.join(RES, "parameter_sensitivity.json"),
                                "w"), indent=1)
    print("WROTE results/parameter_sensitivity.json")


if __name__ == "__main__":
    main()
