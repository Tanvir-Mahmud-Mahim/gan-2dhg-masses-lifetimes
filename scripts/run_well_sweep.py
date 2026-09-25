"""Masses and occupations against sheet density, with the finite barrier.

The density dependence behind the prediction of Fig. 2(c) of the Letter.  Each point is a
full self-consistent solution with the finite AlN barrier, at two valence band
offsets within the published range, 0.3 and 0.7 eV.  Masses are local
derivatives at each branch's own Fermi wavevector (kp6_het.mass_at_kf).

Outputs results/well_sweep.json.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import kp6_het as H            # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
KW = dict(L_bar=3.0, dz=0.09375, n_kt=16, kt_max=2.4, max_iter=120,
          tol=2e-5, mix=0.5, sparse=True)


def main():
    out = []
    t0 = time.time()
    for vbo in (0.7, 0.3):
        for ns in (2.0e13, 3.0e13, 4.0e13, 4.6e13, 5.5e13, 6.5e13):
            s = H.self_consistent_het(p_s_cm2=ns, vbo_eV=vbo, **KW)
            z, p = s["z"], s["p_of_z"]
            rec = {"vbo_eV": vbo, "p_s_cm2": ns, "converged": s["converged"],
                   "iterations": s["iterations"],
                   "centroid_nm": float(np.trapezoid(z * p, z)
                                        / np.trapezoid(p, z)),
                   "EF_minus_E0_meV": 1000.0 * (s["EF"] - s["E_of_k"][0, 0]),
                   "bands": []}
            per = s["per_subband_nm2"]
            for b in range(len(per)):
                if per[b] <= 0:
                    continue
                kF, m = H.mass_at_kf(s, b)
                rec["bands"].append({
                    "index": b, "n_cm2": float(per[b] * 1e14), "kF": kF,
                    "m": m, "edge_meV": float(1000.0 * (
                        s["E_of_k"][0, b] - s["E_of_k"][0, 0]))})
            out.append(rec)
            print(f"vbo {vbo} ns {ns:.1e} conv {s['converged']} "
                  f"masses {[round(b['m'], 3) for b in rec['bands']]} "
                  f"t {time.time() - t0:.0f}s", flush=True)
            json.dump(out, open(os.path.join(RES, "well_sweep.json"), "w"),
                      indent=1)
    print("WROTE results/well_sweep.json")


if __name__ == "__main__":
    main()
