"""In-plane dispersion of the occupied subbands on a fine wavevector grid.

Used for Fig. 2(b) of the Letter (figures/prb_fig1, panel b).  The branches are followed by eigenvector continuity
(kp6_het.track_branches), in the converged potential of results/well_het.json.

The same is done for the polarization-closure well (results/well_het_pol.json)
to show how the states that extend across the GaN layer depend on the
potential beyond the gas: their energy at k = 0 relative to the ground subband
edge, and the light-hole mass hbar^2 k / (dE/dk) along the tracked branch.

Outputs results/dispersion.json.
"""

import json
import os
import sys

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import kp6_het as H            # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


HB2_2M0 = 0.0380998212       # eV nm^2


def tracked(fname, k):
    d = json.load(open(os.path.join(RES, fname)))
    z, V = np.array(d["z"]), np.array(d["V"])
    prof = H.profile(z, d["vbo_eV"])

    def at(kk):
        w, v = eigsh(sps.csc_matrix(H.build_operator(kk, 0.0, z, V, prof)),
                     k=10, sigma=-0.2, which="LM")
        o = np.argsort(w)
        return w[o], v[:, o]

    E = H.track_branches(k, at, n_states=10)
    w0 = at(0.0)[0]
    return d, E, w0


def main():
    k = np.linspace(1e-3, 2.05, 83)
    d, E, _ = tracked("well_het.json", k)
    out = {"k_per_nm": k.tolist(), "E_eV": E[:, :4].T.tolist(),
           "E0_eV": float(E[0, 0]), "EF_eV": d["EF"], "vbo_eV": d["vbo_eV"]}

    kk = np.linspace(0.40, 0.95, 56)
    for fname in ("well_het.json", "well_het_pol.json"):
        _, Ek, w0 = tracked(fname, kk)
        m = HB2_2M0 * 2.0 * kk / np.gradient(Ek[:, 2], kk)
        out["closure_" + fname] = {
            "k_per_nm": kk.tolist(), "light_mass": m.tolist(),
            "levels_at_k0_meV": (1e3 * (w0 - w0[0])).tolist()}
        print(fname, "third level at k=0:", round(1e3 * (w0[4] - w0[0]), 1),
              "meV; light mass at 0.55, 0.70, 0.95:",
              [round(float(np.interp(x, kk, m)), 3) for x in (0.55, 0.70, 0.95)])
    json.dump(out, open(os.path.join(RES, "dispersion.json"), "w"))
    print("WROTE results/dispersion.json")


if __name__ == "__main__":
    main()
