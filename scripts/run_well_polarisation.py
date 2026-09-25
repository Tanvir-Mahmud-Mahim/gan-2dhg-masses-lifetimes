"""Self-consistent well with the interface charge from the calculated polarization.

The Letter closes the electrostatics at the measured sheet density, so that
the field vanishes beyond the gas (results/well_het.json).  This script saves
the alternative closure in full, with the bound charge at the interface set to
the spontaneous plus piezoelectric polarization discontinuity
(run_strain_polarisation.sigma_polarisation) and the excess over the measured
hole density compensated far from the interface.  Beyond the gas the valence
band then continues to bend away from the Fermi level, instead of lying flat a
few millielectronvolts below it, and the states that extend across the GaN
layer are pushed far from the Fermi level.  The Landau-level calculation is
repeated on this potential as a test of its sensitivity to the closure.

Outputs results/well_het_pol.json, in the format of results/well_het.json.
"""

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from gan2dhg import kp6_het as H             # noqa: E402

spec = importlib.util.spec_from_file_location(
    "sp", os.path.join(HERE, "run_strain_polarisation.py"))
SP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SP)

RES = os.path.join(HERE, '..', 'results')


def main():
    st = SP.strain_state(0.0)
    sig, _ = SP.sigma_polarisation(st[0])
    s = H.self_consistent_het(p_s_cm2=4.6e13, vbo_eV=0.7, strain=st,
                              sigma_cm2=sig, **SP.KW)
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
           "sigma_cm2": float(sig), "masses": masses}
    json.dump(out, open(os.path.join(RES, "well_het_pol.json"), "w"))
    print("WROTE results/well_het_pol.json", s["converged"],
          [round(m["m_CR"], 3) for m in masses])


if __name__ == "__main__":
    main()
