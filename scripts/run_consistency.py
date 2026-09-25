"""Are the two measured subband densities compatible with the computed dispersions?

The oscillation frequencies fix the Fermi wavevector of each subband
independently of any mass: k_F = (2 pi n)^(1/2) with n = 0.80 and
3.80 x 10^13 cm^-2 (spin-degenerate pairs).  In equilibrium both pockets end at
one Fermi level, so E_heavy(k_F,H) and E_light(k_F,L), measured from a common
reference, must be equal.  This script evaluates both with the computed
subband dispersions of the finite-barrier solution, for the two electrostatic
closures and the two ends of the offset range, and reports the mismatch.

It also reports the average ("secant") light-hole mass that would remove the
mismatch with the computed subband edges held fixed,

    m_secant = hbar^2 k_F,L^2 / (2 [E_heavy(k_F,H) - E_light(0)]),

and compares it with the secant mass of the computed light-hole branch.  The
light-heavy separation at k = 0 is reported to show that it does not depend on
the closure or the offset.

For parabolic subbands the same condition reads
    Delta = pi hbar^2 (n_H / m_H - n_L / m_L),
which with the measured densities and heavy mass gives the light mass for
which the computed Delta is recovered; this is evaluated in closed form.

Outputs results/consistency.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from gan2dhg import kp6_het as H                     # noqa: E402
from gan2dhg.kp6 import HB2_2M0_eVnm2                # noqa: E402
import run_strain_polarisation as SP                 # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
N_L, N_H = 0.080, 0.380          # nm^-2, measured pairs
M_H = 1.92
KF_L, KF_H = np.sqrt(2 * np.pi * N_L), np.sqrt(2 * np.pi * N_H)
KW = dict(L_bar=3.0, dz=0.09375, n_kt=16, kt_max=2.4, max_iter=150, tol=2e-5,
          mix=0.4, sparse=True)


def pair_energy(sol, k, pair):
    w = H.solve(k, 0.0, sol["z"], sol["V"], sol["prof"], n_states=6,
                sparse=True)[0]
    return float(np.mean(w[2 * pair:2 * pair + 2]))


def analyse(label, sol):
    e_h0 = pair_energy(sol, 1e-3, 0)
    e_l0 = pair_energy(sol, 1e-3, 1)
    e_h = pair_energy(sol, KF_H, 0)
    e_l = pair_energy(sol, KF_L, 1)
    rec = {"case": label,
           "Delta_meV": 1000 * (e_l0 - e_h0),
           "E_heavy_at_kFH_meV": 1000 * (e_h - e_h0),
           "E_light_at_kFL_meV": 1000 * (e_l - e_h0),
           "mismatch_meV": 1000 * (e_l - e_h),
           "light_secant_mass_required": HB2_2M0_eVnm2 * KF_L ** 2 / (e_h - e_l0),
           "light_secant_mass_computed": HB2_2M0_eVnm2 * KF_L ** 2 / (e_l - e_l0)}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in rec.items()}), flush=True)
    return rec


def parabolic_delta(m_l):
    """pi hbar^2 (n_H/m_H - n_L/m_L) in meV (hbar^2/m0 = 2 HB2_2M0)."""
    return 1000 * np.pi * 2 * HB2_2M0_eVnm2 * (N_H / M_H - N_L / m_l)


def main():
    sig_b, _ = SP.sigma_polarisation(SP.strain_state(0.0)[0])
    out = {"kF_light_per_nm": KF_L, "kF_heavy_per_nm": KF_H, "cases": []}
    for vbo in (0.7, 0.3):
        for name, sig in (("closure A (field vanishes beyond the gas)", None),
                          ("closure B (interface charge from polarization)",
                           sig_b)):
            sol = H.self_consistent_het(p_s_cm2=4.6e13, vbo_eV=vbo,
                                        sigma_cm2=sig, **KW)
            out["cases"].append(analyse(f"{name}, offset {vbo} eV", sol))
    out["parabolic"] = {
        "formula": "Delta = pi hbar^2 (n_H/m_H - n_L/m_L)",
        "Delta_meV_for_m_L": {str(m): parabolic_delta(m)
                              for m in (0.30, 0.46, 0.53, 0.57)}}
    print(out["parabolic"])
    json.dump(out, open(os.path.join(RES, "consistency.json"), "w"), indent=1)
    print("WROTE results/consistency.json")


if __name__ == "__main__":
    main()
