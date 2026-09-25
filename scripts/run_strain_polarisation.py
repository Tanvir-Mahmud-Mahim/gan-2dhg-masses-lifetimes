"""Piezoelectric polarisation, strain relaxation, and the subband occupations.

WHERE PIEZOELECTRICITY ENTERS
-----------------------------
GaN grown pseudomorphically on relaxed AlN is under biaxial compression, and
its polarisation is the sum of the spontaneous part and the piezoelectric part
(Bernardini, Fiorentini and Vanderbilt, Phys. Rev. B 56, R10024 (1997)),

    P_pz = e33 eps_zz + e31 (eps_xx + eps_yy)
         = 2 eps_xx (e31 - e33 C13 / C33).

The bound sheet charge at the interface is the discontinuity of the total
polarisation (Ambacher et al., J. Appl. Phys. 85, 3222 (1999)),

    sigma = P_sp(AlN) - [P_sp(GaN) + P_pz(GaN)],

negative for this metal-polar structure, and it is this charge that binds the
hole gas.  Strain enters the calculation in two further places: through the
deformation potentials in the six-band Hamiltonian, which set the splitting
between the heavy and light bands, and through sigma.

In the Letter the electrostatics are closed at the MEASURED sheet density:
the interface charge that the gas balances is set equal to p_s, so the field
vanishes beyond the gas.  That choice makes the result independent of the
polarisation constants, which are known less well than the density.  This
script measures what the alternative costs: it computes sigma including the
piezoelectric term, places the difference sigma - p_s far from the interface
(so that a residual field extends across the rest of the GaN), and repeats
the self-consistent solution.  It then scans the strain state of the GaN from
pseudomorphic towards relaxed, changing both the deformation-potential terms
and sigma, at fixed measured sheet density.

The polarisation constants are those of Bernardini et al. (Table II):
P_sp = -0.081 and -0.029 C/m^2, e33 = 1.46 and 0.73 C/m^2, e31 = -0.60 and
-0.49 C/m^2 for AlN and GaN.  Elastic constants are those used everywhere
else in this work (Extended Data Table 1 of Chang et al.).

Outputs results/strain_polarisation.json.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import kp6_het as H            # noqa: E402
from gan2dhg.kp6 import ALN, GAN             # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
QE = 1.602176634e-19

PSP_ALN, PSP_GAN = -0.081, -0.029     # C/m^2
E33_GAN, E31_GAN = 0.73, -0.49        # C/m^2

P_S = 4.6e13
KW = dict(L_bar=3.0, dz=0.09375, n_kt=16, kt_max=2.4, max_iter=120,
          tol=2e-5, mix=0.5, sparse=True)


def strain_state(r):
    """Biaxial strain of GaN with a fraction r of the mismatch relaxed."""
    a_par = ALN["a_A"] + r * (GAN["a_A"] - ALN["a_A"])
    exx = (a_par - GAN["a_A"]) / GAN["a_A"]
    ezz = -2.0 * GAN["C13"] / GAN["C33"] * exx
    return exx, exx, ezz


def sigma_polarisation(exx):
    """Interface bound charge, as a density of holes it can bind (cm^-2)."""
    ppz = 2.0 * exx * (E31_GAN - E33_GAN * GAN["C13"] / GAN["C33"])
    sigma = PSP_ALN - (PSP_GAN + ppz)          # C/m^2, negative
    return -sigma / QE * 1e-4, ppz


def summarise(sol):
    per = sol["per_subband_nm2"]
    occ = [s for s in range(len(per)) if per[s] > 0]
    z, p = sol["z"], sol["p_of_z"]
    rec = {"converged": sol["converged"], "iterations": sol["iterations"],
           "centroid_nm": float(np.trapezoid(z * p, z) / np.trapezoid(p, z)),
           "EF_minus_E0_meV": 1000.0 * (sol["EF"] - sol["E_of_k"][0, 0]),
           "branches": []}
    for s in occ:
        kF, m = H.mass_at_kf(sol, s)
        rec["branches"].append({"index": s, "n_cm2": float(per[s] * 1e14),
                                "m_CR": m,
                                "edge_meV": float(1000.0 * (
                                    sol["E_of_k"][0, s] - sol["E_of_k"][0, 0]))})
    br = rec["branches"]
    heavy = [x for x in br if x["index"] in (0, 1)]
    light = [x for x in br if x["index"] in (2, 3)]
    rec["m_heavy"] = float(np.mean([x["m_CR"] for x in heavy]))
    rec["m_light"] = float(np.mean([x["m_CR"] for x in light])) if light else None
    rec["p_light_cm2"] = float(sum(x["n_cm2"] for x in light))
    rec["p_heavy_cm2"] = float(sum(x["n_cm2"] for x in heavy))
    rec["separation_meV"] = float(np.mean([x["edge_meV"] for x in light])) \
        if light else None
    rec["third_pair_occupied"] = bool(any(x["index"] >= 4 for x in br))
    return rec


def main():
    t0 = time.time()
    out = {"constants": {"Psp_AlN": PSP_ALN, "Psp_GaN": PSP_GAN,
                         "e33_GaN": E33_GAN, "e31_GaN": E31_GAN,
                         "source": "Bernardini, Fiorentini and Vanderbilt, "
                                   "PRB 56, R10024 (1997), Table II"}}
    exx0 = strain_state(0.0)[0]
    sig0, ppz0 = sigma_polarisation(exx0)
    out["pseudomorphic"] = {
        "exx": exx0, "P_pz_C_m2": ppz0,
        "sigma_total_cm2": sig0,
        "sigma_spontaneous_only_cm2": -(PSP_ALN - PSP_GAN) / QE * 1e-4,
        "piezo_share_of_sigma": float(ppz0 / (PSP_GAN + ppz0 - PSP_ALN)),
        "measured_p_s_cm2": P_S}
    print(json.dumps(out["pseudomorphic"], indent=1), flush=True)

    rows = []
    for vbo in (0.7,):
        for r in (0.0, 0.1, 0.25, 0.5):
            st = strain_state(r)
            sig, ppz = sigma_polarisation(st[0])
            for closure in ("field vanishes beyond gas",
                            "interface charge from polarisation"):
                s_cm2 = None if closure.startswith("field") else sig
                if s_cm2 is not None and s_cm2 < P_S:
                    # The calculated polarization charge is smaller than the
                    # measured hole density: it cannot bind the gas, and this
                    # strain state is incompatible with the measurement for
                    # these polarization constants.
                    rows.append({"vbo_eV": vbo, "relaxed_fraction": r,
                                 "exx": st[0], "ezz": st[2],
                                 "closure": closure, "sigma_cm2": sig,
                                 "bound": False})
                    print(f"vbo {vbo} r {r:.2f} polarization charge "
                          f"{sig:.3e} below p_s: gas not bound", flush=True)
                    continue
                sol = H.self_consistent_het(p_s_cm2=P_S, vbo_eV=vbo,
                                            strain=st, sigma_cm2=s_cm2, **KW)
                rec = summarise(sol)
                rec.update({"vbo_eV": vbo, "relaxed_fraction": r, "bound": True,
                            "exx": st[0], "ezz": st[2], "closure": closure,
                            "sigma_cm2": sig if s_cm2 else P_S,
                            "interface_field_MV_cm":
                                (sig if s_cm2 else P_S) * 1e4 * QE
                                / (10.4 * 8.8541878128e-12) * 1e-8})
                rows.append(rec)
                print(f"vbo {vbo} r {r:.2f} {closure[:18]:18s} conv "
                      f"{rec['converged']} mh {rec['m_heavy']:.3f} ml "
                      f"{rec['m_light']} pl {rec['p_light_cm2']:.3e} sep "
                      f"{rec['separation_meV']} 3rd "
                      f"{rec['third_pair_occupied']}  t {time.time()-t0:.0f}",
                      flush=True)
                json.dump(dict(out, rows=rows), open(os.path.join(
                    RES, "strain_polarisation.json"), "w"), indent=1)
    out["rows"] = rows
    # relaxed fraction at which the calculated polarization charge falls to
    # the measured sheet density
    from scipy.optimize import brentq
    out["relaxation_at_which_sigma_equals_p_s"] = float(brentq(
        lambda r: sigma_polarisation(strain_state(r)[0])[0] - P_S, 0.0, 1.0))
    json.dump(out, open(os.path.join(RES, "strain_polarisation.json"), "w"),
              indent=1)
    print("WROTE results/strain_polarisation.json")


if __name__ == "__main__":
    main()
