"""Remove the hard-wall barrier, and measure what removing it changes.

The Letter previously treated the AlN barrier as infinite, on the grounds that
the valence band offset is of order an electron volt while the subband energies
are tens of millielectronvolts.  That argument is sound for the heavy branch
but weak for the light one, and the light branch carries the mass and the
occupation that the experiments disagree about.  This script solves the same
problem with a finite barrier, scans the offset across the whole published
range rather than adopting one value, and records the sensitivity to every
choice that had to be made.

Outputs results/barrier.json.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import kp6_het as H          # noqa: E402
from gan2dhg import kp6_well as KW        # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RES, exist_ok=True)

DZ = 0.09375          # nm, the spacing of the N = 161 hard-wall grid
NKT = 16
KTMAX = 2.4


def summarise_het(sol, label):
    per = sol["per_subband_nm2"]
    occ = [s for s in range(len(per)) if per[s] > 0]
    z, p = sol["z"], sol["p_of_z"]
    inb = z < 0
    out = {
        "label": label,
        "vbo_eV": sol["vbo_eV"],
        "converged": sol["converged"],
        "iterations": sol["iterations"],
        "EF_minus_E0_meV": 1000.0 * (sol["EF"] - sol["E_of_k"][0, 0]),
        "occupations_cm2": [float(per[s] * 1e14) for s in occ],
        "total_cm2": float(per.sum() * 1e14),
        "centroid_nm": float(np.trapezoid(z * p, z) / np.trapezoid(p, z)),
        "rms_nm": None,
        "barrier_fraction": float(np.trapezoid(p[inb], z[inb])
                                  / np.trapezoid(p, z)),
        "branches": [],
    }
    zc = out["centroid_nm"]
    out["rms_nm"] = float(np.sqrt(np.trapezoid((z - zc) ** 2 * p, z)
                                  / np.trapezoid(p, z)))
    for s in occ:
        kF, m = H.mass_at_kf(sol, s)
        out["branches"].append({
            "index": s, "n_cm2": float(per[s] * 1e14), "kF_per_nm": kF,
            "m_CR": m,
            "edge_meV": float(1000.0 * (sol["E_of_k"][0, s]
                                        - sol["E_of_k"][0, 0])),
        })
    return out


def rashba(sol, pair_lo, solver):
    """Splitting of a Kramers pair at the lower partner's own k_F, in meV."""
    per = sol["per_subband_nm2"]
    kF = np.sqrt(4.0 * np.pi * per[pair_lo])
    w = solver(kF)
    return float(1000.0 * (w[pair_lo + 1] - w[pair_lo])), float(kF)


def main():
    t0 = time.time()
    out = {}

    # ---- baseline: the hard wall, with the counting and the derivative fixed
    print("hard wall baseline ...", flush=True)
    sol0 = KW.self_consistent(p_s_cm2=4.6e13, N=161, n_kt=NKT, kt_max=KTMAX,
                              max_iter=90, mix=0.6, tol=2e-5, verbose=False)
    per0 = sol0["per_subband_nm2"]
    hard = {"converged": bool(sol0["converged"]),
            "EF_minus_E0_meV": 1000.0 * (sol0["EF"] - sol0["E_of_k"][0, 0]),
            "centroid_nm": KW.centroid(sol0), "rms_nm": KW.rms_width(sol0),
            "total_cm2": float(per0.sum() * 1e14), "branches": []}
    for s in range(6):
        if per0[s] <= 0:
            continue
        kF, m = KW.cyclotron_mass_refined(sol0, s)
        _, m_coarse = KW.cyclotron_mass_at_kf(sol0, s)
        hard["branches"].append({
            "index": s, "n_cm2": float(per0[s] * 1e14), "kF_per_nm": kF,
            "m_CR": m, "m_CR_coarse_grid": m_coarse,
            "edge_meV": float(1000.0 * (sol0["E_of_k"][0, s]
                                        - sol0["E_of_k"][0, 0]))})
    zh, Vh, sth = sol0["z"], sol0["V"], sol0["strain"]
    hard["rashba_light_meV"], hard["kF_light"] = rashba(
        sol0, 2, lambda k: KW.solve_subbands(k, zh, Vh, sth, n_states=6)[0])
    hard["rashba_heavy_meV"], hard["kF_heavy"] = rashba(
        sol0, 0, lambda k: KW.solve_subbands(k, zh, Vh, sth, n_states=6)[0])
    out["hard_wall"] = hard
    print(f"  done {time.time()-t0:.0f}s", flush=True)

    # ---- finite barrier across the published range of the offset
    out["finite_barrier"] = []
    for vbo in (0.30, 0.50, 0.70, 0.80):
        print(f"finite barrier vbo={vbo} ...", flush=True)
        sol = H.self_consistent_het(vbo_eV=vbo, L_bar=3.0, dz=DZ, n_kt=NKT,
                                    kt_max=KTMAX, max_iter=90, tol=2e-5,
                                    mix=0.6)
        rec = summarise_het(sol, f"vbo={vbo}")
        z, V, pr = sol["z"], sol["V"], sol["prof"]
        rec["rashba_light_meV"], _ = rashba(
            sol, 2, lambda k: H.solve(k, 0.0, z, V, pr, n_states=6)[0])
        rec["rashba_heavy_meV"], _ = rashba(
            sol, 0, lambda k: H.solve(k, 0.0, z, V, pr, n_states=6)[0])
        out["finite_barrier"].append(rec)
        print(f"  done {time.time()-t0:.0f}s", flush=True)

        if abs(vbo - 0.70) < 1e-9:
            # Angle-resolved interband Bloch overlap, computed not fitted.
            th = np.linspace(0.0, np.pi, 19)
            ov = [H.bloch_overlap(sol, 2, 0, float(t)) for t in th]
            ov_hh = [H.bloch_overlap(sol, 0, 0, float(t)) for t in th]
            ov_ll = [H.bloch_overlap(sol, 2, 2, float(t)) for t in th]
            out["overlap"] = {"theta_rad": th.tolist(),
                              "light_to_heavy": ov,
                              "heavy_to_heavy": ov_hh,
                              "light_to_light": ov_ll,
                              "vbo_eV": vbo}
            print(f"  overlap done {time.time()-t0:.0f}s", flush=True)

    # ---- sensitivity of the barrier result to the choices it required
    sens = []
    for label, kw in (
            ("GaN parameters in the barrier",
             dict(vbo_eV=0.70, gan_params_in_barrier=True, L_bar=3.0)),
            ("barrier region 5 nm instead of 3 nm",
             dict(vbo_eV=0.30, L_bar=5.0)),
            ("AlN spin-orbit 19 meV instead of 22 meV",
             dict(vbo_eV=0.70, L_bar=3.0,
                  aln=dict(H.ALN_KP, D_SO=0.019))),
    ):
        print(f"sensitivity: {label} ...", flush=True)
        sol = H.self_consistent_het(dz=DZ, n_kt=NKT, kt_max=KTMAX,
                                    max_iter=90, tol=2e-5, mix=0.6, **kw)
        sens.append(summarise_het(sol, label))
        print(f"  done {time.time()-t0:.0f}s", flush=True)
    out["sensitivity"] = sens

    out["runtime_s"] = time.time() - t0
    json.dump(out, open(os.path.join(RES, "barrier.json"), "w"), indent=1)
    print("WROTE results/barrier.json", flush=True)


if __name__ == "__main__":
    main()
