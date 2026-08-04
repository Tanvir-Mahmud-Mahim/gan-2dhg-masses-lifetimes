"""Spin splitting in the asymmetric well, with branches tracked and converged.

An earlier evaluation of this splitting read it off fixed eigenvalue indices.
That is unsafe here: the light pair and the crystal-field split-off pair cross
within the occupied range of in-plane wavevector, so fixed indices report the
splitting of whichever pair occupies them and jump at the crossing.  Branches
are followed by eigenvector continuity instead, and the result is checked for
convergence in the grid spacing before it is quoted.

Outputs results/rashba.json.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gan2dhg import kp6_het as H
from gan2dhg import kp6_well as KW
from gan2dhg.kp6 import biaxial_strain_on_AlN

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
KK = np.linspace(0.15, 1.10, 20)

def hard_wall(N):
    dz = 15.0/(N-1)
    sol = KW.self_consistent(p_s_cm2=4.6e13, N=N, n_kt=16, kt_max=2.4,
                             max_iter=90, mix=0.6, tol=2e-5, verbose=False)
    z = sol["z"]
    prof = H.profile(z, vbo_eV=50.0)
    fake = {"z": z, "V": sol["V"], "prof": prof,
            "per_subband_nm2": sol["per_subband_nm2"]}
    return sol, fake

out = {}
t0=time.time()
for N in (121, 161, 241):
    sol, fake = hard_wall(N)
    per = sol["per_subband_nm2"]
    kF_l = float(np.sqrt(4*np.pi*per[2])); kF_h = float(np.sqrt(4*np.pi*per[0]))
    sp_l = H.rashba_vs_k(fake, KK, pair_lo=2)
    sp_h = H.rashba_vs_k(fake, KK, pair_lo=0)
    out[f"hard_wall_N{N}"] = {
        "N": N, "kF_light": kF_l, "kF_heavy": kF_h,
        "k": KK.tolist(), "split_light_meV": sp_l.tolist(),
        "split_heavy_meV": sp_h.tolist(),
        "at_kF_light_meV": float(np.interp(kF_l, KK, sp_l)),
        "at_kF_heavy_meV": float(np.interp(kF_h, KK, sp_h)),
        "light_mass": float(KW.cyclotron_mass_refined(sol,2)[1]),
    }
    print(N, "light@kF", round(out[f"hard_wall_N{N}"]["at_kF_light_meV"],4),
          "heavy@kF", round(out[f"hard_wall_N{N}"]["at_kF_heavy_meV"],4),
          f"{time.time()-t0:.0f}s", flush=True)

for vbo in (0.30, 0.70):
    sol = H.self_consistent_het(vbo_eV=vbo, L_bar=3.0, dz=0.09375, n_kt=16,
                                kt_max=2.4, max_iter=90, tol=2e-5, mix=0.6)
    per = sol["per_subband_nm2"]
    kF_l = float(np.sqrt(4*np.pi*per[2])); kF_h = float(np.sqrt(4*np.pi*per[0]))
    sp_l = H.rashba_vs_k(sol, KK, pair_lo=2)
    sp_h = H.rashba_vs_k(sol, KK, pair_lo=0)
    out[f"vbo_{vbo}"] = {"vbo_eV": vbo, "kF_light": kF_l, "kF_heavy": kF_h,
        "k": KK.tolist(), "split_light_meV": sp_l.tolist(),
        "split_heavy_meV": sp_h.tolist(),
        "at_kF_light_meV": float(np.interp(kF_l, KK, sp_l)),
        "at_kF_heavy_meV": float(np.interp(kF_h, KK, sp_h))}
    print(vbo, "light@kF", round(out[f"vbo_{vbo}"]["at_kF_light_meV"],4),
          f"{time.time()-t0:.0f}s", flush=True)

json.dump(out, open(os.path.join(RES,"rashba.json"),"w"), indent=1)
print("WROTE results/rashba.json")
