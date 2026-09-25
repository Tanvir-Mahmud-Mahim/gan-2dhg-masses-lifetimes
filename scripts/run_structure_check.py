"""Sensitivity of the masses to the sample and to the AlN spin-orbit splitting.

1. The cyclotron-resonance study (Wang et al., Appl. Phys. Lett. 126, 213102,
   2025) used a heterostructure of the same design but not the same sample as
   the quantum-oscillation study: an 8.2 nm undoped GaN layer, and a total
   sheet density of 0.65 + 4.6 = 5.25e13 cm^-2 from its own fits.  The
   self-consistent masses are recomputed for that layer thickness and density.
2. The spin-orbit splitting of AlN is taken as 22 meV, the rounded value of the
   21.7 meV computed by de Carvalho et al. for the direction parallel to the c
   axis; the same work gives 23.5 meV perpendicular to it.  Both, and 19 meV,
   are tried.

Each case is a full self-consistent solution with the finite AlN barrier.
Masses are local derivatives at each branch's own Fermi wavevector.

Outputs results/structure_check.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import kp6_het as H            # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
KW = dict(L_bar=3.0, dz=0.09375, n_kt=16, kt_max=2.4, max_iter=120,
          tol=2e-5, mix=0.5, sparse=True)


def masses(s):
    per = s["per_subband_nm2"]
    out = []
    for b in range(len(per)):
        if per[b] <= 0:
            continue
        kF, m = H.mass_at_kf(s, b)
        out.append({"index": b, "n_cm2": float(per[b] * 1e14), "m": m})
    return out


def pairs(bands):
    """Average the two spin states of each pair, ordered heavy then light."""
    bands = sorted(bands, key=lambda d: -d["n_cm2"])
    hh = 0.5 * (bands[0]["m"] + bands[1]["m"])
    lh = 0.5 * (bands[2]["m"] + bands[3]["m"]) if len(bands) >= 4 else None
    p_lh = (bands[2]["n_cm2"] + bands[3]["n_cm2"]) if len(bands) >= 4 else 0.0
    return hh, lh, p_lh


def run(label, **kw):
    s = H.self_consistent_het(**{**KW, **kw})
    b = masses(s)
    hh, lh, p_lh = pairs(b)
    rec = {"label": label, "converged": bool(s["converged"]),
           "m_hh": hh, "m_lh": lh, "p_lh_cm2": p_lh, "bands": b,
           "args": {k: v for k, v in kw.items() if k != "aln"}}
    if "aln" in kw:
        rec["args"]["aln_D_SO"] = kw["aln"]["D_SO"]
    print(f"{label:40s} conv {s['converged']}  m_hh {hh:.4f}  m_lh {lh:.4f}"
          f"  p_lh {p_lh:.3e}", flush=True)
    return rec


def main():
    out = []
    for vbo in (0.3, 0.7):
        out.append(run(f"reference 15 nm, 4.6e13, vbo {vbo}",
                       p_s_cm2=4.6e13, vbo_eV=vbo, L_gan=15.0))
        out.append(run(f"CR sample 8.2 nm, 5.25e13, vbo {vbo}",
                       p_s_cm2=5.25e13, vbo_eV=vbo, L_gan=8.2))
    for dso in (0.019, 0.0217, 0.0235):
        out.append(run(f"AlN D_SO {1000 * dso:.1f} meV, vbo 0.7",
                       p_s_cm2=4.6e13, vbo_eV=0.7, L_gan=15.0,
                       aln=dict(H.ALN_KP, D_SO=dso)))
    json.dump(out, open(os.path.join(RES, "structure_check.json"), "w"),
              indent=1)
    print("WROTE results/structure_check.json")


if __name__ == "__main__":
    main()
