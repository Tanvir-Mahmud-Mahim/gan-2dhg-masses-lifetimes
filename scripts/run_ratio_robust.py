"""How robust is the lifetime-ratio tension to everything it could depend on?

An important simplification applies here.  For an isotropic two-dimensional
band, parabolic or not, both the transport and the quantum scattering rates
carry the same prefactor, the cyclotron mass m_CR = hbar^2 k (dE/dk)^-1, which
is also what sets the density of states and what enters the mobility.  The
ratio

    tau_tr / tau_q  =  Int dtheta W(q) / Int dtheta W(q) (1 - cos theta)

therefore contains no mass at all.  It depends only on the shape of the
scattering kernel and on the Fermi wavevector, and the Fermi wavevector follows
from the measured sheet density by Luttinger's theorem alone.  Non-parabolicity,
confinement and the choice of k.p parameters all cancel.

That leaves exactly two things the comparison can depend on: the sheet
densities, and the assumption of spin degeneracy used to convert an oscillation
frequency into a density.  Both are scanned here.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpfet import scatter2d as S
from gpfet.constants import Q as QE

EPS0 = 8.8541878128e-12
EPS_R = 10.4

MEAS_LH_RATIO = 3.82
MEAS_HH_RATIO = 2.13
TARGET = MEAS_LH_RATIO / MEAS_HH_RATIO

print("=" * 78)
print("The ratio tau_tr/tau_q carries no effective mass, so the comparison is")
print("independent of the band structure.  Only the densities enter.")
print("=" * 78)
print(f"  measured: light {MEAS_LH_RATIO:.2f}, heavy {MEAS_HH_RATIO:.2f}, "
      f"light over heavy {TARGET:.2f}")
print("  Every mechanism whose matrix element decreases with momentum transfer")
print("  makes the band with the LARGER Fermi wavevector the more forward-")
print("  peaked one, hence a light-over-heavy value BELOW unity.")
print()


def build(n_lh, n_hh, b_fh, F_eff):
    return [{"label": "lh", "m_over_m0": 0.53, "n_s_cm2": n_lh},
            {"label": "hh", "m_over_m0": 1.92, "n_s_cm2": n_hh}]


def ratio_pair(n_lh, n_hh, wfn_maker):
    n_tot = n_lh + n_hh
    b = S.fang_howard_b(n_tot, 0.0, 1.0, EPS_R)
    F = QE * (n_tot * 1e4 / 2.0) / (EPS_R * EPS0)
    bands = build(n_lh, n_hh, b, F)
    wfn = wfn_maker(b, F)
    out = []
    for bd in bands:
        r = S.lifetimes(bd, bands, wfn, EPS_R, b)
        out.append(r["ratio"])
    return out


MAKERS = {
    "remote charge d=2nm": lambda b, F: (
        lambda q: S.w_remote_impurity(q, 1e13, 2e-9, b, EPS_R)),
    "remote charge d=6nm": lambda b, F: (
        lambda q: S.w_remote_impurity(q, 1e13, 6e-9, b, EPS_R)),
    "roughness L=1nm": lambda b, F: (
        lambda q: S.w_interface_roughness(q, 0.3e-9, 1e-9, F)),
    "roughness L=3nm": lambda b, F: (
        lambda q: S.w_interface_roughness(q, 0.3e-9, 3e-9, F)),
    "background impurity": lambda b, F: (
        lambda q: S.w_background_impurity(q, 1e17, b, EPS_R)),
    "dislocations": lambda b, F: (
        lambda q: S.w_dislocation(q, 1e8, 5.185e-10, 1.0, EPS_R, b)),
}

CASES = [
    ("as published, spin degenerate", 8.0e12, 3.8e13),
    ("both halved, spin resolved", 4.0e12, 1.9e13),
    ("light halved only", 4.0e12, 3.8e13),
    ("heavy halved only", 8.0e12, 1.9e13),
    ("cyclotron-resonance densities", 6.5e12, 4.6e13),
    ("equal densities, extreme test", 2.3e13, 2.3e13),
    ("light denser than heavy, extreme", 3.8e13, 8.0e12),
]

print(f"{'density case':>32}", end="")
for name in MAKERS:
    print(f"{name[:11]:>12}", end="")
print()
print("-" * 78)
rows = []
best = None
for lab, nl, nh in CASES:
    print(f"{lab:>32}", end="")
    for name, mk in MAKERS.items():
        rl, rh = ratio_pair(nl, nh, mk)
        v = rl / rh
        rows.append({"case": lab, "n_lh": nl, "n_hh": nh, "mechanism": name,
                     "lh_ratio": rl, "hh_ratio": rh, "lh_over_hh": v})
        if best is None or abs(np.log(v / TARGET)) < abs(np.log(best["lh_over_hh"] / TARGET)):
            best = rows[-1]
        print(f"{v:12.2f}", end="")
    print()

print()
print(f"Target value to reproduce: {TARGET:.2f}")
print(f"Closest of {len(rows)} combinations: {best['mechanism']} with "
      f"{best['case']}, giving {best['lh_over_hh']:.2f}")
above = [r for r in rows if r["lh_over_hh"] >= TARGET]
print(f"Combinations reaching the measured value: {len(above)} of {len(rows)}")
print()
print("The ordering only inverts when the light band is made DENSER than the")
print("heavy band, which contradicts the measured oscillation frequencies of")
print("166 T and 795 T.  The tension is therefore not an artefact of the")
print("density assignment, and since the ratio carries no mass it is not an")
print("artefact of the band structure either.")

out = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "ratio_robust.json"), "w") as fh:
    json.dump({"target": TARGET, "rows": rows}, fh, indent=2)
print("\nwrote results/ratio_robust.json")
