"""Does interband scattering explain the measured lifetime ratios?

Single-band treatments predict that the subband with the larger Fermi
wavevector has the more forward-peaked scattering and therefore the larger
tau_tr/tau_q.  The measurement is the other way round.  The one term that can
invert the ordering is interband scattering, because its momentum transfer is
bounded below by |k_F1 - k_F2| and so never becomes forward-peaked, and because
it loads asymmetrically: the light band, with the smaller density of states,
sees the heavy band as a large phase space to scatter into, while the reverse
is weak.
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
M_Z = 1.0

LH = {"label": "light hole", "m_over_m0": 0.53, "n_s_cm2": 8.0e12}
HH = {"label": "heavy hole", "m_over_m0": 1.92, "n_s_cm2": 3.8e13}
BANDS = [LH, HH]

MU_HALL = {"light hole": 1900.0, "heavy hole": 400.0}
TAU_Q = {"light hole": 0.15, "heavy hole": (0.17, 0.24)}

n_tot = LH["n_s_cm2"] + HH["n_s_cm2"]
b_fh = S.fang_howard_b(n_tot, 0.0, M_Z, EPS_R)
F_EFF = QE * (n_tot * 1e4 / 2.0) / (EPS_R * EPS0)

t_tr = {lab: S.tau_from_mobility(MU_HALL[lab], bd["m_over_m0"])
        for lab, bd in zip([b["label"] for b in BANDS], BANDS)}
meas = {
    "light hole": {"tau_tr": t_tr["light hole"], "tau_q": 0.15e-12,
                   "ratio": t_tr["light hole"] / 0.15e-12},
    "heavy hole": {"tau_tr": t_tr["heavy hole"], "tau_q": 0.205e-12,
                   "ratio": t_tr["heavy hole"] / 0.205e-12},
}

MECHS = {
    "remote charge d=2nm":
        lambda q: S.w_remote_impurity(q, 1e13, 2e-9, b_fh, EPS_R),
    "remote charge d=10nm":
        lambda q: S.w_remote_impurity(q, 1e13, 10e-9, b_fh, EPS_R),
    "roughness L=1nm":
        lambda q: S.w_interface_roughness(q, 0.3e-9, 1e-9, F_EFF),
    "roughness L=3nm":
        lambda q: S.w_interface_roughness(q, 0.3e-9, 3e-9, F_EFF),
    "background impurity":
        lambda q: S.w_background_impurity(q, 1e17, b_fh, EPS_R),
}

kF = [S.fermi_wavevector(b["n_s_cm2"]) for b in BANDS]
print("=" * 78)
print("Momentum transfer available to each channel")
print("=" * 78)
print(f"  k_F  light = {kF[0]*1e-9:.3f} 1/nm,  heavy = {kF[1]*1e-9:.3f} 1/nm")
print(f"  intraband LH: q from 0 to {2*kF[0]*1e-9:.3f} 1/nm")
print(f"  intraband HH: q from 0 to {2*kF[1]*1e-9:.3f} 1/nm")
print(f"  INTERBAND   : q from {abs(kF[0]-kF[1])*1e-9:.3f} to "
      f"{(kF[0]+kF[1])*1e-9:.3f} 1/nm  (never reaches zero)")

print()
print("=" * 78)
print("Two-band lifetimes including interband scattering")
print("Overlap is the squared Bloch overlap between the light and heavy")
print("subbands, the one quantity not fixed by the measurement.")
print("=" * 78)
print(f"{'mechanism':>22} {'overlap':>8} {'LH ratio':>9} {'HH ratio':>9} "
      f"{'LH/HH':>7} {'LH interband':>13} {'HH interband':>13}")
print(f"{'MEASURED':>22} {'':>8} {meas['light hole']['ratio']:9.2f} "
      f"{meas['heavy hole']['ratio']:9.2f} "
      f"{meas['light hole']['ratio']/meas['heavy hole']['ratio']:7.2f}")
print("-" * 78)

rows = []
for name, wfn in MECHS.items():
    for ov in (0.0, 0.05, 0.2, 0.5, 1.0):
        O = np.array([[1.0, ov], [ov, 1.0]])
        r = S.two_band_lifetimes(BANDS, wfn, EPS_R, b_fh, overlap=O)
        rl, rh = r["ratio"]
        fi = r["interband_fraction_of_quantum_rate"]
        rows.append({"mechanism": name, "overlap": ov,
                     "lh_ratio": float(rl), "hh_ratio": float(rh),
                     "lh_over_hh": float(rl / rh),
                     "lh_interband_frac": float(fi[0]),
                     "hh_interband_frac": float(fi[1]),
                     "mu": r["mu_cm2Vs"].tolist()})
        print(f"{name:>22} {ov:8.2f} {rl:9.2f} {rh:9.2f} {rl/rh:7.2f} "
              f"{100*fi[0]:12.0f}% {100*fi[1]:12.0f}%")
    print()

target = meas['light hole']['ratio'] / meas['heavy hole']['ratio']
ok = [r for r in rows if r["lh_over_hh"] > 1.0]
print("=" * 78)
print(f"Measured LH/HH ratio of ratios = {target:.2f}")
print(f"Combinations reaching LH/HH above unity: {len(ok)} of {len(rows)}")
for r in sorted(ok, key=lambda z: abs(z["lh_over_hh"] - target))[:6]:
    print(f"  {r['mechanism']:>22} overlap {r['overlap']:.2f}  "
          f"LH/HH = {r['lh_over_hh']:.2f}  "
          f"(LH {r['lh_ratio']:.2f}, HH {r['hh_ratio']:.2f})")

out = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "interband.json"), "w") as fh:
    json.dump({"measured": {k: {kk: float(vv) for kk, vv in v.items()}
                            for k, v in meas.items()},
               "rows": rows}, fh, indent=2)
print("\nwrote results/interband.json")
