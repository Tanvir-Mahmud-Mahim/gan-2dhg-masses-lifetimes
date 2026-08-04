"""Which elastic mechanism reproduces the measured tau_tr/tau_q of each band?

The measured ratios are dimensionless and independent of disorder strength, so
this is a test with no free amplitude.  Only the geometry of each mechanism
enters: the standoff distance of remote charge, the correlation length of the
roughness, the dislocation density.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpfet import scatter2d as S

# --- Measured, from data/gan_2dhg_measured.yaml ----------------------------
# Chang et al., Nat. Electron. 9, 346 (2026), doi 10.1038/s41928-026-01590-8
# Wang et al., Appl. Phys. Lett. 126, 213102 (2025), doi 10.1063/5.0273413
LH = {"label": "light hole", "m_over_m0": 0.53, "n_s_cm2": 8.0e12}
HH = {"label": "heavy hole", "m_over_m0": 1.92, "n_s_cm2": 3.8e13}
BANDS = [LH, HH]

MEAS = {
    "light hole": {"mu_hall": 1900.0, "mu_q": 368.0, "tau_q_ps": 0.15},
    "heavy hole": {"mu_hall": 400.0, "mu_q": 183.5, "tau_q_ps": 0.205},
}

EPS_R = 10.4          # static dielectric constant of GaN, perpendicular to c
M_Z = 1.0             # out-of-plane hole mass for the envelope, order unity
C_LAT = 5.185e-10     # m

# Structure, Chang et al.: 15 nm GaN, 2DHG at the buried GaN/AlN interface,
# top 5 nm of the GaN doped with Mg.  Ionised Mg therefore stands off the
# channel by 10 to 15 nm.
T_GAN_NM = 15.0
MG_CAP_NM = 5.0

n_tot = LH["n_s_cm2"] + HH["n_s_cm2"]
b_fh = S.fang_howard_b(n_tot, 0.0, M_Z, EPS_R)

print("=" * 74)
print("Measured lifetimes, derived from published mobilities and masses")
print("=" * 74)
print(f"{'band':>12} {'k_F (1e7/cm)':>13} {'tau_tr (ps)':>12} "
      f"{'tau_q (ps)':>11} {'ratio':>7}")
meas_ratio = {}
for bd in BANDS:
    lab = bd["label"]
    kF = S.fermi_wavevector(bd["n_s_cm2"]) * 1e-2      # 1/m -> 1/cm
    t_tr = S.tau_from_mobility(MEAS[lab]["mu_hall"], bd["m_over_m0"])
    t_q = MEAS[lab]["tau_q_ps"] * 1e-12
    meas_ratio[lab] = t_tr / t_q
    print(f"{lab:>12} {kF*1e-7:13.3f} {t_tr*1e12:12.3f} "
          f"{t_q*1e12:11.3f} {t_tr/t_q:7.2f}")
print(f"\nFang-Howard b = {b_fh*1e-9:.3f} 1/nm  -> "
      f"channel extent ~ {3.0/b_fh*1e9:.1f} nm")
print(f"Screening wavevector q_s = {S.screening_wavevector(BANDS, EPS_R)*1e-9:.3f}"
      f" 1/nm; 2k_F(LH) = {2*S.fermi_wavevector(LH['n_s_cm2'])*1e-9:.3f}, "
      f"2k_F(HH) = {2*S.fermi_wavevector(HH['n_s_cm2'])*1e-9:.3f} 1/nm")

# --- Candidate mechanisms --------------------------------------------------
F_EFF = Q_EFF = None
from gpfet.constants import Q as QE
EPS0 = 8.8541878128e-12
# Effective field pressing the gas to the interface, e(n_depl + n_s/2)/eps
F_EFF = QE * (n_tot * 1e4 / 2.0) / (EPS_R * EPS0)

MECHS = []
for d_nm in (2.0, 5.0, 10.0, 15.0):
    MECHS.append((
        f"remote ionised charge, d = {d_nm:.0f} nm",
        lambda q, d=d_nm * 1e-9: S.w_remote_impurity(q, 1e13, d, b_fh, EPS_R)))
for lam_nm in (0.5, 1.0, 2.0, 4.0, 8.0):
    MECHS.append((
        f"interface roughness, Lambda = {lam_nm:.1f} nm",
        lambda q, L=lam_nm * 1e-9: S.w_interface_roughness(q, 0.3e-9, L, F_EFF)))
MECHS.append((
    "background impurities in channel",
    lambda q: S.w_background_impurity(q, 1e17, b_fh, EPS_R)))
MECHS.append((
    "threading dislocations",
    lambda q: S.w_dislocation(q, 1e8, C_LAT, 1.0, EPS_R, b_fh)))

print()
print("=" * 74)
print("Computed tau_tr/tau_q by mechanism (screened, two-band)")
print("=" * 74)
print(f"{'mechanism':>36} {'LH ratio':>10} {'HH ratio':>10} {'LH/HH':>9}")
print(f"{'MEASURED':>36} {meas_ratio['light hole']:10.2f} "
      f"{meas_ratio['heavy hole']:10.2f} "
      f"{meas_ratio['light hole']/meas_ratio['heavy hole']:9.2f}")
print("-" * 74)

rows = []
for name, wfn in MECHS:
    r = {}
    for bd in BANDS:
        r[bd["label"]] = S.lifetimes(bd, BANDS, wfn, EPS_R, b_fh)
    rl = r["light hole"]["ratio"]
    rh = r["heavy hole"]["ratio"]
    rows.append({"mechanism": name, "lh_ratio": rl, "hh_ratio": rh,
                 "lh_over_hh": rl / rh,
                 "lh_mu_tr": r["light hole"]["mu_tr_cm2Vs"],
                 "hh_mu_tr": r["heavy hole"]["mu_tr_cm2Vs"]})
    print(f"{name:>36} {rl:10.2f} {rh:10.2f} {rl/rh:9.2f}")

print()
print("KEY TEST.  Experiment gives a LARGER tau_tr/tau_q for the LIGHT hole,")
print("which has the SMALLER Fermi wavevector.  Every mechanism whose matrix")
print("element falls with q predicts the opposite ordering, because the band")
print("with the larger k_F samples larger scattering angles and is therefore")
print("more forward-peaked in relative terms.  A mechanism reproduces the data")
print("only if the LH/HH column exceeds unity.")

out = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "scattering.json"), "w") as fh:
    json.dump({"measured": {k: {"ratio": v} for k, v in meas_ratio.items()},
               "mechanisms": rows,
               "b_fh_per_m": b_fh, "F_eff_V_per_m": F_EFF}, fh, indent=2)
print("\nwrote results/scattering.json")
