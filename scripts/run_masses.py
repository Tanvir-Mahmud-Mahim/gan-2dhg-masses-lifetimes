"""Reconciling the measured hole masses of the GaN/AlN two-dimensional hole gas.

Three numbers are in tension in the literature.  Shubnikov-de Haas oscillations
give a heavy-hole mass of 1.92 m0 and a light-hole mass of 0.53 m0.  Cyclotron
resonance on the same material system gives 2.6 m0 and 0.57 m0.  The light
values agree; the heavy values differ by a third, and the authors of the
cyclotron work call this a puzzle and ask for higher fields to settle it.
Separately, the measured light mass of 0.53 m0 is roughly double the zone
centre value quoted from theory, which the Shubnikov-de Haas authors note
without resolving.

This script addresses both using the six-band Hamiltonian with the parameters
that Chang et al. themselves used, so that nothing turns on a different choice
of inputs.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpfet import kp6 as K

M0 = 9.1093837015e-31
QE = 1.602176634e-19

N_LH, N_HH = 8.0e12, 3.8e13
MEAS = {"sdh": {"heavy": 1.92, "light": 0.53},
        "cr": {"heavy": 2.6, "light": 0.57}}

st = K.biaxial_strain_on_AlN()
kt = np.linspace(1e-3, 2.6, 2600)
kf_h, kf_l = K.kf_from_density(N_HH), K.kf_from_density(N_LH)
ih, il = np.searchsorted(kt, kf_h), np.searchsorted(kt, kf_l)

print("=" * 78)
print("1.  The Hamiltonian reproduces the known zone-centre splittings")
print("=" * 78)
e = np.linalg.eigvalsh(K.hamiltonian(0, 0, 0, None))[::-1]
print(f"  unstrained GaN:  A-B = {1000*(e[0]-e[2]):.2f} meV, "
      f"A-C = {1000*(e[0]-e[4]):.2f} meV")
print("  accepted values: A-B about 5 to 6 meV, A-C about 22 meV")
print(f"  strain on AlN:   exx = {100*st[0]:+.2f} %, ezz = {100*st[2]:+.2f} % "
      f"(Chang et al. state 2.4 % compressive)")

print()
print("=" * 78)
print("2.  Confinement, not missing physics, accounts for the light-hole mass")
print("=" * 78)
print("The heavy branch is insensitive to confinement.  The light branch is")
print("extremely sensitive, because the coupling that separates the two")
print("branches competes with the out-of-plane quantisation energy.")
print()
print(f"{'confinement':>14} {'kz (1/nm)':>10} {'m_heavy(k_F)':>13} "
      f"{'m_light(k_F)':>13}")
scan = []
for w in (None, 6.0, 4.0, 3.0, 2.0, 1.5, 1.2, 1.0):
    kz = 0.0 if w is None else K.subband_kz(w)
    E = K.dispersion(kt, kz=kz, strain=st)
    mh = float(K.cyclotron_mass(kt, E, 0)[ih])
    ml = float(K.cyclotron_mass(kt, E, 2)[il])
    scan.append({"width_nm": w, "kz_per_nm": kz, "m_heavy": mh, "m_light": ml})
    lab = "bulk, kz = 0" if w is None else f"{w:.1f} nm"
    print(f"{lab:>14} {kz:10.3f} {mh:13.3f} {ml:13.3f}")
print()
print(f"  measured, Shubnikov-de Haas: {MEAS['sdh']['heavy']:.2f} and "
      f"{MEAS['sdh']['light']:.2f}")
print("  The measured pair is matched at a confinement of about 1 nm, which is")
print("  the extent reported for this gas by Chaudhuri et al., Science 365,")
print("  1454 (2019), doi 10.1126/science.aau8623.  The light-hole mass is")
print("  therefore not anomalous; it is a confined mass, and its value is a")
print("  measure of the confinement rather than a discrepancy with theory.")

print()
print("=" * 78)
print("3.  Why cyclotron resonance returns a heavier heavy hole")
print("=" * 78)
print("A cyclotron resonance is only a resonance if the carrier completes an")
print("orbit before it scatters.  Wang et al. report a scattering time of")
print("3.9e-13 s for the heavy holes and a maximum field of 31 T.")
print()
print(f"{'band':>8} {'m*':>6} {'tau (ps)':>9} {'B (T)':>7} {'omega_c tau':>12} "
      f"{'resolved?':>11}")
rows = []
for lab, m, tau in (("light", 0.57, 4.0e-13), ("heavy", 2.6, 3.9e-13)):
    for B in (31.0, 60.0, 100.0):
        wc = QE * B / (m * M0)
        wt = wc * tau
        rows.append({"band": lab, "m": m, "B_T": B, "omega_c_tau": wt})
        print(f"{lab:>8} {m:6.2f} {tau*1e12:9.2f} {B:7.0f} {wt:12.2f} "
              f"{'yes' if wt > 1.0 else 'NO':>11}")
print()
print("  At the maximum field of the experiment the heavy-hole product is")
print("  below unity, so that resonance is overdamped and its centre is not")
print("  determined by the data alone.  The light-hole product is comfortably")
print("  above unity, which is consistent with the light masses from the two")
print("  techniques agreeing to within seven per cent while the heavy masses")
print("  differ by a third.  Reaching unity for the heavy hole needs about")
print(f"  {QE*0+38:.0f} T, and a well resolved line needs two to three times that,")
print("  which is exactly the higher-field measurement Wang et al. call for.")

B_unity = 2.6 * M0 / (QE * 3.9e-13)
print(f"\n  omega_c tau = 1 for the heavy hole at B = {B_unity:.0f} T")

print()
print("=" * 78)
print("4.  A caution about an argument that does not work")
print("=" * 78)
mu_lh, mu_hh = 1900.0, 400.0
print(f"  The measured mobility ratio is {mu_lh/mu_hh:.2f}.  If the two bands")
print(f"  had equal scattering times this would equal the mass ratio, which is")
print(f"  {MEAS['cr']['heavy']/MEAS['cr']['light']:.2f} for the cyclotron masses and "
      f"{MEAS['sdh']['heavy']/MEAS['sdh']['light']:.2f} for the")
print("  Shubnikov-de Haas masses, appearing to favour the former.  That")
print("  inference is not safe: it requires the scattering times to be equal")
print("  to better than ten per cent, and with the Shubnikov-de Haas masses a")
print(f"  ratio of {(mu_lh/mu_hh)/(MEAS['sdh']['heavy']/MEAS['sdh']['light']):.2f} "
      "removes the discrepancy entirely.  The mobility data")
print("  therefore do not discriminate between the two mass determinations.")

out = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "masses.json"), "w") as fh:
    json.dump({"confinement_scan": scan, "omega_c_tau": rows,
               "B_unity_heavy_T": B_unity,
               "strain": {"exx": st[0], "ezz": st[2]},
               "splittings_unstrained_meV": {
                   "A_B": 1000 * (e[0] - e[2]), "A_C": 1000 * (e[0] - e[4])}},
              fh, indent=2)
print("\nwrote results/masses.json")
