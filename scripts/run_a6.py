"""The parameter A6 and the light-hole subband.

A6 multiplies the term i A6 k_z (k_x + i k_y) of the six-band Hamiltonian,
the only term that couples motion along the growth axis to motion in the
plane.  In a gas 0.4 nm wide k_z is large, and this term, rather than the
in-plane parameters, sets the in-plane mass of the light subband.  A6 is also
the least well determined parameter in the published sets: Rinke et al.
(Phys. Rev. B 77, 075202) give -3.202, while Punya and Lambrecht (Phys. Rev. B
85, 195147) obtain -1.55 by a direct fit and -3.31 in the quasi-cubic
approximation, and state that their parameters agree with those of Rinke et
al. except for A6.

usage:
  run_a6.py ellipticity             -> results/a6_ellipticity.json
  run_a6.py scan NAME [NAME ...]    -> results/a6_scan_NAME.json
       NAME in: A07 (closure A, 0.7 eV), A03, B07, B03
  run_a6.py apply                   -> results/a6_apply.json and the two wells
                                       used by run_landau.py
                                       (results/well_het_pol03.json,
                                        results/well_het_pol03_A6.json)

ELLIPTICITY.  A k.p Hamiltonian describes a valence band only if every band
curves away from the band edge at large wavevector (the quadratic form is
negative definite).  With the other parameters fixed, this fails beyond a
critical |A6|; the threshold is found by bisection on the largest eigenvalue
of H(k)/k^2 at |k| = 500 nm^-1 over all polar angles.

SCAN.  For each closure and offset, A6 is scaled and the full self-consistent
solution is repeated at the measured sheet density.  The factor at which the
light-hole occupation equals the measured 0.80 x 10^13 cm^-2 is found by
linear interpolation, together with the light-hole mass there.

APPLY.  With the factor fixed in this way on the quantum-oscillation sample:
the grid check, the prediction for the cyclotron-resonance sample (8.2 nm
GaN) at its quoted total density and at the density obtained when its
heavy-hole spectral weight is re-read with the heavy mass of 1.92 m0, and the
two polarization-closure wells at 0.3 eV used for the Landau levels.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from gan2dhg import kp6_het as H             # noqa: E402
from gan2dhg import kp6                      # noqa: E402
from gan2dhg.kp6 import GAN                  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
KW = dict(L_bar=3.0, dz=0.09375, n_kt=16, kt_max=2.4, max_iter=150, tol=2e-5,
          mix=0.4, sparse=True, n_states=8)
P_LH_MEAS = 0.80
SERIES = {"A07": (0.7, "A", (1.0, 1.30, 1.35, 1.38, 1.40, 1.42)),
          "A03": (0.3, "A", (1.0, 1.20, 1.25, 1.30, 1.35)),
          "B07": (0.7, "B", (1.0, 1.35, 1.40)),
          "B03": (0.3, "B", (1.0, 1.20, 1.25, 1.27, 1.31))}


def sigma_b():
    import run_strain_polarisation as SP
    return SP.sigma_polarisation(SP.strain_state(0.0)[0])[0]


def gan_with(f):
    g = dict(GAN)
    g["A6"] = GAN["A6"] * f
    return g


def solve_case(f, vbo, closure, p_s=4.6e13, L_gan=15.0, **extra):
    kw = dict(KW); kw.update(extra)
    sig = sigma_b() if closure == "B" else None
    return H.self_consistent_het(p_s_cm2=p_s, vbo_eV=vbo, L_gan=L_gan,
                                 sigma_cm2=sig, gan=gan_with(f), **kw)


def summarise(sol, **tags):
    per = sol["per_subband_nm2"]
    bands = []
    for b in range(len(per)):
        if per[b] > 0:
            kF, m = H.mass_at_kf(sol, b, n_states=8)
            bands.append({"index": b, "n_cm2": float(per[b] * 1e14),
                          "kF_per_nm": kF, "m_CR": m,
                          "edge_meV": float(1000 * (sol["E_of_k"][0, b]
                                                    - sol["E_of_k"][0, 0]))})
    rec = dict(tags); rec.update(converged=bool(sol["converged"]),
                                 occupied_branches=len(bands), bands=bands)
    if len(bands) == 4:
        rec.update(m_hh=0.5 * (bands[0]["m_CR"] + bands[1]["m_CR"]),
                   m_lh=0.5 * (bands[2]["m_CR"] + bands[3]["m_CR"]),
                   p_lh_1e13=(bands[2]["n_cm2"] + bands[3]["n_cm2"]) / 1e13,
                   p_hh_1e13=(bands[0]["n_cm2"] + bands[1]["n_cm2"]) / 1e13,
                   Delta_meV=bands[2]["edge_meV"])
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in rec.items() if k != "bands"}), flush=True)
    return rec


# ---------------------------------------------------------------------------
def max_curvature(f, k=500.0, n_theta=361):
    p = gan_with(f)
    th = np.linspace(0.0, np.pi / 2, n_theta)
    vals = [np.linalg.eigvalsh(kp6.hamiltonian(k * np.sin(t), 0.0,
                                               k * np.cos(t), None, p)).max()
            / k ** 2 for t in th]
    i = int(np.argmax(vals))
    return float(vals[i]), float(np.degrees(th[i]))


def ellipticity():
    from scipy.optimize import brentq
    f_c = brentq(lambda f: max_curvature(f)[0], 1.2, 1.6, xtol=1e-5)
    out = {"critical_factor": f_c, "critical_A6": GAN["A6"] * f_c,
           "published_A6": GAN["A6"],
           "largest_eigenvalue_of_H_over_k2_eV_nm2": {
               str(f): max_curvature(f)[0] for f in (1.0, 1.2, 1.27, 1.3,
                                                     1.35, 1.40, 1.45)},
           "angle_from_c_deg_at_critical": max_curvature(f_c + 1e-3)[1]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RES, "a6_ellipticity.json"), "w"),
              indent=1)


def scan(names):
    for name in names:
        vbo, closure, factors = SERIES[name]
        rows = []
        for f in factors:
            rows.append(summarise(solve_case(f, vbo, closure), series=name,
                                  vbo=vbo, closure=closure, factor=f,
                                  A6=GAN["A6"] * f))
            json.dump(rows, open(os.path.join(RES, f"a6_scan_{name}.json"),
                                 "w"), indent=1)


def fitted(name):
    rows = [r for r in json.load(open(os.path.join(RES, f"a6_scan_{name}.json")))
            if r.get("p_lh_1e13") is not None and r["converged"]]
    rows.sort(key=lambda r: r["factor"])
    for a, b in zip(rows[:-1], rows[1:]):
        if a["p_lh_1e13"] <= P_LH_MEAS <= b["p_lh_1e13"]:
            t = (P_LH_MEAS - a["p_lh_1e13"]) / (b["p_lh_1e13"] - a["p_lh_1e13"])
            return {"factor": a["factor"] + t * (b["factor"] - a["factor"]),
                    "m_lh": a["m_lh"] + t * (b["m_lh"] - a["m_lh"]),
                    "m_hh": a["m_hh"] + t * (b["m_hh"] - a["m_hh"])}
    return None


def save_well(sol, fname, factor):
    per = sol["per_subband_nm2"]
    E0 = sol["E_of_k"][0]
    masses = []
    for b in range(4):
        kF, m = H.mass_at_kf(sol, b, n_states=8)
        masses.append({"subband": b, "n_cm2": float(per[b] * 1e14),
                       "kF_per_nm": kF, "m_CR": m,
                       "E_edge_meV": float(1000 * (E0[b] - E0[0]))})
    json.dump({"z": sol["z"].tolist(), "V": sol["V"].tolist(),
               "EF": float(sol["EF"]), "vbo_eV": sol["vbo_eV"],
               "E_of_k": sol["E_of_k"].tolist(), "kt": sol["kt"].tolist(),
               "per_subband_nm2": [float(p) for p in per], "masses": masses,
               "A6_factor": factor, "A6": GAN["A6"] * factor,
               "sigma_cm2": sol["sigma_cm2"]},
              open(os.path.join(RES, fname), "w"))


def apply():
    fits = {n: fitted(n) for n in SERIES}
    print("fits", json.dumps(fits, indent=1), flush=True)
    out = {"fits_to_measured_light_occupation": fits}
    fA7 = fits["A07"]["factor"]
    fA3 = fits["A03"]["factor"]
    # grid check at the 0.7 eV fit
    out["grid_check"] = [
        summarise(solve_case(fA7, 0.7, "A"), what="reference grid", factor=fA7),
        summarise(solve_case(fA7, 0.7, "A", dz=0.125), what="dz 0.125 nm",
                  factor=fA7),
        summarise(solve_case(fA7, 0.7, "A", L_bar=4.0), what="barrier 4 nm",
                  factor=fA7)]
    # cyclotron-resonance sample: 8.2 nm GaN; total density as quoted (0.65 +
    # 4.6) and with the heavy spectral weight re-read at 1.92 m0:
    # N_hh = 4.6 x 1.92 / 2.6 = 3.40, total 4.05 (x 10^13 cm^-2)
    cr = []
    for vbo, f in ((0.7, fA7), (0.3, fA3)):
        for p_s in (5.25e13, 4.05e13):
            for ff in (1.0, f):
                cr.append(summarise(solve_case(ff, vbo, "A", p_s=p_s,
                                               L_gan=8.2),
                                    sample="cyclotron resonance", vbo=vbo,
                                    p_s=p_s, factor=ff))
    out["cyclotron_resonance_sample"] = cr
    # wells for the Landau levels: polarization closure, 0.3 eV
    fB3 = fits["B03"]["factor"]
    for f, fname in ((1.0, "well_het_pol03.json"),
                     (fB3, "well_het_pol03_A6.json")):
        sol = solve_case(f, 0.3, "B")
        out.setdefault("landau_wells", []).append(
            summarise(sol, file=fname, factor=f))
        save_well(sol, fname, f)
    json.dump(out, open(os.path.join(RES, "a6_apply.json"), "w"), indent=1)
    print("WROTE results/a6_apply.json")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ellipticity":
        ellipticity()
    elif cmd == "scan":
        scan(sys.argv[2:])
    elif cmd == "apply":
        apply()
