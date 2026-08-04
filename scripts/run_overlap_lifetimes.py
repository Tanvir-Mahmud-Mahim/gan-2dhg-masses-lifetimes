"""Use the COMPUTED interband Bloch overlap, and resolve the Rashba splitting.

Two limitations of the scattering analysis are removed here.

The interband Bloch overlap was previously carried as a free parameter,
scanned from zero to unity, because its value was said not to be fixed by the
measurement.  That was too pessimistic: the envelope-function solver returns
the full six-component spinor at every wavevector, so the overlap between the
light and heavy states on their own Fermi circles can simply be computed.  It
also depends on the angle between initial and final wavevectors, and therefore
belongs inside the angular integral rather than multiplying it.

Spin splitting was previously neglected, following the measurement papers.
Here the two Kramers pairs are resolved into four branches whose Fermi
wavevectors differ by the computed Rashba splitting, and the coupled Boltzmann
system is solved on all four.

Outputs results/overlap_lifetimes.json.  Requires results/barrier.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import scatter2d as S            # noqa: E402
from gan2dhg.constants import HBAR, M0, PI, Q  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
EPS_R = 10.4
EPS0 = 8.8541878128e-12

N_L, N_H = 0.80e13, 3.80e13          # measured pair densities, cm^-2
M_L, M_H = 0.53, 1.92                # measured masses
MEAS = {"light": 3.82, "heavy": 2.13}


def main():
    bar = json.load(open(os.path.join(RES, "barrier.json")))
    ov = bar["overlap"]
    th = np.array(ov["theta_rad"])
    o_lh = np.array(ov["light_to_heavy"])
    o_hh = np.array(ov["heavy_to_heavy"])
    o_ll = np.array(ov["light_to_light"])

    def interp_sym(table):
        """Even, 2 pi periodic interpolation of an overlap sampled on [0, pi]."""
        def f(t):
            t = np.mod(t, 2.0 * PI)
            t = np.where(t > PI, 2.0 * PI - t, t)
            return np.interp(t, th, table)
        return f

    f_lh, f_hh, f_ll = (interp_sym(o_lh), interp_sym(o_hh), interp_sym(o_ll))

    out = {
        "overlap_light_to_heavy": {
            "at_0": float(o_lh[0]), "at_pi_over_2": float(np.interp(PI / 2, th, o_lh)),
            "at_pi": float(o_lh[-1]),
            "angle_averaged": float(np.trapezoid(o_lh, th) / PI),
            "min": float(o_lh.min()), "max": float(o_lh.max())},
        "overlap_heavy_to_heavy_angle_averaged":
            float(np.trapezoid(o_hh, th) / PI),
        "overlap_light_to_light_angle_averaged":
            float(np.trapezoid(o_ll, th) / PI),
    }

    bands = [{"m_over_m0": M_L, "n_s_cm2": N_L},
             {"m_over_m0": M_H, "n_s_cm2": N_H}]
    b = S.fang_howard_b(N_L + N_H, 0.0, 1.9, EPS_R)
    F_EFF = Q * ((N_L + N_H) * 1e4 / 2.0) / (EPS_R * EPS0)
    mech = {
        "remote charge, d = 12 nm":
            lambda q: S.w_remote_impurity(q, 5e12, 12e-9, b, EPS_R),
        "roughness, L = 1 nm":
            lambda q: S.w_interface_roughness(q, 0.3e-9, 1.0e-9, F_EFF),
        "roughness, L = 3 nm":
            lambda q: S.w_interface_roughness(q, 0.3e-9, 3.0e-9, F_EFF),
        "background impurities":
            lambda q: S.w_background_impurity(q, 1e17, b, EPS_R),
        "dislocations":
            lambda q: S.w_dislocation(q, 1e4, 4.982e-10, 0.3, EPS_R, b),
    }

    def ov_fn(i, j, theta):
        if i == j:
            return f_ll(theta) if i == 0 else f_hh(theta)
        return f_lh(theta)

    # ---- two subbands, overlap computed rather than scanned ---------------
    rows = []
    for name, w in mech.items():
        r_par = S.two_band_lifetimes(bands, w, EPS_R, b,
                                     overlap=np.ones((2, 2)))
        r_com = S.two_band_lifetimes(bands, w, EPS_R, b, overlap_fn=ov_fn)
        rows.append({
            "mechanism": name,
            "ratio_unit_overlap": [float(x) for x in r_par["ratio"]],
            "ratio_computed_overlap": [float(x) for x in r_com["ratio"]],
            "light_over_heavy_unit": float(r_par["ratio"][0] / r_par["ratio"][1]),
            "light_over_heavy_computed":
                float(r_com["ratio"][0] / r_com["ratio"][1]),
            "interband_fraction_computed":
                [float(x) for x in r_com["interband_fraction_of_quantum_rate"]],
        })
    out["two_band"] = rows
    out["max_light_over_heavy_computed"] = max(
        r["light_over_heavy_computed"] for r in rows)
    out["measured_light_over_heavy"] = MEAS["light"] / MEAS["heavy"]

    # ---- how much the Rashba splitting can matter -------------------------
    # The four branches must NOT be fed to the coupled solver with the
    # pair-summed overlap above: that overlap already sums over the Kramers
    # partner, so applying it to each partner separately counts the final
    # states twice.  What the splitting does to the ratio is instead bounded
    # through the only thing it changes here, the Fermi wavevectors.
    fb = [r for r in bar["finite_barrier"]
          if abs(r["vbo_eV"] - 0.70) < 1e-9][0]

    def dk_rel(n_cm2, m_over_m0, dE_meV):
        kF = S.fermi_wavevector(n_cm2)
        dE = dE_meV * 1e-3 * Q
        return float(dE * m_over_m0 * M0 / (2.0 * HBAR ** 2 * kF) / kF)

    rel_l = dk_rel(N_L, M_L, fb["rashba_light_meV"])
    rel_h = dk_rel(N_H, M_H, fb["rashba_heavy_meV"])
    rows_r = []
    for name, w in mech.items():
        base = S.two_band_lifetimes(bands, w, EPS_R, b, overlap_fn=ov_fn)
        shifted = [{"m_over_m0": M_L, "n_s_cm2": N_L * (1 + rel_l) ** 2},
                   {"m_over_m0": M_H, "n_s_cm2": N_H * (1 + rel_h) ** 2}]
        sh = S.two_band_lifetimes(shifted, w, EPS_R, b, overlap_fn=ov_fn)
        rows_r.append({
            "mechanism": name,
            "light_over_heavy": float(base["ratio"][0] / base["ratio"][1]),
            "light_over_heavy_kF_shifted":
                float(sh["ratio"][0] / sh["ratio"][1])})
    out["rashba_bound"] = {
        "splitting_light_meV": fb["rashba_light_meV"],
        "splitting_heavy_meV": fb["rashba_heavy_meV"],
        "relative_dk_light": rel_l, "relative_dk_heavy": rel_h,
        "rows": rows_r,
        "max_fractional_change": max(
            abs(r["light_over_heavy_kF_shifted"] / r["light_over_heavy"] - 1.0)
            for r in rows_r)}

    json.dump(out, open(os.path.join(RES, "overlap_lifetimes.json"), "w"),
              indent=1)
    print("WROTE results/overlap_lifetimes.json")
    print("  computed light-to-heavy overlap, angle averaged:",
          round(out["overlap_light_to_heavy"]["angle_averaged"], 4))
    print("  max light/heavy ratio, two band, computed overlap:",
          round(out["max_light_over_heavy_computed"], 4))
    print("  largest fractional change from the Rashba splitting:",
          round(out["rashba_bound"]["max_fractional_change"], 5))
    print("  measured:", round(out["measured_light_over_heavy"], 4))


if __name__ == "__main__":
    main()
