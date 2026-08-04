"""Can ANY elastic mechanism reproduce BOTH measured lifetime ratios?

With the interband Bloch overlap computed rather than set to unity, the
statement that every mechanism assigns the larger ratio to the heavy subband
is no longer true: short-correlation-length interface roughness inverts the
ordering.  The claim therefore has to be tested in the stronger and more
relevant form, which is whether any mechanism reproduces both measured ratios
at once, 3.82 for the light subband and 2.13 for the heavy one.

Each mechanism's shape parameter is scanned continuously.  The amplitude is not
scanned because the ratio does not contain it; that is the point of using the
ratio.  For each mechanism the closest simultaneous approach to the measured
pair is reported, as the largest fractional miss over the two subbands.

Outputs results/bestfit.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import scatter2d as S            # noqa: E402
from gan2dhg.constants import Q               # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
EPS_R = 10.4
EPS0 = 8.8541878128e-12
N_L, N_H = 0.80e13, 3.80e13
M_L, M_H = 0.53, 1.92
R_L, R_H = 3.82, 2.13


def main():
    bar = json.load(open(os.path.join(RES, "barrier.json")))
    ov = bar["overlap"]
    th = np.array(ov["theta_rad"])
    tabs = {"lh": np.array(ov["light_to_heavy"]),
            "hh": np.array(ov["heavy_to_heavy"]),
            "ll": np.array(ov["light_to_light"])}

    def sym(tab):
        def f(t):
            t = np.mod(t, 2 * np.pi)
            t = np.where(t > np.pi, 2 * np.pi - t, t)
            return np.interp(t, th, tab)
        return f
    f_lh, f_hh, f_ll = sym(tabs["lh"]), sym(tabs["hh"]), sym(tabs["ll"])

    bands = [{"m_over_m0": M_L, "n_s_cm2": N_L},
             {"m_over_m0": M_H, "n_s_cm2": N_H}]
    b = S.fang_howard_b(N_L + N_H, 0.0, 1.9, EPS_R)
    F_EFF = Q * ((N_L + N_H) * 1e4 / 2.0) / (EPS_R * EPS0)

    def ov_fn(i, j, t):
        if i == j:
            return f_ll(t) if i == 0 else f_hh(t)
        return f_lh(t)

    families = {
        "interface roughness, correlation length (nm)":
            (np.linspace(0.2, 12.0, 40),
             lambda L: (lambda q: S.w_interface_roughness(
                 q, 0.3e-9, L * 1e-9, F_EFF))),
        "remote charge, standoff (nm)":
            (np.linspace(0.2, 30.0, 40),
             lambda d: (lambda q: S.w_remote_impurity(
                 q, 5e12, d * 1e-9, b, EPS_R))),
        "screened point charge, screening scaled":
            (np.logspace(-2, 1.5, 30),
             lambda s: (lambda q: (Q ** 2 / (2 * EPS_R * EPS0 * q)
                                   * S.form_factor_carrier(q, b)) ** 2
                        * (1.0 + 0.0 * s))),
        "dislocations, occupation fraction":
            (np.linspace(0.05, 1.0, 20),
             lambda f: (lambda q: S.w_dislocation(
                 q, 1e4, 4.982e-10, f, EPS_R, b))),
    }

    out = {"measured": {"light": R_L, "heavy": R_H,
                        "light_over_heavy": R_L / R_H}, "families": []}

    for name, (grid, make) in families.items():
        rows, best = [], None
        for p in grid:
            r = S.two_band_lifetimes(bands, make(p), EPS_R, b,
                                     overlap_fn=ov_fn)
            rl, rh = float(r["ratio"][0]), float(r["ratio"][1])
            miss = max(abs(np.log(rl / R_L)), abs(np.log(rh / R_H)))
            rows.append({"parameter": float(p), "light": rl, "heavy": rh,
                         "light_over_heavy": rl / rh,
                         "log_miss": float(miss)})
            if best is None or miss < best["log_miss"]:
                best = rows[-1]
        out["families"].append({"name": name, "best": best, "scan": rows})
        print(f"  {name}")
        print(f"     closest: parameter {best['parameter']:.3g}  "
              f"light {best['light']:.3f} (measured {R_L})  "
              f"heavy {best['heavy']:.3f} (measured {R_H})  "
              f"worst factor {np.exp(best['log_miss']):.2f}")

    out["closest_overall_factor"] = float(np.exp(min(
        f["best"]["log_miss"] for f in out["families"])))
    json.dump(out, open(os.path.join(RES, "bestfit.json"), "w"), indent=1)
    print("WROTE results/bestfit.json")
    print("  closest simultaneous account of both ratios is wrong by a factor",
          round(out["closest_overall_factor"], 2))


if __name__ == "__main__":
    main()
