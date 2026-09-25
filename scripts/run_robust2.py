"""Turn the remaining stated limitations of the scattering analysis into numbers.

Each block here removes, or bounds, one of the caveats that the Letter
previously only declared.

  local field             a local-field factor of the standard one-parameter
                          shape is applied explicitly and scanned from zero
                          (random phase approximation) to unity

  temperature             the thermal smearing of each Fermi surface is
                          computed rather than asserted to be negligible

  basal-plane warping     a hexagonal warping of the Fermi contour is imposed
                          parametrically and the ratio is recomputed, so the
                          caveat becomes a bound

Each perturbation is reported as the largest fractional change it produces in
either subband's ratio, relative to the unperturbed calculation.  The scan of
the screening strength itself is made in run_tension.py, through the coupled
two-subband solver, where it is compared directly with the measurement.

Outputs results/robust2.json.
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
KB = 1.380649e-23

# Measured, from the quantum oscillation study; the scattering comparison is
# made against the measurement, so the measured densities are used here.
N_L_CM2, N_H_CM2 = 0.80e13, 3.80e13
M_L, M_H = 0.53, 1.92


def bands_of(nl=N_L_CM2, nh=N_H_CM2, ml=M_L, mh=M_H):
    return [{"m_over_m0": ml, "n_s_cm2": nl},
            {"m_over_m0": mh, "n_s_cm2": nh}]


def b_fh(bands):
    return S.fang_howard_b(sum(b["n_s_cm2"] for b in bands), 0.0,
                           1.9, EPS_R)


# ---------------------------------------------------------------------------
# A screening family that contains the random phase approximation
# ---------------------------------------------------------------------------

def dielectric_family(q, bands, b, qs_scale=1.0, gamma=0.0):
    """eps(q) with the screening strength scaled and a local field applied.

    qs_scale multiplies the screening wavevector, so qs_scale -> 0 is the
    unscreened limit and large qs_scale is over-screening.  gamma is the
    strength of a local-field factor of the standard shape

        G(q) = gamma * q / (2 sqrt(q^2 + k_F^2)),

    which suppresses the response at large q; gamma = 0 is the random phase
    approximation.  No particular value is adopted: the point of the scan is
    that the ordering of the two subbands' ratios does not depend on it.
    """
    pol = 0.0
    kF_tot = 0.0
    for bd in bands:
        kF = S.fermi_wavevector(bd["n_s_cm2"])
        kF_tot = max(kF_tot, kF)
        dos = bd["m_over_m0"] * M0 / (PI * HBAR**2)
        if q <= 2.0 * kF:
            pol += dos
        else:
            pol += dos * (1.0 - np.sqrt(1.0 - (2.0 * kF / q) ** 2))
    qs = qs_scale * Q**2 * pol / (2.0 * EPS_R * 8.8541878128e-12)
    G = gamma * q / (2.0 * np.sqrt(q * q + kF_tot * kF_tot))
    return 1.0 + (qs / q) * S.form_factor_carrier(q, b) * (1.0 - G)


def ratio_for(band, bands, w_fn, b, qs_scale=1.0, gamma=0.0, warp=0.0):
    """tau_tr/tau_q for one subband, with an optionally warped Fermi contour.

    A hexagonal warping is imposed as k_F(phi) = k_F [1 + warp cos(6 phi)].
    The six-band Hamiltonian used in this work is exactly isotropic in the
    plane, verified to machine precision, so warp is not a property of the
    model; it is an externally imposed distortion whose only purpose is to
    bound what a warping of that size could do to the ratio.
    """
    kF0 = S.fermi_wavevector(band["n_s_cm2"])
    nphi = 48 if warp else 1
    phis = np.linspace(0.0, 2.0 * PI, nphi, endpoint=False)
    thetas = np.linspace(1e-4, 2.0 * PI - 1e-4, 721)
    num = den = 0.0
    for phi in phis:
        ki = kF0 * (1.0 + warp * np.cos(6.0 * phi))
        kf = kF0 * (1.0 + warp * np.cos(6.0 * (phi + thetas)))
        q = np.sqrt(np.maximum(ki**2 + kf**2
                               - 2.0 * ki * kf * np.cos(thetas), 1e-30))
        w = np.array([w_fn(qq) for qq in q])
        w = w / np.array([dielectric_family(qq, bands, b, qs_scale, gamma)
                          for qq in q]) ** 2
        num += np.trapezoid(w, thetas)
        den += np.trapezoid(w * (1.0 - np.cos(thetas)), thetas)
    return float(num / den)


def mechanisms(b):
    return {
        "remote charge, d = 12 nm":
            lambda q: S.w_remote_impurity(q, 5e12, 12e-9, b, EPS_R),
        "remote charge, d = 5 nm":
            lambda q: S.w_remote_impurity(q, 5e12, 5e-9, b, EPS_R),
        "roughness, L = 1 nm":
            lambda q: S.w_interface_roughness(q, 0.3e-9, 1.0e-9, 8.0e8),
        "roughness, L = 3 nm":
            lambda q: S.w_interface_roughness(q, 0.3e-9, 3.0e-9, 8.0e8),
        "background impurities":
            lambda q: S.w_background_impurity(q, 1e17, b, EPS_R),
        "dislocations":
            lambda q: S.w_dislocation(q, 1e4, 4.982e-10, 0.3, EPS_R, b),
    }


def max_change(rows, key, base_value, relevant_below=None):
    """Largest fractional change of either subband's ratio from the base.

    With relevant_below, only mechanisms whose unperturbed ratios both lie
    below that value are counted: a mechanism giving ratios in the hundreds
    is an order of magnitude or more from the measurement whatever the
    perturbation does, and its sensitivity has no bearing on the comparison.
    """
    base = {r["mechanism"]: r for r in rows if r[key] == base_value}
    worst = 0.0
    for r in rows:
        b0 = base[r["mechanism"]]
        if relevant_below is not None and max(
                b0["ratio_light"], b0["ratio_heavy"]) >= relevant_below:
            continue
        for band in ("ratio_light", "ratio_heavy"):
            worst = max(worst, abs(r[band] / b0[band] - 1.0))
    return float(worst)


def main():
    out = {}
    bands = bands_of()
    b = b_fh(bands)
    mech = mechanisms(b)

    # ---- 2. explicit local field, scanned --------------------------------
    rows = []
    for gamma in (0.0, 0.25, 0.5, 0.75, 1.0):
        for name, w in mech.items():
            rl = ratio_for(bands[0], bands, w, b, gamma=gamma)
            rh = ratio_for(bands[1], bands, w, b, gamma=gamma)
            rows.append({"gamma": gamma, "mechanism": name,
                         "ratio_light": rl, "ratio_heavy": rh,
                         "light_over_heavy": rl / rh})
    out["local_field_scan"] = rows
    out["local_field_max_change"] = max_change(rows, "gamma", 0.0)
    out["local_field_max_change_relevant"] = max_change(rows, "gamma", 0.0,
                                                        relevant_below=20.0)
    out["local_field_never_raises_a_ratio"] = bool(all(
        r["ratio_light"] <= b["ratio_light"] + 1e-12
        and r["ratio_heavy"] <= b["ratio_heavy"] + 1e-12
        for r in rows for b in rows
        if b["mechanism"] == r["mechanism"] and b["gamma"] == 0.0
        and r["mechanism"] != "dislocations"))

    # ---- 3. thermal smearing of the Fermi surfaces -----------------------
    # E_F of each subband above its own edge, from the measured density and
    # the measured mass, and the fractional smearing kT/E_F at 3 K.
    therm = []
    for nm, n_cm2, m in (("light", N_L_CM2, M_L), ("heavy", N_H_CM2, M_H)):
        kF = S.fermi_wavevector(n_cm2)
        EF_J = HBAR**2 * kF**2 / (2.0 * m * M0)
        for T in (2.0, 3.0):
            therm.append({"band": nm, "T_K": T,
                          "EF_meV": EF_J / Q * 1e3,
                          "T_F_K": EF_J / KB,
                          "kT_over_EF": KB * T / EF_J,
                          "dq_over_2kF": 0.5 * KB * T / EF_J})
    out["thermal"] = therm

    # ---- 4. warping bound -------------------------------------------------
    rows = []
    for warp in (0.0, 0.02, 0.05, 0.10):
        for name, w in mech.items():
            rl = ratio_for(bands[0], bands, w, b, warp=warp)
            rh = ratio_for(bands[1], bands, w, b, warp=warp)
            rows.append({"warp": warp, "mechanism": name,
                         "ratio_light": rl, "ratio_heavy": rh,
                         "light_over_heavy": rl / rh})
    out["warp_scan"] = rows
    out["warp_max_change"] = max_change(rows, "warp", 0.0)
    out["warp_max_change_relevant"] = max_change(rows, "warp", 0.0,
                                                 relevant_below=20.0)

    json.dump(out, open(os.path.join(RES, "robust2.json"), "w"), indent=1)
    print("WROTE results/robust2.json")
    for key in ("local_field", "warp"):
        print(f"  {key}: largest fractional change of either ratio = "
              f"{out[key + '_max_change']:.4f}; for mechanisms with both "
              f"ratios below 20: {out[key + '_max_change_relevant']:.4f}")
    print("  local field never raises a ratio (dislocations excepted):",
          out["local_field_never_raises_a_ratio"])
    for t in out["thermal"]:
        print(f"  {t['band']} T = {t['T_K']} K: kT/EF = {t['kT_over_EF']:.4f}")


if __name__ == "__main__":
    main()
