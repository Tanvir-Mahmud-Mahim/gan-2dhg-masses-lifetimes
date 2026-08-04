"""Replace the variational envelope in the scattering with the computed one.

The scattering calculation used Fang-Howard form factors, a one-parameter
variational envelope for a triangular well, while the subband calculation in
the same Letter solves for the envelope exactly.  That inconsistency is removed
here: the form factors are evaluated directly from the self-consistent hole
distribution,

    F_carrier(q) = Int dz Int dz' rho(z) rho(z') exp(-q |z - z'|),
    F_remote(q)  = Int dz rho(z) exp(-q z),

with rho normalised to unity, and compared with the variational forms.  Since
the ratio tau_tr/tau_q depends only on the SHAPE of the matrix element, the
question is not whether the two agree in magnitude but whether they agree in
shape over the range of momentum transfer the Fermi circles sample.

Outputs results/formfactor.json.
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


def exact_form_factors(z_nm, rho):
    """Return callables F_carrier(q_per_m) and F_remote(q_per_m)."""
    z = np.asarray(z_nm) * 1e-9
    r = np.asarray(rho, dtype=float)
    r = r / np.trapezoid(r, z)

    def F_c(q):
        e = np.exp(-q * np.abs(z[:, None] - z[None, :]))
        return float(np.trapezoid(np.trapezoid(e * r[None, :], z, axis=1)
                                  * r, z))

    def F_r(q):
        return float(np.trapezoid(r * np.exp(-q * (z - z.min())), z))

    return F_c, F_r


def main():
    well = json.load(open(os.path.join(RES, "well.json")))
    z = np.array(well["z"])
    rho = np.array(well["p_of_z"])
    F_c, F_r = exact_form_factors(z, rho)

    b = S.fang_howard_b(N_L + N_H, 0.0, 1.9, EPS_R)
    kF_l = S.fermi_wavevector(N_L)
    kF_h = S.fermi_wavevector(N_H)

    qs = np.linspace(1e6, 2.0 * kF_h * 1.05, 60)
    tab = [{"q_per_nm": float(q * 1e-9),
            "F_carrier_exact": F_c(q),
            "F_carrier_fang_howard": float(S.form_factor_carrier(q, b)),
            "F_remote_exact": F_r(q),
            "F_remote_fang_howard": float(S.form_factor_remote(q, b))}
           for q in qs]

    out = {"fang_howard_b_per_nm": float(b * 1e-9),
           "kF_light_per_nm": float(kF_l * 1e-9),
           "kF_heavy_per_nm": float(kF_h * 1e-9),
           "table": tab}

    # The quantity that matters is the ratio, so recompute it both ways.
    bands = [{"m_over_m0": M_L, "n_s_cm2": N_L},
             {"m_over_m0": M_H, "n_s_cm2": N_H}]
    F_EFF = Q * ((N_L + N_H) * 1e4 / 2.0) / (EPS_R * EPS0)

    def eps_exact(q):
        pol = 0.0
        from gan2dhg.constants import HBAR, M0, PI
        for bd in bands:
            kF = S.fermi_wavevector(bd["n_s_cm2"])
            dos = bd["m_over_m0"] * M0 / (PI * HBAR ** 2)
            pol += dos if q <= 2 * kF else dos * (
                1.0 - np.sqrt(1.0 - (2.0 * kF / q) ** 2))
        qsc = Q ** 2 * pol / (2.0 * EPS_R * EPS0)
        return 1.0 + (qsc / q) * F_c(q)

    def ratio(band, w_fn, exact):
        kF = S.fermi_wavevector(band["n_s_cm2"])
        th = np.linspace(1e-4, 2 * np.pi - 1e-4, 501)
        q = 2.0 * kF * np.sin(th / 2.0)
        w = np.array([w_fn(qq, exact) for qq in q])
        e = np.array([eps_exact(qq) if exact
                      else S.dielectric(qq, bands, EPS_R, b) for qq in q])
        w = w / e ** 2
        return float(np.trapezoid(w, th)
                     / np.trapezoid(w * (1 - np.cos(th)), th))

    def w_remote(q, exact):
        v = Q ** 2 / (2.0 * EPS_R * EPS0 * q)
        ff = F_r(q) if exact else S.form_factor_remote(q, b)
        return (v * np.exp(-q * 12e-9) * ff) ** 2

    def w_rough(q, exact):
        return S.w_interface_roughness(q, 0.3e-9, 3.0e-9, F_EFF)

    def w_bg(q, exact):
        v = Q ** 2 / (2.0 * EPS_R * EPS0 * q)
        ff = F_c(q) if exact else S.form_factor_carrier(q, b)
        return (v * ff) ** 2

    rows = []
    for name, w in (("remote charge, d = 12 nm", w_remote),
                    ("roughness, L = 3 nm", w_rough),
                    ("background impurities", w_bg)):
        rl_v, rh_v = ratio(bands[0], w, False), ratio(bands[1], w, False)
        rl_e, rh_e = ratio(bands[0], w, True), ratio(bands[1], w, True)
        rows.append({"mechanism": name,
                     "variational": {"light": rl_v, "heavy": rh_v,
                                     "light_over_heavy": rl_v / rh_v},
                     "computed_envelope": {"light": rl_e, "heavy": rh_e,
                                           "light_over_heavy": rl_e / rh_e}})
    out["ratios"] = rows
    json.dump(out, open(os.path.join(RES, "formfactor.json"), "w"), indent=1)
    print("WROTE results/formfactor.json")
    for r in rows:
        print(f"  {r['mechanism']:<26} L/H variational "
              f"{r['variational']['light_over_heavy']:.4f}   computed envelope "
              f"{r['computed_envelope']['light_over_heavy']:.4f}")


if __name__ == "__main__":
    main()
