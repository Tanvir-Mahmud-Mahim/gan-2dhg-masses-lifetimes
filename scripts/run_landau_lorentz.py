"""Does the band structure change the light-hole Dingle slope?

run_landau_dingle.py chooses a Gaussian level width so that the emulated
Dingle analysis returns the measured 368 cm^2/Vs; it cannot tell how much of
that slope the band structure itself supplies.  Here every Landau level is
instead given a Lorentzian profile of half width Gamma = hbar / (2 tau_q) with a
known quantum lifetime.  For a single parabolic band the Lifshitz-Kosevich
theory then gives exactly the Dingle factor exp(-pi / omega_c tau_q), so a
returned mobility that differs from the input measures damping added by the band
structure: the spin splitting of the light-hole pair, the Zeeman term, the
mixing with heavy-hole levels, and (at fixed density) the oscillation of the
chemical potential.

The input lifetime is quoted as a mobility with the measured light-hole mass,
0.53 m0, as in the scattering calculation.  The Dingle analysis is the one of
run_landau_dingle.py (1.8 K, windows two periods wide centred from 28 to 72 T).
A control applies the same analysis to a single sinusoid with the Lifshitz-
Kosevich factors.

At fixed sheet density every level, heavy or light, carries the same width, so
the heavy-hole levels are sharper than the measured heavy-hole oscillations
imply and pin the chemical potential more strongly than in the experiment; the
fixed-chemical-potential analysis is free of this and is the reference.

Outputs results/landau_lorentz.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_landau_dingle as LD                              # noqa: E402

RES = os.path.join(HERE, '..', 'results')
HBAR = 6.582119569e-16      # eV s
E = 1.602176634e-19
M0 = 9.1093837e-31
M_CONV = 0.53
INPUTS = (0.0368, 0.10, 0.18)       # m^2/Vs
WELLS = ("well_het_pol03.json", "well_het_pol03_A6.json")


def gamma_of(mu_q):
    """Lorentzian half width (eV) for a quantum mobility converted with 0.53 m0."""
    tau = mu_q * M_CONV * M0 / E
    return HBAR / (2 * tau)


def dos_at_mu(RL, levels, B, gam, T, mu_fixed=None):
    D = RL.L.degeneracy_per_nm2(B)
    El = np.sort(np.asarray(levels))
    kT = RL.KB * T
    if mu_fixed is None:
        f = lambda mu: D * np.sum(0.5 + np.arctan((mu - El) / gam) / np.pi) - RL.P_S_NM2
        mu = brentq(f, RL.EF0 - 0.03, RL.E_MAX - 0.005, xtol=1e-11)
    else:
        mu = mu_fixed
    e = np.linspace(mu - 25 * kT, mu + 25 * kT, 1001)
    de = e[1] - e[0]
    g = D * np.sum((gam / np.pi) / ((e[:, None] - El[None, :]) ** 2 + gam ** 2), axis=1)
    x = np.clip((e - mu) / kT, -60, 60)
    mdf = np.exp(x) / (1 + np.exp(x)) ** 2 / kT
    return float(np.sum(g * mdf) * de), mu


def control(RL, inv, mu_in, m):
    """Single sinusoid with the Lifshitz-Kosevich factors, same analysis."""
    X = 2 * np.pi ** 2 * RL.KB * 1.8 * m / (RL.HBAR_E_OVER_M0 / inv)
    y = np.exp(-np.pi * inv / mu_in) * (X / np.sinh(X)) * np.cos(2 * np.pi * 165.3 * inv)
    fL = LD.light_frequency(RL, inv, y)
    return LD.dingle_mu(RL, inv, y, fL, m)


def main():
    out = {"inputs_cm2Vs": [m * 1e4 for m in INPUTS], "mass_for_conversion": M_CONV,
           "wells": {}}
    for well in WELLS:
        RL = LD.load_rl(well)
        cache = json.load(open(RL.CACHE))
        Bs = np.array(cache["B_T"])
        o = np.argsort(1 / Bs)
        inv = (1 / Bs)[o]
        m0 = float(np.mean([m["m_CR"] for m in RL._d["masses"] if m["subband"] in (2, 3)]))
        rec = {"A6_factor": RL._d.get("A6_factor", 1.0), "kappa": {},
               "control": {str(mu * 1e4): control(RL, inv, mu, m0) * 1e4 for mu in INPUTS}}
        for kap in ("-2.0", "0.0", "2.0"):
            spectra = cache["kappa"][kap]
            rows = []
            for mu_in in INPUTS:
                gam = gamma_of(mu_in)
                res = [dos_at_mu(RL, spectra[i], Bs[i], gam, 1.8) for i in o]
                g_n = np.array([r[0] for r in res])
                mu_ref = float(np.mean([r[1] for r in res]))
                g_mu = np.array([dos_at_mu(RL, spectra[i], Bs[i], gam, 1.8, mu_ref)[0]
                                 for i in o])
                row = {"mu_q_in": mu_in * 1e4, "Gamma_meV": gam * 1e3}
                for tag, g in (("fixed_density", g_n), ("fixed_chemical_potential", g_mu)):
                    fL = LD.light_frequency(RL, inv, g)
                    row[tag] = {"light_frequency_T": fL,
                                "mu_q_returned": LD.dingle_mu(RL, inv, g, fL, m0) * 1e4}
                rows.append(row)
                print(well, kap, round(mu_in * 1e4),
                      {k: (round(row[k]["light_frequency_T"], 1), round(row[k]["mu_q_returned"]))
                       for k in ("fixed_density", "fixed_chemical_potential")}, flush=True)
            rec["kappa"][kap] = rows
        out["wells"][well] = rec
    json.dump(out, open(os.path.join(RES, "landau_lorentz.json"), "w"), indent=1)
    print("WROTE results/landau_lorentz.json")


if __name__ == "__main__":
    main()
