"""Landau levels analysed with a level width fixed by the measured Dingle slope.

run_landau.py sets the Gaussian level width so that the damping at one field,
45 T, equals the Dingle factor of the measured light-hole quantum mobility.
Because a Gaussian level damps the oscillations as exp(-c / B^2) and not as
exp(-c / B), matching at one field does not reproduce the measured Dingle
slope.  Here the width is instead chosen so that the Dingle analysis of the
computed oscillations, done as the experiment did it, returns the measured
368 cm^2/Vs:

  amplitude of the light-hole component at 1.8 K in windows two periods wide
  centred from 28 to 72 T, divided by X / sinh X, and ln(amplitude) fitted by
  a straight line in 1/B; mu_q = -pi / slope.

With that width the Lifshitz-Kosevich analysis of run_landau.py (windows one
period wide, temperatures 1.8 to 15 K) is repeated, both at fixed sheet
density (the physical case: the chemical potential then oscillates) and at a
fixed chemical potential, which isolates the contribution of the
chemical-potential oscillations.

usage: run_landau_dingle.py WELL [WELL ...]
       (wells written by run_a6.py; their Landau levels must first be cached
        by run_landau.py WELL OUT)
Outputs results/landau_dingle.json.
"""

import importlib.util
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
MU_Q_MEAS = 0.0368          # m^2/Vs
KAPPAS = ("0.0", "-2.0", "2.0")


def load_rl(well):
    argv = sys.argv
    sys.argv = ['run_landau.py', well, 'unused.json', '0']
    spec = importlib.util.spec_from_file_location(
        'RL_' + well.replace('.', '_'), os.path.join(HERE, 'run_landau.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def thermal_dos(RL, E, B, sigma, temps, mu_fixed=None):
    """As run_landau.thermal_dos_at_ef, optionally at a fixed chemical potential."""
    D = RL.L.degeneracy_per_nm2(B)
    E = np.sort(np.asarray(E))
    lo, hi = RL.EF0 - 0.080, RL.E_MAX
    de = 2e-5
    e = np.arange(lo, hi, de)
    n_below = np.sum(E < lo - 6 * sigma)
    act = E[E >= lo - 6 * sigma]
    g = D * np.exp(-0.5 * ((e[:, None] - act[None, :]) / sigma) ** 2).sum(
        axis=1) / (np.sqrt(2 * np.pi) * sigma)
    tail = D * np.sum(0.5 * erfc((act - lo) / (np.sqrt(2) * sigma)))
    out, mus = [], []
    for T in temps:
        kT = RL.KB * T
        if mu_fixed is None:
            def dens(mu):
                x = np.clip((e - mu) / kT, -60, 60)
                return (D * n_below + tail + np.sum(g / (1 + np.exp(x))) * de
                        - RL.P_S_NM2)
            mu = brentq(dens, lo + 0.01, hi - 0.01, xtol=1e-10)
        else:
            mu = mu_fixed
        x = np.clip((e - mu) / kT, -60, 60)
        mdf = np.exp(x) / (1 + np.exp(x)) ** 2 / kT
        out.append(float(np.sum(g * mdf) * de))
        mus.append(mu)
    return np.array(out), np.array(mus)


def grids(RL, Bs, spectra, sigma, temps, mu_fixed=None):
    inv = 1.0 / Bs
    o = np.argsort(inv)
    res = [thermal_dos(RL, spectra[i], Bs[i], sigma, temps, mu_fixed) for i in o]
    return inv[o], np.array([r[0] for r in res]).T, np.array([r[1] for r in res]).T


def light_frequency(RL, inv, g0):
    w = (inv >= 1 / 72) & (inv <= 1 / 32)
    freqs = np.linspace(40.0, 250.0, 841)
    spec = [RL.amplitude(inv, g0, f, inv[w].min(), inv[w].max()) for f in freqs]
    return float(freqs[int(np.argmax(spec))])


def dingle_mu(RL, inv, g0, fL, m_lk):
    per = 1.0 / fL
    centres = np.linspace(1 / 72, 1 / 28, 24)
    A = []
    for c in centres:
        w = (inv >= c - per) & (inv <= c + per)
        x, y = inv[w], g0[w]
        y = y - np.polyval(np.polyfit(x, y, 2), x)
        han = np.hanning(len(x))
        A.append(2 * abs(np.sum(han * y * np.exp(-2j * np.pi * fL * x))) / np.sum(han))
    A = np.array(A)
    X = 2 * np.pi ** 2 * RL.KB * 1.8 * m_lk / (RL.HBAR_E_OVER_M0 / centres)
    A = A / (X / np.sinh(X))
    s, _ = np.polyfit(centres, np.log(A), 1)
    return float(-np.pi / s)


def lk_rows(RL, inv, g, fL):
    w = (inv >= 1 / 72) & (inv <= 1 / 32)
    amps = [RL.amplitude(inv, g[t], fL, inv[w].min(), inv[w].max())
            for t in range(len(RL.TEMPS))]
    out = {"window_32_72": RL.lk_mass(amps, RL.TEMPS, 1.0 / np.mean(inv[w]))}
    per = 1.0 / fL
    for Bc in (32.0, 40.0, 48.0, 56.0, 64.0, 72.0):
        lo, hi = 1.0 / Bc - 0.5 * per, 1.0 / Bc + 0.5 * per
        a = [RL.amplitude(inv, g[t], fL, lo, hi) for t in range(len(RL.TEMPS))]
        out[str(int(Bc))] = RL.lk_mass(a, RL.TEMPS, Bc)
    return out


def analyse_well(well):
    RL = load_rl(well)
    cache = json.load(open(RL.CACHE))
    Bs = np.array(cache["B_T"])
    m0 = float(np.mean([m["m_CR"] for m in RL._d["masses"] if m["subband"] in (2, 3)]))
    res = {"well": well, "A6_factor": RL._d.get("A6_factor", 1.0),
           "vbo_eV": RL._d["vbo_eV"], "m_light_zero_field": m0,
           "p_light_1e13": sum(m["n_cm2"] for m in RL._d["masses"]
                               if m["subband"] in (2, 3)) / 1e13,
           "kappa": {}}
    for kap in KAPPAS:
        if kap not in cache["kappa"]:
            continue
        spectra = cache["kappa"][kap]
        cache_g = {}

        def mu_of(sig):
            inv, g, _ = grids(RL, Bs, spectra, sig, np.array([1.8]))
            fL = light_frequency(RL, inv, g[0])
            cache_g[sig] = fL
            return dingle_mu(RL, inv, g[0], fL, m0)

        # bracket and solve mu_q(sigma) = measured
        lo, hi = 1.0e-3, 5.0e-3
        sig = brentq(lambda s: mu_of(s) - MU_Q_MEAS, lo, hi, xtol=2e-5)
        inv, g, mus = grids(RL, Bs, spectra, sig, RL.TEMPS)
        fL = light_frequency(RL, inv, g[0])
        mu_ref = float(np.mean(mus[0]))
        inv2, g2, _ = grids(RL, Bs, spectra, sig, RL.TEMPS, mu_fixed=mu_ref)
        rec = {"sigma_meV": sig * 1e3, "light_frequency_T": fL,
               "dingle_mu_q_cm2Vs": dingle_mu(RL, inv, g[0], fL, m0) * 1e4,
               "LK_fixed_density": lk_rows(RL, inv, g, fL),
               "LK_fixed_chemical_potential": lk_rows(RL, inv2, g2, fL),
               "chemical_potential_range_1p8K_meV": float(1e3 * (mus[0].max() - mus[0].min()))}
        res["kappa"][kap] = rec
        print(well, "kappa", kap, json.dumps({k: (round(v, 3) if isinstance(v, float) else
                                                 {kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v)
                                             for k, v in rec.items()}), flush=True)
    return res


def main():
    wells = sys.argv[1:] or ["well_het_pol03.json", "well_het_pol03_A6.json"]
    fn = os.path.join(RES, "landau_dingle.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for w in wells:
        out[w] = analyse_well(w)
        json.dump(out, open(fn, "w"), indent=1)
    print("WROTE results/landau_dingle.json")


if __name__ == "__main__":
    main()
