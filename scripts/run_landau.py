"""Landau levels of the self-consistent well, and the mass a Lifshitz-Kosevich
analysis of them returns.

The light-hole mass reported from quantum oscillations rises linearly from
0.48 m0 at 32 T to 0.69 m0 at 72 T and extrapolates to 0.30 m0 at zero field.
The zero-field value agrees with the self-consistent band structure.  Whether
the field dependence is also a property of the band structure can only be
decided by computing the Landau-level spectrum itself, which this script does
with the six-band operator in the self-consistent potential
(src/gan2dhg/landau.py), and by then analysing that spectrum in the way the
experiment analysed its data.

WHAT IS COMPUTED
----------------
1. The Landau levels below E_F + 40 meV on a grid uniform in 1/B from 25 to
   125 T, with the self-consistent potential of the finite-barrier solution
   at an offset of 0.7 eV held fixed (the sheet density is fixed, so the
   confining field does not change with B).  The grid is truncated at -2 nm
   and +8 nm, which moves the bound levels at the two Fermi wavevectors by
   less than 0.03 meV (test suite).  The levels are cached in
   results/landau_levels.json.

2. The oscillatory density of states at the Fermi level, which the
   longitudinal resistance follows in the Shubnikov-de Haas effect: each level
   is broadened by a Gaussian of width sigma, the Fermi level is fixed at
   every field and temperature by the total sheet density, and the density of
   states is thermally averaged with -df/dE on an energy grid of 0.02 meV.
   The width sigma is set so that the damping at 45 T equals the Dingle factor
   exp(-pi / mu_q B) of the measured light-hole quantum mobility,
   368 cm^2/Vs; half that width tests the sensitivity.

3. A Lifshitz-Kosevich analysis of those oscillations, done as the
   measurement was done: the amplitude at the light-hole frequency is taken
   at temperatures from 1.8 to 15 K and the thermal factor X / sinh X, with
   X = 2 pi^2 k_B T m / (hbar e B), is fitted for m.  It is done over the
   whole 32 to 72 T window (the analogue of the reported field-averaged
   0.53 m0), and in windows one oscillation period wide centred at fields
   from 32 to 72 T (the analogue of the field-resolved values).

4. All of the above for three values of the unknown valence-band magnetic
   parameter kappa_L (landau.py), which sets the Zeeman splitting beyond the
   free-electron spin term, so that the Zeeman energy is tested and not
   assumed.

Outputs results/landau.json.
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gan2dhg import kp6_het as H             # noqa: E402
from gan2dhg import landau as L              # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')

# Optional arguments: an alternative zero-field solution and output names, so
# that the same analysis can be repeated on the polarization-closure well
# (scripts/run_well_polarisation.py):
#   python run_landau.py well_het_pol.json landau_pol.json 0
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
WELL = _args[0] if len(_args) > 0 else "well_het.json"
OUT = _args[1] if len(_args) > 1 else "landau.json"
CACHE = os.path.join(RES, "landau_levels.json" if WELL == "well_het.json"
                     else "landau_levels_" + WELL)
KB = 8.617333262e-5              # eV/K
HBAR_E_OVER_M0 = 1.1576764e-4    # eV/T
P_S_NM2 = 4.6e13 * 1e-14
MU_Q_L = 0.0368                  # m^2/Vs, measured light-hole quantum mobility
TEMPS = np.array([1.8, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
KAPPAS = tuple(float(x) for x in _args[2].split(",")) if len(_args) > 2 \
    else (0.0, -2.0, 2.0)
# Uniform in 1/B: 320 points from 80 to 25 T, extended with the same step to
# 125 T.  The extension is a numerical device only: a window one oscillation
# period wide centred at 72 T reaches beyond 80 T on its high-field side.
_base = np.linspace(1.0 / 80.0, 1.0 / 25.0, 320)
DINV = _base[1] - _base[0]
_ext = 1.0 / 80.0 - DINV * np.arange(1, int((1.0 / 80.0 - 1.0 / 125.0) / DINV) + 1)
INV_B = np.sort(np.concatenate([_ext, _base]))

_d = json.load(open(os.path.join(RES, WELL)))
_z = np.array(_d["z"])
_m = (_z <= 8.0 + 1e-9) & (_z >= -2.0 - 1e-9)
Z = _z[_m]
V = np.array(_d["V"])[_m]
PROF = H.profile(Z, _d["vbo_eV"])
EF0 = _d["EF"]
E_MAX = EF0 + 0.040


def levels_at(args):
    B, kappa = args
    return B, kappa, L.spectrum(B, Z, V, PROF, E_MAX, kappa_L=kappa).tolist()


def all_levels():
    """Levels on the full grid, reusing any that are already cached."""
    cache = {"B_T": [], "kappa": {str(k): [] for k in KAPPAS}}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE))
    have = {round(b, 9) for b in cache["B_T"]}
    Bs = 1.0 / INV_B
    need = [float(B) for B in Bs if round(B, 9) not in have]
    if need:
        jobs = [(B, k) for k in KAPPAS for B in need]
        with Pool(2) as p:
            res = p.map(levels_at, jobs, chunksize=2)
        for k in KAPPAS:
            new = {r[0]: r[2] for r in res if r[1] == k}
            old = dict(zip(cache["B_T"], cache["kappa"][str(k)]))
            old.update(new)
            cache["kappa"][str(k)] = [old[b] for b in sorted(old)]
        cache["B_T"] = sorted(set(cache["B_T"]) | set(need))
        json.dump(cache, open(CACHE, "w"))
    Bs = np.array(cache["B_T"])
    return Bs, {k: cache["kappa"][str(k)] for k in KAPPAS}


# ---------------------------------------------------------------------------
# Thermally averaged density of states at the Fermi level
# ---------------------------------------------------------------------------

def thermal_dos_at_ef(E, B, sigma, temps):
    """<g>_T at the Fermi level for each temperature, fixed sheet density."""
    D = L.degeneracy_per_nm2(B)
    E = np.sort(np.asarray(E))
    lo, hi = EF0 - 0.080, E_MAX
    de = 2e-5
    e = np.arange(lo, hi, de)
    n_below = np.sum(E < lo - 6 * sigma)
    act = E[E >= lo - 6 * sigma]
    g = D * np.exp(-0.5 * ((e[:, None] - act[None, :]) / sigma) ** 2).sum(
        axis=1) / (np.sqrt(2 * np.pi) * sigma)
    # weight of the active levels lying below the grid
    from scipy.special import erfc
    tail = D * np.sum(0.5 * erfc((act - lo) / (np.sqrt(2) * sigma)))
    out = []
    for T in temps:
        kT = KB * T

        def dens(mu):
            x = np.clip((e - mu) / kT, -60, 60)
            return (D * n_below + tail + np.sum(g / (1 + np.exp(x))) * de
                    - P_S_NM2)

        mu = brentq(dens, lo + 0.01, hi - 0.01, xtol=1e-10)
        x = np.clip((e - mu) / kT, -60, 60)
        mdf = np.exp(x) / (1 + np.exp(x)) ** 2 / kT
        out.append(float(np.sum(g * mdf) * de))
    return np.array(out)


def lk_mass(amps, temps, B):
    """Fit A(T) proportional to X / sinh X for m (units of m0)."""
    amps = np.asarray(amps)

    def model(m):
        X = 2.0 * np.pi ** 2 * KB * temps * m / (HBAR_E_OVER_M0 * B)
        return X / np.sinh(X)

    def cost(m):
        mod = model(m)
        a = np.dot(mod, amps) / np.dot(mod, mod)
        return np.sum((amps / amps[0] - a * mod / amps[0]) ** 2)

    r = minimize_scalar(cost, bounds=(0.02, 3.0), method="bounded")
    return float(r.x)


def amplitude(inv, y, f, lo, hi):
    """Amplitude of the component at frequency f in the window lo <= 1/B <= hi,
    after removing a linear background, with a Hann taper."""
    w = (inv >= lo) & (inv <= hi)
    x, yy = inv[w], y[w]
    c = np.polyfit(x, yy, 1)
    yy = yy - np.polyval(c, x)
    han = np.hanning(len(x))
    return float(2.0 * abs(np.sum(han * yy * np.exp(-2j * np.pi * f * x)))
                 / np.sum(han))


def analyse(Bs, spectra, sigma):
    inv = 1.0 / Bs
    order = np.argsort(inv)
    inv, Bs = inv[order], Bs[order]
    spectra = [spectra[i] for i in order]
    g = np.array([thermal_dos_at_ef(E, B, sigma, TEMPS)
                  for B, E in zip(Bs, spectra)]).T          # (T, B)

    # light-hole frequency: peak of the lowest-temperature spectrum, 32-72 T
    w = (Bs >= 32.0) & (Bs <= 72.0)
    freqs = np.linspace(40.0, 250.0, 841)
    spec = [amplitude(inv, g[0], f, inv[w].min(), inv[w].max())
            for f in freqs]
    f_l = float(freqs[int(np.argmax(spec))])

    out = {"sigma_meV": sigma * 1e3, "light_frequency_T": f_l,
           "inv_B": inv.tolist(), "g_T": g.tolist(), "temps": TEMPS.tolist()}
    amps = [amplitude(inv, g[t], f_l, inv[w].min(), inv[w].max())
            for t in range(len(TEMPS))]
    B_eff = 1.0 / np.mean(inv[w])
    out["window_32_72"] = {"B_eff_T": float(B_eff), "amplitudes": amps,
                           "m_LK": lk_mass(amps, TEMPS, B_eff)}

    rows = []
    period = 1.0 / f_l
    for Bc in (32.0, 40.0, 48.0, 56.0, 64.0, 72.0):
        lo, hi = 1.0 / Bc - 0.5 * period, 1.0 / Bc + 0.5 * period
        if lo < inv.min() or hi > inv.max():
            continue
        a = [amplitude(inv, g[t], f_l, lo, hi) for t in range(len(TEMPS))]
        rows.append({"B_centre_T": Bc, "amplitudes": a,
                     "m_LK": lk_mass(a, TEMPS, Bc)})
    out["field_resolved"] = rows
    return out


def main():
    t0 = time.time()
    Bs, spectra = all_levels()
    print(f"levels ready: {len(Bs)} fields, {time.time() - t0:.0f} s",
          flush=True)

    m_calc = float(np.mean([m["m_CR"] for m in _d["masses"]
                            if m["subband"] in (2, 3)]))
    hwc45 = HBAR_E_OVER_M0 * 45.0 / m_calc
    sig_d = hwc45 * np.sqrt(np.pi / (MU_Q_L * 45.0)) / (np.sqrt(2.0) * np.pi)
    out = {"B_T": Bs.tolist(), "E_F0": EF0,
           "m_light_zero_field": m_calc, "sigma_dingle_meV": sig_d * 1e3,
           "measured": {"m_32T": 0.48, "m_72T": 0.69, "m_avg_32_72": 0.53,
                        "m_B0_extrapolated": 0.30},
           "kappa": {}}
    for k in KAPPAS:
        rec = {"analysis": []}
        for sig in (sig_d, 0.5 * sig_d):
            a = analyse(Bs, spectra[k], sig)
            rec["analysis"].append(a)
            fr = ", ".join(f"{r['B_centre_T']:.0f} T: {r['m_LK']:.3f}"
                           for r in a["field_resolved"])
            print(f"kappa {k:+.1f} sigma {sig * 1e3:.2f} meV  f_L "
                  f"{a['light_frequency_T']:.1f} T  32-72 T window m_LK "
                  f"{a['window_32_72']['m_LK']:.3f}  |  {fr}", flush=True)
        out["kappa"][str(k)] = rec
    out["well"] = WELL
    json.dump(out, open(os.path.join(RES, OUT), "w"))
    print(f"WROTE results/{OUT}  {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
