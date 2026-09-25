"""Two coexisting scattering mechanisms, and what the light-hole quantum
mobility requires.

Every calculation uses the coupled two-subband Boltzmann equation with the
computed angle-dependent Bloch overlap (results/barrier.json), as in
run_tension.py, and the measured densities.

1. SINGLE MECHANISMS.  From the continuous scans of run_tension.py: the
   largest light-hole ratio among all cases in which the light-hole ratio
   exceeds the heavy-hole one.

2. PAIRS.  The rates of two mechanisms add.  For every pair of shape
   parameters (the same library as run_tension.py) and a relative weight
   scanned over ten decades, the two ratios tau_tr/tau_q are computed and the
   pair closest to the measurement is kept (largest |ln| distance of the two
   ratios).

3. ABSOLUTE FIT.  For each pair, the two amplitudes are fitted to all four
   measured mobilities (Hall 1900 and 400; quantum 368 and 183, the middle of
   167 to 200 cm^2/Vs) by least squares in the logarithm.  Three free numbers
   (one shape parameter per mechanism is scanned, the two amplitudes fitted)
   face four data.  The fitted amplitudes are converted to physical
   quantities: rms roughness height, dislocation line density times occupation
   squared, remote sheet density, background volume density.

4. STEP-LIKE ROUGHNESS.  Exponentially correlated roughness, whose spectrum
   2 pi D^2 L^2 / (1 + q^2 L^2)^(3/2) has a long small-q tail, as a single
   mechanism.

5. INHOMOGENEITY.  A Gaussian spread of relative rms s in the sheet density
   damps an oscillation of frequency F by exp(-2 pi^2 (F s)^2 / B^2).  A
   linear Dingle fit in 1/B over 25 to 72 T absorbs this into an apparent
   rate: pi/mu_app = pi/mu_q + 2 pi^2 F^2 s^2 (1/25 + 1/72) T^-1.  The spread
   needed to lower the light-hole quantum mobility from the value that the
   transport-setting component of the three best two-mechanism fits gives
   alone to 368 cm^2/Vs is computed, and the field at which the same spread alone
   would damp the heavy-hole oscillations (F = 795 T) by exp(-pi), the
   visibility convention used for the heavy-hole estimate, is reported.
   A spread confined to the light subband is expressed as the rms variation
   of the light-heavy separation it would require.

Outputs results/mixtures.json.
"""

import itertools
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import run_tension as RT                                  # noqa: E402
from gan2dhg import measured as MS, scatter2d as S        # noqa: E402
from gan2dhg.constants import Q, M0                       # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
H_E = 4.135667696e-15            # h/e, T m^2
MEAS = np.array([1900.0, 400.0, 368.0, 183.5])   # cm^2/Vs
# Base amplitudes used in run_tension.families
BASE = {"interface roughness": ("rms height (nm)", 0.3),
        "remote ionised charge": ("sheet density (cm^-2)", 5e12),
        "charged dislocations": ("N f^2 (cm^-2)", 1e4),
        "background impurities": ("volume density (cm^-3)", 1e17)}


def setup():
    ov_fn = RT.overlap_function()
    NL, NH = MS.N_L_CM2, MS.N_H_CM2
    bands = [{"m_over_m0": MS.M_L, "n_s_cm2": NL},
             {"m_over_m0": MS.M_H, "n_s_cm2": NH}]
    b = S.fang_howard_b(NL + NH, 0.0, 1.9, RT.EPS_R)
    F_EFF = Q * ((NL + NH) * 1e4 / 2.0) / (RT.EPS_R * RT.EPS0)
    kF = np.array([S.fermi_wavevector(n) for n in (NL, NH)])
    v = kF / np.array([MS.M_L, MS.M_H])
    m_kg = np.array([MS.M_L, MS.M_H]) * M0
    return ov_fn, bands, b, F_EFF, v, m_kg


def main():
    ov_fn, bands, b, F_EFF, v, m_kg = setup()

    def solve(A, Bk):
        M = np.diag(A.sum(1)) - Bk * (v[None, :] / v[:, None])
        tau = np.linalg.solve(M, np.ones(2))
        tq = 1.0 / A.sum(1)
        return tau / tq, Q * tau / m_kg * 1e4, Q * tq / m_kg * 1e4

    out = {}
    # 1. single mechanisms (from results/tension.json)
    ten = json.load(open(os.path.join(RES, "tension.json")))
    inv = [r["ratio"][0] for fam in ten["scan"] for r in fam["scan"]
           if r["ratio"][0] > r["ratio"][1]]
    out["single_mechanisms"] = {
        "cases": sum(len(f["scan"]) for f in ten["scan"]),
        "cases_with_light_ratio_above_heavy": len(inv),
        "largest_light_ratio_in_those_cases": max(inv) if inv else None}
    print(out["single_mechanisms"], flush=True)

    # library
    fam = RT.families(b, F_EFF, fine=True)
    lib = []
    for name, (pname, grid, make) in fam.items():
        for p in grid:
            A, Bk = S.coupling_matrices(bands, make(p), RT.EPS_R, b,
                                        overlap_fn=ov_fn)
            lib.append({"name": name, "p": float(p), "A": A, "B": Bk})

    # 2. ratio mixtures
    tl, th = MS.R_L, MS.R_H
    W = np.logspace(-5, 5, 201)
    best = {}
    for i, j in itertools.combinations(range(len(lib)), 2):
        e1, e2 = lib[i], lib[j]
        n1, n2 = e1["A"].sum(1)[0], e2["A"].sum(1)[0]
        for w in W:
            r = solve(e1["A"] / n1 + w * e2["A"] / n2,
                      e1["B"] / n1 + w * e2["B"] / n2)[0]
            miss = max(abs(np.log(r[0] / tl)), abs(np.log(r[1] / th)))
            key = (e1["name"], e2["name"])
            if key not in best or miss < best[key]["miss"]:
                best[key] = {"miss": miss, "p1": e1["p"], "p2": e2["p"],
                             "weight": float(w), "ratios": r.tolist()}
    out["ratio_pairs"] = [{"mechanisms": list(k), "factor": float(np.exp(d["miss"])),
                           **{kk: vv for kk, vv in d.items() if kk != "miss"}}
                          for k, d in sorted(best.items(), key=lambda x: x[1]["miss"])]
    for r in out["ratio_pairs"]:
        print("pair", r["mechanisms"], round(r["factor"], 3),
              [round(x, 2) for x in r["ratios"]], flush=True)
    # trajectory of the closest pair as its relative weight is varied (Fig. 3c)
    top = out["ratio_pairs"][0]
    e1 = next(e for e in lib if e["name"] == top["mechanisms"][0] and e["p"] == top["p1"])
    e2 = next(e for e in lib if e["name"] == top["mechanisms"][1] and e["p"] == top["p2"])
    n1, n2 = e1["A"].sum(1)[0], e2["A"].sum(1)[0]
    traj = []
    for w in np.logspace(-4, 4, 161):
        r = solve(e1["A"] / n1 + w * e2["A"] / n2, e1["B"] / n1 + w * e2["B"] / n2)[0]
        traj.append({"weight": float(w), "ratios": r.tolist()})
    out["closest_pair_trajectory"] = {"mechanisms": top["mechanisms"],
                                      "p1": top["p1"], "p2": top["p2"],
                                      "points": traj}

    # 3. absolute fit to four mobilities
    grids = {"interface roughness": np.linspace(0.2, 12, 60),
             "remote ionised charge": np.linspace(0.2, 30, 60),
             "charged dislocations": [1.0], "background impurities": [1.0]}
    alib = {k: [(p, S.coupling_matrices(bands, fam[k][2](p), RT.EPS_R, b,
                                        overlap_fn=ov_fn)) for p in g]
            for k, g in grids.items()}
    names = list(alib)
    fits = []
    for n1, n2 in itertools.combinations_with_replacement(names, 2):
        for i, (p1, e1) in enumerate(alib[n1]):
            for j, (p2, e2) in enumerate(alib[n2]):
                if n1 == n2 and j <= i:
                    continue

                def resid(x):
                    a1, a2 = np.exp(x)
                    _, mt, mq = solve(a1 * e1[0] + a2 * e2[0],
                                      a1 * e1[1] + a2 * e2[1])
                    return np.log(np.r_[mt, mq] / MEAS)
                r = min((least_squares(resid, np.array(x0, float))
                         for x0 in [(0, 0), (2, -2), (-2, 2), (4, 0), (0, 4),
                                    (-4, 0), (0, -4)]), key=lambda z: z.cost)
                a1, a2 = np.exp(r.x)
                _, mt, mq = solve(a1 * e1[0] + a2 * e2[0], a1 * e1[1] + a2 * e2[1])
                fits.append({"mechanisms": [n1, n2], "p1": float(p1),
                             "p2": float(p2), "amplitude_factors": [a1, a2],
                             "worst_factor": float(np.exp(np.max(np.abs(r.fun)))),
                             "mu_tr": mt.tolist(), "mu_q": mq.tolist()})
    fits.sort(key=lambda d: d["worst_factor"])
    bestfit = {}
    for f in fits:
        k = tuple(f["mechanisms"])
        if k not in bestfit:
            phys = []
            for name, a in zip(f["mechanisms"], f["amplitude_factors"]):
                label, base = BASE[name]
                val = base * np.sqrt(a) if name == "interface roughness" else base * a
                phys.append({"mechanism": name, "quantity": label, "value": float(val)})
            f["physical_amplitudes"] = phys
            bestfit[k] = f
    # contribution of each component: mobilities it would give alone, and its
    # share of the light- and heavy-hole quantum and transport rates
    for f in bestfit.values():
        comps = [dict(alib[n])[p] for n, p in zip(f["mechanisms"], (f["p1"], f["p2"]))]
        parts = []
        for (A, Bk), a in zip(comps, f["amplitude_factors"]):
            _, mt, mq = solve(a * A, a * Bk)
            parts.append({"mu_tr_alone": mt.tolist(), "mu_q_alone": mq.tolist(),
                          "share_of_quantum_rate": (a * A.sum(1)).tolist()})
        tot_q = np.sum([pp["share_of_quantum_rate"] for pp in parts], axis=0)
        for pp in parts:
            pp["share_of_quantum_rate"] = (np.array(pp["share_of_quantum_rate"]) / tot_q).tolist()
            pp["share_of_light_transport_rate"] = 1.0 / pp["mu_tr_alone"][0] / sum(
                1.0 / q["mu_tr_alone"][0] for q in parts)
        f["components"] = parts
    out["absolute_fits"] = list(bestfit.values())
    for f in out["absolute_fits"]:
        print("abs", f["mechanisms"], round(f["worst_factor"], 3),
              [(p["quantity"], "%.3g" % p["value"]) for p in f["physical_amplitudes"]],
              flush=True)

    # 4. exponentially correlated roughness
    rows = []
    for lam in np.geomspace(0.2, 100, 40):
        def w_exp(q, L=lam * 1e-9, D=0.3e-9):
            return 2 * np.pi * D ** 2 * L ** 2 / (1 + (q * L) ** 2) ** 1.5 \
                * (Q * F_EFF) ** 2
        A, Bk = S.coupling_matrices(bands, w_exp, RT.EPS_R, b, overlap_fn=ov_fn)
        r, mt, mq = solve(A, Bk)
        s = mt[0] / 1900.0
        rows.append({"correlation_length_nm": float(lam), "ratios": r.tolist(),
                     "rms_height_for_light_Hall_1900_nm": 0.3 * np.sqrt(s),
                     "mu_tr": (mt / s).tolist(), "mu_q": (mq / s).tolist()})
    out["exponential_roughness"] = {
        "scan": rows,
        "largest_light_ratio": max(r["ratios"][0] for r in rows)}
    print("exp roughness: largest light ratio",
          round(out["exponential_roughness"]["largest_light_ratio"], 3), flush=True)

    # 5. inhomogeneity
    F_L, F_H = 166.0, 795.0
    xsum = 1 / 25 + 1 / 72
    mu_meas = 0.0368
    inh = []
    # elastic light-hole quantum mobility without the long-range component:
    # that of the component setting the light-hole transport rate, in the
    # three pairs that fit the four mobilities best
    els = []
    for f in out["absolute_fits"][:3]:
        c = max(f["components"], key=lambda pp: pp["share_of_light_transport_rate"])
        els.append(c["mu_q_alone"][0] * 1e-4)
    for mu_el in (min(els), max(els)):
        extra = np.pi / mu_meas - np.pi / mu_el
        s = np.sqrt(extra / (2 * np.pi ** 2 * F_L ** 2 * xsum))
        B_vis = F_H * s * np.sqrt(2 * np.pi)
        # light-only spread: delta F_L -> delta n_L -> delta(separation)
        dF = F_L * s
        dn_nm2 = 2 * dF / H_E * 1e-18
        dos = MS.M_L / (np.pi * 2 * 38.09982)        # nm^-2 meV^-1 (pair)
        inh.append({"elastic_light_mu_q_cm2Vs": mu_el * 1e4,
                    "relative_density_spread_needed": float(s),
                    "field_below_which_heavy_oscillations_damped_by_exp_minus_pi_T": float(B_vis),
                    "light_only_rms_separation_needed_meV": float(dn_nm2 / dos)})
    out["inhomogeneity"] = inh
    for r in inh:
        print("inhomogeneity", {k: round(v, 4) for k, v in r.items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "mixtures.json"), "w"), indent=1)
    print("WROTE results/mixtures.json")


if __name__ == "__main__":
    main()
