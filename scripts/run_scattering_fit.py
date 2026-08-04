"""Can any mixture of elastic mechanisms reproduce all four measured lifetimes?

Four numbers are measured: a transport and a quantum lifetime for each of the
two subbands.  A mixture of N mechanisms has N free amplitudes.  With two
mechanisms the problem is over-determined by two, so it is a test rather than
a fit.

The rates add, so for a mixture

    1/tau_q(band)  = sum_k  A_k * rq_k(band)
    1/tau_tr(band) = sum_k  A_k * rt_k(band)

with the SAME amplitudes A_k entering both bands and both lifetimes.  That is
the constraint that makes this a test.

HONEST TREATMENT OF THE SOFTEST NUMBER
--------------------------------------
The heavy-hole quantum lifetime is the least secure quantity in the whole
data set.  Chang et al. obtain the light-hole value from a Dingle analysis
(mu_q = 368 +/- 14 cm^2/Vs) but state the heavy-hole value only as an estimate
from the field at which its oscillations become visible, quoting 167 to 200
cm^2/Vs.  Every conclusion below is therefore reported as a function of that
quantity rather than at a single assumed value.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gan2dhg import scatter2d as S
from gan2dhg.constants import Q as QE

EPS0 = 8.8541878128e-12
EPS_R = 10.4
M_Z = 1.0
C_LAT = 5.185e-10

LH = {"label": "light hole", "m_over_m0": 0.53, "n_s_cm2": 8.0e12}
HH = {"label": "heavy hole", "m_over_m0": 1.92, "n_s_cm2": 3.8e13}
BANDS = [LH, HH]

MU_HALL = {"light hole": 1900.0, "heavy hole": 400.0}
TAU_Q_LH_PS = 0.15

n_tot = LH["n_s_cm2"] + HH["n_s_cm2"]
b_fh = S.fang_howard_b(n_tot, 0.0, M_Z, EPS_R)
F_EFF = QE * (n_tot * 1e4 / 2.0) / (EPS_R * EPS0)

MECHS = {
    "remote charge d=10nm":
        lambda q: S.w_remote_impurity(q, 1e13, 10e-9, b_fh, EPS_R),
    "remote charge d=2nm":
        lambda q: S.w_remote_impurity(q, 1e13, 2e-9, b_fh, EPS_R),
    "roughness L=1nm":
        lambda q: S.w_interface_roughness(q, 0.3e-9, 1e-9, F_EFF),
    "roughness L=3nm":
        lambda q: S.w_interface_roughness(q, 0.3e-9, 3e-9, F_EFF),
    "background impurity":
        lambda q: S.w_background_impurity(q, 1e17, b_fh, EPS_R),
    "dislocations":
        lambda q: S.w_dislocation(q, 1e8, C_LAT, 1.0, EPS_R, b_fh),
}

# Unit rates for each mechanism and band (s^-1 at the reference amplitude).
RATE = {}
for name, wfn in MECHS.items():
    RATE[name] = {}
    for bd in BANDS:
        rq, rt = S._rates(bd, BANDS, wfn, EPS_R, b_fh)
        RATE[name][bd["label"]] = (rq, rt)


def residual(mech_names, tau_q_hh_ps):
    """Best non-negative mixture and its worst relative error."""
    target = []
    for bd in BANDS:
        lab = bd["label"]
        t_tr = S.tau_from_mobility(MU_HALL[lab], bd["m_over_m0"])
        t_q = (TAU_Q_LH_PS if lab == "light hole" else tau_q_hh_ps) * 1e-12
        target += [1.0 / t_q, 1.0 / t_tr]
    target = np.array(target)

    A = np.zeros((4, len(mech_names)))
    for j, nm in enumerate(mech_names):
        row = []
        for bd in BANDS:
            rq, rt = RATE[nm][bd["label"]]
            row += [rq, rt]
        A[:, j] = row

    # Fit in relative terms so all four observables carry equal weight.
    W = np.diag(1.0 / target)
    x, _ = nnls(W @ A, W @ target)
    pred = A @ x
    rel = np.abs(pred - target) / target
    return x, pred, target, rel.max()


LABELS = ["1/tau_q  LH", "1/tau_tr LH", "1/tau_q  HH", "1/tau_tr HH"]

print("=" * 78)
print("Can a mixture of two mechanisms reproduce all four lifetimes?")
print("Worst relative error over the four observables, per pair.")
print("=" * 78)
names = list(MECHS)
print(f"{'mechanism pair':>46}", end="")
for tq in (0.17, 0.205, 0.24):
    print(f"  tau_q,HH={tq:.2f}ps", end="")
print()
best = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pair = [names[i], names[j]]
        line = f"{pair[0] + ' + ' + pair[1]:>46}"
        errs = []
        for tq in (0.17, 0.205, 0.24):
            _, _, _, e = residual(pair, tq)
            errs.append(e)
            line += f"{100*e:16.0f}%"
        print(line)
        best.append((min(errs), pair))

print()
best.sort()
print(f"Best two-mechanism combination: {best[0][1]}, "
      f"worst error {100*best[0][0]:.0f}%")

print()
print("=" * 78)
print("What heavy-hole quantum lifetime would each mechanism REQUIRE?")
print("Given the measured transport lifetimes, each mechanism fixes tau_q/tau_tr")
print("for both bands.  Anchoring on the well-determined light-hole numbers, we")
print("ask what heavy-hole quantum lifetime follows, and compare with the")
print("0.17 to 0.24 ps that Chang et al. estimate.")
print("=" * 78)
t_tr_hh = S.tau_from_mobility(MU_HALL["heavy hole"], HH["m_over_m0"])
t_tr_lh = S.tau_from_mobility(MU_HALL["light hole"], LH["m_over_m0"])
print(f"{'mechanism':>26} {'tau_q,LH implied':>18} {'tau_q,HH implied':>18}"
      f" {'in range?':>11}")
print(f"{'MEASURED':>26} {TAU_Q_LH_PS:16.3f}ps {'0.17 - 0.24':>16}ps")
print("-" * 78)
rows = []
for name in names:
    rq_l, rt_l = RATE[name]["light hole"]
    rq_h, rt_h = RATE[name]["heavy hole"]
    tq_l = t_tr_lh * (rt_l / rq_l)
    tq_h = t_tr_hh * (rt_h / rq_h)
    ok = "yes" if 0.17e-12 <= tq_h <= 0.24e-12 else "NO"
    rows.append({"mechanism": name, "tau_q_LH_ps": tq_l * 1e12,
                 "tau_q_HH_ps": tq_h * 1e12, "hh_in_range": ok == "yes"})
    print(f"{name:>26} {tq_l*1e12:16.3f}ps {tq_h*1e12:16.3f}ps {ok:>11}")

out = os.path.join(os.path.dirname(__file__), "..", "results")
with open(os.path.join(out, "scattering_fit.json"), "w") as fh:
    json.dump({"implied_tau_q": rows,
               "best_pair": {"mechanisms": best[0][1],
                             "worst_rel_error": best[0][0]}}, fh, indent=2)
print("\nwrote results/scattering_fit.json")
