"""Can interactions between the holes make the light holes heavier?

The light holes are a minority (0.8 of 4.6 x 10^13 cm^-2) immersed in a
heavy-hole Fermi sea with six times their density of states.  Band-structure
calculations, including the GW calculation for bulk GaN quoted by Chang et
al., do not contain the dressing of a light hole by the density fluctuations
of this Fermi sea.  This script computes it at the level of the random phase
approximation (on-shell GW, src/gan2dhg/rpa2d.py) for a two-component gas.

1. Benchmark: the strictly 2D electron gas at r_s = 1, 2, 3, 5, against the
   on-shell RPA masses tabulated by Asgari et al., Phys. Rev. B 71, 045323
   (2005), Table I: 1.033, 1.168, 1.322, 1.696.
2. The GaN gas: light (m = 0.27, 0.30 or 0.53 m0, n = 0.8 x 10^13 cm^-2) and
   heavy (m = 1.92 m0, n = 3.8 x 10^13 cm^-2) holes, background dielectric
   constant 10.4 or 9.5, strictly 2D or with the Fang-Howard form factor
   (b = 5.72 nm^-1).  Each component is also computed alone, so that the part
   of the enhancement due to the other component is visible.

Outputs results/many_body.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np                                      # noqa: E402
from gan2dhg import rpa2d as R                          # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
A_BOHR = 0.0529177
ASGARI = {1: 1.033, 2: 1.168, 3: 1.322, 5: 1.696}


def main():
    out = {"benchmark": [], "gan": []}
    for rs, ref in ASGARI.items():
        n = 1.0 / (np.pi * (rs * A_BOHR) ** 2)
        s = R.System([R.Species(1.0, n)], eps=1.0)
        r = R.mass_ratio(s, 0)
        out["benchmark"].append({"r_s": rs, "m_star_over_m": r,
                                 "Asgari_2005": ref})
        print(f"benchmark r_s {rs}: {r:.4f} (Asgari et al. {ref})", flush=True)
    for eps in (10.4, 9.5):
        for ffname, ff in (("strict 2D", None),
                           ("Fang-Howard b = 5.72 nm^-1",
                            R.fang_howard_form_factor(5.72))):
            for mL in (0.27, 0.30, 0.53):
                two = R.System([R.Species(mL, 0.08), R.Species(1.92, 0.38)],
                               eps=eps, ff=ff)
                rec = {"eps": eps, "interaction": ffname, "m_light": mL,
                       "light_in_two_component_gas": R.mass_ratio(two, 0),
                       "heavy_in_two_component_gas": R.mass_ratio(two, 1),
                       "light_alone": R.mass_ratio(
                           R.System([R.Species(mL, 0.08)], eps=eps, ff=ff), 0),
                       "heavy_alone": R.mass_ratio(
                           R.System([R.Species(1.92, 0.38)], eps=eps, ff=ff),
                           0)}
                out["gan"].append(rec)
                print({k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in rec.items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "many_body.json"), "w"), indent=1)
    print("WROTE results/many_body.json")


if __name__ == "__main__":
    main()
