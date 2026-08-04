# Hole masses and lifetimes in the GaN/AlN two-dimensional hole gas

Code and derived data for the manuscript *Origin of the conflicting hole masses
in the GaN/AlN two-dimensional hole gas*.

No experiment was performed for this work. Every experimental number used is a
published value, and every one is recorded in
[`data/gan_2dhg_measured.yaml`](data/gan_2dhg_measured.yaml) together with its
source and a note of whether the full text or only the abstract was available
when it was read. That file also carries the record of the citation audit: each
DOI resolved and compared field by field against the bibliography, and each
claim attributed to a reference checked against the primary text.

## What the code does

Three magnetotransport experiments have been applied to the same
polarisation-induced GaN/AlN two-dimensional hole gas, and the band-resolved
parameters they return do not agree. This repository contains the analysis that
resolves two of those disagreements and sharpens the third.

| module | purpose |
| --- | --- |
| `src/gan2dhg/kp6.py` | Six-band wurtzite valence band Hamiltonian, with strain and branch tracking by eigenvector continuity |
| `src/gan2dhg/kp6_well.py` | Self-consistent envelope-function solution of the polarisation well with a hard wall: `k_z -> -i d/dz` coupled to Poisson at fixed sheet density |
| `src/gan2dhg/kp6_het.py` | The same problem with a finite AlN barrier: position-dependent band parameters, symmetric (BenDaniel-Duke) discretisation, vector in-plane wavevector, and the interband Bloch overlap computed from the spinors |
| `src/gan2dhg/scatter2d.py` | Two-dimensional elastic scattering: transport and quantum lifetimes for remote charge, interface roughness, background impurities and dislocations, screened, with interband coupling and an angle-dependent overlap |

GaN parameters are taken from Extended Data Table 1 of Chang *et al.*,
Nat. Electron. **9**, 346 (2026), so that the calculation uses the same inputs
as the measurement it is compared against. AlN parameters are from Rinke
*et al.*, Phys. Rev. B **77**, 075202 (2008), the source from which the GaN
parameters descend, with the AlN spin-orbit splitting from de Carvalho
*et al.*, Appl. Phys. Lett. **97**, 232101 (2010). Nothing is adjusted.

## Reproducing every number in the paper

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # 36 physics tests

python scripts/run_well.py                 # -> results/well.json       hard-wall baseline
python scripts/run_barrier.py              # -> results/barrier.json    finite barrier, offset scanned
python scripts/run_rashba.py               # -> results/rashba.json     tracked spin splitting
python scripts/run_well_sweep.py           # -> results/well_sweep.json masses against density
python scripts/run_anchor.py               # -> results/anchor.json
python scripts/run_scattering.py           # -> results/scattering.json
python scripts/run_scattering_fit.py       # -> results/scattering_fit.json
python scripts/run_interband.py            # -> results/interband.json
python scripts/run_ratio_robust.py         # -> results/ratio_robust.json
python scripts/run_overlap_lifetimes.py    # -> results/overlap_lifetimes.json
python scripts/run_bestfit.py              # -> results/bestfit.json
python scripts/run_formfactor.py           # -> results/formfactor.json
python scripts/run_robust2.py              # -> results/robust2.json
python scripts/run_beyond.py               # -> results/beyond.json
python scripts/run_phaseshift.py           # -> results/phaseshift.json

python scripts/figures3.py                 # -> figures/prb_fig1, prb_fig2 at 1000 dpi
python scripts/figure_overview.py          # -> figures/prb_fig0 at 1000 dpi
```

`run_barrier.py` must be run before `run_overlap_lifetimes.py`,
`run_bestfit.py` and `run_beyond.py`, which read the computed Bloch overlap
from its output; `run_well.py` must be run before `run_formfactor.py` and
`figure_overview.py`. Figures read from the JSON written by the analysis
scripts, so no figure can drift from a number quoted in the text.

## Principal results

- With a finite barrier and no free parameters, the calculation gives a
  heavy-hole mass of 1.92 to 1.99 m0 across the published range of the valence
  band offset, against a measured 1.92 +/- 0.16 m0, and a zero-field light-hole
  mass of 0.26 to 0.33 m0, which brackets the 0.30 m0 obtained by extrapolating
  the measurement to zero field. The dispersion is not the source of the
  reported anomalies.
- The heavy-hole masses from cyclotron resonance and quantum oscillations
  differ because that resonance is overdamped: `omega_c tau = 0.82` at the
  highest field applied.
- The ratio `tau_tr/tau_q` carries neither the disorder amplitude nor the
  effective mass, and its measured value requires long-range forward-peaked
  scattering, not the short-range mechanism to which it has been attributed.
  With the interband Bloch overlap computed rather than assumed, no elastic
  mechanism reproduces the measured ratios of both subbands: over a continuous
  scan of every mechanism the closest simultaneous account is wrong by a factor
  of two, and neither correlated disorder nor scattering beyond the Born
  approximation removes the discrepancy.
- The asymmetric well gives a Rashba splitting that is present but whose size
  is not predicted: it depends strongly and non-monotonically on the valence
  band offset, falling from 2.2 meV at 0.3 eV to 0.4 meV at 0.7 eV and passing
  through zero near 0.9 eV.

## Tests

The suite checks quantities known independently of the implementation rather
than merely exercising it: Hermiticity at random wavevectors, time-reversal
symmetry of the confined spectrum, the closed-form zone-centre eigenvalues, the
accepted unstrained splittings, basal-plane isotropy, the reported pseudomorphic
strain, and the spin-orbit crossover above which the heavy and light labels are
meaningful. The well solver is required to reproduce the bulk masses when the
confining field is switched off, to conserve the total sheet density, to give
the analytic interface field from Poisson's equation, and to return exactly
zero spin splitting when an inversion centre is restored. The occupation rule is
checked against the definition `n = k_F^2 / 4 pi` for a single spin-resolved
branch; the heterostructure operator is required to reproduce the hard-wall
operator when the barrier region is removed; the computed Bloch overlap is
required to be unity for a state with itself; and the tracked spin splitting is
required to remain smooth across the crossing at which a splitting read off
fixed eigenvalue indices collapses. On the scattering side the suite pins the
exact limits `tau_tr/tau_q = 1` for isotropic scattering and `1/2` for pure
backscattering, the invariance of that ratio under both disorder amplitude and
effective mass, the reduction of the coupled two-subband solver to the
single-subband result when the interband overlap vanishes, the equality of the
transport and total cross sections for pure s-wave scattering, the reduction of
the variable-phase solver to the two-dimensional Born phase shift for a weak
potential, and the agreement of the numerically solved linearised Boltzmann
equation with the closed-form transport rate.

## Methodological cautions

Three points are recorded because each affects results and each produces
plausible numbers when handled wrongly.

1. Confinement represented by a fixed wavevector `k_z = pi/w` inserted into the
   bulk Hamiltonian is structurally wrong, not merely imprecise: it inserts
   `k_z` as a number into a coupling linear in `k_z`, whose expectation
   vanishes for a bound state. The proxy returns a light-hole mass near
   0.51 m0, close to the field-averaged measurement and so apparently correct.
   The self-consistent solve returns 0.24 to 0.33 m0 depending on the barrier.
2. Every eigenvalue of the six-band operator is a single spin-resolved branch
   and holds `k_F^2 / 4 pi` carriers. The spin-degenerate rule `k_F^2 / 2 pi`
   describes a Kramers pair; applied to each partner separately it counts every
   spin twice, and the self-consistent loop then converges to half the intended
   sheet density while Poisson's equation is supplied with the full one.
3. Band branches followed by energy order are relabelled wherever two of them
   cross. The light pair crosses the second heavy pair within the occupied
   range of in-plane wavevector, so a spin splitting read off fixed eigenvalue
   indices collapses there by two orders of magnitude. Branches are assigned by
   eigenvector continuity as an assignment problem.

## Licence

See [LICENSE](LICENSE).
