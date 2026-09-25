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

Quantum oscillations, a two-carrier Hall analysis (on one sample) and terahertz
cyclotron resonance (on a second sample of similar design and density) have
been applied to polarization-induced GaN/AlN two-dimensional hole gases, and
the subband-resolved parameters they return do not agree. This repository
contains the analysis: the heavy-hole mass and the difference between the two
heavy-hole masses are accounted for; the light-hole mass and occupation are
shown to be one zero-field discrepancy, traced to the band parameter A6; and
the lifetime ratios are shown to require two coexisting scattering mechanisms.

| module | purpose |
| --- | --- |
| `src/gan2dhg/kp6.py` | Six-band wurtzite valence band Hamiltonian, with strain and branch tracking by eigenvector continuity |
| `src/gan2dhg/kp6_well.py` | Self-consistent envelope-function solution of the polarization well with a hard wall: `k_z -> -i d/dz` coupled to Poisson at fixed sheet density |
| `src/gan2dhg/kp6_het.py` | The same problem with a finite AlN barrier: position-dependent band parameters, symmetric (BenDaniel-Duke) discretization, vector in-plane wavevector, interband Bloch overlap computed from the spinors, optional strain state and interface charge |
| `src/gan2dhg/landau.py` | Landau levels of the six-band envelope operator in a field along the growth axis: exact separation into blocks by the ladder-operator structure, free-electron and scanned valence-band Zeeman terms |
| `src/gan2dhg/scatter2d.py` | Two-dimensional elastic scattering: transport and quantum lifetimes for remote charge, interface roughness, background impurities and dislocations, screened, with the coupled two-subband Boltzmann equation and an angle-dependent overlap |
| `src/gan2dhg/measured.py` | The published values used in the scattering comparison, including the mass-free measured ratio of Hall to quantum mobility |
| `src/gan2dhg/rpa2d.py` | On-shell RPA (GW) quasiparticle mass of a multicomponent two-dimensional gas, checked against published values for the electron gas |

GaN parameters are taken from Extended Data Table 1 of Chang *et al.*,
Nat. Electron. **9**, 346 (2026), so that the calculation uses the same inputs
as the measurement it is compared against. AlN parameters are from Rinke
*et al.*, Phys. Rev. B **77**, 075202 (2008), with the AlN spin-orbit splitting
from de Carvalho *et al.*, Appl. Phys. Lett. **97**, 232101 (2010).
Polarization constants are from Bernardini, Fiorentini and Vanderbilt,
Phys. Rev. B **56**, R10024 (1997). With these parameters nothing is
adjusted. The one parameter that is then rescaled, A6, is fixed by the measured
light-hole density alone (`scripts/run_a6.py`).

## Reproducing every number in the paper

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                     # physics tests

python scripts/run_well.py                     # -> results/well.json               hard-wall baseline
python scripts/run_barrier.py                  # -> results/barrier.json            finite barrier, offset scanned, Bloch overlap
python scripts/run_well_het.py                 # -> results/well_het.json           finite-barrier well at 0.7 eV, saved in full
python scripts/run_rashba.py                   # -> results/rashba.json             tracked spin splitting
python scripts/run_well_sweep.py               # -> results/well_sweep.json         masses against density, finite barrier
python scripts/run_structure_check.py          # -> results/structure_check.json    cyclotron-resonance sample, AlN spin-orbit splitting
python scripts/run_dispersion.py               # -> results/dispersion.json         fine dispersion for Fig. 2(b) of the Letter
python scripts/run_strain_polarisation.py      # -> results/strain_polarisation.json piezoelectric charge, strain relaxation
python scripts/run_well_polarisation.py        # -> results/well_het_pol.json       well with the polarization closure
python scripts/run_landau.py                   # -> results/landau.json             Landau levels, Lifshitz-Kosevich emulation
python scripts/run_landau.py well_het_pol.json landau_pol.json 0   # same, polarization closure
python scripts/run_consistency.py              # -> results/consistency.json         measured densities against computed dispersions
python scripts/run_parameter_sensitivity.py    # -> results/parameter_sensitivity.json each parameter scaled by 0.8 and 1.2
python scripts/run_a6.py ellipticity           # -> results/a6_ellipticity.json      largest |A6| for a bounded Hamiltonian
python scripts/run_a6.py scan A07 A03 B07 B03  # -> results/a6_scan_*.json           A6 against light-hole occupation and mass
python scripts/run_a6.py apply                 # -> results/a6_apply.json, results/well_het_pol03*.json
python scripts/run_landau.py well_het_pol03.json landau_pol03.json         # Landau levels, published A6, 0.3 eV
python scripts/run_landau.py well_het_pol03_A6.json landau_pol03_A6.json   # the same with the rescaled A6
python scripts/run_landau_dingle.py            # -> results/landau_dingle.json       level width fixed by the Dingle slope
python scripts/run_many_body.py                # -> results/many_body.json           RPA mass of the two-component gas
python scripts/run_tension.py                  # -> results/tension.json            coupled two-subband lifetimes, scans, two-carrier fit
python scripts/run_overlap_lifetimes.py        # -> results/overlap_lifetimes.json
python scripts/run_formfactor.py               # -> results/formfactor.json
python scripts/run_robust2.py                  # -> results/robust2.json            warping, local field, temperature
python scripts/run_beyond.py                   # -> results/beyond.json             exact Boltzmann, inelastic bounds, correlated disorder
python scripts/run_phaseshift.py               # -> results/phaseshift.json         beyond the Born approximation
python scripts/run_mixtures.py                 # -> results/mixtures.json           two coexisting mechanisms, inhomogeneity

python scripts/figures3.py                     # -> figures/prb_fig1, prb_fig2 at 1000 dpi
python scripts/figure_overview.py              # -> figures/prb_fig0 at 1000 dpi
```

Run the scripts in the order listed: later ones read the converged potential
(`results/well_het.json`) and the computed Bloch overlap
(`results/barrier.json`). The Landau-level script caches its levels in
`results/landau_levels*.json`; deleting the cache forces a full recomputation,
which takes about an hour and a half on two cores. Figures read from the JSON
written by the analysis scripts, so no figure can drift from a number quoted in
the text.

## Principal results

- With the published parameters and a finite barrier, the calculation gives a
  heavy-hole mass of 1.92 to 1.99 m0 across the published range of the valence
  band offset, against a measured 1.92 +/- 0.16 m0.
- The heavy-hole masses from cyclotron resonance and quantum oscillations differ
  because that resonance is overdamped: `omega_c tau = 0.82` at the highest
  field applied.
- The light holes carry one zero-field discrepancy. With the computed
  dispersions the two measured subband densities cannot share a Fermi level;
  they require an average light-hole mass of 0.46 to 0.49 m0 against 0.28 to
  0.35 m0 computed, the same discrepancy as the measured light-hole masses
  (0.53 m0 from quantum oscillations, 0.57 m0 from cyclotron resonance). The
  linear extrapolation of the field-dependent mass to 0.30 m0 is not a
  zero-field mass.
- Of all band parameters only A6 moves the light subband alone. Rescaled to the
  measured light-hole density (A6 = -4.11 to -4.50 in three of the four
  combinations of offset and closure; published -3.20) it gives a light-hole
  mass of 0.54 to 0.56 m0 (tied to the density through the density of states),
  leaves the heavy holes unchanged, and gives 0.50 to 0.64 m0 for the
  cyclotron-resonance sample (measured 0.57). Its Landau levels give a
  Lifshitz-Kosevich mass of 0.45 to 0.46 m0 at 32 T and 0.45 to 0.58 m0 at
  72 T (reported 0.48 and 0.69), depending on the unknown Zeeman parameter.
  The required value exceeds the published ones and lies close to
  |A6| = 4.44, beyond which the six-band Hamiltonian is unbounded; it is an
  effective parameter.
- Interactions with the heavy-hole Fermi sea (two-component RPA) enhance the
  light-hole mass by 12 to 23 percent and the heavy-hole mass by 11 to 39
  percent.
- The measured ordering of the lifetime ratios (light 5.2, heavy 2.2) cannot
  arise from any single mechanism. Two coexisting mechanisms of different range
  reproduce both ratios within 6 percent and all four mobilities within 21
  percent, but only with an implausibly strong long-range component, which
  supplies three quarters of both quantum rates; the light-hole quantum
  mobility remains the open anomaly. Solving the coupled
  Boltzmann equation in a magnetic field shows that the two-carrier Hall
  analysis is not the cause.

## Tests

The suite checks quantities known independently of the implementation rather
than merely exercising it: Hermiticity, time-reversal symmetry, closed-form
zone-center eigenvalues, accepted splittings, basal-plane isotropy, the
reported strain, the occupation rule `n = k_F^2 / 4 pi` for a single
spin-resolved branch, the reduction of the heterostructure operator to the
hard-wall one, the agreement of the sparse and dense eigen-solvers, the
normalization of the computed Bloch overlap and the continuity of the tracked
spin splitting; for the Landau levels, the parabolic limit, equality of the
spectra at `B` and `-B`, the Onsager count of states and the grid truncation;
for transport, the exact limits `tau_tr/tau_q = 1` and `1/2`, invariance under
disorder amplitude and mass, the two-carrier form of the coupled
magnetoconductivity against an explicit least-squares fit, the reduction to
independent Drude channels, the `s`-wave identity of the cross sections, the
Born limit of the variable-phase solver and the exact solution of the
linearized Boltzmann equation; the piezoelectric polarization formula; the
parameter override, the bound on A6 and the parabolic density condition; and,
for the many-body mass, the limits of the Lindhard function and the published
on-shell mass of the two-dimensional electron gas.

## Methodological cautions

1. Confinement represented by a fixed wavevector `k_z = pi/w` inserted into the
   bulk Hamiltonian is wrong in kind for the coupling linear in `k_z`, whose
   expectation vanishes for a bound state. It returns a light-hole mass near
   0.51 m0, close to the field-averaged measurement and so apparently correct.
2. Every eigenvalue of the six-band operator is a single spin-resolved branch
   and holds `k_F^2 / 4 pi` carriers, not `k_F^2 / 2 pi`.
3. Branches followed by energy order are relabeled wherever two of them cross;
   they must be followed by eigenvector continuity.
4. The measured lifetime ratio should be formed from the two mobilities, which
   need no mass, not from separately quoted lifetimes.

## Licence

See [LICENSE](LICENSE).
