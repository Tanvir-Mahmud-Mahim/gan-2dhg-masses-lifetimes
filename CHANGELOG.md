# Changelog

All notable changes to this repository are listed here, newest first.
No release or git tag has been made, so the entries are grouped by date. The
package carries the version string `2.0.0` (`src/gan2dhg/__init__.py`) since
5 August 2026. Every entry before the documentation entry is read from the
git history.

## Documentation (30 September 2026)

Documentation only; no code, data or result changed.

- README restructured as a step-by-step guide: plain-language overview,
  file tree, installation (including the fact that NumPy 2.0 or newer is
  needed because the code calls `numpy.trapezoid`), quick start, a table of
  all scripts with what each reads and writes, which script makes which
  figure, module table, parameter and measurement sources with a summary of
  the citation audits, built-in checks, notes, version history, and citation.
  All content of the previous README is kept.
- In the list of scripts, `run_well_polarisation.py` now comes before
  `run_dispersion.py`, which reads its output.
- Added this CHANGELOG and `CITATION.cff`.

## 26 September 2026

Heavy-hole quantum mobility from the published oscillations.

- `src/gan2dhg/sdh.py` (new): how strongly each subband's oscillation appears
  in `rho_xx` of a two-carrier gas; later extended with the density-of-states
  factor in sigma_xx, the spin factor from two harmonics and a
  non-perturbative `rho_xx`; cites Dmitriev *et al.*, Rev. Mod. Phys. 84,
  1709 (2012).
- `src/gan2dhg/scatter2d.py`: polarization-fluctuation, misfit-line and
  power-law disorder kernels.
- `data/chang2026_fig2_digitized.json` (new): Fig. 2 of Chang *et al.*
  digitized, first from a rendered page, then at the native resolution of the
  embedded image; provenance record updated.
- New scripts: `run_heavy_quantum_mobility.py`, `run_landau_lorentz.py`,
  `run_revised_mobilities.py`, `run_forward_calibration.py`,
  `run_light_corrected.py`, `disorder_fit.py`; figure scripts updated.
- New results: `heavy_quantum_mobility.json`, `landau_lorentz.json`,
  `revised_mobilities.json`, `forward_calibration.json`,
  `light_corrected_fits.json`.
- Tests: 56, then 61.
- README: new scripts and revised principal results.

## 25 September 2026

Revision of the analysis.

- `src/gan2dhg/landau.py` (new, Landau levels) and `src/gan2dhg/measured.py`
  (new, published values in one place); `kp6_het.py` gains a sparse solver,
  strain and interface-charge options, and later a parameter override.
- `src/gan2dhg/rpa2d.py` (new): two-component RPA mass.
- New scripts: `run_landau.py`, `run_tension.py`, `run_dispersion.py`,
  `run_strain_polarisation.py`, `run_well_het.py`,
  `run_well_polarisation.py`, `run_structure_check.py`, `run_a6.py`,
  `run_consistency.py`, `run_landau_dingle.py`, `run_many_body.py`,
  `run_mixtures.py`, `run_parameter_sensitivity.py`.
- Removed scripts and their results: `run_anchor.py`, `run_scattering.py`,
  `run_scattering_fit.py`, `run_interband.py`, `run_ratio_robust.py`,
  `run_bestfit.py`.
- Measured lifetime ratios corrected (mobility ratio instead of lifetime
  ratio) and the reported range of the Hall fit included.
- Provenance record: revision audit, second citation audit (samples, King
  offset, Hall-fit ranges), light-hole revision sources (Punya and Lambrecht
  2012, Asgari *et al.* 2005).
- Tests: 45, then 51.
- README rewritten for the revised analysis.

## 5 August 2026

Finite barrier and renamed package.

- Package renamed from `gpfet` to `gan2dhg` (version string `2.0.0`);
  `kp6_het.py` (new) solves the well with a finite AlN barrier.
- New scripts: `run_barrier.py`, `run_beyond.py`, `run_formfactor.py`,
  `run_overlap_lifetimes.py`, `run_phaseshift.py`, `run_rashba.py`,
  `run_robust2.py`, `run_bestfit.py`, `figure_overview.py`.
- Derived data regenerated with corrected occupation counting and mass
  derivative.
- Tests: 36.
- Citation audit recorded in the provenance file.
- README updated; manuscript title changed to "Origin of the conflicting hole
  masses in the GaN/AlN two-dimensional hole gas".
- `gitignore.txt` replaced by `.gitignore`.

## 4 August 2026

First version.

- Package `gpfet` with the six-band k.p Hamiltonian and the two-dimensional
  scattering module; provenance record `data/gan_2dhg_measured.yaml`; physics
  test suite (20 tests); analysis scripts `run_masses.py`,
  `run_scattering.py`, `run_scattering_fit.py`, `run_interband.py`,
  `run_ratio_robust.py`, `run_anchor.py` and `figures2.py`, with their
  results.
- License changed from MIT to Apache 2.0.
- Self-consistent envelope-function well solver (`kp6_well.py`),
  `run_well.py`, `run_well_sweep.py` and `figures3.py`; tests: 28.
  `run_masses.py` and `figures2.py` removed.
