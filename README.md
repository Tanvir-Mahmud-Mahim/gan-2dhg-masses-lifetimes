# Hole masses and lifetimes in the GaN/AlN two-dimensional hole gas

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Code and derived data for the manuscript **"Origin of the conflicting hole
masses in the GaN/AlN two-dimensional hole gas"**. The figure script calls it
a Letter for *Physical Review B*, with Supplemental Material. No journal
reference or DOI is recorded in this repository.

Code copyright: Tanvir M. Mahim, A.S.M. Mohsin, and M. Mosaddequr Rahman (the
copyright holders named in [`LICENSE`](LICENSE)). The author list of the
manuscript is not recorded in this repository.

- Repository: https://github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes
- Record of every published number used, with its source and the citation audit:
  [`data/gan_2dhg_measured.yaml`](data/gan_2dhg_measured.yaml)

No experiment was performed for this work. Every experimental number used is a
published value, and every one is recorded in
[`data/gan_2dhg_measured.yaml`](data/gan_2dhg_measured.yaml) together with its
source and a note of whether the full text or only the abstract was available
when it was read. That file also carries the record of the citation audit: each
DOI resolved and compared field by field against the bibliography, and each
claim attributed to a reference checked against the primary text
(summarised in [Section 8](#8-where-the-numbers-come-from)).

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](#5-the-scripts-step-by-step)
6. [Which script makes which figure](#6-which-script-makes-which-figure)
7. [The Python modules](#7-the-python-modules)
8. [Where the numbers come from](#8-where-the-numbers-come-from)
9. [Built-in checks](#9-built-in-checks)
10. [Notes on the calculations](#10-notes-on-the-calculations)
11. [Version history](#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

---

## 1. The idea in one minute

When a thin layer of gallium nitride (GaN) is grown on aluminium nitride (AlN),
the built-in electric polarization of the two crystals pulls a thin sheet of
mobile positive charge carriers, called **holes** (missing electrons), against
the interface. This sheet is a **two-dimensional hole gas** (2DHG). It forms
without acceptor doping.

The holes fall into two groups, **heavy holes** and **light holes**, which
behave as if they had different masses. This apparent mass is the
**effective mass**, written in units of the free-electron mass m0. Each group
fills its own **subband** (a set of allowed energies). How fast the holes lose
their motion to disorder is described by a **lifetime**, or equivalently a
**mobility**.

Three kinds of published measurement have probed this gas:

- **quantum oscillations**: the resistance oscillates in a strong magnetic
  field (up to 72 T), and the pattern gives each subband's density, mass and
  "quantum mobility" (Chang *et al.*, 2026). The quantum mobility measures how
  long a hole keeps its quantum state before any scattering event, small-angle
  or large;
- **a two-carrier Hall analysis** on the same sample, which gives each
  subband's "Hall mobility" (how easily holes drift in an electric field;
  small-angle scattering hardly reduces it) from low-field data (up to 9 T) (Chang *et al.*,
  using the fitting procedure of Dill *et al.*, 2025);
- **terahertz cyclotron resonance**: absorption of terahertz light at a
  frequency set by the mass, in fields up to 31 T, on a second sample of similar
  design and density (Wang *et al.*, 2025).

The subband-resolved numbers they return do not agree. This repository contains
the analysis of these disagreements. It computes the band structure of the gas
from a **six-band k·p model** (Chuang and Chang, 1996): a 6 x 6 matrix formula
for the energies of the valence-band states near the top of the valence band,
controlled by tabulated parameters such as A1 to A6. It solves this together with
the electrostatics of the gas, computes the energy levels in a magnetic field
(**Landau levels**), models the scattering of holes by several kinds of
disorder, and re-analyses the published oscillation data.

In the words of the original description: the heavy-hole mass and the
difference between the two heavy-hole masses are accounted for; the light-hole
mass and occupation are shown to be one zero-field discrepancy, traced to the
band parameter A6; the heavy-hole quantum mobility is re-derived from a Dingle
analysis of the published oscillations and from the ratio of their amplitudes;
and the four measured mobilities are shown to fix the spectrum of the
long-range disorder.

### Principal results

These are the results as stated in the README before this documentation
update, kept word for word.

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
- With Lorentzian Landau levels of known lifetime the emulated Dingle analysis
  returns 348 to 430 cm2/Vs for an input of 368 (rescaled A6, fixed chemical
  potential), so the band structure does not produce the light-hole Dingle
  slope.
- The measured ordering of the lifetime ratios (light 5.2, heavy 2.2) cannot
  arise from any single mechanism. Solving the coupled Boltzmann equation in a
  magnetic field shows that the two-carrier Hall analysis is not the cause.
- The heavy-hole quantum mobility of 167 to 200 cm2/Vs was estimated from the
  field at which the heavy-hole oscillations appear. At 60 to 67 T the heavy
  holes carry 95 percent of sigma_xx; with the density-of-states factor in
  sigma_xx (Dmitriev et al., Rev. Mod. Phys. 84, 1709 (2012)) the same relative
  oscillation moves `rho_xx` 6.0 times more for the heavy holes. The
  native-resolution digitization of Fig. 2 of Chang et al. returns their
  light-hole Dingle mobility (352 to 385 cm2/Vs at 1.8 to 6.0 K, reported
  368 +/- 14) and heavy-hole mass (1.97 +/- 0.13 m0, reported 1.92 +/- 0.16).
  A joint Dingle fit of the heavy-hole oscillation at seven temperatures gives
  74 cm2/Vs (95 percent range 58 to 97), which a non-perturbative model of
  `rho_xx` calibrates to 94 (77 to 113); the heavy-to-light amplitude ratio,
  with the light-hole spin factor 0.745 measured from the second harmonic,
  gives 95 (90 to 101) to first order and 96 (87 to 106) without that
  approximation. The adopted value is 95 cm2/Vs; the heavy-hole ratio becomes
  4.2.
- A spread of the local Fermi level cannot replace long-range scattering: it
  would make the heavy-hole oscillation 4 to 20 times weaker than measured.
- With interface roughness, a long-range component with |V(q)|^2 ~ q^-p fits
  the four mobilities within 3 percent for any heavy-hole value from 75 to
  200 cm2/Vs, with p rising monotonically; at 95, p = 3.55 (3.2 if the
  light-hole value is corrected like the heavy-hole one), between charged lines
  in the interface (p = 3) and charged lines threading the gas (p = 4).
  Combinations that fit within 2 percent all contain threading line charges
  with N f^2 = 2 to 4e9 cm^-2, far above the substrate dislocation density; a
  relaxation of the GaN by 0.35 to 0.5 percent would create both misfit and
  threading lines. Their origin is open.

A few terms used above, explained once: the **Fermi level** is the energy up to
which the subbands are filled; **Lifshitz-Kosevich (LK)** analysis extracts a
mass from how the oscillations weaken with temperature; a **Dingle** analysis
extracts the quantum mobility from how they weaken at lower field; **RPA**
(random phase approximation) is a common approximation for the interaction
between holes; `rho_xx` and `sigma_xx` are the longitudinal resistivity and
conductivity; `omega_c tau` is the number of cyclotron turns a hole makes
before it scatters (below about 1 a resonance is overdamped, that is,
smeared out); the **Zeeman** splitting is the energy splitting of the two spin
states in a magnetic field, and the "Zeeman parameter" sets its size.

---

## 2. What is in this repository

```
gan-2dhg-masses-lifetimes/
|-- README.md               this guide
|-- CHANGELOG.md            what changed, by date (from the git history)
|-- CITATION.cff            citation details (drives the "Cite this repository" button)
|-- LICENSE                 Apache-2.0 license
|-- requirements.txt        Python packages to install
|-- .gitignore
|-- data/
|   |-- gan_2dhg_measured.yaml        every published value used, its source, and the citation audits
|   `-- chang2026_fig2_digitized.json Fig. 2a, 2c and 2f of Chang et al. read at native image resolution
|-- src/gan2dhg/            the Python package (see Section 7)
|   |-- __init__.py         package docstring and version string ("2.0.0")
|   |-- constants.py        physical constants taken from scipy.constants, and unit helpers
|   |-- kp6.py              six-band wurtzite valence-band Hamiltonian and GaN parameters
|   |-- kp6_well.py         self-consistent well with a hard wall at the interface
|   |-- kp6_het.py          self-consistent well with a finite AlN barrier; AlN parameters
|   |-- landau.py           Landau levels of the six-band operator in a magnetic field
|   |-- scatter2d.py        two-dimensional elastic scattering, transport and quantum lifetimes
|   |-- measured.py         the published values used in the scattering comparison
|   |-- sdh.py              weight of each subband's oscillation in rho_xx; non-perturbative rho_xx
|   `-- rpa2d.py            RPA (GW) quasiparticle mass of a multicomponent 2D gas
|-- scripts/                analysis and figure scripts (see Section 5)
|   |-- run_*.py            27 analysis scripts, each writing one or more files in results/
|   |-- disorder_fit.py     shared helper: fits disorder models to the four measured mobilities
|   |-- figures3.py         draws figures/prb_fig1 and prb_fig2
|   `-- figure_overview.py  draws figures/prb_fig0
|-- results/                41 JSON files written by the scripts (committed, so figures can be redrawn)
`-- tests/
    `-- test_kp6.py         61 physics tests (pytest)
```

`results/` is part of the repository: the computed numbers can be read from
it, and the figures redrawn from it, without recomputing anything. The largest files there are the
cached Landau levels (`results/landau_levels*.json`, 0.6 to 2.7 MB each).
The figure scripts write to a folder `figures/`, which they create; it is not
stored in the repository.

---

## 3. Installation

The code was checked here with **Python 3.11** (3.11.15). The repository does
not state a minimum Python version.

```
pip install -r requirements.txt
```

This installs `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7`, `pyyaml>=6.0`
and `pytest>=7.0`. On 2026-09-30 it installed numpy 2.4.6, scipy 1.17.1,
matplotlib 3.11.2, pyyaml 6.0.3 and pytest 9.1.1.

**NumPy 2.0 or newer is needed in practice.** The code calls
`numpy.trapezoid`, which first appeared in NumPy 2.0, although
`requirements.txt` still allows `numpy>=1.24`.

`pyyaml` is not imported by any script; it is there for reading
`data/gan_2dhg_measured.yaml`. There is no installable package: the scripts
that use it and the test file add `src/` to the Python path themselves, and
all file paths are relative to the script, so the commands work from any
working directory.

**Fonts (optional).** The figure scripts ask for the serif font "Nimbus Roman"
and fall back to "Liberation Serif" or "DejaVu Serif" when it is missing; in
that case matplotlib prints `findfont` warnings and the figures are still
written.

---

## 4. Quick start: three ways to use the code

Run all commands from the repository folder.

### Way A: check that everything works

```
python -m pytest tests/ -q
```

All 61 tests must pass. Measured here: `61 passed in 643.26s (0:10:43)` on a
shared 2-core machine that was heavily loaded by other jobs; times on another
machine will differ.

### Way B: redraw the figures from the committed results (about 20 seconds)

```
python scripts/figure_overview.py      # -> figures/prb_fig0.png and .pdf
python scripts/figures3.py             # -> figures/prb_fig1 and prb_fig2, .png and .pdf
```

Measured here: 7.5 s and 11.5 s. The figures are written at 1000 dpi.

### Way C: recompute the results

Run the scripts of [Section 5](#5-the-scripts-step-by-step) in the order
given there, then Way B. The scripts **overwrite the committed files in
`results/`**; `git diff --stat results/` shows what changed. Most scripts were
not re-timed for this guide. The only run times recorded in the repository are
those of the finite-barrier offset scan (`run_barrier.py`, whose output records a run time of 2347 s) and
a full recomputation of the Landau levels (about an hour and a half on two
cores; see the notes under the table in Section 5). `run_well.py` and
`run_rashba.py` did not finish within 15 minutes on the machine used here.

---

## 5. The scripts, step by step

The order below is the order of the original README, with one change:
`run_well_polarisation.py` is moved before `run_dispersion.py`, because
`run_dispersion.py` reads its output (`results/well_het_pol.json`).
"Reads" lists the files that a script needs (in `results/` unless another
folder is given); a script that
imports another script reads what that one reads.

| Step | Command | What it does | Reads (in `results/` unless a folder is given) | Time* | Writes (in `results/` unless a folder is given) |
|---|---|---|---|---|---|
| 0 | `python -m pytest tests/ -q` | 61 physics tests (Section 9) | - | 10 min 43 s | nothing |
| 1 | `python scripts/run_well.py` | Hard-wall baseline: self-consistent six-band well at 4.6e13 cm^-2 | - | long (stopped unfinished after 15 min here) | `well.json` |
| 2 | `python scripts/run_barrier.py` | Finite AlN barrier at offsets 0.3, 0.5, 0.7, 0.8 eV; sensitivity to each choice; computed Bloch overlap (how strongly a light-hole state and a heavy-hole state overlap on the scale of the crystal cell, which sets how easily disorder scatters a hole from one subband to the other) | - | 2347 s recorded in the output (machine not recorded) | `barrier.json` |
| 3 | `python scripts/run_well_het.py` | Finite-barrier well at 0.7 eV, saved in full; the potential used by later scripts | - | not re-timed | `well_het.json` |
| 4 | `python scripts/run_rashba.py` | Spin (Rashba) splitting, with branches tracked by eigenvector continuity and checked for grid convergence | - | long (stopped unfinished after 15 min here) | `rashba.json` |
| 5 | `python scripts/run_well_sweep.py` | Masses and occupations against sheet density (2.0 to 6.5e13 cm^-2) at offsets 0.3 and 0.7 eV | - | not re-timed | `well_sweep.json` |
| 6 | `python scripts/run_structure_check.py` | The cyclotron-resonance sample (8.2 nm GaN, 5.25e13 cm^-2) and the AlN spin-orbit splitting (19, 21.7, 23.5 meV) | - | not re-timed | `structure_check.json` |
| 7 | `python scripts/run_strain_polarisation.py` | Piezoelectric interface charge; strain relaxed from pseudomorphic towards relaxed | - | not re-timed | `strain_polarisation.json` |
| 8 | `python scripts/run_well_polarisation.py` | Well with the interface charge from the calculated polarization (the "polarization closure") | - | not re-timed | `well_het_pol.json` |
| 9 | `python scripts/run_dispersion.py` | Fine in-plane dispersion of the occupied subbands, for both closures | `well_het.json`, `well_het_pol.json` | 5 min 57 s | `dispersion.json` |
| 10 | `python scripts/run_landau.py` | Landau levels of the 0.7 eV well; Lifshitz-Kosevich emulation for three Zeeman parameters | `well_het.json` | see note** | `landau.json`, cache `landau_levels.json` |
| 11 | `python scripts/run_landau.py well_het_pol.json landau_pol.json 0` | The same for the polarization closure, Zeeman parameter 0 only | `well_het_pol.json` | see note** | `landau_pol.json`, cache `landau_levels_well_het_pol.json` |
| 12 | `python scripts/run_consistency.py` | Can the two measured subband densities share one Fermi level with the computed dispersions? | - | not re-timed | `consistency.json` |
| 13 | `python scripts/run_parameter_sensitivity.py` | Each band parameter scaled by 0.8 and 1.2 in turn | - | not re-timed | `parameter_sensitivity.json` |
| 14 | `python scripts/run_a6.py ellipticity` | Largest \|A6\| for which the six-band Hamiltonian is bounded | - | 2 s | `a6_ellipticity.json` |
| 15 | `python scripts/run_a6.py scan A07 A03 B07 B03` | A6 scaled until the light-hole occupation equals the measured one, for the two electrostatic closures (A: measured density, B: polarization; Section 10) at offsets 0.7 and 0.3 eV | - | not re-timed | `a6_scan_A07.json`, `a6_scan_A03.json`, `a6_scan_B07.json`, `a6_scan_B03.json` |
| 16 | `python scripts/run_a6.py apply` | Uses the fitted A6: grid check, cyclotron-resonance sample, and the two wells for the Landau levels | `a6_scan_*.json` | not re-timed | `a6_apply.json`, `well_het_pol03.json`, `well_het_pol03_A6.json` |
| 17 | `python scripts/run_landau.py well_het_pol03.json landau_pol03.json` | Landau levels, published A6, 0.3 eV | `well_het_pol03.json` | see note** | `landau_pol03.json`, cache `landau_levels_well_het_pol03.json` |
| 18 | `python scripts/run_landau.py well_het_pol03_A6.json landau_pol03_A6.json` | The same with the rescaled A6 | `well_het_pol03_A6.json` | see note** | `landau_pol03_A6.json`, cache `landau_levels_well_het_pol03_A6.json` |
| 19 | `python scripts/run_landau_dingle.py` | Level width chosen so that the emulated Dingle analysis returns the measured 368 cm2/Vs; LK analysis repeated | the two wells of step 16 and the level caches of steps 17 and 18 | not re-timed | `landau_dingle.json` |
| 20 | `python scripts/run_landau_lorentz.py` | Lorentzian levels of known lifetime: does the band structure change the Dingle slope? | the same | not re-timed | `landau_lorentz.json` |
| 21 | `python scripts/run_many_body.py` | Two-component RPA mass; benchmark against the 2D electron gas | - | 7 min 24 s | `many_body.json` |
| 22 | `python scripts/run_tension.py` | Can any single elastic mechanism give both measured lifetime ratios? Coupled two-subband Boltzmann equation, also in a magnetic field | `barrier.json` | not re-timed | `tension.json` |
| 23 | `python scripts/run_overlap_lifetimes.py` | Lifetimes with the computed interband overlap and the Rashba splitting resolved | `barrier.json` | 1 s | `overlap_lifetimes.json` |
| 24 | `python scripts/run_formfactor.py` | Form factors from the computed hole distribution instead of the variational (Fang-Howard) one | `well.json` | 2 s | `formfactor.json` |
| 25 | `python scripts/run_robust2.py` | Warping, local-field factor, temperature | - | 9 s | `robust2.json` |
| 26 | `python scripts/run_beyond.py` | Exact Boltzmann solution, inelastic bounds, correlated disorder | `barrier.json` | not re-timed | `beyond.json` |
| 27 | `python scripts/run_phaseshift.py` | Beyond the Born approximation (which treats the disorder as a weak, first-order disturbance): exact phase shifts for a screened centre | - | 4 min 11 s | `phaseshift.json` |
| 28 | `python scripts/run_mixtures.py` | Two coexisting mechanisms; inhomogeneity | `tension.json` | not re-timed | `mixtures.json` |
| 29 | `python scripts/run_heavy_quantum_mobility.py` | Heavy-hole quantum mobility from the digitized oscillations: digitization checks, envelope and ratio routes | `data/chang2026_fig2_digitized.json` | not re-timed | `heavy_quantum_mobility.json` |
| 30 | `python scripts/run_forward_calibration.py` | Non-perturbative calibration of both routes | `heavy_quantum_mobility.json` | not re-timed | `forward_calibration.json` |
| 31 | `python scripts/run_revised_mobilities.py` | Adopted heavy-hole value, spectral exponent p, combinations of mechanisms | `heavy_quantum_mobility.json`, `forward_calibration.json` | not re-timed | `revised_mobilities.json` |
| 32 | `python scripts/run_light_corrected.py` | The same fits with the light-hole value corrected like the heavy-hole one | `forward_calibration.json`, `revised_mobilities.json` | not re-timed | `light_corrected_fits.json` |
| 33 | `python scripts/figures3.py` | Draws `prb_fig1` and `prb_fig2` (Letter Figs. 2 and 3; Section 6) | see Section 6 | 11.5 s | `figures/prb_fig1`, `figures/prb_fig2` |
| 34 | `python scripts/figure_overview.py` | Draws the overview figure `prb_fig0` (Letter Fig. 1) | `well_het.json` | 7.5 s | `figures/prb_fig0` |

\*Times measured here on a shared 2-core machine that was also running other
jobs (load average 2 to 4), with the committed `results/` in place;
times on another machine will differ. "Not re-timed" means the step was not
run for this guide. "Stopped unfinished" means the step was started here but
stopped after 15 minutes, so its full run time is not known (`run_well.py` had
completed 3 of at most 90 self-consistency iterations by then). Every analysis
step that finished here reproduced the committed file exactly, except
`dispersion.json`, whose numbers differ by at most 2e-11 (floating-point
rounding).

\*\*`run_landau.py` stores the computed levels in a cache file
(`results/landau_levels*.json`, committed). With the cache present it reuses
the levels and only repeats the analysis. Deleting the cache forces a full
recomputation, which the authors record as taking about an hour and a half on
two cores; the script uses two worker processes.

`scripts/disorder_fit.py` is not run by itself: it holds the shared fitting
of disorder models to the four mobilities, used by steps 29 to 32.
`run_landau_dingle.py` accepts other well files as arguments (their levels must
first be cached by `run_landau.py`); by default it uses the two wells of step 16.
`run_landau_lorentz.py` takes no arguments: its two wells are fixed in the
script (the two of step 16).

---

## 6. Which script makes which figure

Figures read their data from the JSON written by the analysis scripts, so
they cannot drift from the computed numbers. The exceptions are numbers written
into the scripts themselves: the labels of `prb_fig0`; in `prb_fig1`, the
measured points of panel (c) (1.92 +/- 0.16 and 0.53 +/- 0.01 m0, drawn at
4.6 x 10^13 cm^-2) and the cyclotron-resonance point of panel (d) (0.57 m0 at
31 T); and panels (a) and (b) of `prb_fig2`. The file names are those the scripts
write. The Letter figure numbers are taken from comments in the scripts
(`run_well_het.py`: "Figs. 2(a) and 2(b)"; `run_dispersion.py`: "Fig. 2(b) of
the Letter (figures/prb_fig1, panel b)"; `run_well_sweep.py`: "Fig. 2(c)";
`run_tension.py`: "Fig. 3(c)"), and from the provenance file, which calls the
heterostructure drawing "Fig. 1(a)".

| File | Letter figure | Panels | Data from | Drawn by |
|---|---|---|---|---|
| `figures/prb_fig0` | Fig. 1 | (a) the measured heterostructure of Chang *et al.* with an expanded view of the interface and the computed hole density; (b) what each probe returns, where they disagree, and what this work concludes | `well_het.json` (step 3); the other labels are written into the script | `figure_overview.py` |
| `figures/prb_fig1` | Fig. 2 | (a) self-consistent well and hole distribution; (b) in-plane dispersion with the Fermi wavevectors; (c) mass against sheet density, measured points and the rescaled-A6 light mass; (d) light-hole LK mass from the computed Landau levels against the reported field dependence | `well_het.json`, `dispersion.json`, `well_sweep.json`, `a6_apply.json`, `landau_dingle.json`, `landau.json`; the measured points in (c) and the cyclotron-resonance point in (d) are written into the script | `figures3.py` (`figure1()`) |
| `figures/prb_fig2` | Fig. 3 | (a) `omega_c tau` against field for the cyclotron-resonance masses and lifetimes; (b) angular character of scattering; (c) the two lifetime ratios, computed for each mechanism against the reported values and the revised heavy-hole value | (a) and (b) are computed inside the script from the values of Wang *et al.*; (c) `tension.json`, `revised_mobilities.json`, `src/gan2dhg/measured.py` | `figures3.py` (`figure2()`) |

Each figure is written as `.png` and `.pdf`.

---

## 7. The Python modules

| Module | Purpose |
| --- | --- |
| `src/gan2dhg/kp6.py` | Six-band wurtzite valence band Hamiltonian (Chuang and Chang, Phys. Rev. B 54, 2491 (1996)), with strain and branch tracking by eigenvector continuity; the GaN parameter set |
| `src/gan2dhg/kp6_well.py` | Self-consistent envelope-function solution of the polarization well with a hard wall: `k_z -> -i d/dz` coupled to Poisson at fixed sheet density |
| `src/gan2dhg/kp6_het.py` | The same problem with a finite AlN barrier: position-dependent band parameters, symmetric (BenDaniel-Duke) discretization (a way of writing the equations that keeps them consistent where the material changes at the interface), vector in-plane wavevector, interband Bloch overlap computed from the spinors, optional strain state and interface charge; the AlN parameter set |
| `src/gan2dhg/landau.py` | Landau levels of the six-band envelope operator in a field along the growth axis: exact separation into blocks by the ladder-operator structure, free-electron and scanned valence-band Zeeman terms |
| `src/gan2dhg/scatter2d.py` | Two-dimensional elastic scattering: transport and quantum lifetimes for remote charge, interface roughness, background impurities, threading dislocations, charged lines in the interface (misfit dislocations), fluctuations of the interface polarization charge and a power-law spectrum, screened, with the coupled two-subband Boltzmann equation and an angle-dependent overlap |
| `src/gan2dhg/measured.py` | The published values used in the scattering comparison, including the mass-free measured ratio of Hall to quantum mobility |
| `src/gan2dhg/sdh.py` | How strongly each subband's density-of-states oscillation appears in `rho_xx` of a two-carrier gas (through the scattering rates and through the density-of-states factor of sigma_xx), the heavy-hole quantum mobility implied by a measured ratio of the two oscillation amplitudes, the spin factor from two harmonics, and a non-perturbative `rho_xx` with Lorentzian Landau levels to all harmonics |
| `src/gan2dhg/rpa2d.py` | On-shell RPA (GW) quasiparticle mass of a multicomponent two-dimensional gas, checked against published values for the electron gas |
| `src/gan2dhg/constants.py` | Physical constants read from `scipy.constants` at run time, and unit helpers. Its docstring says "CODATA 2018", but the values are whatever the installed SciPy provides (SciPy 1.17.1, used here, provides CODATA 2022). `kp6.py`, `figures3.py` and the tests instead write the electron mass in directly as 9.1093837015e-31 kg (the CODATA 2018 value); SciPy 1.17.1 gives 9.1093837139e-31 kg, a relative difference of about 1e-8 |
| `src/gan2dhg/__init__.py` | Package docstring and `__version__ = "2.0.0"` |

---

## 8. Where the numbers come from

**The rule.** The provenance file states: "No number enters the manuscript
unless verified: true." Each entry records the source it was read from and how
it was read: `primary_pdf` (the full text was opened and the number read from
it), `abstract` (only the abstract was accessible) or `bibliographic_record`.
In its own words, "Nothing in this file was taken from memory or from a
secondary summary."

### GaN band parameters (no adjustment)

GaN parameters are taken from Extended Data Table 1 of Chang *et al.*,
Nat. Electron. **9**, 346 (2026), doi:10.1038/s41928-026-01590-8, read from
the arXiv full text (arXiv:2501.16213), so that the calculation uses the same
inputs as the measurement it is compared against. That table attributes the
A parameters to Rinke *et al.*, Phys. Rev. B **77**, 075202 (2008), and the
deformation potentials to Yan *et al.*, Phys. Rev. B **90**, 125118 (2014).
Values as they appear in `src/gan2dhg/kp6.py` (`GAN`):

| Quantity | Value |
|---|---|
| A1 ... A6 | -5.947, -0.528, 5.414, -2.512, -2.510, -3.202 |
| crystal-field and spin-orbit splitting | 0.010 eV, 0.017 eV |
| lattice constants a, c | 3.189, 5.185 Angstrom |
| elastic constants C11, C12, C13, C33, C44 | 390, 145, 106, 398, 105 (GPa; `run_beyond.py` uses 390e9 and 398e9 Pa) |
| deformation potentials acz, act, (acz - D1), (act - D2) | -11.3, -4.9, -6.07, -8.88 eV (D1 and D2 follow from these) |
| deformation potentials D3, D4, D5, D6 | 5.38, -2.69, -2.56, -3.88 eV |

GaN is taken as pseudomorphically strained to the AlN lattice constant
(3.112 Angstrom), which gives an in-plane strain of -2.41 percent and
+1.29 percent along the growth axis (`kp6.biaxial_strain_on_AlN`). Chang
*et al.* state a 2.4 percent compressive strain; the provenance file notes
that it is stated, not measured.

### AlN, polarization, band offset

- **AlN** (`kp6_het.ALN_KP`): A1 to A6 = -3.991, -0.311, 3.671, -1.147,
  -1.329, -1.952 and crystal-field splitting -0.295 eV from Rinke *et al.*
  (2008). The A parameters are from Table V. For the crystal-field splitting
  the sources in the repository disagree: `kp6_het.py` cites Table III, while
  the provenance file lists it with the Table V values it re-checked. Rinke *et al.* do not
  give the AlN spin-orbit splitting; it is taken as 22 meV from de Carvalho *et al.*, Appl. Phys. Lett.
  **97**, 232101 (2010) (21.7 meV parallel, 23.5 meV perpendicular to the c
  axis). 19, 21.7 and 23.5 meV give the same masses to four figures
  (`results/structure_check.json`).
- **Polarization constants**: Bernardini, Fiorentini and Vanderbilt, Phys. Rev.
  B **56**, R10024 (1997), Table II: P_sp = -0.081 and -0.029 C/m2;
  e33 = 1.46 and 0.73 C/m2; e31 = -0.60 and -0.49 C/m2 for AlN and GaN. The
  interface charge is the discontinuity of the total polarization (Ambacher
  *et al.*, J. Appl. Phys. **85**, 3222 (1999), cited only for this principle).
- **Valence band offset**: not fixed; scanned. Rizzi *et al.* (1999) give
  0.3 +/- 0.1 eV for this growth order; King *et al.* (1998) give 0.5 +/- 0.2
  and 0.8 +/- 0.2 eV. The offsets scanned, 0.3 to 0.8 eV, span these central
  values.
- **Dielectric constant**: 10.4 is used throughout (`eps_r=10.4`); the
  many-body script also tries 9.5. The source of these two values is not
  recorded in the code or the provenance file.

With these parameters nothing is adjusted. The one parameter that is then
rescaled, A6, is fixed by the measured light-hole density alone
(`scripts/run_a6.py`). For comparison, Punya and Lambrecht, Phys. Rev. B
**85**, 195147 (2012), give GaN A6 = -1.55 (-3.31 in the quasi-cubic
approximation). The A6 value of Vurgaftman and Meyer (2003) is not quoted
because that paper was not accessible.

### The measurements

The two experiments were made on **different samples**:

| | Quantum oscillations and Hall (Chang *et al.* 2026) | Cyclotron resonance (Wang *et al.*, Appl. Phys. Lett. **126**, 213102 (2025)) |
|---|---|---|
| GaN layer | 15 nm, top 5 nm Mg-doped | 8.2 nm, undoped, no Mg-doped cap |
| below it | 10 x [Al0.95Ga0.05N, 2-3 monolayers / 25 nm AlN], 500 nm AlN buffer, bulk Al-polar AlN (dislocations below 1e4 cm^-2) | 700 nm AlN buffer |
| method | pulsed field to 72 T, 1.8 to 15 K; Hall fit 0 to 9 T at 3 K | time-domain terahertz spectroscopy, pulsed field to 31 T, 8 K |

(Fig. 1a of Chang *et al.* draws 5.5 nm of p-GaN above 10 nm of undoped GaN;
the Methods text gives a 15 nm GaN layer whose last 5 nm are Mg-doped. Both
are recorded; the fits change by less than 2 percent between them.)

Values used (from `data/gan_2dhg_measured.yaml`, sections `sdh` and
`cyclotron_resonance`; the scattering comparison takes them from
`src/gan2dhg/measured.py`):

| Quantity | Light holes | Heavy holes |
|---|---|---|
| oscillation frequency | 166 +/- 2 T | 795 +/- 24 T |
| sheet density (quantum oscillations) | (0.80 +/- 0.01) x 10^13 cm^-2 | (3.8 +/- 0.1) x 10^13 cm^-2 |
| mass (quantum oscillations) | 0.53 +/- 0.01 m0 (32 to 72 T); 0.48 at 32 T, 0.69 at 72 T, 0.30 extrapolated to zero field | 1.92 +/- 0.16 m0 |
| quantum mobility | 368 +/- 14 cm2/Vs (Dingle analysis) | 167 to 200 cm2/Vs (estimated from the oscillation onset at 50 to 60 T) |
| Hall mobility, 3 K | about 1900 (fit range 1858 to 1986) cm2/Vs | about 400 (381 to 464) cm2/Vs |
| Hall to quantum mobility, as stated | about 5 | 2 to 2.5 |
| mass (cyclotron resonance) | 0.57 +/- 0.01 m0 | 2.6 +/- 0.2 m0 |
| sheet density (cyclotron resonance) | (0.65 +/- 0.03) x 10^13 cm^-2 | (4.6 +/- 0.2) x 10^13 cm^-2 |
| scattering time (cyclotron resonance) | (4.0 +/- 0.2) x 10^-13 s | (3.9 +/- 0.2) x 10^-13 s |

The calculations use the total sheet density 4.6 x 10^13 cm^-2
(0.8 + 3.8). The measured lifetime ratio is formed from the two mobilities,
which need no mass: 1900/368 = 5.16 for the light holes, 400/(167 to 200) =
2.00 to 2.40 for the heavy holes. The quoted light-hole quantum lifetime,
0.15 ps, is not consistent with 368 cm2/Vs at 0.53 m0 (which gives 0.111 ps),
so it is not used. Theory values quoted by Chang *et al.* (light 0.29 m0 k.p,
0.27 m0 GW; heavy 1.6 to 2.1 m0 k.p, 1.93 m0 GW) are recorded separately and
marked as theory, not measurement.

**Digitized data.** `data/chang2026_fig2_digitized.json` holds Fig. 2a and 2c
of arXiv:2501.16213v1 read at the native resolution of the image embedded in
the PDF (2498 x 1677 pixels, 0.24 ohm per pixel, ten temperatures), with the
axis calibration, together with the normalized heavy-hole points of Fig. 2f.
It replaces an earlier digitization at 1224 x 1584 pixels. The heavy-to-light
amplitude ratio, the envelope fit (74 cm2/Vs raw, 94 calibrated), the
light-hole spin factor (0.745) and the adopted heavy-hole quantum mobility
(95 cm2/Vs) are results of this work, not published values.

**Other sources used as benchmarks or models.** Asgari *et al.*, Phys. Rev. B
**71**, 045323 (2005), Table I (on-shell RPA masses of the 2D electron gas,
1.033, 1.168, 1.322, 1.696 at r_s = 1, 2, 3, 5), used only as a numerical
benchmark. Dmitriev, Mirlin, Polyakov and Zudov, Rev. Mod. Phys. **84**, 1709
(2012), Sec. II.C.2, for the response of sigma_xx to the density-of-states
oscillation. Dill *et al.*, J. Appl. Phys. **137**, 025702 (2025), read from
the arXiv version, for the two-carrier fitting procedure and the observation
that both mobilities saturate below 20 K; its erratum (J. Appl. Phys. **139**,
249901 (2026)) was not accessible, and no number in the manuscript is taken
from Dill *et al.* or its erratum.

### The citation audits recorded in the provenance file

| Audit (key in `meta`) | Date | What was checked | Outcome |
|---|---|---|---|
| `citation_audit` | 2026-08-04 | Every reference in the Letter and Supplemental Material; each DOI resolved (Crossref, OpenAlex, publisher page or author-hosted PDF) and compared field by field: title, author order, journal, volume, issue, page or article number, year | All matched. Two defined but uncited entries deleted (Dill *et al.*, Appl. Phys. Lett. 127, 032105 (2025); Ponce, Jena and Giustino, Phys. Rev. B 100, 085204 (2019)). Fang and Howard (1966) and de Carvalho *et al.* (2010) added to the Letter's reference list, as APS requires. Claims attributed to Chang, Wang, Dill, Costa, Rizzi and King *et al.* checked against the text. |
| `revision_audit` | 2026-09-24 | The revised Letter and Supplemental Material, including the two references added (Bernardini 1997, Ambacher 1999) | All nineteen journal articles matched. Correction: the measured light-hole ratio had been formed as 0.573/0.150 = 3.82 from lifetimes; it is now the mobility ratio 1900/368 = 5.16 (heavy 2.00 to 2.40). |
| `second_revision_audit` | 2026-09-25 | Every citation and attributed claim re-read against the primary text where accessible; all nineteen DOIs through the Crossref REST API | All matched. Corrections: King *et al.* offsets 0.5 +/- 0.2 and 0.8 +/- 0.2 eV (0.86 had been a text-extraction artifact); Rizzi *et al.* value 0.3 +/- 0.1 eV; the three experiments were not made on one sample (text, Fig. 1 caption and Sec. S8 corrected; new calculation for the cyclotron-resonance sample); the reported Hall-fit ranges added to the ratio ranges; wording on spin degeneracy, strain, Costa *et al.*, Huang and Wu, Bader *et al.*, and an unverifiable superlative corrected. |
| `light_hole_revision` | 2026-09-25 | Statements added with the light-hole and two-mechanism analysis | Wang *et al.* and Chang *et al.* procedures quoted; Rinke *et al.* GaN A6 = -3.202; Punya and Lambrecht (2012) and Asgari *et al.* (2005) read and checked against Crossref; Vurgaftman and Meyer (2003) not accessible, so its A6 is not quoted. |

The file also records, with quotations, three open questions stated in print by
the original authors (the field-dependent light-hole mass, the disagreement of
the two heavy-hole masses, and the unidentified elastic scattering mechanism)
and a caveat on the temperature exponent taken from the arXiv version of Dill
*et al.*.

---

## 9. Built-in checks

**`tests/test_kp6.py`** (61 tests, run with `python -m pytest tests/ -q`).
The suite checks quantities known independently of the implementation rather
than merely exercising it: Hermiticity, time-reversal symmetry, closed-form
zone-center eigenvalues, accepted splittings, basal-plane isotropy, the
reported strain, the occupation rule `n = k_F^2 / 4 pi` for a single
spin-resolved branch, the reduction of the heterostructure operator to the
hard-wall one, the agreement of the sparse and dense eigen-solvers, the
normalization of the computed Bloch overlap and the continuity of the tracked
spin splitting; for the Landau levels, the parabolic limit, equality of the
spectra at `B` and `-B`, the Onsager count of states (the number of states each Landau level holds) and the grid truncation;
for transport, the exact limits `tau_tr/tau_q = 1` and `1/2`, invariance under
disorder amplitude and mass, the two-carrier form of the coupled
magnetoconductivity against an explicit least-squares fit, the reduction to
independent Drude channels, the `s`-wave identity of the cross sections, the
Born limit of the variable-phase solver and the exact solution of the
linearized Boltzmann equation; for the oscillation amplitudes, the resistance
of a single carrier, the density-of-states weights without intersubband
scattering, the inversion of the amplitude ratio, the reduction of the
polarization-fluctuation spectrum to point charges, the reported Dingle
mobility and heavy-hole mass from the digitization, the factor 2 of the full
response for one carrier in a strong field, the closed-form Lorentzian density
of states, the spin factor from two harmonics, the exponent 3 of the
misfit-line kernel and the reduction of the non-perturbative `rho_xx` to first
order; the piezoelectric polarization formula; the
parameter override, the bound on A6 and the parabolic density condition; and,
for the many-body mass, the limits of the Lindhard function and the published
on-shell mass of the two-dimensional electron gas.

**Checks inside the scripts.**

- `run_heavy_quantum_mobility.py` first validates the digitization against
  what Chang *et al.* extract from the same figure (light-hole Dingle mobility,
  light- and heavy-hole LK masses, the normalized amplitudes of their Fig. 2f).
- `run_many_body.py` benchmarks the RPA mass against Table I of Asgari *et al.*
- `run_phaseshift.py` checks that the exact phase-shift result approaches the
  Born value when the potential is scaled towards zero.
- `run_landau_lorentz.py` applies the same Dingle analysis to a single
  sinusoid with the Lifshitz-Kosevich factors as a control.
- `run_barrier.py` records the sensitivity of the finite-barrier result to each
  numerical and physical choice.

---

## 10. Notes on the calculations

- **Conventions** (`kp6_well.py`). Energies are hole energies in eV, increasing
  downward from the valence band maximum, so the ground subband is the lowest
  eigenvalue. The coordinate z runs in nanometres from the GaN/AlN interface.
  In-plane wavevectors are in 1/nm. `rpa2d.py` uses meV and nm.
- **Electrostatic closure.** The Letter closes the electrostatics at the
  measured sheet density: the interface charge that the gas balances is set
  equal to the hole density, so the field vanishes beyond the gas
  (`results/well_het.json`). The alternative "polarization closure" sets the
  interface charge from the calculated spontaneous plus piezoelectric
  polarization and places the excess far from the interface
  (`results/well_het_pol.json`); both are reported.
- **Finite barrier.** The published valence band offset spans a wide range, so
  it is scanned rather than chosen. The 0.7 eV well (`results/well_het.json`)
  is the default potential of the Landau-level, dispersion and overlap
  calculations; the A6 analysis adds two polarization-closure wells at 0.3 eV.
- **The linear-in-k term A7** is taken as zero, and the shear deformation
  terms D5 and D6 vanish for the biaxial strain treated here (`kp6.py`).
- **Landau levels.** Computed on a grid uniform in 1/B from 25 to 125 T, with
  the zero-field potential held fixed; the grid is truncated at -2 nm and +8 nm,
  which moves the relevant levels by less than 0.03 meV (tested). The
  valence-band magnetic parameter has no measured value for GaN holes, so it is
  scanned (0, -2 and +2 by default) and never fitted.
- **Figures follow the numbers.** The computed curves read from the JSON
  written by the analysis scripts; some measured points and labels are written
  into the figure scripts (listed in Section 6).

**Known inconsistencies in the repository** (found while writing this guide;
not changed, because this update touches documentation only):

- `requirements.txt` allows `numpy>=1.24`, but the code needs NumPy 2.0 or
  newer (`numpy.trapezoid`; Section 3).
- The docstring of `scripts/figure_overview.py` says panel (a) draws "the
  heterostructure that all three experiments were performed on", and the
  docstring of `src/gan2dhg/scatter2d.py` says the three experiments were made
  "on the same" gas. The second citation audit in the provenance file retracts
  this: the cyclotron-resonance sample was a different one.
- `src/gan2dhg/constants.py` says "CODATA 2018" but reads whatever
  `scipy.constants` provides (CODATA 2022 in SciPy 1.17.1), while `kp6.py`,
  `figures3.py` and the tests write in the CODATA 2018 electron mass
  (Section 7).
- The source table for the AlN crystal-field splitting: Table III of Rinke
  *et al.* in `kp6_het.py`, Table V in the provenance file (Section 8).
- A comment in `kp6_het.py` says the midpoint of 21.7 and 23.5 meV is used
  for the AlN spin-orbit splitting; the code uses 22 meV, which
  `run_structure_check.py` and the provenance file describe as the rounded
  21.7 meV.
- `figure_overview.py` labels the in-plane strain -2.42 percent;
  `kp6.biaxial_strain_on_AlN` gives -2.41 percent.
- The original README asked for the scripts to be run in its listed order, but
  `run_dispersion.py` came before `run_well_polarisation.py`, whose output it
  reads (corrected in Section 5).

**Methodological cautions** recorded by the authors:

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
5. In a two-carrier gas the two oscillations do not appear in `rho_xx` with
   equal weight, so the field at which an oscillation first becomes visible is
   not a measure of its Dingle factor alone.

---

## 11. Version history

No release or git tag has been made. The package carries the version string
`2.0.0` (`src/gan2dhg/__init__.py`), set when the package was renamed on
5 August 2026. The development stages below are read from the git history;
details are in [CHANGELOG.md](CHANGELOG.md).

| Date | Stage | Tests |
|---|---|---|
| 30 Sep 2026 | Documentation: this README restructured, CHANGELOG and CITATION added; code, data and results unchanged | 61 |
| 26 Sep 2026 | Heavy-hole quantum mobility from the digitized oscillations (`sdh.py`, digitized Fig. 2, forward calibration, revised mobilities) | 61 |
| 25 Sep 2026 | Revision: Landau levels, coupled magnetotransport, strain and polarization, second citation audit, light-hole analysis (A6, many-body mass); obsolete scripts removed | 51 |
| 5 Aug 2026 | Package renamed `gan2dhg`; finite AlN barrier; first citation audit (dated 2026-08-04) recorded; current manuscript title | 36 |
| 4 Aug 2026 | First version (package `gpfet`); self-consistent well solver; license changed from MIT to Apache 2.0 | 28 |

---

## 12. How to cite

No journal reference, DOI or author list for the manuscript is recorded in
this repository. Until one exists, please cite the repository. The names
below are the copyright holders named in `LICENSE`, not a recorded author
list of the manuscript. GitHub shows a
**"Cite this repository"** button in the right-hand column, which reads
`CITATION.cff`.

> T. M. Mahim, A.S.M. Mohsin, and M. M. Rahman, gan-2dhg-masses-lifetimes:
> code and derived data for "Origin of the conflicting hole masses in the
> GaN/AlN two-dimensional hole gas" (2026),
> https://github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes

When using the published values, please also cite the original measurements
(Chang *et al.*, Nat. Electron. 9, 346 (2026); Wang *et al.*, Appl. Phys. Lett.
126, 213102 (2025)); their full references are in
`data/gan_2dhg_measured.yaml`.

---

## 13. License and contact

Code: Apache License 2.0 (see [`LICENSE`](LICENSE)).

Questions and bug reports: please open an issue on this repository, or
contact Tanvir M. Mahim, BRAC University (tanvir.mahim@bracu.ac.bd).
