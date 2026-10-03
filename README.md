# Hole masses and lifetimes in the GaN/AlN two-dimensional hole gas

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This repository holds the code and derived data for the manuscript
**"Origin of the conflicting hole masses in the GaN/AlN two-dimensional hole
gas"**. The figure script calls it a Letter for *Physical Review B*, with
Supplemental Material. You will not find a journal reference or DOI in this
repository.

The code copyright belongs to Tanvir M. Mahim, A.S.M. Mohsin, and M. Mosaddequr
Rahman (the copyright holders named in [`LICENSE`](LICENSE)). The author list
of the manuscript is not recorded here.

- Repository: https://github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes
- Record of every published number used, with its source and the citation audit:
  [`data/gan_2dhg_measured.yaml`](data/gan_2dhg_measured.yaml)

No experiment was done for this work. Every experimental number here is a
published value. Where each value comes from, and how the citations were
checked, is described in [DETAILS.md](DETAILS.md#more-on-the-introduction-the-record-of-published-values).

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step)
6. [Which script makes which figure](DETAILS.md#6-which-script-makes-which-figure)
7. [The Python modules](DETAILS.md#7-the-python-modules)
8. [Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from)
9. [Built-in checks](DETAILS.md#9-built-in-checks)
10. [Notes on the calculations](DETAILS.md#10-notes-on-the-calculations)
11. [Version history](DETAILS.md#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

Sections 5 to 11 are in [DETAILS.md](DETAILS.md), together with
[extra notes for Sections 1 to 4](DETAILS.md#extra-notes-for-sections-1-to-4).
A short guide to them is in [More details](#more-details-sections-5-to-11).

---

## 1. The idea in one minute

Take a thin layer of gallium nitride (GaN) grown on aluminum nitride (AlN).
The built-in electric polarization of the two crystals pulls a thin sheet of
mobile positive charge carriers against the interface. These carriers are
called **holes** (missing electrons). The sheet is a
**two-dimensional hole gas** (2DHG). It forms without acceptor doping.

The holes fall into two groups, **heavy holes** and **light holes**. They
behave as if they had different masses. Each group fills its own **subband**
(a set of allowed energies).

Three kinds of published measurement have probed this gas: quantum
oscillations, a two-carrier Hall analysis, and terahertz cyclotron resonance.
The numbers they give for each subband do not agree.

This repository holds the analysis of these disagreements. It computes the
band structure of the gas from a **six-band k·p model** (Chuang and Chang,
1996). The code solves it together with the electrostatics of the gas. It also
computes the energy levels in a magnetic field (**Landau levels**), models how
several kinds of disorder scatter the holes, and analyzes the published
oscillation data again.

**Main results.** In the words of the original description:

- the heavy-hole mass and the difference between the two heavy-hole masses
  are accounted for;
- the light-hole mass and occupation are shown to be one zero-field
  discrepancy, traced to the band parameter A6;
- the heavy-hole quantum mobility is re-derived from a Dingle analysis of the
  published oscillations and from the ratio of their amplitudes;
- and the four measured mobilities are shown to fix the spectrum of the
  long-range disorder.

The full background, the principal results with all their numbers, and a short
glossary of terms are in
[DETAILS.md](DETAILS.md#more-on-section-1-the-full-background).

---

## 2. What is in this repository

- **data/**: the record of every published value used, its source, and the
  citation audits (`data/gan_2dhg_measured.yaml`). It also holds Fig. 2a, 2c
  and 2f of Chang *et al.*, digitized (`data/chang2026_fig2_digitized.json`).
- **src/gan2dhg/**: the Python package `gan2dhg` (see
  [Section 7](DETAILS.md#7-the-python-modules)).
- **scripts/**: the analysis and figure scripts (see
  [Section 5](DETAILS.md#5-the-scripts-step-by-step)).
- **results/**: the JSON files written by the scripts.
- **tests/**: `tests/test_kp6.py`, with 61 physics tests (pytest).
- At the top level: `requirements.txt`, `LICENSE`, `CITATION.cff` and
  [CHANGELOG.md](CHANGELOG.md).

`results/` is part of the repository. You can read the computed numbers from
it and redraw the figures from it without recomputing anything. The figure
scripts write to a folder `figures/`, which they create. It is not stored in
the repository.

The full annotated file tree is in
[DETAILS.md](DETAILS.md#more-on-section-2-the-full-file-tree).

---

## 3. Installation

I checked the code with **Python 3.11** (3.11.15). The repository does not
state a minimum Python version.

```
pip install -r requirements.txt
```

This installs `numpy>=2.0`, `scipy>=1.13`, `matplotlib>=3.8.4`, `pyyaml>=6.0`
and `pytest>=7.0`. The code calls `numpy.trapezoid`, which first appeared in
NumPy 2.0. So NumPy 2.0 is the minimum.

There is no installable package, so the scripts that use it and the test file
add `src/` to the Python path themselves. All file paths are relative to the
script, so the commands work from any working directory.

The versions I installed, the tests with the minimum versions, and a note on
fonts are in
[DETAILS.md](DETAILS.md#more-on-section-3-versions-checks-and-fonts).

---

## 4. Quick start: three ways to use the code

Run all commands from the repository folder.

### Way A: check that everything works

```
python -m pytest tests/ -q
```

All 61 tests must pass.

### Way B: redraw the figures from the committed results (about 20 seconds)

```
python scripts/figure_overview.py      # -> figures/prb_fig0.png and .pdf
python scripts/figures3.py             # -> figures/prb_fig1 and prb_fig2, .png and .pdf
```

The figures are written at 1000 dpi.

### Way C: recompute the results

Run the scripts of [Section 5](DETAILS.md#5-the-scripts-step-by-step) in the
order given there, then Way B. Be aware that the scripts **overwrite the
committed files in `results/`**. You can see what changed with
`git diff --stat results/`.

My test and run times, and the scripts that take long, are in
[DETAILS.md](DETAILS.md#more-on-section-4-run-times-and-notes).

---

## More details (Sections 5 to 11)

The full notes are in [DETAILS.md](DETAILS.md). Here is what each section
holds:

- [5. The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step): every
  command in order, what it reads and writes, and its run time where measured.
- [6. Which script makes which figure](DETAILS.md#6-which-script-makes-which-figure):
  the three Letter figures, their panels, and their data files.
- [7. The Python modules](DETAILS.md#7-the-python-modules): what each module of
  the package `gan2dhg` does.
- [8. Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from):
  the band parameters, the measured values, and the citation audits.
- [9. Built-in checks](DETAILS.md#9-built-in-checks): what the 61 tests and
  the checks inside the scripts cover.
- [10. Notes on the calculations](DETAILS.md#10-notes-on-the-calculations):
  conventions, known inconsistencies, and methodological cautions.
- [11. Version history](DETAILS.md#11-version-history): the development
  stages, by date.

---

## 12. How to cite

No journal reference, DOI or author list for the manuscript is recorded in
this repository. Until one exists, please cite the repository. The names
below are the copyright holders named in `LICENSE`, not a recorded author
list of the manuscript. On GitHub you will also see a
**"Cite this repository"** button in the right-hand column, which reads
`CITATION.cff`.

> T. M. Mahim, A.S.M. Mohsin, and M. M. Rahman, gan-2dhg-masses-lifetimes:
> code and derived data for "Origin of the conflicting hole masses in the
> GaN/AlN two-dimensional hole gas" (2026),
> https://github.com/Tanvir-Mahmud-Mahim/gan-2dhg-masses-lifetimes

If you use the published values, please also cite the original measurements
(Chang *et al.*, Nat. Electron. 9, 346 (2026); Wang *et al.*, Appl. Phys. Lett.
126, 213102 (2025)). You can find their full references in
`data/gan_2dhg_measured.yaml`.

---

## 13. License and contact

Code: Apache License 2.0 (see [`LICENSE`](LICENSE)).

If you have a question or find a bug, please open an issue on this
repository, or contact me, Tanvir M. Mahim, BRAC University
(tanvir.mahim@bracu.ac.bd).
