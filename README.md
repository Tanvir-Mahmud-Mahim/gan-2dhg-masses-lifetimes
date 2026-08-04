# Hole masses and lifetimes of the GaN/AlN two-dimensional hole gas

Code and derived data for the manuscript *Reconciling the measured hole masses
and lifetimes of the GaN/AlN two-dimensional hole gas*.

No experiment was performed for this work. Every experimental number used is a
published value, and every one of them is recorded in
[`data/gan_2dhg_measured.yaml`](data/gan_2dhg_measured.yaml) together with its
source and with a note of whether the full text or only the abstract was
available when it was read. Nothing was taken from memory or from a secondary
summary.

## What the code does

Three magnetotransport experiments have been applied to the same
polarisation-induced GaN/AlN two-dimensional hole gas, and the band-resolved
parameters they return do not agree. This repository contains the analysis that
resolves two of those disagreements and sharpens the third.

| module | purpose |
| --- | --- |
| `src/gpfet/kp6.py` | Six-band wurtzite valence band Hamiltonian, with strain, confinement and branch tracking by eigenvector continuity |
| `src/gpfet/scatter2d.py` | Two-dimensional elastic scattering: transport and quantum lifetimes for remote charge, interface roughness, background impurities and dislocations, screened, with interband coupling |

Parameters for the Hamiltonian are taken from Extended Data Table 1 of Chang
*et al.*, Nat. Electron. **9**, 346 (2026), so that the calculation uses the
same inputs as the measurement it is compared against.

## Reproducing every number in the paper

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # 20 physics tests
python scripts/run_masses.py               # -> results/masses.json
python scripts/run_scattering.py           # -> results/scattering.json
python scripts/run_scattering_fit.py       # -> results/scattering_fit.json
python scripts/run_interband.py            # -> results/interband.json
python scripts/run_ratio_robust.py         # -> results/ratio_robust.json
python scripts/figures2.py                 # -> figures/fig1, fig2 at 1000 dpi
```

Figures read from the JSON written by the analysis scripts, so no figure can
drift from a number quoted in the text.

## Tests

The suite checks quantities known independently of the implementation rather
than merely exercising it: Hermiticity at random wavevectors, Kramers
degeneracy, the closed-form zone-centre eigenvalues, the accepted unstrained
splittings, basal-plane isotropy, the reported pseudomorphic strain, and the
spin-orbit crossover above which the heavy and light labels are meaningful. On
the scattering side it pins the exact limits `tau_tr/tau_q = 1` for isotropic
scattering and `1/2` for pure backscattering, the invariance of that ratio
under both disorder amplitude and effective mass, and the reduction of the
coupled two-subband solver to the single-subband result when the interband
overlap vanishes.

## Two errors found during this work

Recorded because they affected intermediate results before they were caught.

1. The six-band Hamiltonian was first transcribed from memory and was not
   Hermitian. `numpy.linalg.eigvalsh` reads only one triangle, so it
   symmetrised the matrix silently and returned plausible but wrong splittings.
   It was caught by comparing against the known zone-centre values, and the
   Hamiltonian now asserts Hermiticity on every call.
2. Band branches were first followed by a greedy overlap match, which exchanged
   the light and crystal-field split-off branches where they cross and produced
   a spurious kink in the dispersion. Branches are now assigned optimally.

## Earlier, separate project

An earlier project on the sign of the temperature coefficient in GaN p-channel
transistors lives in its own repository,
[`gan-pfet-thermal-cooperativity`](https://github.com/Tanvir-Mahmud-Mahim/gan-pfet-thermal-cooperativity).
Nothing here depends on it. Two cautions if it is ever picked up again:

- Its `dft/u_relax.py` does not do what it claims. It imports `U_INT` by value
  from `gan_scf`, so reassigning `gan_scf.U_INT` never reaches the cell builder
  and all of its runs are identical. Its output is meaningless.
- Its strain results reproduce a deformation potential that is already
  published, and do so about thirty per cent below the accepted value. They are
  not a new result.

## Licence

See [LICENSE](LICENSE).
