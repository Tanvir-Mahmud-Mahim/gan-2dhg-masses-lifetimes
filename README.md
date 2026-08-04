# Hole masses and scattering in the GaN/AlN two-dimensional hole gas

Code and derived data for the manuscript *Hole masses and scattering in the
GaN/AlN two-dimensional hole gas: what three magnetotransport probes do and do
not agree on*.

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
| `src/gpfet/kp6.py` | Six-band wurtzite valence band Hamiltonian, with strain and branch tracking by eigenvector continuity |
| `src/gpfet/kp6_well.py` | Self-consistent envelope-function solution of the polarisation well: `k_z -> -i d/dz` coupled to Poisson at fixed sheet density |
| `src/gpfet/scatter2d.py` | Two-dimensional elastic scattering: transport and quantum lifetimes for remote charge, interface roughness, background impurities and dislocations, screened, with interband coupling |

Parameters for the Hamiltonian are taken from Extended Data Table 1 of Chang
*et al.*, Nat. Electron. **9**, 346 (2026), so that the calculation uses the
same inputs as the measurement it is compared against. Nothing is adjusted.

## Reproducing every number in the paper

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # 28 physics tests, about 80 s

python scripts/run_well.py                 # -> results/well.json        (~3 min)
python scripts/run_well_sweep.py           # -> results/well_sweep.json  (~16 min)
python scripts/run_scattering.py           # -> results/scattering.json
python scripts/run_scattering_fit.py       # -> results/scattering_fit.json
python scripts/run_interband.py            # -> results/interband.json
python scripts/run_ratio_robust.py         # -> results/ratio_robust.json
python scripts/figures3.py                 # -> figures/prb_fig1, prb_fig2 at 1000 dpi
```

Figures read from the JSON written by the analysis scripts, so no figure can
drift from a number quoted in the text.

## Principal results

- The self-consistent calculation reproduces the heavy-hole mass, 1.875 m0
  against a measured 1.92 +/- 0.16 m0, and the zero-field light-hole mass,
  0.246 m0 against the 0.30 m0 obtained by extrapolating the measurement to
  zero field. The dispersion is not the source of the reported anomalies.
- The heavy-hole masses from cyclotron resonance and quantum oscillations
  differ because that resonance is overdamped: `omega_c tau = 0.82` at the
  highest field applied.
- The ratio `tau_tr/tau_q` carries neither the disorder amplitude nor the
  effective mass, and its measured value requires long-range forward-peaked
  scattering, not the short-range mechanism to which it has been attributed.
  No elastic mechanism reproduces the measured ratio for both subbands.
- The asymmetric well gives a Rashba splitting of 1.5 meV in the light subband
  at its Fermi wavevector, which both experiments' analyses assume away.

## Tests

The suite checks quantities known independently of the implementation rather
than merely exercising it: Hermiticity at random wavevectors, time-reversal
symmetry of the confined spectrum, the closed-form zone-centre eigenvalues, the
accepted unstrained splittings, basal-plane isotropy, the reported pseudomorphic
strain, and the spin-orbit crossover above which the heavy and light labels are
meaningful. The well solver is required to reproduce the bulk masses when the
confining field is switched off, to conserve the total sheet density, to give
the analytic interface field from Poisson's equation, and to return exactly
zero spin splitting when an inversion centre is restored. On the scattering
side it pins the exact limits `tau_tr/tau_q = 1` for isotropic scattering and
`1/2` for pure backscattering, the invariance of that ratio under both disorder
amplitude and effective mass, and the reduction of the coupled two-subband
solver to the single-subband result when the interband overlap vanishes.

## Three errors found during this work

Recorded because each affected intermediate results before it was caught, and
each produced plausible numbers while wrong.

1. The six-band Hamiltonian was first transcribed from memory and was not
   Hermitian. `numpy.linalg.eigvalsh` reads only one triangle, so it
   symmetrised the matrix silently and returned plausible but wrong splittings.
   Caught by comparing against the known zone-centre values; the Hamiltonian
   now asserts Hermiticity on every call.
2. Band branches were first followed by a greedy overlap match, which exchanged
   the light and crystal-field split-off branches where they cross and produced
   a spurious kink in the dispersion. Branches are now assigned optimally.
3. Confinement was first represented by a hard-wall proxy, `k_z = pi/w`. That is
   structurally wrong, not merely imprecise: it inserts `k_z` as a number into a
   coupling linear in `k_z`, whose expectation vanishes for a bound state. The
   proxy returned a light-hole mass near 0.51 m0, close to the field-averaged
   measurement and so apparently correct. The self-consistent solve returns
   0.246 m0. The proxy survives only in a test, labelled unreliable, so that the
   discrepancy stays visible.

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
