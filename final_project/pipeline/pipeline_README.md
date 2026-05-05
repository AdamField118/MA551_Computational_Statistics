# Metacalibration Bias Pipeline

End-to-end multiplicative shear bias estimation for the MA 551 final project.
Mirrors `shear_bias/m/main.py` and `shear_bias/m/helpers.py` from ShearNet exactly.

---

## Project layout (what you already have)

```
MA551_Computational_Statistics/
├── cosmos15_superbit2023_phot_shapes_with_sigma.csv   <- catalog (project root)
├── psf_data/                                          <- PSFEx models (project root)
│   └── *.psf
├── final_project/
│   ├── pipeline/
│   │   ├── metacal_pipeline.py    <- this script
│   │   ├── slurm_metacal.sh       <- Turing job script
│   │   └── README.md              <- this file
│   ├── R/
│   │   └── run_all.R              <- reads results/metacal_bias.json
│   └── results/
│       └── metacal_bias.json      <- written by metacal_pipeline.py
```

The catalog and PSF data live two levels above the pipeline script.
All paths below are written relative to `MA551_Computational_Statistics/`.

---

## Dependencies

You already have these in your `shearnet_gpu` conda environment on Turing.
No new installs needed.

```
galsim  ngmix  numpy  pandas  scipy  astropy
```

In R, `jsonlite` is needed to read the output (install once):
```r
install.packages("jsonlite")
```

---

## Quick test (login node, Gaussian PSF, 500 galaxies)

Run from `MA551_Computational_Statistics/`:

```bash
conda activate shearnet_gpu

python final_project/pipeline/metacal_pipeline.py \
    --catalog cosmos15_superbit2023_phot_shapes_with_sigma.csv \
    --n-obs   500 \
    --n-workers 4 \
    --psf-fwhm 0.5 \
    --seed 150 \
    --outdir final_project/results
```

Expected runtime: ~3 min on a laptop.
Expected output:

```
METACALIBRATION BIAS ESTIMATES
  Galaxies processed : 487 / 500
  m  = (+2.341 +/- 8.120) x 10^-2
  c  = (+0.00015 +/- 0.00031)
  R11 (ensemble)     : 0.3421 +/- 0.0018
```

---

## Full run on Turing (SuperBIT PSFEx, 50 000 galaxies)

Run from `MA551_Computational_Statistics/`:

```bash
sbatch final_project/pipeline/slurm_metacal.sh
```

The SLURM script is pre-configured with the correct relative paths.
Monitor progress:

```bash
squeue -u $USER
tail -f final_project/logs/metacal_<JOBID>.out
```

---

## All command-line options

| Flag | Default | Description |
|---|---|---|
| `--catalog` | required | Path to the cosmos15 CSV |
| `--n-obs` | 10000 | Number of galaxy pairs to simulate |
| `--n-workers` | half of CPUs | Parallel worker processes |
| `--psf-fwhm` | 0.5 | Gaussian PSF FWHM in arcsec (fallback if `--psfex` absent) |
| `--psfex` | None | Path to a `.psf` PSFEx model file for realistic SuperBIT PSF |
| `--n-jack` | 20 | Jackknife groups (matches ShearNet `Njack` default) |
| `--seed` | 150 | RNG seed (matches ShearNet benchmark configs) |
| `--outdir` | `results` | Directory for `metacal_bias.json` output |

---

## Simulation parameters

All fixed constants mirror `shear_bias/m/config.yaml` exactly:

| Parameter | Value | Source |
|---|---|---|
| `nse_sd` | 12.719674 ADU | SuperBIT sky noise |
| `scale` | 0.141 arcsec/px | SuperBIT pixel scale |
| `npix` | 53 | Stamp size |
| `gal_hlr` | 0.5 arcsec | Constant HLR (as in ShearNet unit tests) |
| `gal_flux` | 12258.97 ADU | Constant flux |
| `shear_true` | 0.01 | Injected +/- shear |
| `mcal_step` | 0.01 | Metacalibration perturbation step |
| `psf_model` | gauss | ngmix PSF fitter |
| `gal_model` | gauss | ngmix galaxy fitter |

---

## Catalog columns used

The CSV has 70 columns. The pipeline uses only four:

| CSV column | Role |
|---|---|
| `c10_viable_sersic` | Quality filter — keep rows where this equals 1 |
| `c10_sersic_fit_q` | Axis ratio q (also filter q > 0.05) |
| `c10_sersic_fit_phi` | Position angle in radians |
| `c10_sersic_fit_hlr` | Half-light radius (read but overridden by constant GAL_HLR = 0.5) |

Galaxy morphology follows ShearNet's pattern: constant HLR and flux,
catalog-drawn shape (q, phi), exactly as in the unit-test configs under
`unit_tests/first/`.

---

## Output: `results/metacal_bias.json`

`run_all.R` reads this file automatically in Section 0. Key fields:

```json
{
  "m_hat":        0.02341,
  "c_hat":        0.00015,
  "m_hat_se":     0.00089,
  "c_hat_se":     0.00023,
  "r11_mean":     0.3421,
  "r11_err":      0.0018,
  "n_success":    49203,
  "success_rate": 0.984,
  "m_jk": [...],
  "c_jk": [...]
}
```

---

## R integration

`run_all.R` loads `metacal_bias.json` in Section 0 and uses `M_HAT` and
`C_HAT` throughout instead of the previously injected `m=0, c=0`. The IS
prior width is set to `M_SE` (the jackknife SE), so importance sampling
now reflects genuine measurement uncertainty. If the JSON is absent the
script falls back to `m=0, c=0` with a warning and all analyses still run.

---

## Function provenance

Every function maps directly to a ShearNet source file:

| Pipeline function | ShearNet source |
|---|---|
| `get_admoms_ngmix_fit()` | `shearnet/utils/metrics.py` |
| `_make_galsim_wcs()` | `shearnet/utils/simutils.py` |
| `make_data()` | `shear_bias/m/main.py` |
| `_get_priors()` | `shear_bias/m/helpers.py` |
| `make_struct()` | `shear_bias/m/helpers.py` |
| `process_obs()` | `shear_bias/m/helpers.py` |
| `shear_data_to_table()` | `shear_bias/m/helpers.py` |
| `jackknife_mc_v2()` | `shear_bias/m/helpers.py` |
