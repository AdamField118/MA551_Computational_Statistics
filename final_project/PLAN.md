# MA 551 Final Project: Uncertainty Quantification for a Regularized Lensing Inverse Problem

## One-Line Pitch
A Monte Carlo study of how noise and shear bias propagate through a Wiener-filter
mass reconstruction pipeline, covering importance sampling, antithetic variables,
EM component recovery, bootstrap UQ, and power analysis — all tied to a real
astrophysical inverse problem.

---

## The Statistical Problem (Professor-Facing Framing)

We observe noisy measurements of a vector field derived from an unknown image
kappa(theta) via a known linear forward operator F:

    gamma_obs = F * kappa_true + epsilon,    epsilon ~ N(0, sigma_n^2 I)

The regularized MAP estimator (Wiener filter / Tikhonov) is:

    kappa_hat = argmin_kappa || F*kappa - gamma_obs ||^2 + lambda ||kappa||^2

This is a standard Bayesian linear inverse problem with a Gaussian prior.
The astrophysics provides F (the lensing kernel, computed via FFT), kappa
(the projected mass density of a galaxy cluster), and gamma (the gravitational
shear field). None of this physics needs to appear in the statistical framing.

Statistical questions we answer:
1. How does noise propagate through F^{-1}? (Monte Carlo)
2. How does multiplicative bias in gamma_obs affect kappa_hat? (Importance sampling)
3. Can we fit a Gaussian mixture to kappa_hat to recover cluster substructure? (EM)
4. What are pixelwise confidence bands on kappa_hat? (Bootstrap BCa)
5. How many galaxies do we need to detect a 1% bias at 5-sigma? (Power analysis)

---

## Course Coverage Map

| HW / Topic            | Project Component                                      |
|-----------------------|--------------------------------------------------------|
| HW1: Bootstrap, PCA   | BCa confidence bands on kappa_hat; PCA of residuals    |
| HW2: Newton-Raphson   | Iterative KS reconstruction (optional Tikhonov solve)  |
| HW3: Rejection / IS   | IS for bias-integrated reconstruction error            |
| HW3: Antithetic vars  | Variance reduction in MC error estimation              |
| HW4: EM mixture model | EM to recover cluster + substructure components        |
| HW4: Censored EM      | (Analogy: incomplete data / censored kappa pixels)     |

This is strictly more than any single homework: methods interact (IS feeds
into bootstrap; EM estimates are the objects we put CIs on), the data-generating
process is nontrivial, and results require domain interpretation.

---

## Implementation Plan

### Phase 1 — Core Pipeline (implement first)
- `R/lensing.R`     Forward operator F (kappa -> gamma) and KS inverse, both via FFT
- `R/simulate.R`    Galaxy catalog: positions, shape noise, binning to grid
- `R/reconstruct.R` Wiener filter reconstruction with lambda selection

### Phase 2 — Monte Carlo Framework
- `R/mc.R`          Run B replicates, store kappa_hat arrays, compute L2 error
- Antithetic pairs: run (gamma, -gamma) replicates for variance reduction
- Importance sampling: weight replicates by p(m) for bias-integrated error

### Phase 3 — Statistical Analysis
- `R/bootstrap.R`   Pixelwise BCa CIs on kappa_hat; coverage simulation
- `R/em_kappa.R`    EM for Gaussian mixture on kappa_hat map pixels
- `R/power.R`       Power curve for detecting |m| = 0.01 at 5-sigma

### Phase 4 — Report
- `final_report.Rmd`  R Markdown, knitted to PDF, standard HW conventions

---

## Key Parameters (fixed for reproducibility)

```r
N_PIX    <- 64        # grid is 64 x 64 pixels
PIX_DEG  <- 0.05      # pixel scale in arcmin (0.05 arcmin/pix => 3.2 arcmin field)
SIGMA_L  <- 0.5       # true kappa Gaussian sigma in arcmin
KAPPA_0  <- 0.3       # peak convergence
SIGMA_E  <- 0.26      # intrinsic shape noise per component
N_GAL    <- 500       # galaxies per Monte Carlo replicate (varied in power analysis)
B_MC     <- 500       # Monte Carlo replicates
B_BOOT   <- 1000      # bootstrap replicates
LAMBDA   <- 1e-3      # Tikhonov regularization (or chosen by discrepancy principle)
```

---

## Directory Structure

```
final_project/
├── PLAN.md
├── final_report.Rmd
├── R/
│   ├── lensing.R       # forward/inverse operators (FFT-based)
│   ├── simulate.R      # galaxy catalog simulation
│   ├── reconstruct.R   # Wiener filter
│   ├── mc.R            # Monte Carlo framework
│   ├── bootstrap.R     # BCa confidence bands
│   ├── em_kappa.R      # EM for mixture model on kappa_hat
│   └── power.R         # power analysis
└── scripts/
    └── run_all.R       # sources everything, produces all results
```