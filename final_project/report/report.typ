// ============================================================
// MA 551 Final Project Report — Typst
// Uncertainty Quantification for a Regularized Lensing Inverse Problem
// Adam Field · Department of Physics · adfield@wpi.edu
// ============================================================

// --- Page setup -----------------------------------------------------
#set page(
  paper: "us-letter",
  margin: (top: 1in, bottom: 1in, left: 1in, right: 1in),
  numbering: "1",
  number-align: center,
)

#set text(font: "Cambria", size: 12pt, lang: "en")
#set par(justify: true, leading: 0.65em, spacing: 1.2em)
#set heading(numbering: "1.")

// --- Code style
#show raw: set text(size: 10pt, font: "Consolas")

// --- Math shortcuts
#let E = $bb(E)$
#let Var = $op("Var")$
#let Cov = $op("Cov")$
#let Cor = $op("Cor")$

// ============================================================
// TITLE PAGE
// ============================================================
#page(numbering: none)[
  #v(1.5cm)
  #align(center)[
    #text(size: 16pt, weight: "bold")[
      Monte Carlo and Resampling Methods Applied to\ 
      Weak Gravitational Lensing Mass Reconstruction
    ]
    #v(0.8cm)
    #text(size: 12pt)[
      Adam Field #footnote[Department of Physics. Email: #link("mailto:adfield@wpi.edu")[adfield\@wpi.edu]]
    ]
    #v(0.4cm)
    #text(size: 11pt)[May 2026]
  ]

  #v(1.2cm)
  #align(center)[*Abstract*]
  #block(inset: (left: 1.5cm, right: 1.5cm))[
    We present a Monte Carlo and resampling study of uncertainty quantification
    for a weak gravitational lensing mass reconstruction pipeline based on the
    Kaiser--Squires (KS) algorithm. The forward operator mapping convergence to
    shear is implemented in R and validated analytically against the SMPy
    production implementation. Simple Monte Carlo, antithetic variable sampling,
    and importance sampling are applied to characterize reconstruction error
    under shape noise. Parametric bootstrap confidence intervals, both
    percentile and bias-corrected-and-accelerated (BCa), are constructed for
    two scalar summaries of the reconstructed mass map. A coverage simulation
    confirms that 95% BCa intervals achieve empirical coverage of 97.5%.
    A power analysis determines that approximately 24,400 source galaxies are
    required to detect 1% multiplicative shear bias at 5-sigma significance
    with 500 Monte Carlo replicates per condition. The antithetic design is
    found to be degenerate for the $L^2$ reconstruction error, a consequence
    of the estimator's symmetry under sign reversal of the noise, illustrating
    a fundamental condition on variance reduction methods not always emphasized
    in textbook treatments.
  ]
]

// ============================================================
// SECTION 1 — INTRODUCTION
// ============================================================
= Introduction

Weak gravitational lensing is among the most powerful observational probes
of cosmological large-scale structure. Background galaxy images are
coherently distorted by intervening mass along the line of sight, producing
a measurable shear field from which the projected mass density, the
convergence $kappa$, can be reconstructed. The standard linear inversion,
known as Kaiser--Squires reconstruction #cite(<kaiser1993>), maps the
observed shear back to convergence via a Fourier-domain inversion of the
lensing kernel.

From a statistical standpoint the problem is a regularized linear inverse
problem: a known linear forward operator $F$ maps the unknown $kappa$ to
the observable shear $gamma$, and the reconstruction is obtained from noisy
shear measurements subject to the irreducible intrinsic ellipticity scatter
of source galaxies (shape noise). Several core statistical questions arise
naturally: how should uncertainty in the reconstructed mass map be quantified;
what confidence intervals are appropriate for scalar summaries; and how many
source galaxies are required to detect systematic errors in the shear
measurement pipeline?

This project applies the Monte Carlo and resampling methods of MA 551
to answer these questions within a fully controlled simulation environment.
Section 2 describes the lensing forward model, the reconstruction algorithm,
and the noise model. Section 3 presents the simulation study covering simple
Monte Carlo, antithetic variables, importance sampling, and bootstrap
confidence intervals. Section 4 develops the power analysis. Section 5
concludes with connections to the broader weak lensing literature.

// ============================================================
// SECTION 2 — METHODOLOGY
// ============================================================
= Methodology

== The Kaiser--Squires Forward--Inverse System

The convergence $kappa(bold(theta))$ and shear
$gamma = gamma_1 + i gamma_2$ are related through the lensing
potential $psi$, which satisfies $nabla^2 psi = 2kappa$.
In Fourier space, the shear is obtained from $kappa$ via the
Kaiser--Squires kernel $D(bold(k))$ #cite(<kaiser1993>):

$ hat(gamma)(bold(k)) = D(bold(k)) hat(kappa)(bold(k)), quad
  D(bold(k)) = frac(k_1^2 - k_2^2 + 2 i k_1 k_2, k_1^2 + k_2^2), $

with $D(bold(0)) = 0$ by convention (the DC mode is unobservable).
The inverse, which recovers $kappa$ from $gamma$, is:

$ hat(kappa)(bold(k)) = overline(D(bold(k))) hat(gamma)(bold(k)) $

since $|D(bold(k))| = 1$ for $bold(k) != bold(0)$.
Optional Tikhonov regularization replaces the denominator
with $|D|^2 + lambda$, shrinking high-frequency modes.

The implementation in R was validated analytically against the
SMPy#footnote[https://github.com/GeorgeVassilakis/SMPy]
`KaiserSquiresMapper`: both compute
$(k_1^2 - k_2^2) hat(G)_1 + 2k_1 k_2 hat(G)_2) \/ k^2$ for the
E-mode, which is algebraically identical to
$op("Re")[overline(D) hat(gamma)]$ (Notes \#11, \#12). A 10-test
analytical suite confirmed noiseless round-trip error below $10^(-10)$,
$|D(bold(k))| = 1$ to machine precision, and B-mode purity below $10^(-10)$
for a physical convergence field.

== Noise Model and Simulation Setup

We simulate source galaxy shape noise at the pixel level. With $N_"gal"$
galaxies uniformly distributed over an $N times N$ grid, the per-pixel shape
noise standard deviation is #cite(<bartelmann2001>):

$ sigma_"pix" = frac(sigma_e, sqrt(N_"gal" \/ N^2)), $

where $sigma_e = 0.26$ per component is the intrinsic ellipticity dispersion.
The observed shear at each pixel is:

$ gamma_"obs"[i,j] = (1 + m) gamma_"true"[i,j] + c + epsilon[i,j],
  quad epsilon[i,j] tilde cal(N)(0, sigma_"pix"^2), $

where $m$ is a multiplicative bias and $c$ an additive bias injected for
the power analysis. This is the standard Gaussian shape noise model used
in analytical weak lensing studies #cite(<kaiser1993>).

We use a $32 times 32$ pixel grid with pixel scale $0.1$ arcmin, giving a
$3.2$ arcmin field. The true convergence is a Gaussian cluster profile
$kappa_"true"(bold(theta)) = 0.3 exp(-|bold(theta)|^2 \/ 2sigma_ell^2)$
with $sigma_ell = 0.5$ arcmin, shown in @fig-kappa. With $N_"gal" = 5000$ galaxies,
$sigma_"pix" = 0.118$ and the shear signal reaches $|gamma| approx 0.10$,
giving a per-pixel SNR of approximately 0.85.

#figure(
  image("../figures/kappa_report.png", width: 52%),
  caption: [True convergence map $kappa_"true"(bold(theta))$ used throughout the simulation study. The Gaussian cluster profile peaks at $kappa_0 = 0.297$ with scale radius $sigma_ell = 0.5$ arcmin. The field is $3.2 times 3.2$ arcmin² on a $32 times 32$ pixel grid.],
  placement: auto,
) <fig-kappa>

*Reconstruction statistic.* Two scalar summaries of the KS reconstruction
are used throughout: the peak convergence $T_"peak" = max(hat(kappa))$
and the aperture mass
$T_"ap" = sum_(|bold(theta)| <= r_"ap") hat(kappa)(bold(theta)) , delta theta^2$
with $r_"ap" = 0.8$ arcmin. The aperture mass is preferred for inference
because it averages over a spatial region rather than taking the maximum of
a noisy field; this makes it far more stable under the shape noise model.

*DC ambiguity.* The KS inversion always zeros the DC Fourier mode, so
$E[hat(kappa)] = 0$ but $kappa_"true"$ has a non-zero mean. The aperture mass
of the noiseless KS reconstruction ($T_"ap"^"KS" = 0.2504$) differs from the
true aperture mass ($T_"ap"^"true" = 0.3461$) by 9.6%, reflecting the
unrecoverable mass-sheet degeneracy #cite(<bartelmann2001>).

// ============================================================
// SECTION 3 — SIMULATION STUDY
// ============================================================
= Simulation Study

== Simple Monte Carlo

We estimate $E[L_2(m=0)]$, the mean relative $L^2$ reconstruction error at
zero bias, using $B = 500$ independent noise realizations (Notes \#12).
The $L^2$ error is computed on the interior ($4$-pixel border removed) after
subtracting the mean from both reconstructed and true fields to remove the
DC offset. Results are summarized in @tab-mc.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*Summary*], [*True value*], [*MC mean*], [*MC SD*],
    table.hline(stroke: 0.4pt),
    [$L_2$ error],         [0 (m=0)], [1.521], [0.045],
    [Peak $hat(kappa)$],   [0.297],   [0.488], [0.051],
    [Aperture mass],       [0.250#super[\*]], [0.251], [0.015],
    table.hline(stroke: 0.8pt),
    table.cell(colspan: 4)[
      #set text(size: 10pt)
      \* Noiseless KS aperture mass; true-kappa value is 0.346 (DC offset 9.6%).
    ]
  ),
  caption: [Simple Monte Carlo results ($N_"gal" = 5000$, $B = 500$).],
  placement: auto,
) <tab-mc>

The aperture mass estimate is nearly unbiased relative to the noiseless KS
target. The peak kappa estimate shows substantial positive bias (0.488 vs.
0.297 true peak): the maximum of a 1024-pixel noisy field is a noise spike,
not the cluster center. The distributions of both statistics across all $B$
replicates are shown in @fig-mc.

#figure(
  image("../figures/mc_distributions.png", width: 100%),
  caption: [Monte Carlo distributions of peak $hat(kappa)$ (left) and aperture mass $T_"ap"$ (right) across $B = 500$ replicates. The dashed vertical line marks the target value in each case. Peak $hat(kappa)$ is severely upward-biased by noise spikes; aperture mass is nearly unbiased relative to the noiseless KS target.],
  placement: auto,
) <fig-mc>

== Antithetic Variable Sampling

Antithetic sampling pairs each noise draw $epsilon$ with its negation
$-epsilon$, exploiting negative correlation to reduce Monte Carlo variance
(Notes \#13). The variance reduction is:

$ op("Var")(hat(I)_"anti") = frac(sigma^2, 2N)(1 + op("Cor")(g(epsilon), g(-epsilon))). $

For a monotone integrand the correlation is negative, so variance is reduced.
Results for the three summaries at $B = 500$ pairs are:

- *Aperture mass:* +100% variance reduction (correlation = $-1$, exact)
- *Peak kappa:* $-4$% variance reduction (correlation slightly positive)
- *$L_2$ error:* $-100$% variance reduction (degenerate case)

The degenerate result for $L_2$ error is analytically exact, not a numerical
artifact. The reconstruction is linear: $hat(kappa)(epsilon) = F^{-1}(F kappa_"true" + epsilon)$,
so the mean-corrected $L_2$ error satisfies
$L_2(epsilon) = ||epsilon_"KS"|| \/ ||kappa_"true"||$ where
$epsilon_"KS" = F^{-1} epsilon$ is the noise propagated through the KS inverse.
Since this depends only on $||epsilon||$, we have $L_2(epsilon) = L_2(-epsilon)$ exactly,
so antithetic pairs yield identical values. Averaging them doubles the sample
without reducing variance, halving the effective sample size.

Aperture mass achieves perfect negative correlation because it is a
spatially integrated linear functional of $hat(kappa)$, and
$sum hat(kappa)(epsilon) = -sum hat(kappa)(-epsilon)$ (mean zero, linear in $epsilon$).

This illustrates the condition stated in Notes \#13: antithetic sampling
requires that $g$ be *monotone*. The $L_2$ error, being even in $epsilon$,
violates this condition. This finding would directly affect survey design
decisions that rely on variance reduction efficiency.

== Importance Sampling

We estimate the bias-integrated reconstruction error
$E_p[L_2(m)] = integral L_2(m) p(m) , d m$,
where $p(m) = cal(N)(0, 0.05^2)$ is a Gaussian prior on multiplicative bias
(Notes \#12, \#13). The proposal is $q(m) = op("Uniform")(-0.15, 0.15)$,
giving self-normalized importance weights $w_i = p(m_i)/q(m_i)$.

Using $n_"IS" = 50$ proposal draws (each evaluated with $B' = 100$ MC
replicates), the IS estimate is $hat(I)_"IS" = 1.518$ (SE = 0.193), compared
to the uniform grid estimate of 1.520. The small difference reflects that the
error function is nearly flat over the range of $m$ values supported by the
prior, so IS offers little advantage here. This is itself an informative
finding: the reconstruction error is insensitive to small biases at this
galaxy density, meaning the bias signal is buried in shape noise.

== Bootstrap Confidence Intervals

We construct parametric bootstrap confidence intervals for the aperture mass
using $B = 500$ bootstrap replicates at $N_"gal" = 5000$ (Notes \#6, \#7).
In the parametric bootstrap, fresh shape noise is drawn from the known
$cal(N)(0, sigma_"pix"^2)$ distribution for each replicate, consistent with
the known noise model.

The BCa interval adjusts for bias and skewness via the bias-correction factor
$hat(z)_0 = Phi^{-1}(#[$frac(1,B) sum_b bb(1)(hat(T)_b < hat(T))$])$ and
the acceleration $hat(a)$, estimated via column-jackknife on the shear grid
(deleting one column of $N = 32$ pixels at a time, requiring $N$ reconstructions
rather than $N^2$):

$ hat(a) = frac(sum_j (bar(T)_"jack" - hat(T)_((j)))^3,
  6{sum_j (bar(T)_"jack" - hat(T)_((j)))^2}^(3/2)). $

Results are shown in @tab-boot.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*Statistic*], [*Obs.*], [*Bias*], [*SE*], [*95% BCa CI*],
    table.hline(stroke: 0.4pt),
    [Peak $hat(kappa)$], [0.251], [+0.237], [0.051], [degenerate],
    [Aperture mass],     [0.250], [+0.000], [0.015], [$[0.220, 0.278]$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Bootstrap results ($B = 500$, $N_"gal" = 5000$).
            Peak kappa BCa is degenerate ($hat(z)_0 = -infinity$)
            because all bootstrap replicates exceed the noiseless observed value.],
  placement: auto,
) <tab-boot>

The peak kappa BCa is degenerate: every bootstrap replicate draws fresh noise,
so the bootstrap peak (driven by noise spikes) always exceeds the noiseless
reference value. This makes $hat(z)_0 = Phi^{-1}(0) = -infinity$ and the
interval undefined. This is not a coding error; it correctly identifies that
the peak of a noise-dominated field cannot be bootstrapped with a parametric
resampling scheme that adds noise to a noiseless reference. The aperture mass
is well-behaved: bias is negligible ($+0.0003$), and the BCa correction is
small ($hat(z)_0 = -0.045$, $hat(a) = 0.006$), indicating near-symmetry of
the bootstrap distribution.

The acceleration $hat(a) approx 0.006$ near zero reflects the near-linear
relationship between shear and aperture mass. For non-linear statistics or
when the estimator's standard error changes rapidly with the parameter,
$hat(a)$ is substantially nonzero and the BCa correction becomes important.

== Bootstrap Coverage Study

To assess whether the bootstrap intervals achieve their nominal coverage,
we run $n_"outer" = 200$ independent outer Monte Carlo replicates (Notes
\#6). For each outer replicate, one noisy shear observation is generated,
a 95% BCa and percentile CI are computed, and coverage against the noiseless
KS aperture mass is checked.

Results: both BCa and percentile intervals achieve 97.5% empirical coverage
at nominal 95%, with mean CI width 0.059. The slight over-coverage (97.5%
vs. 95%) is consistent with the conservative behavior of the parametric
bootstrap when $sigma_"pix"$ is known exactly and the shape noise distribution
is exactly Gaussian. The near-identical width of BCa and percentile CIs
(0.0585 vs. 0.0586) confirms that the acceleration correction is small, as
expected for this near-linear estimator.

// ============================================================
// SECTION 4 — POWER ANALYSIS
// ============================================================
= Power Analysis

We ask: how many source galaxies $N_"gal"$ are required to detect
multiplicative bias $|m| = 0.01$ at 5-sigma significance, using a
two-sample Welch $t$-test on $B$ independent aperture mass replicates per
condition?

*Why unpaired?* A paired design, sharing noise draws between $m = 0$ and
$m = 0.01$, is degenerate for aperture mass, which is a linear statistic.
The paired difference
$D_b = T_"ap"(m = 0.01) - T_"ap"(m = 0) = 0.01 times T_"ap"^"KS"(m = 0)$
is exactly constant across replicates, giving $op("sd")(D) = 0$ and
$"NCP" = infinity$. The unpaired design preserves the natural noise
contribution to the variance of each condition.

*Analytic power.* The effect size is
$delta = |m| times T_"ap"^"KS" = 0.01 times 0.2504 = 0.00250$.
The aperture mass standard deviation scales as $sigma_"ap" prop 1/sqrt(N_"gal")$
(shape noise model):

$ sigma_"ap"(N_"gal") = sigma_"ap,ref" sqrt(frac(N_"gal,ref", N_"gal")), $

with $sigma_"ap,ref" = 0.0146$ at $N_"ref" = 5000$. The two-sample
non-centrality parameter is:

$ "NCP" = frac(delta, sqrt(2) sigma_"ap"(N_"gal") / sqrt(B)). $

Power results for $B = 500$ replicates per condition are shown in @tab-power,
with the full power curve in @fig-power.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (right, right, right, right),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*$N_"gal"$*], [*NCP*], [*Power*], [*$sigma_"ap"$*],
    table.hline(stroke: 0.4pt),
    [1,000],  [1.21],  [< 0.001], [0.033],
    [2,000],  [1.71],  [0.001],   [0.023],
    [5,000],  [2.70],  [0.008],   [0.015],
    [10,000], [3.83],  [0.096],   [0.010],
    [20,000], [5.41],  [0.610],   [0.007],
    [50,000], [8.55],  [> 0.999], [0.005],
    table.hline(stroke: 0.8pt),
    table.cell(colspan: 4)[
      #set text(size: 10pt)
      5-sigma threshold: $alpha = 2.87 times 10^{-7}$.
      $B = 500$ replicates per condition.
    ]
  ),
  caption: [Power curve for detecting $|m| = 0.01$ at 5-sigma significance.],
  placement: auto,
) <tab-power>

#figure(
  image("../figures/power_curve.png", width: 78%),
  caption: [Power to detect $|m| = 0.01$ at 5-sigma significance as a function of the number of source galaxies $N_"gal"$, for $B = 500$ replicates per condition. The dashed crimson line marks the 80% power threshold at $N_"gal" approx$ 24,400.],
  placement: auto,
) <fig-power>

The power curve shows a rapid transition from negligible power at
$N_"gal" = 5000$ to near-certainty at $N_"gal" = 50000$. The minimum
galaxy count for 80% power is $N_"gal" approx 24,400$ (obtained via
`uniroot`). An MC verification at $N_"gal" = 5000$, $B = 500$ gives
$"NCP" = 2.30$ and power $= 0.002$, in agreement with the analytic
prediction (NCP = 2.70, power = 0.008; the 15% gap in NCP reflects
finite-sample fluctuation at $B = 500$).

This result has direct survey design implications: detecting sub-percent
shear calibration errors at cosmological precision requires tens of thousands
of galaxies per field, a finding consistent with requirements from Stage-IV
weak lensing surveys #cite(<bartelmann2001>).

// ============================================================
// SECTION 5 — CONCLUSIONS
// ============================================================
= Conclusions

We have applied the full toolkit of MA 551 (simple Monte Carlo, antithetic
variables, importance sampling, parametric bootstrap, BCa confidence intervals,
and power analysis) to a weak gravitational lensing mass reconstruction
problem. Several findings merit emphasis.

The antithetic design is degenerate for the $L_2$ reconstruction error, a
consequence of the estimator being an even function of the noise. This
illustrates the theoretical requirement (Notes \#13) that antithetic sampling
requires monotonicity of the integrand. The same linearity that makes
aperture mass tractable for bootstrap inference renders it degenerate for
paired power testing.

The BCa confidence interval for aperture mass achieves 97.5% empirical
coverage at nominal 95%, and is nearly identical to the percentile interval
because the acceleration parameter $hat(a) approx 0.006$ is close to zero.
This reflects the near-linearity of the aperture mass functional. The BCa
interval would differ substantially from the percentile interval for
non-linear statistics such as peak kappa, but peak kappa is itself
problematic for bootstrap inference when noise dominates the signal.

The DC mode ambiguity of the KS reconstruction (specifically the mass-sheet
degeneracy) accounts for a 9.6% deficit between the reconstructed and true
aperture mass. This irreducible offset must be accounted for when comparing
bootstrap intervals to the physical truth; the correct target for bootstrap
inference is the noiseless reconstruction, not the true convergence map.

The power analysis shows that detecting $|m| = 0.01$ at 5-sigma with 500
replicates requires approximately 24,400 source galaxies. This is achievable
with modern wide-field surveys but highlights that per-field shear calibration
at the sub-percent level remains statistically demanding.

Future work will incorporate GalSim-simulated galaxy images with realistic
PSF convolution and metacalibration shear response correction, placing this
statistical analysis in a physically complete pipeline #cite(<rowe2015galsimmodulargalaxyimage>).

// ============================================================
// REFERENCES
// ============================================================
#bibliography("refs.bib", style: "apa")

// ============================================================
// APPENDIX — R CODE
// ============================================================
#pagebreak()
#align(center)[= Appendix: R Code]

Selected key functions are reproduced below. Full source is in the
accompanying project repository: https://github.com/AdamField118/MA551_Computational_Statistics/tree/main.

*Kaiser--Squires inverse (lensing.R):*
```r
ks_inverse <- function(gamma1, gamma2, grid, lambda = 0) {
  N     <- grid$N_pix
  g1_ft <- fft(gamma1)
  g2_ft <- fft(gamma2)
  gamma_ft <- g1_ft + 1i * g2_ft          # preserves N x N dims
  KX <- outer(rep(1, N), grid$kx)
  KY <- outer(grid$ky, rep(1, N))
  K2 <- KX^2 + KY^2;  K2[1, 1] <- 1
  D_re <- (KX^2 - KY^2) / K2
  D_im <- (2 * KX * KY) / K2
  D_complex       <- D_re + 1i * D_im;  D_complex[1, 1] <- 0 + 0i
  Dmod2           <- D_re^2 + D_im^2;   Dmod2[1, 1]     <- 1
  kappa_ft        <- Conj(D_complex) / (Dmod2 + lambda) * gamma_ft
  kappa_ft[1, 1]  <- 0 + 0i
  Re(fft(kappa_ft, inverse = TRUE)) / N^2
}
```

*BCa confidence interval (bootstrap.R):*
```r
bca_ci <- function(boot_out, gamma_true, grid, stat_fn = peak_kappa,
                   conf = 0.95, lambda = 0, N_gal = NULL) {
  t_boot <- boot_out$t_boot;  t_obs <- boot_out$t_obs
  alpha  <- 1 - conf
  ci_perc <- quantile(t_boot, c(alpha/2, 1 - alpha/2), names = FALSE)
  z0      <- qnorm(mean(t_boot < t_obs))
  # Column jackknife for acceleration
  N  <- grid$N_pix
  g1 <- gamma_true$gamma1;  g2 <- gamma_true$gamma2
  t_jack <- numeric(N)
  for (j in seq_len(N)) {
    g1j <- g1;  g1j[, j] <- 0
    g2j <- g2;  g2j[, j] <- 0
    t_jack[j] <- stat_fn(ks_inverse(g1j, g2j, grid, lambda))
  }
  tj <- mean(t_jack)
  a  <- sum((tj - t_jack)^3) / (6 * sum((tj - t_jack)^2)^1.5)
  p_lo <- pnorm(z0 + (z0 + qnorm(alpha/2))     / (1 - a*(z0 + qnorm(alpha/2))))
  p_hi <- pnorm(z0 + (z0 + qnorm(1-alpha/2)) / (1 - a*(z0 + qnorm(1-alpha/2))))
  list(percentile = ci_perc,
       bca        = quantile(t_boot, c(p_lo, p_hi), names = FALSE),
       z0 = z0, a = a)
}
```

*Analytic power for two-sample test (run\_all.R):*
```r
unpaired_power_analytic <- function(N_gal, B, delta, sd_ref, N_gal_ref,
                                    alpha = 2.87e-7) {
  sd_ng  <- sd_ref * sqrt(N_gal_ref / N_gal)
  se     <- sqrt(2) * sd_ng / sqrt(B)
  ncp    <- delta / se
  t_crit <- qnorm(1 - alpha / 2)
  pnorm(ncp - t_crit) + pnorm(-ncp - t_crit)
}
# Minimum N_gal via root-finding:
ng_80 <- uniroot(function(ng) unpaired_power_analytic(ng, B=500, ...) - 0.80,
                 c(1e3, 1e7))$root
```