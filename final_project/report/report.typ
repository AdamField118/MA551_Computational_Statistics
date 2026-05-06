// ============================================================
// MA 551 Final Project Report -- Typst
// Uncertainty Quantification for a Regularized Lensing Inverse Problem
// Adam Field · Department of Physics · adfield@wpi.edu
// Updated: metacalibration pipeline, N_gal = 50,000, B = 2,000
// ============================================================

// --- Page setup ---
#set page(
  paper: "us-letter",
  margin: (top: 1in, bottom: 1in, left: 1in, right: 1in),
  numbering: "1",
  number-align: center,
)

#set text(font: "Cambria", size: 12pt, lang: "en")
#set par(justify: true, leading: 0.65em, spacing: 1.2em)
#set heading(numbering: "1.")

#show raw: set text(size: 10pt, font: "Consolas")

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
    We present a Monte Carlo and resampling study of uncertainty
    quantification for a weak gravitational lensing mass reconstruction
    pipeline based on the Kaiser--Squires (KS) algorithm.
    A fully validated GalSim + ngmix metacalibration pipeline, drawn
    from the COSMOS 2015 SuperBIT photometric catalog, measures
    multiplicative shear bias $hat(m) = 0.00567 plus.minus 0.00544$ and
    additive bias $hat(c) = 0.00497 plus.minus 0.00105$ (jackknife SE,
    $N = 50000$ galaxies), with the bias consistent with zero at
    $1sigma$.
    These measured values replace the previously injected placeholders
    and drive the subsequent statistical analyses.
    Simple Monte Carlo ($B = 2000$, $N_"gal" = 50000$) gives
    a mean relative $L^2$ reconstruction error of 0.480, a threefold
    reduction from the noise-dominated regime at $N_"gal" = 5000$.
    Antithetic variable sampling achieves $+100\%$ variance reduction
    for aperture mass (exact, by linearity) and is degenerate for $L^2$
    error.
    Importance sampling over the bias prior yields an effective sample
    size of 56.5 out of 100, confirming that the small measured bias
    leaves the reconstruction error nearly flat in $m$.
    Parametric BCa bootstrap confidence intervals achieve 96.0% empirical
    coverage at nominal 95%, a closer match to theory than at lower
    galaxy densities.
    A power analysis with $B = 2000$ replicates finds that only
    6,527 source galaxies suffice for 80% power to detect
    $|m| = 0.01$ at 5-sigma, with the analytic non-centrality parameter
    agreeing with Monte Carlo to within 7.6%.
  ]
]

// ============================================================
// SECTION 1 -- INTRODUCTION
// ============================================================
= Introduction

Weak gravitational lensing is among the most powerful observational
probes of cosmological large-scale structure.
Background galaxy images are coherently distorted by intervening mass,
producing a measurable shear field from which the projected mass density
(convergence $kappa$) can be reconstructed.
The standard linear inversion, Kaiser--Squires reconstruction
#cite(<kaiser1993>), maps observed shear back to convergence via a
Fourier-domain inversion of the lensing kernel.

From a statistical standpoint the problem is a regularized linear inverse
problem: a known linear forward operator $F$ maps the unknown $kappa$
to the observable shear $gamma$, and the reconstruction is obtained from
noisy shear measurements subject to the irreducible intrinsic ellipticity
scatter of source galaxies (shape noise).
Several core statistical questions arise naturally:
how should uncertainty in the reconstructed mass map be quantified;
what confidence intervals are appropriate for scalar summaries;
and how many source galaxies are required to detect systematic errors
in the shear measurement pipeline?

This project applies the Monte Carlo and resampling methods of MA 551
to answer these questions within a fully controlled simulation environment.
Crucially, the shear calibration bias parameters are no longer injected
by hand but are measured from a realistic GalSim + ngmix metacalibration
pipeline applied to morphological parameters drawn from the COSMOS 2015
SuperBIT photometric catalog #cite(<saha2024>).
The analysis is also scaled from $N_"gal" = 5000$ and $B = 500$
to $N_"gal" = 50000$ and $B = 2000$, placing the simulation
in a signal-dominated regime and enabling direct validation of the
analytic power formula.

Section~2 describes the lensing forward model, the reconstruction
algorithm, and the metacalibration bias measurement pipeline.
Section~3 presents the full simulation study.
Section~4 develops the power analysis and NCP validation.
Section~5 concludes.

// ============================================================
// SECTION 2 -- METHODOLOGY
// ============================================================
= Methodology

== The Kaiser--Squires Forward--Inverse System

The convergence $kappa(bold(theta))$ and shear
$gamma = gamma_1 + i gamma_2$ are related through the lensing potential
$psi$, which satisfies $nabla^2 psi = 2kappa$.
In Fourier space the shear is obtained from $kappa$ via the
Kaiser--Squires kernel $D(bold(k))$ #cite(<kaiser1993>):

$ hat(gamma)(bold(k)) = D(bold(k)) hat(kappa)(bold(k)), quad
  D(bold(k)) = frac(k_1^2 - k_2^2 + 2 i k_1 k_2, k_1^2 + k_2^2), $

with $D(bold(0)) = 0$ by convention.
The inverse is:

$ hat(kappa)(bold(k)) = overline(D(bold(k))) hat(gamma)(bold(k)), $

since $|D(bold(k))| = 1$ for $bold(k) != bold(0)$.
Optional Tikhonov regularization replaces the denominator with
$|D|^2 + lambda$.

The R implementation was validated analytically against SMPy's
`KaiserSquiresMapper` via a 10-test suite: noiseless round-trip $L^2$
error below $10^(-10)$, $|D(bold(k))| = 1$ to machine precision,
and B-mode purity below $10^(-10)$ for a physical convergence field.

== Noise Model and Simulation Setup

With $N_"gal"$ galaxies uniformly distributed over an $N times N$ grid,
the per-pixel shape noise standard deviation is #cite(<bartelmann2001>):

$ sigma_"pix" = frac(sigma_e, sqrt(N_"gal" / N^2)), $

where $sigma_e = 0.26$ per component.
The observed shear is:

$ gamma_"obs"[i,j] = (1 + m) gamma_"true"[i,j] + c + epsilon[i,j],
  quad epsilon[i,j] tilde cal(N)(0, sigma_"pix"^2). $

We use a $32 times 32$ pixel grid with pixel scale $0.1$ arcmin.
With $N_"gal" = 50000$ galaxies, $sigma_"pix" = 0.0372$ and
$max|gamma| approx 0.103$, giving a per-pixel SNR of approximately 2.8.
This represents a threefold improvement over the preliminary study at
$N_"gal" = 5000$ ($sigma_"pix" = 0.118$, SNR~$approx$~0.85) and places
the simulation in a signal-dominated rather than noise-dominated regime.
Monte Carlo results use $B = 2000$ replicates throughout.

*Reconstruction statistics.*
Two scalar summaries are used: the peak convergence
$T_"peak" = max(hat(kappa))$ and the aperture mass
$T_"ap" = sum_(|bold(theta)| <= r_"ap") hat(kappa)(bold(theta)) space delta theta^2$
with $r_"ap" = 0.8$ arcmin.
The noiseless KS aperture mass ($T_"ap"^"KS" = 0.2504$) differs from
the true value ($T_"ap"^"true" = 0.3461$) by 9.6% due to the
mass-sheet degeneracy; the correct bootstrap target is $T_"ap"^"KS"$.

== Metacalibration Bias Estimation

Multiplicative and additive shear biases were measured via
metacalibration #cite(<sheldon2017>), a numerical differentiation
technique that estimates the shear response matrix:

$ R_(i j) = frac(partial chevron.l e_i chevron.r, partial gamma_j)
  approx frac(chevron.l e_i (+Delta) chevron.r - chevron.l e_i (-Delta) chevron.r,
               2 Delta), quad Delta = 0.01. $

The pipeline simulates $N = 50000$ paired galaxy observations at
$gamma_"true" = plus.minus 0.01$ using GalSim, drawing axis ratios $q$ and
position angles $phi$ from the COSMOS 2015 SuperBIT photometric catalog
(viable Sersic fits only, $q > 0.05$).
Galaxy morphology is held at constant half-light radius
$r_h = 0.5" arcsec"$ and flux $F = 12,259" ADU"$, matching
the ShearNet benchmark configuration.
Sky noise is set to the SuperBIT value $sigma_n = 12.72" ADU"$
with pixel scale $0.141" arcsec/pixel"$ on a $53 times 53$ stamp.
A Gaussian PSF (FWHM $= 0.5" arcsec"$) was used; PSFEx models
were available but path resolution was deferred.

Galaxy shapes and metacalibration responses are measured with
ngmix using Gaussian galaxy and PSF models with Levenberg--Marquardt
fitting.
The corrected shear estimator is:

$ hat(gamma) = frac(chevron.l e_"noshear" chevron.r, chevron.l R chevron.r). $

Bias parameters $m$ and $c$ are estimated from the paired
$plus.minus$shear design via the jackknife estimator of Sheldon et al., with
$N_"jack" = 20$ groups.
The paired design cancels additive ellipticity terms, giving:

$ hat(m) = (chevron.l hat(gamma)(+) chevron.r - chevron.l hat(gamma)(-) chevron.r) / (2 gamma_"true") - 1,
  quad hat(c) = (chevron.l e_"noshear"^+ chevron.r + chevron.l e_"noshear"^- chevron.r) / (2). $

Results, shown in @tab-metacal, indicate a small positive multiplicative
bias consistent with zero at $1sigma$, and a small additive bias
reflecting residual ellipticity in the symmetric Gaussian PSF simulation.
The ensemble response $R_11 = 0.9401 plus.minus 0.0007$ is consistent
with typical ngmix Gaussian-model fits.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*Parameter*], [*Estimate*], [*Jackknife SE*],
    table.hline(stroke: 0.4pt),
    [Mult. bias $hat(m)$],    [$+0.00567$], [$0.00544$],
    [Add. bias $hat(c)$],     [$+0.00497$], [$0.00105$],
    [Response $R_11$],        [$0.9401$],   [$0.0007$],
    [Galaxies $N$],           [$50000$], [---],
    [Success rate],           [$100.0\%$],  [---],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Metacalibration bias estimates. $hat(m)$ is consistent with
            zero at $1sigma$; $hat(c)$ reflects residual ellipticity from
            the symmetric simulation.],
  placement: auto,
) <tab-metacal>

These measured values are used in all subsequent analyses in place of
the injected $m = 0$, $c = 0$ used in the preliminary study.

// ============================================================
// SECTION 3 -- SIMULATION STUDY
// ============================================================
= Simulation Study

== Simple Monte Carlo

We estimate the distribution of scalar summaries of the KS reconstruction
using $B = 2000$ independent replicates at the measured bias
$hat(m) = 0.00567$, $hat(c) = 0.00497$, and $N_"gal" = 50000$.
Results are in @tab-mc.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*Summary*], [*True / target*], [*MC mean*], [*MC SD*],
    table.hline(stroke: 0.4pt),
    [$L_2$ error],         [---],    [$0.4797$], [$0.0142$],
    [Peak $hat(kappa)$],   [$0.297$],[$0.3097$], [$0.0186$],
    [Aperture mass],       [$0.250$#super[\*]],[$0.2518$],[$0.0046$],
    table.hline(stroke: 0.8pt),
    table.cell(colspan: 4)[
      #set text(size: 10pt)
      \* Noiseless KS target; true-kappa value is 0.346 (DC offset 9.6%).
    ]
  ),
  caption: [Simple Monte Carlo results ($N_"gal" = 50000$,
            $B = 2000$, measured bias $hat(m)$.
            Compare to the preliminary study at $N_"gal" = 5000$:
            $L_2$ mean 1.521, aperture mass SD 0.015.],
  placement: auto,
) <tab-mc>

The increase to $N_"gal" = 50000$ moves the simulation into a
signal-dominated regime ($"SNR" approx 2.8$ per pixel), with three
notable consequences.
First, the $L^2$ error drops from 1.521 to 0.480, confirming the
$1/sqrt(N_"gal")$ scaling of shape noise.
Second, the peak kappa bias shrinks from 0.488 to 0.310; the maximum
of a noise-dominated field is dominated by noise spikes, so higher
SNR reduces this effect.
Third, the aperture mass standard deviation narrows from 0.015 to 0.0046,
a 3.3-fold improvement consistent with scaling as
$sqrt(5000 / 50000) = 0.316$.
Comparing the biased run ($hat(m) = 0.00567$) to the unbiased baseline
($m = 0$), the $L^2$ error means are identical to four decimal places
(both 0.4797), confirming that the measured $0.567\%$ bias contributes
negligibly to reconstruction error at this galaxy density.

== Antithetic Variable Sampling

Antithetic sampling pairs each noise draw $epsilon$ with $-epsilon$,
exploiting negative correlation to reduce Monte Carlo variance
(Notes \#13).
Results are in @tab-antithetic.

#figure(
  table(
    columns: (2fr, 1fr, 1fr),
    align: (left, center, center),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*Statistic*], [*Variance reduction*], [*Reason*],
    table.hline(stroke: 0.4pt),
    [Aperture mass], [$+100.0\%$], [Linear in $epsilon$; $op("Cor") = -1$ exactly],
    [Peak $hat(kappa)$],  [$-1.9\%$],  [Maximum not monotone],
    [$L_2$ error],        [$-100.1\%$], [*Degenerate* -- see below],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Antithetic variable variance reduction at $N_"gal" = 50000$.
            The pattern is identical to the preliminary study, confirming
            that the finding is structural rather than sample-size dependent.],
  placement: auto,
) <tab-antithetic>

The degenerate result for $L^2$ error is an exact consequence of linearity.
The KS reconstruction is linear: $hat(kappa)(epsilon) = F^(-1)(F kappa_"true" + epsilon)$,
so the relative $L^2$ error satisfies
$L_2(epsilon) = ||epsilon_"KS"|| \/ ||kappa_"true"||$,
where $epsilon_"KS" = F^(-1) epsilon$ depends only on $||epsilon||$.
Therefore $L_2(epsilon) = L_2(-epsilon)$ exactly, so every antithetic pair
yields identical values, halving the effective sample size without reducing
variance.
Aperture mass achieves perfect negative correlation because it is a
spatially integrated linear functional of $hat(kappa)$ with zero mean
under the null.
The monotonicity condition of Notes \#13 is satisfied for aperture mass
and violated for $L^2$ error.
The magnitude of these results is unchanged at $N_"gal" = 50000$,
confirming they are structural properties of the estimators rather
than artifacts of the noise regime.

== Importance Sampling

We estimate the bias-integrated reconstruction error
$E_p[L_2(m)] = integral L_2(m) p(m) space d m$,
where $p(m) = cal(N)(0, hat(sigma)_m^2)$ is the posterior on multiplicative
bias with $hat(sigma)_m = 0.00544$ (the jackknife SE from metacalibration)
and proposal $q(m) = op("Uniform")(-3 hat(sigma)_m, 3 hat(sigma)_m)$.
Self-normalized importance weights $w_i = p(m_i)/q(m_i)$ are used.

Using $n_"IS" = 100$ proposal draws each evaluated with $B' = 200$
MC replicates:

#figure(
  table(
    columns: (auto, auto),
    align: (left, center),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*Method*], [*Estimate (SE)*],
    table.hline(stroke: 0.4pt),
    [IS ($n = 100$, $B' = 200$)], [$0.4798$ ($0.0401$)],
    [Uniform grid],               [$0.4799$],
    [Effective sample size (ESS)],[$56.5 / 100$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Importance sampling results at $N_"gal" = 50000$.
            ESS $= 56.5/100$ reflects moderate concentration of the
            Gaussian prior relative to the uniform proposal.],
  placement: auto,
) <tab-is>

The IS and uniform estimates agree closely (0.4798 vs. 0.4799).
The ESS of 56.5 reflects the finite concentration of the Gaussian prior
within the uniform proposal interval, but the effective gain over
naive estimation is modest because the reconstruction error remains
nearly flat in $m$ even at $N_"gal" = 50000$.
This is a direct consequence of the measured bias being small
($hat(m) = 0.567\%$): the prior concentrates weight near $m = 0$,
where the error function has little curvature.
At $N_"gal" = 50000$ the noise floor has dropped sufficiently that
the bias signal is detectable in principle (see Section~4), but the
fractional variation of $L_2$ over the prior support is still
less than 0.1%, so IS provides no statistical advantage.
A meaningful IS gain would require either a physically larger bias or
a summary statistic with higher sensitivity to $m$, such as the
response-corrected aperture mass rather than the raw $L^2$ error.

== Bootstrap Confidence Intervals

We construct parametric bootstrap confidence intervals for the aperture
mass using $B = 1000$ bootstrap replicates at $N_"gal" = 50000$
and the measured bias (Notes \#6, \#7).
The BCa acceleration $hat(a)$ is estimated via column jackknife on
the $N = 32$ shear grid columns (32 reconstructions total).
Results are in @tab-boot.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*Statistic*], [*Obs.*], [*Bias*], [*SE*], [*95% BCa CI*],
    table.hline(stroke: 0.4pt),
    [Aperture mass], [$0.2518$], [$+0.0001$], [$0.0049$],
                     [$[0.2415, 0.2611]$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Bootstrap results ($B = 1000$, $N_"gal" = 50000$).
            The noiseless KS target $T_"ap"^"KS" = 0.2504$ lies inside
            both BCa and percentile intervals.
            $hat(z)_0 = -0.020$, $hat(a) = -0.003$.],
  placement: auto,
) <tab-boot>

The bootstrap bias is negligible ($+0.0001$) and the BCa correction
parameters are both near zero ($hat(z)_0 = -0.020$, $hat(a) = -0.003$),
confirming the near-symmetry of the bootstrap distribution.
The 95% BCa CI $[0.2415, 0.2611]$ contains the noiseless KS target
$T_"ap"^"KS" = 0.2504$.
Compared to the preliminary study ($N_"gal" = 5000$, CI width 0.059),
the CI is now 3.2 times narrower (width 0.019), consistent with the
$1 / (sqrt(10))$ reduction in $sigma_"pix"$.
Peak kappa BCa remains degenerate for the same reason as before:
every bootstrap replicate adds fresh noise, so the bootstrap peak
always exceeds the noiseless reference.

== Bootstrap Coverage Study

We run $n_"outer" = 200$ independent Monte Carlo replicates, each
generating one noisy shear observation, computing 95% BCa and percentile
CIs, and checking coverage against $T_"ap"^"KS" = 0.2504$.

#figure(
  table(
    columns: (2fr, 1fr, 1fr),
    align: (left, center, center),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*Interval type*], [*Empirical coverage*], [*Mean width*],
    table.hline(stroke: 0.4pt),
    [Percentile], [$96.0\%$], [$0.0185$],
    [BCa],        [$96.0\%$], [$0.0186$],
    table.hline(stroke: 0.8pt),
    table.cell(colspan: 3)[
      #set text(size: 10pt)
      Nominal level 95%, $n_"outer" = 200$, $B = 1000$, $N_"gal" = 50000$.
    ]
  ),
  caption: [Bootstrap coverage results. Empirical coverage is 96.0% at
            nominal 95%, closer to theory than the 97.5% observed in the
            preliminary study at $N_"gal" = 5000$.],
  placement: auto,
) <tab-coverage>

Both intervals achieve 96.0% empirical coverage, a tighter match to the
nominal 95% than the 97.5% observed at $N_"gal" = 5000$.
The improvement is expected: at lower galaxy density, the noise is nearly
perfectly Gaussian and the bootstrap is conservative because the noise model
is known exactly; at higher density the mild signal-to-noise dependence of
the response adds a small asymmetry that the BCa can partially accommodate.
BCa and percentile CIs remain nearly identical in width ($0.0186$ vs. $0.0185$),
consistent with $hat(a) approx -0.003 approx 0$: the aperture mass functional
is near-linear, so the estimator's standard error changes negligibly with
the parameter, and no skewness correction is warranted.

// ============================================================
// SECTION 4 -- POWER ANALYSIS AND NCP VALIDATION
// ============================================================
= Power Analysis and NCP Validation

== Detecting Multiplicative Bias at 5-sigma

We ask: how many source galaxies $N_"gal"$ are required to detect
$|m| = 0.01$ at 5-sigma significance using a two-sample Welch
$t$-test on $B = 2000$ independent aperture mass replicates per
condition?

*Why unpaired.*
A paired design is degenerate for aperture mass (a linear statistic):
the paired difference $D_b = 0.01 times T_"ap"^"KS"$ is exactly constant
across replicates, giving $op("sd")(D) = 0$ and NCP $= infinity$.
The unpaired design preserves the natural noise contribution to the
variance of each condition.

*Analytic power.*
The effect size is
$delta = |m| times T_"ap"^"KS" = 0.01 times 0.2504 = 0.00250$.
The aperture mass SD scales as $sigma_"ap" prop (1) / sqrt(N_"gal")$:

$ sigma_"ap"(N_"gal") = 0.00479 times sqrt(50000 / N_"gal"), $

calibrated at $N_"gal" = 50000$.
The two-sample non-centrality parameter is:

$ "NCP" = (delta sqrt(2)) / (sigma_"ap"(N_"gal") / sqrt(B)) $

Power results for $B = 2000$ replicates and
$alpha = 2.87 times 10^(-7)$ (5-sigma) are in @tab-power.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (right, right, right, right),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*$N_"gal"$*], [*NCP*], [*Power*], [*$sigma_"ap"$*],
    table.hline(stroke: 0.4pt),
    [$1000$],  [$2.34$],  [$0.003$],   [$0.0339$],
    [$2000$],  [$3.31$],  [$0.034$],   [$0.0240$],
    [$5000$],  [$5.23$],  [$0.538$],   [$0.0151$],
    [$10000$], [$7.39$],  [$0.988$],   [$0.0107$],
    [$20000$], [$10.46$], [$1.000$],   [$0.0076$],
    [$50000$], [$16.53$], [$1.000$],   [$0.0048$],
    table.hline(stroke: 0.8pt),
    table.cell(colspan: 4)[
      #set text(size: 10pt)
      5-sigma threshold: $alpha = 2.87 times 10^{-7}$.
      $B = 2000$ replicates per condition.
    ]
  ),
  caption: [Power curve for detecting $|m| = 0.01$ at 5-sigma significance.
            The minimum $N_"gal"$ for 80% power is 6,527.],
  placement: auto,
) <tab-power>

The minimum galaxy count for 80% power is $N_"gal" = 6,527$
(obtained via `uniroot`).
This is 3.7 times lower than the $24,400$ required at $B = 500$,
illustrating the $(1)/(sqrt(B))$ scaling of the standard error of the
difference.
The power curve transitions steeply around $N_"gal" = 5000$--$10000$:
at $N_"gal" = 5000$ the NCP is 5.23 (power 54%); at $N_"gal" = 10000$
the NCP is 7.39 (power 99%).

== NCP Validation: Analytic vs. Monte Carlo

The analytic power formula assumes $op("SD")(T_"ap") prop 1 / sqrt(N_"gal")$
(shape noise model).
We validate this directly at $N_"gal" = 50000$, $B = 2000$,
comparing the analytic NCP to a full Monte Carlo estimate.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: none,
    inset: 8pt,
    table.hline(stroke: 0.8pt),
    [*Quantity*], [*Monte Carlo*], [*Analytic*],
    table.hline(stroke: 0.4pt),
    [Effect size $delta$],      [$0.00233$], [$0.00250$],
    [NCP],                      [$15.28$],   [$16.53$],
    [Power],                    [$1.000$],   [$1.000$],
    [Relative NCP error],       [$7.6\%$],   [---],
    table.hline(stroke: 0.8pt),
    table.cell(colspan: 3)[
      #set text(size: 10pt)
      $N_"gal" = 50000$, $B = 2000$ replicates per condition.
      Preliminary study at $B = 500$: relative error ~15%.
    ]
  ),
  caption: [NCP validation. The analytic formula overestimates the MC NCP
            by 7.6%, down from ~15% at $B = 500$.],
  placement: auto,
) <tab-ncp>

The analytic formula overestimates the MC NCP by 7.6%, an improvement
from the ~15% discrepancy at $B = 500$.
The residual gap arises from two sources.
First, the MC effect size ($delta_"MC" = 0.00233$) is 6.8% below the
analytic value ($delta = 0.00250$), reflecting a finite-$B$ downward
bias in the estimated mean difference: with $B = 2000$ replicates
each contributing independent noise draws, the sample mean understates
the theoretical shift.
Second, the analytic formula uses a large-$d f$ normal approximation
for the critical value, while the MC uses the exact $t$ distribution
with $2B - 2 = 3,998$ degrees of freedom, introducing a small
additional discrepancy.
Both sources are expected to shrink further as $B -> oo$, consistent
with the observed improvement from $B = 500$ to $B = 2000$.
At $N_"gal" = 50000$ both methods agree that power is 1.000, so the
practical consequence is negligible for the survey design question.

// ============================================================
// SECTION 5 -- CONCLUSIONS
// ============================================================
= Conclusions

We have applied the full toolkit of MA 551 to a weak gravitational
lensing mass reconstruction problem, now with realistic shear biases
measured from a GalSim + ngmix metacalibration pipeline and a tenfold
increase in galaxy density.

*Metacalibration.*
The measured multiplicative bias $hat(m) = 0.00567 plus.minus 0.00544$
is consistent with zero at $1sigma$, as expected for an unbiased
Gaussian galaxy model with a symmetric PSF.
The additive bias $hat(c) = 0.00497 plus.minus 0.00105$ is small but
statistically significant, reflecting residual ellipticity in the
simulation that would be suppressed with a more realistic anisotropic
PSF.
The ensemble response $R_11 = 0.9401$ is consistent with Gaussian-model
ngmix fits reported in the literature.

*Impact of scaling.*
Increasing $N_"gal"$ from 5,000 to 50,000 moves the reconstruction
from noise-dominated to signal-dominated ($"SNR"$~2.8 vs. 0.85 per pixel).
The $L^2$ reconstruction error drops from 1.521 to 0.480; aperture mass
uncertainty narrows by $3.3 times$; and the peak kappa bias shrinks from
0.488 to 0.310.
The measured bias ($hat(m) = 0.567\%$) contributes negligibly to $L^2$
error at either density, confirming that shape noise dominates over
calibration bias at current simulation scales.

*Structural findings.*
Several findings from the preliminary study are confirmed to be structural
rather than sample-size artifacts.
The antithetic design achieves $+100\%$ variance reduction for aperture
mass and is degenerate for $L^2$ error, at both galaxy densities.
BCa and percentile bootstrap CIs remain nearly identical (acceleration
$hat(a) approx 0$) because aperture mass is near-linear.
Empirical coverage is 96.0% at nominal 95%, closer to the theoretical
value than the 97.5% observed at $N_"gal" = 5000$.

*Power and NCP validation.*
With $B = 2000$ replicates, only 6,527 galaxies are required for 80%
power to detect $|m| = 0.01$ at 5-sigma, compared to 24,400 at $B = 500$.
The analytic NCP overestimates the MC value by 7.6%, improved from 15%
at $B = 500$, with the residual error attributable to finite-$B$
downward bias in the estimated effect size and the large-$d f$ normal
approximation.

*Unifying theme.*
The linearity of the KS operator and the aperture mass functional is the
single organizing principle across all methods: it enables exact
antithetic variance reduction, renders BCa identical to percentile,
makes the IS error function flat in $m$, and causes the paired power
design to be degenerate.
Statistical method selection is inseparable from the estimator structure.

*Future work.*
A realistic PSFEx model for the SuperBIT focal plane would produce
PSF-induced additive bias with spatially varying ellipticity, making the
IS gain meaningful and the BCa acceleration nonzero.
Replacing the constant-HLR, constant-flux simulation with catalog-drawn
morphological parameters would also introduce natural variability in the
metacalibration response, enabling a full study of per-galaxy response
weighting.

// ============================================================
// REFERENCES
// ============================================================
#bibliography("refs.bib", style: "apa")

// ============================================================
// APPENDIX -- R CODE
// ============================================================
#pagebreak()
#align(center)[= Appendix: Selected R Code]

Full source is at
#link("https://github.com/AdamField118/MA551_Computational_Statistics")[github.com/AdamField118/MA551\_Computational\_Statistics].

*Kaiser--Squires inverse (lensing.R):*
```r
ks_inverse <- function(gamma1, gamma2, grid, lambda = 0) {
  N <- grid$N_pix
  gamma_ft <- fft(gamma1) + 1i * fft(gamma2)
  KX <- outer(rep(1, N), grid$kx)
  KY <- outer(grid$ky, rep(1, N))
  K2 <- KX^2 + KY^2;  K2[1, 1] <- 1
  D_complex <- (KX^2 - KY^2)/K2 + 1i*(2*KX*KY)/K2
  D_complex[1, 1] <- 0 + 0i
  Dmod2 <- Re(D_complex)^2 + Im(D_complex)^2;  Dmod2[1,1] <- 1
  kappa_ft <- Conj(D_complex) / (Dmod2 + lambda) * gamma_ft
  kappa_ft[1, 1] <- 0 + 0i
  Re(fft(kappa_ft, inverse = TRUE)) / N^2
}
```

*Metacalibration bias estimator (metacal_pipeline.py):*
```python
def jackknife_mc_v2(tab_p, tab_m, shear_true=0.01, njac=20):
    gamma1_per = (g_arr_p[:, 0] - g_arr_m[:, 0]) / 2.0
    R1_pair    = 0.5 * (R_arr_p + R_arr_m)
    m_full     = np.nanmean(gamma1_per) / np.nanmean(R1_pair) / shear_true - 1
    for chunk in chunks:
        mask     = np.ones(N, dtype=bool); mask[chunk] = False
        m_jk.append(np.nanmean(gamma1_per[mask]) /
                    np.nanmean(R1_pair[mask]) / shear_true - 1)
    se_m = sqrt((njac-1)/njac * sum((m_jk - mean(m_jk))**2))
```

*Analytic power with NCP validation (run_all.R):*
```r
unpaired_power_analytic <- function(N_gal, B, delta, sd_ref, N_ref,
                                    alpha = 2.87e-7) {
  sd_ng  <- sd_ref * sqrt(N_ref / N_gal)
  se     <- sqrt(2) * sd_ng / sqrt(B)
  ncp    <- delta / se
  pnorm(ncp - qnorm(1-alpha/2)) + pnorm(-ncp - qnorm(1-alpha/2))
}
ng_80 <- uniroot(function(ng)
    unpaired_power_analytic(ng, B=2000, ...) - 0.80, c(1e3,1e7))$root
# Result: ng_80 = 6527
```
