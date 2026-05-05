#import "@preview/polylux:0.4.0": *
#import "./slides.typ": *

#set page(paper: "presentation-16-9", margin: 30pt)

// ── Title ──────────────────────────────────────────────────────────────────
#title-slide(
  [Monte Carlo and Resampling Methods Applied to Weak Gravitational Lensing Mass Reconstruction],
  [MA 551 - Computational Statistics],
  (
    ("Adam Field", "Department of Physics"),
  ),
  date: [May 4, 2026],
)

// ── Section 1: Motivation ──────────────────────────────────────────────────
#section-slide([Motivation], number: [01],
  subtitle: [Why lensing? Why statistics?])

#content-slide([Weak Gravitational Lensing])[
  Massive structures bend light from background galaxies, distorting their
  apparent shapes coherently across the sky.

  #v(10pt)

  *The observable:* galaxy ellipticity $e = e_1 + i e_2$

  *What we want:* projected mass density $kappa(bold(theta))$ (convergence)

  #v(10pt)

  - Direct probe of dark matter, dark energy equation of state
  - Stage-IV surveys (Rubin, Euclid, Roman) are already underway
  - Systematic errors at the *sub-percent level* will limit cosmological inference

  #v(10pt)

  *The core problem:* how do we propagate uncertainty, detect bias, and
  determine how much data we need?
]

#content-slide([The Statistical Problem])[
  The mass reconstruction pipeline has two stages:

  + *Forward model:* $kappa arrow.r gamma$ via a known linear FFT operator $F$
  + *Inverse:* recover $hat(kappa)$ from noisy shear observations $gamma_"obs"$

  Each source galaxy contributes an ellipticity measurement with
  intrinsic scatter $sigma_e = 0.26$ per component ("shape noise"):

  $ gamma_"obs"[i,j] = (1+m)gamma_"true"[i,j] + epsilon[i,j], quad
    epsilon[i,j] tilde cal(N)(0, sigma_e^2 \/ n_"gal/pix") $

  *Key questions (all from MA 551):*
  - What is the reconstruction error, and how variable is it? *(MC)*
  - Can we quantify uncertainty efficiently? *(Antithetic, IS)*
  - What confidence intervals are valid? *(Bootstrap, BCa)*
  - How many galaxies to detect 1% bias at 5$sigma$? *(Power analysis)*
]

// ── Section 2: Methods ─────────────────────────────────────────────────────
#section-slide([The Kaiser--Squires Algorithm], number: [02],
  subtitle: [A validated R implementation])

#math-slide(
  [Kaiser--Squires Inversion],
  $ hat(kappa)(bold(k)) = overline(D(bold(k))) hat(gamma)(bold(k)),
    quad D(bold(k)) = frac(k_1^2 - k_2^2 + 2 i k_1 k_2, k_1^2 + k_2^2) $,
  before: [
    Shear and convergence are related in Fourier space via the KS kernel $D(bold(k))$.
    Since $|D(bold(k))| = 1$ for all $bold(k) != bold(0)$, the inverse is simply the conjugate.
    Optional Tikhonov regularization: replace $|D|^2$ with $|D|^2 + lambda$.
  ],
  after: [
    *DC ambiguity:* $D(bold(0)) = 0$ always, so the mean of $kappa$ is unrecoverable
    (mass-sheet degeneracy). The noiseless KS aperture mass (0.250) differs from the
    true value (0.346) by 9.6% due to this irreducible offset.
  ],
)

#content-slide([Analytical Validation: 10-Test Suite])[
  The R implementation was validated against SMPy's `KaiserSquiresMapper`
  (production Python KS). Both compute the same algebraic expression.

  #v(-10pt)

  #table(
    columns: (2fr, 0.5fr),
    stroke: none,
    inset: 6pt,
    table.hline(stroke: 0.5pt),
    [*Test*], [*Result*],
    table.hline(stroke: 0.3pt),
    [FFT frequency convention matches `numpy.fft.fftfreq`], [PASS],
    [$|D(bold(k))| = 1$ for all $bold(k) != bold(0)$ (machine precision)], [PASS],
    [Noiseless round-trip $L_2$ error $< 10^(-10)$], [PASS],
    [Linearity: $F(a kappa_1 + b kappa_2) = a F(kappa_1) + b F(kappa_2)$], [PASS],
    [Parseval / adjoint: $chevron.l F kappa, gamma chevron.r = chevron.l kappa, F^* gamma chevron.r$], [PASS],
    [B-mode purity $< 10^(-10)$ for physical $kappa$], [PASS],
    [SMPy exact formula equivalence: max diff $< 10^(-14)$], [PASS],
    [Single-frequency (sinusoidal) analytical solution], [PASS],
    [DC non-contamination: adding constant to $kappa$ does not change $gamma$], [PASS],
    [Peak amplitude recovered to $10^(-10)$ (DC-corrected)], [PASS],
    table.hline(stroke: 0.5pt),
  )
]

// ── Section 3: Simulation Study ───────────────────────────────────────────
#section-slide([Simulation Study], number: [03],
  subtitle: [Monte Carlo, antithetic variables, importance sampling])

#content-slide([Simulation Setup])[
  #table(
    columns: (1fr, 1fr),
    stroke: none,
    inset: 0pt,
    [
      #v(-14pt)
      *Grid:* $32 times 32$ pixels, $0.1$ arcmin/pixel \
      *Field:* $3.2 times 3.2$ arcmin$#box(width: 0pt)^2$ \
      *True $kappa$:* Gaussian cluster, peak $= 0.297$, $sigma_ell = 0.5$ arcmin \
      *Galaxies:* $N_"gal" = 5,000$ \
      *Noise:* $sigma_"pix" = 0.118$, max $|gamma| = 0.103$ \
      *SNR per pixel:* $approx 0.85$ (noise-dominated)

      #v(0pt)

      *Two scalar summaries:*
      - Peak $hat(kappa)$: maximum pixel value
      - Aperture mass: $T_"ap" = sum_(r <= 0.8) hat(kappa) space delta theta^2$

      Aperture mass is preferred: it integrates over a region rather than
      taking the maximum of a noisy field.
    ],
    [
      #align(center)[
        #box(
          stroke: (paint: muted, thickness: 3pt),
          inset: 0pt,
        )[
          #image("../figures/kappa_true.png", width: 92%)
        ]
      ]
    ]
  )
]

#content-slide([Simple Monte Carlo: $B = 500$ Replicates])[
  Estimate $bb(E)[L_2(m=0)]$ and the distribution of scalar summaries
  under pure shape noise (Notes \#12).

  #v(10pt)

  #table(
    columns: (2fr, 1fr, 1fr, 1fr),
    stroke: none,
    inset: 7pt,
    table.hline(stroke: 0.6pt),
    [*Summary*], [*True value*], [*MC mean*], [*MC SD*],
    table.hline(stroke: 0.3pt),
    [$L_2$ error], [0 ($m=0$)], [1.521], [0.045],
    [Peak $hat(kappa)$], [0.297], [*0.488*], [0.051],
    [Aperture mass], [0.250$#box(width: 0pt)^dagger$], [0.251], [0.015],
    table.hline(stroke: 0.6pt),
  )
  #text(size: 10pt)[$dagger$ Noiseless KS value; true-$kappa$ aperture mass is 0.346 (DC offset 9.6%).]

  #v(6pt)

  - *Peak kappa is severely upward-biased:* the maximum of 1024 noisy pixels
    is a noise spike, not the cluster. Bootstrap is not valid for this statistic.
  - *Aperture mass is nearly unbiased* relative to the correct target (noiseless KS).
  - $L_2$ error mean of 1.52 reflects the noise-dominated regime at $N_"gal" = 5000$.
]

#content-slide([Antithetic Variable Sampling (Notes \#13)])[
  Pair each noise draw $epsilon$ with $-epsilon$. For monotone $g$:

  $ op("Var")(hat(I)_"anti") = frac(sigma^2, 2N)(1 + op("Cor")(g(epsilon), g(-epsilon))) < frac(sigma^2, 2N) $

  #v(-10pt)

  #table(
    columns: (1fr, 1fr, 2fr),
    stroke: none,
    inset: 7pt,
    table.hline(stroke: 0.6pt),
    [*Statistic*], [*Variance reduction*], [*Reason*],
    table.hline(stroke: 0.3pt),
    [Aperture mass], [*+100%*], [Linear in $epsilon$; $op("Cor") = -1$ exactly],
    [Peak $hat(kappa)$], [$-4%$], [Maximum is not monotone],
    [$L_2$ error], [*$-100%$*], [*Degenerate* (see below)],
    table.hline(stroke: 0.6pt),
  )

  #v(-10pt)

  *The degenerate case:* $L_2(epsilon) = ||F^(-1) epsilon|| \/ ||kappa_"true"||$ depends
  only on $||epsilon||$, so $L_2(epsilon) = L_2(-epsilon)$ *exactly*.
  Antithetic pairs are identical; averaging them halves the effective sample size
  without any variance reduction.

  #v(-10pt)

  This illustrates the key condition from Notes \#13: *antithetic sampling
  requires the integrand to be monotone.* An even function in $epsilon$ violates this.
]

#content-slide([Importance Sampling Over Bias (Notes \#12, \#13)])[
  Estimate the bias-integrated error:
  $ bb(E)_p[L_2(m)] = integral L_2(m) p(m) , d m, quad p(m) = cal(N)(0, 0.05^2) $

  Proposal: $q(m) = "Uniform"(-0.15, 0.15)$. Self-normalized weights $w_i = p(m_i)/q(m_i)$.

  #v(-10pt)

  #table(
    columns: (1fr, 1fr),
    stroke: none,
    inset: 7pt,
    table.hline(stroke: 0.6pt),
    [*Method*], [*Estimate (SE)*],
    table.hline(stroke: 0.3pt),
    [IS ($n = 50$, $B' = 100$)], [1.518 (0.193)],
    [Uniform grid], [1.520],
    table.hline(stroke: 0.6pt),
  )

  #v(-10pt)

  The IS and uniform estimates agree closely. This is itself informative:
  *the reconstruction error is nearly flat in $m$ at this galaxy density.*
  The bias signal is buried in shape noise, so IS offers no efficiency gain.
  A more informative regime (higher $N_"gal"$) would show larger IS benefit
  near $m = 0$ where the prior concentrates.
]

// ── Section 4: Bootstrap ──────────────────────────────────────────────────
#section-slide([Bootstrap Inference], number: [04],
  subtitle: [BCa confidence intervals and coverage])

#content-slide([Parametric Bootstrap Setup (Notes \#6, \#7)])[
  *Parametric bootstrap:* draw $B = 500$ fresh noise realizations from the
  known $cal(N)(0, sigma_"pix"^2)$ distribution. The noise model is known exactly.

  #v(-10pt)

  *BCa correction factors:*

  - *Bias correction* $hat(z)_0 = Phi^(-1)(B^(-1) sum_b bb(1)(hat(T)_b^* < hat(T)))$: adjusts for median shift
  - *Acceleration* $hat(a)$: estimated via *column jackknife* on the shear grid.
    Delete column $j$, recompute KS reconstruction. Requires $N = 32$ reconstructions
    (not $N^2$). For near-linear estimators, $hat(a) approx 0$.

  #v(-10pt)

  #table(
    columns: (2fr, 1fr, 1fr, 1fr, 1.5fr),
    stroke: none,
    inset: 7pt,
    table.hline(stroke: 0.6pt),
    [*Statistic*], [*Obs.*], [*Bias*], [*SE*], [*95% BCa CI*],
    table.hline(stroke: 0.3pt),
    [Peak $hat(kappa)$], [0.251], [+0.237], [0.051], [degenerate],
    [Aperture mass], [0.250], [+0.000], [0.015], [$[0.220, 0.278]$],
    table.hline(stroke: 0.6pt),
  )

  #v(-10pt)

  Peak BCa fails: every bootstrap replicate adds fresh noise, so the bootstrap
  peak *always exceeds* the noiseless reference. $hat(z)_0 = Phi^(-1)(0) = -infinity$.
]

#content-slide([Coverage Study: $n_"outer" = 200$ Replicates])[
  #v(-10pt)

  For each outer replicate: generate one noisy observation, compute BCa and
  percentile CI, check coverage against the noiseless KS aperture mass target.

  #v(-10pt)

  #table(
    columns: (2fr, 1fr, 1fr),
    stroke: none,
    inset: 7pt,
    table.hline(stroke: 0.6pt),
    [*Interval type*], [*Empirical coverage*], [*Mean width*],
    table.hline(stroke: 0.3pt),
    [Percentile], [97.5%], [0.0586],
    [BCa], [97.5%], [0.0585],
    table.hline(stroke: 0.6pt),
  )

  #v(-10pt)

  *Interpretation:*
  - Both achieve 97.5% at nominal 95%: slightly conservative, as expected when
    $sigma_"pix"$ is known exactly and the noise is exactly Gaussian.
  - BCa $approx$ percentile here because $hat(a) = 0.006 approx 0$: the aperture
    mass functional is nearly linear in the shear, so the estimator's SE is
    essentially constant and no skewness correction is needed.
  - BCa would diverge from percentile for non-linear statistics (e.g., peak kappa)
    or at lower SNR where asymmetry becomes important.
]

// ── Section 5: Power Analysis ─────────────────────────────────────────────
#section-slide([Power Analysis], number: [05],
  subtitle: [How many galaxies to detect 1% bias at 5$sigma$?])

#content-slide([Design: Why Unpaired?])[

  *Goal:* detect multiplicative bias $|m| = 0.01$ via two-sample Welch $t$-test
  on $B$ independent aperture mass replicates per condition.

  *The paired design is degenerate for linear statistics.*

  Since aperture mass is linear in the shear, the paired difference is:
  $ D_b = T_"ap"(m=0.01) - T_"ap"(m=0) = 0.01 times T_"ap"^"KS" = "const" $

  Every replicate gives exactly the same $D_b$, so $op("sd")(D) = 0$ and NCP $= infinity$.
  This is the same linearity that caused the antithetic degeneracy.

]

// ── Section 5: Power Analysis ─────────────────────────────────────────────

#content-slide([Design: Why Unpaired?])[

  *Unpaired design:* independent noise in each condition. Effect size is analytic:
  $ delta = |m| times T_"ap"^"KS" = 0.01 times 0.2504 = 0.00250 $

  SD scales as $1/sqrt(N_"gal")$ from the shape noise model:
  $ sigma_"ap"(N_"gal") = 0.0146 times sqrt(5000 / N_"gal") $
]

#content-slide([Power Curve: Detecting $|m| = 0.01$ at 5$sigma$])[
  Non-centrality parameter and power for $B = 500$ replicates per condition,
  significance threshold $alpha = 2.87 times 10^(-7)$ (5-sigma).

  #v(-10pt)

  #table(
    columns: (1fr, 1fr, 1fr, 1fr),
    stroke: none,
    inset: 7pt,
    table.hline(stroke: 0.6pt),
    [*$N_"gal"$*], [*NCP*], [*Power*], [*$sigma_"ap"$*],
    table.hline(stroke: 0.3pt),
    [1,000],  [1.21], [$< 0.001$], [0.033],
    [2,000],  [1.71], [0.001],    [0.023],
    [5,000],  [2.70], [0.008],    [0.015],
    [10,000], [3.83], [0.096],    [0.010],
    [20,000], [5.41], [0.610],    [0.007],
    [*50,000*], [*8.55*], [*$> 0.999$*], [*0.005*],
    table.hline(stroke: 0.6pt),
  )

  #v(-10pt)

  *Minimum $N_"gal"$ for 80% power:* $approx$ 24,400 (analytic `uniroot`).

  MC verification at $N_"gal" = 5000$, $B = 500$: NCP $= 2.30$, power $= 0.002$
  (analytic: 2.70, 0.008 -- 15% gap from finite-$B$ fluctuation, as expected).
]

// ── Section 6: Conclusions ────────────────────────────────────────────────
#section-slide([Conclusions], number: [06])

#content-slide([Summary of Findings])[
  #table(
    columns: (auto, 1fr),
    stroke: none,
    inset: 7pt,
    [*Method*], [*Key finding*],
    table.hline(stroke: 0.3pt),
    [Simple MC], [Aperture mass nearly unbiased; peak kappa severely biased by noise spikes],
    [Antithetic], [+100% reduction for aperture mass; *degenerate* for $L_2$ error (even function)],
    [IS], [No gain here: error flat in $m$ at $N_"gal"=5000$; bias buried in noise],
    [Bootstrap], [Aperture mass BCa well-behaved ($hat(a) approx 0$); peak kappa BCa undefined],
    [Coverage], [97.5% empirical at nominal 95%; BCa $approx$ percentile for linear functionals],
    [Power], [24,400 galaxies for 80% power at 5$sigma$ detecting $|m|=0.01$, $B=500$],
  )

  *Unifying theme:* the linearity of the KS operator creates a consistent pattern.
  Linear statistics (aperture mass) are tractable for bootstrap and IS but
  degenerate for antithetic and paired designs. Non-linear statistics (peak)
  are biased and bootstrap-invalid. Statistical method choice is inseparable
  from the structure of the estimator.
]

#content-slide([Connections to MA 551 Curriculum])[
  Every analysis in this project maps directly to course material:

  #v(-10pt)

  - *Notes \#3--4:* simulation-based hypothesis testing, critical values
  - *Notes \#6:* the bootstrap, empirical CDF, resampling
  - *Notes \#7:* BCa CI, jackknife, acceleration factor
  - *Notes \#12:* Monte Carlo integration, simple MC estimation
  - *Notes \#13:* antithetic variables (monotonicity condition), importance
    sampling (proposal design), control variates

  The project is approximately 80% statistical analysis and 20% forward modeling.
  The astrophysics provides a concrete inverse problem where the statistical
  methods produce physically interpretable results.

  #v(-10pt)

  *Future work:* GalSim-simulated galaxy images with realistic PSF convolution
  and metacalibration shear response correction, placing this analysis in a
  physically complete pipeline.
]

#ack-slide((
  ("George Vassilakis ",    "Open-source Kaiser--Squires reference implementation."),
))

#end-slide(
  // bib: bibliography("bib.bib", title: [#v(-1.2em)])
)
