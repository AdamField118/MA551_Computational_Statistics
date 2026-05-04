# Presentation Script
# SLIDE 1 — Title

"Hey everyone, so my project is on applying the Monte Carlo and resampling
methods from this class to a real astrophysics problem — weak gravitational
lensing mass reconstruction. The name sounds intimidating but I promise the
statistics is the main event here, not the physics."

# SLIDE 2 — Section break: Motivation

(just transition, no talking needed — let the slide breathe)


# SLIDE 3 — Weak Gravitational Lensing


"So the basic idea is this: massive objects — galaxy clusters, dark matter —
bend light as it travels to us. That bending distorts the shapes of background
galaxies in a coherent way. So if you measure how galaxy shapes are distorted
across the sky, you can work backwards to figure out where the mass is.

The thing we want to recover is called the convergence, kappa — basically the
projected mass density. And the question this project is really about is: how
do you do statistics on that reconstruction? How do you quantify uncertainty,
detect bias, figure out how much data you need?"


# SLIDE 4 — The Statistical Problem


"The pipeline has two steps. First, there's a forward model that maps kappa to
a quantity called shear, gamma — that's the actual observable, the distortion
field. Second, you invert that to get kappa back from noisy shear measurements.

The noise comes from the fact that galaxies have random intrinsic shapes —
that's called shape noise, sigma_e around 0.26 per component. So every galaxy
gives you a noisy estimate of the local shear.

The m in the equation is a multiplicative bias — real shear estimators are
never perfectly calibrated, so they can be off by some factor (1+m). That's
what I'm trying to detect in the power analysis later."


# SLIDE 5 — Section break: KS Algorithm


(transition)


# SLIDE 6 — Kaiser-Squires Inversion


"The reconstruction method is called Kaiser-Squires. It's a Fourier-domain
inversion — you take the shear, Fourier transform it, apply this kernel D(k),
take the real part, and you get back kappa.

The key property is that the magnitude of D(k) is exactly 1 everywhere except
at k=0. So the inverse is just the conjugate — very clean.

The one thing to know: the DC mode, k=0, is always zero. That means the mean
of kappa is unrecoverable — there's an irreducible ambiguity called the
mass-sheet degeneracy. It matters later when I talk about what the bootstrap
is actually targeting."


# SLIDE 7 — Analytical Validation: 10-Test Suite


"Before I trusted any of the statistical results, I validated the R
implementation against a production Python library called SMPy. I wrote ten
analytical tests — things like verifying the round-trip error is below 1e-10,
checking the kernel magnitude, B-mode purity.

Everything passed. And actually the two implementations are algebraically
identical — I proved it line by line. So I'm confident the forward and inverse
operators are correct."


# SLIDE 8 — Section break: Simulation Study


(transition)


# SLIDE 9 — Simulation Setup


"So here's the setup. I'm working on a 32x32 grid, 0.1 arcmin per pixel,
5000 source galaxies. The true kappa — that's the image on the right — is
just a Gaussian blob centered in the field, peaking at 0.297.

With 5000 galaxies the per-pixel signal-to-noise is about 0.85, so this is
firmly noise-dominated. That turns out to drive a lot of interesting behavior.

I focus on two scalar summaries of the reconstruction throughout: peak kappa,
and aperture mass, which is just the integral of kappa within some radius.
Spoiler: aperture mass behaves much better statistically, and the reason is
exactly what you'd expect from the course — it's linear."


# SLIDE 10 — Simple Monte Carlo


"Simple MC, 500 replicates. The L2 error averages around 1.5 — high, because
we're noise-dominated.

The interesting result is peak kappa: the true peak is 0.297, but the MC mean
is 0.488. Massive positive bias. The reason is just that the maximum of 1024
noisy pixels is dominated by noise spikes, not the cluster. Bootstrap isn't
valid for this statistic — I'll come back to that.

Aperture mass, on the other hand, mean of 0.251 against a target of 0.250.
Nearly unbiased. That's because integration averages out the noise."


# SLIDE 11 — Antithetic Variable Sampling


"For antithetic sampling I pair each noise draw epsilon with negative epsilon.
The variance reduction depends on the correlation between g(epsilon) and
g(-epsilon) — if the function is monotone you get negative correlation and
variance goes down.

Aperture mass: +100% reduction, correlation exactly -1. Makes sense — it's
linear in the noise, so flipping the sign flips the output exactly.

Peak kappa: slightly negative result, the maximum isn't monotone.

But the interesting case is L2 error. The variance reduction is -100%, meaning
antithetic sampling actually makes it worse. Why? Because L2 error only depends
on the norm of the noise, not its sign. So L2(epsilon) equals L2(-epsilon)
exactly — every pair gives identical values, you're just halving your effective
sample size. This is a direct consequence of the monotonicity condition from
Notes 13."


# SLIDE 12 — Importance Sampling


"For importance sampling I'm estimating the bias-integrated reconstruction
error — basically integrating L2(m) over a Gaussian prior on the multiplicative
bias m.

The IS estimate and the uniform grid estimate are almost identical: 1.518 vs
1.520. No efficiency gain. But that's actually the informative result — it
means the reconstruction error is nearly flat as a function of m at this galaxy
density. The bias signal is buried in shape noise. You'd need a lot more
galaxies before IS would pay off here."


# SLIDE 13 — Section break: Bootstrap


(transition)


# SLIDE 14 — Parametric Bootstrap Setup


"For the bootstrap I'm using a parametric setup — the noise distribution is
known exactly, so each replicate draws fresh Gaussian noise from the known
sigma_pix.

For BCa I need the acceleration parameter a-hat, which I estimate via column
jackknife on the shear grid. Delete one column of pixels at a time, rerun
the KS reconstruction — that's only N=32 reconstructions rather than N^2.

Peak kappa BCa is completely degenerate. Every bootstrap replicate adds fresh
noise, so the bootstrap peak always exceeds the noiseless reference. z0 hits
negative infinity. This isn't a bug — it correctly tells you that this
statistic can't be bootstrapped this way.

Aperture mass: bias essentially zero, acceleration 0.006, BCa CI [0.220, 0.278].
Well-behaved."


# SLIDE 15 — Coverage Study


"To check the intervals I ran 200 outer replicates — generate a fresh noisy
observation, compute the CI, check whether the target falls inside.

Both BCa and percentile hit 97.5% at nominal 95%. Slightly conservative, which
makes sense when you know the noise model exactly and the distribution is
exactly Gaussian — you're not losing anything to estimation.

BCa and percentile are almost identical in width because a-hat is so close to
zero. The aperture mass is near-linear, so the estimator's SE barely changes
with the parameter. BCa would start to diverge for a genuinely nonlinear
statistic."


# SLIDE 16 — Section break: Power Analysis


(transition)


# SLIDE 17 — Design: Why Unpaired? (part 1)


"For the power analysis I want to know how many galaxies I need to detect a 1%
multiplicative bias at 5-sigma.

You might think: use a paired t-test, share the noise draws between m=0 and
m=0.01 conditions. But aperture mass is linear in the shear, so the paired
difference D_b = 0.01 times the noiseless aperture mass — a constant. Every
replicate gives exactly the same value. Standard deviation is zero, NCP is
infinite. The paired design is completely degenerate for any linear statistic.

Same linearity that caused the antithetic issue."


# SLIDE 18 — Design: Why Unpaired? (part 2)


"So I use an unpaired design — independent noise in each condition. The effect
size is analytic: delta = 0.01 times the noiseless KS aperture mass, about
0.0025. And the SD scales as 1 over root N_gal from the shape noise model, so
I can write down the NCP analytically."


# SLIDE 19 — Power Curve


"Here's the power table. At 5000 galaxies — our simulation regime — power is
basically zero. At 50,000 you're at essentially 1. The crossover for 80% power
is around 24,400 galaxies, which I found analytically with uniroot.

I verified one MC point at 5000 galaxies and got NCP of 2.30 versus analytic
2.70 — about 15% off, which is just finite-B noise at 500 replicates. In the
right ballpark.

The practical implication: detecting sub-percent shear calibration errors
requires tens of thousands of galaxies per field. That's consistent with what
Stage-IV surveys like Rubin and Euclid are designed to achieve."


# SLIDE 20 — Section break: Conclusions


(transition)


# SLIDE 21 — Summary of Findings


"So pulling it all together — the thing I find most satisfying about this
project is that there's one underlying reason for all the weird results:
linearity of the KS operator.

Aperture mass is linear, so antithetic sampling gives you perfect variance
reduction but paired designs are degenerate. Peak kappa is nonlinear, so
neither works cleanly. Bootstrap BCa is well-behaved for linear statistics
but undefined for nonlinear ones. The same thread runs through every result.

Statistical method choice really is inseparable from the structure of the
estimator."


# SLIDE 22 — Connections to MA 551


"Everything in the project maps directly to specific lecture notes — bootstrap
from 6 and 7, Monte Carlo from 12, antithetic and IS from 13. I tried to make
sure the project was more than just stringing the methods together — the
methods actually interact and inform each other here."


# SLIDE 23 — Acknowledgements


"Quick thanks to Prof. Wang for the course, and George Vassilakis whose
open-source KS implementation I used as a reference for validation."


# SLIDE 24 — Questions


"That's it — happy to take questions."


# LIKELY QUESTIONS + SHORT ANSWERS


Q: Why is the L2 error so high (1.5)?
A: "The grid is noise-dominated — per-pixel SNR is under 1.
    L2 error is normalized by the true kappa norm, so when noise is large
    relative to the signal, 1.5 is expected. It's not a calibration issue."

Q: Why not just use the true kappa as the bootstrap target?
A: "Because the KS inversion zeros the DC mode, so the reconstructed map
    always has mean zero — it can never match the true kappa mean. The
    correct target is the noiseless KS reconstruction, not the physical truth."

Q: Couldn't you just use a different statistic that's robust to the peak bias?
A: "Exactly — that's why aperture mass is the right statistic. It's the
    standard choice in the lensing literature for exactly this reason.
    Peak kappa is useful for detection but not for inference."

Q: What would you do differently with more time?
A: "Run with GalSim-simulated images and NGmix with metacalibration for shape
    measurement — that would give realistic PSF convolution and a measured
    rather than injected bias, which is the physically relevant scenario."
