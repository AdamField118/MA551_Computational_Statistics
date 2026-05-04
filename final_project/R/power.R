# =============================================================================
# R/power.R
# Power analysis for detecting multiplicative shear bias m in the KS
# mass reconstruction pipeline.
#
# Scientific question:
#   How many source galaxies N_gal are required to detect a multiplicative
#   bias |m| = m_target at significance level alpha with power 1-beta?
#
# Statistical framing:
#   For each N_gal, run B MC replicates at m=0 and B replicates at m=m_target.
#   Use a two-sample t-test on the L2 reconstruction error (or peak kappa).
#   Power = fraction of replicates where the test correctly rejects H0: m=0.
#   The power curve power(N_gal) gives the required sample size.
#
#   Alternatively (and more directly connected to real surveys): fit a linear
#   regression L2_error ~ m across a range of m values; the slope beta_hat
#   and its SE give a t-statistic for detecting bias.  Minimum N_gal is the
#   point where the 5-sigma detection requirement is met.
#
# Main functions:
#   power_two_sample()   -- power curve via two-sample t-test
#   power_regression()   -- power via slope of L2_error ~ m (regression method)
#   power_curve()        -- full power curve over a grid of N_gal values
#   min_ngal_for_power() -- binary search for minimum N_gal
# =============================================================================


# -----------------------------------------------------------------------------
# simulate_l2_errors()
# Run B MC replicates at a given (N_gal, m) and return L2 errors.
# Internal helper used by all power functions.
# -----------------------------------------------------------------------------
.sim_l2 <- function(gamma_true, kappa_true, grid, B, N_gal, sigma_e,
                    m, lambda, seed) {
  run_mc(gamma_true, kappa_true, grid,
         B       = B,
         N_gal   = N_gal,
         sigma_e = sigma_e,
         m       = m,
         lambda  = lambda,
         seed    = seed)[, "l2_err"]
}


# -----------------------------------------------------------------------------
# power_two_sample()
# Estimate power for detecting m vs m=0 at a given N_gal.
# Uses a one-sided Welch t-test: H0: mu(L2|m=0) >= mu(L2|m_target).
# (Bias increases L2 error, so the alternative is L2|m > L2|m=0.)
#
# Arguments:
#   gamma_true, kappa_true, grid : lensing objects
#   m_target    : multiplicative bias to detect
#   N_gal       : galaxy count to evaluate
#   B           : MC replicates per hypothesis
#   alpha       : significance level (default 0.05, or 2.87e-7 for 5-sigma)
#   sigma_e     : shape noise
#   lambda      : KS regularization
#   seed        : RNG seed
#
# Returns: estimated power (scalar in [0,1])
# -----------------------------------------------------------------------------
power_two_sample <- function(gamma_true, kappa_true, grid,
                             m_target = 0.01,
                             N_gal    = 500L,
                             B        = 500L,
                             alpha    = 2.87e-7,   # 5-sigma two-tailed
                             sigma_e  = 0.26,
                             lambda   = 0,
                             seed     = 42L) {
  l2_null <- .sim_l2(gamma_true, kappa_true, grid, B, N_gal, sigma_e,
                     m = 0,        lambda = lambda, seed = seed)
  l2_alt  <- .sim_l2(gamma_true, kappa_true, grid, B, N_gal, sigma_e,
                     m = m_target, lambda = lambda, seed = seed + 1000L)

  # One-sided Welch t-test: H1: mu_alt > mu_null
  test    <- t.test(l2_alt, l2_null, alternative = "greater",
                    var.equal = FALSE)
  # Analytic power from the non-central t approximation
  n1      <- length(l2_alt);  n2 <- length(l2_null)
  se_diff <- sqrt(var(l2_alt) / n1 + var(l2_null) / n2)
  delta   <- mean(l2_alt) - mean(l2_null)
  df_w    <- (var(l2_alt)/n1 + var(l2_null)/n2)^2 /
             ((var(l2_alt)/n1)^2/(n1-1) + (var(l2_null)/n2)^2/(n2-1))
  t_crit  <- qt(1 - alpha, df = df_w)
  ncp     <- delta / se_diff
  power   <- 1 - pt(t_crit, df = df_w, ncp = ncp)

  list(power   = power,
       delta   = delta,
       se_diff = se_diff,
       ncp     = ncp,
       t_obs   = test$statistic,
       p_value = test$p.value,
       N_gal   = N_gal,
       m_target = m_target)
}


# -----------------------------------------------------------------------------
# power_regression()
# Regression-based power: fit L2_error ~ m over a grid of m values.
# Power = probability that the slope beta_hat is significantly positive.
#
# This is more efficient than two-sample testing because it uses information
# from multiple m values (analogous to the importance sampling in HW4/mc.R).
# It also directly estimates the slope dL2/dm, which has physical meaning.
#
# Returns: list with slope estimate, SE, t-stat, power, and fitted values
# -----------------------------------------------------------------------------
power_regression <- function(gamma_true, kappa_true, grid,
                             m_grid   = seq(-0.05, 0.05, by = 0.01),
                             N_gal    = 500L,
                             B_per_m  = 200L,
                             alpha    = 2.87e-7,
                             sigma_e  = 0.26,
                             lambda   = 0,
                             seed     = 42L) {
  n_m    <- length(m_grid)
  l2_bar <- numeric(n_m)
  l2_se  <- numeric(n_m)

  for (i in seq_along(m_grid)) {
    reps      <- .sim_l2(gamma_true, kappa_true, grid, B_per_m, N_gal,
                         sigma_e, m = m_grid[i], lambda = lambda,
                         seed = seed + i * 100L)
    l2_bar[i] <- mean(reps)
    l2_se[i]  <- sd(reps) / sqrt(B_per_m)
  }

  # Weighted least squares: L2_bar_i ~ beta0 + beta1 * m_i
  # Weights = 1/SE^2 (inverse variance)
  w_i  <- 1 / pmax(l2_se^2, 1e-20)
  fit  <- lm(l2_bar ~ m_grid, weights = w_i)
  coef_sum <- summary(fit)$coefficients

  beta_hat  <- coef_sum["m_grid", "Estimate"]
  beta_se   <- coef_sum["m_grid", "Std. Error"]
  t_stat    <- beta_hat / beta_se
  df_resid  <- fit$df.residual
  t_crit    <- qt(1 - alpha / 2, df = df_resid)   # two-sided
  power_est <- 1 - pt(t_crit - abs(t_stat), df = df_resid) +
               pt(-t_crit - abs(t_stat), df = df_resid)

  list(
    beta_hat  = beta_hat,
    beta_se   = beta_se,
    t_stat    = t_stat,
    power     = power_est,
    df        = df_resid,
    m_grid    = m_grid,
    l2_bar    = l2_bar,
    l2_se     = l2_se,
    fit       = fit,
    N_gal     = N_gal,
    alpha     = alpha
  )
}


# -----------------------------------------------------------------------------
# power_curve()
# Compute power at each N_gal in a grid using the two-sample method.
# The result is the power curve used to read off the required sample size.
#
# Arguments:
#   ngal_grid : vector of N_gal values to evaluate (e.g., 2^(7:14))
#   ...       : other arguments passed to power_two_sample()
#
# Returns: data.frame with N_gal, power, delta, se_diff, ncp
# -----------------------------------------------------------------------------
power_curve <- function(gamma_true, kappa_true, grid,
                        ngal_grid = c(500, 1000, 2000, 5000, 10000, 20000),
                        m_target  = 0.01,
                        B         = 500L,
                        alpha     = 2.87e-7,
                        sigma_e   = 0.26,
                        lambda    = 0,
                        seed      = 42L,
                        verbose   = TRUE) {
  results <- vector("list", length(ngal_grid))

  for (i in seq_along(ngal_grid)) {
    ng <- ngal_grid[i]
    if (verbose)
      cat(sprintf("  N_gal = %6d ... ", ng))

    res <- power_two_sample(gamma_true, kappa_true, grid,
                            m_target = m_target,
                            N_gal    = ng,
                            B        = B,
                            alpha    = alpha,
                            sigma_e  = sigma_e,
                            lambda   = lambda,
                            seed     = seed)
    results[[i]] <- res
    if (verbose)
      cat(sprintf("power = %.3f  delta = %.4f  NCP = %.2f\n",
                  res$power, res$delta, res$ncp))
  }

  data.frame(
    N_gal    = ngal_grid,
    power    = sapply(results, `[[`, "power"),
    delta    = sapply(results, `[[`, "delta"),
    se_diff  = sapply(results, `[[`, "se_diff"),
    ncp      = sapply(results, `[[`, "ncp"),
    m_target = m_target,
    alpha    = alpha
  )
}


# -----------------------------------------------------------------------------
# min_ngal_for_power()
# Binary search for the minimum N_gal achieving target power (default 0.80).
# Uses the analytic power approximation from power_two_sample() for speed.
#
# Arguments:
#   target_power : desired power (default 0.80)
#   ngal_lo, ngal_hi : search bracket
#   tol          : convergence tolerance on N_gal
#   ...          : passed to power_two_sample()
#
# Returns: estimated minimum N_gal and the power at that N_gal
# -----------------------------------------------------------------------------
min_ngal_for_power <- function(gamma_true, kappa_true, grid,
                               target_power = 0.80,
                               m_target     = 0.01,
                               ngal_lo      = 500L,
                               ngal_hi      = 100000L,
                               B            = 300L,
                               alpha        = 2.87e-7,
                               sigma_e      = 0.26,
                               lambda       = 0,
                               seed         = 42L,
                               tol          = 100L,
                               verbose      = TRUE) {

  .power_at <- function(ng) {
    power_two_sample(gamma_true, kappa_true, grid,
                     m_target = m_target, N_gal = as.integer(ng),
                     B = B, alpha = alpha, sigma_e = sigma_e,
                     lambda = lambda, seed = seed)$power
  }

  # Check bracket
  p_lo <- .power_at(ngal_lo)
  p_hi <- .power_at(ngal_hi)
  if (verbose)
    cat(sprintf("Power at N_gal=%d: %.3f\n", ngal_lo, p_lo))
  if (verbose)
    cat(sprintf("Power at N_gal=%d: %.3f\n", ngal_hi, p_hi))

  if (p_lo >= target_power) {
    if (verbose) cat("Lower bound already achieves target power.\n")
    return(list(N_gal = ngal_lo, power = p_lo))
  }
  if (p_hi < target_power) {
    if (verbose) cat("Upper bound does not achieve target power; increase ngal_hi.\n")
    return(list(N_gal = ngal_hi, power = p_hi))
  }

  # Binary search
  iter <- 0L
  while (ngal_hi - ngal_lo > tol) {
    ng_mid <- as.integer((ngal_lo + ngal_hi) / 2)
    p_mid  <- .power_at(ng_mid)
    iter   <- iter + 1L
    if (verbose)
      cat(sprintf("  iter %2d: N_gal=%6d  power=%.3f\n", iter, ng_mid, p_mid))
    if (p_mid < target_power) ngal_lo <- ng_mid else ngal_hi <- ng_mid
  }

  ng_final <- as.integer((ngal_lo + ngal_hi) / 2)
  p_final  <- .power_at(ng_final)
  if (verbose)
    cat(sprintf("Minimum N_gal for power >= %.2f: %d  (actual power: %.3f)\n",
                target_power, ng_final, p_final))

  list(N_gal = ng_final, power = p_final, iter = iter)
}


# -----------------------------------------------------------------------------
# analytic_power_approx()
# Closed-form power approximation for the two-sample test, given an analytic
# model for how reconstruction error scales with N_gal and m.
#
# Model (from shape noise theory):
#   sigma_L2(N_gal) ~ C / sqrt(N_gal)    (L2 error scales as sigma_pix)
#   delta_L2(m)     ~ |m| * mu_L2        (bias shifts mean proportionally)
#
# This gives an analytic power curve without any simulation, useful for
# understanding the scaling before running the full MC.
#
# Arguments:
#   mu_L2   : mean L2 error at m=0 (from MC)
#   sd_L2   : SD of L2 error at m=0, reference N_gal_ref
#   N_gal_ref : N_gal at which sd_L2 was measured
#   m_target  : bias to detect
#   ngal_grid : N_gal values to evaluate
#   alpha     : significance level
#
# Returns: data.frame with N_gal and analytic_power
# -----------------------------------------------------------------------------
analytic_power_approx <- function(mu_L2, sd_L2, N_gal_ref,
                                  m_target  = 0.01,
                                  ngal_grid = c(500, 1000, 2000, 5000, 10000),
                                  alpha     = 2.87e-7) {
  # SD scales as 1/sqrt(N_gal) relative to reference
  sd_at <- function(ng) sd_L2 * sqrt(N_gal_ref / ng)

  # Effect size: bias shifts mean L2 by |m| * mu_L2 (linear approximation)
  delta <- abs(m_target) * mu_L2

  t_crit  <- qnorm(1 - alpha / 2)   # approx for large df
  power_v <- sapply(ngal_grid, function(ng) {
    se  <- sqrt(2) * sd_at(ng)   # pooled SE (two equal groups)
    ncp <- delta / se
    pnorm(ncp - t_crit) + pnorm(-ncp - t_crit)
  })

  data.frame(
    N_gal          = ngal_grid,
    analytic_power = power_v,
    delta          = delta,
    sd_per_ng      = sapply(ngal_grid, sd_at)
  )
}
