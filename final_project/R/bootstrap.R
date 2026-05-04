# =============================================================================
# R/bootstrap.R
# Bootstrap confidence intervals (percentile and BCa) for scalar summaries
# of the KS mass reconstruction: peak_kappa and aperture_mass.
#
# Design:
#   Parametric bootstrap -- resample shape noise from its known distribution
#   N(0, sigma_pix^2).  This is appropriate because the noise model is known
#   exactly (unlike an empirical bootstrap which resamples observed data).
#
#   BCa acceleration is estimated via delete-one-column jackknife on the
#   pixelised shear grid (each "observation" is one column of gamma pixels).
#   This is O(N) reconstructions rather than O(N^2) and is standard practice
#   for spatial data.
#
# Main functions:
#   boot_ks()          -- single bootstrap run, returns B summary values
#   bca_ci()           -- BCa CI from a bootstrap vector
#   boot_coverage()    -- empirical coverage study over many outer replicates
#   boot_bias_se()     -- bias and SE of a summary statistic
# =============================================================================


# -----------------------------------------------------------------------------
# boot_ks()
# Parametric bootstrap for a scalar summary of the KS reconstruction.
#
# Arguments:
#   gamma_true : list($gamma1, $gamma2) -- true shear (no noise)
#   grid       : from make_grid()
#   stat_fn    : function(kappa_hat, grid) -> scalar summary
#   B          : number of bootstrap replicates
#   N_gal      : total source galaxies (sets sigma_pix)
#   sigma_e    : per-component shape noise
#   m          : multiplicative bias (injected)
#   c_bias     : additive bias (injected)
#   lambda     : KS regularization
#   seed       : RNG seed
#
# Returns: list with
#   $t_boot    : length-B vector of bootstrap summary values
#   $t_obs     : observed value (single noiseless or reference reconstruction)
#   $sigma_pix : noise per pixel used
# -----------------------------------------------------------------------------
boot_ks <- function(gamma_true, grid,
                    stat_fn  = peak_kappa,
                    B        = 1000L,
                    N_gal    = 500L,
                    sigma_e  = 0.26,
                    m        = 0,
                    c_bias   = 0,
                    lambda   = 0,
                    seed     = 42L) {
  set.seed(seed)
  N         <- grid$N_pix
  sigma_pix <- sigma_e / sqrt(N_gal / N^2)

  # Biased (but noiseless) reference shear
  g1_ref <- (1 + m) * gamma_true$gamma1 + c_bias
  g2_ref <- (1 + m) * gamma_true$gamma2 + c_bias

  # Observed value: noiseless reference reconstruction
  kappa_obs <- ks_inverse(g1_ref, g2_ref, grid, lambda = lambda)
  t_obs     <- stat_fn(kappa_obs)

  # Bootstrap replicates: add fresh noise each time
  t_boot <- numeric(B)
  for (b in seq_len(B)) {
    g1_b    <- g1_ref + matrix(rnorm(N^2, 0, sigma_pix), N, N)
    g2_b    <- g2_ref + matrix(rnorm(N^2, 0, sigma_pix), N, N)
    kappa_b <- ks_inverse(g1_b, g2_b, grid, lambda = lambda)
    t_boot[b] <- stat_fn(kappa_b)
  }

  list(t_boot = t_boot, t_obs = t_obs, sigma_pix = sigma_pix,
       N_gal = N_gal, B = B, lambda = lambda)
}


# -----------------------------------------------------------------------------
# bca_ci()
# BCa (bias-corrected and accelerated) confidence interval.
#
# Arguments:
#   boot_out  : output from boot_ks()
#   gamma_true, grid : needed for jackknife acceleration estimate
#   stat_fn   : same summary function used in boot_ks()
#   conf      : nominal coverage (default 0.95)
#   lambda    : KS regularization
#
# Returns: list with
#   $percentile : c(lower, upper) -- standard percentile CI
#   $bca        : c(lower, upper) -- BCa CI
#   $z0         : bias-correction
#   $a          : acceleration
# -----------------------------------------------------------------------------
bca_ci <- function(boot_out, gamma_true, grid,
                   stat_fn = peak_kappa,
                   conf    = 0.95,
                   lambda  = 0,
                   m       = 0,
                   c_bias  = 0,
                   N_gal   = NULL) {
  t_boot <- boot_out$t_boot
  t_obs  <- boot_out$t_obs
  B      <- length(t_boot)
  alpha  <- 1 - conf

  # --- Percentile CI ---
  ci_perc <- quantile(t_boot, c(alpha / 2, 1 - alpha / 2), names = FALSE)

  # --- Bias correction z0 ---
  z0 <- qnorm(mean(t_boot < t_obs))

  # --- Acceleration a via column-jackknife on the shear grid ---
  # Delete one column of pixels at a time, recompute noiseless summary.
  # Using noiseless reconstruction for jackknife (acceleration is a
  # property of the estimator geometry, not the noise).
  N     <- grid$N_pix
  N_gal <- if (is.null(N_gal)) boot_out$N_gal else N_gal
  g1_ref <- (1 + m) * gamma_true$gamma1 + c_bias
  g2_ref <- (1 + m) * gamma_true$gamma2 + c_bias

  t_jack <- numeric(N)
  for (j in seq_len(N)) {
    g1_j   <- g1_ref;  g1_j[, j] <- 0
    g2_j   <- g2_ref;  g2_j[, j] <- 0
    kh_j   <- ks_inverse(g1_j, g2_j, grid, lambda = lambda)
    t_jack[j] <- stat_fn(kh_j)
  }
  tj_mean <- mean(t_jack)
  num_a   <- sum((tj_mean - t_jack)^3)
  den_a   <- 6 * sum((tj_mean - t_jack)^2)^1.5
  a       <- if (abs(den_a) > 1e-15) num_a / den_a else 0

  # --- BCa quantile levels ---
  z_lo  <- qnorm(alpha / 2)
  z_hi  <- qnorm(1 - alpha / 2)
  p_lo  <- pnorm(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
  p_hi  <- pnorm(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

  # Clamp to avoid out-of-range quantiles
  p_lo <- pmax(0.001, pmin(0.999, p_lo))
  p_hi <- pmax(0.001, pmin(0.999, p_hi))

  ci_bca <- quantile(t_boot, c(p_lo, p_hi), names = FALSE)

  list(percentile = ci_perc, bca = ci_bca,
       z0 = z0, a = a, p_lo = p_lo, p_hi = p_hi)
}


# -----------------------------------------------------------------------------
# boot_bias_se()
# Bootstrap bias and standard error for a summary statistic.
# -----------------------------------------------------------------------------
boot_bias_se <- function(boot_out) {
  t_boot <- boot_out$t_boot
  t_obs  <- boot_out$t_obs
  list(
    bias = mean(t_boot) - t_obs,
    se   = sd(t_boot),
    mean = mean(t_boot),
    t_obs = t_obs
  )
}


# -----------------------------------------------------------------------------
# boot_coverage()
# Empirical coverage study: run n_outer outer MC replicates, each producing
# one noisy observation; compute BCa and percentile CIs; check coverage
# against the true summary value.
#
# This answers: "Do 95% BCa intervals actually contain the truth 95% of
# the time under shape noise?"
#
# Arguments:
#   kappa_true : N x N true convergence
#   gamma_true : list($gamma1, $gamma2)
#   grid       : from make_grid()
#   stat_fn    : scalar summary (peak_kappa or aperture_mass_wrapper)
#   truth      : true value of stat_fn(kappa_true)
#   n_outer    : number of outer replicates (each gives one CI)
#   B          : bootstrap replicates per outer replicate
#   N_gal      : source galaxy count
#   conf       : nominal coverage level
#   lambda     : KS regularization
#   m          : multiplicative bias
#   seed       : base RNG seed
#
# Returns: list with empirical coverage, CI widths, all CIs
# -----------------------------------------------------------------------------
boot_coverage <- function(kappa_true, gamma_true, grid,
                          stat_fn  = peak_kappa,
                          truth    = NULL,
                          n_outer  = 200L,
                          B        = 500L,
                          N_gal    = 500L,
                          conf     = 0.95,
                          lambda   = 0,
                          m        = 0,
                          c_bias   = 0,
                          sigma_e  = 0.26,
                          seed     = 42L) {

  if (is.null(truth)) truth <- stat_fn(kappa_true, grid)

  N         <- grid$N_pix
  sigma_pix <- sigma_e / sqrt(N_gal / N^2)
  g1_ref    <- (1 + m) * gamma_true$gamma1 + c_bias
  g2_ref    <- (1 + m) * gamma_true$gamma2 + c_bias

  ci_perc <- matrix(NA_real_, n_outer, 2)
  ci_bca  <- matrix(NA_real_, n_outer, 2)
  t_obs_all <- numeric(n_outer)

  cat(sprintf("Bootstrap coverage study: n_outer=%d, B=%d, N_gal=%d, conf=%.2f\n",
              n_outer, B, N_gal, conf))

  for (i in seq_len(n_outer)) {
    set.seed(seed + i)

    # One outer noisy observation (the "data" for this replicate)
    g1_obs_i <- g1_ref + matrix(rnorm(N^2, 0, sigma_pix), N, N)
    g2_obs_i <- g2_ref + matrix(rnorm(N^2, 0, sigma_pix), N, N)
    kappa_i  <- ks_inverse(g1_obs_i, g2_obs_i, grid, lambda = lambda)
    t_i      <- stat_fn(kappa_i)
    t_obs_all[i] <- t_i

    # Bootstrap by resampling noise around this observation
    t_boot_i <- numeric(B)
    for (b in seq_len(B)) {
      g1_b   <- g1_obs_i + matrix(rnorm(N^2, 0, sigma_pix), N, N)
      g2_b   <- g2_obs_i + matrix(rnorm(N^2, 0, sigma_pix), N, N)
      kh_b   <- ks_inverse(g1_b, g2_b, grid, lambda = lambda)
      t_boot_i[b] <- stat_fn(kh_b)
    }

    # BCa acceleration (column jackknife)
    t_jack  <- numeric(N)
    for (j in seq_len(N)) {
      g1_j  <- g1_obs_i; g1_j[, j] <- 0
      g2_j  <- g2_obs_i; g2_j[, j] <- 0
      kh_j  <- ks_inverse(g1_j, g2_j, grid, lambda = lambda)
      t_jack[j] <- stat_fn(kh_j)
    }
    tj_mean <- mean(t_jack)
    num_a   <- sum((tj_mean - t_jack)^3)
    den_a   <- 6 * sum((tj_mean - t_jack)^2)^1.5
    a_i     <- if (abs(den_a) > 1e-15) num_a / den_a else 0

    z0_i    <- qnorm(mean(t_boot_i < t_i))
    alpha   <- 1 - conf
    z_lo    <- qnorm(alpha / 2); z_hi <- qnorm(1 - alpha / 2)
    p_lo    <- pnorm(z0_i + (z0_i + z_lo) / (1 - a_i * (z0_i + z_lo)))
    p_hi    <- pnorm(z0_i + (z0_i + z_hi) / (1 - a_i * (z0_i + z_hi)))
    p_lo    <- pmax(0.001, pmin(0.999, p_lo))
    p_hi    <- pmax(0.001, pmin(0.999, p_hi))

    alpha_v    <- 1 - conf
    ci_perc[i, ] <- quantile(t_boot_i, c(alpha_v / 2, 1 - alpha_v / 2),
                              names = FALSE)
    ci_bca[i, ]  <- quantile(t_boot_i, c(p_lo, p_hi), names = FALSE)

    if (i %% 50 == 0)
      cat(sprintf("  outer replicate %d/%d done\n", i, n_outer))
  }

  cover_perc <- mean(ci_perc[, 1] <= truth & truth <= ci_perc[, 2])
  cover_bca  <- mean(ci_bca[, 1]  <= truth & truth <= ci_bca[, 2])
  width_perc <- mean(ci_perc[, 2] - ci_perc[, 1])
  width_bca  <- mean(ci_bca[, 2]  - ci_bca[, 1])

  cat(sprintf("\nCoverage results (nominal = %.0f%%):\n", conf * 100))
  cat(sprintf("  Percentile CI: coverage = %.3f  mean width = %.4f\n",
              cover_perc, width_perc))
  cat(sprintf("  BCa CI:        coverage = %.3f  mean width = %.4f\n",
              cover_bca, width_bca))

  list(
    cover_perc = cover_perc,  cover_bca  = cover_bca,
    width_perc = width_perc,  width_bca  = width_bca,
    ci_perc    = ci_perc,     ci_bca     = ci_bca,
    t_obs      = t_obs_all,   truth      = truth,
    conf       = conf,        n_outer    = n_outer
  )
}


# -----------------------------------------------------------------------------
# make_apmass_fn()
# Returns a one-argument closure over grid for use as a stat_fn.
# Usage: apmass_stat <- make_apmass_fn(grid)
#        boot_ks(..., stat_fn = apmass_stat)
# -----------------------------------------------------------------------------
make_apmass_fn <- function(grid, r_ap = 0.8) {
  function(kappa_hat) aperture_mass(kappa_hat, grid, r_ap = r_ap)
}

# Convenience alias kept for backward compatibility
apmass_fn <- function(kappa_hat, grid, r_ap = 0.8) {
  aperture_mass(kappa_hat, grid, r_ap = r_ap)
}
