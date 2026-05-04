# =============================================================================
# R/mc.R
# Monte Carlo framework for UQ on the lensing inverse problem.
# Covers: simple MC, antithetic variables, importance sampling.
#
# Dependencies (sourced by run_all.R before this file):
#   lensing.R  -- ks_inverse, l2_error, peak_kappa, aperture_mass
#   simulate.R -- add_shape_noise_to_grid
#   ks_smpy.R  -- ks_reconstruct, smpy_available
# =============================================================================


# -----------------------------------------------------------------------------
# one_replicate()
# Single MC replicate: simulate noise, reconstruct, compute summary stats.
# Returns a named numeric vector: c(l2_err, peak, apmass).
#
# Arguments:
#   gamma_true : true shear field (list with $gamma1, $gamma2)
#   kappa_true : true convergence (N x N matrix)
#   grid       : from make_grid()
#   N_gal      : total source galaxies in the field
#   sigma_e    : per-component intrinsic shape dispersion
#   m          : multiplicative bias (injected directly)
#   c          : additive bias (injected directly)
#   lambda     : Tikhonov regularization (native R fallback only)
#   use_smpy   : if TRUE, prefer SMPy backend when available
#   seed       : optional RNG seed
# -----------------------------------------------------------------------------
one_replicate <- function(gamma_true, kappa_true, grid,
                          N_gal    = 500,
                          sigma_e  = 0.26,
                          m        = 0,
                          c        = 0,
                          lambda   = 1e-3,
                          use_smpy = TRUE,
                          seed     = NULL) {

  obs <- add_shape_noise_to_grid(gamma_true, grid,
                                 N_gal = N_gal, sigma_e = sigma_e,
                                 m = m, c = c, seed = seed)

  # Dispatch to SMPy if requested and available; otherwise native R KS
  if (use_smpy &&
      exists("ks_reconstruct", mode = "function") &&
      exists("smpy_available", mode = "function") &&
      smpy_available()) {
    kappa_hat <- ks_reconstruct(obs$gamma1_est, obs$gamma2_est, grid,
                                smoothing_sigma = 1.0,
                                verbose = FALSE)$kappa_hat
  } else {
    kappa_hat <- ks_inverse(obs$gamma1_est, obs$gamma2_est, grid, lambda = lambda)
  }

  c(
    l2_err = l2_error(kappa_hat, kappa_true),
    peak   = peak_kappa(kappa_hat),
    apmass = aperture_mass(kappa_hat, grid)
  )
}


# -----------------------------------------------------------------------------
# run_mc()
# Simple Monte Carlo: B independent replicates.
# Returns a B x 3 matrix (l2_err, peak, apmass per replicate).
# -----------------------------------------------------------------------------
run_mc <- function(gamma_true, kappa_true, grid,
                   B        = 500,
                   N_gal    = 500,
                   sigma_e  = 0.26,
                   m        = 0,
                   c        = 0,
                   lambda   = 1e-3,
                   use_smpy = TRUE,
                   seed     = 42) {
  set.seed(seed)
  t(replicate(B, one_replicate(gamma_true, kappa_true, grid,
                                N_gal = N_gal, sigma_e = sigma_e,
                                m = m, c = c, lambda = lambda,
                                use_smpy = use_smpy)))
}


# -----------------------------------------------------------------------------
# run_mc_antithetic()
# Antithetic variable MC: B/2 pairs (noise, -noise) per replicate.
#
# Because the KS reconstruction is linear in the observed shear, negating
# the pixel noise produces a negatively correlated kappa_hat, reducing the
# MC variance of any symmetric summary (e.g., L2 error).
#
# Returns a list:
#   $standard     : B x 3 matrix from B independent draws
#   $antithetic   : B x 3 matrix from B/2 antithetic pairs (same budget)
#   $pct_reduction: estimated variance reduction (%) per summary
# -----------------------------------------------------------------------------
run_mc_antithetic <- function(gamma_true, kappa_true, grid,
                              B        = 500,
                              N_gal    = 500,
                              sigma_e  = 0.26,
                              m        = 0,
                              lambda   = 1e-3,
                              use_smpy = TRUE,
                              seed     = 42) {
  stopifnot(B %% 2 == 0)
  set.seed(seed)

  N       <- grid$N_pix
  n_pairs <- B / 2
  n_pix   <- N_gal / N^2
  sigma_p <- sigma_e / sqrt(n_pix)

  # Pre-generate noise matrices for all B replicates
  noise1_list <- vector("list", n_pairs)
  noise2_list <- vector("list", n_pairs)
  for (b in seq_len(n_pairs)) {
    eps1 <- matrix(rnorm(N^2, 0, sigma_p), N, N)
    eps2 <- matrix(rnorm(N^2, 0, sigma_p), N, N)
    noise1_list[[b]] <- list(e1 = eps1, e2 = eps2)
    noise2_list[[b]] <- list(e1 = -eps1, e2 = -eps2)   # antithetic
  }

  .run_noise <- function(noise) {
    g1_obs <- (1 + m) * gamma_true$gamma1 + noise$e1
    g2_obs <- (1 + m) * gamma_true$gamma2 + noise$e2

    if (use_smpy &&
        exists("ks_reconstruct", mode = "function") &&
        exists("smpy_available", mode = "function") &&
        smpy_available()) {
      kh <- ks_reconstruct(g1_obs, g2_obs, grid,
                           smoothing_sigma = 1.0, verbose = FALSE)$kappa_hat
    } else {
      kh <- ks_inverse(g1_obs, g2_obs, grid, lambda = lambda)
    }

    c(l2_err = l2_error(kh, kappa_true),
      peak   = peak_kappa(kh),
      apmass = aperture_mass(kh, grid))
  }

  res_pos  <- t(sapply(noise1_list, .run_noise))
  res_neg  <- t(sapply(noise2_list, .run_noise))

  # Antithetic estimator: average each pair
  res_anti <- (res_pos + res_neg) / 2

  var_standard   <- apply(res_pos, 2, var) / B
  var_antithetic <- apply(res_anti, 2, var) / n_pairs
  pct_reduction  <- 100 * (var_standard - var_antithetic) / var_standard

  list(
    standard      = rbind(res_pos, res_neg),
    antithetic    = res_anti,
    pct_reduction = pct_reduction
  )
}


# -----------------------------------------------------------------------------
# run_mc_is()
# Importance sampling over multiplicative bias m.
#
# Goal: estimate E_p[epsilon(m)] = int epsilon(m) p(m) dm
# where p(m) ~ N(0, sigma_m^2) (prior on multiplicative bias) and
# epsilon(m) is the reconstruction error at bias level m.
#
# Proposal: q(m) = Uniform(-3*sigma_m, 3*sigma_m)
# Weight:   w(m) = p(m) / q(m)
#
# Self-normalized IS estimator is used for numerical stability.
# For the report: this is the direct analog of HW3 importance sampling.
#
# Returns: list with IS estimate, SE, plain uniform-grid estimate, and raw
#          sample data for plotting.
# -----------------------------------------------------------------------------
run_mc_is <- function(gamma_true, kappa_true, grid,
                      B_per_m  = 100,
                      n_is     = 50,
                      sigma_m  = 0.05,
                      N_gal    = 500,
                      sigma_e  = 0.26,
                      lambda   = 1e-3,
                      use_smpy = TRUE,
                      seed     = 42) {
  set.seed(seed)

  # Sample m values from proposal q = Uniform(-3*sigma_m, 3*sigma_m)
  m_range <- 3 * sigma_m
  m_vals  <- runif(n_is, -m_range, m_range)

  # Importance weights w(m) = p(m) / q(m)
  p_m <- dnorm(m_vals, 0, sigma_m)
  q_m <- dunif(m_vals, -m_range, m_range)
  w   <- p_m / q_m

  # For each m, estimate epsilon(m) via B_per_m MC reps
  eps_hat <- numeric(n_is)
  for (i in seq_along(m_vals)) {
    reps       <- run_mc(gamma_true, kappa_true, grid,
                         B = B_per_m, N_gal = N_gal,
                         sigma_e = sigma_e, m = m_vals[i],
                         lambda = lambda, use_smpy = use_smpy,
                         seed = seed + i)
    eps_hat[i] <- mean(reps[, "l2_err"])
  }

  # Self-normalized IS estimate
  w_norm  <- w / sum(w)
  is_est  <- sum(w_norm * eps_hat)
  is_var  <- var(w * eps_hat) / n_is

  uniform_est <- mean(eps_hat)

  list(
    m_vals      = m_vals,
    eps_hat     = eps_hat,
    weights     = w_norm,
    is_estimate = is_est,
    is_se       = sqrt(is_var),
    uniform_est = uniform_est
  )
}
