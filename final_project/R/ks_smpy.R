# =============================================================================
# R/ks_smpy.R
# Kaiser-Squires mass reconstruction using SMPy via reticulate.
#
# SMPy (https://github.com/CosmoStat/smpy) is the production KS implementation
# used in weak lensing pipelines (e.g. SuperBIT).  This wrapper calls it from
# R via reticulate, with an automatic fallback to a native R FFT implementation
# if SMPy is not available in the Python environment.
#
# Primary function:
#   ks_reconstruct()   -- dispatches to SMPy or native R
#
# Helper:
#   .ks_native_r()     -- pure-R FFT KS (fallback, for comparison)
#   .ks_smpy()         -- SMPy wrapper (preferred)
#   smpy_available()   -- returns TRUE if SMPy can be imported
#
# Usage:
#   source("R/ks_smpy.R")
#   kappa_hat <- ks_reconstruct(gamma1_obs, gamma2_obs, grid)
# =============================================================================

suppressPackageStartupMessages(library(reticulate))


# -----------------------------------------------------------------------------
# smpy_available()
# Check whether SMPy is importable in the active Python environment.
# -----------------------------------------------------------------------------
smpy_available <- function() {
  tryCatch({
    reticulate::import("smpy.config", convert = FALSE)
    TRUE
  }, error = function(e) FALSE)
}


# -----------------------------------------------------------------------------
# .ks_smpy()
# Internal: Kaiser-Squires via SMPy's KaiserSquiresMapper.
#
# Arguments:
#   gamma1, gamma2 : N x N observed shear matrices (biased + noisy)
#   smoothing_sigma: Gaussian smoothing scale (pixels) applied by SMPy
#
# Returns: N x N convergence map (E-mode, DC-corrected)
# -----------------------------------------------------------------------------
.ks_smpy <- function(gamma1, gamma2, smoothing_sigma = 1.0) {
  smpy_config <- reticulate::import("smpy.config", convert = FALSE)
  smpy_ks     <- reticulate::import(
    "smpy.mapping_methods.kaiser_squires.kaiser_squires", convert = FALSE)
  np          <- reticulate::import("numpy", convert = FALSE)

  # R matrices -> numpy arrays (SMPy expects [row, col] = [y, x] convention)
  g1_np <- np$array(gamma1, dtype = "float64")
  g2_np <- np$array(gamma2, dtype = "float64")

  # Build SMPy config dict and set smoothing
  cfg <- smpy_config$Config$from_defaults("kaiser_squires")$to_dict()
  cfg[["methods"]][["kaiser_squires"]][["smoothing"]][["sigma"]] <- smoothing_sigma

  # Run KS reconstruction; create_maps returns (kappa_E, kappa_B)
  mapper    <- smpy_ks$KaiserSquiresMapper(cfg)
  result    <- mapper$create_maps(g1_np, g2_np)
  kappa_raw <- as.matrix(reticulate::py_to_r(result[[1]]))

  # DC (sky) correction: subtract mean of outer 15% border pixels
  N <- nrow(kappa_raw)
  m <- max(1L, as.integer(ceiling(0.15 * N)))
  sky_vals <- c(
    kappa_raw[seq_len(m), ],
    kappa_raw[seq(N - m + 1L, N), ],
    kappa_raw[, seq_len(m)],
    kappa_raw[, seq(N - m + 1L, N)]
  )
  kappa_raw - mean(sky_vals, na.rm = TRUE)
}


# -----------------------------------------------------------------------------
# .ks_native_r()
# Internal: Kaiser-Squires via native R FFT (used as fallback).
# Implements the standard KS inversion in Fourier space:
#   kappa_hat(k) = Re[ conj(D(k)) * gamma_hat(k) ]
# where D(k) = (k1^2 - k2^2 + 2i*k1*k2) / (k1^2 + k2^2).
#
# This is mathematically equivalent to SMPy's implementation; the difference
# is that SMPy applies configurable smoothing and has been tested against
# numerous weak lensing datasets.
# -----------------------------------------------------------------------------
.ks_native_r <- function(gamma1, gamma2, grid) {
  # Forward-source this from lensing.R if not already loaded
  if (!exists("ks_inverse", mode = "function")) {
    stop("ks_inverse() not found. Source R/lensing.R before R/ks_smpy.R.")
  }
  ks_inverse(gamma1, gamma2, grid, lambda = 0)
}


# -----------------------------------------------------------------------------
# ks_reconstruct()
#
# Unified Kaiser-Squires interface.  Prefers SMPy; falls back to native R.
#
# Arguments:
#   gamma1, gamma2   : N x N observed shear matrices
#   grid             : from make_grid() (used only for native-R fallback)
#   smoothing_sigma  : Gaussian smoothing scale for SMPy (pixels; 0 = none)
#   force_native     : if TRUE, always use native R implementation
#   verbose          : if TRUE, print which backend is used
#
# Returns: list with
#   $kappa_hat  : N x N reconstruction
#   $backend    : "smpy" or "native_r"
# -----------------------------------------------------------------------------
ks_reconstruct <- function(gamma1, gamma2, grid,
                           smoothing_sigma = 1.0,
                           force_native    = FALSE,
                           verbose         = TRUE) {
  if (!force_native && smpy_available()) {
    if (verbose) message("ks_reconstruct: using SMPy backend")
    kappa_hat <- .ks_smpy(gamma1, gamma2, smoothing_sigma = smoothing_sigma)
    return(list(kappa_hat = kappa_hat, backend = "smpy"))
  }

  if (!force_native && verbose) {
    message("ks_reconstruct: SMPy not found, falling back to native R FFT")
  } else if (force_native && verbose) {
    message("ks_reconstruct: using native R FFT backend (forced)")
  }

  kappa_hat <- .ks_native_r(gamma1, gamma2, grid)
  list(kappa_hat = kappa_hat, backend = "native_r")
}


# -----------------------------------------------------------------------------
# ks_compare_backends()
#
# Diagnostic: run both backends on the same input and report the
# pixel-wise difference.  Useful for validating the native R fallback.
#
# Returns: list with $smpy, $native, $max_diff, $rel_diff, $corr
# -----------------------------------------------------------------------------
ks_compare_backends <- function(gamma1, gamma2, grid,
                                smoothing_sigma = 0.0) {
  if (!smpy_available()) {
    stop("SMPy not available; cannot run comparison.")
  }

  k_smpy   <- .ks_smpy(gamma1, gamma2, smoothing_sigma = smoothing_sigma)
  k_native <- .ks_native_r(gamma1, gamma2, grid)

  diff     <- k_smpy - k_native
  max_diff <- max(abs(diff))
  rel_diff <- max_diff / (max(abs(k_smpy)) + 1e-12)
  corr_val <- cor(as.vector(k_smpy), as.vector(k_native))

  cat(sprintf("Backend comparison (smoothing_sigma = %.1f):\n", smoothing_sigma))
  cat(sprintf("  max |SMPy - native R|: %.4e\n",   max_diff))
  cat(sprintf("  relative max diff:     %.4e\n",   rel_diff))
  cat(sprintf("  pixel correlation:     %.6f\n",   corr_val))

  list(
    smpy     = k_smpy,
    native   = k_native,
    max_diff = max_diff,
    rel_diff = rel_diff,
    corr     = corr_val
  )
}
