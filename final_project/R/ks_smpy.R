# =============================================================================
# R/ks_smpy.R  (patched: reticulate loaded conditionally)
# Kaiser-Squires mass reconstruction using SMPy via reticulate.
# Falls back to native R FFT if SMPy / reticulate is not available.
# =============================================================================

# Load reticulate only if installed — avoids hard crash on HPC where SMPy
# is not present but the native R fallback works fine.
.reticulate_available <- requireNamespace("reticulate", quietly = TRUE)
if (.reticulate_available) {
  suppressPackageStartupMessages(library(reticulate))
}

smpy_available <- function() {
  if (!.reticulate_available) return(FALSE)
  tryCatch({
    reticulate::import("smpy.config", convert = FALSE)
    TRUE
  }, error = function(e) FALSE)
}

.ks_smpy <- function(gamma1, gamma2, smoothing_sigma = 1.0) {
  smpy_config <- reticulate::import("smpy.config", convert = FALSE)
  smpy_ks     <- reticulate::import(
    "smpy.mapping_methods.kaiser_squires.kaiser_squires", convert = FALSE)
  np          <- reticulate::import("numpy", convert = FALSE)

  g1_np <- np$array(gamma1, dtype = "float64")
  g2_np <- np$array(gamma2, dtype = "float64")

  cfg <- smpy_config$Config$from_defaults("kaiser_squires")$to_dict()
  cfg[["methods"]][["kaiser_squires"]][["smoothing"]][["sigma"]] <- smoothing_sigma

  mapper    <- smpy_ks$KaiserSquiresMapper(cfg)
  result    <- mapper$create_maps(g1_np, g2_np)
  kappa_raw <- as.matrix(reticulate::py_to_r(result[[1]]))

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

.ks_native_r <- function(gamma1, gamma2, grid) {
  if (!exists("ks_inverse", mode = "function")) {
    stop("ks_inverse() not found. Source R/lensing.R before R/ks_smpy.R.")
  }
  ks_inverse(gamma1, gamma2, grid, lambda = 0)
}

ks_reconstruct <- function(gamma1, gamma2, grid,
                           smoothing_sigma = 1.0,
                           force_native    = FALSE,
                           verbose         = TRUE) {
  if (!force_native && smpy_available()) {
    if (verbose) message("ks_reconstruct: using SMPy backend")
    kappa_hat <- .ks_smpy(gamma1, gamma2, smoothing_sigma = smoothing_sigma)
    return(list(kappa_hat = kappa_hat, backend = "smpy"))
  }

  if (verbose) {
    if (force_native) {
      message("ks_reconstruct: using native R FFT backend (forced)")
    } else {
      message("ks_reconstruct: SMPy/reticulate not available, using native R FFT")
    }
  }

  kappa_hat <- .ks_native_r(gamma1, gamma2, grid)
  list(kappa_hat = kappa_hat, backend = "native_r")
}

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
  cat(sprintf("  max |SMPy - native R|: %.4e\n", max_diff))
  cat(sprintf("  relative max diff:     %.4e\n", rel_diff))
  cat(sprintf("  pixel correlation:     %.6f\n", corr_val))
  list(smpy=k_smpy, native=k_native,
       max_diff=max_diff, rel_diff=rel_diff, corr=corr_val)
}