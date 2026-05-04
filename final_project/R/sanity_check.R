# =============================================================================
# scripts/sanity_check.R
# Full pipeline smoke test: lensing operators, simulation, KS reconstruction.
#
# Run from the project root:
#   source("scripts/sanity_check.R")
# =============================================================================

.get_proj_root <- function() {
  this_file <- tryCatch(
    normalizePath(sys.frame(1)$ofile),
    error = function(e) {
      tryCatch(
        normalizePath(rstudioapi::getActiveDocumentContext()$path),
        error = function(e2) NULL
      )
    }
  )
  if (!is.null(this_file)) normalizePath(file.path(dirname(this_file), ".."))
  else getwd()
}

proj_root <- .get_proj_root()
setwd(proj_root)
cat(sprintf("Working directory: %s\n", getwd()))

source(file.path("R", "lensing.R"))
source(file.path("R", "simulate.R"))
source(file.path("R", "ks_smpy.R"))
source(file.path("R", "mc.R"))

set.seed(42)

# ------ 1. Grid and true kappa ------
grid       <- make_grid(N_pix = 64, pix = 0.05)
kappa_true <- make_true_kappa(grid, kappa0 = 0.3, sigma_l = 0.5)

cat(sprintf("Grid: %d x %d pixels, %.2f arcmin field\n",
            grid$N_pix, grid$N_pix, grid$L))
cat(sprintf("True kappa: peak = %.4f, mean = %.6f\n",
            max(kappa_true), mean(kappa_true)))

# ------ 2. Forward operator ------
gamma_true <- ks_forward(kappa_true, grid)
cat(sprintf("True gamma1: [%.4f, %.4f]\n",
            min(gamma_true$gamma1), max(gamma_true$gamma1)))
cat(sprintf("True gamma2: [%.4f, %.4f]\n",
            min(gamma_true$gamma2), max(gamma_true$gamma2)))

# ------ 3. Noiseless KS inversion (native R) ------
kappa_noiseless <- ks_inverse(gamma_true$gamma1, gamma_true$gamma2, grid, lambda = 0)
cat(sprintf("Noiseless KS L2 error (native R): %.2e (should be ~0)\n",
            l2_error(kappa_noiseless, kappa_true)))

# ------ 4. Simulate shape noise using proper simulate.R ------
obs <- add_shape_noise_to_grid(gamma_true, grid, N_gal = 500000, sigma_e = 0.26, seed = 42)
cat(sprintf("Shape noise sigma_pix: %.4f  (n_gal/pix: %.2f)\n",
            obs$sigma_pix, obs$n_gal_pix))

# ------ 5. KS reconstruction -- try SMPy, fall back to native R ------
cat("\n--- KS Reconstruction ---\n")
ks_out <- ks_reconstruct(obs$gamma1_est, obs$gamma2_est, grid,
                         smoothing_sigma = 1.0, verbose = TRUE)
kappa_hat <- ks_out$kappa_hat
cat(sprintf("Backend used: %s\n", ks_out$backend))
cat(sprintf("KS L2 error (N_gal=500000): %.4f\n",
            l2_error(kappa_hat, kappa_true)))
cat(sprintf("Peak kappa_hat: %.4f  (true peak: %.4f)\n",
            peak_kappa(kappa_hat), peak_kappa(kappa_true)))

# ------ 6. Backend comparison (if SMPy is available) ------
if (smpy_available()) {
  cat("\n--- Backend Comparison (SMPy vs native R, no smoothing) ---\n")
  cmp <- ks_compare_backends(gamma_true$gamma1, gamma_true$gamma2, grid,
                              smoothing_sigma = 0.0)
  if (cmp$rel_diff < 1e-3) {
    cat("  PASS: backends agree to < 0.1%\n")
  } else {
    cat("  WARN: backends differ by", round(cmp$rel_diff * 100, 2), "%\n")
  }
} else {
  cat("\nSMPy not found in Python environment.\n")
  cat("To use SMPy backend: pip install smpy (or pip install git+https://github.com/CosmoStat/smpy)\n")
  cat("Then point reticulate to the correct Python: use_python('/path/to/python')\n")
}

# ------ 7. MC smoke test ------
cat("\nRunning quick MC (B = 20)...\n")
mc_quick <- run_mc(gamma_true, kappa_true, grid,
                   B = 20, N_gal = 500000, lambda = 1e-3)
cat(sprintf("MC L2 error: mean = %.4f, sd = %.4f\n",
            mean(mc_quick[, "l2_err"]), sd(mc_quick[, "l2_err"])))
cat(sprintf("MC peak kappa: mean = %.4f  (true = %.4f)\n",
            mean(mc_quick[, "peak"]), peak_kappa(kappa_true)))

# ------ 8. Visualization ------
par(mfrow = c(2, 2), mar = c(2, 2, 2, 1))
image(grid$x, grid$y, kappa_true,
      col = hcl.colors(64, "YlOrRd", rev = TRUE),
      main = "True kappa")
image(grid$x, grid$y, gamma_true$gamma1,
      col = hcl.colors(64, "Blue-Red 3"),
      main = "True gamma1")
image(grid$x, grid$y, kappa_hat,
      col = hcl.colors(64, "YlOrRd", rev = TRUE),
      main = sprintf("KS recon [%s] L2=%.3f",
                     ks_out$backend, l2_error(kappa_hat, kappa_true)))
image(grid$x, grid$y, kappa_hat - kappa_true,
      col = hcl.colors(64, "Blue-Red 3"),
      main = "Residual (kappa_hat - kappa_true)")

cat("\nSanity check complete. Next: run scripts/run_all.R\n")
