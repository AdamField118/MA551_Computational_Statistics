# =============================================================================
# scripts/sanity_check.R
# Verifies the full pipeline works end-to-end before running the full analysis.
# Run with: source("scripts/sanity_check.R") from the project root, OR
# open this file in RStudio and click the Source button.
# =============================================================================

# --- Robust project root detection ------------------------------------------
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
  if (!is.null(this_file)) {
    normalizePath(file.path(dirname(this_file), ".."))
  } else {
    getwd()
  }
}

proj_root <- .get_proj_root()
setwd(proj_root)
cat(sprintf("Working directory: %s\n", getwd()))

source(file.path("R", "lensing.R"))
source(file.path("R", "simulate.R"))
source(file.path("R", "mc.R"))

set.seed(42)

# ------ 1. Build grid and true kappa field ------
grid      <- make_grid(N_pix = 64, pix = 0.05)
kappa_true <- make_true_kappa(grid, kappa0 = 0.3, sigma_l = 0.5)

cat(sprintf("Grid: %d x %d pixels, %.2f arcmin field\n",
            grid$N_pix, grid$N_pix, grid$L))
cat(sprintf("True kappa: peak = %.4f, mean = %.6f\n",
            max(kappa_true), mean(kappa_true)))

# ------ 2. Forward operator: kappa -> gamma ------
gamma_true <- ks_forward(kappa_true, grid)

cat(sprintf("True gamma1: range [%.4f, %.4f]\n",
            min(gamma_true$gamma1), max(gamma_true$gamma1)))
cat(sprintf("True gamma2: range [%.4f, %.4f]\n",
            min(gamma_true$gamma2), max(gamma_true$gamma2)))

# ------ 3. Noiseless KS reconstruction (sanity: should recover kappa_true) ------
kappa_hat_noiseless <- ks_inverse(gamma_true$gamma1, gamma_true$gamma2, grid, lambda = 0)
cat(sprintf("Noiseless KS recovery L2 error: %.6f (should be near 0)\n",
            l2_error(kappa_hat_noiseless, kappa_true)))

# ------ 4. One noisy replicate ------
obs <- add_shape_noise_to_grid(gamma_true, grid, N_gal = 500, sigma_e = 0.26)
kappa_hat <- ks_inverse(obs$gamma1_est, obs$gamma2_est, grid, lambda = 1e-3)
cat(sprintf("Noisy KS recovery L2 error (N_gal=500): %.4f\n",
            l2_error(kappa_hat, kappa_true)))
cat(sprintf("Peak kappa_hat: %.4f  (true peak: %.4f)\n",
            peak_kappa(kappa_hat), peak_kappa(kappa_true)))

# ------ 5. Visualization ------
par(mfrow = c(2, 2), mar = c(2, 2, 2, 1))

image(grid$x, grid$y, kappa_true,
      col = hcl.colors(64, "YlOrRd", rev = TRUE),
      main = "True kappa", xlab = "x (arcmin)", ylab = "y (arcmin)")

image(grid$x, grid$y, gamma_true$gamma1,
      col = hcl.colors(64, "Blue-Red 3"),
      main = "True gamma1", xlab = "x (arcmin)", ylab = "y (arcmin)")

image(grid$x, grid$y, kappa_hat,
      col = hcl.colors(64, "YlOrRd", rev = TRUE),
      main = sprintf("KS reconstruction (N_gal=500, L2=%.3f)",
                     l2_error(kappa_hat, kappa_true)),
      xlab = "x (arcmin)", ylab = "y (arcmin)")

image(grid$x, grid$y, kappa_hat - kappa_true,
      col = hcl.colors(64, "Blue-Red 3"),
      main = "Residual (kappa_hat - kappa_true)",
      xlab = "x (arcmin)", ylab = "y (arcmin)")

# ------ 6. Quick MC run ------
cat("\nRunning quick MC (B=20) to verify no errors...\n")
mc_quick <- run_mc(gamma_true, kappa_true, grid,
                   B = 20, N_gal = 500, lambda = 1e-3)
cat(sprintf("MC L2 error: mean = %.4f, sd = %.4f\n",
            mean(mc_quick[, "l2_err"]), sd(mc_quick[, "l2_err"])))
cat(sprintf("MC peak kappa: mean = %.4f  (true = %.4f)\n",
            mean(mc_quick[, "peak"]), peak_kappa(kappa_true)))

cat("\nSanity check passed. Pipeline is working.\n")
cat("Next: run scripts/run_all.R for the full analysis.\n")