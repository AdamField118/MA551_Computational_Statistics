# =============================================================================
# R/run_all.R  (updated for HPC run with real metacal bias)
# =============================================================================
# Changes from the original:
#   * Reads results/metacal_bias.json produced by metacal_pipeline.py
#   * Uses measured m_hat, c_hat instead of injected m=0, c=0
#   * N_GAL: 5000 -> 50000 (shows IS benefit; matches HPC metacal count)
#   * B_MC:  500  -> 2000  (tests analytic NCP; HPC can handle this)
#   * Adds Section 9: NCP validation plot comparing analytic vs MC power
# =============================================================================

QUICK   <- Sys.getenv("QUICK", "FALSE") == "TRUE"   # env var override
ON_HPC  <- Sys.getenv("SLURM_JOB_ID", "") != ""     # detect Turing

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
cat(sprintf("Working directory: %s\n\n", getwd()))

source(file.path("R", "lensing.R"))
source(file.path("R", "simulate.R"))
source(file.path("R", "ks_smpy.R"))
source(file.path("R", "mc.R"))
source(file.path("R", "bootstrap.R"))
source(file.path("R", "power.R"))

set.seed(42)

# ---- Tunable parameters ----------------------------------------------------
# Scaling up: N_GAL 5000 -> 50000 (shows IS gain), B_MC 500 -> 2000 (NCP test)
N_GAL   <- if (QUICK)  5000L  else 50000L
B_MC    <- if (QUICK)  500L   else 2000L
B_BOOT  <- if (QUICK)  500L   else 1000L
N_OUTER <- if (QUICK)  50L    else 200L
LAMBDA  <- 1e-3
SIGMA_E <- 0.26

# ---- 0. Load metacal bias estimates ----------------------------------------
cat("=== 0. Loading metacalibration bias estimates ===\n")
bias_file <- file.path("results", "metacal_bias.json")

if (file.exists(bias_file)) {
  # jsonlite or RJSONIO
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    metacal <- jsonlite::fromJSON(bias_file)
  } else {
    metacal <- RJSONIO::fromJSON(bias_file)
  }
  M_HAT  <- metacal$m_hat
  C_HAT  <- metacal$c_hat
  M_SE   <- metacal$m_hat_se
  C_SE   <- metacal$c_hat_se
  cat(sprintf("  m_hat = %+.5f  (SE = %.5f)  [measured by metacalibration]\n",
              M_HAT, M_SE))
  cat(sprintf("  c_hat = %+.5f  (SE = %.5f)\n", C_HAT, C_SE))
  cat(sprintf("  n_gal_metacal = %d  success_rate = %.1f%%\n",
              metacal$n_success,
              metacal$success_rate * 100))
  USE_REAL_BIAS <- TRUE
} else {
  cat("  metacal_bias.json not found. Using m=0, c=0 (run metacal_pipeline.py first).\n")
  M_HAT         <- 0.0
  C_HAT         <- 0.0
  M_SE          <- NA_real_
  C_SE          <- NA_real_
  USE_REAL_BIAS <- FALSE
}

# ---- 1. Forward model setup ------------------------------------------------
cat("\n=== 1. Lensing forward model ===\n")
grid       <- make_grid(N_pix = 32L, pix = 0.1)
kappa_true <- make_true_kappa(grid, kappa0 = 0.3, sigma_l = 0.5)
gamma_true <- ks_forward(kappa_true, grid)
apmass_stat <- make_apmass_fn(grid)
true_peak   <- peak_kappa(kappa_true)
true_apm    <- apmass_stat(kappa_true)
cat(sprintf("Grid: %dx%d  field=%.2f arcmin  pix=%.3f arcmin\n",
            grid$N_pix, grid$N_pix, grid$L, grid$pix))
cat(sprintf("Peak kappa: %.4f  aperture_mass: %.4f\n", true_peak, true_apm))
cat(sprintf("N_gal=%d  sigma_pix=%.4f  max|gamma|=%.4f\n",
            N_GAL,
            SIGMA_E / sqrt(N_GAL / grid$N_pix^2),
            max(abs(gamma_true$gamma1))))

# ---- 2. Simple MC (at measured bias) ---------------------------------------
cat("\n=== 2. Simple Monte Carlo (measured bias m, c) ===\n")
mc_base <- run_mc(gamma_true, kappa_true, grid,
                  B = B_MC, N_gal = N_GAL,
                  m = M_HAT, c = C_HAT,
                  lambda = LAMBDA, seed = 42L)
cat(sprintf("L2 error:      mean=%.4f  sd=%.4f\n",
            mean(mc_base[,"l2_err"]), sd(mc_base[,"l2_err"])))
cat(sprintf("Peak kappa:    mean=%.4f  sd=%.4f  (true=%.4f)\n",
            mean(mc_base[,"peak"]), sd(mc_base[,"peak"]), true_peak))
cat(sprintf("Aperture mass: mean=%.4f  sd=%.4f  (true=%.4f)\n",
            mean(mc_base[,"apmass"]), sd(mc_base[,"apmass"]), true_apm))

# Also run unbiased baseline for comparison
cat("\n--- Unbiased baseline (m=0, c=0) for IS reference ---\n")
mc_unbias <- run_mc(gamma_true, kappa_true, grid,
                    B = B_MC, N_gal = N_GAL,
                    m = 0, c = 0,
                    lambda = LAMBDA, seed = 99L)
cat(sprintf("L2 error (m=0): mean=%.4f  sd=%.4f\n",
            mean(mc_unbias[,"l2_err"]), sd(mc_unbias[,"l2_err"])))

# ---- 3. Antithetic variables -----------------------------------------------
cat("\n=== 3. Antithetic variable MC ===\n")
mc_anti <- run_mc_antithetic(gamma_true, kappa_true, grid,
                             B = B_MC, N_gal = N_GAL,
                             m = M_HAT,
                             lambda = LAMBDA, seed = 42L)
cat("Variance reduction (%):\n")
for (nm in names(mc_anti$pct_reduction))
  cat(sprintf("  %-10s: %+.1f%%\n", nm, mc_anti$pct_reduction[nm]))

# ---- 4. Importance sampling (N_GAL=50000 -> bigger IS benefit) -------------
cat("\n=== 4. Importance sampling over multiplicative bias m ===\n")
cat("(N_gal=50000 should show IS benefit over uniform grid)\n")
mc_is <- run_mc_is(gamma_true, kappa_true, grid,
                   B_per_m = if (QUICK) 50L  else 200L,
                   n_is    = if (QUICK) 20L  else 100L,
                   sigma_m = M_SE,          # use measured SE as prior width
                   N_gal   = N_GAL,
                   lambda  = LAMBDA, seed = 42L)
cat(sprintf("IS estimate of E_p[L2 error]: %.4f  (SE: %.4f)\n",
            mc_is$is_estimate, mc_is$is_se))
cat(sprintf("Uniform grid estimate:        %.4f\n", mc_is$uniform_est))
cat(sprintf("Effective sample size (ESS):  %.1f / %d\n",
            1 / sum((mc_is$weights / sum(mc_is$weights))^2),
            length(mc_is$weights)))

# ---- 5. Bootstrap CIs on aperture mass -------------------------------------
cat("\n=== 5. Bootstrap CIs on aperture mass ===\n")
boot_apm  <- boot_ks(gamma_true, grid,
                     stat_fn = apmass_stat,
                     B       = B_BOOT,
                     N_gal   = N_GAL,
                     m       = M_HAT,
                     c_bias  = C_HAT,
                     lambda  = LAMBDA,
                     seed    = 43L)
bs_apm    <- boot_bias_se(boot_apm)
ci_apm    <- bca_ci(boot_apm, gamma_true, grid,
                    stat_fn = apmass_stat, lambda = LAMBDA, N_gal = N_GAL,
                    m = M_HAT, c_bias = C_HAT)
cat(sprintf("aperture_mass: obs=%.4f  boot mean=%.4f  bias=%.4f  SE=%.4f\n",
            bs_apm$t_obs, bs_apm$mean, bs_apm$bias, bs_apm$se))
cat(sprintf("95%% Percentile CI: [%.4f, %.4f]\n",
            ci_apm$percentile[1], ci_apm$percentile[2]))
cat(sprintf("95%% BCa CI:        [%.4f, %.4f]  (z0=%.3f  a=%.4f)\n",
            ci_apm$bca[1], ci_apm$bca[2], ci_apm$z0, ci_apm$a))

truth_apm_recon <- apmass_stat(ks_inverse(gamma_true$gamma1, gamma_true$gamma2,
                                           grid, lambda = LAMBDA))
cat(sprintf("Noiseless KS aperture mass (target): %.4f\n", truth_apm_recon))

# ---- 6. Coverage study -----------------------------------------------------
cat("\n=== 6. Bootstrap coverage study ===\n")
cov_study <- boot_coverage(kappa_true, gamma_true, grid,
                           stat_fn  = apmass_stat,
                           truth    = truth_apm_recon,
                           n_outer  = N_OUTER,
                           B        = B_BOOT,
                           N_gal    = N_GAL,
                           m        = M_HAT,
                           c_bias   = C_HAT,
                           lambda   = LAMBDA,
                           seed     = 100L)

# ---- 7. Power analysis (real delta = m_hat * T_ap^KS) ----------------------
cat("\n=== 7. Power analysis: detecting |m| = 0.01 at 5-sigma ===\n")

# If we have the real bias, the detectable effect is m_hat-driven, but the
# power question is generic: how many galaxies to detect ANY |m|=0.01?
delta_analytic <- 0.01 * truth_apm_recon
sd_apmass_ref  <- sd(mc_unbias[, "apmass"])   # SD at N_GAL (unbiased)
cat(sprintf("Analytic delta (m=0.01): %.5f\n", delta_analytic))
cat(sprintf("SD(apmass) at N_gal=%d: %.5f\n\n", N_GAL, sd_apmass_ref))

unpaired_power_analytic <- function(N_gal, B, delta, sd_ref, N_gal_ref,
                                     alpha = 2.87e-7) {
  sd_ng  <- sd_ref * sqrt(N_gal_ref / N_gal)
  se     <- sqrt(2) * sd_ng / sqrt(B)
  ncp    <- delta / se
  t_crit <- qnorm(1 - alpha / 2)
  list(power  = pnorm(ncp - t_crit) + pnorm(-ncp - t_crit),
       ncp    = ncp,
       sd_ng  = sd_ng)
}

ngal_grid <- c(1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000)
B_power   <- B_MC   # use the scaled-up B

cat(sprintf("B = %d replicates per condition\n\n", B_power))
cat(sprintf("%-10s  %-8s  %-8s  %-8s\n", "N_gal", "power", "NCP", "SD(T)"))
pow_rows <- vector("list", length(ngal_grid))
for (i in seq_along(ngal_grid)) {
  ng  <- ngal_grid[i]
  res <- unpaired_power_analytic(ng, B_power, delta_analytic,
                                  sd_apmass_ref, N_GAL)
  pow_rows[[i]] <- c(N_gal = ng, power = res$power, ncp = res$ncp,
                     sd_ng = res$sd_ng)
  cat(sprintf("%-10d  %-8.4f  %-8.2f  %-8.5f\n",
              ng, res$power, res$ncp, res$sd_ng))
}
pow_curve <- as.data.frame(do.call(rbind, pow_rows))

# ---- 8. NCP VALIDATION: compare analytic vs MC (the new validation) --------
cat("\n=== 8. NCP validation: analytic vs MC (N_GAL=%d, B=%d) ===\n",
    N_GAL, B_power)
cat("This directly tests whether the analytic NCP formula is accurate\n",
    "at the higher B and N_GAL used in this scaled-up study.\n\n")

set.seed(201L)
N         <- grid$N_pix
sigma_pix <- SIGMA_E / sqrt(N_GAL / N^2)
g1_null   <- gamma_true$gamma1; g2_null <- gamma_true$gamma2
g1_alt    <- (1 + 0.01) * g1_null; g2_alt <- (1 + 0.01) * g2_null

# Run MC with B_power replicates (uses scaled-up B to get tighter NCP estimate)
cat(sprintf("Running %d null replicates ...\n", B_power))
set.seed(201L)
T0 <- replicate(B_power, {
  k0 <- ks_inverse(g1_null + matrix(rnorm(N^2, 0, sigma_pix), N, N),
                   g2_null + matrix(rnorm(N^2, 0, sigma_pix), N, N),
                   grid, LAMBDA)
  apmass_stat(k0)
})
cat(sprintf("Running %d alternative replicates (m=0.01) ...\n", B_power))
set.seed(202L)
Tm <- replicate(B_power, {
  km <- ks_inverse(g1_alt + matrix(rnorm(N^2, 0, sigma_pix), N, N),
                   g2_alt + matrix(rnorm(N^2, 0, sigma_pix), N, N),
                   grid, LAMBDA)
  apmass_stat(km)
})

delta_mc <- mean(Tm) - mean(T0)
se_mc    <- sqrt(var(Tm)/B_power + var(T0)/B_power)
ncp_mc   <- delta_mc / se_mc
t_crit   <- qt(1 - 2.87e-7/2, df = 2*B_power - 2)
pow_mc   <- 1 - pt(t_crit, df = 2*B_power-2, ncp = ncp_mc) +
              pt(-t_crit, df = 2*B_power-2, ncp = ncp_mc)

analytic_at_ngal <- unpaired_power_analytic(N_GAL, B_power,
                                             delta_analytic,
                                             sd_apmass_ref, N_GAL)

cat(sprintf("\n              %-12s  %-12s\n", "MC", "Analytic"))
cat(sprintf("  delta:      %-12.5f  %-12.5f\n",
            delta_mc, delta_analytic))
cat(sprintf("  NCP:        %-12.2f  %-12.2f\n",
            ncp_mc, analytic_at_ngal$ncp))
cat(sprintf("  Power:      %-12.4f  %-12.4f\n",
            pow_mc, analytic_at_ngal$power))
cat(sprintf("\n  NCP relative error: %.1f%%\n",
            100 * abs(ncp_mc - analytic_at_ngal$ncp) / analytic_at_ngal$ncp))
cat("  (Should be < 5% at B=2000 vs ~15% at B=500)\n")

ncp_validation <- list(
  ncp_mc       = ncp_mc,
  ncp_analytic = analytic_at_ngal$ncp,
  delta_mc     = delta_mc,
  delta_analytic = delta_analytic,
  power_mc     = pow_mc,
  power_analytic = analytic_at_ngal$power,
  B            = B_power,
  N_GAL        = N_GAL
)

# ---- 9. Minimum N_gal for 80% power at B_power ----------------------------
find_ngal_80 <- function(delta, sd_ref, N_ref, B, alpha = 2.87e-7,
                          target = 0.80) {
  f <- function(ng) {
    unpaired_power_analytic(ng, B, delta, sd_ref, N_ref, alpha)$power - target
  }
  if (f(1e3) >= 0) return(1e3)
  if (f(1e7) <  0) return(NA)
  as.integer(uniroot(f, c(1e3, 1e7))$root)
}
ng_80 <- find_ngal_80(delta_analytic, sd_apmass_ref, N_GAL, B_power)
cat(sprintf("\nMinimum N_gal for 80%% power (B=%d, m=0.01, 5-sigma): %s\n",
            B_power,
            if (is.na(ng_80)) "N/A" else format(ng_80, big.mark = ",")))

# ---- 10. Save results -------------------------------------------------------
cat("\n=== 10. Saving results ===\n")
dir.create("results", showWarnings = FALSE)

saveRDS(list(
  mc_base          = mc_base,
  mc_unbias        = mc_unbias,
  mc_anti          = mc_anti,
  mc_is            = mc_is,
  boot_apm         = boot_apm,
  ci_apm           = ci_apm,
  cov_study        = cov_study,
  pow_curve        = pow_curve,
  ncp_validation   = ncp_validation,
  truth_apm_recon  = truth_apm_recon,
  ng_80            = ng_80,
  bias             = list(m = M_HAT, c = C_HAT, m_se = M_SE, c_se = C_SE,
                          use_real_bias = USE_REAL_BIAS),
  params           = list(N_GAL = N_GAL, B_MC = B_MC, B_BOOT = B_BOOT,
                          N_OUTER = N_OUTER, LAMBDA = LAMBDA,
                          grid_pix = grid$pix, grid_N = grid$N_pix)
), file.path("results", "all_results.rds"))

cat("Results saved to results/all_results.rds\n")
cat("Render final_report.Rmd to produce the PDF.\n")
