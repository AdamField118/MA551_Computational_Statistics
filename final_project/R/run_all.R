# =============================================================================
# scripts/run_all.R
# Full analysis pipeline for MA 551 final project.
# Sources all R modules and produces all results needed for the report.
#
# Runtime estimate (N_gal=5000, B=500, n_outer=200): ~15-30 min on a laptop.
# For a quick smoke test, set QUICK <- TRUE (reduces B and n_outer).
#
# Run from project root:
#   source("scripts/run_all.R")
# =============================================================================

QUICK <- FALSE   # set TRUE for fast smoke test

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
N_GAL   <- if (QUICK) 5000L  else 5000L
B_MC    <- if (QUICK) 100L   else 500L
B_BOOT  <- if (QUICK) 200L   else 500L
N_OUTER <- if (QUICK) 50L    else 200L
LAMBDA  <- 1e-3
SIGMA_E <- 0.26

# ---- 1. Forward model setup ------------------------------------------------
cat("=== 1. Lensing forward model ===\n")
# 32x32 grid: 1024 pixels, 5000 gal -> ~4.9 gal/pix -> sigma_pix ~ 0.12
# Signal (shear peak) ~ 0.1 -> SNR per pixel ~ 0.8, workable with lambda
grid       <- make_grid(N_pix = 32L, pix = 0.1)
kappa_true <- make_true_kappa(grid, kappa0 = 0.3, sigma_l = 0.5)
gamma_true <- ks_forward(kappa_true, grid)
apmass_stat <- make_apmass_fn(grid)   # closure -- used throughout
true_peak   <- peak_kappa(kappa_true)
true_apm    <- apmass_stat(kappa_true)
cat(sprintf("Grid: %dx%d  field=%.2f arcmin  pix=%.3f arcmin\n",
            grid$N_pix, grid$N_pix, grid$L, grid$pix))
cat(sprintf("Peak kappa: %.4f  aperture_mass: %.4f\n", true_peak, true_apm))
cat(sprintf("N_gal/pixel: %.1f  sigma_pix: %.4f  max|gamma|: %.4f\n",
            N_GAL / grid$N_pix^2,
            SIGMA_E / sqrt(N_GAL / grid$N_pix^2),
            max(abs(gamma_true$gamma1))))

# ---- 2. Simple MC (baseline) -----------------------------------------------
cat("\n=== 2. Simple Monte Carlo (m=0) ===\n")
mc_base <- run_mc(gamma_true, kappa_true, grid,
                  B = B_MC, N_gal = N_GAL, lambda = LAMBDA, seed = 42L)
cat(sprintf("L2 error:     mean=%.4f  sd=%.4f\n",
            mean(mc_base[,"l2_err"]), sd(mc_base[,"l2_err"])))
cat(sprintf("Peak kappa:   mean=%.4f  sd=%.4f  (true=%.4f)\n",
            mean(mc_base[,"peak"]), sd(mc_base[,"peak"]), true_peak))
cat(sprintf("Aperture mass: mean=%.4f  sd=%.4f  (true=%.4f)\n",
            mean(mc_base[,"apmass"]), sd(mc_base[,"apmass"]), true_apm))

# ---- 3. Antithetic variables -----------------------------------------------
cat("\n=== 3. Antithetic variable MC ===\n")
mc_anti <- run_mc_antithetic(gamma_true, kappa_true, grid,
                             B = B_MC, N_gal = N_GAL, lambda = LAMBDA, seed = 42L)
cat("Variance reduction (%):\n")
for (nm in names(mc_anti$pct_reduction))
  cat(sprintf("  %-10s: %+.1f%%\n", nm, mc_anti$pct_reduction[nm]))

# ---- 4. Importance sampling over bias m ------------------------------------
cat("\n=== 4. Importance sampling over multiplicative bias m ===\n")
mc_is <- run_mc_is(gamma_true, kappa_true, grid,
                   B_per_m = if (QUICK) 50L else 100L,
                   n_is    = if (QUICK) 20L else 50L,
                   sigma_m = 0.05, N_gal = N_GAL,
                   lambda = LAMBDA, seed = 42L)
cat(sprintf("IS estimate of E_p[L2 error]: %.4f  (SE: %.4f)\n",
            mc_is$is_estimate, mc_is$is_se))
cat(sprintf("Uniform grid estimate:        %.4f\n", mc_is$uniform_est))

# ---- 5. Bootstrap CIs on peak_kappa ----------------------------------------
cat("\n=== 5. Bootstrap CIs on peak_kappa ===\n")
boot_peak <- boot_ks(gamma_true, grid,
                     stat_fn = peak_kappa,
                     B       = B_BOOT,
                     N_gal   = N_GAL,
                     lambda  = LAMBDA,
                     seed    = 42L)
bs_peak   <- boot_bias_se(boot_peak)
cat(sprintf("peak_kappa: obs=%.4f  boot mean=%.4f  bias=%.4f  SE=%.4f\n",
            bs_peak$t_obs, bs_peak$mean, bs_peak$bias, bs_peak$se))

ci_peak   <- bca_ci(boot_peak, gamma_true, grid,
                    stat_fn = peak_kappa, lambda = LAMBDA, N_gal = N_GAL)
cat(sprintf("95%% Percentile CI: [%.4f, %.4f]\n",
            ci_peak$percentile[1], ci_peak$percentile[2]))
cat(sprintf("95%% BCa CI:        [%.4f, %.4f]  (z0=%.3f  a=%.4f)\n",
            ci_peak$bca[1], ci_peak$bca[2], ci_peak$z0, ci_peak$a))

# ---- 6. Bootstrap CIs on aperture mass -------------------------------------
cat("\n=== 6. Bootstrap CIs on aperture mass ===\n")
apmass_stat <- make_apmass_fn(grid)   # closure over grid
boot_apm  <- boot_ks(gamma_true, grid,
                     stat_fn = apmass_stat,
                     B       = B_BOOT,
                     N_gal   = N_GAL,
                     lambda  = LAMBDA,
                     seed    = 43L)
bs_apm    <- boot_bias_se(boot_apm)
cat(sprintf("aperture_mass: obs=%.4f  boot mean=%.4f  bias=%.4f  SE=%.4f\n",
            bs_apm$t_obs, bs_apm$mean, bs_apm$bias, bs_apm$se))

ci_apm    <- bca_ci(boot_apm, gamma_true, grid,
                    stat_fn = apmass_stat, lambda = LAMBDA, N_gal = N_GAL)
cat(sprintf("95%% Percentile CI: [%.4f, %.4f]\n",
            ci_apm$percentile[1], ci_apm$percentile[2]))
cat(sprintf("95%% BCa CI:        [%.4f, %.4f]  (z0=%.3f  a=%.4f)\n",
            ci_apm$bca[1], ci_apm$bca[2], ci_apm$z0, ci_apm$a))

# Noiseless KS aperture mass -- this is what the estimator actually targets
# (DC is zeroed, so it differs from true_apm by the mean kappa contribution)
truth_apm_recon <- apmass_stat(ks_inverse(gamma_true$gamma1, gamma_true$gamma2,
                                           grid, lambda = LAMBDA))
cat(sprintf("True aperture mass:           %.4f\n", true_apm))
cat(sprintf("Noiseless KS aperture mass:   %.4f  (DC offset = %.4f)\n",
            truth_apm_recon, true_apm - truth_apm_recon))

# ---- 7. Coverage study (aperture_mass, more reliable than peak) ------------
cat("\n=== 7. Bootstrap coverage study (BCa vs percentile) ===\n")
cat("Truth = noiseless KS reconstruction (what the estimator targets).\n")
cov_study <- boot_coverage(kappa_true, gamma_true, grid,
                           stat_fn  = apmass_stat,
                           truth    = truth_apm_recon,
                           n_outer  = N_OUTER,
                           B        = B_BOOT,
                           N_gal    = N_GAL,
                           lambda   = LAMBDA,
                           seed     = 100L)

# ---- 8. Power analysis: unpaired two-sample t-test on aperture_mass --------
cat("\n=== 8. Power analysis: detecting |m| = 0.01 at 5-sigma ===\n")
cat("Statistic: aperture_mass.  Unpaired: independent noise in each condition.\n")
cat("Note: paired design is degenerate because aperture_mass o ks_inverse\n")
cat("is linear in shear -> D = m * const for all replicates, sd(D) = 0.\n\n")

# Analytic delta: E[apmass(m)] - E[apmass(0)] = m * apmass(ks_inverse(g_true))
# Analytic sd(apmass): from MC at N_GAL
delta_analytic  <- 0.01 * truth_apm_recon
sd_apmass_ref   <- sd(mc_base[, "apmass"])   # at N_GAL
cat(sprintf("Analytic delta (m=0.01): %.5f\n", delta_analytic))
cat(sprintf("SD(apmass) at N_gal=%d: %.5f\n\n", N_GAL, sd_apmass_ref))

# Power function using analytic formula:
# SD(apmass) scales as 1/sqrt(N_gal) (shape noise model).
# Two-sample Welch test with B replicates per condition.
unpaired_power_analytic <- function(N_gal, B, delta, sd_ref, N_gal_ref,
                                    alpha = 2.87e-7) {
  sd_ng  <- sd_ref * sqrt(N_gal_ref / N_gal)   # SD at this N_gal
  se     <- sqrt(2) * sd_ng / sqrt(B)           # SE of difference
  ncp    <- delta / se
  t_crit <- qnorm(1 - alpha / 2)               # large df approx
  power  <- pnorm(ncp - t_crit) + pnorm(-ncp - t_crit)
  list(power = power, ncp = ncp, sd_ng = sd_ng, se = se)
}

ngal_grid <- c(1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000)
B_power   <- if (QUICK) 200L else 500L

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

# Verify one MC point against analytic (at N_GAL)
cat(sprintf("\nMC verification at N_gal=%d, B=%d:\n", N_GAL, B_power))
set.seed(201L)
N         <- grid$N_pix
sigma_pix <- SIGMA_E / sqrt(N_GAL / N^2)
g1_null   <- gamma_true$gamma1; g2_null <- gamma_true$gamma2
g1_alt    <- (1 + 0.01) * g1_null; g2_alt <- (1 + 0.01) * g2_null
T0 <- replicate(B_power, {
  k0 <- ks_inverse(g1_null + matrix(rnorm(N^2,0,sigma_pix),N,N),
                   g2_null + matrix(rnorm(N^2,0,sigma_pix),N,N), grid, LAMBDA)
  apmass_stat(k0)
})
set.seed(202L)
Tm <- replicate(B_power, {
  km <- ks_inverse(g1_alt + matrix(rnorm(N^2,0,sigma_pix),N,N),
                   g2_alt + matrix(rnorm(N^2,0,sigma_pix),N,N), grid, LAMBDA)
  apmass_stat(km)
})
delta_mc <- mean(Tm) - mean(T0)
se_mc    <- sqrt(var(Tm)/B_power + var(T0)/B_power)
ncp_mc   <- delta_mc / se_mc
t_crit   <- qt(1 - 2.87e-7/2, df = 2*B_power - 2)
pow_mc   <- 1 - pt(t_crit, df=2*B_power-2, ncp=ncp_mc) +
              pt(-t_crit, df=2*B_power-2, ncp=ncp_mc)
cat(sprintf("  MC delta=%.5f  NCP=%.2f  power=%.4f\n", delta_mc, ncp_mc, pow_mc))
cat(sprintf("  Analytic:         NCP=%.2f  power=%.4f\n",
            unpaired_power_analytic(N_GAL, B_power, delta_analytic,
                                     sd_apmass_ref, N_GAL)$ncp,
            unpaired_power_analytic(N_GAL, B_power, delta_analytic,
                                     sd_apmass_ref, N_GAL)$power))

# Minimum N_gal for 80% power at B=B_power replicates
find_ngal_80 <- function(delta, sd_ref, N_ref, B, alpha = 2.87e-7,
                          target = 0.80) {
  f <- function(ng) {
    unpaired_power_analytic(ng, B, delta, sd_ref, N_ref, alpha)$power - target
  }
  if (f(1e3) >= 0) return(1e3)
  if (f(1e7) < 0)  return(NA)
  as.integer(uniroot(f, c(1e3, 1e7))$root)
}
ng_80 <- find_ngal_80(delta_analytic, sd_apmass_ref, N_GAL, B_power)
cat(sprintf("\nMinimum N_gal for 80%% power (B=%d, m=0.01, 5-sigma): %s\n",
            B_power, if (is.na(ng_80)) "N/A" else format(ng_80, big.mark=",")))

# ---- 9. Save results -------------------------------------------------------
cat("\n=== 9. Saving results ===\n")
dir.create("results", showWarnings = FALSE)

saveRDS(list(
  mc_base          = mc_base,
  mc_anti          = mc_anti,
  mc_is            = mc_is,
  boot_peak        = boot_peak,
  boot_apm         = boot_apm,
  ci_peak          = ci_peak,
  ci_apm           = ci_apm,
  cov_study        = cov_study,
  pow_curve        = pow_curve,
  truth_apm_recon  = truth_apm_recon,
  ng_80            = ng_80,
  params           = list(N_GAL=N_GAL, B_MC=B_MC, B_BOOT=B_BOOT,
                          N_OUTER=N_OUTER, LAMBDA=LAMBDA,
                          grid_pix=grid$pix, grid_N=grid$N_pix)
), file.path("results", "all_results.rds"))

cat("Results saved to results/all_results.rds\n")
cat("Run report/final_report.Rmd to render the report.\n")