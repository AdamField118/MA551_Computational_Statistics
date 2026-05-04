# =============================================================================
# plot_report_figures.R
# Generates three publication-quality figures for the final report:
#
#   figures/kappa_report.png      -- true convergence map (print-ready)
#   figures/mc_distributions.png  -- MC distributions of peak & aperture mass
#   figures/power_curve.png       -- power vs N_gal curve
#
# All parameters are read from results/all_results.rds so every number
# matches the report exactly.
#
# Run from the project root:
#   source("scripts/plot_report_figures.R")
# =============================================================================

library(ggplot2)
library(patchwork)

rds_path <- file.path("results", "all_results.rds")
out_dir  <- "figures"
dir.create(out_dir, showWarnings = FALSE)

if (!file.exists(rds_path))
  stop("Run scripts/run_all.R first to produce results/all_results.rds")

res <- readRDS(rds_path)

# Shared theme for print (white background, clean grid)
theme_report <- function(base_size = 11) {
  theme_bw(base_size = base_size) %+replace%
    theme(
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey92", colour = NA),
      plot.title       = element_text(size = base_size, face = "bold",
                                      hjust = 0.5)
    )
}

crimson <- "#AC2B37"

# =============================================================================
# Figure 1: True convergence map (print version — no title, white bg)
# =============================================================================
source(file.path("R", "lensing.R"))

grid       <- make_grid(N_pix = res$params$grid_N, pix = res$params$grid_pix)
kappa_true <- make_true_kappa(grid, kappa0 = 0.3, sigma_l = 0.5)

df_kappa <- expand.grid(x = grid$x, y = grid$y)
df_kappa$kappa <- as.vector(kappa_true)

p_kappa <- ggplot(df_kappa, aes(x = x, y = y, fill = kappa)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_gradientn(
    colours = c("white", "#FFF0CC", "#FFAA33", "#CC4400", "#661100", "#000000"),
    name    = expression(kappa),
    limits  = c(0, max(kappa_true)),
    breaks  = c(0, 0.1, 0.2, round(max(kappa_true), 3)),
    labels  = c("0.00", "0.10", "0.20",
                sprintf("%.3f", max(kappa_true)))
  ) +
  coord_fixed() +
  labs(x = "arcmin", y = "arcmin") +
  theme_report() +
  theme(legend.key.height = unit(1.2, "cm"))

ggsave(file.path(out_dir, "kappa_report.png"),
       p_kappa, width = 3.8, height = 3.5, dpi = 220)
cat("Saved: figures/kappa_report.png\n")

# =============================================================================
# Figure 2: MC distributions — peak kappa (biased) vs aperture mass (unbiased)
# =============================================================================
mc <- as.data.frame(res$mc_base)

# True / target values
true_peak <- 0.297          # max(kappa_true), deterministic
true_apm  <- res$truth_apm_recon  # noiseless KS target (what bootstrap targets)

p_peak <- ggplot(mc, aes(x = peak)) +
  geom_histogram(bins = 30, fill = crimson, alpha = 0.75, colour = "white") +
  geom_vline(xintercept = true_peak, linetype = "dashed", linewidth = 0.8,
             colour = "black") +
  annotate("text", x = true_peak - 0.005, y = Inf,
           label = sprintf("truth\n%.3f", true_peak),
           hjust = 1, vjust = 1.3, size = 3.2) +
  labs(x = expression(hat(kappa)[peak]),
       y = "Count",
       title = expression("Peak "*hat(kappa)*"  (biased)")) +
  theme_report()

p_apm <- ggplot(mc, aes(x = apmass)) +
  geom_histogram(bins = 30, fill = "#2B6CAC", alpha = 0.75, colour = "white") +
  geom_vline(xintercept = true_apm, linetype = "dashed", linewidth = 0.8,
             colour = "black") +
  annotate("text", x = true_apm + 0.001, y = Inf,
           label = sprintf("KS target\n%.3f", true_apm),
           hjust = 0, vjust = 1.3, size = 3.2) +
  labs(x = expression(T[ap]),
       y = "Count",
       title = "Aperture mass  (unbiased)") +
  theme_report()

p_mc <- p_peak + p_apm +
  plot_annotation(
    caption = sprintf("B = %d Monte Carlo replicates, N_gal = %d.",
                      res$params$B_MC, res$params$N_GAL)
  )

ggsave(file.path(out_dir, "mc_distributions.png"),
       p_mc, width = 7, height = 3.2, dpi = 220)
cat("Saved: figures/mc_distributions.png\n")

# =============================================================================
# Figure 3: Power curve — power vs N_gal (smooth analytic + grid points)
# =============================================================================

# Reconstruct the analytic formula from saved RDS quantities
delta_analytic <- 0.01 * res$truth_apm_recon
sd_ref         <- sd(res$mc_base[, "apmass"])
N_gal_ref      <- res$params$N_GAL
B_power        <- res$params$B_MC
alpha_5sig     <- 2.87e-7

power_analytic <- function(N_gal, B = B_power, delta = delta_analytic,
                            sd_r = sd_ref, N_r = N_gal_ref,
                            alpha = alpha_5sig) {
  sd_ng  <- sd_r * sqrt(N_r / N_gal)
  se     <- sqrt(2) * sd_ng / sqrt(B)
  ncp    <- delta / se
  t_crit <- qnorm(1 - alpha / 2)
  pnorm(ncp - t_crit) + pnorm(-ncp - t_crit)
}

# Dense smooth curve
ngal_smooth <- exp(seq(log(500), log(250000), length.out = 300))
df_smooth   <- data.frame(
  N_gal = ngal_smooth,
  power = sapply(ngal_smooth, power_analytic)
)

# Grid points from RDS (for overplotted dots)
df_grid <- as.data.frame(res$pow_curve)

# 80% crossing point
ng_80 <- uniroot(function(ng) power_analytic(ng) - 0.80,
                 c(1000, 200000))$root

p_power <- ggplot(df_smooth, aes(x = N_gal, y = power)) +
  geom_hline(yintercept = 0.80, linetype = "dashed",
             colour = "grey50", linewidth = 0.6) +
  geom_hline(yintercept = 0.50, linetype = "dotted",
             colour = "grey70", linewidth = 0.5) +
  geom_vline(xintercept = ng_80, linetype = "dashed",
             colour = crimson, linewidth = 0.6, alpha = 0.7) +
  geom_line(colour = crimson, linewidth = 1) +
  geom_point(data = df_grid,
             aes(x = N_gal, y = power),
             colour = crimson, size = 2.2, shape = 19) +
  annotate("text",
           x = ng_80 * 1.08, y = 0.08,
           label = sprintf("N = %s", format(round(ng_80), big.mark = ",")),
           hjust = 0, colour = crimson, size = 3.3) +
  annotate("text", x = 600, y = 0.825,
           label = "80%", hjust = 0, colour = "grey40", size = 3.2) +
  scale_x_log10(
    breaks = c(1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000),
    labels = c("1k", "2k", "5k", "10k", "20k", "50k", "100k", "200k")
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(
    x = expression(N[gal]~"(source galaxies)"),
    y = "Power",
    title = expression("Power to detect "*"|"*m*"|"~"= 0.01 at 5"*sigma)
  ) +
  theme_report()

ggsave(file.path(out_dir, "power_curve.png"),
       p_power, width = 5.5, height = 3.5, dpi = 220)
cat("Saved: figures/power_curve.png\n")

cat("\nAll report figures saved to figures/\n")
