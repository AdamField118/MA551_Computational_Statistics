# =============================================================================
# plot_kappa.R
# Generates a presentation-ready kappa_true heatmap for the simulation setup
# slide.  Reads grid parameters from results/all_results.rds so the figure
# is guaranteed to match the run that produced the report numbers.
#
# Run from the project root:
#   source("scripts/plot_kappa.R")   or   Rscript scripts/plot_kappa.R
#
# Output: figures/kappa_true.png  (also .pdf if SAVE_PDF <- TRUE)
# =============================================================================

library(ggplot2)

# ── 0. Paths -----------------------------------------------------------------
rds_path <- file.path("results", "all_results.rds")
r_dir    <- "R"
out_dir  <- "figures"
dir.create(out_dir, showWarnings = FALSE)

# ── 1. Load grid parameters from the RDS ------------------------------------
# kappa_true itself is not stored, but the grid dims used in the run are.
if (file.exists(rds_path)) {
  res    <- readRDS(rds_path)
  N_PIX  <- res$params$grid_N    # 32
  PIX    <- res$params$grid_pix  # 0.1 arcmin
  cat(sprintf("Grid params from RDS: N_pix = %d, pix = %.3f arcmin\n",
              N_PIX, PIX))
} else {
  # Fallback to the values hard-coded in run_all.R
  message("RDS not found; using hard-coded grid params from run_all.R")
  N_PIX <- 32L
  PIX   <- 0.1
}

# ── 2. Source lensing.R and recreate the map --------------------------------
source(file.path(r_dir, "lensing.R"))

grid       <- make_grid(N_pix = N_PIX, pix = PIX)
kappa_true <- make_true_kappa(grid, kappa0 = 0.3, sigma_l = 0.5)

cat(sprintf("kappa_true: peak = %.4f, mean = %.6f\n",
            max(kappa_true), mean(kappa_true)))

# ── 3. Tidy data frame for ggplot -------------------------------------------
df <- expand.grid(
  x = grid$x,   # arcmin
  y = grid$y
)
df$kappa <- as.vector(kappa_true)

# ── 4. Plot ------------------------------------------------------------------
# Colour scale: white (low) -> dark orange -> crimson (high).
# This reads clearly on both dark and light slide backgrounds.
p <- ggplot(df, aes(x = x, y = y, fill = kappa)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_gradientn(
    colours  = c("#0D0D0D", "#3B1005", "#8B2500", "#CC4A00",
                 "#E8800A", "#F5C842", "#FFFDE0"),
    name     = expression(kappa),
    limits   = c(0, max(kappa_true)),
    breaks   = c(0, 0.1, 0.2, max(kappa_true)),
    labels   = c("0.0", "0.1", "0.2",
                 sprintf("%.3f", max(kappa_true)))
  ) +
  coord_fixed() +
  labs(
    title    = expression("True convergence"~kappa(bold(theta))),
    subtitle = expression(kappa[0]~"= 0.297,"~sigma[l]~"= 0.5 arcmin,  32"~"\u00d7"~"32 grid"),
    x        = expression(theta[1]~"(arcmin)"),
    y        = expression(theta[2]~"(arcmin)")
  ) +
  theme_void(base_size = 13) +
  theme(
    plot.background  = element_rect(fill = "#0D0D0D", colour = NA),
    panel.background = element_rect(fill = "#0D0D0D", colour = NA),
    plot.title       = element_text(colour = "white",  size = 14,
                                    face = "bold", hjust = 0.5,
                                    margin = margin(b = 4)),
    plot.subtitle    = element_text(colour = "#999999", size = 10,
                                    hjust = 0.5,
                                    margin = margin(b = 8)),
    legend.position  = "right",
    legend.title     = element_text(colour = "white",  size = 11),
    legend.text      = element_text(colour = "#CCCCCC", size = 9),
    plot.margin      = margin(12, 12, 12, 12)
  )

# ── 5. Save -----------------------------------------------------------------
SAVE_PDF <- FALSE   # set TRUE if you also want a vector PDF

png_path <- file.path(out_dir, "kappa_true.png")
ggsave(png_path, p, width = 4.5, height = 4.2, dpi = 220, bg = "#0D0D0D")
cat(sprintf("Saved: %s\n", png_path))

if (SAVE_PDF) {
  pdf_path <- file.path(out_dir, "kappa_true.pdf")
  ggsave(pdf_path, p, width = 4.5, height = 4.2)
  cat(sprintf("Saved: %s\n", pdf_path))
}
