# =============================================================================
# R/simulate.R
# Galaxy catalog simulation and shape noise addition.
#
# Provides: add_shape_noise_to_grid()
#
# Physics summary:
#   Each simulated galaxy contributes an ellipticity estimate
#     eps_i = (1 + m) * gamma(x_i) + c + e_i,    e_i ~ N(0, sigma_e^2 / 2)
#   where gamma(x_i) is the true shear at galaxy position x_i,
#   m is multiplicative bias, c is additive bias, and sigma_e is the
#   per-component intrinsic shape dispersion (~0.26 per component).
#
#   Galaxies are assumed uniformly distributed; we bin onto the kappa grid
#   (no actual positions needed for grid-level statistics).
#   The effective per-pixel noise is sigma_e / sqrt(n_gal_per_pixel).
#
# Statistical framing:
#   This is the noise model driving the inverse problem.  Shape noise is
#   irreducible (intrinsic galaxy ellipticities); varying N_gal directly
#   controls the signal-to-noise in the shear map and therefore the
#   reconstruction error kappa_hat - kappa_true.
# =============================================================================


# -----------------------------------------------------------------------------
# add_shape_noise_to_grid()
#
# Simulates a noisy, possibly biased shear observation on a pixelised grid.
#
# Arguments:
#   gamma_true : list with $gamma1 and $gamma2 (N x N real matrices)
#   grid       : from make_grid()
#   N_gal      : total number of source galaxies in the field
#   sigma_e    : per-component intrinsic ellipticity dispersion
#   m          : multiplicative bias (scalar; 0 = unbiased)
#   c          : additive bias (scalar; 0 = unbiased)
#   seed       : optional RNG seed for reproducibility
#
# Returns: list with
#   $gamma1_est  : N x N estimated gamma1 (biased + noisy)
#   $gamma2_est  : N x N estimated gamma2 (biased + noisy)
#   $sigma_pix   : per-pixel noise standard deviation (scalar)
#   $n_gal_pix   : expected number of galaxies per pixel (scalar)
# -----------------------------------------------------------------------------
add_shape_noise_to_grid <- function(gamma_true, grid,
                                    N_gal   = 500,
                                    sigma_e = 0.26,
                                    m       = 0,
                                    c       = 0,
                                    seed    = NULL) {
  if (!is.null(seed)) set.seed(seed)

  N          <- grid$N_pix
  n_gal_pix  <- N_gal / N^2          # expected galaxies per pixel
  sigma_pix  <- sigma_e / sqrt(n_gal_pix)

  # Apply linear shear bias model: gamma_obs = (1+m) * gamma_true + c
  g1_biased  <- (1 + m) * gamma_true$gamma1 + c
  g2_biased  <- (1 + m) * gamma_true$gamma2 + c

  # Add independent Gaussian shape noise
  noise1     <- matrix(rnorm(N^2, 0, sigma_pix), N, N)
  noise2     <- matrix(rnorm(N^2, 0, sigma_pix), N, N)

  list(
    gamma1_est = g1_biased + noise1,
    gamma2_est = g2_biased + noise2,
    sigma_pix  = sigma_pix,
    n_gal_pix  = n_gal_pix
  )
}


# -----------------------------------------------------------------------------
# simulate_metacal_response()
#
# Simulates a metacalibration-style response matrix R for the shear estimator.
# For a perfect estimator with no model bias, R = diag(1, 1).
# For a biased estimator, R = diag(1+m, 1+m) where m is the mult. bias.
#
# In the project we directly inject bias via m in add_shape_noise_to_grid;
# this function is provided for completeness and power analysis use.
# -----------------------------------------------------------------------------
simulate_metacal_response <- function(m = 0) {
  matrix(c(1 + m, 0, 0, 1 + m), 2, 2)
}


# -----------------------------------------------------------------------------
# galaxy_catalog()
#
# Generates a full galaxy catalog with positions and shapes.
# Each galaxy is assigned a random position in the field and an
# intrinsic ellipticity drawn from a Gaussian.  The observed ellipticity
# is the sum of the lensed shear (interpolated to the galaxy position)
# plus the intrinsic shape noise.
#
# Arguments:
#   gamma_true : list with $gamma1, $gamma2 (N x N matrices on grid)
#   grid       : from make_grid()
#   N_gal      : number of galaxies
#   sigma_e    : per-component intrinsic shape dispersion
#   m, c       : bias parameters
#   seed       : optional seed
#
# Returns: data.frame with columns x, y, e1_obs, e2_obs, gamma1_true, gamma2_true
# -----------------------------------------------------------------------------
galaxy_catalog <- function(gamma_true, grid,
                           N_gal   = 500,
                           sigma_e = 0.26,
                           m       = 0,
                           c       = 0,
                           seed    = NULL) {
  if (!is.null(seed)) set.seed(seed)

  # Random galaxy positions in [xmin, xmax] x [ymin, ymax]
  x_gal <- runif(N_gal, min(grid$x), max(grid$x))
  y_gal <- runif(N_gal, min(grid$y), max(grid$y))

  # Bilinear interpolation of true shear to galaxy positions
  g1_at_gal <- bilinear_interp(gamma_true$gamma1, grid, x_gal, y_gal)
  g2_at_gal <- bilinear_interp(gamma_true$gamma2, grid, x_gal, y_gal)

  # Intrinsic ellipticity noise
  e1_int <- rnorm(N_gal, 0, sigma_e)
  e2_int <- rnorm(N_gal, 0, sigma_e)

  # Observed ellipticity: bias model + intrinsic noise
  e1_obs <- (1 + m) * g1_at_gal + c + e1_int
  e2_obs <- (1 + m) * g2_at_gal + c + e2_int

  data.frame(
    x           = x_gal,
    y           = y_gal,
    e1_obs      = e1_obs,
    e2_obs      = e2_obs,
    gamma1_true = g1_at_gal,
    gamma2_true = g2_at_gal
  )
}


# -----------------------------------------------------------------------------
# bilinear_interp()
# Bilinear interpolation of an N x N grid matrix at scattered (x, y) points.
# Used internally by galaxy_catalog().
# -----------------------------------------------------------------------------
bilinear_interp <- function(mat, grid, x_pts, y_pts) {
  N   <- grid$N_pix
  x0  <- min(grid$x); x1 <- max(grid$x)
  y0  <- min(grid$y); y1 <- max(grid$y)

  # Normalize to [0, N-1] index space
  xi  <- pmin(pmax((x_pts - x0) / (x1 - x0) * (N - 1), 0), N - 2)
  yi  <- pmin(pmax((y_pts - y0) / (y1 - y0) * (N - 1), 0), N - 2)

  ix  <- floor(xi); dx <- xi - ix
  iy  <- floor(yi); dy <- yi - iy

  # Bilinear weights; R matrices are indexed [row, col] = [y, x]
  i00 <- cbind(iy + 1,      ix + 1)
  i10 <- cbind(iy + 2,      ix + 1)
  i01 <- cbind(iy + 1,      ix + 2)
  i11 <- cbind(iy + 2,      ix + 2)

  (1 - dy) * (1 - dx) * mat[i00] +
  dy       * (1 - dx) * mat[i10] +
  (1 - dy) * dx       * mat[i01] +
  dy       * dx       * mat[i11]
}
