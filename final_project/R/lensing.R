# =============================================================================
# R/lensing.R
# Core weak lensing operators: forward (kappa -> gamma) and KS inverse.
#
# Bug fixed (vs original): complex(real=mat, imaginary=mat) strips matrix
# dimensions in R, producing a flat vector.  All complex matrix construction
# now uses  mat_re + 1i * mat_im  which preserves dimensions.
# =============================================================================

make_grid <- function(N_pix = 64, pix = 0.05) {
  coords <- (seq_len(N_pix) - N_pix / 2 - 0.5) * pix
  freqs  <- (seq_len(N_pix) - 1L)
  freqs  <- ifelse(freqs <= N_pix / 2, freqs, freqs - N_pix)
  freqs  <- freqs / (N_pix * pix)
  list(N_pix = N_pix, pix = pix, x = coords, y = coords,
       kx = freqs, ky = freqs, L = N_pix * pix)
}

make_true_kappa <- function(grid, kappa0 = 0.3, sigma_l = 0.5,
                             substructure = FALSE, kappa0_sub = 0.15,
                             sigma_sub = 0.3, offset_sub = c(0.8, 0.0)) {
  X     <- outer(rep(1, grid$N_pix), grid$x)
  Y     <- outer(grid$y, rep(1, grid$N_pix))
  kappa <- kappa0 * exp(-(X^2 + Y^2) / (2 * sigma_l^2))
  if (substructure) {
    dx    <- X - offset_sub[1]
    dy    <- Y - offset_sub[2]
    kappa <- kappa + kappa0_sub * exp(-(dx^2 + dy^2) / (2 * sigma_sub^2))
  }
  kappa
}

# -----------------------------------------------------------------------------
# ks_forward()
# Maps convergence kappa (N x N real matrix) to shear (gamma1, gamma2).
# -----------------------------------------------------------------------------
ks_forward <- function(kappa, grid) {
  N        <- grid$N_pix
  kappa_ft <- fft(kappa)

  KX <- outer(rep(1, N), grid$kx)   # kx varies along columns
  KY <- outer(grid$ky, rep(1, N))   # ky varies along rows
  K2 <- KX^2 + KY^2
  K2[1, 1] <- 1

  D_re <- (KX^2 - KY^2) / K2
  D_im <- (2 * KX * KY) / K2

  # FIX: use arithmetic (+, *) to build complex matrix; complex() drops dims
  D_complex        <- D_re + 1i * D_im
  D_complex[1, 1]  <- 0 + 0i

  gamma_ft <- D_complex * kappa_ft

  gamma1 <- Re(fft(gamma_ft, inverse = TRUE)) / N^2
  gamma2 <- Im(fft(gamma_ft, inverse = TRUE)) / N^2

  list(gamma1 = gamma1, gamma2 = gamma2)
}

# -----------------------------------------------------------------------------
# ks_inverse()
# Kaiser-Squires inverse: recovers kappa from (gamma1, gamma2).
# Optional Wiener-filter regularization with regularization parameter lambda.
# -----------------------------------------------------------------------------
ks_inverse <- function(gamma1, gamma2, grid, lambda = 0) {
  N     <- grid$N_pix
  g1_ft <- fft(gamma1)
  g2_ft <- fft(gamma2)

  # FIX: g1_ft + 1i*g2_ft is equivalent to complex(Re(g1)-Im(g2), Im(g1)+Re(g2))
  # but preserves the N x N matrix structure that fft() returns.
  gamma_ft <- g1_ft + 1i * g2_ft

  KX <- outer(rep(1, N), grid$kx)
  KY <- outer(grid$ky, rep(1, N))
  K2 <- KX^2 + KY^2
  K2[1, 1] <- 1

  D_re <- (KX^2 - KY^2) / K2
  D_im <- (2 * KX * KY) / K2

  # FIX: same as ks_forward -- use + 1i * to keep matrix dims
  D_complex        <- D_re + 1i * D_im
  D_complex[1, 1]  <- 0 + 0i

  Dmod2        <- D_re^2 + D_im^2
  Dmod2[1, 1]  <- 1

  kappa_ft        <- Conj(D_complex) / (Dmod2 + lambda) * gamma_ft
  kappa_ft[1, 1]  <- 0 + 0i

  Re(fft(kappa_ft, inverse = TRUE)) / N^2
}

# -----------------------------------------------------------------------------
# l2_error(), peak_kappa(), aperture_mass()
# -----------------------------------------------------------------------------
l2_error <- function(kappa_hat, kappa_true, border = 4L) {
  idx <- (border + 1L):(nrow(kappa_true) - border)
  kh  <- kappa_hat[idx, idx]  - mean(kappa_hat[idx, idx])
  kt  <- kappa_true[idx, idx] - mean(kappa_true[idx, idx])
  sqrt(sum((kh - kt)^2) / sum(kt^2))
}

peak_kappa <- function(kappa_hat) max(kappa_hat)

aperture_mass <- function(kappa_hat, grid, r_ap = 0.8) {
  X    <- outer(rep(1, grid$N_pix), grid$x)
  Y    <- outer(grid$y, rep(1, grid$N_pix))
  mask <- (X^2 + Y^2) <= r_ap^2
  sum(kappa_hat[mask]) * grid$pix^2
}