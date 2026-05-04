# =============================================================================
# R/lensing.R
# Core weak lensing operators: forward (kappa -> gamma) and KS inverse.
#
# Physics summary (can be skipped in the report):
#   The lensing potential psi satisfies Laplace^2 psi = 2*kappa.
#   In Fourier space the convergence and shear are related by:
#     gamma_hat(k) = D(k) * kappa_hat(k)
#   where D(k) = (k1^2 - k2^2 + 2i*k1*k2) / (k1^2 + k2^2)
#   is the Kaiser-Squires kernel (undefined at k=0; set to 0).
#   The KS inverse is just the conjugate: kappa_hat = Re( conj(D) * gamma_hat ).
#
# In statistical terms: F is a known linear operator with a closed-form
# pseudoinverse, making this an ideal testbed for studying how noise
# propagates through a regularized inverse problem.
# =============================================================================

# -----------------------------------------------------------------------------
# make_grid()
# Returns a list with the spatial grid and its Fourier-space frequencies.
# N_pix : number of pixels along each axis (square grid)
# pix   : pixel scale (arcmin)
# -----------------------------------------------------------------------------
make_grid <- function(N_pix = 64, pix = 0.05) {
  # Spatial coordinates (centered at 0)
  coords <- (seq_len(N_pix) - N_pix / 2 - 0.5) * pix

  # Fourier frequencies (cycles per arcmin), using standard FFT convention
  freqs  <- (seq_len(N_pix) - 1L)
  freqs  <- ifelse(freqs <= N_pix / 2, freqs, freqs - N_pix)
  freqs  <- freqs / (N_pix * pix)

  list(
    N_pix  = N_pix,
    pix    = pix,
    x      = coords,
    y      = coords,
    kx     = freqs,
    ky     = freqs,
    L      = N_pix * pix   # field side length in arcmin
  )
}

# -----------------------------------------------------------------------------
# make_true_kappa()
# Gaussian convergence profile: kappa(r) = kappa0 * exp(-r^2 / 2*sigma_l^2)
# Optionally adds a secondary component (substructure) offset from center.
# -----------------------------------------------------------------------------
make_true_kappa <- function(grid,
                            kappa0  = 0.3,
                            sigma_l = 0.5,
                            substructure = FALSE,
                            kappa0_sub   = 0.15,
                            sigma_sub    = 0.3,
                            offset_sub   = c(0.8, 0.0)) {
  X  <- outer(rep(1, grid$N_pix), grid$x)   # N x N matrix of x coords
  Y  <- outer(grid$y, rep(1, grid$N_pix))   # N x N matrix of y coords

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
# Returns a list with two N x N real matrices.
# -----------------------------------------------------------------------------
ks_forward <- function(kappa, grid) {
  N  <- grid$N_pix

  # 2D FFT of kappa
  kappa_ft <- fft(kappa)

  # Build KS kernel D(kx, ky) on the Fourier grid
  # Outer products give the 2D frequency arrays
  KX <- outer(rep(1, N), grid$kx)   # kx varies along columns
  KY <- outer(grid$ky, rep(1, N))   # ky varies along rows

  K2 <- KX^2 + KY^2
  K2[1, 1] <- 1      # avoid division by zero at DC; set gamma_ft[1,1] = 0 below

  # Real and imaginary parts of D(k) = (kx^2 - ky^2 + 2i*kx*ky) / (kx^2 + ky^2)
  D_re <- (KX^2 - KY^2) / K2
  D_im <- (2 * KX * KY) / K2

  # Compute gamma1_ft = Re(D) * kappa_ft, gamma2_ft = Im(D) * kappa_ft
  # (D is applied as complex multiplication in Fourier space)
  D_complex <- complex(real = D_re, imaginary = D_im)
  D_complex[1, 1] <- 0 + 0i   # zero DC mode

  gamma_ft <- D_complex * kappa_ft

  # Inverse FFT and take real part (imaginary part should be ~0 for real kappa)
  gamma1 <- Re(fft(gamma_ft, inverse = TRUE)) / N^2
  gamma2 <- Im(fft(gamma_ft, inverse = TRUE)) / N^2

  # Note: gamma1 = Re(gamma), gamma2 = Im(gamma) using the complex shear convention.
  # Equivalently:
  #   gamma1 = (psi_xx - psi_yy) / 2
  #   gamma2 = psi_xy
  list(gamma1 = gamma1, gamma2 = gamma2)
}

# -----------------------------------------------------------------------------
# ks_inverse()
# Kaiser-Squires inverse: recovers kappa from (gamma1, gamma2).
# Applies conj(D) / |D|^2 = conj(D) in Fourier space (since |D|=1 away from DC).
# Optional Wiener-filter regularization with regularization parameter lambda.
# -----------------------------------------------------------------------------
ks_inverse <- function(gamma1, gamma2, grid, lambda = 0) {
  N  <- grid$N_pix

  # FFT of observed shear components
  g1_ft <- fft(gamma1)
  g2_ft <- fft(gamma2)

  # Reconstruct complex gamma in Fourier space: gamma_ft = g1_ft + i*g2_ft
  gamma_ft <- complex(real = Re(g1_ft) - Im(g2_ft),
                      imaginary = Im(g1_ft) + Re(g2_ft))

  # Build KS kernel (same as forward)
  KX <- outer(rep(1, N), grid$kx)
  KY <- outer(grid$ky, rep(1, N))
  K2 <- KX^2 + KY^2
  K2[1, 1] <- 1

  D_re <- (KX^2 - KY^2) / K2
  D_im <- (2 * KX * KY) / K2
  D_complex <- complex(real = D_re, imaginary = D_im)
  D_complex[1, 1] <- 0 + 0i

  # kappa_ft = conj(D) * gamma_ft, with optional Tikhonov regularization
  # Wiener filter: kappa_ft = conj(D) / (|D|^2 + lambda) * gamma_ft
  # Since |D|=1 away from DC: Wiener filter = conj(D)/(1 + lambda) * gamma_ft
  Dmod2 <- D_re^2 + D_im^2
  Dmod2[1, 1] <- 1  # avoid zero denominator at DC

  kappa_ft <- Conj(D_complex) / (Dmod2 + lambda) * gamma_ft
  kappa_ft[1, 1] <- 0 + 0i  # set DC (mean) to zero

  Re(fft(kappa_ft, inverse = TRUE)) / N^2
}

# -----------------------------------------------------------------------------
# l2_error()
# Normalized L2 reconstruction error: ||kappa_hat - kappa_true|| / ||kappa_true||
# Evaluated on the interior only (strip 'border' pixels from each edge)
# to avoid FFT boundary artefacts.
# -----------------------------------------------------------------------------
l2_error <- function(kappa_hat, kappa_true, border = 4L) {
  idx <- (border + 1L):(nrow(kappa_true) - border)
  num <- sum((kappa_hat[idx, idx] - kappa_true[idx, idx])^2)
  den <- sum(kappa_true[idx, idx]^2)
  sqrt(num / den)
}

# -----------------------------------------------------------------------------
# peak_kappa()
# Returns the peak value of the reconstructed kappa map.
# A simple scalar summary for bootstrap confidence intervals.
# -----------------------------------------------------------------------------
peak_kappa <- function(kappa_hat) max(kappa_hat)

# -----------------------------------------------------------------------------
# aperture_mass()
# Aperture mass statistic: sum of kappa_hat pixels within radius r_ap (arcmin).
# A second scalar summary for bootstrap and power analysis.
# -----------------------------------------------------------------------------
aperture_mass <- function(kappa_hat, grid, r_ap = 0.8) {
  X   <- outer(rep(1, grid$N_pix), grid$x)
  Y   <- outer(grid$y, rep(1, grid$N_pix))
  mask <- (X^2 + Y^2) <= r_ap^2
  sum(kappa_hat[mask]) * grid$pix^2
}