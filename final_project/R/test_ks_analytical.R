# =============================================================================
# tests/test_ks_analytical.R
#
# Analytical validation of the R Kaiser-Squires implementation in lensing.R
# against SMPy's KaiserSquiresMapper.create_maps() formulas.
#
# FINDINGS (from reading both source files):
#   The R ks_inverse() and SMPy's KaiserSquiresMapper are mathematically
#   identical for the E-mode with lambda=0 and no smoothing.  The same
#   KS kernel D(k) = (kx^2 - ky^2 + 2i*kx*ky)/(kx^2+ky^2) appears in
#   both; the two expressions below are algebraically equal:
#
#     R:     Re[ conj(D(k)) * (FFT(g1) + i*FFT(g2)) ]
#     SMPy:  Re[ IFFT( ((k1^2-k2^2)*G1 + 2*k1*k2*G2) / k^2 ) ]
#
#   Both zero the DC mode (k=0).  FFT normalisation is the same (both
#   divide by N^2).  The g2 sign flip in SMPy only applies to RA/Dec
#   catalogs; for pixelised grid simulations it is absent.
#
# TEST INVENTORY:
#   T01  Grid frequencies match numpy fftfreq convention
#   T02  KS kernel |D(k)| = 1 for all k != 0
#   T03  Forward-inverse round-trip: L2 error < 1e-10 (noiseless, interior)
#   T04  Linearity of forward operator
#   T05  Adjoint / Parseval check: <Fk, g> = <k, F*g>
#   T06  B-mode purity: E-only kappa produces zero B-mode
#   T07  SMPy exact formula ported to R matches ks_inverse() output
#   T08  Known analytical solution: single-frequency kappa
#   T09  DC mode zeroed: mean kappa does not affect shear
#   T10  Peak recovery: reconstructed peak position matches truth
#
# Run from project root:
#   source("tests/test_ks_analytical.R")
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
source(file.path("R", "lensing.R"))

set.seed(42)

# ---- Test harness -----------------------------------------------------------
.pass  <- 0L
.fail  <- 0L
.results <- list()

ok <- function(name, condition, detail = "") {
  if (isTRUE(condition)) {
    cat(sprintf("  PASS  %s\n", name))
    .pass <<- .pass + 1L
  } else {
    cat(sprintf("  FAIL  %s\n", name))
    if (nchar(detail) > 0) cat(sprintf("        %s\n", detail))
    .fail <<- .fail + 1L
  }
  .results[[name]] <<- condition
}

# ---- Shared fixtures --------------------------------------------------------
N   <- 64L
pix <- 0.05
grid       <- make_grid(N_pix = N, pix = pix)
kappa_true <- make_true_kappa(grid, kappa0 = 0.3, sigma_l = 0.5)

# Compute forward shear once; reused by multiple tests
gamma_true <- ks_forward(kappa_true, grid)
g1_true    <- gamma_true$gamma1
g2_true    <- gamma_true$gamma2

# Helper: normalised interior L2 error (strip 4-pixel border)
border <- 4L
idx    <- (border + 1L):(N - border)
l2_interior <- function(a, b) {
  num <- sqrt(sum((a[idx, idx] - b[idx, idx])^2))
  den <- sqrt(sum(b[idx, idx]^2))
  num / (den + 1e-30)
}

# ---- T01: Grid frequencies match numpy fftfreq convention ------------------
cat("\nT01: Grid frequency convention\n")
{
  # numpy fftfreq(N) = [0, 1, ..., N/2-1, -N/2, ..., -1] / N (then /pix for physical)
  # R's make_grid does:
  #   freqs = (0..N-1); freqs[freqs > N/2] <- freqs - N; freqs / (N*pix)
  freqs_expected <- (0:(N - 1L))
  freqs_expected <- ifelse(freqs_expected <= N / 2, freqs_expected,
                           freqs_expected - N)
  freqs_expected <- freqs_expected / (N * pix)

  ok("T01a: kx matches expected fftfreq convention",
     max(abs(grid$kx - freqs_expected)) < 1e-12,
     sprintf("max diff = %.2e", max(abs(grid$kx - freqs_expected))))

  ok("T01b: ky matches expected fftfreq convention",
     max(abs(grid$ky - freqs_expected)) < 1e-12)

  ok("T01c: DC mode is at index [1,1]",
     grid$kx[1] == 0 && grid$ky[1] == 0)

  ok("T01d: Nyquist frequency is at index [N/2+1]",
     abs(grid$kx[N / 2 + 1L] - 1 / (2 * pix)) < 1e-12)
}

# ---- T02: KS kernel |D(k)| = 1 for k != 0 ---------------------------------
cat("\nT02: KS kernel unit magnitude\n")
{
  KX <- outer(rep(1, N), grid$kx)
  KY <- outer(grid$ky, rep(1, N))
  K2 <- KX^2 + KY^2

  # Away from DC, |D|^2 = (kx^2-ky^2)^2/(kx^2+ky^2)^2 + 4kx^2ky^2/(kx^2+ky^2)^2
  #                      = (kx^2+ky^2)^2/(kx^2+ky^2)^2 = 1
  D_re   <- (KX^2 - KY^2) / pmax(K2, 1e-30)
  D_im   <- (2 * KX * KY) / pmax(K2, 1e-30)
  D_mod2 <- D_re^2 + D_im^2

  # Exclude DC pixel
  nondc       <- K2 > 0
  max_err_mag <- max(abs(D_mod2[nondc] - 1))
  ok("T02a: |D(k)| = 1 for all k != 0 (max err < 1e-14)",
     max_err_mag < 1e-14,
     sprintf("max |D|^2 - 1| = %.2e", max_err_mag))

  ok("T02b: D(k) is even: D(-k) = D(k)",
   {
     D_cplx <- D_re + 1i * D_im
     err <- 0
     for (m in c(2, 5, 10)) {
       km <- m + 1L            # index for +m frequency
       kn <- N - m + 1L        # index for -m frequency
       # D(-kx,-ky) should equal D(kx,ky)
       err <- max(err, abs(D_cplx[km, km] - D_cplx[kn, kn]))
     }
     err < 1e-14
   })
}

# ---- T03: Forward-inverse round-trip (noiseless) ---------------------------
cat("\nT03: Forward-inverse round-trip\n")
{
  kappa_hat <- ks_inverse(g1_true, g2_true, grid, lambda = 0)

  # DC ambiguity: both R and SMPy zero the DC mode, so mean(kappa) is not
  # preserved.  Compare on the mean-subtracted interior.
  kh_int  <- kappa_hat[idx, idx] - mean(kappa_hat[idx, idx])
  kt_int  <- kappa_true[idx, idx] - mean(kappa_true[idx, idx])
  rel_err <- sqrt(sum((kh_int - kt_int)^2)) / sqrt(sum(kt_int^2))

  ok("T03a: Noiseless round-trip L2 error < 1e-10",
     rel_err < 1e-10,
     sprintf("rel L2 = %.2e", rel_err))

  ok("T03b: Peak position preserved",
     which.min(abs(kappa_hat - max(kappa_hat))) ==
       which.min(abs(kappa_true - max(kappa_true))))

  # Wiener filter (lambda > 0) should smooth: peak lower, error higher
  kappa_wiener <- ks_inverse(g1_true, g2_true, grid, lambda = 1e-2)
  ok("T03c: Tikhonov lambda > 0 reduces peak (regularization acts)",
     max(kappa_wiener) < max(kappa_hat),
     sprintf("peak(lambda=0)=%.4f  peak(lambda=0.01)=%.4f",
             max(kappa_hat), max(kappa_wiener)))
}

# ---- T04: Linearity of forward operator ------------------------------------
cat("\nT04: Linearity of ks_forward()\n")
{
  a <- 2.7; b <- -1.3
  kappa2      <- make_true_kappa(grid, kappa0 = 0.5, sigma_l = 0.8,
                                  substructure = TRUE)
  gamma1_a    <- ks_forward(kappa_true, grid)
  gamma1_b    <- ks_forward(kappa2, grid)
  gamma1_comb <- ks_forward(a * kappa_true + b * kappa2, grid)

  err_g1 <- max(abs(gamma1_comb$gamma1 -
                      (a * gamma1_a$gamma1 + b * gamma1_b$gamma1)))
  err_g2 <- max(abs(gamma1_comb$gamma2 -
                      (a * gamma1_a$gamma2 + b * gamma1_b$gamma2)))
  ok("T04a: F(a*k1 + b*k2) = a*F(k1) + b*F(k2) for gamma1",
     err_g1 < 1e-14,
     sprintf("max err = %.2e", err_g1))
  ok("T04b: F(a*k1 + b*k2) = a*F(k1) + b*F(k2) for gamma2",
     err_g2 < 1e-14,
     sprintf("max err = %.2e", err_g2))
}

# ---- T05: Adjoint / Parseval check -----------------------------------------
cat("\nT05: Parseval / adjoint check <F*kappa, gamma> = <kappa, F^adj*gamma>\n")
{
  # Forward: F maps kappa -> (gamma1, gamma2)
  # Adjoint: F^adj maps (gamma1, gamma2) -> kappa via the same KS kernel
  # For this linear FFT operator, the adjoint equals the inverse (since |D|=1).
  # So <F(k), g> = <k, F^{-1}(g)> up to the DC ambiguity.
  #
  # Test with two random fields.
  set.seed(1234)
  k_test <- matrix(rnorm(N^2), N, N)
  g1_rnd <- matrix(rnorm(N^2), N, N)
  g2_rnd <- matrix(rnorm(N^2), N, N)

  gamma_from_k <- ks_forward(k_test, grid)
  kappa_from_g <- ks_inverse(g1_rnd, g2_rnd, grid, lambda = 0)

  # <F(k), (g1,g2)> = sum(gamma1*g1) + sum(gamma2*g2)
  lhs <- sum(gamma_from_k$gamma1 * g1_rnd) + sum(gamma_from_k$gamma2 * g2_rnd)
  # <k, F^{-1}(g1,g2)> = sum(k * kappa_from_g)
  rhs <- sum(k_test * kappa_from_g)
  rel <- abs(lhs - rhs) / (abs(lhs) + 1e-30)

  ok("T05: <F*kappa, gamma> ~ <kappa, F^adj*gamma> (rel err < 1e-8)",
     rel < 1e-8,
     sprintf("lhs=%.6e  rhs=%.6e  rel=%.2e", lhs, rhs, rel))
}

# ---- T06: B-mode purity ----------------------------------------------------
cat("\nT06: B-mode purity for E-only kappa\n")
{
  # For a physical convergence field, shear has no B-mode content.
  # SMPy's B-mode formula: kappa_B = Re[ IFFT( ((k1^2-k2^2)*G2 - 2k1k2*G1)/k^2 ) ]
  # Port this exactly to R:
  N    <- grid$N_pix
  KX   <- outer(rep(1, N), grid$kx)
  KY   <- outer(grid$ky, rep(1, N))
  K2   <- KX^2 + KY^2
  K2[1, 1] <- .Machine$double.eps   # SMPy uses eps, not 1

  g1_ft <- fft(g1_true)
  g2_ft <- fft(g2_true)

  kappa_b_hat <- ((KX^2 - KY^2) * g2_ft - 2 * KX * KY * g1_ft) / K2
  kappa_b     <- Re(fft(kappa_b_hat, inverse = TRUE)) / N^2
  kappa_b[1, 1] <- 0

  max_b   <- max(abs(kappa_b[idx, idx]))
  max_kap <- max(abs(kappa_true[idx, idx]))
  frac_b  <- max_b / max_kap

  ok("T06: B-mode amplitude < 1e-10 * E-mode peak for physical kappa",
     frac_b < 1e-10,
     sprintf("max|B-mode|/max|kappa| = %.2e", frac_b))
}

# ---- T07: SMPy exact formula ported to R -----------------------------------
cat("\nT07: SMPy exact formula equivalence\n")
{
  # Port SMPy's KaiserSquiresMapper.create_maps() line-by-line to R.
  # SMPy source (kaiser_squires.py lines 83-99):
  #   k1, k2 = np.meshgrid(fftfreq(npix_ra), fftfreq(npix_dec))
  #   k_squared = k1**2 + k2**2
  #   k_squared = np.where(k_squared==0, eps, k_squared)
  #   kappa_e_hat = (1/k_squared)*((k1**2-k2**2)*g1_hat + 2*k1*k2*g2_hat)
  #   kappa_e = Re(ifft2(kappa_e_hat))
  #
  # Note: SMPy's fftfreq is dimensionless (cycles/pixel).
  # R's grid$kx is in cycles/arcmin = fftfreq_dimensionless / pix.
  # In D(k) the pix factors cancel, so we can use R's frequencies directly.

  N   <- grid$N_pix
  eps <- .Machine$double.eps

  # Dimensionless frequencies (cycles/pixel), matching SMPy exactly
  fftfreq_r <- function(n) {
    f <- 0:(n - 1L)
    ifelse(f <= n %/% 2L, f, f - n) / n
  }
  k1 <- outer(rep(1, N), fftfreq_r(N))   # kx varies along columns
  k2 <- outer(fftfreq_r(N), rep(1, N))   # ky varies along rows

  k_sq        <- k1^2 + k2^2
  k_sq[k_sq == 0] <- eps

  g1_hat <- fft(g1_true)
  g2_hat <- fft(g2_true)

  kappa_e_hat_smpy <- ((k1^2 - k2^2) * g1_hat + 2 * k1 * k2 * g2_hat) / k_sq
  kappa_e_smpy     <- Re(fft(kappa_e_hat_smpy, inverse = TRUE)) / N^2

  # Compare to R's ks_inverse (lambda=0)
  kappa_r <- ks_inverse(g1_true, g2_true, grid, lambda = 0)

  # DC mode: SMPy leaves k=0 with ~0 (eps denominator), R explicitly zeros it.
  # Both should be negligible at DC; compare on interior.
  kappa_r[1, 1]    <- 0
  kappa_e_smpy[1, 1] <- 0

  max_diff <- max(abs(kappa_r[idx, idx] - kappa_e_smpy[idx, idx]))
  rel_diff <- max_diff / (max(abs(kappa_e_smpy[idx, idx])) + 1e-30)

  ok("T07a: Max absolute difference (interior) < 1e-14",
     max_diff < 1e-14,
     sprintf("max|R - SMPy| = %.2e", max_diff))

  ok("T07b: Relative difference (interior) < 1e-12",
     rel_diff < 1e-12,
     sprintf("rel diff = %.2e", rel_diff))

  # Also verify the B-mode formula
  kappa_b_hat_smpy <- ((k1^2 - k2^2) * g2_hat - 2 * k1 * k2 * g1_hat) / k_sq
  kappa_b_smpy     <- Re(fft(kappa_b_hat_smpy, inverse = TRUE)) / N^2
  ok("T07c: SMPy B-mode of E-only kappa < 1e-12",
     max(abs(kappa_b_smpy[idx, idx])) < 1e-12,
     sprintf("max|B_smpy| = %.2e", max(abs(kappa_b_smpy[idx, idx]))))
}

# ---- T08: Known analytical solution (single-frequency kappa) ---------------
cat("\nT08: Analytical solution for sinusoidal kappa\n")
{
  # kappa(x,y) = cos(2*pi*m*x/L) is a pure Fourier mode.
  # Its KS shear in Fourier space has the analytical form:
  #   gamma_hat(k) = D(k) * kappa_hat(k)
  # For kappa = cos(2*pi*m*x/L), kappa_hat has delta spikes at +-km.
  # We verify that ks_forward and ks_inverse preserve this exactly.

  m_mode <- 3L   # spatial frequency in grid units
  L      <- grid$L
  X      <- outer(rep(1, N), grid$x)
  kappa_sin <- cos(2 * pi * m_mode * X / L)

  gamma_sin  <- ks_forward(kappa_sin, grid)
  kappa_back <- ks_inverse(gamma_sin$gamma1, gamma_sin$gamma2, grid, lambda = 0)

  # After round-trip, should recover kappa_sin (up to DC mode / mean)
  diff_sin   <- kappa_back - kappa_sin
  diff_sin   <- diff_sin - mean(diff_sin)   # remove mean (DC ambiguity)
  rel_sin    <- sqrt(sum(diff_sin[idx, idx]^2)) / sqrt(sum(kappa_sin[idx, idx]^2))

  ok("T08: Single-frequency round-trip L2 error < 1e-10",
     rel_sin < 1e-10,
     sprintf("rel L2 = %.2e", rel_sin))
}

# ---- T09: DC mode (mean kappa) does not affect shear -----------------------
cat("\nT09: DC mode (mean kappa) does not contaminate shear\n")
{
  # Adding a constant to kappa shifts the mean, which maps to k=0.
  # Since D(k=0) = 0, the forward shear should be unaffected.
  c_offset      <- 5.0
  kappa_shifted <- kappa_true + c_offset
  gamma_shifted <- ks_forward(kappa_shifted, grid)

  err_g1 <- max(abs(gamma_shifted$gamma1 - g1_true))
  err_g2 <- max(abs(gamma_shifted$gamma2 - g2_true))
  ok("T09a: Adding constant to kappa does not change gamma1",
     err_g1 < 1e-14,
     sprintf("max|delta gamma1| = %.2e", err_g1))
  ok("T09b: Adding constant to kappa does not change gamma2",
     err_g2 < 1e-14,
     sprintf("max|delta gamma2| = %.2e", err_g2))

  # Correspondingly, ks_inverse always returns zero-mean kappa (DC zeroed)
  kappa_back <- ks_inverse(g1_true, g2_true, grid, lambda = 0)
  ok("T09c: ks_inverse zeros DC => mean(kappa_hat) is zero",
   abs(mean(kappa_back)) < 1e-12,
   sprintf("|mean(kappa_hat)| = %.2e", abs(mean(kappa_back))))
}

# ---- T10: Peak recovery ----------------------------------------------------
cat("\nT10: Peak recovery (single cluster)\n")
{
  kappa_hat  <- ks_inverse(g1_true, g2_true, grid, lambda = 0)
  true_peak  <- which(kappa_true == max(kappa_true), arr.ind = TRUE)
  recon_peak <- which(kappa_hat  == max(kappa_hat),  arr.ind = TRUE)

  # Peak should be at the same pixel (or within 1 pixel)
  dist <- sqrt((true_peak[1] - recon_peak[1])^2 +
                 (true_peak[2] - recon_peak[2])^2)
  ok("T10a: Peak pixel within 1 pixel of truth",
     dist <= 1.0,
     sprintf("pixel dist = %.2f  true=(%d,%d)  recon=(%d,%d)",
             dist, true_peak[1], true_peak[2],
             recon_peak[1], recon_peak[2]))

  # Peak value: noiseless reconstruction should match to 1e-10
  peak_err_rel <- abs(max(kappa_hat) - max(kappa_true)) / max(kappa_true)
  kappa_hat_dm  <- kappa_hat  - mean(kappa_hat[idx, idx])
  kappa_true_dm <- kappa_true - mean(kappa_true[idx, idx])
  peak_err_rel  <- abs(max(kappa_hat_dm[idx,idx]) - max(kappa_true_dm[idx,idx])) /
                    max(kappa_true_dm[idx,idx])
  ok("T10b: Peak amplitude recovered to 1e-10 (DC-corrected)",
    peak_err_rel < 1e-10,
    sprintf("rel peak err = %.2e", peak_err_rel))
}

# ---- Summary ----------------------------------------------------------------
cat(sprintf("\n=== Results: %d passed, %d failed ===\n", .pass, .fail))
if (.fail == 0L) {
  cat("All tests PASSED. R KS implementation is analytically validated.\n")
  cat("It is mathematically identical to SMPy's KaiserSquiresMapper (E-mode, lambda=0).\n")
} else {
  cat(sprintf("WARNING: %d test(s) failed. Inspect output above.\n", .fail))
}
