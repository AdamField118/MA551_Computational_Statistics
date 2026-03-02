# ============================================================
# Synchronized Optimizer Comparison: SGD, Momentum, Adam
# MA 551 -- Adam: Adaptive Moment Estimation
#
# Output:
#   optimizer_comparison.png  -- synchronized static summary panel
#   optimizer_trajectories.png -- trajectory-only panel for slide 9
#   optimizer_comparison.gif  -- animated version (requires gifski)
#
# Packages: ggplot2, gganimate, patchwork, gifski
# ============================================================

library(ggplot2)
library(gganimate)
library(patchwork)

# ── 1. Objective function ─────────────────────────────────────────────────────
# Anisotropic quadratic: f(theta) = (theta1^2)/2 + (theta2^2)/50
# Gradient: (theta1, theta2/25)
# Minimum at origin.
# The strong anisotropy (curvatures 1 vs 0.04) makes SGD oscillate badly
# and makes Adam's per-coordinate scaling visually obvious.

f      <- function(t1, t2) t1^2 / 2 + t2^2 / 200
grad_f <- function(t1, t2) c(t1, t2 / 100)

# ── 2. Stochastic gradient (add isotropic noise) ──────────────────────────────
stoch_grad <- function(t1, t2, noise_sd = 0.15) {
  g <- grad_f(t1, t2)
  g + rnorm(2, sd = noise_sd)
}

# ── 3. Optimizer implementations ─────────────────────────────────────────────
run_sgd <- function(theta0, eta, n_iter, noise_sd = 0.15, seed = 42) {
  set.seed(seed)
  theta <- theta0
  history <- data.frame(iter = 0, t1 = theta[1], t2 = theta[2],
                        loss = f(theta[1], theta[2]), method = "SGD")
  for (t in seq_len(n_iter)) {
    g <- stoch_grad(theta[1], theta[2], noise_sd)
    theta <- theta - eta * g
    history <- rbind(history,
      data.frame(iter = t, t1 = theta[1], t2 = theta[2],
                 loss = f(theta[1], theta[2]), method = "SGD"))
  }
  history
}

run_momentum <- function(theta0, eta, gamma = 0.9, n_iter, noise_sd = 0.15, seed = 42) {
  set.seed(seed)
  theta <- theta0
  v     <- c(0, 0)
  history <- data.frame(iter = 0, t1 = theta[1], t2 = theta[2],
                        loss = f(theta[1], theta[2]), method = "Momentum")
  for (t in seq_len(n_iter)) {
    g <- stoch_grad(theta[1], theta[2], noise_sd)
    v <- gamma * v + eta * g
    theta <- theta - v
    history <- rbind(history,
      data.frame(iter = t, t1 = theta[1], t2 = theta[2],
                 loss = f(theta[1], theta[2]), method = "Momentum"))
  }
  history
}

run_adam <- function(theta0, eta, beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
                     n_iter, noise_sd = 0.15, seed = 42) {
  set.seed(seed)
  theta <- theta0
  m <- c(0, 0)
  v <- c(0, 0)
  history <- data.frame(iter = 0, t1 = theta[1], t2 = theta[2],
                        loss = f(theta[1], theta[2]), method = "Adam")
  for (t in seq_len(n_iter)) {
    g  <- stoch_grad(theta[1], theta[2], noise_sd)
    m  <- beta1 * m + (1 - beta1) * g
    v  <- beta2 * v + (1 - beta2) * g^2
    mh <- m / (1 - beta1^t)
    vh <- v / (1 - beta2^t)
    theta <- theta - eta * mh / (sqrt(vh) + eps)
    history <- rbind(history,
      data.frame(iter = t, t1 = theta[1], t2 = theta[2],
                 loss = f(theta[1], theta[2]), method = "Adam"))
  }
  history
}

# ── 4. Run all optimizers ─────────────────────────────────────────────────────
theta0   <- c(3.0, 12.0)
n_iter   <- 80

sgd_hist  <- run_sgd(theta0,      eta = 1.80, n_iter = n_iter, noise_sd = 0.02)
mom_hist  <- run_momentum(theta0, eta = 0.15, gamma = 0.85, n_iter = n_iter, noise_sd = 0.02)
adam_hist <- run_adam(theta0,     eta = 0.50, n_iter = n_iter, noise_sd = 0.02)

all_hist <- rbind(sgd_hist, mom_hist, adam_hist)
all_hist$method <- factor(all_hist$method, levels = c("SGD", "Momentum", "Adam"))

# ── 5. Contour grid ───────────────────────────────────────────────────────────
grid <- expand.grid(t1 = seq(-5, 5.5, length.out = 200),
                    t2 = seq(-10, 10, length.out = 200))
grid$loss <- f(grid$t1, grid$t2)

method_colors <- c("SGD" = "#E64B35", "Momentum" = "#4DBBD5", "Adam" = "#00A087")

# ── 6. Static summary panel (optimizer_comparison.png) ───────────────────────
# Left: full trajectories. Right: loss curves.

p_traj <- ggplot() +
  geom_contour(data = grid, aes(x = t1, y = t2, z = loss),
               color = "grey75", bins = 18, linewidth = 0.3) +
  geom_path(data = all_hist,
            aes(x = t1, y = t2, color = method, group = method),
            linewidth = 0.8, alpha = 0.85) +
  geom_point(data = subset(all_hist, iter == 0)[1, ],
             aes(x = t1, y = t2), shape = 4, size = 3, color = "black") +
  geom_point(data = subset(all_hist, iter == n_iter),
             aes(x = t1, y = t2, color = method), size = 2.5, shape = 19) +
  scale_color_manual(values = method_colors) +
  labs(x = expression(theta[1]), y = expression(theta[2]),
       title = "Parameter space trajectories", color = NULL) +
  coord_fixed() +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title = element_text(size = 11, face = "bold"))

p_loss <- ggplot(all_hist, aes(x = iter, y = log10(loss + 1e-8),
                               color = method, group = method)) +
  geom_line(linewidth = 0.9, alpha = 0.9) +
  scale_color_manual(values = method_colors) +
  labs(x = "Iteration", y = expression(log[10](f(theta))),
       title = "Loss vs. iteration", color = NULL) +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom",
        plot.title = element_text(size = 11, face = "bold"))

p_combined <- p_traj + p_loss +
  plot_layout(widths = c(1, 1)) &
  theme(legend.position = "bottom")

ggsave("midterm_presentation/optimizer_comparison.png", p_combined,
       width = 10, height = 4.5, dpi = 180, bg = "white")
cat("Saved: optimizer_comparison.png\n")

# ── 7. Trajectory-only panel (optimizer_trajectories.png) ────────────────────
# Used on the Interpretation slide (slide 9).
ggsave("midterm_presentation/optimizer_trajectories.png", p_traj + theme(legend.position = "right"),
       width = 5.5, height = 4.5, dpi = 180, bg = "white")
cat("Saved: optimizer_trajectories.png\n")

# ── 8. Animated version (optimizer_comparison.gif) ───────────────────────────
# Requires the gifski package: install.packages("gifski")
# The animation reveals trajectories step-by-step in sync with the loss curve.

p_traj_anim <- ggplot() +
  geom_contour(data = grid, aes(x = t1, y = t2, z = loss),
               color = "grey75", bins = 18, linewidth = 0.3) +
  geom_path(data = all_hist,
            aes(x = t1, y = t2, color = method, group = method),
            linewidth = 0.8, alpha = 0.85) +
  geom_point(data = all_hist,
             aes(x = t1, y = t2, color = method, group = method),
             size = 2.5) +
  scale_color_manual(values = method_colors) +
  labs(x = expression(theta[1]), y = expression(theta[2]),
       title = "Iteration: {frame_along}", color = NULL) +
  coord_fixed() +
  theme_classic(base_size = 11) +
  theme(legend.position = "bottom") +
  transition_reveal(iter) +         # <-- reveal trajectory step-by-step
  ease_aes("linear")

p_loss_anim <- ggplot(all_hist, aes(x = iter, y = log10(loss + 1e-8),
                                    color = method, group = method)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  scale_color_manual(values = method_colors) +
  labs(x = "Iteration", y = expression(log[10](f(theta))), color = NULL) +
  theme_classic(base_size = 11) +
  theme(legend.position = "bottom") +
  transition_reveal(iter) +
  ease_aes("linear")

# Render each animation separately, then stitch side-by-side with magick.
# If magick is unavailable, save individually.

cat("Rendering trajectory animation...\n")
anim_traj <- animate(p_traj_anim, nframes = n_iter + 1, fps = 10,
                     width = 480, height = 420, renderer = gifski_renderer())

cat("Rendering loss animation...\n")
anim_loss <- animate(p_loss_anim, nframes = n_iter + 1, fps = 10,
                     width = 480, height = 420, renderer = gifski_renderer())

# Stitch side-by-side using magick (install.packages("magick"))
if (requireNamespace("magick", quietly = TRUE)) {
  library(magick)
  frames_traj <- image_read(anim_traj)
  frames_loss <- image_read(anim_loss)
  combined    <- image_append(c(frames_traj, frames_loss))  # stacks vertically
  # For side-by-side, append frame-by-frame:
  n_frames <- length(frames_traj)
  side_by_side <- vector("list", n_frames)
  for (i in seq_len(n_frames)) {
    side_by_side[[i]] <- image_append(c(frames_traj[i], frames_loss[i]))
  }
  anim_combined <- image_join(side_by_side)
  image_write_gif(anim_combined, "midterm_presentation/optimizer_comparison.gif", delay = 1/10)
  cat("Saved: optimizer_comparison.gif (side-by-side)\n")
} else {
  # Fallback: save individually
  anim_save("optimizer_trajectories.gif", anim_traj)
  anim_save("optimizer_loss.gif",         anim_loss)
  cat("magick not found. Saved individual GIFs.\n")
  cat("Install magick for the stitched side-by-side GIF.\n")
}