library(magick)
gif <- image_read("midterm_presentation/optimizer_comparison.gif")
dir.create("midterm_presentation/gif_frames", showWarnings = FALSE)
for (i in seq_along(gif)) {
  frame <- image_convert(gif[i], format = "png", depth = 8, colorspace = "sRGB")
  image_write(frame, path = sprintf("midterm_presentation/gif_frames/frame%03d.png", i - 1))
}
cat(sprintf("Extracted %d frames\n", length(gif)))