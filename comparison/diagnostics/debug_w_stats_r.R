#!/usr/bin/env Rscript
# Compare W statistics between R knockoff package and Python implementation

library(knockoff)

# Load data saved from Python
z <- as.matrix(read.csv("debug_z_scaled.csv", header=FALSE))
Xk <- as.matrix(read.csv("debug_knockoffs.csv", header=FALSE))
y <- as.numeric(read.csv("debug_y.csv", header=FALSE)$V1)
W_py <- as.numeric(read.csv("debug_W_py.csv", header=FALSE)$V1)

cat(paste(rep("=", 70), collapse=""), "\n")
cat("R Knockoff W statistics comparison\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

cat("Data shapes:\n")
cat("  Z:", nrow(z), "x", ncol(z), "\n")
cat("  Knockoffs:", nrow(Xk), "x", ncol(Xk), "\n")
cat("  Y:", length(y), "\n")
cat("  Python W:", length(W_py), "\n\n")

# Compute W statistics using R's stat.glmnet_lambdasmax
cat("Computing R W statistics with stat.glmnet_lambdasmax...\n")
W_r <- stat.glmnet_lambdasmax(z, Xk, y)

cat("\nR W statistics summary:\n")
cat("  Range: [", min(W_r), ",", max(W_r), "]\n")
cat("  # positive:", sum(W_r > 0), ", # negative:", sum(W_r < 0), "\n")

cat("\nPython W statistics summary (for reference):\n")
cat("  Range: [", min(W_py), ",", max(W_py), "]\n")
cat("  # positive:", sum(W_py > 0), ", # negative:", sum(W_py < 0), "\n")

cat("\n" , paste(rep("=", 70), collapse=""), "\n")
cat("Comparison:\n")
cat(paste(rep("=", 70), collapse=""), "\n")
cat("  Correlation:", cor(W_r, W_py), "\n")
cat("  Mean absolute difference:", mean(abs(W_r - W_py)), "\n")
cat("  Max absolute difference:", max(abs(W_r - W_py)), "\n")

# Per-feature comparison
cat("\nPer-feature comparison (sorted by R W value):\n")
cat(sprintf("%6s %12s %12s %12s\n", "Z_idx", "W_r", "W_py", "diff"))
ord <- order(W_r, decreasing=TRUE)
for (i in ord) {
    cat(sprintf("%6d %12.6f %12.6f %12.6f\n", i-1, W_r[i], W_py[i], W_r[i] - W_py[i]))
}

# Compute thresholds
cat("\n" , paste(rep("=", 70), collapse=""), "\n")
cat("Threshold comparison (FDR=0.1):\n")
cat(paste(rep("=", 70), collapse=""), "\n")

knockoff_threshold <- function(W, fdr, offset=1) {
    ts <- sort(unique(abs(W)))
    threshold <- Inf
    for (t in ts) {
        num <- offset + sum(W <= -t)
        denom <- max(1, sum(W >= t))
        if (num / denom <= fdr) {
            threshold <- t
            break
        }
    }
    return(threshold)
}

t_r_offset0 <- knockoff_threshold(W_r, 0.1, offset=0)
t_r_offset1 <- knockoff_threshold(W_r, 0.1, offset=1)
t_py_offset0 <- knockoff_threshold(W_py, 0.1, offset=0)
t_py_offset1 <- knockoff_threshold(W_py, 0.1, offset=1)

cat("  R (offset=0): threshold=", t_r_offset0, ", selected=", sum(W_r >= t_r_offset0), "\n")
cat("  R (offset=1): threshold=", t_r_offset1, ", selected=", sum(W_r >= t_r_offset1), "\n")
cat("  Python (offset=0): threshold=", t_py_offset0, ", selected=", sum(W_py >= t_py_offset0), "\n")
cat("  Python (offset=1): threshold=", t_py_offset1, ", selected=", sum(W_py >= t_py_offset1), "\n")

if (t_r_offset0 < Inf) {
    sel <- which(W_r >= t_r_offset0) - 1
    cat("\n  R selections (offset=0): Z", paste(sel, collapse=", Z"), "\n")
}

# Save R W stats for further analysis
write.csv(data.frame(W_r=W_r), "debug_W_r.csv", row.names=FALSE)
cat("\nSaved R W stats to debug_W_r.csv\n")
