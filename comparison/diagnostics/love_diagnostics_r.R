#!/usr/bin/env Rscript
#' LOVE Diagnostic Script - R Implementation
#'
#' Runs R LOVE/SLIDE with fixed delta (no CV) and saves all intermediate values
#' for comparison with Python implementation.
#'
#' Usage:
#'   Rscript love_diagnostics_r.R --data_path /path/to/data.csv --delta 0.05 --output_dir ./diagnostics_output_r

library(argparse)

# Get the directory of this script to source LOVE-SLIDE functions
# Use commandArgs to get script path when run via Rscript
script_path <- NULL
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- sub("--file=", "", file_arg)
  script_dir <- dirname(normalizePath(script_path))
} else {
  script_dir <- "."
}

# Source all LOVE-SLIDE R functions
love_slide_dir <- file.path(dirname(dirname(script_dir)), "src", "loveslide", "LOVE-SLIDE")
if (!dir.exists(love_slide_dir)) {
  # Try alternate path
  love_slide_dir <- "/ix/djishnu/Aaron/1_general_use/SLIDE_py/src/loveslide/LOVE-SLIDE"
}

cat("Sourcing LOVE-SLIDE functions from:", love_slide_dir, "\n")
for (f in list.files(love_slide_dir, pattern = "\\.R$", full.names = TRUE)) {
  source(f)
}


run_love_diagnostics <- function(X, delta = 0.05, thresh_fdr = 0.2, output_dir = "./love_diagnostics_r") {
  #' Run LOVE with diagnostic output at each step.
  #'
  #' @param X Data matrix (n x p)
  #' @param delta Fixed delta value (no CV)
  #' @param thresh_fdr FDR threshold for correlation matrix
  #' @param output_dir Directory to save diagnostic outputs

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  n <- nrow(X)
  p <- ncol(X)
  cat(sprintf("Data shape: n=%d, p=%d\n", n, p))

  # Step 1: Preprocessing (center and scale)
  cat("\n=== Step 1: Preprocessing ===\n")
  X_scaled <- scale(X, center = TRUE, scale = TRUE)

  write.csv(X_scaled, file.path(output_dir, "01_X_scaled.csv"), row.names = FALSE)
  cat(sprintf("X_scaled: min=%.6f, max=%.6f, mean=%.6f\n",
              min(X_scaled), max(X_scaled), mean(X_scaled)))

  # Step 2: Compute correlation matrix
  cat("\n=== Step 2: Correlation Matrix ===\n")
  R_hat <- cor(X_scaled)
  Sigma <- cov(X_scaled)

  write.csv(R_hat, file.path(output_dir, "02_R_hat_raw.csv"), row.names = FALSE)
  write.csv(Sigma, file.path(output_dir, "02_Sigma_raw.csv"), row.names = FALSE)
  cat(sprintf("R_hat: min=%.6f, max=%.6f\n", min(R_hat), max(R_hat)))
  cat(sprintf("Sigma: min=%.6f, max=%.6f\n", min(Sigma), max(Sigma)))

  # Step 3: FDR thresholding
  cat("\n=== Step 3: FDR Thresholding ===\n")

  # Use R SLIDE's threshSigma function
  # First compute p-values manually
  sigma_pvals <- apply(R_hat, c(1, 2), function(x) { corrToP(x, n = n)$p })
  write.csv(sigma_pvals, file.path(output_dir, "03_pvalues_raw.csv"), row.names = FALSE)

  # Benjamini-Hochberg correction
  p_adjusted <- matrix(p.adjust(sigma_pvals, method = 'BH'), nrow = nrow(sigma_pvals))
  write.csv(p_adjusted, file.path(output_dir, "03_pvalues_adjusted.csv"), row.names = FALSE)

  # Apply threshold
  kept_entries <- apply(p_adjusted, c(1, 2), function(x) { ifelse(x <= thresh_fdr, 1, 0) })
  R_thresh <- R_hat * kept_entries
  # Note: R SLIDE applies thresholding to sigma (correlation), not Sigma (covariance) directly
  # But for fair comparison, let's also threshold the covariance
  Sigma_thresh <- Sigma * kept_entries

  write.csv(kept_entries, file.path(output_dir, "03_kept_entries.csv"), row.names = FALSE)
  write.csv(R_thresh, file.path(output_dir, "03_R_hat_thresholded.csv"), row.names = FALSE)
  write.csv(Sigma_thresh, file.path(output_dir, "03_Sigma_thresholded.csv"), row.names = FALSE)

  n_kept <- sum(kept_entries) - p  # Exclude diagonal
  n_total <- p * (p - 1)
  cat(sprintf("Kept %d/%d off-diagonal entries (%.1f%%)\n", n_kept, n_total, 100 * n_kept / n_total))

  # Step 4: Scale delta
  cat("\n=== Step 4: Delta Scaling ===\n")
  se_est <- apply(X_scaled, 2, sd)
  delta_scaled <- delta * sqrt(log(max(p, n)) / n)

  write.csv(se_est, file.path(output_dir, "04_se_est.csv"), row.names = FALSE)
  cat(sprintf("delta_raw=%.6f, delta_scaled=%.6f\n", delta, delta_scaled))
  cat(sprintf("se_est: min=%.6f, max=%.6f, mean=%.6f\n", min(se_est), max(se_est), mean(se_est)))

  writeLines(c(
    sprintf("delta_raw=%f", delta),
    sprintf("delta_scaled=%f", delta_scaled),
    sprintf("log(max(p,n))/n = %f", log(max(p, n)) / n),
    sprintf("sqrt(log(max(p,n))/n) = %f", sqrt(log(max(p, n)) / n))
  ), file.path(output_dir, "04_delta_values.txt"))

  # Step 5: Find row maxima
  # Note: R SLIDE uses sigma (correlation), so we use R_thresh here
  # But Python uses Sigma, so let's use Sigma_thresh for consistency
  cat("\n=== Step 5: Find Row Maxima ===\n")

  # Use correlation matrix as R SLIDE does (sigma is actually R_hat in getLatentFactors.R)
  abs_sigma <- abs(R_thresh)  # R SLIDE uses correlation matrix
  diag(abs_sigma) <- 0

  result_max <- findRowMax(abs_sigma)
  max_vals <- result_max$max_vals
  max_inds <- result_max$max_inds

  write.csv(abs_sigma, file.path(output_dir, "05_off_Sigma.csv"), row.names = FALSE)
  write.csv(max_vals, file.path(output_dir, "05_row_max_values.csv"), row.names = FALSE)
  write.csv(max_inds, file.path(output_dir, "05_row_max_indices.csv"), row.names = FALSE)
  cat(sprintf("Ms: min=%.6f, max=%.6f\n", min(max_vals), max(max_vals)))

  # Step 6: Find pure nodes
  cat("\n=== Step 6: Find Pure Nodes ===\n")
  result_pure <- findPureNode(
    abs_sigma = abs_sigma,
    delta = delta_scaled,
    max_vals = max_vals,
    max_inds = max_inds,
    se_est = se_est
  )
  pure_list <- result_pure$pure_list
  pure_vec <- as.vector(unlist(result_pure$pure_vec))

  sink(file.path(output_dir, "06_pure_indices.txt"))
  cat(sprintf("Number of groups: %d\n", length(pure_list)))
  cat(sprintf("Total pure variables: %d\n\n", length(pure_vec)))
  for (i in seq_along(pure_list)) {
    cat(sprintf("Group %d: %s\n", i, paste(sort(pure_list[[i]]), collapse = ", ")))
  }
  cat(sprintf("\nAll pure indices: %s\n", paste(sort(pure_vec), collapse = ", ")))
  sink()

  if (length(pure_vec) > 0) {
    write.csv(sort(pure_vec), file.path(output_dir, "06_pure_vec.csv"), row.names = FALSE)
  }

  cat(sprintf("Found %d groups with %d pure variables\n", length(pure_list), length(pure_vec)))

  # Step 7: Find sign partition
  cat("\n=== Step 7: Sign Partition ===\n")
  signed_pure_list <- findSignPureNode(pure_list = pure_list, sigma = R_thresh)

  sink(file.path(output_dir, "07_sign_partition.txt"))
  for (i in seq_along(signed_pure_list)) {
    group <- signed_pure_list[[i]]
    cat(sprintf("Group %d: pos=%s, neg=%s\n", i,
                paste(group$pos, collapse = ", "),
                paste(group$neg, collapse = ", ")))
  }
  sink()

  # Step 8: Recover AI matrix
  cat("\n=== Step 8: Recover AI Matrix ===\n")
  AI <- recoverAI(pure_list = signed_pure_list, p = p)
  K <- ncol(AI)

  write.csv(AI, file.path(output_dir, "08_AI_matrix.csv"), row.names = FALSE)
  cat(sprintf("AI shape: %d x %d (K=%d)\n", nrow(AI), ncol(AI), K))

  # Step 9: Estimate C (covariance of Z)
  cat("\n=== Step 9: Estimate C Matrix ===\n")
  C_hat <- estC(sigma = R_thresh, AI = AI)

  write.csv(C_hat, file.path(output_dir, "09_C_hat.csv"), row.names = FALSE)
  cat(sprintf("C_hat shape: %d x %d\n", nrow(C_hat), ncol(C_hat)))
  cat(sprintf("C_hat diagonal: %s\n", paste(round(diag(C_hat), 6), collapse = ", ")))

  # Step 10: Estimate Gamma (error variance)
  cat("\n=== Step 10: Estimate Gamma ===\n")
  Gamma_hat <- rep(0, p)
  if (length(pure_vec) > 0) {
    Gamma_hat[pure_vec] <- diag(R_thresh[pure_vec, pure_vec]) -
      diag(AI[pure_vec, , drop = FALSE] %*% C_hat %*% t(AI[pure_vec, , drop = FALSE]))
  }
  Gamma_hat[Gamma_hat < 0] <- 0

  write.csv(Gamma_hat, file.path(output_dir, "10_Gamma_hat.csv"), row.names = FALSE)
  cat(sprintf("Gamma_hat: min=%.6f, max=%.6f\n", min(Gamma_hat), max(Gamma_hat)))

  # Save summary
  cat("\n=== Summary ===\n")
  summary_list <- list(
    n = n,
    p = p,
    delta_raw = delta,
    delta_scaled = delta_scaled,
    thresh_fdr = thresh_fdr,
    K = K,
    n_pure_variables = length(pure_vec),
    n_groups = length(pure_list),
    pure_indices = pure_list,
    pure_vec = pure_vec
  )

  saveRDS(summary_list, file.path(output_dir, "summary.rds"))

  sink(file.path(output_dir, "summary.txt"))
  for (nm in names(summary_list)) {
    val <- summary_list[[nm]]
    if (is.list(val)) {
      cat(sprintf("%s: [list with %d elements]\n", nm, length(val)))
    } else if (length(val) > 10) {
      cat(sprintf("%s: [vector of length %d]\n", nm, length(val)))
    } else {
      cat(sprintf("%s: %s\n", nm, paste(val, collapse = ", ")))
    }
  }
  sink()

  cat(sprintf("K (number of latent factors): %d\n", K))
  cat(sprintf("Pure variables: %d\n", length(pure_vec)))
  cat(sprintf("Groups: %d\n", length(pure_list)))

  return(summary_list)
}


# Main execution
main <- function() {
  parser <- ArgumentParser(description = "LOVE Diagnostic Script - R")
  parser$add_argument("--data_path", type = "character", required = TRUE,
                      help = "Path to data CSV (samples x features)")
  parser$add_argument("--delta", type = "double", default = 0.05,
                      help = "Fixed delta value (no CV)")
  parser$add_argument("--thresh_fdr", type = "double", default = 0.2,
                      help = "FDR threshold")
  parser$add_argument("--output_dir", type = "character", default = "./love_diagnostics_r",
                      help = "Output directory")

  args <- parser$parse_args()

  # Load data
  cat(sprintf("Loading data from %s\n", args$data_path))
  df <- read.csv(args$data_path, row.names = 1, check.names = FALSE)
  X <- as.matrix(df)
  cat(sprintf("Loaded data: %d x %d\n", nrow(X), ncol(X)))

  # Run diagnostics
  run_love_diagnostics(
    X = X,
    delta = args$delta,
    thresh_fdr = args$thresh_fdr,
    output_dir = args$output_dir
  )

  cat(sprintf("\nDiagnostic outputs saved to: %s\n", args$output_dir))
}

# Run if executed as script
if (!interactive()) {
  main()
}
