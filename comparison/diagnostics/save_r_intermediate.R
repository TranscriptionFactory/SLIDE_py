#!/usr/bin/env Rscript
#
# Diagnostic script to save R LOVE intermediate results for step-by-step comparison
#
# This runs R LOVE step-by-step and saves each intermediate result to files
# that can be loaded by the Python comparison script.
#
# Usage: Rscript save_r_intermediate.R <data_file> <output_dir> [--mode hetero|homo] [--fixed-delta VALUE]
#

suppressPackageStartupMessages({
  library(igraph)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat("Usage: Rscript save_r_intermediate.R <data_file> <output_dir> [--mode hetero|homo] [--fixed-delta VALUE]\n")
  quit(status = 1)
}

data_file <- args[1]
output_dir <- args[2]
mode <- "hetero"  # Default
fixed_delta <- NULL
seed_value <- 42  # Fixed seed for reproducibility

i <- 3
while (i <= length(args)) {
  if (args[i] == "--mode" && i < length(args)) {
    mode <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--fixed-delta" && i < length(args)) {
    fixed_delta <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--seed" && i < length(args)) {
    seed_value <- as.integer(args[i + 1])
    i <- i + 2
  } else {
    i <- i + 1
  }
}

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Get script directory and source R files
script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
if (length(script_path) == 0) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}

r_dir <- file.path(dirname(dirname(script_dir)), "src", "loveslide", "love_pkg", "R")
cat("Loading R source files from:", r_dir, "\n")

# Source individual R files
source(file.path(r_dir, "Utilities.R"))
source(file.path(r_dir, "Score.R"))
source(file.path(r_dir, "EstPureHomo.R"))
source(file.path(r_dir, "EstPureHetero.R"))
source(file.path(r_dir, "CV.R"))

cat("================================================================================\n")
cat("R LOVE Step-by-Step Diagnostic\n")
cat("================================================================================\n")
cat("Data file:", data_file, "\n")
cat("Output dir:", output_dir, "\n")
cat("Mode:", mode, "\n")
cat("Seed:", seed_value, "\n")
if (!is.null(fixed_delta)) cat("Fixed delta:", fixed_delta, "\n")
cat("================================================================================\n\n")

# Set seed for reproducibility
set.seed(seed_value)

# Load data
cat("Step 0: Loading data...\n")
X_df <- read.csv(data_file, row.names = 1)
X <- as.matrix(X_df)
n <- nrow(X)
p <- ncol(X)

cat("  Data dimensions:", n, "samples x", p, "features\n")

# Save raw input
write.csv(X, file.path(output_dir, "step0_X_input.csv"), row.names = TRUE)
cat("  Saved: step0_X_input.csv\n")

# Step 1: Center the data
cat("\nStep 1: Centering data...\n")
X_centered <- scale(X, TRUE, FALSE)
write.csv(X_centered, file.path(output_dir, "step1_X_centered.csv"), row.names = TRUE)
cat("  Max centering diff:", max(abs(colMeans(X_centered))), "\n")
cat("  Saved: step1_X_centered.csv\n")

if (mode == "hetero") {
  # ============================================================================
  # Heterogeneous case (pure_homo = FALSE)
  # ============================================================================

  # Step 2: Compute correlation matrix
  cat("\nStep 2: Computing correlation matrix...\n")
  R_hat <- cor(X_centered)
  Sigma <- cov(X_centered)

  write.csv(R_hat, file.path(output_dir, "step2_R_corr.csv"), row.names = FALSE)
  write.csv(Sigma, file.path(output_dir, "step2_Sigma_cov.csv"), row.names = FALSE)
  cat("  R_hat dims:", dim(R_hat), "\n")
  cat("  Saved: step2_R_corr.csv, step2_Sigma_cov.csv\n")

  # Step 3: Compute score matrix
  cat("\nStep 3: Computing score matrix (q=2)...\n")
  score_res <- Score_mat(R_hat, q = 2, exact = FALSE)
  score_mat <- score_res$score
  moments_mat <- score_res$moments

  write.csv(score_mat, file.path(output_dir, "step3_score_mat.csv"), row.names = FALSE)
  write.csv(moments_mat, file.path(output_dir, "step3_moments_M.csv"), row.names = FALSE)

  # Get score statistics
  score_vals <- score_mat[!is.na(score_mat)]
  cat("  Score matrix stats: min=", min(score_vals), ", max=", max(score_vals),
      ", median=", median(score_vals), "\n")
  cat("  Saved: step3_score_mat.csv, step3_moments_M.csv\n")

  # Step 4: Generate/use delta grid and select delta
  cat("\nStep 4: Delta selection...\n")

  if (!is.null(fixed_delta)) {
    delta_min <- fixed_delta
    delta_grid <- fixed_delta
    cat("  Using fixed delta:", fixed_delta, "\n")
  } else {
    # Generate delta grid (simplified version - single run)
    max_pure <- ifelse(n > p, 1.0, 0.8)
    ndelta <- 50

    # Get row minima of score matrix
    row_mins <- apply(score_mat, 1, function(x) min(x[!is.na(x)], na.rm = TRUE))
    row_mins <- row_mins[is.finite(row_mins)]

    delta_max <- quantile(row_mins, max_pure)
    delta_min_val <- min(score_mat, na.rm = TRUE)
    delta_grid <- seq(delta_max, delta_min_val, length.out = ndelta)

    # For deterministic testing, use the median of the grid
    delta_min <- delta_grid[length(delta_grid) %/% 2]  # Middle of grid
    cat("  Generated delta grid: [", min(delta_grid), ", ", max(delta_grid), "]\n")
    cat("  Selected delta_min:", delta_min, "\n")
  }

  write.csv(data.frame(delta = delta_grid), file.path(output_dir, "step4_delta_grid.csv"), row.names = FALSE)
  write.csv(data.frame(delta_min = delta_min), file.path(output_dir, "step4_delta_min.csv"), row.names = FALSE)
  cat("  Saved: step4_delta_grid.csv, step4_delta_min.csv\n")

  # Step 5: Est_Pure - find parallel rows
  cat("\nStep 5: Est_Pure (finding parallel rows)...\n")
  pure_res <- Est_Pure(score_mat, delta_min)

  K_init <- pure_res$K
  I_init <- pure_res$I
  I_part_init <- pure_res$I_part

  cat("  Initial K:", K_init, "\n")
  cat("  Initial |I|:", length(I_init), "\n")
  cat("  First few I:", head(I_init, 20), "...\n")

  write.csv(data.frame(K = K_init), file.path(output_dir, "step5_K_init.csv"), row.names = FALSE)
  write.csv(data.frame(I = I_init), file.path(output_dir, "step5_I_init.csv"), row.names = FALSE)

  # Save I_part as a JSON-like structure
  I_part_df <- do.call(rbind, lapply(seq_along(I_part_init), function(k) {
    data.frame(group = k, idx = I_part_init[[k]])
  }))
  write.csv(I_part_df, file.path(output_dir, "step5_I_part_init.csv"), row.names = FALSE)
  cat("  Saved: step5_K_init.csv, step5_I_init.csv, step5_I_part_init.csv\n")

  # Step 6: Est_BI_C - estimate B, C, Gamma (FIRST CALL - this is the one used!)
  cat("\nStep 6: Est_BI_C (first and only call)...\n")

  if (K_init >= 2) {
    BI_C_res <- Est_BI_C(moments_mat, R_hat, I_part_init, I_init)

    B_hat <- BI_C_res$B
    C_hat <- BI_C_res$C
    B_left_inv <- BI_C_res$B_left_inv
    Gamma_hat <- BI_C_res$Gamma

    cat("  B dims:", dim(B_hat), "\n")
    cat("  C dims:", dim(C_hat), "\n")
    cat("  B_left_inv dims:", dim(B_left_inv), "\n")
    cat("  Gamma stats: min=", min(Gamma_hat), ", max=", max(Gamma_hat), "\n")

    write.csv(B_hat, file.path(output_dir, "step6_B_hat.csv"), row.names = FALSE)
    write.csv(C_hat, file.path(output_dir, "step6_C_hat.csv"), row.names = FALSE)
    write.csv(B_left_inv, file.path(output_dir, "step6_B_left_inv.csv"), row.names = FALSE)
    write.csv(data.frame(Gamma = Gamma_hat), file.path(output_dir, "step6_Gamma.csv"), row.names = FALSE)
    cat("  Saved: step6_B_hat.csv, step6_C_hat.csv, step6_B_left_inv.csv, step6_Gamma.csv\n")

    # Step 7: Re_Est_Pure (updates I_part, but does NOT change BI_C_res!)
    cat("\nStep 7: Re_Est_Pure (updates partition)...\n")
    est_I_updated <- Re_Est_Pure(X_centered, Sigma, moments_mat, I_part_init, Gamma_hat)
    est_I_set_updated <- as.numeric(unlist(est_I_updated))

    K_updated <- length(est_I_updated)
    cat("  Updated K:", K_updated, " (was:", K_init, ")\n")
    cat("  Updated |I|:", length(est_I_set_updated), " (was:", length(I_init), ")\n")

    # Save updated partition
    I_part_updated_df <- do.call(rbind, lapply(seq_along(est_I_updated), function(k) {
      data.frame(group = k, idx = est_I_updated[[k]])
    }))
    write.csv(data.frame(K = K_updated), file.path(output_dir, "step7_K_updated.csv"), row.names = FALSE)
    write.csv(data.frame(I = est_I_set_updated), file.path(output_dir, "step7_I_updated.csv"), row.names = FALSE)
    write.csv(I_part_updated_df, file.path(output_dir, "step7_I_part_updated.csv"), row.names = FALSE)
    cat("  Saved: step7_K_updated.csv, step7_I_updated.csv, step7_I_part_updated.csv\n")

    # IMPORTANT: R does NOT re-call Est_BI_C here!
    # The B_hat, C_hat, Gamma_hat from step 6 are used directly

    # Step 8: Compute final A matrix
    cat("\nStep 8: Computing final A matrix...\n")
    D_Sigma <- diag(Sigma)

    # B_hat <- sqrt(D_Sigma) * B_hat
    B_hat_scaled <- sqrt(D_Sigma) * B_hat
    D_B <- apply(abs(B_hat_scaled), 2, max)
    A_hat <- t(t(B_hat_scaled) / D_B)
    C_hat_final <- D_B * C_hat  # Note: R uses column-wise recycled multiplication

    cat("  A dims:", dim(A_hat), "\n")
    cat("  A stats: min=", min(A_hat), ", max=", max(A_hat), "\n")

    write.csv(B_hat_scaled, file.path(output_dir, "step8_B_scaled.csv"), row.names = FALSE)
    write.csv(data.frame(D_B = D_B), file.path(output_dir, "step8_D_B.csv"), row.names = FALSE)
    write.csv(A_hat, file.path(output_dir, "step8_A_hat.csv"), row.names = FALSE)
    write.csv(C_hat_final, file.path(output_dir, "step8_C_final.csv"), row.names = FALSE)
    cat("  Saved: step8_B_scaled.csv, step8_D_B.csv, step8_A_hat.csv, step8_C_final.csv\n")

    # Step 9: Compute final Gamma
    cat("\nStep 9: Computing final Gamma...\n")
    Gamma_final <- Gamma_hat * D_Sigma
    Gamma_final[Gamma_final < 0] <- 0

    cat("  Gamma stats: min=", min(Gamma_final), ", max=", max(Gamma_final), "\n")

    write.csv(data.frame(Gamma = Gamma_final), file.path(output_dir, "step9_Gamma_final.csv"), row.names = FALSE)
    cat("  Saved: step9_Gamma_final.csv\n")

    # Step 10: Final I_hat and I_hat_part
    cat("\nStep 10: Final pure variable indices...\n")
    I_hat <- est_I_set_updated
    I_hat_part <- FindSignPureNode(I_part_init, Sigma)  # Note: R uses ORIGINAL I_part!

    cat("  Final K:", ncol(A_hat), "\n")
    cat("  Final |I|:", length(I_hat), "\n")

    write.csv(data.frame(I = I_hat), file.path(output_dir, "step10_I_final.csv"), row.names = FALSE)

    # Convert I_hat_part to dataframe
    I_hat_part_df <- do.call(rbind, lapply(seq_along(I_hat_part), function(k) {
      data.frame(group = k, pos = paste(I_hat_part[[k]]$pos, collapse = ","),
                 neg = paste(I_hat_part[[k]]$neg, collapse = ","))
    }))
    write.csv(I_hat_part_df, file.path(output_dir, "step10_I_part_final.csv"), row.names = FALSE)
    cat("  Saved: step10_I_final.csv, step10_I_part_final.csv\n")

  } else {
    cat("  K < 2, skipping Est_BI_C and remaining steps\n")
  }

} else {
  # ============================================================================
  # Homogeneous case (pure_homo = TRUE)
  # ============================================================================
  cat("\nHomogeneous case (pure_homo = TRUE)\n")

  # Step 2: Compute covariance and standard errors
  cat("\nStep 2: Computing covariance and standard errors...\n")
  Sigma <- cov(X_centered)
  se_est <- apply(X_centered, 2, sd)

  write.csv(Sigma, file.path(output_dir, "step2_Sigma_cov.csv"), row.names = FALSE)
  write.csv(data.frame(se = se_est), file.path(output_dir, "step2_se_est.csv"), row.names = FALSE)
  cat("  Saved: step2_Sigma_cov.csv, step2_se_est.csv\n")

  # Step 3: Delta grid
  cat("\nStep 3: Delta grid for homogeneous case...\n")
  if (!is.null(fixed_delta)) {
    deltaGrids <- fixed_delta * sqrt(log(max(p, n)) / n)
    optDelta <- deltaGrids
  } else {
    delta_seq <- seq(0.1, 1.1, 0.1)
    deltaGrids <- delta_seq * sqrt(log(max(p, n)) / n)
    optDelta <- deltaGrids[length(deltaGrids) %/% 2]  # Middle of grid
  }

  write.csv(data.frame(delta = deltaGrids), file.path(output_dir, "step3_deltaGrids.csv"), row.names = FALSE)
  write.csv(data.frame(optDelta = optDelta), file.path(output_dir, "step3_optDelta.csv"), row.names = FALSE)
  cat("  Saved: step3_deltaGrids.csv, step3_optDelta.csv\n")

  # Step 4: EstAI
  cat("\nStep 4: EstAI...\n")
  resultAI <- EstAI(Sigma, optDelta, se_est, merge = FALSE)

  A_hat <- resultAI$AI
  I_hat <- resultAI$pureVec
  I_hat_part <- resultAI$pureSignInd

  cat("  K:", ncol(A_hat), "\n")
  cat("  |I|:", length(I_hat), "\n")

  write.csv(A_hat, file.path(output_dir, "step4_A_hat.csv"), row.names = FALSE)
  write.csv(data.frame(I = I_hat), file.path(output_dir, "step4_I_hat.csv"), row.names = FALSE)
  cat("  Saved: step4_A_hat.csv, step4_I_hat.csv\n")

  # Step 5: EstC
  cat("\nStep 5: EstC...\n")
  C_hat <- EstC(Sigma, A_hat, diagonal = FALSE)

  write.csv(C_hat, file.path(output_dir, "step5_C_hat.csv"), row.names = FALSE)
  cat("  Saved: step5_C_hat.csv\n")

  # Step 6: Gamma
  cat("\nStep 6: Computing Gamma...\n")
  Gamma_hat <- rep(0, p)
  Gamma_hat[I_hat] <- diag(Sigma[I_hat, I_hat]) - diag(A_hat[I_hat,] %*% C_hat %*% t(A_hat[I_hat,]))
  Gamma_hat[Gamma_hat < 0] <- 0

  write.csv(data.frame(Gamma = Gamma_hat), file.path(output_dir, "step6_Gamma.csv"), row.names = FALSE)
  cat("  Saved: step6_Gamma.csv\n")
}

# Summary
cat("\n================================================================================\n")
cat("DIAGNOSTIC COMPLETE\n")
cat("================================================================================\n")
cat("All intermediate results saved to:", output_dir, "\n")
cat("\nNext: Run compare_step_by_step.py with the same data file\n")
