#!/usr/bin/env Rscript
#
# R script to run SLIDE on input data and save outputs for comparison with Python
#
# Usage: Rscript run_slide_R.R <x_file> <y_file> [tag] [options]
#
# Arguments:
#   x_file          Path to X matrix CSV (samples as rows, features as columns)
#   y_file          Path to Y vector CSV (samples as rows, single column)
#   tag             Optional run name tag (default: basename of x file)
#   --delta VALUE   Delta value (default: 0.1)
#   --lambda VALUE  Lambda value (default: 0.5)
#   --spec VALUE    Specificity threshold (default: 0.1)
#   --fdr VALUE     FDR threshold (default: 0.1)
#   --niter VALUE   Number of SLIDE iterations (default: 500)
#
# Examples:
#   Rscript run_slide_R.R /path/to/X.csv /path/to/Y.csv
#   Rscript run_slide_R.R /path/to/X.csv /path/to/Y.csv my_experiment --delta 0.1 --lambda 0.5
#

suppressPackageStartupMessages({
  library(foreach)
  library(doParallel)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat("Usage: Rscript run_slide_R.R <x_file> <y_file> [tag] [options]\n")
  cat("\nArguments:\n")
  cat("  x_file          Path to X matrix CSV (samples as rows, features as columns)\n")
  cat("  y_file          Path to Y vector CSV (samples as rows, single column)\n")
  cat("  tag             Optional run name tag (default: basename of x file)\n")
  cat("  --delta VALUE   Delta value (default: 0.1)\n")
  cat("  --lambda VALUE  Lambda value (default: 0.5)\n")
  cat("  --spec VALUE    Specificity threshold (default: 0.1)\n")
  cat("  --fdr VALUE     FDR threshold (default: 0.1)\n")
  cat("  --niter VALUE   Number of SLIDE iterations (default: 500)\n")
  cat("\nExamples:\n")
  cat("  Rscript run_slide_R.R /path/to/X.csv /path/to/Y.csv my_experiment\n")
  quit(status = 1)
}

x_file <- args[1]
y_file <- args[2]

# Check if data files exist
if (!file.exists(x_file)) {
  cat("ERROR: X data file not found:", x_file, "\n")
  quit(status = 1)
}
if (!file.exists(y_file)) {
  cat("ERROR: Y data file not found:", y_file, "\n")
  quit(status = 1)
}

# Parse optional arguments
tag <- NULL
delta <- 0.1
lambda <- 0.5
spec <- 0.1
fdr <- 0.1
niter <- 500
thresh_fdr <- 0.2

i <- 3
while (i <= length(args)) {
  if (args[i] == "--delta" && i < length(args)) {
    delta <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--lambda" && i < length(args)) {
    lambda <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--spec" && i < length(args)) {
    spec <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--fdr" && i < length(args)) {
    fdr <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--niter" && i < length(args)) {
    niter <- as.integer(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--thresh-fdr" && i < length(args)) {
    thresh_fdr <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (is.null(tag) && !startsWith(args[i], "--")) {
    tag <- args[i]
    i <- i + 1
  } else {
    i <- i + 1
  }
}

# Default tag to basename without extension
if (is.null(tag)) {
  tag <- tools::file_path_sans_ext(basename(x_file))
}

# Get script directory for sourcing R files
script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
if (length(script_path) == 0) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}

# Source all SLIDE R files
slide_r_dir <- "/ix/djishnu/Aaron/1_general_use/SLIDE/R"

r_files <- c(
  "Rutils.R", "LP.R", "SLIDE.R", "adjustSign.R", "calcDefaultFsize.R",
  "calcZMatrix.R", "checkElement.R", "clustMem.R", "cvDelta.R",
  "dantzig.R", "estAI.R", "estAJDant.R", "estBeta.R", "estC.R",
  "estOmega.R", "estSigmaTJ.R", "findFrequent.R", "findOptIter.R",
  "findPureNode.R", "findRowMax.R", "findRowMaxInd.R", "findSignPureNode.R",
  "getFolds.R", "getLatentFactors.R", "getTopFeatures.R", "interUnion.R",
  "interactionSLIDE.R", "makeHeatmap.R", "makePosDef.R", "marginalSLIDE.R",
  "merge.R", "offSum.R", "pairwiseInteractions.R", "permA.R", "predZ.R",
  "prediction.R", "pureRowInd.R", "recoverAI.R", "recoverGroup.R",
  "runSLIDE.R", "selectFrequent.R", "selectLongest.R", "selectShortFreq.R",
  "signPerm.R", "singleton.R", "solveRow.R", "calcControlPerformance.R",
  "sampleCV.R"
)

cat("Loading R source files from:", slide_r_dir, "\n")
for (f in r_files) {
  fpath <- file.path(slide_r_dir, f)
  if (file.exists(fpath)) {
    source(fpath)
  } else {
    cat("WARNING: R source file not found:", fpath, "\n")
  }
}

# Create output directory
output_dir <- file.path(script_dir, "outputs", tag)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat(rep("=", 60), "\n", sep = "")
cat("SLIDE R Analysis\n")
cat("  X file: ", x_file, "\n", sep = "")
cat("  Y file: ", y_file, "\n", sep = "")
cat("  Tag: ", tag, "\n", sep = "")
cat("  Output dir: ", output_dir, "\n", sep = "")
cat("  Delta: ", delta, "\n", sep = "")
cat("  Lambda: ", lambda, "\n", sep = "")
cat("  Spec: ", spec, "\n", sep = "")
cat("  FDR: ", fdr, "\n", sep = "")
cat("  Niter: ", niter, "\n", sep = "")
cat("  Thresh FDR: ", thresh_fdr, "\n", sep = "")
cat(rep("=", 60), "\n", sep = "")

# Setup parallel backend
n_cores <- min(parallel::detectCores() - 1, 10)
cl <- makeCluster(n_cores)
registerDoParallel(cl)
cat("Using", n_cores, "cores for parallel processing\n")

# Load data
cat("\nLoading data...\n")
X <- as.matrix(read.csv(x_file, row.names = 1, check.names = FALSE))
Y <- as.matrix(read.csv(y_file, row.names = 1))

# Replace spaces in feature names with underscores (matching R SLIDE behavior)
colnames(X) <- gsub(" ", "_", colnames(X))

n <- nrow(X)
p <- ncol(X)

cat("X dimensions: ", n, " samples x ", p, " features\n", sep = "")
cat("Y dimensions: ", nrow(Y), " samples x ", ncol(Y), " columns\n", sep = "")

# Check dimensions
if (n <= p) {
  cat("\nWARNING: n (", n, ") <= p (", p, ")\n", sep = "")
  cat("SLIDE requires more samples than features for reliable estimation.\n\n")
}

# Standardize X
X_std <- scale(X, TRUE, TRUE)

# Save standardized data for Python to use (ensures same data)
write.csv(X, file.path(output_dir, paste0(tag, "_X_matrix.csv")), row.names = TRUE)
write.csv(Y, file.path(output_dir, paste0(tag, "_Y_matrix.csv")), row.names = TRUE)
write.csv(X_std, file.path(output_dir, paste0(tag, "_X_std_matrix.csv")), row.names = TRUE)

# ============================================================================
# Step 1: Get Latent Factors
# ============================================================================
cat("\n", rep("-", 60), "\n", sep = "")
cat("Step 1: Getting Latent Factors (getLatentFactors)\n")
cat(rep("-", 60), "\n", sep = "")

set.seed(123)
t_start <- Sys.time()

all_latent_factors <- tryCatch({
  getLatentFactors(
    x = X,
    x_std = X_std,
    y = Y,
    sigma = NULL,
    delta = delta,
    lambda = lambda,
    rep_cv = 50,
    alpha_level = 0.05,
    thresh_fdr = thresh_fdr,
    out_path = NULL
  )
}, error = function(e) {
  cat("ERROR in getLatentFactors:", conditionMessage(e), "\n")
  return(NULL)
})

t_end <- Sys.time()

if (!is.null(all_latent_factors)) {
  cat("\nLatent Factors Results:\n")
  cat("  K (number of LFs): ", all_latent_factors$K, "\n", sep = "")
  cat("  Number of pure variables: ", length(all_latent_factors$I), "\n", sep = "")
  cat("  A shape: ", nrow(all_latent_factors$A), " x ", ncol(all_latent_factors$A), "\n", sep = "")
  cat("  C shape: ", nrow(all_latent_factors$C), " x ", ncol(all_latent_factors$C), "\n", sep = "")
  cat("  Gamma length: ", length(all_latent_factors$Gamma), "\n", sep = "")
  cat("  opt_delta: ", all_latent_factors$opt_delta, "\n", sep = "")
  cat("  opt_lambda: ", all_latent_factors$opt_lambda, "\n", sep = "")
  cat("  Time: ", format(t_end - t_start), "\n", sep = "")

  # Save outputs
  write.csv(all_latent_factors$A, file.path(output_dir, paste0(tag, "_A.csv")), row.names = TRUE)
  write.csv(all_latent_factors$C, file.path(output_dir, paste0(tag, "_C.csv")), row.names = TRUE)
  write.csv(data.frame(Gamma = all_latent_factors$Gamma),
            file.path(output_dir, paste0(tag, "_Gamma.csv")), row.names = FALSE)
  write.csv(data.frame(I = all_latent_factors$I),
            file.path(output_dir, paste0(tag, "_I.csv")), row.names = FALSE)
  write.csv(data.frame(K = all_latent_factors$K,
                       opt_delta = all_latent_factors$opt_delta,
                       opt_lambda = all_latent_factors$opt_lambda),
            file.path(output_dir, paste0(tag, "_params.csv")), row.names = FALSE)

  saveRDS(all_latent_factors, file.path(output_dir, paste0(tag, "_AllLatentFactors.rds")))

} else {
  cat("Skipping remaining steps due to error in getLatentFactors.\n")
  stopCluster(cl)
  quit(status = 1)
}

# ============================================================================
# Step 2: Calculate Z Matrix
# ============================================================================
cat("\n", rep("-", 60), "\n", sep = "")
cat("Step 2: Calculating Z Matrix (calcZMatrix)\n")
cat(rep("-", 60), "\n", sep = "")

t_start <- Sys.time()

z_matrix <- tryCatch({
  predZ(X_std, all_latent_factors)
}, error = function(e) {
  cat("ERROR in predZ:", conditionMessage(e), "\n")
  return(NULL)
})

t_end <- Sys.time()

if (!is.null(z_matrix)) {
  colnames(z_matrix) <- paste0("Z", 1:ncol(z_matrix))
  rownames(z_matrix) <- rownames(X_std)

  cat("\nZ Matrix Results:\n")
  cat("  Z shape: ", nrow(z_matrix), " x ", ncol(z_matrix), "\n", sep = "")
  cat("  Time: ", format(t_end - t_start), "\n", sep = "")

  # Save Z matrix
  write.csv(z_matrix, file.path(output_dir, paste0(tag, "_Z.csv")), row.names = TRUE)

} else {
  cat("Skipping remaining steps due to error in Z matrix calculation.\n")
  stopCluster(cl)
  quit(status = 1)
}

# ============================================================================
# Step 3: Run SLIDE
# ============================================================================
cat("\n", rep("-", 60), "\n", sep = "")
cat("Step 3: Running SLIDE (runSLIDE)\n")
cat(rep("-", 60), "\n", sep = "")

set.seed(123)
t_start <- Sys.time()

SLIDE_res <- tryCatch({
  runSLIDE(
    y = Y,
    y_path = NULL,
    z_path = NULL,
    z_matrix = z_matrix,
    all_latent_factors = all_latent_factors,
    lf_path = NULL,
    method = 4,
    do_interacts = TRUE,
    fdr = fdr,
    niter = niter,
    spec = spec,
    f_size = NULL
  )
}, error = function(e) {
  cat("ERROR in runSLIDE:", conditionMessage(e), "\n")
  return(NULL)
})

t_end <- Sys.time()

if (!is.null(SLIDE_res)) {
  cat("\nSLIDE Results:\n")
  cat("  Marginal LFs: ", paste(SLIDE_res$marginal_vals, collapse = ", "), "\n", sep = "")
  cat("  Number of marginals: ", length(SLIDE_res$marginal_vals), "\n", sep = "")
  cat("  Number of interactions: ", nrow(SLIDE_res$interaction), "\n", sep = "")
  if (nrow(SLIDE_res$interaction) > 0) {
    cat("  Interaction pairs:\n")
    for (i in 1:min(5, nrow(SLIDE_res$interaction))) {
      cat("    Z", SLIDE_res$interaction$p1[i], " - Z", SLIDE_res$interaction$p2[i], "\n", sep = "")
    }
    if (nrow(SLIDE_res$interaction) > 5) cat("    ...\n")
  }
  cat("  f_size: ", SLIDE_res$SLIDE_param['f_size'], "\n", sep = "")
  cat("  Time: ", format(t_end - t_start), "\n", sep = "")

  # Save outputs
  write.csv(data.frame(marginal = SLIDE_res$marginal_vals),
            file.path(output_dir, paste0(tag, "_marginal_LFs.csv")), row.names = FALSE)
  write.csv(SLIDE_res$interaction,
            file.path(output_dir, paste0(tag, "_interactions.csv")), row.names = FALSE)
  write.csv(data.frame(t(SLIDE_res$SLIDE_param)),
            file.path(output_dir, paste0(tag, "_SLIDE_params.csv")), row.names = FALSE)

  saveRDS(SLIDE_res, file.path(output_dir, paste0(tag, "_SLIDE_res.rds")))

} else {
  cat("SLIDE failed to run.\n")
}

# Cleanup
stopCluster(cl)

# ============================================================================
# Summary
# ============================================================================
cat("\n")
cat(rep("=", 60), "\n", sep = "")
cat("R outputs saved to:", output_dir, "\n")
cat("Now run: python run_slide_py.py", x_file, y_file, tag, "\n")
cat(rep("=", 60), "\n", sep = "")
