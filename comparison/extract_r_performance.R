#!/usr/bin/env Rscript
# Extract performance metrics from R SLIDE output folder and save to CSV

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript extract_r_performance.R <output_folder>")
}

folder_path <- args[1]

# Read RDS files
sample_cv_file <- file.path(folder_path, "sampleCV_model.RDS")
slide_lfs_file <- file.path(folder_path, "SLIDE_LFs.rds")
all_lfs_file <- file.path(folder_path, "AllLatentFactors.rds")
control_perf_file <- file.path(folder_path, "ControlPerformance.rds")

# Check if minimal required files exist (SLIDE_LFs and AllLatentFactors)
if (!file.exists(slide_lfs_file) || !file.exists(all_lfs_file)) {
  stop("Required RDS files (SLIDE_LFs.rds, AllLatentFactors.rds) not found in ", folder_path)
}

# Initialize result list
result <- list()

# Read sampleCV data and calculate performance (if file exists)
result$true_score <- NA

if (file.exists(sample_cv_file)) {
  sample_cv_data <- readRDS(sample_cv_file)
  if (!is.null(sample_cv_data)) {
  true_y <- NULL
  pred_y <- NULL

  # Try different ways to access the data
  if (is.list(sample_cv_data)) {
    # Check for samplesCV element (common in lm objects)
    if ("samplesCV" %in% names(sample_cv_data)) {
      cv_data <- sample_cv_data$samplesCV
      # samplesCV might be a list with true_y and pred_y vectors
      if (is.list(cv_data)) {
        if ("true_y" %in% names(cv_data)) true_y <- cv_data$true_y
        if ("pred_y" %in% names(cv_data)) pred_y <- cv_data$pred_y
      } else if (is.data.frame(cv_data)) {
        if ("true_y" %in% names(cv_data)) true_y <- cv_data$true_y
        if ("pred_y" %in% names(cv_data)) pred_y <- cv_data$pred_y
      }
    }
    # Try direct access to true_y and pred_y
    if (is.null(true_y) && "true_y" %in% names(sample_cv_data)) {
      true_y <- sample_cv_data$true_y
    }
    if (is.null(pred_y) && "pred_y" %in% names(sample_cv_data)) {
      pred_y <- sample_cv_data$pred_y
    }
  } else if (is.data.frame(sample_cv_data)) {
    if ("true_y" %in% names(sample_cv_data)) true_y <- sample_cv_data$true_y
    if ("pred_y" %in% names(sample_cv_data)) pred_y <- sample_cv_data$pred_y
  }

  # Calculate performance if we found the data
  if (!is.null(true_y) && !is.null(pred_y)) {
    # Check if binary or continuous
    unique_vals <- unique(true_y)
    if (length(unique_vals) == 2) {
      # Binary - calculate AUC
      if (requireNamespace("pROC", quietly = TRUE)) {
        result$true_score <- as.numeric(pROC::auc(true_y, pred_y))
      } else if (requireNamespace("ROCR", quietly = TRUE)) {
        pred_obj <- ROCR::prediction(pred_y, true_y)
        result$true_score <- as.numeric(ROCR::performance(pred_obj, "auc")@y.values[[1]])
      }
    } else {
      # Continuous - calculate Spearman correlation
      result$true_score <- cor(true_y, pred_y, method = "spearman")
    }
  }
  }
}

# Read control performance if available
result$partial_random <- NA
result$full_random <- NA
if (file.exists(control_perf_file)) {
  control_data <- readRDS(control_perf_file)
  if (is.data.frame(control_data)) {
    # Check if there's a column indicating the type (FullRandom/PartialRandom)
    # Usually first column is the type/label, second column is the value
    if (ncol(control_data) >= 2) {
      # Get column names
      type_col <- names(control_data)[1]
      value_col <- names(control_data)[2]

      # Filter and calculate means
      full_random_rows <- control_data[control_data[[type_col]] == "FullRandom", ]
      if (nrow(full_random_rows) > 0) {
        result$full_random <- mean(full_random_rows[[value_col]], na.rm = TRUE)
      }

      partial_random_rows <- control_data[control_data[[type_col]] == "PartialRandom", ]
      if (nrow(partial_random_rows) > 0) {
        result$partial_random <- mean(partial_random_rows[[value_col]], na.rm = TRUE)
      }
    } else if ("partial" %in% names(control_data)) {
      result$partial_random <- mean(control_data$partial, na.rm = TRUE)
    } else if ("full" %in% names(control_data)) {
      result$full_random <- mean(control_data$full, na.rm = TRUE)
    }
  } else if (is.list(control_data) && length(control_data) >= 3) {
    result$partial_random <- control_data[[2]]
    result$full_random <- control_data[[3]]
  } else if (is.numeric(control_data) && length(control_data) >= 3) {
    result$partial_random <- control_data[2]
    result$full_random <- control_data[3]
  }
}

# Read AllLatentFactors to get number of LFs
all_lfs_data <- readRDS(all_lfs_file)
result$num_LFs <- NA

if (is.list(all_lfs_data)) {
  # Try to get K (number of latent factors)
  if ("K" %in% names(all_lfs_data)) {
    result$num_LFs <- all_lfs_data$K
  } else if ("A" %in% names(all_lfs_data)) {
    # A matrix should have dimensions (features x K)
    if (is.matrix(all_lfs_data$A)) {
      result$num_LFs <- ncol(all_lfs_data$A)
    }
  }
} else if (is.matrix(all_lfs_data) || is.data.frame(all_lfs_data)) {
  result$num_LFs <- ncol(all_lfs_data)
}

# Read SLIDE_LFs to get marginal and interaction counts
slide_lfs_data <- readRDS(slide_lfs_file)
result$num_marginals <- 0
result$num_interactors <- 0

if (is.list(slide_lfs_data)) {
  # Try to find marginal and interaction info
  for (name in names(slide_lfs_data)) {
    item <- slide_lfs_data[[name]]
    if (grepl("marginal", name, ignore.case = TRUE)) {
      if (is.data.frame(item)) {
        result$num_marginals <- nrow(item)
      } else if (is.vector(item)) {
        result$num_marginals <- length(item)
      }
    } else if (grepl("interact", name, ignore.case = TRUE)) {
      if (is.data.frame(item)) {
        result$num_interactors <- nrow(item)
      } else if (is.vector(item)) {
        result$num_interactors <- length(item)
      }
    }
  }

  # Alternative: check feature_list files
  if (result$num_marginals == 0) {
    feature_files <- list.files(folder_path, pattern = "feature_list_Z.*\\.txt", full.names = TRUE)
    if (length(feature_files) > 0) {
      # Count unique features across all Z files
      all_features <- c()
      for (ff in feature_files) {
        features <- read.table(ff, header = TRUE, stringsAsFactors = FALSE)
        if (nrow(features) > 0) {
          all_features <- c(all_features, features$names)
        }
      }
      result$num_marginals <- length(unique(all_features))
    }
  }
}

# Convert to data frame and save
result_df <- as.data.frame(result)
output_file <- file.path(folder_path, "performance_metrics.csv")
write.csv(result_df, output_file, row.names = FALSE)

cat("Successfully extracted performance metrics to", output_file, "\n")
