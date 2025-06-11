########################################################################################
##########                                                                ##############
##########              Latent-model based OVErlap clustering             ##############
##########                                                                ##############
########################################################################################

#' @title LOVE: Latent-model based OVErlapping clustering
#'
#' @description Perform overlapping (variable) clustering of a \eqn{p}-
#' dimensional feature generated from the latent factor model
#' \deqn{X = AZ + E}
#' with identifiability conditions on \eqn{A} and \eqn{Cov(Z)}.
#'
#' @param X A \eqn{n} by \eqn{p} data matrix.
#' @param lbd The grid of leading constant of \eqn{\lambda}.
#' @param mu The leading constant used for thresholding the loading matrix.
#' @param est_non_pure_row String. Procedure used for estimating the non-pure
#'   rows. One of \{"HT", "ST", "Dantzig"\}.
#' @param diagonal Logical. If TRUE, the covariance matrix of \eqn{Z} is
#'   diagonal; else FALSE.
#' @param verbose Logical. Set FALSE to suppress printing the progress.
#' @param pure_homo Logical. TRUE if the pure loadings have the same magnitude.
#' @param delta The grid of leading constant of \eqn{\delta}.
#' @param merge Logical. If TRUE, take the union of all candidate pure variables;
#'   otherwise, take the intersection.
#' @param rep_CV The number of repetitions used for cross validation.
#' @inheritParams KfoldCV_delta
#'
#' @return A list of objects including: \itemize{
#'   \item \code{K} The estimated number of clusters.
#'   \item \code{pureVec} The estimated set of pure variables.
#'   \item \code{pureInd} The estimated partition of pure variables.
#'   \item \code{group} The estimated clusters (indices of each cluster).
#'   \item \code{A} The estimated \eqn{p} by \eqn{K} assignment matrix.
#'   \item \code{C} The covariance matrix of \eqn{Z}.
#'   \item \code{Omega} The precision matrix of \eqn{Z}.
#'   \item \code{Gamma} The diagonal of the covariance matrix of \eqn{E}.
#'   \item \code{optDelta} The selected value of \eqn{\delta}.
#' }
#'
#' @details \code{LOVE} performs overlapping clustering of the feature variables
#'   \eqn{X} generated from the latent factor model
#'   \deqn{X = AZ+E}
#'   where the loading matrix \eqn{A} and the covariance matrix of \eqn{Z}
#'   satisfy certain identifiability conditions. The main goal is to estimate
#'   the loading matrix \eqn{A} whose support is used to form overlapping groups
#'   of \eqn{X}.
#'
#'   The first step estimates the pure loadings, defined as the rows of \eqn{A}
#'   that are proportional to canonical vectors. When the pure loadings are
#'   expected to have the same magnitudes (up to the sign), for instance,
#'   \deqn{A_{1.} = (1, 0, 0), A_{2.} = (-1, 0, 0),}
#'   the estimation of pure loadings is done via setting \code{pure_homo} to
#'   \code{TRUE}. When different magnitudes are expected for the pure loadings,
#'   such as \deqn{A_{1.} = (1, 0, 0), A_{2.} = (-0.5, 0, 0),}
#'   the estimation uses a different approach by setting setting \code{pure_homo}
#'   to \code{FALSE}.
#'
#'   The second step estimates the non-pure (mixed) loadings of \eqn{A}. Three
#'   procedures are available as specified by \code{est_non_pure_row}. The choice
#'   "HT" specifies the estimation via hard-thresholding that is computationally
#'   fast while "ST" uses soft-thresholding instead. Both "ST" and "Dantzig"
#'   resort to solving linear programs. Another difference of "Dantzig" from "HT"
#'   and "ST" is that the former does not require to estimate the precision
#'   matrix of \eqn{Z}.
#'
#'
#' @examples
#' p <- 6
#' n <- 100
#' K <- 2
#' A <- rbind(c(1, 0), c(-1, 0), c(0, 1), c(0, 1), c(1/3, 2/3), c(1/2, -1/2))
#' Z <- matrix(rnorm(n * K, sd = sqrt(2)), n, K)
#' E <- matrix(rnorm(n * p), n, p)
#' X <- Z %*% t(A) + E
#' res_LOVE <- LOVE(X, pure_homo = FALSE, delta = NULL)
#' res_LOVE <- LOVE(X, pure_homo = TRUE, delta = seq(0.1, 1.1 ,0.1))
#'
#' @export
#' @import stats
#' @references
#'
#' Bing, X., Bunea, F., Yang N and Wegkamp, M. (2020) \emph{Adaptive
#' estimation in structured factor models with applications to overlapping clustering},
#' Annals of Statistics, Vol.48(4) 2055 - 2081, August 2020.
#' \url{https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-4/Adaptive-estimation-in-structured-factor-models-with-applications-to-overlapping/10.1214/19-AOS1877.short}
#'
#' Bing, X., Bunea, F. and Wegkamp, M. (2021) \emph{Detecting approximate replicate
#' components of a high-dimensional random vector with latent structure}.
#' \url{https://arxiv.org/abs/2010.02288}.

script_path <- getSrcDirectory(function(){})
script_dir <- dirname(script_path)
thresh_path <- file.path(dirname(script_dir), "LOVE-SLIDE", "threshSigma.R")

source(thresh_path)



LOVE <- function(X, lbd = 0.5, mu = 0.5, est_non_pure_row = "HT", thresh_fdr = NULL,
                 verbose = FALSE, pure_homo = FALSE, diagonal = FALSE, 
                 delta = NULL, merge = FALSE, rep_CV = 50, ndelta = 1, 
                 q = 2, exact = FALSE, max_pure = NULL, nfolds = 10, 
                 out_path = NULL, gene_names = NULL, sample_names = NULL) {


  # Adding back the gene and sample names that were removed in python -> R conversion
  if (!is.null(gene_names)) {
    colnames(X) <- gene_names
  }
  if (!is.null(sample_names)) {
    rownames(X) <- sample_names
  }

  n <- nrow(X);  p <- ncol(X)

  X <- scale(X, T, T)  # centering
  if (verbose) {
    cat("x max: ", max(X), ", x min: ", min(X), "\n")
  }

  if (!is.null(out_path)) {
    # Save X to a CSV file in the output directory
    write.csv(X, file.path(out_path, "X.csv"), row.names = FALSE)
  }

  se_est <- apply(X, 2, stats::sd)   # estimate the standard errors of each feature

  if (verbose) {
    cat("se_est max: ", max(se_est) , ", se_est min: ", min(se_est), "\n")
  }

  if (!is.null(out_path)) {
    write.csv(se_est, file.path(out_path, "se_est.csv"), row.names = FALSE)
  }

  # original LOVE delta selection
  # deltaGrids <- delta * sqrt(log(max(p, n)) / n)
  # if (verbose)
  #   cat("Select delta by using data splitting...\n")
  # optDelta <- ifelse(length(deltaGrids) > 1,
  #                    median(replicate(rep_CV, CV_delta(X, deltaGrids, diagonal,
  #                                                      se_est, merge))),
  #                    deltaGrids)

  # SLIDE delta selection
  optDelta <- delta * sqrt(log(max(p, n)) / n)

  if (verbose) {
    cat("optDelta: ", optDelta, "\n")
  }

  # delta selection isn't done in SLIDE
  # if (verbose)
  #   cat("Finish selecting delta and start estimating the pure loadings...\n")

  # Sigma <- cov(X)
  Sigma <- cor(X) # SLIDE uses correlation matrix

  if (verbose) {
    cat("sigma max: ", max(Sigma), ", sigma min: ", min(Sigma), "\n")
  }

  # SLIDE added this to threshold sigma to control for FDR 
  if (!is.null(thresh_fdr)) {
    control_fdr <- threshSigma(x = X,
                              sigma = Sigma,
                              thresh = thresh_fdr)
    Sigma <- control_fdr$thresh_sigma
    kept_entries <- control_fdr$kept_entries
  } else {
    kept_entries <- matrix(1, nrow = nrow(Sigma), ncol = ncol(Sigma))
  }

  if (!is.null(out_path)) {
    write.csv(Sigma, file.path(out_path, "Sigma.csv"), row.names = FALSE)
    write.csv(kept_entries, file.path(out_path, "kept_entries.csv"), row.names = FALSE)
  }

  ##### pure_homo starts from here
  R_hat <- cor(X)
  Sigma <- abs(Sigma) # This has become a thresholded correlation matrix, which can be negative
  # Sigma <- cov(X)

  if (verbose)
    cat("Select delta by using", nfolds, "fold cross-validation...\n")

  # Find parallel rows and its partition
  CV_res <- KfoldCV_delta(X, optDelta, ndelta, q, exact, nfolds, max_pure, verbose)
  pure_res <- CV_res$est_pure
  est_I <- pure_res$I_part
  est_I_set <- pure_res$I

  if (verbose) {
    cat('delta: ', CV_res$delta_min, '\n')
  }

  # optDelta <- CV_res$delta_min

  if (verbose)
    cat("Start estimating the pure loadings...\n")

  # Post-select pure variables
  if (length(est_I) >= 2) {
    BI_C_res <- Est_BI_C(CV_res$moments, R_hat, est_I, est_I_set)
    est_I <- Re_Est_Pure(X, Sigma, CV_res$moments, est_I, BI_C_res$Gamma)
    est_I_set <- as.numeric(unlist(est_I))
  } else if (length(est_I) <= 1) {
    cat("Algorithm fails since <2 pure variables found.\n")
    stop()
  }

  D_Sigma <- diag(Sigma)
  B_hat <- BI_C_res$B
  R_Z <- BI_C_res$C

  # Estimate the loading matrix A and C.

  B_hat <- sqrt(D_Sigma) * B_hat
  D_B <- apply(abs(B_hat), 2, max)
  A_hat <- t(t(B_hat) / D_B)
  C_hat <- D_B  * R_Z

  if (diagonal)
    C_hat <- diag(diag(C_hat))

  I_hat <- est_I_set
  I_hat_part <- FindSignPureNode(pure_res$I_part, Sigma)

  Gamma_hat <- BI_C_res$Gamma * D_Sigma
  
  #### pure_homo ends from here

  Gamma_hat[Gamma_hat < 0] <- 0

  if (verbose)
    cat("Finish estimating the pure loadings...\n")



  if (length(I_hat) == p)  {### the case that all variables are pure
    # group <- resultAI$pureSignInd
    group <- I_hat_part
    Omega <- NULL
  } else {

    if (diagonal)
      Omega <- diag(diag(C_hat) ** (-1))
    else {
      # # SLIDE doesn't do lambda selection
      # if (verbose)
      #   cat("Select lambda for estimating the precision of Z...\n")

      # lbdGrids <- lbd * optDelta

      # optLbd <- ifelse(length(lbd) > 1,
      #                  median(replicate(rep_CV, CV_lbd(X, lbdGrids, A_hat, I_hat, diagonal))),
      #                  lbdGrids)

      # if (verbose) {
      #   cat("Select lambda =", optLbd, "with leading constant",
      #       min(which(lbdGrids >= optLbd)),"...\n")
      #   cat("Start estimating the precision of Z...\n")
      # }
      optLbd <- lbd
      Omega <- estOmega(optLbd, C_hat)
    }

    if (verbose) {
      cat("Estimate the non-pure loadings by",
          switch(est_non_pure_row,
                 "HT" = "Hard Thresholding",
                 "ST" = "Soft Thresholding",
                 "Dantzig" = "Dantzig"), "...\n")
    }

    Y <- EstY(Sigma, A_hat, I_hat)

    if (est_non_pure_row == "Dantzig") {
      AI <- abs(A_hat[I_hat, ])
      sigma_bar_sup <- max(solve(crossprod(AI), t(AI)) %*% se_est[I_hat])

      AJ <- EstAJDant(
        C_hat, 
        Y,
        mu * optDelta * sigma_bar_sup,  # orignal LOVE parameter
        sigma_bar_sup + se_est[-I_hat]
        )
      
    } else {

      threshold <- mu * optDelta * norm(Omega, "I")

      if (est_non_pure_row == "HT")
        AJ <- threshA(t(Omega %*% Y), threshold)
      else if (est_non_pure_row == "ST")
        AJ <- EstAJInv(Omega, Y, threshold)
      else
        cat("Unknown method of estimating the non-pure rows.\n")
    }

    A_hat[-I_hat, ] <- AJ
    group <-recoverGroup(A_hat)
  }
  return(list(K = ncol(A_hat),
              pureVec = I_hat,
              pureInd = I_hat_part,
              group = group,
              A = A_hat,
              C = C_hat,
              Omega = Omega,
              Gamma = Gamma_hat
              # optDelta = optDelta
              ))
}
