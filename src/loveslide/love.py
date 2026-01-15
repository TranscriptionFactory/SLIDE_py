"""
Python wrapper for the LOVE algorithm.
Uses the pure Python implementation instead of R.
"""

import warnings
import numpy as np

from .love_pkg.love import LOVE


def call_love(X, lbd=0.5, mu=0.5, est_non_pure_row="HT", thresh_fdr=0.2, verbose=False,
              pure_homo=False, diagonal=False, delta=None, merge=False,
              rep_CV=50, ndelta=50, q=2, exact=False, max_pure=None, nfolds=10,
              outpath='.', **kwargs):
    """
    Python wrapper for calling the LOVE function.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The input data matrix (n samples x p features).
    lbd : float, optional
        Leading constant for lambda (precision estimation). Default is 0.5.
    mu : float, optional
        Leading constant for thresholding the loading matrix. Default is 0.5.
    est_non_pure_row : str, optional
        Method for non-pure rows: "HT", "ST", or "Dantzig". Default is "HT".
    thresh_fdr : float, optional
        Deprecated. Was used in R implementation only.
    verbose : bool, optional
        If True, print progress. Default is False.
    pure_homo : bool, optional
        If True, pure loadings have the same magnitude. Default is False.
    diagonal : bool, optional
        If True, covariance matrix of Z is diagonal. Default is False.
    delta : float, list, or np.ndarray, optional
        Grid of delta values for thresholding. Default is None.
    merge : bool, optional
        If True, take union of candidate pure variables; else intersection.
        Default is False.
    rep_CV : int, optional
        Number of repetitions for cross validation. Default is 50.
    ndelta : int, optional
        Length of delta grid. Default is 50.
    q : int, optional
        Type of score (2 or np.inf). Default is 2.
    exact : bool, optional
        Compute Inf score exactly via LP. Default is False.
    max_pure : float, optional
        Max proportion of pure variables. Default is None.
    nfolds : int, optional
        Number of folds for cross-validation. Default is 10.
    outpath : str, optional
        Deprecated. Was used in R implementation only.
    **kwargs : dict
        Additional keyword arguments (ignored for compatibility).

    Returns
    -------
    dict
        Dictionary containing the results from the LOVE analysis:
        - K: int, estimated number of clusters
        - pureVec: np.ndarray, indices of pure variables
        - pureInd: list of dicts with 'pos' and 'neg' indices
        - group: list of dicts with cluster assignments
        - A: np.ndarray, p x K estimated assignment matrix
        - C: np.ndarray, K x K covariance matrix of Z
        - Omega: np.ndarray or None, K x K precision matrix of Z
        - Gamma: np.ndarray, p-dim diagonal of error covariance
        - optDelta: float, selected delta value
    """

    # Deprecation warnings for R-specific parameters
    if thresh_fdr != 0.2:
        warnings.warn(
            "thresh_fdr parameter is deprecated (was R-only). It has no effect.",
            DeprecationWarning,
            stacklevel=2
        )
    if outpath != '.':
        warnings.warn(
            "outpath parameter is deprecated (was R-only). It has no effect.",
            DeprecationWarning,
            stacklevel=2
        )

    # Extract numpy array from DataFrame if needed
    X_array = X.values if hasattr(X, 'values') else np.asarray(X)

    # Convert delta to numpy array if provided as scalar or list
    if delta is not None and not isinstance(delta, np.ndarray):
        delta = np.atleast_1d(delta)

    # Call Python LOVE
    result = LOVE(
        X=X_array,
        lbd=lbd,
        mu=mu,
        est_non_pure_row=est_non_pure_row,
        verbose=verbose,
        pure_homo=pure_homo,
        diagonal=diagonal,
        delta=delta,
        merge=merge,
        rep_CV=rep_CV,
        ndelta=ndelta,
        q=q,
        exact=exact,
        max_pure=max_pure,
        nfolds=nfolds
    )

    return result
