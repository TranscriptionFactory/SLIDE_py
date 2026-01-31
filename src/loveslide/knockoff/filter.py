"""Main knockoff filter pipeline."""

from dataclasses import dataclass
from typing import Optional, Callable, Union, List, Dict, Any
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from scipy import linalg

from .create import create_second_order, create_fixed, KnockoffVariables
from .solve import create_solve_equi, create_solve_sdp, create_solve_asdp
from .utils import is_posdef


@dataclass
class KnockoffResult:
    """
    Result of the knockoff filter procedure.

    Attributes
    ----------
    X : np.ndarray
        Matrix of original variables.
    Xk : np.ndarray
        Matrix of knockoff variables.
    y : np.ndarray
        Response vector.
    statistic : np.ndarray
        Computed test statistics W.
    threshold : float
        Selection threshold.
    selected : np.ndarray
        Indices of selected variables.
    feature_names : list, optional
        Names of selected features (if provided).
    """
    X: np.ndarray
    Xk: np.ndarray
    y: np.ndarray
    statistic: np.ndarray
    threshold: float
    selected: np.ndarray
    feature_names: Optional[List[str]] = None

    def __repr__(self) -> str:
        n_selected = len(self.selected)
        p = self.X.shape[1]
        return (
            f"KnockoffResult(\n"
            f"  n_features={p},\n"
            f"  n_selected={n_selected},\n"
            f"  selected={self.selected.tolist()},\n"
            f"  threshold={self.threshold:.4f}\n"
            f")"
        )


def knockoff_threshold(
    W: np.ndarray,
    fdr: float = 0.10,
    offset: int = 1,
    **kwargs
) -> float:
    """
    Compute the threshold for the knockoff filter.

    Parameters
    ----------
    W : array-like of shape (p,)
        Test statistics.
    fdr : float, default=0.10
        Target false discovery rate.
    offset : {0, 1}, default=1
        The value 1 yields a slightly more conservative procedure ("knockoffs+")
        that controls the FDR according to the usual definition, while an
        offset of 0 controls a modified FDR.

    Returns
    -------
    float
        The threshold for variable selection.
    """
    W = np.asarray(W)

    if offset not in [0, 1]:
        raise ValueError("offset must be either 0 or 1")

    # Candidate thresholds: 0 and absolute values of W
    ts = np.sort(np.concatenate([[0], np.abs(W)]))

    # For each threshold, compute FDP estimate
    for t in ts:
        numerator = offset + np.sum(W <= -t)
        denominator = max(1, np.sum(W >= t))
        ratio = numerator / denominator
        if ratio <= fdr:
            return t

    return np.inf


def knockoff_filter(
    X: np.ndarray,
    y: np.ndarray,
    knockoffs: Optional[Callable] = None,
    statistic: Optional[Callable] = None,
    fdr: float = 0.10,
    offset: int = 1,
    feature_names: Optional[List[str]] = None,
    **kwargs
) -> KnockoffResult:
    """
    Run the Knockoff Filter for controlled variable selection.

    This function creates knockoffs, computes importance statistics,
    and selects variables while controlling the false discovery rate.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix of predictors.
    y : array-like of shape (n,)
        Response vector.
    knockoffs : callable, optional
        Function to construct knockoffs. Takes a (n, p) matrix as input
        and returns knockoff variables. Default: create_second_order.
    statistic : callable, optional
        Function to compute importance statistics. Takes (X, Xk, y) as input
        and returns a (p,) array of statistics W.
        Default: stat_glmnet_coefdiff.
    fdr : float, default=0.10
        Target false discovery rate.
    offset : {0, 1}, default=1
        Offset for computing the rejection threshold.
        1 = more conservative ("knockoffs+"), 0 = modified FDR.
    feature_names : list of str, optional
        Names of features for labeling selected variables.

    Returns
    -------
    KnockoffResult
        Object containing X, Xk, y, statistic, threshold, and selected.

    References
    ----------
    Candes et al., Panning for Gold: Model-free Knockoffs for
    High-dimensional Controlled Variable Selection, arXiv:1610.02351 (2016).

    Barber and Candes, Controlling the false discovery rate via knockoffs.
    Ann. Statist. 43 (2015), no. 5, 2055--2085.

    Examples
    --------
    >>> import numpy as np
    >>> from knockoff import knockoff_filter
    >>> n, p = 100, 50
    >>> X = np.random.randn(n, p)
    >>> beta = np.zeros(p)
    >>> beta[:5] = 3.0
    >>> y = X @ beta + np.random.randn(n)
    >>> result = knockoff_filter(X, y, fdr=0.1)
    >>> print(result.selected)
    """
    # Convert inputs to numpy arrays
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    # Validate input types
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must be numeric")

    if not (np.issubdtype(y.dtype, np.number) or np.issubdtype(y.dtype, np.object_)):
        raise ValueError("y must be numeric or categorical")

    if y.ndim > 1:
        y = y.ravel()

    if offset not in [0, 1]:
        raise ValueError("offset must be either 0 or 1")

    # Validate dimensions
    n, p = X.shape
    if len(y) != n:
        raise ValueError(f"Length of y ({len(y)}) must match number of rows in X ({n})")

    # Extract feature names if not provided
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = list(X.columns)

    # Set default knockoff constructor
    if knockoffs is None:
        knockoffs = create_second_order

    # Set default statistic
    if statistic is None:
        from .stats import stat_glmnet_coefdiff
        statistic = stat_glmnet_coefdiff

    # Handle create_fixed specially (needs y for augmentation)
    if knockoffs is create_fixed:
        knockoffs = lambda x: create_fixed(x, y=y)

    # Create knockoff variables
    knock_variables = knockoffs(X)

    # Handle different return types from knockoff constructors
    if isinstance(knock_variables, KnockoffVariables):
        X = knock_variables.X
        Xk = knock_variables.Xk
        if knock_variables.y is not None:
            y = knock_variables.y
    elif isinstance(knock_variables, np.ndarray):
        Xk = knock_variables
    else:
        raise ValueError(
            f"Knockoff constructor returned unexpected type: {type(knock_variables)}"
        )

    # Compute statistics
    W = statistic(X, Xk, y)

    # Run the knockoff filter
    t = knockoff_threshold(W, fdr=fdr, offset=offset)
    selected = np.sort(np.where(W >= t)[0])

    # Get feature names for selected variables
    selected_names = None
    if feature_names is not None:
        selected_names = [feature_names[i] for i in selected]

    return KnockoffResult(
        X=X,
        Xk=Xk,
        y=y,
        statistic=W,
        threshold=t,
        selected=selected,
        feature_names=selected_names
    )


@dataclass
class VotingResult:
    """
    Result of the knockoff voting procedure (SLIDE-style).

    Attributes
    ----------
    selection_counts : np.ndarray
        Number of times each variable was selected across iterations.
    selection_frequency : np.ndarray
        Proportion of iterations where each variable was selected (= W).
    selected : np.ndarray
        Indices of variables selected in >= spec proportion of runs.
    threshold : float
        The spec threshold used for selection.
    niter : int
        Number of knockoff iterations.
    spec : float
        Specificity threshold.
    min_selections : int
        Minimum selections required (= ceiling(niter * spec)).
    """
    selection_counts: np.ndarray
    selection_frequency: np.ndarray  # This is the "W" for voting
    selected: np.ndarray
    threshold: float
    niter: int
    spec: float
    min_selections: int

    def __repr__(self) -> str:
        n_selected = len(self.selected)
        p = len(self.selection_counts)
        return (
            f"VotingResult(\n"
            f"  n_features={p},\n"
            f"  n_selected={n_selected},\n"
            f"  selected={self.selected.tolist()},\n"
            f"  niter={self.niter},\n"
            f"  spec={self.spec},\n"
            f"  min_selections={self.min_selections}\n"
            f")"
        )


def _run_single_knockoff(args):
    """
    Worker function for parallel knockoff execution.

    Returns the indices of selected variables for one iteration.
    """
    X, y, knockoffs, statistic, fdr, offset, seed = args

    # Set random seed for this iteration
    np.random.seed(seed)

    try:
        result = knockoff_filter(
            X, y,
            knockoffs=knockoffs,
            statistic=statistic,
            fdr=fdr,
            offset=offset
        )
        return result.selected.tolist()
    except Exception as e:
        warnings.warn(f"Knockoff iteration failed (seed={seed}): {e}")
        return []


def _prepare_knockoff_cache(
    X: np.ndarray,
    method: str = 'asdp',
    shrink: bool = False,
    match_r: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Pre-compute invariant values for knockoff generation.

    These computations are CONSTANT across all voting iterations since X doesn't change.
    By caching them, we avoid redundant SDP solving, covariance estimation, and Cholesky
    decomposition on every iteration.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Matrix of predictors.
    method : str, default='asdp'
        Method for minimizing correlation between original and knockoffs.
        One of 'asdp', 'sdp', 'equi'.
    shrink : bool, default=False
        Whether to shrink the estimated covariance matrix.
    match_r : bool, default=False
        If True, skip the condition number check for auto-shrinkage.

    Returns
    -------
    dict
        Dictionary containing cached values:
        - 'mu': Mean vector (p,)
        - 'Sigma': Covariance matrix (p, p)
        - 'diag_s': Diagonal of knockoff covariance (p,)
        - 'SigmaInv_s': Sigma^{-1} @ diag(s) (p, p)
        - 'L': Cholesky decomposition of Sigma_k (p, p)
        - 'method': Method used (may differ from input if fallback occurred)
        - 'degenerate': True if knockoff construction failed
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    if method not in ['asdp', 'sdp', 'equi']:
        raise ValueError(f"method must be 'asdp', 'sdp', or 'equi', got '{method}'")

    # Do not use ASDP unless p > 500
    if p <= 500 and method == 'asdp':
        method = 'sdp'

    # Auto-enable shrinkage for n <= 1.25*p (R-style regularization)
    if not shrink and n <= 1.25 * p:
        warnings.warn(
            f"n={n}, p={p} (n/p={n/p:.2f}): Insufficient samples for stable covariance. "
            f"Auto-enabling Ledoit-Wolf shrinkage to match R's knockoff.filter behavior."
        )
        shrink = True

    # Estimate the mean vector
    mu = np.mean(X, axis=0)

    # Estimate the covariance matrix
    if not shrink:
        Sigma = np.cov(X, rowvar=False, ddof=1)
        if Sigma.ndim == 0:
            Sigma = np.array([[Sigma]])

        if not is_posdef(Sigma):
            shrink = True

        # Auto-enable shrinkage for ill-conditioned matrices (unless match_r)
        if not shrink and not match_r:
            cond_num = np.linalg.cond(Sigma)
            if cond_num > 1e5:
                warnings.warn(
                    f"Covariance matrix is ill-conditioned (cond={cond_num:.1e}). "
                    f"Auto-enabling Ledoit-Wolf shrinkage for better knockoff power."
                )
                shrink = True

    if shrink:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(X)
            Sigma = lw.covariance_
        except ImportError:
            warnings.warn(
                "sklearn is not installed. Using manual shrinkage."
            )
            S = np.cov(X, rowvar=False, ddof=1)
            if S.ndim == 0:
                S = np.array([[S]])
            trace_S = np.trace(S)
            shrinkage_param = min(1.0, max(0.0, (p / n)))
            Sigma = (1 - shrinkage_param) * S + shrinkage_param * (trace_S / p) * np.eye(p)

    # Compute diag_s using the solver
    if method == 'equi':
        diag_s = create_solve_equi(Sigma)
    elif method == 'sdp':
        diag_s = create_solve_sdp(Sigma)
    else:  # asdp
        diag_s = create_solve_asdp(Sigma)

    diag_s = np.asarray(diag_s)
    if diag_s.ndim == 2:
        diag_s = np.diag(diag_s)

    # Check for degenerate SDP solution and fall back to equicorrelated
    max_s = np.max(diag_s) if len(diag_s) > 0 else 0
    if np.all(diag_s == 0) or max_s < 1e-6:
        if method in ['sdp', 'asdp']:
            warnings.warn(
                f"SDP solver returned degenerate solution (max diag_s={max_s:.2e}). "
                f"Falling back to equicorrelated method for robustness."
            )
            diag_s = create_solve_equi(Sigma)
            method = 'equi'
            max_s = np.max(diag_s) if len(diag_s) > 0 else 0

            if np.all(diag_s == 0) or max_s < 1e-6:
                warnings.warn(
                    "Both SDP and equicorrelated methods failed. "
                    "Knockoffs will have no power."
                )
                return {
                    'mu': mu,
                    'Sigma': Sigma,
                    'diag_s': diag_s,
                    'SigmaInv_s': None,
                    'L': None,
                    'method': method,
                    'degenerate': True
                }

    # Compute knockoff distribution parameters
    diag_s_matrix = np.diag(diag_s)
    SigmaInv_s = linalg.solve(Sigma, diag_s_matrix)

    # Sigma_k = 2*diag(s) - diag(s) @ SigmaInv_s
    Sigma_k = 2 * diag_s_matrix - diag_s_matrix @ SigmaInv_s

    # Cholesky decomposition with R-style scaled regularization
    try:
        L = linalg.cholesky(Sigma_k, lower=True)
    except linalg.LinAlgError:
        max_diag = np.max(np.diag(Sigma_k))
        eps = 1e-10 * max(1.0, max_diag)
        try:
            L = linalg.cholesky(Sigma_k + eps * np.eye(p), lower=True)
        except linalg.LinAlgError:
            while eps < 1:
                eps *= 10
                try:
                    L = linalg.cholesky(Sigma_k + eps * np.eye(p), lower=True)
                    break
                except linalg.LinAlgError:
                    continue
            else:
                raise ValueError("Cholesky decomposition failed even with large regularization")

    return {
        'mu': mu,
        'Sigma': Sigma,
        'diag_s': diag_s,
        'SigmaInv_s': SigmaInv_s,
        'L': L,
        'method': method,
        'degenerate': False
    }


def _sample_knockoffs_from_cache(
    X: np.ndarray,
    cache: Dict[str, Any]
) -> np.ndarray:
    """
    Generate knockoff variables using pre-computed cache.

    This is the ONLY computation that varies across voting iterations -
    the random sampling step.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Matrix of original variables.
    cache : dict
        Pre-computed cache from _prepare_knockoff_cache.

    Returns
    -------
    np.ndarray of shape (n, p)
        Matrix of knockoff variables.
    """
    if cache.get('degenerate', False):
        # Return copy of X if knockoff construction failed
        return X.copy()

    n, p = X.shape
    mu = cache['mu']
    SigmaInv_s = cache['SigmaInv_s']
    L = cache['L']

    # mu_k = X - (X - mu) @ SigmaInv_s
    mu_k = X - (X - mu) @ SigmaInv_s

    # Sample knockoffs: X_k = mu_k + randn(n, p) @ L.T
    X_k = mu_k + np.random.randn(n, p) @ L.T

    return X_k


def knockoff_filter_voting(
    X: np.ndarray,
    y: np.ndarray,
    knockoffs: Optional[Callable] = None,
    statistic: Optional[Callable] = None,
    fdr: float = 0.10,
    offset: int = 0,
    niter: int = 500,
    spec: float = 0.1,
    n_jobs: int = 1,
    base_seed: int = 42,
    verbose: bool = False,
    match_r: bool = False,
    use_cache: bool = True,
    **kwargs
) -> VotingResult:
    """
    Run the Knockoff Filter with SLIDE-style voting for stable variable selection.

    This function runs the knockoff filter multiple times with different random
    seeds and selects variables that appear in at least `spec` proportion of runs.
    This matches R's SLIDE voting methodology.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix of predictors.
    y : array-like of shape (n,)
        Response vector.
    knockoffs : callable, optional
        Function to construct knockoffs. Default: create_second_order.
    statistic : callable, optional
        Function to compute importance statistics. Default: stat_glmnet_coefdiff.
    fdr : float, default=0.10
        Target false discovery rate for each knockoff run.
    offset : {0, 1}, default=0
        Offset for knockoff threshold. 0 = standard knockoff (matches R's knockoff.filter),
        1 = knockoffs+ (more conservative).
    niter : int, default=500
        Number of knockoff iterations to run.
    spec : float, default=0.1
        Specificity threshold - keep variables selected in >= spec * niter runs.
    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all available cores.
    base_seed : int, default=42
        Base seed for reproducibility. Each iteration uses base_seed + i.
    verbose : bool, default=False
        Print progress information.
    match_r : bool, default=False
        If True, skip the automatic Ledoit-Wolf shrinkage condition number
        check to match R's knockoff.filter behavior. Use this for exact
        R compatibility testing in n ~ p boundary cases.
    use_cache : bool, default=True
        If True (default), pre-compute invariant quantities (covariance, SDP solution,
        Cholesky decomposition) once before the voting loop. This provides 3-4x speedup
        by avoiding redundant computations. Only applies when using default knockoffs.
        Set to False to use the original uncached behavior.

    Returns
    -------
    VotingResult
        Object containing selection_counts, selection_frequency (W), selected,
        and voting parameters.

    Notes
    -----
    This implements the SLIDE voting approach:
    1. Run knockoff filter `niter` times with different random seeds
    2. Count how often each variable is selected
    3. Keep variables selected in >= spec proportion of runs

    The `selection_frequency` field serves as the "W" statistic for voting,
    representing the proportion of runs where each variable was selected.

    Performance Optimization (use_cache=True):
    When using default second-order knockoffs, the covariance matrix estimation,
    SDP solver, and Cholesky decomposition are CONSTANT across all voting iterations
    since X doesn't change. By pre-computing these once, we avoid redundant work
    and achieve 3-4x speedup (e.g., from ~23 min to ~6-8 min for 500 iterations).

    References
    ----------
    Barber and Candes, Controlling the false discovery rate via knockoffs.
    Ann. Statist. 43 (2015), no. 5, 2055--2085.

    Examples
    --------
    >>> import numpy as np
    >>> from loveslide.knockoff import knockoff_filter_voting
    >>> n, p = 100, 50
    >>> X = np.random.randn(n, p)
    >>> beta = np.zeros(p)
    >>> beta[:5] = 3.0
    >>> y = X @ beta + np.random.randn(n)
    >>> result = knockoff_filter_voting(X, y, niter=100, spec=0.1)
    >>> print(result.selected)
    """
    # Convert inputs
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if y.ndim > 1:
        y = y.ravel()

    n, p = X.shape

    # Validate parameters
    if niter < 1:
        raise ValueError(f"niter must be >= 1, got {niter}")
    if not 0 < spec <= 1:
        raise ValueError(f"spec must be in (0, 1], got {spec}")

    # Set defaults
    # If match_r is True and no custom knockoffs function provided, wrap create_second_order
    if knockoffs is None:
        if match_r:
            knockoffs = lambda X: create_second_order(X, match_r=True)
        else:
            knockoffs = create_second_order
    if statistic is None:
        from .stats import stat_glmnet_coefdiff
        statistic = stat_glmnet_coefdiff

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = max(1, min(n_jobs, niter))

    # Initialize selection counts
    selection_counts = np.zeros(p, dtype=np.int32)

    if verbose:
        print(f"Running {niter} knockoff iterations with {n_jobs} jobs...")

    # Prepare arguments for each iteration
    # Note: We can't pickle lambdas, so we pass knockoffs/statistic as None
    # and rely on defaults, or we run sequentially

    if n_jobs == 1:
        # Sequential execution (safer, works with custom knockoffs/statistic)
        # Check if we can use caching optimization (default knockoffs and use_cache enabled)
        can_use_cache = use_cache and (knockoffs is create_second_order or
                     (knockoffs is not None and callable(knockoffs) and
                      hasattr(knockoffs, '__name__') and knockoffs.__name__ == '<lambda>' and match_r))

        if can_use_cache:
            # OPTIMIZED PATH: Pre-compute invariant quantities once
            # This avoids redundant SDP solving, covariance estimation, and Cholesky
            # decomposition on every iteration (3-4x speedup)
            if verbose:
                print("  Using cached knockoff computation (optimized path)...")

            try:
                cache = _prepare_knockoff_cache(X, method='asdp', shrink=False, match_r=match_r)
            except Exception as e:
                warnings.warn(f"Cache preparation failed: {e}. Falling back to uncached path.")
                can_use_cache = False

        if can_use_cache:
            # Fast path with caching
            for i in range(niter):
                seed = base_seed + i
                np.random.seed(seed)

                try:
                    # Only random sampling varies per iteration
                    Xk = _sample_knockoffs_from_cache(X, cache)

                    # Compute statistics and threshold
                    W = statistic(X, Xk, y)
                    t = knockoff_threshold(W, fdr=fdr, offset=offset)
                    selected_iter = np.where(W >= t)[0]

                    for idx in selected_iter:
                        selection_counts[idx] += 1

                    if verbose and (i + 1) % 50 == 0:
                        print(f"  Completed {i + 1}/{niter} iterations")

                except Exception as e:
                    warnings.warn(f"Knockoff iteration {i} failed: {e}")
        else:
            # Original path for custom knockoffs
            for i in range(niter):
                seed = base_seed + i
                np.random.seed(seed)

                try:
                    result = knockoff_filter(
                        X, y,
                        knockoffs=knockoffs,
                        statistic=statistic,
                        fdr=fdr,
                        offset=offset
                    )
                    for idx in result.selected:
                        selection_counts[idx] += 1

                    if verbose and (i + 1) % 50 == 0:
                        print(f"  Completed {i + 1}/{niter} iterations")

                except Exception as e:
                    warnings.warn(f"Knockoff iteration {i} failed: {e}")
    else:
        # Parallel execution (only works with default knockoffs/statistic)
        # Note: check for lambda (match_r wrapper) or non-default knockoffs
        is_default_knockoffs = (knockoffs is create_second_order or
                                (match_r and callable(knockoffs) and knockoffs.__name__ == '<lambda>'))
        if not is_default_knockoffs or statistic is not None:
            warnings.warn(
                "Parallel execution with custom knockoffs/statistic may not work. "
                "Falling back to sequential execution."
            )
            return knockoff_filter_voting(
                X, y, knockoffs=knockoffs, statistic=statistic,
                fdr=fdr, offset=offset, niter=niter, spec=spec,
                n_jobs=1, base_seed=base_seed, verbose=verbose,
                match_r=match_r, **kwargs
            )

        # Prepare arguments
        args_list = [
            (X, y, None, None, fdr, offset, base_seed + i)
            for i in range(niter)
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_run_single_knockoff, args) for args in args_list]

            for i, future in enumerate(as_completed(futures)):
                try:
                    selected = future.result()
                    for idx in selected:
                        selection_counts[idx] += 1
                except Exception as e:
                    warnings.warn(f"Knockoff iteration failed: {e}")

                if verbose and (i + 1) % 50 == 0:
                    print(f"  Completed {i + 1}/{niter} iterations")

    # Compute selection frequency (this is the "W" for voting)
    selection_frequency = selection_counts / niter

    # Select variables appearing in >= spec proportion of runs
    min_selections = int(np.ceil(niter * spec))
    selected = np.where(selection_counts >= min_selections)[0]
    selected = np.sort(selected)

    if verbose:
        print(f"  Selected {len(selected)} variables (>= {min_selections} selections)")

    return VotingResult(
        selection_counts=selection_counts,
        selection_frequency=selection_frequency,
        selected=selected,
        threshold=spec,
        niter=niter,
        spec=spec,
        min_selections=min_selections
    )
