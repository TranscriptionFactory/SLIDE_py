"""Main knockoff filter pipeline."""

from dataclasses import dataclass
from typing import Optional, Callable, Union, List, Dict, Any
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .create import create_second_order, create_fixed, KnockoffVariables


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


def knockoff_filter_voting(
    X: np.ndarray,
    y: np.ndarray,
    knockoffs: Optional[Callable] = None,
    statistic: Optional[Callable] = None,
    fdr: float = 0.10,
    offset: int = 1,
    niter: int = 500,
    spec: float = 0.1,
    n_jobs: int = 1,
    base_seed: int = 42,
    verbose: bool = False,
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
    offset : {0, 1}, default=1
        Offset for knockoff threshold (1 = knockoffs+).
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
    if knockoffs is None:
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
        if knockoffs is not create_second_order or statistic is not None:
            warnings.warn(
                "Parallel execution with custom knockoffs/statistic may not work. "
                "Falling back to sequential execution."
            )
            return knockoff_filter_voting(
                X, y, knockoffs=knockoffs, statistic=statistic,
                fdr=fdr, offset=offset, niter=niter, spec=spec,
                n_jobs=1, base_seed=base_seed, verbose=verbose, **kwargs
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
