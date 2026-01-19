import numpy as np
import pandas as pd
import os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import math
from pqdm.processes import pqdm
from functools import partial
from tqdm import tqdm
import copy
import logging

from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, lasso_path

logger = logging.getLogger(__name__)

# Path to Python knockoffs package
KNOCKOFF_PYTHON_PATH = '/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter'


def _single_knockoff_iteration_python(z, y, fdr, method, shrink, offset, statistic,
                                       mu, Sigma, diag_s):
    """Execute a single knockoff filter iteration with cached covariance.

    This function is designed to be called in parallel via joblib.
    """
    if KNOCKOFF_PYTHON_PATH not in sys.path:
        sys.path.insert(0, KNOCKOFF_PYTHON_PATH)

    from knockoff.filter import knockoff_filter, knockoff_threshold
    from knockoff.create import create_gaussian, KnockoffVariables

    # Generate knockoffs using cached SDP solution
    Xk = create_gaussian(z, mu, Sigma, method=method, diag_s=diag_s)

    # Compute statistics
    W = statistic(z, Xk, y.flatten())

    # Run the knockoff filter
    t = knockoff_threshold(W, fdr=fdr, offset=offset)
    selected = np.where(W >= t)[0]

    return selected.tolist() if len(selected) > 0 else []


def _single_knockoff_iteration_knockpy(z, y, fdr, kp_method, shrinkage, offset, kp_fstat,
                                        mu, Sigma, S):
    """Execute a single knockoff filter iteration with cached covariance for knockpy.

    This function is designed to be called in parallel via joblib.
    """
    from knockpy.knockoffs import GaussianSampler
    from knockpy.knockoff_filter import KnockoffFilter

    # Create sampler with cached covariance
    ksampler = GaussianSampler(X=z, mu=mu, Sigma=Sigma, S=S, method=kp_method)
    Xk = ksampler.sample_knockoffs()

    # Create filter and compute statistics
    kfilter = KnockoffFilter(ksampler=ksampler, fstat=kp_fstat)
    kfilter.forward(X=z, y=y.flatten(), Xk=Xk, fdr=fdr, shrinkage=shrinkage)

    # Make selections with specified offset
    selected_mask = kfilter.make_selections(fdr=fdr, offset=offset)
    selected = np.where(selected_mask)[0]

    return selected.tolist() if len(selected) > 0 else []


class Knockoffs():

    def __init__(self, y, z2, model='LR'):

        # self.z1 = self.scale_features(z1)
        self.z2 = self.scale_features(z2)
        self.y = y
        self.n = self.y.shape[0]
        # self.n, self.k = self.z1.shape
        self.l = self.z2.shape[1] 
        # self.interaction_terms = self.get_interaction_terms(self.z1, self.z2) 
        
        if model == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif model == 'LR':
            self.model = LinearRegression()
        else:
            raise ValueError('Model not supported')
    
    def add_z1(self, z1=None, marginal_idxs=None):
        if marginal_idxs is not None and z1 is None:
            z1 = self.z2[:, marginal_idxs]
            self.z2 = np.delete(self.z2, marginal_idxs, axis=1)

        n, self.k = z1.shape

        assert n == self.n

        self.z1 = self.scale_features(z1)
        self.interaction_terms = self.get_interaction_terms(self.z1, self.z2) 
    
    @staticmethod
    def scale_features(X, minmax=False, feature_range=(-1, 1)):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if minmax:
            scaler = MinMaxScaler(feature_range=feature_range)
        else:
            scaler = StandardScaler()
        
        scaler.fit(X)
        return scaler.transform(X)

    @staticmethod
    def get_interaction_terms(z_matrix, plm_embedding):
        '''
        @return: interactions in shape of (n_samples, n_LFs, plm_embed_dim)
        '''

        # If only one dimension, need to reshape to 2D for einsum to work as expected
        if len(z_matrix.shape) == 1:
            n = z_matrix.shape[0]
            z_matrix = z_matrix.reshape(n, -1)
        
        if len(plm_embedding.shape) == 1:
            n = plm_embedding.shape[0]
            plm_embedding = plm_embedding.reshape(n, -1)
        
        assert z_matrix.shape[0] == plm_embedding.shape[0]
        return np.einsum('ij,ik->ijk', z_matrix, plm_embedding)

    @staticmethod
    def filter_knockoffs_iterative_r(z, y, fdr=0.1, niter=1, spec=0.2, **kwargs):
        """Run knockoff filter using R package via rpy2."""
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        pandas2ri.activate()
        z_r = pandas2ri.py2rpy(pd.DataFrame(z))
        y_r = pandas2ri.py2rpy(pd.Series(y.flatten()))

        knockoff = importr('knockoff')

        results = []
        for _ in range(niter):
            result = knockoff.knockoff_filter(
                X=z_r,
                y=y_r,
                knockoffs=knockoff.create_second_order,
                statistic=knockoff.stat_glmnet_lambdasmax,
                offset=0,
                fdr=fdr
            )
            selected = result.rx2('selected')
            results.append(pandas2ri.rpy2py(selected))

        pandas2ri.deactivate()

        results = np.concatenate(results, axis=0)
        results = results - 1  # Convert to 0-based indexing

        idx, counts = np.unique(results, return_counts=True)
        sig_idxs = idx[np.where(counts >= spec * niter)]

        return sig_idxs

    @staticmethod
    def filter_knockoffs_iterative_python(z, y, fdr=0.1, niter=1, spec=0.2,
                                          method='asdp', shrink=False, offset=0,
                                          fstat='glmnet_lambdasmax', n_jobs=-1, **kwargs):
        """Run knockoff filter using pure Python package with parallel processing.

        Parameters
        ----------
        z : np.ndarray
            Feature matrix.
        y : np.ndarray
            Response vector.
        fdr : float
            Target false discovery rate.
        niter : int
            Number of knockoff iterations.
        spec : float
            Proportion threshold for selection frequency.
        method : str
            Knockoff construction method: 'asdp' (default), 'sdp', or 'equi'.
            - 'asdp': Approximate SDP, faster for high-dimensional data
            - 'sdp': Full semidefinite programming, more power but may fail
            - 'equi': Equicorrelated, always works but lower power
        shrink : bool
            Whether to use Ledoit-Wolf covariance shrinkage.
        offset : int
            Knockoff procedure offset:
            - 0: Original knockoff (more power, controls modified FDR) - R default
            - 1: Knockoff+ (conservative, controls exact FDR)
        fstat : str
            Feature statistic method:
            - 'glmnet_lambdasmax': Signed max of glmnet lasso path (matches R's stat.glmnet_lambdasmax)
            - 'glmnet_lambdadiff': Lambda difference statistic
            - 'glmnet_coefdiff': Coefficient difference at CV-selected lambda
        n_jobs : int
            Number of parallel jobs. -1 uses all available cores.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        np.ndarray
            Indices of selected variables.
        """
        if KNOCKOFF_PYTHON_PATH not in sys.path:
            sys.path.insert(0, KNOCKOFF_PYTHON_PATH)

        from knockoff.stats import stat_glmnet_lambdasmax, stat_glmnet_lambdadiff, stat_glmnet_coefdiff
        from knockoff.solve import create_solve_equi, create_solve_sdp, create_solve_asdp
        from knockoff.utils import is_posdef

        # Map fstat names to functions
        fstat_map = {
            'glmnet_lambdasmax': stat_glmnet_lambdasmax,
            'glmnet_lambdadiff': stat_glmnet_lambdadiff,
            'glmnet_coefdiff': stat_glmnet_coefdiff,
        }
        statistic = fstat_map.get(fstat, stat_glmnet_lambdasmax)

        z = np.asarray(z, dtype=np.float64)
        y = np.asarray(y)

        # Pre-compute covariance and SDP solution (OPTIMIZATION: avoid recomputing each iteration)
        mu = np.mean(z, axis=0)

        if not shrink:
            Sigma = np.cov(z, rowvar=False)
            if Sigma.ndim == 0:
                Sigma = np.array([[Sigma]])
            if not is_posdef(Sigma):
                shrink = True

        if shrink:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(z)
            Sigma = lw.covariance_

        # Solve SDP once (OPTIMIZATION: this is expensive and doesn't change between iterations)
        p = z.shape[1]
        if p <= 500 and method == 'asdp':
            method = 'sdp'  # Use full SDP for small problems

        logger.info(f"Pre-computing SDP solution for {p} features using method={method}")
        try:
            if method == 'equi':
                diag_s = create_solve_equi(Sigma)
            elif method == 'asdp':
                diag_s = create_solve_asdp(Sigma)
            else:
                diag_s = create_solve_sdp(Sigma)
        except ImportError as e:
            # Fall back to equi if cvxpy not available for SDP/ASDP
            logger.warning(f"SDP/ASDP method failed ({e}), falling back to equi method")
            method = 'equi'
            diag_s = create_solve_equi(Sigma)

        logger.info(f"Running {niter} knockoff iterations with {n_jobs} parallel jobs")

        # Run iterations in parallel (OPTIMIZATION: embarrassingly parallel)
        # Only use multiprocessing for large niter to justify process spawning overhead
        if n_jobs == 1 or niter < 200:
            # Sequential execution for small niter or explicit single-threaded
            results = []
            for _ in range(niter):
                selected = _single_knockoff_iteration_python(
                    z, y, fdr, method, shrink, offset, statistic, mu, Sigma, diag_s
                )
                results.extend(selected)
        else:
            # Parallel execution for large niter
            results_nested = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
                delayed(_single_knockoff_iteration_python)(
                    z, y, fdr, method, shrink, offset, statistic, mu, Sigma, diag_s
                )
                for _ in range(niter)
            )
            # Flatten results
            results = [item for sublist in results_nested for item in sublist]

        if len(results) == 0:
            return np.array([], dtype=int)

        results = np.array(results)
        idx, counts = np.unique(results, return_counts=True)
        sig_idxs = idx[np.where(counts >= spec * niter)]

        return sig_idxs

    @staticmethod
    def _compute_glmnet_lambdasmax(X, Xk, y, nlambda=500, eps=0.0005):
        """Compute W statistics matching R's stat.glmnet_lambdasmax.

        This implements the signed maximum of lasso path statistic using a
        grid of lambda values similar to glmnet's default behavior, rather
        than the exact LARS path used by knockpy's 'lsm' statistic.

        Includes the random swap symmetrization from R's implementation to
        ensure unbiased W statistics.

        Parameters
        ----------
        X : np.ndarray
            Original feature matrix (n x p).
        Xk : np.ndarray
            Knockoff feature matrix (n x p).
        y : np.ndarray
            Response vector (n,).
        nlambda : int
            Number of lambda values in the grid (default 500, matching glmnet).
        eps : float
            Ratio of lambda_min/lambda_max (default 0.0005, matching glmnet's 1/2000).

        Returns
        -------
        np.ndarray
            W statistics for each feature (p,).
        """
        n, p = X.shape
        y = y.flatten()

        # Random swap for symmetry (matching R's stat.glmnet_lambdasmax)
        # This ensures W statistics are unbiased w.r.t. the ordering of X vs Xk
        swap = np.random.binomial(1, 0.5, size=p)
        X_swap = X * (1 - swap) + Xk * swap
        Xk_swap = X * swap + Xk * (1 - swap)

        # Combine swapped X and knockoffs: [X_swap, Xk_swap]
        X_full = np.hstack([X_swap, Xk_swap])

        # Compute lambda grid (matching glmnet's log-scale grid)
        # lambda_max is where all coefficients become zero
        lambda_max = np.max(np.abs(X_full.T @ y)) / n
        lambda_min = lambda_max * eps
        lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), nlambda)

        # Compute lasso path using sklearn (matches glmnet coordinate descent)
        try:
            _, coef_path, _ = lasso_path(X_full, y, alphas=lambdas, max_iter=10000)
        except Exception:
            # Fallback to simpler grid if path computation fails
            _, coef_path, _ = lasso_path(X_full, y, n_alphas=nlambda, max_iter=10000)
            lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), coef_path.shape[1])

        # Find entry times (lambda at which each feature first becomes nonzero)
        Z = np.zeros(p)  # Entry times for original features
        Z_k = np.zeros(p)  # Entry times for knockoff features

        for j in range(p):
            # Original feature entry time (multiply by n to match R's glmnet behavior)
            nonzero = np.where(np.abs(coef_path[j, :]) > 1e-10)[0]
            Z[j] = lambdas[nonzero[0]] * n if len(nonzero) > 0 else 0

            # Knockoff feature entry time (multiply by n to match R's glmnet behavior)
            nonzero_k = np.where(np.abs(coef_path[p + j, :]) > 1e-10)[0]
            Z_k[j] = lambdas[nonzero_k[0]] * n if len(nonzero_k) > 0 else 0

        # Compute signed max statistic: W_j = max(Z_j, Z_k_j) * sign(Z_j - Z_k_j)
        W = np.maximum(Z, Z_k) * np.sign(Z - Z_k)

        # Adjust signs based on swap (matching R's behavior)
        W = W * (1 - 2 * swap)

        return W

    @staticmethod
    def _knockoff_threshold(W, fdr, offset=1):
        """Compute knockoff threshold with configurable offset.

        Parameters
        ----------
        W : np.ndarray
            Knockoff W statistics.
        fdr : float
            Target false discovery rate.
        offset : int
            0 for original knockoff (more power, controls modified FDR),
            1 for knockoff+ (conservative, controls FDR).

        Returns
        -------
        float
            Threshold value, or np.inf if no threshold satisfies FDR.
        """
        # Get unique positive W values as candidate thresholds (ascending order)
        W_abs = np.abs(W)
        candidates = np.sort(W_abs[W_abs > 0])

        # Find minimum threshold that satisfies FDR constraint
        threshold = np.inf
        for t in candidates:
            numerator = offset + np.sum(W <= -t)
            denominator = max(1, np.sum(W >= t))
            if numerator / denominator <= fdr:
                threshold = t
                break
        return threshold

    @staticmethod
    def filter_knockoffs_iterative_knockpy(z, y, fdr=0.1, niter=1, spec=0.2,
                                           method='mvr', shrink=False,
                                           offset=0, fstat='lsm', n_jobs=-1, **kwargs):
        """Run knockoff filter using knockpy package with parallel processing.

        Parameters
        ----------
        z : np.ndarray
            Feature matrix.
        y : np.ndarray
            Response vector.
        fdr : float
            Target false discovery rate.
        niter : int
            Number of knockoff iterations.
        spec : float
            Proportion threshold for selection frequency.
        method : str
            Knockoff construction method:
            - 'mvr': Minimum Variance-based Reconstructability (default, often best)
            - 'sdp': Semidefinite programming (uses DSDP if scikit-dsdp installed)
            - 'equicorrelated': Always works, lower power
            - 'maxent': Maximum entropy
            - 'mmi': Minimize mutual information
        shrink : bool
            Whether to use Ledoit-Wolf covariance shrinkage.
        offset : int
            Knockoff procedure offset:
            - 0: Original knockoff (more power, controls modified FDR)
            - 1: Knockoff+ (conservative, controls exact FDR)
            Default is 0 for more power.
        fstat : str
            Feature statistic method:
            - 'lsm': Signed maximum of lasso path (uses knockpy's lars_path)
            - 'glmnet': Grid-based lasso path (matches R's stat.glmnet_lambdasmax)
            - 'lasso': Cross-validated lasso coefficient differences
            - 'lcd': Lasso coefficient differences (no CV)
            - 'ols': OLS coefficient differences
            Default is 'lsm' to match R SLIDE.
        n_jobs : int
            Number of parallel jobs. -1 uses all available cores.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        np.ndarray
            Indices of selected variables.
        """
        from knockpy.knockoffs import GaussianSampler

        z = np.asarray(z, dtype=np.float64)
        y = np.asarray(y)

        # Map method names for compatibility
        method_map = {
            'asdp': 'sdp',  # knockpy doesn't have asdp, sdp uses DSDP
            'equi': 'equicorrelated',
        }
        kp_method = method_map.get(method, method)

        # Configure shrinkage
        shrinkage = 'ledoitwolf' if shrink else None

        # Handle custom 'glmnet' fstat
        use_glmnet_stat = (fstat == 'glmnet')
        kp_fstat = 'lsm' if use_glmnet_stat else fstat

        # Pre-compute covariance and S matrix (OPTIMIZATION: avoid recomputing each iteration)
        mu = np.mean(z, axis=0)

        if shrink:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(z)
            Sigma = lw.covariance_
        else:
            Sigma = np.cov(z, rowvar=False)
            if Sigma.ndim == 0:
                Sigma = np.array([[Sigma]])

        # Create a temporary sampler to compute S matrix once
        logger.info(f"Pre-computing knockoff S matrix for {z.shape[1]} features using method={kp_method}")
        temp_sampler = GaussianSampler(X=z, mu=mu, Sigma=Sigma, method=kp_method)
        S = temp_sampler.S  # Cache the S matrix

        logger.info(f"Running {niter} knockoff iterations with {n_jobs} parallel jobs")

        if use_glmnet_stat:
            # For glmnet stat, we need to use our custom implementation
            # This is harder to parallelize with knockpy's API, so use a different approach
            results = []
            for _ in range(niter):
                ksampler = GaussianSampler(X=z, mu=mu, Sigma=Sigma, S=S, method=kp_method)
                Xk = ksampler.sample_knockoffs()
                W = Knockoffs._compute_glmnet_lambdasmax(z, Xk, y)
                threshold = Knockoffs._knockoff_threshold(W, fdr, offset=offset)
                if threshold < np.inf:
                    selected = np.where(W >= threshold)[0]
                    results.extend(selected.tolist())
        else:
            # Run iterations in parallel (OPTIMIZATION: embarrassingly parallel)
            # Only use multiprocessing for large niter to justify process spawning overhead
            if n_jobs == 1 or niter < 200:
                # Sequential execution for small niter
                results = []
                for _ in range(niter):
                    selected = _single_knockoff_iteration_knockpy(
                        z, y, fdr, kp_method, shrinkage, offset, kp_fstat, mu, Sigma, S
                    )
                    results.extend(selected)
            else:
                # Parallel execution for large niter
                results_nested = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
                    delayed(_single_knockoff_iteration_knockpy)(
                        z, y, fdr, kp_method, shrinkage, offset, kp_fstat, mu, Sigma, S
                    )
                    for _ in range(niter)
                )
                # Flatten results
                results = [item for sublist in results_nested for item in sublist]

        if len(results) == 0:
            return np.array([], dtype=int)

        results = np.array(results)
        idx, counts = np.unique(results, return_counts=True)
        sig_idxs = idx[np.where(counts >= spec * niter)]

        return sig_idxs

    @staticmethod
    def filter_knockoffs_iterative(z, y, fdr=0.1, niter=1, spec=0.2, n_workers=-1, backend='r',
                                   method='asdp', shrink=False, offset=0, fstat='glmnet_lambdasmax'):
        """
        Run knockoff filter to find significant variables.

        Parameters
        ----------
        z : np.ndarray
            Feature matrix.
        y : np.ndarray
            Response vector.
        fdr : float
            Target false discovery rate.
        niter : int
            Number of knockoff iterations.
        spec : float
            Proportion threshold for selection frequency.
        n_workers : int
            Number of parallel workers. -1 uses all available cores. Default: -1.
        backend : str
            Which knockoff implementation: 'r' (default), 'python', or 'knockpy'.
        method : str
            Knockoff construction method:
            - For 'python' backend: 'asdp' (default), 'sdp', or 'equi'
            - For 'knockpy' backend: 'mvr' (default), 'sdp', 'equicorrelated', 'maxent', 'mmi'
        shrink : bool
            Whether to use Ledoit-Wolf covariance shrinkage (Python/knockpy backends only).
        offset : int
            Knockoff procedure offset:
            - 0: Original knockoff (more power, controls modified FDR) - R default
            - 1: Knockoff+ (conservative, controls exact FDR)
        fstat : str
            Feature statistic method:
            - For 'python' backend: 'glmnet_lambdasmax' (default), 'glmnet_lambdadiff', 'glmnet_coefdiff'
            - For 'knockpy' backend: 'lsm', 'glmnet', 'lasso', 'lcd', 'ols'

        Returns
        -------
        np.ndarray
            Indices of selected variables.
        """
        if backend == 'knockpy':
            return Knockoffs.filter_knockoffs_iterative_knockpy(
                z, y, fdr=fdr, niter=niter, spec=spec, method=method, shrink=shrink,
                offset=offset, fstat=fstat, n_jobs=n_workers)
        elif backend == 'python':
            return Knockoffs.filter_knockoffs_iterative_python(
                z, y, fdr=fdr, niter=niter, spec=spec, method=method, shrink=shrink,
                offset=offset, fstat=fstat, n_jobs=n_workers)
        else:
            return Knockoffs.filter_knockoffs_iterative_r(z, y, fdr=fdr, niter=niter, spec=spec)
    
    def fit_linear(self, z_matrix, y):
        '''fit z-matrix in linear part to get LP'''
        reg = self.model.fit(z_matrix, y)
        
        LP = reg.predict(z_matrix)
        beta = reg.coef_       

        return LP, beta


    @staticmethod
    def select_short_freq(z, y, spec=0.3, fdr=0.1, niter=1000, f_size=100, n_workers=-1, backend='r',
                          method='asdp', shrink=False, offset=0, fstat='glmnet_lambdasmax'):
        """
        Find significant variables using second order knockoffs across subsets of features.

        Parameters
        ----------
        z : np.ndarray or pandas.DataFrame
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray or pandas.DataFrame
            Response vector of shape (n_samples,)
        spec : float
            Proportion threshold to consider a variable frequently selected
        fdr : float
            Target false discovery rate
        niter : int
            Number of knockoff iterations
        f_size : int
            Target size for each feature subset
        n_workers : int
            Number of parallel workers. -1 uses all available cores. Default: -1.
        backend : str
            Which knockoff implementation: 'r' (default), 'python', or 'knockpy'.
        method : str
            Knockoff construction method:
            - For 'python' backend: 'asdp' (default), 'sdp', or 'equi'
            - For 'knockpy' backend: 'mvr' (default), 'sdp', 'equicorrelated', 'maxent', 'mmi'
        shrink : bool
            Whether to use Ledoit-Wolf covariance shrinkage (Python/knockpy backends only).
        offset : int
            Knockoff procedure offset:
            - 0: Original knockoff (more power, controls modified FDR) - R default
            - 1: Knockoff+ (conservative, controls exact FDR)
        fstat : str
            Feature statistic method:
            - For 'python' backend: 'glmnet_lambdasmax' (default), 'glmnet_lambdadiff', 'glmnet_coefdiff'
            - For 'knockpy' backend: 'lsm', 'glmnet', 'lasso', 'lcd', 'ols'

        Returns
        -------
        np.ndarray
            Array of selected variable indices
        """
        z = Knockoffs.scale_features(z)
        y = y.copy()

        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values

        if isinstance(z, pd.DataFrame):
            z = z.values

        n_features = z.shape[1]
        n_splits = math.ceil(n_features / f_size)
        feature_split = math.ceil(n_features / n_splits)
        feature_starts = list(range(0, n_features, feature_split))
        feature_stops = [min(start + feature_split, n_features) for start in feature_starts]

        screen_var = []

        for start, stop in tqdm(zip(feature_starts, feature_stops),
                                total=len(feature_starts),
                                desc="Processing subsets"):

            subset_z = z[:, start:stop]

            selected_indices = Knockoffs.filter_knockoffs_iterative(
                subset_z, y, fdr=fdr, niter=niter, spec=spec, n_workers=n_workers, backend=backend,
                method=method, shrink=shrink, offset=offset, fstat=fstat
            )

            selected_indices = selected_indices + start

            if len(selected_indices) > 0:
                screen_var.extend(selected_indices)

        screen_var = np.array(screen_var)

        if n_splits > 1 and len(screen_var) > 1:
            subset_z = z[:, screen_var]
            final_var = Knockoffs.filter_knockoffs_iterative(
                subset_z, y, fdr=fdr, niter=niter, spec=spec, n_workers=n_workers, backend=backend,
                method=method, shrink=shrink, offset=offset, fstat=fstat
            )
            final_var = screen_var[final_var]
        else:
            final_var = screen_var

        return final_var


