import numpy as np
import pandas as pd
import os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import math
from pqdm.processes import pqdm
from functools import partial
from tqdm import tqdm
import copy

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, lasso_path

# Path to Python knockoffs package
KNOCKOFF_PYTHON_PATH = '/ix/djishnu/Aaron/1_general_use/knockoff-filter/knockoff-filter'


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
                                          method='asdp', shrink=False, **kwargs):
        """Run knockoff filter using pure Python package.

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
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        np.ndarray
            Indices of selected variables.
        """
        if KNOCKOFF_PYTHON_PATH not in sys.path:
            sys.path.insert(0, KNOCKOFF_PYTHON_PATH)

        from knockoff import knockoff_filter
        from knockoff.create import create_second_order

        # Create knockoff generator with specified parameters
        def knockoff_generator(X):
            return create_second_order(X, method=method, shrink=shrink)

        results = []
        for _ in range(niter):
            result = knockoff_filter(z, y.flatten(), fdr=fdr, knockoffs=knockoff_generator)
            if len(result.selected) > 0:
                results.extend(result.selected.tolist())

        if len(results) == 0:
            return np.array([], dtype=int)

        results = np.array(results)
        idx, counts = np.unique(results, return_counts=True)
        sig_idxs = idx[np.where(counts >= spec * niter)]

        return sig_idxs

    @staticmethod
    def _compute_glmnet_lambdasmax(X, Xk, y, nlambda=100, eps=0.001):
        """Compute W statistics matching R's stat.glmnet_lambdasmax.

        This implements the signed maximum of lasso path statistic using a
        grid of lambda values similar to glmnet's default behavior, rather
        than the exact LARS path used by knockpy's 'lsm' statistic.

        Parameters
        ----------
        X : np.ndarray
            Original feature matrix (n x p).
        Xk : np.ndarray
            Knockoff feature matrix (n x p).
        y : np.ndarray
            Response vector (n,).
        nlambda : int
            Number of lambda values in the grid (default 100, matching glmnet).
        eps : float
            Ratio of lambda_min/lambda_max (default 0.001, matching glmnet).

        Returns
        -------
        np.ndarray
            W statistics for each feature (p,).
        """
        n, p = X.shape
        y = y.flatten()

        # Combine X and knockoffs: [X, Xk]
        X_full = np.hstack([X, Xk])

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
            # Original feature entry time
            nonzero = np.where(np.abs(coef_path[j, :]) > 1e-10)[0]
            Z[j] = lambdas[nonzero[0]] if len(nonzero) > 0 else 0

            # Knockoff feature entry time
            nonzero_k = np.where(np.abs(coef_path[p + j, :]) > 1e-10)[0]
            Z_k[j] = lambdas[nonzero_k[0]] if len(nonzero_k) > 0 else 0

        # Compute signed max statistic: W_j = max(Z_j, Z_k_j) * sign(Z_j - Z_k_j)
        W = np.maximum(Z, Z_k) * np.sign(Z - Z_k)

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
                                           offset=0, fstat='lsm', **kwargs):
        """Run knockoff filter using knockpy package.

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
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        np.ndarray
            Indices of selected variables.
        """
        from knockpy import KnockoffFilter
        from knockpy.knockoffs import GaussianSampler

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

        # Create knockoff filter
        kfilter = KnockoffFilter(
            ksampler='gaussian',
            fstat=kp_fstat,
            knockoff_kwargs={'method': kp_method}
        )

        results = []
        for _ in range(niter):
            if use_glmnet_stat:
                # Use custom glmnet-equivalent statistic
                # First, generate knockoffs using knockpy's sampler
                ksampler = GaussianSampler(
                    X=z,
                    method=kp_method,
                    shrinkage=shrinkage
                )
                Xk = ksampler.sample_knockoffs()

                # Compute W using our glmnet-equivalent method
                W = Knockoffs._compute_glmnet_lambdasmax(z, Xk, y)

                # Compute threshold and select
                threshold = Knockoffs._knockoff_threshold(W, fdr, offset=offset)
                if threshold < np.inf:
                    selected = np.where(W >= threshold)[0]
                else:
                    selected = np.array([], dtype=int)
            else:
                # Use knockpy's built-in fstat
                rejections = kfilter.forward(
                    X=z,
                    y=y.flatten(),
                    fdr=fdr,
                    shrinkage=shrinkage
                )

                # Use knockpy's built-in selection with specified offset
                # This uses the data-dependent threshold from Barber & Candes (2015)
                selected_mask = kfilter.make_selections(fdr=fdr, offset=offset)
                selected = np.where(selected_mask)[0]

            if len(selected) > 0:
                results.extend(selected.tolist())

        if len(results) == 0:
            return np.array([], dtype=int)

        results = np.array(results)
        idx, counts = np.unique(results, return_counts=True)
        sig_idxs = idx[np.where(counts >= spec * niter)]

        return sig_idxs

    @staticmethod
    def filter_knockoffs_iterative(z, y, fdr=0.1, niter=1, spec=0.2, n_workers=1, backend='r',
                                   method='asdp', shrink=False, offset=0, fstat='lsm'):
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
            Number of parallel workers (unused currently).
        backend : str
            Which knockoff implementation: 'r' (default), 'python', or 'knockpy'.
        method : str
            Knockoff construction method:
            - For 'python' backend: 'asdp' (default), 'sdp', or 'equi'
            - For 'knockpy' backend: 'mvr' (default), 'sdp', 'equicorrelated', 'maxent', 'mmi'
        shrink : bool
            Whether to use Ledoit-Wolf covariance shrinkage (Python/knockpy backends only).
        offset : int
            Knockoff procedure offset (knockpy backend only):
            - 0: Original knockoff (more power, controls modified FDR)
            - 1: Knockoff+ (conservative, controls exact FDR)
        fstat : str
            Feature statistic method (knockpy backend only):
            - 'lsm': Signed maximum of lasso path (default)
            - 'glmnet': Grid-based lasso path (matches R's stat.glmnet_lambdasmax)
            - 'lasso': Cross-validated lasso
            - 'lcd': Lasso coefficient differences
            - 'ols': OLS coefficient differences

        Returns
        -------
        np.ndarray
            Indices of selected variables.
        """
        if backend == 'knockpy':
            return Knockoffs.filter_knockoffs_iterative_knockpy(
                z, y, fdr=fdr, niter=niter, spec=spec, method=method, shrink=shrink, offset=offset, fstat=fstat)
        elif backend == 'python':
            return Knockoffs.filter_knockoffs_iterative_python(
                z, y, fdr=fdr, niter=niter, spec=spec, method=method, shrink=shrink)
        else:
            return Knockoffs.filter_knockoffs_iterative_r(z, y, fdr=fdr, niter=niter, spec=spec)
    
    def fit_linear(self, z_matrix, y):
        '''fit z-matrix in linear part to get LP'''
        reg = self.model.fit(z_matrix, y)
        
        LP = reg.predict(z_matrix)
        beta = reg.coef_       

        return LP, beta


    @staticmethod
    def select_short_freq(z, y, spec=0.3, fdr=0.1, niter=1000, f_size=100, n_workers=1, backend='r',
                          method='asdp', shrink=False, fstat='lsm'):
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
            Number of parallel workers
        backend : str
            Which knockoff implementation: 'r' (default), 'python', or 'knockpy'.
        method : str
            Knockoff construction method:
            - For 'python' backend: 'asdp' (default), 'sdp', or 'equi'
            - For 'knockpy' backend: 'mvr' (default), 'sdp', 'equicorrelated', 'maxent', 'mmi'
        shrink : bool
            Whether to use Ledoit-Wolf covariance shrinkage (Python/knockpy backends only).
        fstat : str
            Feature statistic (knockpy backend only):
            - 'lsm': Signed max of lasso path (default)
            - 'glmnet': Grid-based lasso (matches R)
            - 'lasso': Cross-validated lasso

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
                method=method, shrink=shrink, fstat=fstat
            )

            selected_indices = selected_indices + start

            if len(selected_indices) > 0:
                screen_var.extend(selected_indices)

        screen_var = np.array(screen_var)

        if n_splits > 1 and len(screen_var) > 1:
            subset_z = z[:, screen_var]
            final_var = Knockoffs.filter_knockoffs_iterative(
                subset_z, y, fdr=fdr, niter=niter, spec=spec, n_workers=n_workers, backend=backend,
                method=method, shrink=shrink, fstat=fstat
            )
            final_var = screen_var[final_var]
        else:
            final_var = screen_var

        return final_var


