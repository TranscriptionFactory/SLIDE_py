import numpy as np 
import pandas as pd 
import os, pickle
from concurrent.futures import ProcessPoolExecutor
import math
from pqdm.processes import pqdm
from functools import partial
from tqdm import tqdm
import copy

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
# from knockpy import KnockoffFilter


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
    def filter_knockoffs_iterative(z, y, fdr=0.1, niter=1, spec=0.2, n_workers=1):
        '''
        @return: mask of 0,1 significant interaction terms where 1 is significant
        '''
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        
        # Convert numpy arrays to R objects
        pandas2ri.activate()
        z_r = pandas2ri.py2rpy(pd.DataFrame(z))
        y_r = pandas2ri.py2rpy(pd.Series(y.flatten()))
        
        # Import R packages
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

        results = np.concatenate(results, axis=0)
        results = results - 1 # Convert to 0-based indexing

        idx, counts = np.unique(results, return_counts=True)
        sig_idxs = idx[np.where(counts >= spec * niter)]

        return sig_idxs
    
    def fit_linear(self, z_matrix, y):
        '''fit z-matrix in linear part to get LP'''
        reg = self.model.fit(z_matrix, y)
        
        LP = reg.predict(z_matrix)
        beta = reg.coef_       

        return LP, beta


    @staticmethod
    def select_short_freq(z, y, spec=0.3, fdr=0.1, niter=1000, f_size=100, n_workers=1):
        """
        Find significant variables using second order knockoffs across subsets of features.

        Parameters:
        -----------
        z : np.ndarray or pandas.DataFrame
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray or pandas.DataFrame
            Response vector of shape (n_samples,)
        spec : float
            Proportion threshold to consider a variable frequently selected
        fdr : float
            Target false discovery rate
        elbow : bool
            Whether to use elbow method to select frequent variables
        niter : int
            Number of knockoff iterations
        f_size : int
            Target size for each feature subset
        parallel : bool
            Whether to run iterations in parallel

        Returns:

        --------
        list
            List of selected variable indices
        """
        # Scale the input features
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

            # Run knockoffs on this subset
            selected_indices = Knockoffs.filter_knockoffs_iterative(subset_z, y, fdr=fdr, niter=niter, spec=spec, n_workers=n_workers)
        
            # Adjust indices to account for subset
            selected_indices = selected_indices + start
            
            if len(selected_indices) > 0:
                screen_var.extend(selected_indices)

        # Aggregation step if multiple splits

        screen_var = np.array(screen_var)

        if n_splits > 1 and len(screen_var) > 1:
            subset_z = z[:, screen_var]
            final_var = Knockoffs.filter_knockoffs_iterative(subset_z, y, fdr=fdr, niter=niter, spec=spec, n_workers=n_workers)
            final_var = screen_var[final_var] # index the candidate indices to get the final significant indices
        else:
            final_var = screen_var

        return final_var


