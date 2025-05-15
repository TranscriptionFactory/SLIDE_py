import numpy as np 
import pandas as pd
import os 
import pickle
from itertools import product
from tools import init_data, calc_default_fsize, show_params
from love import call_love
from knockoffs import Knockoffs
# from ER import EssentialRegression
from plotting import Plotter
from score import Estimator

class SLIDE:
    def __init__(self, input_params):
        self.data, self.input_params = init_data(input_params)

    def calc_default_fsize(self, K):
        n_rows = self.data.X.shape[0]
        return self.input_params.get('f_size', calc_default_fsize(n_rows, K))

    def show_params(self):
        show_params(self.input_params, self.data)

    def load_love(self, love_res_path):
        try:
            with open(love_res_path, 'rb') as f:
                self.love_result = pickle.load(f)

            self.A = pd.DataFrame(
                self.love_result['A'], 
                index=self.data.X.columns, 
                columns=[f"Z{i}" for i in range(self.love_result['A'].shape[1])]
            )

            self.latent_factors = self.calc_z_matrix(self.love_result)
        
        except Exception as e:
            print(f"Error loading LOVE result: {e}")
            return
    
    def get_aucs(self, z_matrix, n_iters=10, scaler='standard'):
        model = Estimator(model='linear', scaler=scaler)
        scores = model.evaluate(
            z_matrix, self.data.Y, 
            n_iters=n_iters
        )
        return scores
        


class OptimizeSLIDE(SLIDE):
    def __init__(self, input_params):
        super().__init__(input_params)
    
    def get_latent_factors(self, x, y, delta, mu=0.5, lbd=0.1, rep_CV=50, pure_homo=True, verbose=False, thresh_fdr=0.2, 
                           outpath='.', LOVE_version='LOVE'):
        
        assert LOVE_version in ['LOVE', 'SLIDE'], "LOVE_version must be either 'LOVE' or 'SLIDE'"
        
        love_result = call_love(
            X=x.values, 
            lbd=lbd, 
            mu=mu, 
            rep_CV=rep_CV, 
            pure_homo=pure_homo, 
            delta=delta, 
            verbose=verbose,
            thresh_fdr=thresh_fdr,
            LOVE_version=LOVE_version,
            outpath=outpath
        )
        self.love_result = love_result

        love_res_path = os.path.join(outpath, 'love_result.pkl')
        with open(love_res_path, 'wb') as f:
            pickle.dump(self.love_result, f)

        self.A = pd.DataFrame(
            love_result['A'], 
            index=x.columns, 
            columns=[f"Z{i}" for i in range(love_result['A'].shape[1])]
        )
        self.A.to_csv(
            os.path.join(outpath, 'A.csv')
        )

        ### Get thetas-- might not be necessary
        # er_result = EssentialRegression.fit_model(
        #     y = y, 
        #     x = x, 
        #     sigma = sigma,
        #     A_hat = love_result['A'],
        #     Gamma_hat = love_result['Gamma'],
        #     I_hat = love_result['pureVec']
        # )
        # self.er_result = er_result

        ### Get Z matrix from A and Gamma
        self.latent_factors = self.calc_z_matrix(love_result)

        self.latent_factors.to_csv(
            os.path.join(outpath, 'latent_factors.csv')
        )


    def calc_z_matrix(self, love_result):
        A_hat = love_result['A']
        Gamma_hat = love_result['Gamma']
        C_hat = love_result['C']
        X = self.data.X.values

        # Convert Gamma_hat to diagonal matrix and handle zeros
        Gamma_hat = np.where(Gamma_hat == 0, 1e-10, Gamma_hat)
        Gamma_hat_inv = np.diag(Gamma_hat ** (-1))

        # Calculate G_hat matrix
        G_hat = A_hat.T @ Gamma_hat_inv @ A_hat + np.linalg.inv(C_hat)

        # Calculate Z_hat matrix
        Z_hat = X @ Gamma_hat_inv @ A_hat @ np.linalg.pinv(G_hat)

        # Convert to DataFrame with appropriate column names
        Z = pd.DataFrame(
            Z_hat,
            index=self.data.X.index,
            columns=[f"Z{i}" for i in range(Z_hat.shape[1])]
        )
        return Z
    
    def find_standalone_LFs(self, latent_factors, spec, fdr, niter, f_size, n_workers=1):

        machop = Knockoffs(y = self.data.Y.values, z2 = latent_factors.values)

        marginal_idxs = machop.select_short_freq(
            z = latent_factors.values, 
            spec = spec, 
            fdr = fdr, 
            niter = niter, 
            f_size = f_size,
            n_workers = n_workers
        )

        self.marginal_idxs = marginal_idxs
        return machop
    
    def find_interaction_LFs(self, machop, spec, fdr, niter, f_size, n_workers=1):

        machop.add_z1(marginal_idxs=self.marginal_idxs)

        # Flatten interaction terms for knockoff selection
        interaction_terms = machop.interaction_terms.reshape(machop.n, -1)

        # Get significant interactions from flattened array
        sig_interactions = machop.select_short_freq(
            z = interaction_terms,
            spec = spec,
            fdr = fdr,
            niter = niter,
            f_size = f_size,
            n_workers = n_workers
        )

        if len(sig_interactions) == 0:
            self.interaction_pairs = np.array([])
        else:
            # Convert flattened indices back to original x,y coordinates
            x_indices = sig_interactions // machop.l  # Integer division gives x index
            y_indices = sig_interactions % machop.l   # Modulo gives y index

            self.interaction_pairs = np.array([x_indices, y_indices])

    def runSLIDE(self, latent_factors, niter, spec, fdr, verbose=False, n_workers=1, outpath='.'):
        
        f_size = self.calc_default_fsize(latent_factors.shape[1])

        if verbose:
            print(f'Calculated f_size: {f_size}')
            print(f'Finding standalone LF...')

        ### Find standalone LFs
        machop = self.find_standalone_LFs(latent_factors, spec, fdr, niter, f_size, n_workers)

        if len(self.marginal_idxs) == 0:
            print("No standalone LF found")
            return
        
        self.sig_LFs = [f"Z{i}" for i in self.marginal_idxs]
        np.save(os.path.join(outpath, 'sig_LFs.npy'), np.array(self.sig_LFs))

        ### Find interacting LFs
        if verbose:
            print(f'Found {len(self.marginal_idxs)} standalone LF')
            print(f'Finding interacting LF...')

        self.find_interaction_LFs(machop, spec, fdr, niter, f_size, n_workers)

        if verbose:
            print(f'Found {len(self.interaction_pairs)} interacting LF')

        self.sig_interacts = [f"Z{j}" for i, j in self.interaction_pairs]
        np.save(os.path.join(outpath, 'sig_interacts.npy'), np.array(self.sig_interacts))



    def get_LF_genes(self, lf, lf_thresh=0.2):
        """
        Returns a dictionary of lists, categorizing genes into positive and negative based on their loadings.
        
        Parameters:
        - lf: The name of the latent factor (column name in self.latent_factors).
        - lf_thresh: The threshold for the latent factor loadings.

        Returns:
        - Dictionary with 'positive' and 'negative' keys, containing lists of indices (gene names) for each.
        """

        if lf not in self.A.columns:
            raise ValueError(f"Latent factor {lf} not found in A matrix")

        contribution = self.A[lf]
        positive_genes = np.where(contribution > lf_thresh)[0]
        negative_genes = np.where(contribution < -lf_thresh)[0]

        # group = self.love_result['group'][int(lf.replace('Z', ''))]
        # genes = self.data.X.columns
        # positive_genes = genes[np.array(group['pos']) - 1]
        # negative_genes = genes[np.array(group['neg']) - 1] # +1 because LOVE uses 1-indexing

        # Sort genes by absolute loading value in descending order
        loadings = self.A[lf]
        positive_genes = positive_genes[np.argsort(np.abs(loadings[positive_genes]))[::-1]]
        negative_genes = negative_genes[np.argsort(np.abs(loadings[negative_genes]))[::-1]]

        # Return as a dictionary
        return {
            'pos': positive_genes, 
            'neg': negative_genes
        }

    def run_pipeline(self, verbose=True, n_workers=1, LOVE_version='LOVE'):
        if verbose:
            self.show_params()

        delta_iter = self.input_params['delta']
        lambda_iter = self.input_params['lambda']

        for delta_iter, lambda_iter in product(self.input_params['delta'], self.input_params['lambda']):
            
            out_iter = os.path.join(self.input_params['out_path'], f"{delta_iter}_{lambda_iter}")
            os.makedirs(out_iter, exist_ok=True)

            if verbose:
                print(f"Running LOVE with delta={delta_iter} and lambda={lambda_iter}")

            self.get_latent_factors(
                x=self.data.X, 
                y=self.data.Y, 
                delta=delta_iter,
                lbd=lambda_iter, 
                rep_CV=self.input_params['rep_CV'], 
                thresh_fdr=self.input_params['thresh_fdr'],
                pure_homo=self.input_params['pure_homo'],
                verbose=verbose,
                outpath=out_iter, 
                LOVE_version=LOVE_version
            )
            
            if verbose:
                print("\nRunning SLIDE knockoffs...")

            self.runSLIDE(
                self.latent_factors, 
                self.input_params['niter'], 
                self.input_params['spec'], 
                self.input_params['fdr'],
                verbose=verbose,
                n_workers=n_workers,
                outpath=out_iter
            )

            if verbose:
                print("\nSLIDE complete.")

            if len(self.marginal_idxs) == 0:
                return None

            sig_LF_genes = {lf: self.get_LF_genes(lf) for lf in self.sig_LFs}
            sig_interact_genes = {lf: self.get_LF_genes(lf) for lf in self.sig_interacts}

            Plotter.plot_latent_factors(sig_LF_genes, loadings=self.A, outdir=out_iter, title='marginal_LFs')
            Plotter.plot_latent_factors(sig_interact_genes, loadings=self.A, outdir=out_iter, title='interaction_LFs')

            self.score_performance(
                z1=self.latent_factors, 
                z2=self.latent_factors[self.sig_LFs], 
                z3=self.latent_factors[np.concatenate([self.sig_LFs, self.sig_interacts])], 
                n_iters=10, 
                scaler='standard', 
                outdir=out_iter
            )

    def score_performance(self, z1, z2, z3, n_iters=10, scaler='standard', outdir='.'):
        '''
        z1: all LFs from LOVE
        z2: marginal LFs from SLIDE
        z3: marginal + interaction LFs from SLIDE
        '''
        scores = {}
        
        scores['z_matrix'] = self.get_aucs(z1, n_iters, scaler)
        scores['marginals'] = self.get_aucs(z2, n_iters, scaler)
        scores['marginals&interactions'] = self.get_aucs(z3, n_iters, scaler)

        Plotter.plot_scores(scores, outdir=outdir, title='scores')





    


