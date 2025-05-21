import numpy as np 
import pandas as pd
import os 
import pickle
from tqdm import tqdm
from itertools import product
from collections import defaultdict

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
        """
        Calculate the default feature size for the SLIDE algorithm.
        This breaks the data into many chunks to run knockoffs on each separately.

        Args:
            K (int): Number of latent factors.
        """
        n_rows = self.data.X.shape[0]
        return self.input_params.get('f_size', calc_default_fsize(n_rows, K))

    def show_params(self):
        """
        Display the parameters and data.
        """
        show_params(self.input_params, self.data)

    def load_love(self, love_res_path):
        """
        Load the LOVE result and calculate the latent factors.
        This is used when we want to continue running SLIDE from a previous LOVE run.
        """
        try:
            with open(love_res_path, 'rb') as f:
                love_result = pickle.load(f)

            self.A = pd.DataFrame(
                love_result['A'], 
                index=self.data.X.columns, 
                columns=[f"Z{i}" for i in range(love_result['A'].shape[1])]
            )

            self.latent_factors = self.calc_z_matrix(love_result)
            self.love_result = love_result
        
        except Exception as e:
            print(f"Error loading LOVE result: {e}")
            return
    
    def get_aucs(self, z_matrix, n_iters=10, test_size=0.2, scaler='standard'):
        """
        Get the AUCs on a sample x feature matrix. Each iter will fit and test on a different
        random sample of the data.

        Args:
            z_matrix (pd.DataFrame): The z_matrix to evaluate.
            n_iters (int): The number of iterations to run.
            test_size (float): The percentage of data to use as a test set.
            scaler (str): The scaler to use.
        """
        model = Estimator(model='linear', scaler=scaler)
        scores = model.evaluate(
            X=z_matrix, 
            y=self.data.Y, 
            n_iters=n_iters,
            test_size=test_size,
        )
        return scores
    
    def score_performance(self, s1, s2, n_iters=100, test_size=0.2, scaler='standard', outdir='.'):
        """
        Score the performance of the given latent factors relative to each other and to random
        selection of marginals and interactions.
        This visualization is similar to the control plot in original R code.

        s1: standalone LFs 
        s2: interacting LF terms (marginal * interacting)
        """
        scores = defaultdict(list)

        scores['s1'] = self.get_aucs(s1, n_iters, test_size=test_size, scaler=scaler)
        scores['s2'] = self.get_aucs(s2, n_iters, test_size=test_size, scaler=scaler)
        scores['s3'] = self.get_aucs(np.concatenate([s1, s2], axis=1), n_iters, test_size=test_size, scaler=scaler)

        num_marginals = s1.shape[1]
        num_interactions = s2.shape[1]
        n = self.data.X.shape[0]
        real_interaction_terms = Knockoffs.get_interaction_terms(
            self.latent_factors.loc[:, ~self.latent_factors.columns.isin(self.sig_interacts + self.sig_LFs)],  # remove real sig LFs
            s1
        ).reshape(n, -1)


        for _ in tqdm(range(n_iters)):
            s1_random = self.latent_factors.iloc[:, np.random.choice(
                self.latent_factors.shape[1], num_marginals, replace=False
            )]

            interaction_terms = Knockoffs.get_interaction_terms(self.latent_factors, s1_random).reshape(n, -1)
            s2_random = interaction_terms[:, np.random.choice(interaction_terms.shape[1], num_interactions, replace=False)]
            
            s3_random = np.concatenate([s1_random.values, s2_random], axis=1)
            scores['full_random'].append(self.get_aucs(s3_random, n_iters=1, test_size=test_size, scaler=scaler))

            s2_real = real_interaction_terms[:, np.random.choice(real_interaction_terms.shape[1], num_interactions, replace=False)]
            s3_real = np.concatenate([s1.values, s2_real], axis=1)
            scores['partial_random'].append(self.get_aucs(s3_real, n_iters=1, test_size=test_size, scaler=scaler))

        scores['full_random'] = np.array(scores['full_random']).flatten()
        scores['partial_random'] = np.array(scores['partial_random']).flatten()

        Plotter.plot_controlplot(scores, outdir=outdir, title='control_plot')
        return scores

    # def score_performance(self, s1, s2, n_iters=10, scaler='standard', outdir='.'):
    #     '''
    #     Score the performance of the given latent factors relative to each other and to random
    #     selection of marginals and interactions.
    #     This is a new visualization that is not in the original SLIDE paper.

    #     s1: standalone LFs 
    #     s2: interacting LF terms (marginal * interacting)
    #     '''
    #     scores = {}
        
    #     scores['X'] = self.get_aucs(self.data.X, n_iters, scaler)
    #     scores['z_matrix'] = self.get_aucs(self.latent_factors, n_iters, scaler)
    #     scores['s1'] = self.get_aucs(s1, n_iters, scaler)
    #     scores['s2'] = self.get_aucs(s2, n_iters, scaler)
    #     scores['s3'] = self.get_aucs(np.concatenate([s1, s2], axis=1), n_iters, scaler)

    #     ### Compare to randomly selected marginals ###
    #     num_marginals = s1.shape[1]
    #     s1_random = self.latent_factors.iloc[:, np.random.choice(
    #         self.latent_factors.shape[1], num_marginals, replace=False
    #     )]
    #     scores['s1_random'] = self.get_aucs(s1_random, n_iters, scaler)

    #     num_interactions = s2.shape[1]
    #     n = self.data.X.shape[0]
    #     s2_random = Knockoffs.get_interaction_terms(self.latent_factors, s1_random).reshape(n, -1)
    #     s2_random = s2_random[:, np.random.choice(s2_random.shape[1], num_interactions, replace=False)]
    #     scores['s2_random'] = self.get_aucs(s2_random, n_iters, scaler)

    #     s3_random = np.concatenate([s1_random, s2_random], axis=1)
    #     scores['s3_random'] = self.get_aucs(s3_random, n_iters, scaler)

    #     Plotter.plot_scores(scores, outdir=outdir, title='scores')
    #     Plotter.plot_controlplot(scores, outdir=outdir, title='')



class OptimizeSLIDE(SLIDE):
    def __init__(self, input_params):
        super().__init__(input_params)
    
    def get_latent_factors(self, x, y, delta, mu=0.5, lbd=0.1, rep_CV=50, pure_homo=True, verbose=False, thresh_fdr=0.2, 
                           outpath='.', LOVE_version='LOVE'):
        """
        Get the latent factors (aka z_matrix) from the LOVE algorithm.

        Args:
            x (pd.DataFrame): The input data.
            y (pd.Series): The target variable.
            delta (float): The delta parameter.
            mu (float): The mu parameter. Set to 0.5 by default.
            lbd (float): The lambda parameter.
            rep_CV (int): The number of cross-validation folds.
            pure_homo (bool): Whether to use the pure homoscedastic model. Newest LOVE implementation uses False
            verbose (bool): Whether to print verbose output.
            thresh_fdr (float): a numerical constant used for thresholding the correlation matrix to
                                control the false discovery rate
            outpath (str): The path to save the LOVE result.
            LOVE_version (str): original LOVE or SLIDE implementation of LOVE
        """

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

        ### Solve for Z matrix from A, Gamma, and C
        self.latent_factors = self.calc_z_matrix(love_result)

        self.latent_factors.to_csv(
            os.path.join(outpath, 'z_matrix.csv')
        )


    def calc_z_matrix(self, love_result):
        A_hat = love_result['A']
        Gamma_hat = love_result['Gamma']
        C_hat = love_result['C']

        # Z-score X
        x = self.data.X.values
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

        # Convert Gamma_hat to diagonal matrix and handle zeros
        Gamma_hat = np.where(Gamma_hat == 0, 1e-10, Gamma_hat)
        Gamma_hat_inv = np.diag(Gamma_hat ** (-1))

        # Calculate G_hat matrix
        G_hat = A_hat.T @ Gamma_hat_inv @ A_hat + np.linalg.inv(C_hat)

        # Calculate Z_hat matrix
        Z_hat = x @ Gamma_hat_inv @ A_hat @ np.linalg.pinv(G_hat)

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
            n_candidates = machop.z2.shape[1]
            marginal_lf = self.marginal_idxs[sig_interactions // n_candidates]
            z2_cols = np.array([i for i in range(self.latent_factors.shape[1]) if i not in self.marginal_idxs])

            assert len(z2_cols) == n_candidates, "Number of candidates does not match (implementation error)"

            interacting_lf = z2_cols[sig_interactions % n_candidates] 
            self.interaction_pairs = np.array([marginal_lf, interacting_lf])
            self.interaction_terms = interaction_terms[:, sig_interactions]
    

    def run_SLIDE(self, latent_factors, niter, spec, fdr, verbose=False, n_workers=1, outpath='.'):
        
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
        np.savetxt(os.path.join(outpath, 'sig_LFs.txt'), self.sig_LFs, fmt='%s')

        ### Find interacting LFs
        if verbose:
            print(f'Found {len(self.marginal_idxs)} standalone LF')
            print(f'Finding interacting LF...')

        self.find_interaction_LFs(machop, spec, fdr, niter, f_size, n_workers)

        if verbose:
            print(f'Found {len(self.interaction_pairs)} interacting LF')

        self.sig_interacts = [f"Z{j}" for i, j in self.interaction_pairs.T]
        np.savetxt(os.path.join(outpath, 'sig_interacts.txt'), self.sig_interacts, fmt='%s')


    def get_LF_genes(self, lf, lf_thresh=0.03, top_feats=10, outpath=None):
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
        positive_genes = contribution[contribution > lf_thresh]
        negative_genes = contribution[contribution < -lf_thresh]

        # group = self.love_result['group'][int(lf.replace('Z', ''))]
        # genes = self.data.X.columns
        # positive_genes = genes[np.array(group['pos']) - 1]
        # negative_genes = genes[np.array(group['neg']) - 1] # +1 because LOVE uses 1-indexing

        # Sort genes by absolute loading value in descending order
    
        if outpath is not None:
            all_genes = pd.concat([positive_genes, negative_genes])
            all_genes = all_genes.sort_values(key=abs, ascending=False)
            # Save both gene names and their loading values
            np.savetxt(os.path.join(outpath, f'feature_list_{lf}.txt'), 
                      np.column_stack((all_genes.index, all_genes.values)), 
                      fmt='%s\t%.6f')
    
        all_genes = all_genes[:top_feats]
        
        # Return as a dictionary
        return {
            'pos': [x for x in positive_genes.index if x in all_genes], 
            'neg': [x for x in negative_genes.index if x in all_genes]
        }

    def run_pipeline(self, verbose=True, n_workers=1, LOVE_version='LOVE'):
        
        if verbose:
            self.show_params()

        delta_iter = self.input_params['delta']
        lambda_iter = self.input_params['lambda']

        for delta_iter, lambda_iter in product(self.input_params['delta'], self.input_params['lambda']):
            
            out_iter = os.path.join(self.input_params['out_path'], f"{delta_iter}_{lambda_iter}_out")
            os.makedirs(out_iter, exist_ok=True)

            if verbose:
                print(f"Running LOVE with delta={delta_iter} and lambda={lambda_iter}")

            try:
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
            except Exception as e:
                print(f"\nError running LOVE: {e}\n")
                print('##################\n')

                continue
            
            if verbose:
                print("\nRunning SLIDE knockoffs...")

            self.run_SLIDE(
                latent_factors=self.latent_factors, 
                niter=self.input_params['niter'], 
                spec=self.input_params['spec'], 
                fdr=self.input_params['fdr'],
                verbose=verbose,
                n_workers=n_workers,
                outpath=out_iter
            )

            if verbose:
                print("\nSLIDE complete.")

            if len(self.marginal_idxs) > 0:

                sig_LF_genes = {str(lf): self.get_LF_genes(
                    lf=lf, 
                    top_feats=self.input_params['SLIDE_top_feats'],
                    outpath=out_iter) for lf in self.sig_LFs}
                Plotter.plot_latent_factors(sig_LF_genes, loadings=self.A, outdir=out_iter, title='marginal_LFs')

                sig_interact_genes = {str(lf): self.get_LF_genes(
                    lf=lf, 
                    top_feats=self.input_params['SLIDE_top_feats'],
                    outpath=out_iter) for lf in self.sig_interacts}
                
                if len(sig_interact_genes) > 0:
                    Plotter.plot_latent_factors(sig_interact_genes, loadings=self.A, outdir=out_iter, title='interaction_LFs')

                self.score_performance(
                    s1=self.latent_factors[self.sig_LFs], 
                    s2=self.interaction_terms, 
                    n_iters=10, 
                    test_size=0.2,
                    scaler='standard', 
                    outdir=out_iter
                )

            if verbose:
                print(f"\nCompleted {delta_iter}_{lambda_iter}\n")
                print('##################\n')

    