import pandas as pd
from easydict import EasyDict
from collections import defaultdict
import os


def init_data(input_params):

    data = EasyDict()
    input_params = defaultdict(lambda: None, input_params)


    if input_params['x_path'] is None:
        raise ValueError("x_path is not provided")
    
    if input_params['y_path'] is None:
        raise ValueError("y_path is not provided")
    
    if input_params['delta'] is None:
        input_params['delta'] = [0.05, 0.1]

    if input_params['lambda'] is None:
        input_params['lambda'] = [0.1]

    if input_params['niter'] is None:
        input_params['niter'] = 100

    if input_params['spec'] is None:
        input_params['spec'] = 0.2
        
    if input_params['fdr'] is None:
        input_params['fdr'] = 0.05

    if input_params['rep_CV'] is None:
        input_params['rep_CV'] = 50

    if input_params['out_path'] is None:
        input_params['out_path'] = os.getcwd()

    if input_params['thresh_fdr'] is None:
        input_params['thresh_fdr'] = 0.2
    
    data.X = pd.read_csv(input_params['x_path'], index_col=0)
    data.Y = pd.read_csv(input_params['y_path'], index_col=0)

    return data, input_params

def show_params(input_params, data):
    print(f'\n### PARAMETERS ###\n')
    for k, v in input_params.items():
        print(f"{k}: {v}")
    
    print(f'\n####### DATA #######\n')
    print(f'{data.Y.shape[0]} samples')
    print(f'{data.X.shape[1]} features')
    print(f'{(data.Y == 1).values.sum() / len(data.Y) * 100:.1f}% cases')
    print(f'{(data.Y == 0).values.sum() / len(data.Y) * 100:.1f}% controls')
    
    print(f'\n##################\n')



def calc_default_fsize(n_rows, K):
    """
    Calculate the default f_size.

    Parameters:
    - n_rows: integer representing the number of samples
    - K: integer representing the number of latent factors

    Returns:
    - Integer representing the default f_size
    """
    
    # written exactly as in the R code
    f_size = K 
    
    if (n_rows <= K) and (K < 100):
        if abs(n_rows - K) <= 2:
            f_size = n_rows - 2
        else:
            f_size = n_rows
            
    if (n_rows > K) and (K < 100):
        f_size = K
        
    if n_rows < K:
        f_size = n_rows
        
    return f_size




    