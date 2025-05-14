# %%

# %%
import pandas as pd
import numpy as np

# %%


# %%
input_params = {
    'x_path' : '/ix3/djishnu/alw399/SLIDE_PLM/data/slide/inputs/10x_200k_donor1_mait_X.csv',
    'y_path' : '/ix3/djishnu/alw399/SLIDE_PLM/data/slide/inputs/10x_200k_donor1_mait_y.csv',
    'niter' : 5,
    'spec' : 0.1,
    'fdr' : 0.5,
    'rep_CV' : 50,
    'pure_homo' : True,
    'lambda' : [0.5],
    'delta' : [0.1],
    'out_path': '/ix3/djishnu/alw399/SLIDE_py/test_results',
    'thresh_fdr': 0.2
}

# %%
import sys
sys.path.append('src/SLIDE')

# %%
from slide import OptimizeSLIDE
slider = OptimizeSLIDE(input_params)

slider.data.X = slider.data.X.iloc[:500, :500]
slider.data.Y = slider.data.Y.iloc[:500]

# %%
slider.run_pipeline(verbose=True, n_workers=1)
