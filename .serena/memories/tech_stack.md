# SLIDE_py Tech Stack

## Core Technologies

### Languages
- **Python 3.11+**: Main language for orchestration and interfaces
- **R 4.4+**: Core statistical computations via rpy2 integration

### Scientific Computing Stack
- **numpy** (>=1.20): Numerical computing foundation
- **scipy** (>=1.7): Scientific algorithms and statistics
- **scikit-learn** (>=1.0): Machine learning utilities
- **pandas** (>=2.0): Data manipulation and analysis
- **statsmodels** (>=0.14): Additional statistical tools

### Optimization & Solvers
- **cvxpy** (>=1.3): Convex optimization (SDP solver fallback)
- R packages: knockoff, glmnet, MASS, lpsolve, linprog

### Visualization
- **matplotlib** (>=3.7): Base plotting
- **seaborn** (>=0.12): Statistical visualization
- **networkx** (>=3.0): Network/graph analysis
- Optional: altair, wordcloud, adjusttext

### Development & Build Tools
- **pixi**: Environment and dependency management
- **pytest** (>=7.0): Testing framework
- **ruff** (>=0.1): Linting and formatting
- **black** (>=23.0): Code formatting

### Utilities
- **joblib** (>=0.14.1): Parallel computing
- **tqdm** (>=4.60): Progress bars
- **pyyaml** (>=6.0): Configuration files
- **pqdm**: Parallel processing with progress bars
- **easydict**: Enhanced dictionary functionality

## R Integration
- **rpy2** (>=3.5): Python-R interface
- R statistical packages loaded dynamically as needed