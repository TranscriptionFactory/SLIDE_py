# SLIDE_py Codebase Structure

## Project Layout
```
SLIDE_py/
├── src/loveslide/               # Main Python package
│   ├── __init__.py             # Package initialization
│   ├── slide.py                # Main SLIDE class and entry point
│   ├── love.py                 # LOVE algorithm implementation
│   ├── knockoffs.py            # Knockoffs statistical inference
│   ├── plotting.py             # Visualization utilities
│   ├── score.py                # Scoring and estimation classes
│   ├── cv.py                   # Cross-validation utilities
│   ├── tools.py                # Common utilities and helpers
│   │
│   ├── love_python/            # Python implementation of LOVE
│   │   ├── love/               # Core LOVE algorithms
│   │   └── setup.py            # LOVE Python setup
│   │
│   ├── knockoff/               # Knockoff inference components
│   │   ├── stats/              # Knockoff statistics implementations  
│   │   ├── _vendor/            # Vendored dependencies (glmnet)
│   │   ├── pydsdp/             # Python SDP solver interface
│   │   ├── create.py           # Knockoff generation
│   │   ├── filter.py           # Statistical filtering
│   │   ├── solve.py            # Optimization routines
│   │   └── utils.py            # Knockoff utilities
│   │
│   └── dsdp_solver/            # SDP solver components
│       ├── dsdp/               # DSDP solver interface
│       └── convert.py          # Data conversion utilities
│
├── tests/                      # Test suite
│   ├── pytest.ini             # Pytest configuration
│   └── test_*.py               # Test files
│
├── example/                    # Usage examples
├── dist/                       # Distribution packages
├── archive/                    # Archived runs and results
├── runs/                       # Current run outputs
│
├── pyproject.toml              # Python package configuration
├── pixi.toml                   # Pixi environment configuration  
├── setup.py                    # Setuptools configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Key Modules

### Core Components
- **slide.py**: Main `SLIDE` class, pipeline orchestration, parameter management
- **love.py**: Interface to LOVE algorithm (R integration via rpy2)
- **knockoffs.py**: Knockoff inference implementation and utilities
- **score.py**: `Estimator` and `SLIDE_Estimator` classes for model scoring
- **plotting.py**: `Plotter` class for visualization and diagnostics

### Algorithm Implementations
- **love_python/**: Pure Python implementation of LOVE algorithm
- **knockoff/**: Comprehensive knockoff inference framework
  - **stats/**: Different knockoff statistics (lasso, glmnet, etc.)
  - **create.py**: Knockoff variable generation
  - **filter.py**: FDR-controlled selection procedures

### Utilities
- **tools.py**: Data initialization, parameter validation, helper functions
- **cv.py**: Cross-validation utilities for model selection
- **dsdp_solver/**: SDP solver interface for optimization problems