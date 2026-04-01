# SLIDE_py Code Style & Conventions

## Code Style Standards

### Python Style
- **PEP 8** compliance enforced via **ruff** linter
- **Black** for consistent code formatting
- **Line length**: 88 characters (black default)
- **Import organization**: Standard library, third-party, local imports

### Naming Conventions
- **Classes**: PascalCase (e.g., `SLIDE`, `Knockoffs`, `Plotter`)
- **Functions/methods**: snake_case (e.g., `calc_default_fsize`, `run_pipeline`)
- **Variables**: snake_case (e.g., `input_params`, `z_matrix`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `SLIDE_TOP_FEATS`)
- **Private methods**: Leading underscore (e.g., `_validate_params`)

### Documentation Standards
- **Docstrings**: Use for all public classes and methods
- **Type hints**: Preferred but not strictly enforced
- **Comments**: Inline comments for complex logic
- **README.md**: Main documentation with usage examples

## File Organization

### Import Structure
```python
# Standard library imports
import os
import datetime
from glob import glob

# Third-party imports  
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from .tools import init_data
from .love import call_love
```

### Class Structure
- Main functionality in classes (e.g., `SLIDE`, `Knockoffs`)
- Utility functions in separate modules (e.g., `tools.py`)
- R integration encapsulated in specific methods

## Testing Conventions

### Test Organization
- **Location**: `tests/` directory
- **Naming**: `test_*.py` files with `test_*` functions
- **Structure**: Mirror source code structure
- **Markers**: Use `@pytest.mark.slow` for long-running tests

### Test Configuration (pytest.ini)
```ini
testpaths = tests
addopts = -v --tb=short -ra
markers = slow: marks tests as slow
filterwarnings = ignore::DeprecationWarning
```

## R Integration Patterns

### R Code Integration
- R functionality accessed via `rpy2`
- R scripts in subdirectories (e.g., `love_python/love/`)
- Error handling for R-Python interface issues
- R package dependencies managed via pixi/conda

### Data Exchange
- NumPy arrays for numerical data transfer
- Pandas DataFrames for structured data
- Explicit type conversion between R and Python

## Performance Considerations

### Memory Management
- Large datasets handled in chunks where possible
- Progress bars (`tqdm`) for long-running operations
- Parallel processing via `joblib` and `pqdm`

### Caching Strategy
- Results cached to disk when appropriate
- Pickle serialization for Python objects
- Output directories for organized results storage