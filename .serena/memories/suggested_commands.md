# SLIDE_py Development Commands

## Environment Setup
```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Activate development environment
pixi shell -e dev

# Or use specific environment
pixi shell -e default  # Core dependencies only
pixi shell -e dev      # Development tools + R integration + viz
```

## Development Commands

### Testing
```bash
pixi run test                    # Run all tests with pixi
pytest tests/ -v                 # Direct pytest command
pytest tests/ -m "not slow"     # Skip slow tests
```

### Code Quality
```bash
pixi run lint                    # Run ruff linting
pixi run fmt                     # Format code with ruff
ruff check src/                  # Direct ruff check
ruff format src/                 # Direct ruff format
black src/                       # Alternative formatting with black
```

### Package Validation
```bash
pixi run check                   # Check loveslide import
python -c "import loveslide; print(loveslide.__version__)"
```

## Running the Application

### Command Line Interface
```bash
# From src/loveslide/ directory
python slide.py \
  --x_path /path/to/features.csv \
  --y_path /path/to/labels.csv \
  --out_path /path/to/output/
```

### Programmatic Usage
```python
from loveslide import OptimizeSLIDE

params = {
    'x_path': '/path/to/features.csv',
    'y_path': '/path/to/labels.csv', 
    'out_path': '/path/to/output/',
    # ... other parameters
}

slider = OptimizeSLIDE(params)
slider.run_pipeline(verbose=True, n_workers=1)
```

## Git Workflow
```bash
git status                       # Check working directory status
git add .                        # Stage changes
git commit -m "description"      # Commit changes
git push                         # Push to remote

# Development branches
git checkout -b feat/feature-name
git checkout -b fix/bug-name
```

## System Utilities (Linux)
```bash
ls -la                          # List directory contents
find . -name "*.py" -type f     # Find Python files
grep -r "pattern" src/          # Search in source code
cd /path/to/directory           # Change directory
```

## Package Building & Installation
```bash
pip install -e .               # Editable install
pip install loveslide          # Install from PyPI (when available)
python setup.py build_ext --inplace  # Build extensions
```