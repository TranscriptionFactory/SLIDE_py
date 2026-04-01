# SLIDE_py Task Completion Checklist

## Code Quality Checks

### 1. Linting & Formatting
```bash
pixi run lint                    # Check for linting issues
pixi run fmt                     # Auto-format code
ruff check src/ --fix            # Fix auto-fixable issues
```

### 2. Import Validation  
```bash
pixi run check                   # Verify package imports correctly
python -c "import loveslide; print('Import successful')"
```

### 3. Type Checking (Optional)
```bash
# If mypy is available
mypy src/loveslide/
```

## Testing

### Run Test Suite
```bash
pixi run test                    # Full test suite
pytest tests/ -v                 # Verbose output
pytest tests/ -x                 # Stop on first failure
pytest tests/ -m "not slow"     # Skip slow tests for quick check
```

### Test Coverage (if coverage tools available)
```bash
pytest tests/ --cov=src/loveslide --cov-report=html
```

## Documentation

### 1. Update Docstrings
- Ensure all new public methods have docstrings
- Update existing docstrings if functionality changed
- Include parameter types and descriptions

### 2. Update README (if applicable)
- Add new features to README.md
- Update usage examples if API changed
- Update parameter tables if new parameters added

### 3. Add Examples (for new features)
- Create usage examples in `example/` directory
- Update notebook examples if relevant

## Version Control

### 1. Git Workflow
```bash
git status                       # Review changes
git add .                        # Stage all changes (or specific files)
git commit -m "descriptive message"  # Commit with clear message
```

### 2. Commit Message Guidelines
- Use present tense ("Add feature" not "Added feature")
- Be specific and descriptive
- Include issue numbers if applicable
- Format: `type: brief description`
  - `feat:` new features
  - `fix:` bug fixes  
  - `refactor:` code refactoring
  - `test:` adding tests
  - `docs:` documentation updates

## Performance Validation

### 1. Performance Tests (for performance-critical changes)
```bash
# Run performance-sensitive tests
pytest tests/ -m "performance" -v
```

### 2. Memory Usage (for large data changes)
- Monitor memory usage with large datasets
- Ensure no memory leaks in long-running processes

## Integration Testing

### 1. End-to-End Tests
```bash
# Test main pipeline with example data
python src/loveslide/slide.py --help  # Check CLI still works
```

### 2. R Integration Tests
- Ensure R integration still functions correctly
- Test with different R environments if applicable

## Final Checklist

- [ ] Code formatted and linted
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Examples work correctly
- [ ] Git committed with descriptive message
- [ ] No debug print statements left in code
- [ ] Performance hasn't regressed significantly
- [ ] R integration still functional (if modified)