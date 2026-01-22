# SLIDE Build Check System

Automated build validation for Python SLIDE implementation testing knockoff backends against R native baseline.

## Quick Start

```bash
cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

# Submit build check (uses HIV example data, ~10 min total)
bash submit_build_check.sh
```

The script will:
1. Submit SLURM array job (4 parallel tasks)
2. Monitor job progress
3. Run validation automatically when complete
4. Display pass/fail summary
5. Exit with code 0 (pass) or 1 (fail)

## What Gets Tested

### Implementations Compared
- **R_native**: R SLIDE package (baseline reference)
- **Py_pyLOVE_rKO**: Python LOVE + R knockoffs
- **Py_pyLOVE_kf_glmnet**: Python LOVE + knockoff-filter (glmnet)
- **Py_pyLOVE_kf_sklearn**: Python LOVE + knockoff-filter (sklearn)

### Validation Checks
1. **Task Completion**: All 4 tasks finished successfully
2. **Output Files**: Required LOVE and knockoff files present
3. **LOVE Matrices**: A, C, Z, Gamma correlation ≥ 0.7 vs R baseline
4. **Pure Variables**: Jaccard similarity ≥ 0.5 with R
5. **W-Statistics**: All finite (no NaN/Inf values)
6. **Selections**: At least one backend makes feature selections
7. **Error Logs**: No exceptions or crashes in stderr logs

## Files

### Configuration
- `build_check_config.yaml` - Test configuration (HIV example data, single delta/lambda)

### Scripts
- `submit_build_check.sh` - Main entry point (submits, monitors, validates)
- `run_build_check.sh` - SLURM array job script (4 tasks)
- `validate_build.py` - Comprehensive validation with detailed diagnostics

## Usage Examples

### Basic Usage
```bash
# Use default config
bash submit_build_check.sh

# Specify custom config
bash submit_build_check.sh my_config.yaml

# Override output path
bash submit_build_check.sh build_check_config.yaml /path/to/output
```

### Direct SLURM Submission (No Auto-Validation)
```bash
sbatch run_build_check.sh build_check_config.yaml
```

### Manual Validation
```bash
# After job completes, validate manually
python validate_build.py /path/to/build_check_outputs/20260122_143052

# Adjust correlation threshold
python validate_build.py /path/to/output --tolerance 0.8
```

### Monitor Running Job
```bash
# Check job status
squeue -u $USER

# Watch log output (use actual job ID)
tail -f logs/build_check_123456_0.out

# Check task completion
ls build_check_outputs/*/.*_complete
```

## Output Structure

```
build_check_outputs/
  20260122_143052/              # Timestamped run
    R_native/                   # Baseline R implementation
      A.csv, C.csv, z_matrix.csv, Gamma.csv, I.csv
      selected_Z.csv, selected_Int.csv
      feature_list_Z*.csv
    Py_pyLOVE_rKO/              # Python with R knockoffs
      [same structure]
    Py_pyLOVE_kf_glmnet/        # Python with knockoff-filter (glmnet)
      [same structure]
    Py_pyLOVE_kf_sklearn/       # Python with knockoff-filter (sklearn)
      [same structure]
    validation_results.json     # Detailed validation report
    .task0_complete             # Completion markers
    .task1_complete
    .task2_complete
    .task3_complete
```

## Exit Codes

- `0` - All validation checks passed
- `1` - One or more checks failed or error occurred

## Validation Thresholds

Current acceptance criteria:
- LOVE matrix correlation: ≥ 0.7 vs R native
- Pure variable Jaccard: ≥ 0.5 vs R native
- W-statistics: All finite (no NaN/Inf)
- At least one backend must make selections

These thresholds reflect documented differences between R and Python implementations (RNG, glmnet vs sklearn, etc.) and are considered acceptable for production use.

## Customizing Tests

### Test Different Data
Edit `build_check_config.yaml`:
```yaml
x_path: /path/to/your/X.csv
y_path: /path/to/your/Y.csv
```

### Test Multiple Parameter Combinations
Add more delta/lambda values:
```yaml
delta:
  - 0.05
  - 0.1
  - 0.2
lambda:
  - 0.1
  - 0.5
  - 1.0
```
Note: This increases runtime proportionally (9x for 3×3 grid).

### Adjust Validation Strictness
```bash
python validate_build.py /path/to/output --tolerance 0.8  # Stricter
python validate_build.py /path/to/output --tolerance 0.6  # More lenient
```

## Troubleshooting

### Job Fails to Start
- Check SLURM queue: `squeue -u $USER`
- Check quotas: `quota -s`
- Verify modules load: `module load gcc/12.2.0 python/ondemand-jupyter-python3.11 r/4.4.0`

### Validation Fails
1. Check detailed results: `cat build_check_outputs/*/validation_results.json`
2. Review error logs: `cat logs/build_check_*_*.err`
3. Check for missing dependencies: Python knockoff package, R SLIDE package
4. Verify data integrity: Are X and Y valid CSV files?

### Output Directory Not Found
- Wait longer - job may be queued or starting
- Check base path exists: `ls -ld build_check_outputs`
- Verify permissions: Can you write to output directory?

### W-Statistics All NaN
- Usually indicates knockoff generation failed
- Check if covariance matrix is positive definite
- Try different knockoff method (equicorrelated fallback)

## Integration with CI/CD

The build check can be integrated into automated workflows:

```bash
#!/bin/bash
# CI/CD integration example

cd /ix/djishnu/Aaron/1_general_use/SLIDE_py/comparison

# Run build check
bash submit_build_check.sh

# Exit code propagates
if [ $? -eq 0 ]; then
    echo "Build validated - proceeding with deployment"
else
    echo "Build validation failed - blocking deployment"
    exit 1
fi
```

## Expected Runtime

With HIV example data (61 samples × 80 features):
- R native: ~2-3 min
- Python implementations: ~2-3 min each
- Total: ~10 min including job queue time
- Validation: ~30 seconds

## Related Scripts

- `run_comparison.sh` - Full 5-way comparison with multiple backends
- `run_knockoff_comparison.sh` - Detailed knockoff backend comparison
- `compare_outputs.py` - Element-wise numerical comparison
- `compare_latent_factors.py` - Semantic factor comparison
- `compare_full.py` - Combined comprehensive report

## Contact

For issues or questions about the build check system, contact: aar126@pitt.edu
