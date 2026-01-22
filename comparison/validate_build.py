#!/usr/bin/env python3
"""
Build Check Validation Script for Python SLIDE

Comprehensive validation of Python SLIDE knockoff backends against R native baseline.
Checks for:
  - LOVE matrix accuracy (A, C, Z, Gamma)
  - Knockoff generation (W-statistics validity)
  - Selection consistency
  - No crashes/exceptions
  - Performance metrics

Usage:
    python validate_build.py <output_directory>
    python validate_build.py /path/to/build_check_outputs/20260122_143052

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import traceback


class BuildValidator:
    """Validate Python SLIDE build against R baseline."""

    def __init__(self, output_dir, tolerance=0.7):
        """
        Initialize validator.
        
        Args:
            output_dir: Path to timestamped build check output directory
            tolerance: Minimum correlation threshold for LOVE matrices (default: 0.7)
        """
        self.output_dir = Path(output_dir)
        self.tolerance = tolerance
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'output_dir': str(output_dir),
            'checks': {},
            'overall_status': 'UNKNOWN'
        }
        self.implementations = ['R_native', 'Py_pyLOVE_rKO', 'Py_pyLOVE_kf_glmnet', 'Py_pyLOVE_kf_sklearn']
        
    def log(self, message, level='INFO'):
        """Print formatted log message."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {level}: {message}")
        
    def check_task_completion(self):
        """Verify all tasks completed successfully."""
        self.log("Checking task completion markers...")
        status = {}
        
        for i, impl in enumerate(self.implementations):
            marker_file = self.output_dir / f".task{i}_complete"
            exists = marker_file.exists()
            status[impl] = exists
            
            if exists:
                self.log(f"  ✓ {impl}: Task completed", 'INFO')
            else:
                self.log(f"  ✗ {impl}: Task NOT completed", 'ERROR')
        
        self.results['checks']['task_completion'] = status
        return all(status.values())
    
    def check_output_files(self):
        """Verify expected output files exist."""
        self.log("Checking for required output files...")
        
        required_files = {
            'LOVE': ['A.csv', 'C.csv', 'z_matrix.csv', 'Gamma.csv', 'I.csv'],
            'Knockoff': ['selected_Z.csv', 'selected_Int.csv']
        }
        
        status = {}
        for impl in self.implementations:
            impl_dir = self.output_dir / impl
            impl_status = {'exists': impl_dir.exists(), 'files': {}}
            
            if impl_dir.exists():
                for category, files in required_files.items():
                    for fname in files:
                        fpath = impl_dir / fname
                        impl_status['files'][fname] = fpath.exists()
            
            status[impl] = impl_status
            
            # Report
            missing = [f for f, exists in impl_status['files'].items() if not exists]
            if missing:
                self.log(f"  ✗ {impl}: Missing files: {missing}", 'ERROR')
            else:
                self.log(f"  ✓ {impl}: All required files present", 'INFO')
        
        self.results['checks']['output_files'] = status
        return all(all(s['files'].values()) for s in status.values() if s['exists'])
    
    def load_csv(self, filepath, index_col=0):
        """Safely load CSV file."""
        try:
            if not filepath.exists():
                return None
            return pd.read_csv(filepath, index_col=index_col)
        except Exception as e:
            self.log(f"Error loading {filepath}: {e}", 'ERROR')
            return None
    
    def compare_matrices(self, mat1, mat2, name):
        """
        Compare two matrices and return correlation.
        
        Returns:
            dict with correlation, shape_match, and status
        """
        if mat1 is None or mat2 is None:
            return {'status': 'MISSING', 'correlation': None, 'shape_match': False}
        
        if mat1.shape != mat2.shape:
            return {'status': 'SHAPE_MISMATCH', 'correlation': None, 'shape_match': False,
                    'shape1': mat1.shape, 'shape2': mat2.shape}
        
        # Flatten and compute correlation
        arr1 = mat1.values.flatten()
        arr2 = mat2.values.flatten()
        
        # Remove NaN/inf
        valid_mask = np.isfinite(arr1) & np.isfinite(arr2)
        arr1 = arr1[valid_mask]
        arr2 = arr2[valid_mask]
        
        if len(arr1) == 0:
            return {'status': 'NO_VALID_DATA', 'correlation': None, 'shape_match': True}
        
        corr = np.corrcoef(arr1, arr2)[0, 1]
        
        passed = corr >= self.tolerance
        status = 'PASS' if passed else 'FAIL'
        
        return {
            'status': status,
            'correlation': float(corr),
            'shape_match': True,
            'shape': mat1.shape,
            'n_elements': len(arr1)
        }
    
    def validate_love_matrices(self):
        """Compare LOVE outputs between R native and Python implementations."""
        self.log("Validating LOVE matrices against R native baseline...")
        
        results = {}
        r_dir = self.output_dir / 'R_native'
        
        matrices = ['A', 'C', 'z_matrix', 'Gamma']
        
        for impl in self.implementations[1:]:  # Skip R_native (it's the baseline)
            impl_dir = self.output_dir / impl
            impl_results = {}
            
            self.log(f"  Comparing {impl} vs R_native...", 'INFO')
            
            for mat_name in matrices:
                r_mat = self.load_csv(r_dir / f"{mat_name}.csv")
                py_mat = self.load_csv(impl_dir / f"{mat_name}.csv")
                
                comparison = self.compare_matrices(r_mat, py_mat, mat_name)
                impl_results[mat_name] = comparison
                
                if comparison['status'] == 'PASS':
                    self.log(f"    ✓ {mat_name}: r={comparison['correlation']:.4f} (>= {self.tolerance})", 'INFO')
                elif comparison['status'] == 'FAIL':
                    self.log(f"    ✗ {mat_name}: r={comparison['correlation']:.4f} (< {self.tolerance})", 'ERROR')
                else:
                    self.log(f"    ⚠ {mat_name}: {comparison['status']}", 'WARN')
            
            results[impl] = impl_results
        
        self.results['checks']['love_matrices'] = results
        
        # Check if all Python implementations pass
        all_pass = all(
            all(mat['status'] == 'PASS' for mat in impl_results.values())
            for impl_results in results.values()
        )
        return all_pass
    
    def validate_pure_variables(self):
        """Check pure variable detection."""
        self.log("Validating pure variable detection...")
        
        results = {}
        r_dir = self.output_dir / 'R_native'
        r_pure = self.load_csv(r_dir / 'I.csv')
        
        for impl in self.implementations[1:]:
            impl_dir = self.output_dir / impl
            py_pure = self.load_csv(impl_dir / 'I.csv')
            
            if r_pure is None or py_pure is None:
                results[impl] = {'status': 'MISSING'}
                continue
            
            # Compare pure variable indices
            r_indices = set(r_pure.index)
            py_indices = set(py_pure.index)
            
            overlap = len(r_indices & py_indices)
            total = len(r_indices | py_indices)
            jaccard = overlap / total if total > 0 else 0
            
            results[impl] = {
                'status': 'PASS' if jaccard >= 0.5 else 'FAIL',
                'jaccard': float(jaccard),
                'r_count': len(r_indices),
                'py_count': len(py_indices),
                'overlap': overlap
            }
            
            if results[impl]['status'] == 'PASS':
                self.log(f"  ✓ {impl}: Jaccard={jaccard:.3f}, overlap={overlap}/{total}", 'INFO')
            else:
                self.log(f"  ✗ {impl}: Jaccard={jaccard:.3f}, overlap={overlap}/{total}", 'ERROR')
        
        self.results['checks']['pure_variables'] = results
        return all(r['status'] == 'PASS' for r in results.values() if r['status'] != 'MISSING')
    
    def validate_w_statistics(self):
        """Check W-statistics are valid (finite, non-NaN)."""
        self.log("Validating W-statistics...")
        
        results = {}
        
        for impl in self.implementations:
            impl_dir = self.output_dir / impl
            
            # Look for feature lists which contain W statistics
            feature_files = list(impl_dir.glob('feature_list_Z*.csv'))
            
            if not feature_files:
                results[impl] = {'status': 'MISSING', 'reason': 'No feature list files found'}
                self.log(f"  ⚠ {impl}: No feature list files found", 'WARN')
                continue
            
            all_valid = True
            issues = []
            
            for ffile in feature_files:
                df = self.load_csv(ffile, index_col=None)
                if df is None:
                    continue
                
                if 'W' in df.columns:
                    w_stats = df['W'].values
                    
                    # Check for NaN
                    if np.any(np.isnan(w_stats)):
                        all_valid = False
                        issues.append(f"{ffile.name}: Contains NaN")
                    
                    # Check for inf
                    if np.any(np.isinf(w_stats)):
                        all_valid = False
                        issues.append(f"{ffile.name}: Contains Inf")
            
            results[impl] = {
                'status': 'PASS' if all_valid else 'FAIL',
                'n_files': len(feature_files),
                'issues': issues
            }
            
            if all_valid:
                self.log(f"  ✓ {impl}: All W-statistics valid ({len(feature_files)} files)", 'INFO')
            else:
                self.log(f"  ✗ {impl}: W-statistics issues: {issues}", 'ERROR')
        
        self.results['checks']['w_statistics'] = results
        return all(r['status'] == 'PASS' for r in results.values())
    
    def validate_selections(self):
        """Check that at least one backend makes selections."""
        self.log("Validating knockoff selections...")
        
        results = {}
        
        for impl in self.implementations:
            impl_dir = self.output_dir / impl
            
            selected_z = self.load_csv(impl_dir / 'selected_Z.csv', index_col=None)
            selected_int = self.load_csv(impl_dir / 'selected_Int.csv', index_col=None)
            
            n_z = len(selected_z) if selected_z is not None else 0
            n_int = len(selected_int) if selected_int is not None else 0
            
            results[impl] = {
                'n_selected_Z': n_z,
                'n_selected_Int': n_int,
                'total': n_z + n_int
            }
            
            self.log(f"  {impl}: {n_z} latent factors, {n_int} interactions", 'INFO')
        
        self.results['checks']['selections'] = results
        
        # At least one implementation should make selections
        any_selections = any(r['total'] > 0 for r in results.values())
        return any_selections
    
    def check_for_errors(self):
        """Check log files for exceptions and errors."""
        self.log("Checking for errors in log files...")
        
        results = {}
        log_dir = Path(self.output_dir).parent.parent / 'logs'
        
        # Find most recent log files matching this job
        pattern = 'build_check_*.err'
        error_logs = sorted(log_dir.glob(pattern))[-4:] if log_dir.exists() else []
        
        for i, impl in enumerate(self.implementations):
            if i < len(error_logs):
                log_file = error_logs[i]
                
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                    
                    has_error = any(keyword in content.lower() for keyword in 
                                   ['error', 'exception', 'traceback', 'failed'])
                    
                    results[impl] = {
                        'status': 'FAIL' if has_error else 'PASS',
                        'log_file': str(log_file),
                        'has_errors': has_error
                    }
                    
                    if has_error:
                        self.log(f"  ✗ {impl}: Errors found in {log_file}", 'ERROR')
                    else:
                        self.log(f"  ✓ {impl}: No errors in logs", 'INFO')
                        
                except Exception as e:
                    results[impl] = {'status': 'UNKNOWN', 'error': str(e)}
            else:
                results[impl] = {'status': 'UNKNOWN', 'reason': 'Log file not found'}
        
        self.results['checks']['error_logs'] = results
        return all(r['status'] == 'PASS' for r in results.values() if r['status'] != 'UNKNOWN')
    
    def generate_summary_report(self):
        """Generate human-readable summary report."""
        self.log("\n" + "="*80)
        self.log("BUILD CHECK SUMMARY")
        self.log("="*80)
        
        checks = [
            ('Task Completion', 'task_completion'),
            ('Output Files', 'output_files'),
            ('LOVE Matrices', 'love_matrices'),
            ('Pure Variables', 'pure_variables'),
            ('W-Statistics', 'w_statistics'),
            ('Selections', 'selections'),
            ('Error Logs', 'error_logs')
        ]
        
        all_passed = True
        
        for check_name, check_key in checks:
            if check_key not in self.results['checks']:
                self.log(f"{check_name:20s}: SKIPPED", 'WARN')
                continue
            
            check_data = self.results['checks'][check_key]
            
            # Determine overall status for this check
            if isinstance(check_data, dict):
                if check_key in ['task_completion', 'output_files', 'error_logs']:
                    # These have per-implementation boolean/status values
                    statuses = []
                    for impl_data in check_data.values():
                        if isinstance(impl_data, bool):
                            statuses.append(impl_data)
                        elif isinstance(impl_data, dict) and 'status' in impl_data:
                            statuses.append(impl_data['status'] == 'PASS')
                        elif isinstance(impl_data, dict) and 'files' in impl_data:
                            statuses.append(all(impl_data['files'].values()))
                    passed = all(statuses) if statuses else False
                else:
                    # These have nested per-implementation results
                    statuses = []
                    for impl_results in check_data.values():
                        if isinstance(impl_results, dict):
                            if 'status' in impl_results:
                                statuses.append(impl_results['status'] == 'PASS')
                            else:
                                # For nested dicts (like love_matrices)
                                statuses.append(all(
                                    m.get('status') == 'PASS' 
                                    for m in impl_results.values() 
                                    if isinstance(m, dict)
                                ))
                    passed = all(statuses) if statuses else False
            elif isinstance(check_data, bool):
                passed = check_data
            else:
                passed = False
            
            status_str = "✓ PASS" if passed else "✗ FAIL"
            level = 'INFO' if passed else 'ERROR'
            self.log(f"{check_name:20s}: {status_str}", level)
            
            if not passed:
                all_passed = False
        
        self.log("="*80)
        
        if all_passed:
            self.log("OVERALL STATUS: ✓ ALL CHECKS PASSED", 'INFO')
            self.results['overall_status'] = 'PASS'
        else:
            self.log("OVERALL STATUS: ✗ ONE OR MORE CHECKS FAILED", 'ERROR')
            self.results['overall_status'] = 'FAIL'
        
        self.log("="*80 + "\n")
        
        return all_passed
    
    def save_results(self):
        """Save detailed results to JSON file."""
        output_file = self.output_dir / 'validation_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Detailed results saved to: {output_file}")
    
    def run_all_checks(self):
        """Run all validation checks."""
        self.log(f"Starting build validation for: {self.output_dir}")
        self.log(f"Correlation threshold: {self.tolerance}\n")
        
        try:
            self.check_task_completion()
            self.check_output_files()
            self.validate_love_matrices()
            self.validate_pure_variables()
            self.validate_w_statistics()
            self.validate_selections()
            self.check_for_errors()
            
            all_passed = self.generate_summary_report()
            self.save_results()
            
            return all_passed
            
        except Exception as e:
            self.log(f"Validation failed with exception: {e}", 'ERROR')
            self.log(traceback.format_exc(), 'ERROR')
            self.results['overall_status'] = 'ERROR'
            self.results['error'] = str(e)
            self.save_results()
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate Python SLIDE build check against R baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('output_dir', help='Path to timestamped build check output directory')
    parser.add_argument('--tolerance', type=float, default=0.7,
                       help='Minimum correlation threshold for LOVE matrices (default: 0.7)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"ERROR: Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    validator = BuildValidator(args.output_dir, tolerance=args.tolerance)
    passed = validator.run_all_checks()
    
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
