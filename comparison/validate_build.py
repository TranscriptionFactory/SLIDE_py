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
import subprocess
import tempfile
import shutil
import pickle


# Find Rscript path
RSCRIPT_PATH = os.environ.get('RSCRIPT_PATH')
if not RSCRIPT_PATH:
    hpc_paths = [
        '/software/rhel9/manual/install/r/4.5.0/bin/Rscript',
        '/software/rhel9/manual/install/r/4.4.0/bin/Rscript',
        '/usr/bin/Rscript',
    ]
    for path in hpc_paths:
        if os.path.exists(path):
            RSCRIPT_PATH = path
            break
    else:
        RSCRIPT_PATH = shutil.which('Rscript')


def read_rds_matrix(rds_path: Path, key: str) -> pd.DataFrame:
    """
    Read a matrix from an RDS file using R subprocess.
    
    Args:
        rds_path: Path to the RDS file
        key: Key for list elements (e.g., 'A', 'C', 'Gamma' for AllLatentFactors.rds)
    
    Returns:
        DataFrame with the matrix contents, or None if failed
    """
    if not RSCRIPT_PATH:
        return None
    
    if not rds_path.exists():
        return None
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_csv = tmp.name
    
    try:
        r_script = f'''
data <- readRDS("{rds_path}")
if (is.list(data) && "{key}" %in% names(data)) {{
    mat <- data${key}
    if (is.matrix(mat) || is.data.frame(mat)) {{
        write.csv(mat, "{tmp_csv}", row.names=TRUE)
    }} else if (is.vector(mat)) {{
        write.csv(data.frame({key}=mat), "{tmp_csv}", row.names=TRUE)
    }}
}} else {{
    quit(status=1)
}}
'''
        result = subprocess.run(
            [RSCRIPT_PATH, '-e', r_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return None
        
        if not os.path.exists(tmp_csv) or os.path.getsize(tmp_csv) == 0:
            return None
            
        df = pd.read_csv(tmp_csv, index_col=0)
        return df
        
    except Exception:
        return None
    finally:
        Path(tmp_csv).unlink(missing_ok=True)


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

    def find_param_dirs(self, impl_dir: Path) -> list:
        """Find parameter output directories within an implementation directory.
        
        These are directories ending in '_out' (e.g., '0.1_0.5_out').
        """
        if not impl_dir.exists():
            return []
        return sorted([d for d in impl_dir.iterdir() if d.is_dir() and d.name.endswith('_out')])

    def get_first_param_dir(self, impl_dir: Path) -> Path:
        """Get the first (or only) parameter directory for validation.
        
        For build checks, we typically have a single parameter combination.
        Returns impl_dir itself if no param dirs found (backward compatibility).
        """
        param_dirs = self.find_param_dirs(impl_dir)
        if param_dirs:
            return param_dirs[0]
        return impl_dir
        
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
        
        # Different required files for R vs Python implementations
        # R outputs mostly RDS files, Python outputs CSV files
        required_files_python = {
            'LOVE': ['A.csv', 'z_matrix.csv'],
            'Knockoff': ['sig_LFs.txt', 'sig_interacts.txt']
        }
        required_files_r = {
            'LOVE': ['z_matrix.csv'],  # R may only produce z_matrix.csv as CSV
            'Knockoff': ['feature_list_Z*.txt']  # R uses feature_list files, not sig_LFs.txt
        }
        
        status = {}
        for impl in self.implementations:
            impl_dir = self.output_dir / impl
            impl_status = {'exists': impl_dir.exists(), 'files': {}}
            
            is_r = impl.startswith('R_')
            required_files = required_files_r if is_r else required_files_python
            
            if impl_dir.exists():
                for category, files in required_files.items():
                    for fname in files:
                        # Use rglob to search recursively for the file
                        matches = list(impl_dir.rglob(fname))
                        impl_status['files'][fname] = str(matches[0]) if matches else None
            
            status[impl] = impl_status
            
            # Report
            missing = [f for f, path in impl_status['files'].items() if path is None]
            if missing:
                self.log(f"  ✗ {impl}: Missing files: {missing}", 'ERROR')
            else:
                self.log(f"  ✓ {impl}: All required files present", 'INFO')
        
        self.results['checks']['output_files'] = status
        return all(all(p is not None for p in s['files'].values()) for s in status.values() if s['exists'])
    
    def load_csv(self, filepath, index_col=0):
        """Safely load CSV file."""
        try:
            if not filepath.exists():
                return None
            return pd.read_csv(filepath, index_col=index_col)
        except Exception as e:
            self.log(f"Error loading {filepath}: {e}", 'ERROR')
            return None

    def load_python_love_result(self, impl_dir: Path) -> dict:
        """Load Python LOVE result from love_result.pkl.
        
        Keys in love_result.pkl:
        - K, pureVec, pureInd, group, A, C, Omega, Gamma, optDelta
        
        pureInd structure: list of dicts with 'pos' and 'neg' keys containing feature indices
        
        Returns:
            dict with matrices as DataFrames, or empty dict if not found
        """
        pkl_path = impl_dir / 'love_result.pkl'
        if not pkl_path.exists():
            return {}
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            result = {}
            
            # A matrix (features x LFs)
            if 'A' in data and data['A'] is not None:
                A = data['A']
                if isinstance(A, np.ndarray):
                    result['A'] = pd.DataFrame(A)
                elif isinstance(A, pd.DataFrame):
                    result['A'] = A
            
            # C matrix (LF covariance)
            if 'C' in data and data['C'] is not None:
                C = data['C']
                if isinstance(C, np.ndarray):
                    result['C'] = pd.DataFrame(C)
                elif isinstance(C, pd.DataFrame):
                    result['C'] = C
            
            # Gamma (error variance)
            if 'Gamma' in data and data['Gamma'] is not None:
                Gamma = data['Gamma']
                if isinstance(Gamma, np.ndarray):
                    result['Gamma'] = pd.DataFrame(Gamma.flatten(), columns=['Gamma'])
                elif isinstance(Gamma, pd.DataFrame):
                    result['Gamma'] = Gamma
                else:
                    result['Gamma'] = pd.DataFrame([Gamma], columns=['Gamma'])
            
            # Pure variable indices (I) - pureInd is a list of dicts with 'pos' and 'neg' keys
            if 'pureInd' in data and data['pureInd'] is not None:
                pureInd = data['pureInd']
                all_indices = set()
                
                if isinstance(pureInd, list):
                    # Each element is a dict with 'pos' and 'neg' keys
                    for item in pureInd:
                        if isinstance(item, dict):
                            for key in ['pos', 'neg']:
                                if key in item and item[key]:
                                    for idx in item[key]:
                                        all_indices.add(int(idx))
                        elif isinstance(item, (int, np.integer)):
                            all_indices.add(int(item))
                elif isinstance(pureInd, np.ndarray):
                    all_indices = set(int(x) for x in pureInd.flatten())
                
                if all_indices:
                    result['I'] = all_indices
            
            return result
            
        except Exception as e:
            self.log(f"Error loading {pkl_path}: {e}", 'ERROR')
            return {}
    
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
        """Compare LOVE outputs between implementations.
        
        R outputs are stored in AllLatentFactors.rds (containing A, C, Gamma).
        Python implementations output A.csv, z_matrix.csv, and optionally C.csv, Gamma.csv, I.csv.
        
        Strategy:
        1. Compare z_matrix across all implementations (both R and Python have this as CSV)
        2. Compare A, C, Gamma: Python (from love_result.pkl) vs R (from AllLatentFactors.rds)
        3. Compare A, C, Gamma between Python implementations (from love_result.pkl)
        """
        self.log("Validating LOVE matrices...")
        
        results = {}
        
        # Get all param directories and load Python LOVE results
        impl_param_dirs = {}
        py_love_results = {}
        for impl in self.implementations:
            impl_base_dir = self.output_dir / impl
            impl_dir = self.get_first_param_dir(impl_base_dir)
            impl_param_dirs[impl] = impl_dir
            
            if not impl.startswith('R_'):
                py_love_results[impl] = self.load_python_love_result(impl_dir)
        
        # 1. Compare z_matrix across ALL implementations (including R vs Python)
        self.log("  Comparing z_matrix across all implementations...", 'INFO')
        ref_impl = self.implementations[0]  # R_native
        ref_dir = impl_param_dirs[ref_impl]
        ref_z = self.load_csv(ref_dir / 'z_matrix.csv')
        
        for impl in self.implementations[1:]:
            impl_dir = impl_param_dirs[impl]
            impl_z = self.load_csv(impl_dir / 'z_matrix.csv')
            
            comparison = self.compare_matrices(ref_z, impl_z, 'z_matrix')
            results.setdefault(impl, {})['z_matrix'] = comparison
            
            if comparison['status'] == 'PASS':
                self.log(f"    ✓ {impl} z_matrix: r={comparison['correlation']:.4f}", 'INFO')
            elif comparison['status'] == 'FAIL':
                self.log(f"    ✗ {impl} z_matrix: r={comparison['correlation']:.4f}", 'ERROR')
            else:
                self.log(f"    ⚠ {impl} z_matrix: {comparison['status']}", 'WARN')
        
        # 2. Compare A, C, Gamma: Python (from love_result.pkl) vs R (from AllLatentFactors.rds)
        r_impl = 'R_native'
        if r_impl in impl_param_dirs:
            r_dir = impl_param_dirs[r_impl]
            rds_path = r_dir / 'AllLatentFactors.rds'
            
            if rds_path.exists() and RSCRIPT_PATH:
                self.log("  Comparing Python LOVE matrices (A, C, Gamma) against R native...", 'INFO')
                
                for mat_name in ['A', 'C', 'Gamma']:
                    # Load R matrix from RDS
                    r_mat = read_rds_matrix(rds_path, mat_name)
                    if r_mat is None:
                        self.log(f"    ⚠ Could not extract {mat_name} from R AllLatentFactors.rds", 'WARN')
                        continue
                    
                    # Compare each Python implementation against R
                    py_impls = [impl for impl in self.implementations if not impl.startswith('R_')]
                    for impl in py_impls:
                        # Try love_result.pkl first, then CSV
                        py_mat = py_love_results.get(impl, {}).get(mat_name)
                        if py_mat is None:
                            impl_dir = impl_param_dirs[impl]
                            py_mat = self.load_csv(impl_dir / f'{mat_name}.csv')
                        
                        if py_mat is None:
                            self.log(f"    ⚠ {impl} {mat_name}: not found in love_result.pkl or CSV", 'WARN')
                            continue
                        
                        # For A matrix, compare absolute values (sign may differ)
                        if mat_name == 'A':
                            comparison = self.compare_matrices(
                                pd.DataFrame(np.abs(r_mat.values), index=r_mat.index, columns=r_mat.columns),
                                pd.DataFrame(np.abs(py_mat.values), index=py_mat.index if hasattr(py_mat, 'index') else None, 
                                           columns=py_mat.columns if hasattr(py_mat, 'columns') else None),
                                f'{mat_name}_vs_R'
                            )
                        else:
                            comparison = self.compare_matrices(r_mat, py_mat, f'{mat_name}_vs_R')
                        
                        results.setdefault(impl, {})[f'{mat_name}_vs_R'] = comparison
                        
                        if comparison['status'] == 'PASS':
                            self.log(f"    ✓ {impl} {mat_name} vs R: r={comparison['correlation']:.4f}", 'INFO')
                        elif comparison['status'] == 'FAIL':
                            self.log(f"    ✗ {impl} {mat_name} vs R: r={comparison['correlation']:.4f}", 'ERROR')
                        else:
                            self.log(f"    ⚠ {impl} {mat_name} vs R: {comparison['status']}", 'WARN')
            else:
                if not rds_path.exists():
                    self.log("  Skipping R comparison: AllLatentFactors.rds not found", 'WARN')
                elif not RSCRIPT_PATH:
                    self.log("  Skipping R comparison: Rscript not available", 'WARN')
        
        # 3. Compare Python LOVE matrices (A, C, Gamma) between Python implementations
        py_impls = [impl for impl in self.implementations if not impl.startswith('R_')]
        if len(py_impls) >= 2:
            self.log("  Comparing Python LOVE matrices (A, C, Gamma) between Python implementations...", 'INFO')
            py_ref = py_impls[0]
            
            for mat_name in ['A', 'C', 'Gamma']:
                # Get reference from love_result.pkl or CSV
                py_ref_mat = py_love_results.get(py_ref, {}).get(mat_name)
                if py_ref_mat is None:
                    py_ref_dir = impl_param_dirs[py_ref]
                    py_ref_mat = self.load_csv(py_ref_dir / f'{mat_name}.csv')
                
                if py_ref_mat is None:
                    continue
                    
                for impl in py_impls[1:]:
                    # Get comparison target from love_result.pkl or CSV
                    impl_mat = py_love_results.get(impl, {}).get(mat_name)
                    if impl_mat is None:
                        impl_dir = impl_param_dirs[impl]
                        impl_mat = self.load_csv(impl_dir / f'{mat_name}.csv')
                    
                    if impl_mat is None:
                        continue
                    
                    comparison = self.compare_matrices(py_ref_mat, impl_mat, mat_name)
                    results.setdefault(impl, {})[mat_name] = comparison
                    
                    if comparison['status'] == 'PASS':
                        self.log(f"    ✓ {impl} {mat_name}: r={comparison['correlation']:.4f}", 'INFO')
                    elif comparison['status'] == 'FAIL':
                        self.log(f"    ✗ {impl} {mat_name}: r={comparison['correlation']:.4f}", 'ERROR')
                    else:
                        self.log(f"    ⚠ {impl} {mat_name}: {comparison['status']}", 'WARN')
        
        self.results['checks']['love_matrices'] = results
        
        # Check if z_matrix passes for all implementations (main validation)
        z_pass = all(
            impl_results.get('z_matrix', {}).get('status') == 'PASS'
            for impl_results in results.values()
        )
        return z_pass
    
    def validate_pure_variables(self):
        """Check pure variable detection.
        
        Compare Python implementations against R (from AllLatentFactors.rds$I)
        and against each other (from love_result.pkl$pureInd).
        """
        self.log("Validating pure variable detection...")
        
        results = {}
        
        # Get all param directories
        impl_param_dirs = {}
        py_love_results = {}
        for impl in self.implementations:
            impl_base_dir = self.output_dir / impl
            impl_dir = self.get_first_param_dir(impl_base_dir)
            impl_param_dirs[impl] = impl_dir
            
            if not impl.startswith('R_'):
                py_love_results[impl] = self.load_python_love_result(impl_dir)
        
        # Load R pure variable indices from AllLatentFactors.rds
        r_impl = 'R_native'
        r_indices = None
        if r_impl in impl_param_dirs:
            r_dir = impl_param_dirs[r_impl]
            rds_path = r_dir / 'AllLatentFactors.rds'
            
            if rds_path.exists() and RSCRIPT_PATH:
                r_I = read_rds_matrix(rds_path, 'I')
                if r_I is not None:
                    # R indices are 1-based, convert to set
                    r_indices = set(int(x) for x in r_I.values.flatten() if pd.notna(x))
        
        py_impls = [impl for impl in self.implementations if not impl.startswith('R_')]
        
        # Compare Python vs R
        if r_indices is not None:
            self.log("  Comparing Python pure variables against R native...", 'INFO')
            for impl in py_impls:
                py_I = py_love_results.get(impl, {}).get('I')
                if py_I is None:
                    continue
                
                # Python pureInd is 0-based, R is 1-based - convert Python to 1-based
                py_I_1based = set(x + 1 for x in py_I)
                overlap = len(r_indices & py_I_1based)
                total = len(r_indices | py_I_1based)
                jaccard = overlap / total if total > 0 else 0
                
                results[f'{impl}_vs_R'] = {
                    'status': 'PASS' if jaccard >= 0.5 else 'FAIL',
                    'jaccard': float(jaccard),
                    'r_count': len(r_indices),
                    'py_count': len(py_I),
                    'overlap': overlap
                }
                
                if results[f'{impl}_vs_R']['status'] == 'PASS':
                    self.log(f"    ✓ {impl} vs R: Jaccard={jaccard:.3f}, overlap={overlap}/{total}", 'INFO')
                else:
                    self.log(f"    ✗ {impl} vs R: Jaccard={jaccard:.3f}, overlap={overlap}/{total}", 'ERROR')
        
        # Compare Python implementations against each other
        if len(py_impls) >= 2:
            self.log("  Comparing Python pure variables against each other...", 'INFO')
            ref_impl = py_impls[0]
            ref_I = py_love_results.get(ref_impl, {}).get('I')
            
            if ref_I is not None:
                for impl in py_impls[1:]:
                    py_I = py_love_results.get(impl, {}).get('I')
                    if py_I is None:
                        results[impl] = {'status': 'MISSING'}
                        continue
                    
                    overlap = len(ref_I & py_I)
                    total = len(ref_I | py_I)
                    jaccard = overlap / total if total > 0 else 0
                    
                    results[impl] = {
                        'status': 'PASS' if jaccard >= 0.5 else 'FAIL',
                        'jaccard': float(jaccard),
                        'ref_count': len(ref_I),
                        'py_count': len(py_I),
                        'overlap': overlap
                    }
                    
                    if results[impl]['status'] == 'PASS':
                        self.log(f"    ✓ {ref_impl} vs {impl}: Jaccard={jaccard:.3f}, overlap={overlap}/{total}", 'INFO')
                    else:
                        self.log(f"    ✗ {ref_impl} vs {impl}: Jaccard={jaccard:.3f}, overlap={overlap}/{total}", 'ERROR')
            else:
                self.log(f"  Reference {ref_impl} has no pureInd in love_result.pkl", 'WARN')
        
        if not results:
            self.log("  No pure variable data found to compare", 'WARN')
            return True
        
        self.results['checks']['pure_variables'] = results
        return all(r['status'] == 'PASS' for r in results.values() if r.get('status') != 'MISSING')
    
    def validate_w_statistics(self):
        """Check feature list files are valid (no NaN/Inf in numeric columns)."""
        self.log("Validating feature list files...")
        
        results = {}
        
        for impl in self.implementations:
            impl_base_dir = self.output_dir / impl
            impl_dir = self.get_first_param_dir(impl_base_dir)
            
            is_r = impl.startswith('R_')
            
            # Look for feature lists (search recursively)
            # Python uses .csv, R uses .txt (but both are tab-separated with headers)
            if is_r:
                feature_files = list(impl_dir.rglob('feature_list_Z*.txt'))
            else:
                feature_files = list(impl_dir.rglob('feature_list_Z*.csv'))
            
            if not feature_files:
                results[impl] = {'status': 'MISSING', 'reason': 'No feature list files found'}
                self.log(f"  ⚠ {impl}: No feature list files found", 'WARN')
                continue
            
            all_valid = True
            issues = []
            
            for ffile in feature_files:
                try:
                    # Both R (.txt) and Python (.csv) feature lists are tab-separated
                    df = pd.read_csv(ffile, sep='\t', index_col=0)
                    if df is None or df.empty:
                        issues.append(f"{ffile.name}: Empty file")
                        all_valid = False
                        continue
                    
                    # Check numeric columns for NaN/Inf
                    for col in df.select_dtypes(include=[np.number]).columns:
                        vals = df[col].values
                        if np.any(np.isnan(vals)):
                            all_valid = False
                            issues.append(f"{ffile.name}: {col} contains NaN")
                        if np.any(np.isinf(vals)):
                            all_valid = False
                            issues.append(f"{ffile.name}: {col} contains Inf")
                except Exception as e:
                    issues.append(f"{ffile.name}: Read error - {e}")
                    all_valid = False
            
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
            impl_base_dir = self.output_dir / impl
            impl_dir = self.get_first_param_dir(impl_base_dir)
            
            is_r = impl.startswith('R_')
            
            if is_r:
                # R uses feature_list_Z*.txt files for selections
                feature_files = list(impl_dir.glob('feature_list_Z*.txt'))
                n_z = len(feature_files)
                n_int = 0  # R doesn't output separate interaction file
            else:
                # Python uses sig_LFs.txt and sig_interacts.txt
                sig_lfs_path = impl_dir / 'sig_LFs.txt'
                sig_int_path = impl_dir / 'sig_interacts.txt'
                
                n_z = 0
                n_int = 0
                
                if sig_lfs_path.exists():
                    with open(sig_lfs_path) as f:
                        lines = [l.strip() for l in f if l.strip()]
                        n_z = len(lines)
                
                if sig_int_path.exists():
                    with open(sig_int_path) as f:
                        lines = [l.strip() for l in f if l.strip()]
                        n_int = len(lines)
            
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
