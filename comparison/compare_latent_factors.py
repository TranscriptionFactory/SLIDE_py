#!/usr/bin/env python3
"""
SLIDE Implementation Comparison Report Generator

This script generates a comprehensive comparison report across SLIDE implementations,
including completion status, output summaries, performance metrics, and latent factor
content comparisons.

Usage:
    python compare_latent_factors.py <output_path> [options]

Arguments:
    output_path     Path containing implementation subdirectories (R_native, Py_*, etc.)

Options:
    --detailed      Show detailed per-LF comparison
    --param COMBO   Only compare specific parameter combination (e.g., "0.05_0.1_out")
    --lf-only       Only run latent factor comparison (skip status/metrics)
"""

import argparse
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Task Configuration
# =============================================================================

@dataclass
class TaskConfig:
    """Configuration for a SLIDE implementation task."""
    name: str
    description: str
    is_r: bool = False


# Default task configuration (matches run_comparison.sh)
DEFAULT_TASKS = [
    TaskConfig("R_native", "R SLIDE (native R LOVE + R Knockoffs)", is_r=True),
    TaskConfig("Py_rLOVE_rKO", "Python (R LOVE + R Knockoffs)"),
    TaskConfig("Py_rLOVE_knockpy", "Python (R LOVE + knockpy Knockoffs)"),
    TaskConfig("Py_pyLOVE_rKO", "Python (Py LOVE + R Knockoffs)"),
    TaskConfig("Py_pyLOVE_knockpy", "Python (Py LOVE + knockpy Knockoffs)"),
]

# Additional known patterns for discovery
KNOWN_IMPL_PATTERNS = [
    "R_native", "R_outputs",
    "Py_rLOVE_rKO", "Py_rLOVE_knockpy", "Py_rLOVE_pyKO",
    "Py_pyLOVE_rKO", "Py_pyLOVE_knockpy", "Py_pyLOVE_pyKO",
    "Py_Py_LOVE", "Py_R_LOVE", "Py_outputs"
]


# =============================================================================
# Performance Metrics
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for a SLIDE run."""
    true_score: Optional[float] = None
    partial_random: Optional[float] = None
    full_random: Optional[float] = None
    num_marginals: Optional[int] = None
    num_interactions: Optional[int] = None
    num_lfs: Optional[int] = None

    def format_line(self, name: str) -> str:
        """Format metrics as a single line for reporting."""
        auc = f"{self.true_score:.3f}" if self.true_score is not None else "-"
        partial = f"{self.partial_random:.3f}" if self.partial_random is not None else "-"
        full = f"{self.full_random:.3f}" if self.full_random is not None else "-"
        marg = str(self.num_marginals) if self.num_marginals is not None else "-"
        inter = str(self.num_interactions) if self.num_interactions is not None else "-"
        return f"    {name:20s}: AUC={auc} (P={partial} F={full}) M={marg} I={inter}"


class MetricsExtractor:
    """Extract performance metrics from SLIDE outputs."""

    def __init__(self, script_dir: Optional[Path] = None, **kwargs):
        self.script_dir = script_dir or Path(__file__).parent

    def extract_python_metrics(self, path: Path) -> Optional[PerformanceMetrics]:
        """Extract metrics from Python SLIDE output (scores.txt, sig_LFs.txt)."""
        scores_file = path / 'scores.txt'
        lf_file = path / 'sig_LFs.txt'
        interact_file = path / 'sig_interacts.txt'

        metrics = PerformanceMetrics()

        if scores_file.exists():
            content = scores_file.read_text()

            # Parse scores
            for pattern, attr in [
                (r'True Scores?:\s*(-?[\d.]+)', 'true_score'),
                (r'Partial Random:\s*(-?[\d.]+)', 'partial_random'),
                (r'Full Random:\s*(-?[\d.]+)', 'full_random'),
                (r'Number of marginals:\s*(\d+)', 'num_marginals'),
                (r'Number of interactions:\s*(\d+)', 'num_interactions'),
            ]:
                match = re.search(pattern, content)
                if match:
                    val = match.group(1)
                    if attr in ('num_marginals', 'num_interactions'):
                        setattr(metrics, attr, int(val))
                    else:
                        setattr(metrics, attr, float(val))

        # Count LFs from sig_LFs.txt
        if lf_file.exists():
            lines = [l.strip() for l in lf_file.read_text().splitlines() if l.strip()]
            metrics.num_lfs = len(lines)

        # Count interactions from sig_interacts.txt if not in scores
        if metrics.num_interactions is None and interact_file.exists():
            lines = [l.strip() for l in interact_file.read_text().splitlines() if l.strip()]
            metrics.num_interactions = len(lines)

        return metrics if any([
            metrics.true_score, metrics.num_lfs, metrics.num_marginals
        ]) else None

    def extract_r_metrics(self, path: Path) -> Optional[PerformanceMetrics]:
        """Extract metrics from R SLIDE output.

        First tries to read performance_metrics.csv. If not present, attempts
        to generate it using extract_r_performance.R.
        """
        perf_csv = path / 'performance_metrics.csv'
        slide_lfs = path / 'SLIDE_LFs.rds'
        all_lfs = path / 'AllLatentFactors.rds'

        # Try to generate metrics CSV if needed
        if not perf_csv.exists() and slide_lfs.exists():
            r_script = self.script_dir / 'extract_r_performance.R'
            if r_script.exists():
                try:
                    subprocess.run(
                        ['Rscript', str(r_script), str(path)],
                        capture_output=True, timeout=30
                    )
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

        # Read metrics from CSV
        if perf_csv.exists():
            try:
                df = pd.read_csv(perf_csv)
                metrics = PerformanceMetrics()

                for col, attr in [
                    ('true_score', 'true_score'),
                    ('partial_random', 'partial_random'),
                    ('full_random', 'full_random'),
                    ('num_marginals', 'num_marginals'),
                    ('num_interactors', 'num_interactions'),
                    ('num_LFs', 'num_lfs'),
                ]:
                    if col in df.columns:
                        val = df[col].iloc[0]
                        if pd.notna(val):
                            if attr in ('num_marginals', 'num_interactions', 'num_lfs'):
                                setattr(metrics, attr, int(val))
                            else:
                                setattr(metrics, attr, float(val))

                return metrics
            except Exception:
                pass

        # Fallback: count feature_list files for LF count
        feature_files = list(path.glob('feature_list_Z*.txt'))
        if feature_files:
            return PerformanceMetrics(num_lfs=len(feature_files))

        return None


# =============================================================================
# Latent Factor Loading
# =============================================================================

class LatentFactorLoader:
    """Load and normalize latent factor outputs from R and Python SLIDE implementations."""

    def __init__(self, **kwargs):
        pass

    def load_python_outputs(self, path: Path) -> Optional[dict]:
        """Load Python SLIDE outputs.

        Expected files:
        - A.csv: features × LFs, index=feature names, columns=Z0,Z1,...
        - z_matrix.csv: samples × LFs, index=sample names, columns=Z0,Z1,...
        - sig_LFs.txt: one LF name per line (e.g., "Z2", "Z4")
        """
        path = Path(path)
        result = {'type': 'python', 'path': path}

        # Load A matrix
        a_path = path / 'A.csv'
        if a_path.exists():
            result['A'] = pd.read_csv(a_path, index_col=0)
        else:
            result['A'] = None

        # Load Z matrix
        z_path = path / 'z_matrix.csv'
        if z_path.exists():
            result['Z'] = pd.read_csv(z_path, index_col=0)
        else:
            result['Z'] = None

        # Load significant LFs
        sig_path = path / 'sig_LFs.txt'
        if sig_path.exists():
            with open(sig_path) as f:
                result['sig_LFs'] = [line.strip() for line in f if line.strip()]
        else:
            result['sig_LFs'] = []

        # Try to load Python feature_list_Z*.csv files first (more informative)
        py_feature_lists = self._load_python_feature_lists(path)
        if py_feature_lists:
            result['features_per_lf'] = py_feature_lists
            # Update sig_LFs to include all LFs with feature lists
            if py_feature_lists:
                result['sig_LFs'] = sorted(py_feature_lists.keys(), key=lambda x: int(x[1:]))
        else:
            # Fallback to extracting features from A matrix
            result['features_per_lf'] = self._extract_features_from_a(
                result.get('A'), result.get('sig_LFs', [])
            )

        return result if (result.get('A') is not None or result.get('Z') is not None) else None

    def _load_python_feature_lists(self, path: Path) -> dict:
        """Load Python feature list files (feature_list_Z*.csv)."""
        features_per_lf = {}

        for f in path.glob('feature_list_Z*.csv'):
            # Extract LF name from filename
            match = re.search(r'feature_list_(Z\d+)\.csv', f.name)
            if not match:
                continue
            lf_name = match.group(1)

            try:
                df = pd.read_csv(f, sep='\t', index_col=0)
                # Python feature files have: loading, AUC, corr, color columns
                features = df.index.tolist()
                loadings = dict(zip(features, df['loading'])) if 'loading' in df else {}
                features_per_lf[lf_name] = {
                    'features': features,
                    'loadings': loadings,
                }
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        return features_per_lf

    def load_r_outputs(self, path: Path) -> Optional[dict]:
        """Load R SLIDE outputs.

        Expected files:
        - z_matrix.csv: samples × LFs, quoted headers "Z1","Z2",...
        - feature_list_Z*.txt: tab-separated with columns: names, A_loading, corrs, color
        """
        path = Path(path)
        result = {'type': 'r', 'path': path}

        # Load Z matrix
        z_path = path / 'z_matrix.csv'
        if z_path.exists():
            result['Z'] = pd.read_csv(z_path, index_col=0)
            # Normalize column names (remove quotes if present)
            result['Z'].columns = [c.strip('"') for c in result['Z'].columns]
        else:
            result['Z'] = None

        # Load feature lists and build A matrix from them
        feature_lists = self._load_r_feature_lists(path)
        result['features_per_lf'] = feature_lists
        result['sig_LFs'] = list(feature_lists.keys())

        # Build A matrix from feature lists
        result['A'] = self._build_a_from_features(feature_lists)

        return result if (result.get('Z') is not None or result.get('sig_LFs')) else None

    def _load_r_feature_lists(self, path: Path) -> dict:
        """Load R feature list files (feature_list_Z*.txt)."""
        features_per_lf = {}

        for f in path.glob('feature_list_Z*.txt'):
            # Extract LF name from filename
            match = re.search(r'feature_list_(Z\d+)\.txt', f.name)
            if not match:
                continue
            lf_name = match.group(1)

            try:
                df = pd.read_csv(f, sep='\t')
                # Filter out rows with NaN feature names
                df = df[df['names'].notna()]
                features_per_lf[lf_name] = {
                    'features': df['names'].tolist(),
                    'loadings': dict(zip(df['names'], df['A_loading'])),
                    'corrs': dict(zip(df['names'], df['corrs'])) if 'corrs' in df else {},
                }
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        return features_per_lf

    def _extract_features_from_a(self, A: Optional[pd.DataFrame], sig_lfs: list) -> dict:
        """Extract non-zero features per LF from A matrix."""
        if A is None:
            return {}

        features_per_lf = {}
        for lf in sig_lfs:
            if lf not in A.columns:
                continue
            col = A[lf]
            nonzero_mask = col.abs() > 1e-10
            features = col[nonzero_mask].index.tolist()
            loadings = dict(zip(features, col[nonzero_mask].values))
            features_per_lf[lf] = {
                'features': features,
                'loadings': loadings,
            }
        return features_per_lf

    def _build_a_from_features(self, features_per_lf: dict) -> Optional[pd.DataFrame]:
        """Build A matrix from feature lists (for R outputs)."""
        if not features_per_lf:
            return None

        # Collect all features
        all_features = set()
        for lf_data in features_per_lf.values():
            all_features.update(lf_data['features'])

        if not all_features:
            return None

        # Build matrix
        all_features = sorted(all_features)
        lf_names = sorted(features_per_lf.keys(), key=lambda x: int(x[1:]))

        A = pd.DataFrame(0.0, index=all_features, columns=lf_names)
        for lf_name, lf_data in features_per_lf.items():
            for feat, loading in lf_data['loadings'].items():
                A.loc[feat, lf_name] = loading

        return A

    def normalize_lf_name(self, name: str, to_base: str = 'Z0') -> int:
        """Extract numeric index from LF name (handles Z0 vs Z1 indexing)."""
        match = re.match(r'Z(\d+)', name)
        if not match:
            raise ValueError(f"Invalid LF name: {name}")
        return int(match.group(1))


# =============================================================================
# Latent Factor Matching
# =============================================================================

class LatentFactorMatcher:
    """Match corresponding LFs across implementations based on feature overlap or correlation."""

    def __init__(self, **kwargs):
        pass

    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Compute Jaccard index between two feature sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def feature_overlap_matrix(self, features1: dict, features2: dict) -> tuple:
        """Compute pairwise Jaccard similarity matrix between all LFs."""
        lf_names1 = sorted(features1.keys(), key=lambda x: int(x[1:]))
        lf_names2 = sorted(features2.keys(), key=lambda x: int(x[1:]))

        if not lf_names1 or not lf_names2:
            return np.array([[]]), lf_names1, lf_names2

        sim_matrix = np.zeros((len(lf_names1), len(lf_names2)))
        for i, lf1 in enumerate(lf_names1):
            feat1 = set(features1[lf1]['features'])
            for j, lf2 in enumerate(lf_names2):
                feat2 = set(features2[lf2]['features'])
                sim_matrix[i, j] = self.jaccard_similarity(feat1, feat2)

        return sim_matrix, lf_names1, lf_names2

    def a_correlation_matrix(self, A1: pd.DataFrame, A2: pd.DataFrame) -> tuple:
        """Compute pairwise correlation between A matrix columns."""
        if A1 is None or A2 is None:
            return np.array([[]]), [], []

        # Find common features
        common_features = sorted(set(A1.index) & set(A2.index))
        if not common_features:
            return np.array([[]]), [], []

        A1_common = A1.loc[common_features]
        A2_common = A2.loc[common_features]

        lf_names1 = list(A1_common.columns)
        lf_names2 = list(A2_common.columns)

        corr_matrix = np.zeros((len(lf_names1), len(lf_names2)))
        for i, lf1 in enumerate(lf_names1):
            for j, lf2 in enumerate(lf_names2):
                v1 = A1_common[lf1].values
                v2 = A2_common[lf2].values
                # Handle constant vectors
                if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
                    corr_matrix[i, j] = 0.0
                else:
                    corr_matrix[i, j] = np.abs(np.corrcoef(v1, v2)[0, 1])

        return corr_matrix, lf_names1, lf_names2

    def compute_optimal_matching(self, similarity_matrix: np.ndarray,
                                  lf_names1: list, lf_names2: list,
                                  threshold: float = 0.1) -> dict:
        """Find optimal bipartite matching using Hungarian algorithm."""
        if similarity_matrix.size == 0:
            return {
                'matches': [],
                'unmatched1': list(lf_names1),
                'unmatched2': list(lf_names2),
            }

        # Hungarian algorithm maximizes, so negate for minimization
        cost_matrix = -similarity_matrix

        # Pad to square if needed
        n1, n2 = cost_matrix.shape
        max_dim = max(n1, n2)
        if n1 != n2:
            padded = np.zeros((max_dim, max_dim))
            padded[:n1, :n2] = cost_matrix
            cost_matrix = padded

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        matched1 = set()
        matched2 = set()

        for i, j in zip(row_ind, col_ind):
            if i < n1 and j < n2:
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    matches.append((lf_names1[i], lf_names2[j], sim))
                    matched1.add(i)
                    matched2.add(j)

        unmatched1 = [lf_names1[i] for i in range(n1) if i not in matched1]
        unmatched2 = [lf_names2[j] for j in range(n2) if j not in matched2]

        return {
            'matches': sorted(matches, key=lambda x: -x[2]),
            'unmatched1': unmatched1,
            'unmatched2': unmatched2,
        }

    def match_by_feature_overlap(self, impl1: dict, impl2: dict,
                                  threshold: float = 0.1) -> dict:
        """Match LFs by feature overlap (Jaccard similarity)."""
        feat1 = impl1.get('features_per_lf', {})
        feat2 = impl2.get('features_per_lf', {})

        sim_matrix, names1, names2 = self.feature_overlap_matrix(feat1, feat2)
        return self.compute_optimal_matching(sim_matrix, names1, names2, threshold)

    def match_by_a_correlation(self, impl1: dict, impl2: dict,
                                threshold: float = 0.3) -> dict:
        """Match LFs by A matrix column correlation."""
        A1 = impl1.get('A')
        A2 = impl2.get('A')

        # Filter to significant LFs only
        if A1 is not None and impl1.get('sig_LFs'):
            sig_cols = [c for c in impl1['sig_LFs'] if c in A1.columns]
            A1 = A1[sig_cols] if sig_cols else A1
        if A2 is not None and impl2.get('sig_LFs'):
            sig_cols = [c for c in impl2['sig_LFs'] if c in A2.columns]
            A2 = A2[sig_cols] if sig_cols else A2

        corr_matrix, names1, names2 = self.a_correlation_matrix(A1, A2)
        return self.compute_optimal_matching(corr_matrix, names1, names2, threshold)


# =============================================================================
# Latent Factor Comparator
# =============================================================================

class LatentFactorComparator:
    """Compare latent factors across implementations."""

    def __init__(self, detailed: bool = False, **kwargs):
        self.detailed = detailed
        self.loader = LatentFactorLoader()
        self.matcher = LatentFactorMatcher()

    def compare_implementations(self, impl1: dict, impl2: dict,
                                 name1: str, name2: str) -> dict:
        """Compare two implementations and return metrics."""
        results = {
            'impl1': name1,
            'impl2': name2,
            'lf_count1': len(impl1.get('sig_LFs', [])),
            'lf_count2': len(impl2.get('sig_LFs', [])),
        }

        # Feature-based matching
        feat_match = self.matcher.match_by_feature_overlap(impl1, impl2)
        results['feature_matches'] = feat_match

        # Compute mean Jaccard for matched LFs
        if feat_match['matches']:
            results['mean_jaccard'] = np.mean([m[2] for m in feat_match['matches']])
        else:
            results['mean_jaccard'] = 0.0

        # A matrix correlation matching (if available)
        a_match = self.matcher.match_by_a_correlation(impl1, impl2)
        results['a_correlation_matches'] = a_match

        if a_match['matches']:
            results['mean_a_corr'] = np.mean([m[2] for m in a_match['matches']])
        else:
            results['mean_a_corr'] = 0.0

        # Z matrix correlation for matched LFs
        z_corrs = self._compare_z_matrices(impl1, impl2, feat_match['matches'])
        results['z_correlations'] = z_corrs
        if z_corrs:
            results['mean_z_corr'] = np.mean(list(z_corrs.values()))
        else:
            results['mean_z_corr'] = 0.0

        return results

    def _compare_z_matrices(self, impl1: dict, impl2: dict,
                             matches: list) -> dict:
        """Compare Z matrix scores for matched LFs."""
        Z1 = impl1.get('Z')
        Z2 = impl2.get('Z')

        if Z1 is None or Z2 is None:
            return {}

        # Find common samples
        common_samples = sorted(set(Z1.index) & set(Z2.index))
        if not common_samples:
            return {}

        Z1_common = Z1.loc[common_samples]
        Z2_common = Z2.loc[common_samples]

        z_corrs = {}
        for lf1, lf2, _ in matches:
            if lf1 in Z1_common.columns and lf2 in Z2_common.columns:
                v1 = Z1_common[lf1].values
                v2 = Z2_common[lf2].values
                if np.std(v1) > 1e-10 and np.std(v2) > 1e-10:
                    corr = np.abs(np.corrcoef(v1, v2)[0, 1])
                    z_corrs[f"{lf1}<->{lf2}"] = corr

        return z_corrs

    def format_report(self, results: dict) -> str:
        """Format comparison results as a report string."""
        lines = []
        lines.append(f"\n{results['impl1']} <-> {results['impl2']}")
        lines.append("-" * 50)
        lines.append(f"  LF counts: {results['lf_count1']} vs {results['lf_count2']}")

        feat_match = results['feature_matches']
        n_matched = len(feat_match['matches'])
        lines.append(f"  Matched LFs: {n_matched}")

        if results['mean_jaccard'] > 0:
            lines.append(f"  Mean Jaccard (feature overlap): {results['mean_jaccard']:.3f}")
        if results['mean_a_corr'] > 0:
            lines.append(f"  Mean A correlation: {results['mean_a_corr']:.3f}")
        if results['mean_z_corr'] > 0:
            lines.append(f"  Mean Z correlation: {results['mean_z_corr']:.3f}")

        if self.detailed and feat_match['matches']:
            lines.append("\n  Matched pairs:")
            for lf1, lf2, sim in feat_match['matches'][:10]:  # Top 10
                z_key = f"{lf1}<->{lf2}"
                z_corr = results['z_correlations'].get(z_key, float('nan'))
                lines.append(f"    {lf1:>6} <-> {lf2:<6}  Jaccard={sim:.3f}  Z_corr={z_corr:.3f}")

        if feat_match['unmatched1']:
            lines.append(f"\n  Unique to {results['impl1']}: {', '.join(feat_match['unmatched1'][:5])}")
            if len(feat_match['unmatched1']) > 5:
                lines.append(f"    ... and {len(feat_match['unmatched1']) - 5} more")
        if feat_match['unmatched2']:
            lines.append(f"  Unique to {results['impl2']}: {', '.join(feat_match['unmatched2'][:5])}")
            if len(feat_match['unmatched2']) > 5:
                lines.append(f"    ... and {len(feat_match['unmatched2']) - 5} more")

        return '\n'.join(lines)


# =============================================================================
# Report Generator
# =============================================================================

@dataclass
class OutputFiles:
    """Check for presence of key output files."""
    has_a: bool = False  # A.csv or AllLatentFactors.rds
    has_z: bool = False  # z_matrix.csv or feature_list_Z*.txt
    has_lf: bool = False  # sig_LFs.txt or SLIDE_LFs.rds

    def status_str(self) -> str:
        a = "A" if self.has_a else "-"
        z = "Z" if self.has_z else "-"
        lf = "LF" if self.has_lf else "-"
        return f"[{a} {z} {lf}]"


@dataclass
class ReportSummary:
    """Collect summary statistics during report generation."""
    n_implementations: int = 0
    n_completed: int = 0
    n_param_combos: int = 0
    best_lf_match: Optional[tuple] = None  # (impl1, impl2, jaccard, z_corr)
    best_performance: Optional[tuple] = None  # (impl, param, auc)
    total_comparisons: int = 0
    avg_jaccard: float = 0.0
    avg_z_corr: float = 0.0
    _jaccard_values: list = field(default_factory=list)
    _z_corr_values: list = field(default_factory=list)
    # Detailed comparison tracking
    pairwise_results: dict = field(default_factory=dict)  # (impl1, impl2) -> {jaccard, z_corr, matched, ...}
    lf_counts: dict = field(default_factory=dict)  # impl -> [lf_counts per param]
    performance_scores: dict = field(default_factory=dict)  # impl -> [(param, score), ...]

    def update_lf_comparison(self, impl1: str, impl2: str,
                              mean_jaccard: float, mean_z_corr: float,
                              n_matched: int = 0, lf_count1: int = 0, lf_count2: int = 0):
        """Track LF comparison results."""
        self.total_comparisons += 1
        if mean_jaccard > 0:
            self._jaccard_values.append(mean_jaccard)
        if mean_z_corr > 0:
            self._z_corr_values.append(mean_z_corr)

        # Track best match
        if mean_jaccard > 0 or mean_z_corr > 0:
            score = mean_jaccard + mean_z_corr
            if self.best_lf_match is None or score > (self.best_lf_match[2] + self.best_lf_match[3]):
                self.best_lf_match = (impl1, impl2, mean_jaccard, mean_z_corr)

        # Store detailed pairwise results
        key = (impl1, impl2)
        if key not in self.pairwise_results:
            self.pairwise_results[key] = []
        self.pairwise_results[key].append({
            'jaccard': mean_jaccard,
            'z_corr': mean_z_corr,
            'n_matched': n_matched,
            'lf_count1': lf_count1,
            'lf_count2': lf_count2,
        })

    def update_lf_count(self, impl: str, param: str, count: int):
        """Track LF counts per implementation."""
        if impl not in self.lf_counts:
            self.lf_counts[impl] = []
        self.lf_counts[impl].append((param, count))

    def update_performance(self, impl: str, param: str, auc: Optional[float]):
        """Track performance results."""
        if auc is not None:
            if impl not in self.performance_scores:
                self.performance_scores[impl] = []
            self.performance_scores[impl].append((param, auc))

            if auc > 0:
                if self.best_performance is None or auc > self.best_performance[2]:
                    self.best_performance = (impl, param, auc)

    def finalize(self):
        """Calculate final averages."""
        if self._jaccard_values:
            self.avg_jaccard = np.mean(self._jaccard_values)
        if self._z_corr_values:
            self.avg_z_corr = np.mean(self._z_corr_values)

    def get_similarity_ranking(self) -> list:
        """Return implementation pairs ranked by similarity (Jaccard + Z_corr)."""
        rankings = []
        for (impl1, impl2), results in self.pairwise_results.items():
            avg_jaccard = np.mean([r['jaccard'] for r in results if r['jaccard'] > 0]) if any(r['jaccard'] > 0 for r in results) else 0
            avg_z_corr = np.mean([r['z_corr'] for r in results if r['z_corr'] > 0]) if any(r['z_corr'] > 0 for r in results) else 0
            combined_score = avg_jaccard + avg_z_corr
            rankings.append((impl1, impl2, avg_jaccard, avg_z_corr, combined_score))
        return sorted(rankings, key=lambda x: -x[4])

    def get_avg_lf_counts(self) -> dict:
        """Return average LF count per implementation."""
        return {impl: np.mean([c for _, c in counts]) for impl, counts in self.lf_counts.items()}

    def get_avg_performance(self) -> dict:
        """Return average performance per implementation."""
        result = {}
        for impl, scores in self.performance_scores.items():
            valid_scores = [s for _, s in scores if s is not None and s > 0]
            result[impl] = np.mean(valid_scores) if valid_scores else None
        return result


class ReportGenerator:
    """Generate comprehensive SLIDE comparison report."""

    def __init__(self, output_path: Path, tasks: Optional[list] = None,
                 detailed: bool = False, **kwargs):
        self.output_path = Path(output_path)
        self.tasks = tasks or DEFAULT_TASKS
        self.detailed = detailed
        self.metrics_extractor = MetricsExtractor(script_dir=Path(__file__).parent)
        self.lf_loader = LatentFactorLoader()
        self.lf_comparator = LatentFactorComparator(detailed=detailed)
        self.summary = ReportSummary()

    def find_param_dir(self, base_path: Path, combo: str) -> Optional[Path]:
        """Find parameter directory handling R vs Python naming differences."""
        direct = base_path / combo
        if direct.is_dir():
            return direct

        # Try converting integer to decimal: 0.05_1_out -> 0.05_1.0_out
        alt_combo = re.sub(r'_(\d+)_out$', r'_\1.0_out', combo)
        alt_path = base_path / alt_combo
        if alt_path.is_dir():
            return alt_path

        # Try converting decimal to integer: 0.05_1.0_out -> 0.05_1_out
        alt_combo = re.sub(r'_(\d+)\.0_out$', r'_\1_out', combo)
        alt_path = base_path / alt_combo
        if alt_path.is_dir():
            return alt_path

        return None

    def check_output_files(self, path: Path) -> OutputFiles:
        """Check for presence of key output files in a directory."""
        files = OutputFiles()

        # A matrix
        files.has_a = (path / 'A.csv').exists() or (path / 'AllLatentFactors.rds').exists()

        # Z matrix
        files.has_z = (path / 'z_matrix.csv').exists() or bool(list(path.glob('feature_list_Z*.txt')))

        # Significant LFs
        files.has_lf = (path / 'sig_LFs.txt').exists() or (path / 'SLIDE_LFs.rds').exists()

        return files

    def discover_implementations(self) -> dict:
        """Discover available implementations in output path."""
        available = {}

        # Try known patterns first
        for pattern in KNOWN_IMPL_PATTERNS:
            impl_path = self.output_path / pattern
            if impl_path.is_dir():
                available[pattern] = impl_path

        # Discover other directories with *_out subdirectories
        for child in self.output_path.iterdir():
            if child.is_dir() and child.name not in available:
                try:
                    has_outputs = any(d.name.endswith('_out') for d in child.iterdir() if d.is_dir())
                    if has_outputs:
                        available[child.name] = child
                except PermissionError:
                    pass

        return available

    def get_completion_status(self) -> tuple:
        """Check completion status for each task.

        Returns:
            (completed_count, status_dict)
        """
        completed = 0
        status = {}

        for i, task in enumerate(self.tasks):
            task_out = self.output_path / task.name
            marker = self.output_path / f".task{i}_complete"

            if marker.exists():
                status[task.name] = ("COMPLETED", task.description)
                completed += 1
            elif task_out.is_dir():
                status[task.name] = ("PARTIAL", task.description)
            else:
                status[task.name] = ("NOT_STARTED", task.description)

        return completed, status

    def print_header(self):
        """Print report header."""
        logger.info("=" * 62)
        logger.info("SLIDE Implementation Comparison Report")
        logger.info("=" * 62)
        logger.info(f"Output path: {self.output_path}")
        logger.info("=" * 62)
        logger.info("")

    def print_completion_status(self):
        """Print completion status section."""
        logger.info("Completion Status:")
        logger.info("------------------")

        completed, status = self.get_completion_status()

        for i, task in enumerate(self.tasks):
            name = task.name
            if name in status:
                state, desc = status[name]
                if state == "COMPLETED":
                    symbol = "+"
                elif state == "PARTIAL":
                    symbol = "?"
                else:
                    symbol = "x"
                logger.info(f"  [{i}] {symbol} {name:20s}")
                logger.info(f"      {desc}")

        logger.info("")
        logger.info(f"Completed: {completed}/{len(self.tasks)}")
        logger.info("")

        # Track in summary
        self.summary.n_completed = completed
        self.summary.n_implementations = len([s for s in status.values() if s[0] != "NOT_STARTED"])

        return completed

    def print_output_summary(self):
        """Print output summary section."""
        logger.info("=" * 62)
        logger.info("Output Summary")
        logger.info("=" * 62)

        available = self.discover_implementations()

        for i, task in enumerate(self.tasks):
            task_out = self.output_path / task.name

            if not task_out.is_dir():
                # Try to find under different name
                if task.name not in available:
                    continue
                task_out = available[task.name]

            logger.info("")
            logger.info(f"[{i}] {task.name}:")
            logger.info(f"    Directory: {task_out}")

            # List parameter combinations
            combos = sorted([d for d in task_out.iterdir()
                           if d.is_dir() and d.name.endswith('_out')])
            logger.info(f"    Parameter combinations: {len(combos)}")

            if combos:
                logger.info("    Subdirectories:")
                for combo_dir in combos:
                    files = self.check_output_files(combo_dir)
                    logger.info(f"      {combo_dir.name:20s} {files.status_str()}")

    def print_performance_comparison(self, param_filter: Optional[str] = None):
        """Print cross-implementation performance comparison."""
        available = self.discover_implementations()

        if len(available) < 2:
            return

        logger.info("")
        logger.info("=" * 62)
        logger.info("Cross-Implementation Performance Comparison")
        logger.info("=" * 62)

        # Get reference implementation
        ref_name = list(available.keys())[0]
        ref_path = available[ref_name]

        # Get parameter combinations
        combos = sorted([d.name for d in ref_path.iterdir()
                        if d.is_dir() and d.name.endswith('_out')])

        if param_filter:
            combos = [c for c in combos if c == param_filter or
                     self.find_param_dir(ref_path, param_filter) and
                     self.find_param_dir(ref_path, param_filter).name == c]

        for combo in combos:
            logger.info("")
            logger.info(f"Parameter: {combo}")
            logger.info("  Performance metrics:")

            for task in self.tasks:
                name = task.name
                if name not in available:
                    logger.info(f"    {name:20s}: (no results)")
                    continue

                param_dir = self.find_param_dir(available[name], combo)
                if param_dir is None:
                    logger.info(f"    {name:20s}: (no results)")
                    continue

                # Extract metrics
                if task.is_r:
                    metrics = self.metrics_extractor.extract_r_metrics(param_dir)
                else:
                    metrics = self.metrics_extractor.extract_python_metrics(param_dir)

                if metrics:
                    logger.info(metrics.format_line(name))
                    # Track best performance
                    self.summary.update_performance(name, combo, metrics.true_score)
                else:
                    logger.info(f"    {name:20s}: (no metrics)")

    def print_lf_comparison(self, param_filter: Optional[str] = None):
        """Print latent factor content comparison."""
        available = self.discover_implementations()

        if len(available) < 2:
            logger.info("")
            logger.info("Insufficient implementations for LF comparison")
            return

        logger.info("")
        logger.info("=" * 62)
        logger.info("Latent Factor Content Comparison")
        logger.info("=" * 62)

        # Get reference implementation
        ref_name = list(available.keys())[0]
        ref_path = available[ref_name]

        # Get parameter combinations
        combos = sorted([d.name for d in ref_path.iterdir()
                        if d.is_dir() and d.name.endswith('_out')])

        if param_filter:
            found = self.find_param_dir(ref_path, param_filter)
            combos = [found.name] if found else []

        for combo in combos:
            logger.info("")
            logger.info("=" * 62)
            logger.info(f"Parameter: {combo}")
            logger.info("=" * 62)

            # Load all implementations for this combo
            impl_data = {}
            for name, base_path in available.items():
                param_path = self.find_param_dir(base_path, combo)
                if param_path is None:
                    continue

                # Determine if R or Python
                is_r = name.startswith('R_') or name == 'R_native' or name == 'R_outputs'
                if not is_r and list(param_path.glob('feature_list_Z*.txt')):
                    is_r = True
                elif is_r and (param_path / 'A.csv').exists() and (param_path / 'sig_LFs.txt').exists():
                    is_r = False

                if is_r:
                    data = self.lf_loader.load_r_outputs(param_path)
                else:
                    data = self.lf_loader.load_python_outputs(param_path)

                if data:
                    impl_data[name] = data

            if len(impl_data) < 2:
                logger.info(f"  Insufficient data for comparison (found {len(impl_data)} impls)")
                continue

            # Print LF count summary and track counts
            logger.info("\nLF Count Summary:")
            for name, data in impl_data.items():
                n_lfs = len(data.get('sig_LFs', []))
                logger.info(f"  {name:20s}: {n_lfs} significant LFs")
                self.summary.update_lf_count(name, combo, n_lfs)

            # Compare R_native to each Python implementation
            if 'R_native' in impl_data:
                for py_name in impl_data:
                    if py_name == 'R_native':
                        continue
                    results = self.lf_comparator.compare_implementations(
                        impl_data['R_native'], impl_data[py_name],
                        'R_native', py_name
                    )
                    logger.info(self.lf_comparator.format_report(results))
                    # Track summary stats
                    self.summary.update_lf_comparison(
                        'R_native', py_name,
                        results['mean_jaccard'], results['mean_z_corr'],
                        n_matched=len(results['feature_matches']['matches']),
                        lf_count1=results['lf_count1'],
                        lf_count2=results['lf_count2']
                    )

            # Compare Python implementations to each other
            py_impls = [n for n in impl_data if n != 'R_native']
            if len(py_impls) >= 2:
                logger.info("\nPython Implementation Cross-Comparison:")
                for i, name1 in enumerate(py_impls):
                    for name2 in py_impls[i+1:]:
                        results = self.lf_comparator.compare_implementations(
                            impl_data[name1], impl_data[name2],
                            name1, name2
                        )
                        logger.info(self.lf_comparator.format_report(results))
                        # Track summary stats
                        self.summary.update_lf_comparison(
                            name1, name2,
                            results['mean_jaccard'], results['mean_z_corr'],
                            n_matched=len(results['feature_matches']['matches']),
                            lf_count1=results['lf_count1'],
                            lf_count2=results['lf_count2']
                        )

    def print_summary(self):
        """Print summary of key findings."""
        self.summary.finalize()

        logger.info("")
        logger.info("=" * 62)
        logger.info("SUMMARY")
        logger.info("=" * 62)

        logger.info(f"\nImplementations analyzed: {self.summary.n_implementations}")
        logger.info(f"Completed tasks: {self.summary.n_completed}/{len(self.tasks)}")
        logger.info(f"Total pairwise comparisons: {self.summary.total_comparisons}")

        # Average LF counts per implementation
        avg_lf_counts = self.summary.get_avg_lf_counts()
        if avg_lf_counts:
            logger.info("\n" + "-" * 62)
            logger.info("Average LF Counts per Implementation:")
            logger.info("-" * 62)
            for impl, count in sorted(avg_lf_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  {impl:25s}: {count:.1f} LFs")

        # Average performance per implementation
        avg_perf = self.summary.get_avg_performance()
        if avg_perf and any(v is not None for v in avg_perf.values()):
            logger.info("\n" + "-" * 62)
            logger.info("Average Performance per Implementation:")
            logger.info("-" * 62)
            for impl, score in sorted(avg_perf.items(), key=lambda x: -(x[1] or 0)):
                if score is not None:
                    logger.info(f"  {impl:25s}: {score:.3f}")
                else:
                    logger.info(f"  {impl:25s}: -")

        # Similarity ranking (most similar pairs)
        rankings = self.summary.get_similarity_ranking()
        if rankings:
            logger.info("\n" + "-" * 62)
            logger.info("Implementation Similarity Ranking:")
            logger.info("-" * 62)
            logger.info("  (Ranked by combined Jaccard + Z-correlation score)")
            logger.info("")
            logger.info(f"  {'Pair':<45} {'Jaccard':>8} {'Z-corr':>8} {'Score':>8}")
            logger.info(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")
            for impl1, impl2, jaccard, z_corr, score in rankings[:10]:  # Top 10
                pair_name = f"{impl1} <-> {impl2}"
                j_str = f"{jaccard:.3f}" if jaccard > 0 else "-"
                z_str = f"{z_corr:.3f}" if z_corr > 0 else "-"
                s_str = f"{score:.3f}" if score > 0 else "-"
                logger.info(f"  {pair_name:<45} {j_str:>8} {z_str:>8} {s_str:>8}")

        # Pairwise comparison matrix
        if self.summary.pairwise_results:
            self._print_comparison_matrix()

        if self.summary.avg_jaccard > 0:
            logger.info(f"\nOverall average Jaccard similarity: {self.summary.avg_jaccard:.3f}")
        if self.summary.avg_z_corr > 0:
            logger.info(f"Overall average Z correlation: {self.summary.avg_z_corr:.3f}")

        if self.summary.best_lf_match:
            impl1, impl2, jaccard, z_corr = self.summary.best_lf_match
            logger.info(f"\nBest LF match: {impl1} <-> {impl2}")
            logger.info(f"  Jaccard: {jaccard:.3f}, Z correlation: {z_corr:.3f}")

        if self.summary.best_performance:
            impl, param, auc = self.summary.best_performance
            logger.info(f"\nBest performance: {impl} @ {param}")
            logger.info(f"  AUC/Score: {auc:.3f}")

        # Key insights
        logger.info("\n" + "-" * 62)
        logger.info("Key Insights:")
        logger.info("-" * 62)

        if self.summary.avg_jaccard > 0.5:
            logger.info("  + High feature overlap across implementations")
        elif self.summary.avg_jaccard > 0.2:
            logger.info("  ~ Moderate feature overlap across implementations")
        elif self.summary.avg_jaccard > 0:
            logger.info("  - Low feature overlap - implementations finding different features")

        if self.summary.avg_z_corr > 0.8:
            logger.info("  + High Z matrix correlation - similar sample scores")
        elif self.summary.avg_z_corr > 0.5:
            logger.info("  ~ Moderate Z matrix correlation")
        elif self.summary.avg_z_corr > 0:
            logger.info("  - Low Z correlation - sample scores differ between implementations")

        if self.summary.best_lf_match and self.summary.best_lf_match[3] > 0.9:
            impl1, impl2 = self.summary.best_lf_match[:2]
            logger.info(f"  * {impl1} and {impl2} produce nearly identical LFs")

        # Identify clusters of similar implementations
        if rankings:
            high_sim_pairs = [(i1, i2) for i1, i2, j, z, s in rankings if s > 0.5]
            if high_sim_pairs:
                logger.info("\n  Similar implementation groups:")
                for i1, i2 in high_sim_pairs[:3]:
                    logger.info(f"    - {i1} ~ {i2}")

    def _print_comparison_matrix(self):
        """Print a pairwise comparison matrix."""
        # Get unique implementations
        impls = set()
        for (i1, i2) in self.summary.pairwise_results.keys():
            impls.add(i1)
            impls.add(i2)
        impls = sorted(impls)

        if len(impls) < 2:
            return

        # Create short names for display
        short_names = {}
        for impl in impls:
            if impl == 'R_native':
                short_names[impl] = 'R_nat'
            elif impl.startswith('Py_'):
                # Py_rLOVE_rKO -> r_r, Py_pyLOVE_knockpy -> py_kpy
                parts = impl[3:].split('_')
                if len(parts) >= 2:
                    love = 'r' if parts[0].startswith('r') else 'py'
                    ko = 'r' if 'rKO' in impl else 'kpy' if 'knockpy' in impl else 'py'
                    short_names[impl] = f"{love}_{ko}"
                else:
                    short_names[impl] = impl[:8]
            else:
                short_names[impl] = impl[:8]

        logger.info("\n" + "-" * 62)
        logger.info("Pairwise Similarity Matrix (Jaccard | Z-corr):")
        logger.info("-" * 62)

        # Header
        header = "              "
        for impl in impls:
            header += f"{short_names[impl]:>12}"
        logger.info(header)

        # Rows
        for i, impl1 in enumerate(impls):
            row = f"{short_names[impl1]:<12}  "
            for j, impl2 in enumerate(impls):
                if i == j:
                    row += f"{'---':>12}"
                elif i < j:
                    key = (impl1, impl2)
                    if key not in self.summary.pairwise_results:
                        key = (impl2, impl1)
                    if key in self.summary.pairwise_results:
                        results = self.summary.pairwise_results[key]
                        avg_j = np.mean([r['jaccard'] for r in results])
                        avg_z = np.mean([r['z_corr'] for r in results])
                        if avg_j > 0 or avg_z > 0:
                            row += f"{avg_j:.2f}|{avg_z:.2f}".rjust(12)
                        else:
                            row += f"{'0|0':>12}"
                    else:
                        row += f"{'n/a':>12}"
                else:
                    row += f"{'':>12}"  # Lower triangle empty
            logger.info(row)

        logger.info("\n  Legend: Jaccard similarity | Z-matrix correlation")
        logger.info("  Higher values = more similar outputs")

    def generate_full_report(self, param_filter: Optional[str] = None,
                              lf_only: bool = False):
        """Generate the full comparison report."""
        self.print_header()

        if not lf_only:
            completed = self.print_completion_status()
            self.print_output_summary()

            if completed >= 2:
                self.print_performance_comparison(param_filter)

        self.print_lf_comparison(param_filter)
        self.print_summary()

        logger.info("")
        logger.info("=" * 62)
        logger.info("Report complete")
        logger.info("=" * 62)


# =============================================================================
# Main
# =============================================================================

def setup_file_logging(output_file: Path):
    """Add file handler to logger for writing report to file."""
    file_handler = logging.FileHandler(output_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    return file_handler


def main():
    parser = argparse.ArgumentParser(
        description='SLIDE Implementation Comparison Report Generator'
    )
    parser.add_argument('output_path', type=Path,
                       help='Path containing implementation subdirectories')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed per-LF comparison')
    parser.add_argument('--param', type=str, default=None,
                       help='Only compare specific parameter combination (e.g., "0.05_0.1_out")')
    parser.add_argument('--lf-only', action='store_true',
                       help='Only run latent factor comparison (skip status/metrics)')
    parser.add_argument('--output', '-o', type=Path, default=None,
                       help='Output file path (default: <output_path>/comparison_report.txt)')
    parser.add_argument('--no-file', action='store_true',
                       help='Do not write to file, only print to stdout')

    args = parser.parse_args()

    output_path = args.output_path
    if not output_path.is_dir():
        logger.error(f"Output path not found: {output_path}")
        return 1

    # Set up file output
    file_handler = None
    output_file = None
    if not args.no_file:
        output_file = args.output or (output_path / 'comparison_report.txt')
        file_handler = setup_file_logging(output_file)

    report = ReportGenerator(
        output_path=output_path,
        detailed=args.detailed
    )

    report.generate_full_report(
        param_filter=args.param,
        lf_only=args.lf_only
    )

    # Clean up and show output file location
    if file_handler:
        file_handler.close()
        logger.removeHandler(file_handler)
        print(f"\nReport saved to: {output_file}")

    return 0


if __name__ == '__main__':
    exit(main())
