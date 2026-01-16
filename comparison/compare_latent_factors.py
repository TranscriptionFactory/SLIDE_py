#!/usr/bin/env python3
"""
Compare latent factor content across SLIDE implementations.

This script compares the actual latent factor content (features, loadings, sample scores)
across different SLIDE implementations, not just performance metrics.

Key comparisons:
1. Feature overlap (Jaccard similarity) - What features does each LF contain?
2. A matrix correlation - How similar are the loading values?
3. Z matrix agreement - Do sample scores correlate across implementations?
4. LF matching - Identify corresponding LFs across implementations (may have different indices)

Usage:
    python compare_latent_factors.py <output_path> [options]

Arguments:
    output_path     Path containing implementation subdirectories (R_native, Py_*, etc.)

Options:
    --detailed      Show detailed per-LF comparison
    --param COMBO   Only compare specific parameter combination (e.g., "0.05_0.1_out")
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


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
        """Compute pairwise Jaccard similarity matrix between all LFs.

        Returns:
            similarity_matrix: np.ndarray of shape (n_lf1, n_lf2)
            lf_names1: list of LF names from impl1
            lf_names2: list of LF names from impl2
        """
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
        """Compute pairwise correlation between A matrix columns.

        Returns:
            correlation_matrix: np.ndarray of shape (n_lf1, n_lf2)
            lf_names1: list of LF names from impl1
            lf_names2: list of LF names from impl2
        """
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
        """Find optimal bipartite matching using Hungarian algorithm.

        Args:
            similarity_matrix: (n1, n2) similarity scores
            lf_names1: names for rows
            lf_names2: names for columns
            threshold: minimum similarity to consider a valid match

        Returns:
            dict with:
                - 'matches': list of (lf1, lf2, similarity) tuples
                - 'unmatched1': LFs from impl1 without match
                - 'unmatched2': LFs from impl2 without match
        """
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


def find_param_dir(base_path: Path, combo: str) -> Optional[Path]:
    """Find parameter directory handling R vs Python naming differences.

    R uses: 0.05_1_out, Python uses: 0.05_1.0_out
    """
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


def main():
    parser = argparse.ArgumentParser(
        description='Compare latent factor content across SLIDE implementations'
    )
    parser.add_argument('output_path', type=Path,
                       help='Path containing implementation subdirectories')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed per-LF comparison')
    parser.add_argument('--param', type=str, default=None,
                       help='Only compare specific parameter combination (e.g., "0.05_0.1_out")')

    args = parser.parse_args()

    output_path = args.output_path
    if not output_path.is_dir():
        logger.error(f"Output path not found: {output_path}")
        return 1

    # Known implementation name patterns (order matters for comparison priority)
    known_patterns = [
        "R_native", "R_outputs",
        "Py_rLOVE_rKO", "Py_rLOVE_knockpy", "Py_rLOVE_pyKO",
        "Py_pyLOVE_rKO", "Py_pyLOVE_knockpy", "Py_pyLOVE_pyKO",
        "Py_Py_LOVE", "Py_R_LOVE", "Py_outputs"
    ]

    # Find available implementations - try known names first, then discover others
    available = {}
    for name in known_patterns:
        impl_path = output_path / name
        if impl_path.is_dir():
            available[name] = impl_path

    # Also discover any other *_out parent directories
    for child in output_path.iterdir():
        if child.is_dir() and child.name not in available:
            # Check if it has *_out subdirectories (sign of SLIDE output)
            has_outputs = any(d.name.endswith('_out') for d in child.iterdir() if d.is_dir())
            if has_outputs:
                available[child.name] = child

    if len(available) < 2:
        logger.error(f"Need at least 2 implementations to compare, found: {list(available.keys())}")
        return 1

    logger.info("=" * 60)
    logger.info("Latent Factor Comparison")
    logger.info("=" * 60)
    logger.info(f"Output path: {output_path}")
    logger.info(f"Available implementations: {', '.join(available.keys())}")

    loader = LatentFactorLoader()
    comparator = LatentFactorComparator(detailed=args.detailed)

    # Find parameter combinations
    ref_impl = list(available.keys())[0]
    ref_path = available[ref_impl]
    param_combos = sorted([d.name for d in ref_path.iterdir()
                          if d.is_dir() and d.name.endswith('_out')])

    if args.param:
        param_combos = [args.param] if args.param in param_combos else []
        if not param_combos:
            # Try to find it
            found = find_param_dir(ref_path, args.param)
            if found:
                param_combos = [found.name]

    if not param_combos:
        logger.error("No parameter combinations found")
        return 1

    # Compare each parameter combination
    for combo in param_combos:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Parameter: {combo}")
        logger.info("=" * 60)

        # Load all implementations for this combo
        impl_data = {}
        for name, base_path in available.items():
            param_path = find_param_dir(base_path, combo)
            if param_path is None:
                continue

            # Determine if R or Python based on name and files present
            # R outputs have feature_list_Z*.txt, Python has A.csv and sig_LFs.txt
            is_r = (name.startswith('R_') or name == 'R_native' or name == 'R_outputs')
            # Also check file presence as fallback
            if not is_r and list(param_path.glob('feature_list_Z*.txt')):
                is_r = True
            elif is_r and (param_path / 'A.csv').exists() and (param_path / 'sig_LFs.txt').exists():
                is_r = False

            if is_r:
                data = loader.load_r_outputs(param_path)
            else:
                data = loader.load_python_outputs(param_path)

            if data:
                impl_data[name] = data

        if len(impl_data) < 2:
            logger.info(f"  Insufficient data for comparison (found {len(impl_data)} impls)")
            continue

        # Print LF count summary
        logger.info("\nLF Count Summary:")
        for name, data in impl_data.items():
            n_lfs = len(data.get('sig_LFs', []))
            logger.info(f"  {name:20s}: {n_lfs} significant LFs")

        # Compare R_native to each Python implementation
        if 'R_native' in impl_data:
            for py_name in impl_data:
                if py_name == 'R_native':
                    continue
                results = comparator.compare_implementations(
                    impl_data['R_native'], impl_data[py_name],
                    'R_native', py_name
                )
                logger.info(comparator.format_report(results))

        # Also compare Python implementations to each other
        py_impls = [n for n in impl_data if n != 'R_native']
        if len(py_impls) >= 2:
            logger.info("\nPython Implementation Cross-Comparison:")
            for i, name1 in enumerate(py_impls):
                for name2 in py_impls[i+1:]:
                    results = comparator.compare_implementations(
                        impl_data[name1], impl_data[name2],
                        name1, name2
                    )
                    logger.info(comparator.format_report(results))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Comparison complete")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
