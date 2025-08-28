#!/usr/bin/env python3
"""
Comprehensive Statistical Testing for TS-HMM Behavioral Validation

This script performs statistical tests to validate the behavioral significance of:
1. Cluster assignments vs meditation types
2. Transition frequencies and bridge state usage
3. Event-locked analysis around transitions
4. Cross-dataset consistency

Statistical tests used:
- Chi-square tests for categorical associations
- Permutation tests for temporal alignment
- Binomial tests for bridge state frequency
- Non-parametric tests for repeated measures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (chi2_contingency, chisquare, binom_test, 
                        friedmanchisquare, wilcoxon, kruskal, 
                        mannwhitneyu)
try:
    from scipy.stats import permutation_test, bootstrap
except ImportError:
    # For older scipy versions
    permutation_test = None
    bootstrap = None
import warnings
warnings.filterwarnings('ignore')

class BehavioralValidator:
    """Statistical validation of TS-HMM behavioral outcomes."""
    
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.results = {}
        
    def load_data(self):
        """Load relevant datasets for validation."""
        try:
            # Load main HMM output
            self.hmm_data = pd.read_csv(f"{self.data_path}/HMM_output_adjusted.csv")
            
            # Load transitions data if available
            try:
                self.transitions_data = pd.read_csv(f"{self.data_path}/transitions_summary.csv")
            except FileNotFoundError:
                self.transitions_data = None
                
            print(f"Loaded HMM data: {len(self.hmm_data)} observations")
            if self.transitions_data is not None:
                print(f"Loaded transitions data: {len(self.transitions_data)} transitions")
                
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return False
        return True
    
    def test_cluster_meditation_association(self):
        """Test association between cluster labels and meditation types."""
        print("\n=== CLUSTER-MEDITATION TYPE ASSOCIATION ===")
        
        if 'Med_type' not in self.hmm_data.columns or 'transition_label' not in self.hmm_data.columns:
            print("Required columns not found for meditation type analysis")
            return
            
        # Create contingency table
        contingency = pd.crosstab(self.hmm_data['Med_type'], self.hmm_data['transition_label'])
        print(f"Contingency table:\n{contingency}")
        
        # Chi-square test of independence
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
        
        # Effect size (Cramér's V)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))
        
        # Standardized residuals for interpretation
        std_residuals = (contingency - expected) / np.sqrt(expected)
        
        results = {
            'test': 'Chi-square independence',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'significant': p_value < 0.05,
            'effect_size': 'Small' if cramers_v < 0.1 else ('Medium' if cramers_v < 0.3 else 'Large'),
            'interpretation': self._interpret_chi2_result(p_value, cramers_v)
        }
        
        self.results['meditation_association'] = results
        
        print(f"χ² = {chi2_stat:.3f}, p = {p_value:.6f}, Cramér's V = {cramers_v:.3f}")
        print(f"Result: {results['interpretation']}")
        
        # Post-hoc analysis if significant
        if p_value < 0.05:
            print("\nPost-hoc analysis (standardized residuals > |2| are significant):")
            for med_type in contingency.index:
                for cluster in contingency.columns:
                    residual = std_residuals.loc[med_type, cluster]
                    if abs(residual) > 2:
                        direction = "over" if residual > 0 else "under"
                        print(f"  {med_type} → Cluster {cluster}: {direction}-represented (z = {residual:.2f})")
        
        return results
    
    def test_bridge_state_frequency(self):
        """Test if bridge state transitions are significantly enriched."""
        print("\n=== BRIDGE STATE FREQUENCY TEST ===")
        
        if self.transitions_data is None:
            print("No transitions data available")
            return
        
        # Identify available columns for transitions
        transition_cols = [col for col in self.transitions_data.columns 
                          if any(x in col.lower() for x in ['transition', 'from', 'to'])]
        print(f"Available transition columns: {transition_cols}")
        
        # Use 'From State' and 'To State' columns to create transitions
        if 'From State' in self.transitions_data.columns and 'To State' in self.transitions_data.columns:
            # Create transition labels
            transitions = (self.transitions_data['From State'].astype(str) + '→' + 
                          self.transitions_data['To State'].astype(str))
            transition_counts = transitions.value_counts()
            
            # Count transitions involving bridge state (assuming state 3 is bridge)
            bridge_transitions = sum(count for trans, count in transition_counts.items() 
                                   if '3' in trans)
            total_transitions = transition_counts.sum()
            
            # Expected proportion under null (4 out of 9 possible transitions involve bridge)
            expected_prop = 4/9
            observed_prop = bridge_transitions / total_transitions
            
            # Binomial test
            p_value = binom_test(bridge_transitions, total_transitions, expected_prop, alternative='greater')
            
            # Effect size (Cohen's h for proportions)
            cohens_h = 2 * (np.arcsin(np.sqrt(observed_prop)) - np.arcsin(np.sqrt(expected_prop)))
            
            results = {
                'test': 'Binomial test (bridge enrichment)',
                'observed_bridge_transitions': bridge_transitions,
                'total_transitions': total_transitions,
                'observed_proportion': observed_prop,
                'expected_proportion': expected_prop,
                'p_value': p_value,
                'cohens_h': cohens_h,
                'significant': p_value < 0.05,
                'interpretation': f"Bridge transitions {'enriched' if p_value < 0.05 else 'not enriched'}"
            }
            
            self.results['bridge_frequency'] = results
            
            print(f"Observed: {bridge_transitions}/{total_transitions} ({observed_prop:.3f})")
            print(f"Expected: {expected_prop:.3f}")
            print(f"Binomial test p = {p_value:.6f}")
            print(f"Cohen's h = {cohens_h:.3f}")
            print(f"Result: {results['interpretation']}")
            
            return results
        else:
            print("Required columns 'From State' and 'To State' not found")
            return
    
    def test_temporal_alignment(self, event_column='Breath Freq', window_size=5):
        """Test if transitions align with external events using permutation test."""
        print(f"\n=== TEMPORAL ALIGNMENT TEST ({event_column}) ===")
        
        if self.transitions_data is None:
            print("No transitions data available")
            return
            
        # Get event frequencies during transitions
        during_col = f'During {event_column}'
        if during_col not in self.transitions_data.columns:
            print(f"Column {during_col} not found")
            return
            
        observed_during = self.transitions_data[during_col].dropna()
        
        # Compare with before/after periods
        before_col = f'Before {event_column}'
        after_col = f'After {event_column}'
        
        if before_col in self.transitions_data.columns and after_col in self.transitions_data.columns:
            # Drop rows with any missing values
            baseline = pd.concat([
                self.transitions_data[before_col].dropna(),
                self.transitions_data[after_col].dropna()
            ])
            
            if len(observed_during) == 0 or len(baseline) == 0:
                print(f"Insufficient data for {event_column}")
                return
            
            # Permutation test
            def test_statistic(x, y):
                return np.mean(x) - np.mean(y)
            
            # Two-sample permutation test
            observed_stat = test_statistic(observed_during, baseline)
            
            # Manual permutation test (simplified)
            n_permutations = 10000
            combined = np.concatenate([observed_during, baseline])
            n_during = len(observed_during)
            
            perm_stats = []
            for _ in range(n_permutations):
                np.random.shuffle(combined)
                perm_during = combined[:n_during]
                perm_baseline = combined[n_during:]
                perm_stats.append(test_statistic(perm_during, perm_baseline))
            
            p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
            
            # Effect size (Cohen's d) with safety checks
            pooled_std = np.sqrt(((len(observed_during)-1)*np.var(observed_during) + 
                                 (len(baseline)-1)*np.var(baseline)) / 
                                (len(observed_during) + len(baseline) - 2))
            
            cohens_d = observed_stat / pooled_std if pooled_std > 0 else 0
            
            results = {
                'test': 'Permutation test (temporal alignment)',
                'event': event_column,
                'observed_statistic': observed_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'interpretation': f"{'Significant' if p_value < 0.05 else 'No'} temporal alignment"
            }
            
            self.results[f'temporal_alignment_{event_column}'] = results
            
            print(f"Observed difference: {observed_stat:.4f}")
            print(f"Permutation test p = {p_value:.6f}")
            print(f"Cohen's d = {cohens_d:.3f}")
            print(f"Result: {results['interpretation']}")
            
            return results
    
    def test_temporal_alignment_threeway(self, event_column='Breath Freq'):
        """Test if event frequencies differ across before/during/after using Friedman test and post-hoc Wilcoxon tests."""
        print(f"\n=== TEMPORAL ALIGNMENT THREE-WAY TEST (Before/During/After: {event_column}) ===")
        if self.transitions_data is None:
            print("No transitions data available")
            return

        before_col = f'Before {event_column}'
        during_col = f'During {event_column}'
        after_col = f'After {event_column}'

        if not all(col in self.transitions_data.columns for col in [before_col, during_col, after_col]):
            print(f"Missing columns for {event_column}")
            return

        # Drop rows with any missing values
        df = self.transitions_data[[before_col, during_col, after_col]].dropna()
        
        if len(df) < 3:
            print(f"Insufficient data for {event_column} (only {len(df)} complete observations)")
            return
            
        before = df[before_col].values
        during = df[during_col].values
        after = df[after_col].values

        # Check if all values are identical (would cause Friedman test to fail)
        if len(set(before)) == 1 and len(set(during)) == 1 and len(set(after)) == 1:
            print(f"All values identical for {event_column} - no variation to test")
            return

        try:
            # Friedman test (non-parametric repeated measures)
            stat, p_value = stats.friedmanchisquare(before, during, after)
            print(f"Friedman test: χ² = {stat:.3f}, p = {p_value:.6f}")
            
            results = {
                'test': 'Friedman test (temporal alignment threeway)',
                'event': event_column,
                'friedman_stat': stat,
                'friedman_p': p_value,
                'significant': p_value < 0.05,
                'interpretation': f"{'Significant' if p_value < 0.05 else 'No'} difference across periods",
                'before_median': np.median(before),
                'during_median': np.median(during),
                'after_median': np.median(after)
            }
            
            if p_value < 0.05:
                print("Significant difference across periods. Post-hoc Wilcoxon tests:")
                # Pairwise Wilcoxon signed-rank tests
                pairs = [('Before', before, 'During', during),
                         ('During', during, 'After', after),
                         ('Before', before, 'After', after)]
                
                for name1, arr1, name2, arr2 in pairs:
                    try:
                        w_stat, w_p = stats.wilcoxon(arr1, arr2)
                        print(f"  {name1} vs {name2}: Wilcoxon p = {w_p:.6f}")
                        results[f'wilcoxon_{name1.lower()}_vs_{name2.lower()}_p'] = w_p
                    except ValueError as e:
                        print(f"  {name1} vs {name2}: Cannot compute (no differences)")
                        results[f'wilcoxon_{name1.lower()}_vs_{name2.lower()}_p'] = None
            else:
                print("No significant difference across periods.")
                
            self.results[f'temporal_alignment_threeway_{event_column.replace(" ", "_").lower()}'] = results
            return results
            
        except ValueError as e:
            print(f"Error running Friedman test for {event_column}: {e}")
            return

    def test_state_dwell_times(self):
        """Test if bridge state has shorter dwell times than stable states."""
        print("\n=== STATE DWELL TIME ANALYSIS ===")
        
        if 'transition_label' not in self.hmm_data.columns:
            print("No transition labels found")
            return
            
        # Calculate run lengths for each state
        state_runs = []
        current_state = None
        current_length = 0
        
        for state in self.hmm_data['transition_label']:
            if state == current_state:
                current_length += 1
            else:
                if current_state is not None:
                    state_runs.append({'state': current_state, 'length': current_length})
                current_state = state
                current_length = 1
        
        # Add final run
        if current_state is not None:
            state_runs.append({'state': current_state, 'length': current_length})
        
        runs_df = pd.DataFrame(state_runs)
        
        if len(runs_df) == 0:
            print("No state runs found")
            return
            
        # Separate bridge from stable states (assuming state 3 is bridge)
        bridge_runs = runs_df[runs_df['state'] == 3]['length']
        stable_runs = runs_df[runs_df['state'] != 3]['length']
        
        if len(bridge_runs) == 0:
            print("No bridge state runs found")
            return
        if len(stable_runs) == 0:
            print("No stable state runs found")
            return
        
        # Mann-Whitney U test (non-parametric)
        try:
            statistic, p_value = stats.mannwhitneyu(bridge_runs, stable_runs, alternative='less')
            
            # Effect size (rank-biserial correlation)
            r = 1 - (2 * statistic) / (len(bridge_runs) * len(stable_runs))
            
            results = {
                'test': 'Mann-Whitney U (dwell times)',
                'bridge_median_dwell': np.median(bridge_runs),
                'stable_median_dwell': np.median(stable_runs),
                'statistic': statistic,
                'p_value': p_value,
                'effect_size_r': r,
                'significant': p_value < 0.05,
                'interpretation': f"Bridge state {'has shorter' if p_value < 0.05 else 'does not have shorter'} dwell times"
            }
            
            self.results['dwell_times'] = results
            
            print(f"Bridge median dwell: {np.median(bridge_runs):.1f} epochs")
            print(f"Stable median dwell: {np.median(stable_runs):.1f} epochs")
            print(f"Mann-Whitney U p = {p_value:.6f}")
            print(f"Effect size r = {r:.3f}")
            print(f"Result: {results['interpretation']}")
            
            return results
        except ValueError as e:
            print(f"Error in Mann-Whitney U test: {e}")
            return
    
    def test_event_presence_by_period(self, event_column='Breath Freq'):
        """Test if event frequency is significantly greater than zero in each period (Before, During, After)."""
        print(f"\n=== EVENT PRESENCE TESTS ({event_column}) ===")
        if self.transitions_data is None:
            print("No transitions data available")
            return

        periods = ['Before', 'During', 'After']
        results = {}
        for period in periods:
            col = f'{period} {event_column}'
            if col not in self.transitions_data.columns:
                print(f"Missing column: {col}")
                continue
            values = self.transitions_data[col].dropna().values
            if len(values) == 0:
                print(f"No data for {col}")
                continue
            median = np.median(values)
            if np.all(values == 0):
                print(f"{period}: median = 0.000, Wilcoxon p = N/A (all zeros)")
                results[f'{period.lower()}_p'] = None
                results[f'{period.lower()}_median'] = median
            else:
                try:
                    stat, p = stats.wilcoxon(values, zero_method='wilcox', alternative='greater')
                    print(f"{period}: median = {median:.3f}, Wilcoxon p = {p:.6f}")
                    results[f'{period.lower()}_p'] = p
                    results[f'{period.lower()}_median'] = median
                except ValueError as e:
                    print(f"{period}: median = {median:.3f}, Wilcoxon p = N/A ({e})")
                    results[f'{period.lower()}_p'] = None
                    results[f'{period.lower()}_median'] = median
        self.results[f'event_presence_{event_column.replace(" ", "_").lower()}'] = results
        return results

    def test_event_by_transition_type(self, event_column='Breath Freq'):
        """Test event frequency for each transition type separately (median, % nonzero, and Wilcoxon significance)."""
        print(f"\n=== EVENT PRESENCE BY TRANSITION TYPE ({event_column}) ===")
        if self.transitions_data is None:
            print("No transitions data available")
            return
        if 'From State' not in self.transitions_data.columns or 'To State' not in self.transitions_data.columns:
            print("Transition columns not found")
            return
        periods = ['Before', 'During', 'After']
        transition_types = self.transitions_data[['From State', 'To State']].drop_duplicates()
        # For visualization
        summary = []
        for _, row in transition_types.iterrows():
            from_state, to_state = row['From State'], row['To State']
            mask = (self.transitions_data['From State'] == from_state) & (self.transitions_data['To State'] == to_state)
            subset = self.transitions_data[mask]
            print(f"\nTransition {from_state}→{to_state}: n={len(subset)}")
            period_values = {}
            for period in periods:
                col = f'{period} {event_column}'
                if col not in subset.columns:
                    continue
                values = subset[col].dropna().values
                period_values[period] = values
                prop_nonzero = np.mean(values > 0) if len(values) > 0 else 0
                median = np.median(values) if len(values) > 0 else 0
                # Wilcoxon test vs zero
                if len(values) > 0 and np.any(values > 0):
                    try:
                        stat, p = stats.wilcoxon(values, zero_method='wilcox', alternative='greater')
                        print(f"  {period}: median={median:.3f}, %nonzero={prop_nonzero:.2%}, Wilcoxon p={p:.4f}")
                    except ValueError:
                        print(f"  {period}: median={median:.3f}, %nonzero={prop_nonzero:.2%}, Wilcoxon p=N/A")
                else:
                    print(f"  {period}: median={median:.3f}, %nonzero={prop_nonzero:.2%}, Wilcoxon p=N/A")
                summary.append({
                    'Transition': f'{from_state}→{to_state}',
                    'Period': period,
                    'Median': median,
                    '%Nonzero': prop_nonzero,
                })
            # Compare During vs Before
            if 'During' in period_values and 'Before' in period_values:
                try:
                    stat, p = stats.wilcoxon(period_values['During'], period_values['Before'])
                    print(f"  During vs Before: Wilcoxon p={p:.4f}")
                except ValueError:
                    print("  During vs Before: Wilcoxon p=N/A")
            # Compare During vs After
            if 'During' in period_values and 'After' in period_values:
                try:
                    stat, p = stats.wilcoxon(period_values['During'], period_values['After'])
                    print(f"  During vs After: Wilcoxon p={p:.4f}")
                except ValueError:
                    print("  During vs After: Wilcoxon p=N/A")
        # Visualization: heatmap of % nonzero and median
        summary_df = pd.DataFrame(summary)
        pivot_nonzero = summary_df.pivot(index='Transition', columns='Period', values='%Nonzero')
        pivot_median = summary_df.pivot(index='Transition', columns='Period', values='Median')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(pivot_nonzero, annot=True, fmt='.2f', cmap='viridis', ax=axes[0])
        axes[0].set_title(f'% Nonzero {event_column}')
        sns.heatmap(pivot_median, annot=True, fmt='.2f', cmap='magma', ax=axes[1])
        axes[1].set_title(f'Median {event_column}')
        plt.tight_layout()
        plt.show()
        return

    def _interpret_chi2_result(self, p_value, cramers_v):
        """Interpret chi-square test results."""
        if p_value >= 0.05:
            return "No significant association"
        
        significance = "Highly significant" if p_value < 0.001 else "Significant"
        effect = "small" if cramers_v < 0.1 else ("medium" if cramers_v < 0.3 else "large")
        
        return f"{significance} association with {effect} effect size"
    
    def run_full_validation(self):
        """Run all behavioral validation tests."""
        print("=== TS-HMM BEHAVIORAL VALIDATION ===")
        
        if not self.load_data():
            return
        
        # Run all tests
        self.test_cluster_meditation_association()
        self.test_bridge_state_frequency()
        
        # Test event presence and temporal alignment for all cues
        cues = ['Intro Freq', 'Breath Freq', 'Hold Freq', 'Rest Freq', 'END Freq', 'Fast Freq']
        for cue in cues:
            self.test_event_presence_by_period(cue)
            self.test_temporal_alignment_threeway(cue)
            
        self.test_state_dwell_times()
        # Per-transition event analysis for Breath Freq and Hold Freq
        for cue in ['Breath Freq', 'Hold Freq']:
            self.test_event_by_transition_type(cue)
        
        # Summary
        print("\n=== VALIDATION SUMMARY ===")
        significant_tests = []
        for test_name, result in self.results.items():
            # Check for significance in different result structures
            is_significant = (
                result.get('significant', False) or
                (result.get('friedman_p', 1) < 0.05) or
                any(p < 0.05 for key, p in result.items() 
                    if key.endswith('_p') and isinstance(p, (int, float)))
            )
            if is_significant:
                significant_tests.append(test_name)
        
        print(f"Total tests: {len(self.results)}")
        print(f"Significant results: {len(significant_tests)}")
        if significant_tests:
            print(f"Significant tests: {', '.join(significant_tests)}")
        else:
            print("No significant results found")
        
        # Save results with better handling of different result structures
        results_list = []
        for test_name, result in self.results.items():
            # Handle different p-value structures
            p_val = (
                result.get('p_value') or 
                result.get('friedman_p') or
                'Multiple' if any(key.endswith('_p') for key in result.keys()) else 'N/A'
            )
            
            # Handle significance
            is_sig = (
                result.get('significant', False) or
                (result.get('friedman_p', 1) < 0.05) or
                any(p < 0.05 for key, p in result.items() 
                    if key.endswith('_p') and isinstance(p, (int, float)))
            )
            
            results_list.append({
                'test_name': test_name,
                'p_value': p_val,
                'significant': is_sig,
                'interpretation': result.get('interpretation', 'See detailed results')
            })
        
        results_df = pd.DataFrame(results_list)
        
        results_df.to_csv(f"{self.output_dir}/behavioral_validation_results.csv", index=False)
        print(f"\nResults saved to: {self.output_dir}/behavioral_validation_results.csv")
        
        return self.results

def main():
    """Main function to run behavioral validation."""
    data_path = "/Users/a_fin/Desktop/Year 4/Project/Data"
    output_dir = "/Users/a_fin/Desktop/Year 4/Project/ASC_project"
    
    validator = BehavioralValidator(data_path, output_dir)
    results = validator.run_full_validation()
    
    return results

if __name__ == "__main__":
    results = main()