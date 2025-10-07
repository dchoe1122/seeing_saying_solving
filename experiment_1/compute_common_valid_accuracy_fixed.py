#!/usr/bin/env python3
"""
Script to compute accuracy for each ablation using only commonly valid specifications.

For each trial, we find specifications that are valid for ALL ablations in that trial,
then compute accuracy on this commonly valid subset.
"""

import pandas as pd
import glob
import os
import numpy as np
from typing import Dict, List, Set

def load_trial_data(results_dir: str) -> List[pd.DataFrame]:
    """Load all trial CSV files from the results directory."""
    trial_files = glob.glob(os.path.join(results_dir, "trial_*.csv"))
    trial_files.sort()  # Ensure consistent ordering

    dfs = []
    for file in trial_files:
        df = pd.read_csv(file)
        dfs.append(df)

    return dfs

def get_ablation_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Identify ablation columns and their corresponding equivalence columns."""
    ablations = {}

    # Find columns that end with '_tl' (temporal logic columns)
    tl_columns = [col for col in df.columns if col.endswith('_tl') and col != 'dataset_tl']

    for tl_col in tl_columns:
        # Extract ablation name (remove '_tl' suffix)
        ablation_name = tl_col.replace('_tl', '')

        # Find corresponding equivalence column
        equiv_col = tl_col.replace('_tl', '_equivalence')

        if equiv_col in df.columns:
            ablations[ablation_name] = {
                'tl_column': tl_col,
                'equiv_column': equiv_col
            }

    return ablations

def find_commonly_valid_entries_per_trial(df: pd.DataFrame, ablations: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Find entries that are valid for ALL ablations in this specific trial."""

    # Start with all entries
    valid_mask = pd.Series([True] * len(df))

    # For each ablation, mask out entries that are invalid
    for ablation_name, columns in ablations.items():
        equiv_col = columns['equiv_column']
        # Entry is invalid if it has "Invalid data entry" or "Invalid LLM formula"
        invalid_mask = df[equiv_col].isin(["Invalid data entry", "Invalid LLM formula"])
        valid_mask = valid_mask & ~invalid_mask

    return df[valid_mask].copy()

def compute_accuracy_on_common_valid(dfs: List[pd.DataFrame], ablations: Dict[str, Dict[str, str]]) -> Dict[str, List[float]]:
    """Compute accuracy for each ablation on commonly valid specifications."""

    results = {ablation: [] for ablation in ablations.keys()}
    commonly_valid_counts = []

    for trial_idx, df in enumerate(dfs):
        print(f"\nTrial {trial_idx + 1}:")
        print(f"  Total entries: {len(df)}")

        # Find commonly valid entries for this trial
        common_valid_df = find_commonly_valid_entries_per_trial(df, ablations)
        commonly_valid_counts.append(len(common_valid_df))

        print(f"  Commonly valid entries: {len(common_valid_df)}")

        # Compute accuracy for each ablation on the commonly valid subset
        for ablation_name, columns in ablations.items():
            # Count correct predictions (equivalence == 'True') within commonly valid entries
            correct = (common_valid_df[columns['equiv_column']] == 'True').sum()
            total = len(common_valid_df)

            accuracy = correct / total if total > 0 else 0.0
            results[ablation_name].append(accuracy)

            print(f"  {ablation_name}: {correct}/{total} = {accuracy:.4f}")

    return results, commonly_valid_counts

def print_summary_statistics(results: Dict[str, List[float]], commonly_valid_counts: List[int]):
    """Print summary statistics for each ablation."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nCommonly valid entries per trial: {commonly_valid_counts}")
    print(f"Mean commonly valid entries: {np.mean(commonly_valid_counts):.1f}")

    for ablation_name, accuracies in results.items():
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)

        print(f"\n{ablation_name}:")
        print(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Range: [{min_acc:.4f}, {max_acc:.4f}]")
        print(f"  Individual trials: {[f'{acc:.4f}' for acc in accuracies]}")

def save_results(results: Dict[str, List[float]], commonly_valid_counts: List[int], output_file: str):
    """Save results to CSV file."""

    # Prepare data for CSV
    data = []
    num_trials = len(next(iter(results.values())))

    for trial_idx in range(num_trials):
        row = {'trial': trial_idx, 'commonly_valid_count': commonly_valid_counts[trial_idx]}
        for ablation_name, accuracies in results.items():
            row[f'{ablation_name}_accuracy'] = accuracies[trial_idx]
        data.append(row)

    # Add summary statistics
    summary_row = {'trial': 'mean', 'commonly_valid_count': np.mean(commonly_valid_counts)}
    for ablation_name, accuracies in results.items():
        summary_row[f'{ablation_name}_accuracy'] = np.mean(accuracies)
    data.append(summary_row)

    summary_row = {'trial': 'std', 'commonly_valid_count': np.std(commonly_valid_counts)}
    for ablation_name, accuracies in results.items():
        summary_row[f'{ablation_name}_accuracy'] = np.std(accuracies)
    data.append(summary_row)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")

def main():
    results_dir = "trials_500entries_3trials_5examples"

    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return

    print(f"Loading data from: {results_dir}")

    # Load trial data
    dfs = load_trial_data(results_dir)
    print(f"Loaded {len(dfs)} trials")

    if not dfs:
        print("No trial data found!")
        return

    # Identify ablations
    ablations = get_ablation_columns(dfs[0])
    print(f"Found ablations: {list(ablations.keys())}")

    # Compute accuracy on commonly valid specifications
    print("\nComputing accuracy on commonly valid specifications...")
    results, commonly_valid_counts = compute_accuracy_on_common_valid(dfs, ablations)

    # Print summary
    print_summary_statistics(results, commonly_valid_counts)

    # Save results
    output_file = os.path.join(results_dir, "common_valid_accuracy_fixed.csv")
    save_results(results, commonly_valid_counts, output_file)

if __name__ == "__main__":
    main()