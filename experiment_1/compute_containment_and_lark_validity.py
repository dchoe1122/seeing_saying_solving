import argparse
import pandas as pd
from failure_analysis import generate_containment_stats, generate_grammar_validity_stats

argparser = argparse.ArgumentParser()
argparser.add_argument('--experiment_dir', type=str, required=True, help='Directory containing experiment CSV files')

args = argparser.parse_args()
experiment_dir = args.experiment_dir

# Collect results across all trials
all_containment_results = []
all_validity_results = []

for trial_num in range(3):
    experiment_csv = f"{experiment_dir}/trial_{trial_num+1}.csv"
    print(f"Processing {experiment_csv}")
    exp_df = pd.read_csv(experiment_csv)

    containment_df = generate_containment_stats(exp_df)
    validity_df = generate_grammar_validity_stats(exp_df)

    # Add trial number to the dataframes
    containment_df['trial'] = trial_num
    validity_df['trial'] = trial_num

    all_containment_results.append(containment_df)
    all_validity_results.append(validity_df)

# Combine all trial results
combined_containment = pd.concat(all_containment_results, ignore_index=False)
combined_validity = pd.concat(all_validity_results, ignore_index=False)

# Reset index to make ablation a column for easier grouping
combined_containment = combined_containment.reset_index()
combined_validity = combined_validity.reset_index()

# Calculate mean and standard deviation for containment metrics
print("\n=== CONTAINMENT STATISTICS ===")
containment_summary = combined_containment.groupby('ablation').agg({
    'LLM_in_true': ['mean', 'std'],
    'true_in_LLM': ['mean', 'std']
}).round(4)

print("LLM_in_true (Mean ± Std):")
for ablation in containment_summary.index:
    mean_val = containment_summary.loc[ablation, ('LLM_in_true', 'mean')]
    std_val = containment_summary.loc[ablation, ('LLM_in_true', 'std')]
    print(f"  {ablation}: {mean_val:.4f} ± {std_val:.4f}")

print("\ntrue_in_LLM (Mean ± Std):")
for ablation in containment_summary.index:
    mean_val = containment_summary.loc[ablation, ('true_in_LLM', 'mean')]
    std_val = containment_summary.loc[ablation, ('true_in_LLM', 'std')]
    print(f"  {ablation}: {mean_val:.4f} ± {std_val:.4f}")

# Calculate mean and standard deviation for validity metrics
print("\n=== VALIDITY STATISTICS ===")
validity_summary = combined_validity.groupby('ablation').agg({
    'validity': ['mean', 'std']
}).round(4)

print("Validity (Mean ± Std):")
for ablation in validity_summary.index:
    mean_val = validity_summary.loc[ablation, ('validity', 'mean')]
    std_val = validity_summary.loc[ablation, ('validity', 'std')]
    print(f"  {ablation}: {mean_val:.4f} ± {std_val:.4f}")

# Save detailed results to CSV
containment_summary.to_csv(f"{experiment_dir}/containment_summary.csv")
validity_summary.to_csv(f"{experiment_dir}/validity_summary.csv")

print(f"\nDetailed results saved to:")
print(f"  {experiment_dir}/containment_summary.csv")
print(f"  {experiment_dir}/validity_summary.csv")