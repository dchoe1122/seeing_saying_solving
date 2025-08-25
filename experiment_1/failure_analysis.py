import spot
import argparse
import pandas as pd

def parse_args():
    argparser = argparse.ArgumentParser(description="Failure analysis for NL to TL translation")
    argparser.add_argument("--experiment_csv", type=str, required=True, help="Path to CSV containing experiment trial results")
    return argparser.parse_args()


def generate_containment_stats(exp_df):
    containment_results = []
    ablations = [col[:-3] for col in exp_df.columns if col.endswith('_tl')]
    ablations.pop(ablations.index('dataset'))  # remove the dataset column

    for ablation in ablations:
        inaccurate_rows = exp_df[~exp_df[f'{ablation}_equivalence']]
        num_inaccurate = len(inaccurate_rows)
        num_llm_in_true = inaccurate_rows.apply(lambda row: spot.contains(spot.formula(row['dataset_tl']), spot.formula(row[f'{ablation}_tl'])), axis=1).sum()
        num_true_in_llm = inaccurate_rows.apply(lambda row: spot.contains(spot.formula(row[f'{ablation}_tl']), spot.formula(row['dataset_tl'])), axis=1).sum()

        containment_results.append({
            'ablation': ablation,
            'LLM_in_true': num_llm_in_true / num_inaccurate if num_inaccurate > 0 else 0,
            'true_in_LLM': num_true_in_llm / num_inaccurate if num_inaccurate > 0 else 0
        })

    #summarize containment results
    containment_df = pd.DataFrame(containment_results)
    containment_df.set_index('ablation', inplace=True)

    return containment_df


if __name__ == "__main__":
    args = parse_args()
    experiment_csv = args.experiment_csv

    exp_df = pd.read_csv(experiment_csv)

    containment_df = generate_containment_stats(exp_df)
    print(containment_df)