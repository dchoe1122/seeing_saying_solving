import spot
import argparse
import pandas as pd
from prompt_utils import get_llama_bnf_spec
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer

def parse_args():
    argparser = argparse.ArgumentParser(description="Failure analysis for NL to TL translation")
    argparser.add_argument("--experiment_csv", type=str, required=False, help="Path to CSV containing experiment trial results", default="trials_250entries_1trials_10examples/trial_1.csv")
    return argparser.parse_args()

def analyze_constrained_loss(exp_df):
    # This function is meant to analyze the loss in accuracy due to constrained generation
    gap_entries = exp_df[((exp_df['gemma_Pc_equivalence'] == "True") | (exp_df['gemma_Pc_equivalence'] == True)) & ((exp_df['gemma_PC_equivalence'] == "False") | (exp_df['gemma_PC_equivalence'] == False))]
    gap_entries = gap_entries[['nl_sentence', 'propositions', 'dataset_tl', 'gemma_Pc_tl', 'gemma_PC_tl']]
    print(f"Number of entries where constrained generation failed but unconstrained succeeded: {len(gap_entries)}")

    # Check if the unconstrained translations still meet the EBNF grammar using Lark
    for index, row in gap_entries.iterrows():
        grammar = parse_ebnf(get_llama_bnf_spec(propositions=eval(row['propositions'])))
        recognizer = StringRecognizer(grammar.grammar_encoding, grammar.symbol_table["root"])
        
        if recognizer._accept_prefix(row['gemma_Pc_tl'].replace('_','-')):
            print(f"Row {index}: Unconstrained translation is valid (EBNF).\n")
        else:
            print(f"Row {index}: Unconstrained translation is INVALID (EBNF).\nConstrained TL: {row['gemma_PC_tl']}\nUnconstrained TL: {row['gemma_Pc_tl']}\n")

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

    analyze_constrained_loss(exp_df)
    #containment_df = generate_containment_stats(exp_df)
    #print(containment_df)
