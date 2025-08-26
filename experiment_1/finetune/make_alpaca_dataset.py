import json
import os
import pandas as pd

from run_experiment_lazy import process_dataset
from prompt_utils import get_prompt, get_llama_bnf_spec


def make_alpaca_dataset(jsonl_path, output_path="alpaca_dataset.json"):
    # If processed CSV doesn't exist yet, make it
    if os.path.exists("navigation_dataset.csv"):
        dataset_filename = "navigation_dataset.csv"
    else:
        dataset_filename = process_dataset(jsonl_path)

    df = pd.read_csv(dataset_filename)

    alpaca_entries = []
    for _, row in df.iterrows():
        nl_sentence = row["nl_sentence"]
        propositions = eval(row["propositions"]) if isinstance(row["propositions"], str) else row["propositions"]
        ltl_formula = row["dataset_tl"]

        # get the BNF spec for this set of propositions
        bnf_spec = get_llama_bnf_spec(propositions)

        # use grammar guidance baked in
        instruction = get_prompt(
            propositions=propositions,
            task=nl_sentence,
            bnf_spec=bnf_spec,
            grammar_prompt=True,
            few_shot=[]
        )

        input_text = (
            f'Natural Language Requirement - "{nl_sentence}"\n'
            f"Relevant Propositions - {', '.join(propositions)}"
        )

        entry = {
            "instruction": instruction.strip(),
            "input": input_text.strip(),
            "output": ltl_formula.strip(),
        }
        alpaca_entries.append(entry)

    with open(output_path, "w") as f:
        json.dump(alpaca_entries, f, indent=2)

    print(f"✅ Alpaca dataset written to {output_path} with {len(alpaca_entries)} entries")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Path to input JSONL dataset")
    parser.add_argument("--output", type=str, default="alpaca_dataset.json", help="Output Alpaca JSON file")
    args = parser.parse_args()

    make_alpaca_dataset(args.dataset_jsonl, args.output)

