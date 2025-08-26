import json
import os
import pandas as pd

from run_experiment_lazy import process_dataset
from prompt_utils import get_prompt, get_llama_bnf_spec


def make_sharegpt_dataset(jsonl_path, output_path="sharegpt_dataset.jsonl"):
    # If processed CSV doesn't exist yet, make it
    if os.path.exists("navigation_dataset.csv"):
        dataset_filename = "navigation_dataset.csv"
    else:
        dataset_filename = process_dataset(jsonl_path)

    df = pd.read_csv(dataset_filename)

    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            nl_sentence = row["nl_sentence"]
            propositions = eval(row["propositions"]) if isinstance(row["propositions"], str) else row["propositions"]
            ltl_formula = row["dataset_tl"]

            # get grammar-aware instruction
            bnf_spec = get_llama_bnf_spec(propositions)
            instruction = get_prompt(
                propositions=propositions,
                task=nl_sentence,
                bnf_spec=bnf_spec,
                grammar_prompt=True,
                few_shot=[]
            )

            # NL + propositions as human input
            input_text = (
                f'Natural Language Requirement - "{nl_sentence}"\n'
                f"Relevant Propositions - {', '.join(propositions)}"
            )

            conversations = [
                {"from": "system", "value": instruction.strip()},
                {"from": "human", "value": input_text.strip()},
                {"from": "gpt", "value": ltl_formula.strip()},
            ]

            f.write(json.dumps({"conversations": conversations}) + "\n")

    print(f"✅ ShareGPT dataset written to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Path to input JSONL dataset")
    parser.add_argument("--output", type=str, default="sharegpt_dataset.jsonl", help="Output ShareGPT JSONL file")
    args = parser.parse_args()

    make_sharegpt_dataset(args.dataset_jsonl, args.output)

