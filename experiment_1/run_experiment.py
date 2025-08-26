import json
import csv
import time
import os
import re
import argparse

from prompt_utils import get_prompt, get_llama_bnf_spec

# ---------- Lazy import helpers ----------

_gemma_llm = None
_openai_client = None
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    return _tokenizer

def get_grammar_processor(grammar_str):
    from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
    from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

    grammar = IncrementalGrammarConstraint(grammar_str, "root", get_tokenizer())
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar, adapter="llama-cpp-python")
    return grammar_processor

def get_gemma_llm_base():
    """Lazily load the Gemma Llama model."""
    global _gemma_llm
    if _gemma_llm is None:
        from llama_cpp import Llama
        _gemma_llm = Llama.from_pretrained(
            repo_id="google/gemma-3-27b-it-qat-q4_0-gguf",
            filename="gemma-3-27b-it-q4_0.gguf",
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False
        )
    return _gemma_llm


def get_gemma_llm():
    """Lazily load the finetuned Gemma Llama model."""
    global _gemma_llm
    if _gemma_llm is None:
        from llama_cpp import Llama
        _gemma_llm = Llama(
            model_path="./finetune/gemma-3-finetune.gguf",
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False
        )
    return _gemma_llm

def get_openai_client():
    """Lazily initialize the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        import openai
        from openai import OpenAI
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        _openai_client = OpenAI()
    return _openai_client


# ---------- Core functions ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment configuration")
    parser.add_argument("--entries", type=int, required=True,
                        help="Number of dataset entries to use")
    parser.add_argument("--trials", type=int, required=True,
                        help="Number of trials to run")
    parser.add_argument("--examples", type=int, required=True,
                        help="Number of few-shot examples to include")
    parser.add_argument("--dataset_jsonl", type=str, required=True,
                        help="Path to the input JSONL dataset")
    return parser.parse_args()


def process_dataset(jsonl_path):
    from tqdm import tqdm
    import pandas as pd  # only needed if you later load it
    output_filename = f"navigation_dataset.csv"

    with open(jsonl_path, 'r') as json_file, open(output_filename, 'w') as csv_file:
        lines = [json.loads(line) for line in json_file]
        csv_writer = csv.DictWriter(csv_file, fieldnames=['id', 'propositions', 'nl_sentence', 'dataset_tl'])
        csv_writer.writeheader()

        for line in tqdm(lines, desc="Processing dataset entries", unit="entry"):
            id = line['id']
            nl_sentence = ' '.join(line['sentence'])
            logic_ltl = ''.join(line['logic_ltl'])
            logic_ltl = logic_ltl.replace("&", " & ").replace("|", " | ").replace("->", " -> ").replace("U", " U ")
            logic_ltl = re.sub(r"-(?!>)", "~", logic_ltl)

            propositions = line['propositions']
            for prop in propositions:
                if (len(propositions[prop]['prop']) == 1 and len(propositions[prop]['prop'][0]) <= 2):
                    propositions[prop]['prop'] = '_'.join(strip_part_of_speech(word)
                                                          for word in propositions[prop]['prop'][0])
                    propositions[prop].pop('span')
                    logic_ltl = logic_ltl.replace(prop, propositions[prop]['prop'])
                else:
                    print("Error: more than one element")
                    breakpoint()
            propositions = [propositions[prop]['prop'] for prop in propositions]

            csv_writer.writerow({'id': id, 'propositions': propositions, 'nl_sentence': nl_sentence, 'dataset_tl': logic_ltl})

    return output_filename


def strip_part_of_speech(word):
    parts = word.rsplit('_', 1)
    if (parts[1] not in ('v', 'n')):
        print("Error: not a verb or noun")
        breakpoint()
    return parts[0].replace(' ', '_').replace("'", "")


def nl2tl_gemma(nl_sentence, propositions, few_shot_examples, grammar_constraint, grammar_prompt):
    from llama_cpp import LlamaGrammar
    prompt = get_prompt(propositions=propositions, task=nl_sentence,
                        bnf_spec=get_llama_bnf_spec(propositions),
                        few_shot=few_shot_examples, grammar_prompt=grammar_prompt)

    tl_bnf_grammar = LlamaGrammar.from_string(get_llama_bnf_spec(propositions))
    grammar_str = get_llama_bnf_spec(propositions)

    system_prompt = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
    query = [{"role": "user", "content": [{"type": "text",
              "text": f"Natural Language Requirement - \"{nl_sentence}\"\nRelevant Propositions - {str(propositions)[1:-1]}"}]}]

    out = get_gemma_llm_base().create_chat_completion(
        messages=system_prompt + few_shot_examples + query,
        grammar = tl_bnf_grammar
        #logits_processor = [get_grammar_processor(grammar_str)] if grammar_constraint else None
    )

    response = out['choices'][0]['message']['content']
    response = re.sub(r"-(?!>)", "_", response)
    response = response.replace("&", " & ").replace("|", " | ").replace("->", " -> ").replace("U", " U ")
    
    return response.strip()


def nl2tl_gpt4(nl_sentence, propositions, few_shot_examples, grammar_prompt):
    prompt = get_prompt(propositions=propositions, task=nl_sentence,
                        bnf_spec=get_llama_bnf_spec(propositions),
                        few_shot=few_shot_examples, grammar_prompt=grammar_prompt)

    system_prompt = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
    query = [{"role": "user", "content": [{"type": "text",
              "text": f"Natural Language Requirement - \"{nl_sentence}\"\nRelevant Propositions - {str(propositions)[1:-1]}"}]}]
    messages = system_prompt + few_shot_examples + query

    out = get_openai_client().chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        response_format={"type": "text"}
    )

    response = out.choices[0].message.content
    response = re.sub(r"-(?!>)", "_", response)
    return response.strip()


def get_few_shot_examples(df, num_examples):
    df = df.sample(n=num_examples)
    few_shot_examples = []
    for _, row in df.iterrows():
        nl_sentence = row['nl_sentence']
        logic_ltl = row['dataset_tl']
        propositions = eval(row['propositions'])
        user_prompt = f"Natural Language Requirement - \"{nl_sentence}\"\nRelevant Propositions - {str(propositions)[1:-1]}"
        assistant_prompt = logic_ltl
        few_shot_examples.extend([
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_prompt}]}
        ])
    return few_shot_examples


def safe_are_equivalent(true_ltl, llm_ltl):
    import spot
    try:
        return spot.are_equivalent(true_ltl, llm_ltl)
    except Exception:
        try:
            spot.formula(true_ltl)
        except Exception:
            return "Invalid data entry"
        try:
            spot.formula(llm_ltl)
        except Exception:
            return "Invalid LLM formula"
        print([true_ltl, llm_ltl])
        return "ERROR"


# ---------- Main experiment ----------

if __name__ == "__main__":
    from tqdm import tqdm
    import pandas as pd

    args = parse_args()
    num_dataset_entries = args.entries
    num_trials = args.trials
    num_examples = args.examples

    dataset_filename = "navigation_dataset.csv" if os.path.exists("navigation_dataset.csv") else process_dataset(args.dataset_jsonl)
    full_dataset = pd.read_csv(dataset_filename)
    summary_df = pd.DataFrame()

    experiment_dir_name = f"trials_{num_dataset_entries}entries_{num_trials}trials_{num_examples}examples"
    os.makedirs(experiment_dir_name, exist_ok=True)

   # ablations = [
   #     {'name': 'gemma_Pc', 'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, False, True)},
   #     {'name': 'gemma_PC', 'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, True, True)},
   # ]


    ablations = [
        {
            'name': 'gemma_Pc',
            'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, grammar_constraint=False, grammar_prompt=True)
        },
        {
            'name': 'gemma_PC',
            'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, grammar_constraint=True, grammar_prompt=True)
        },
    #    {
    #        'name': 'gemma_pc',
    #        'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, grammar_constraint=False, grammar_prompt=False)
    #    },
    #    {
    #        'name': 'gemma_pC',
    #        'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, grammar_constraint=True, grammar_prompt=False)
    #    },
    ]

    for i in range(num_trials):
        print(f"Trial {i+1}:")
        few_shot_examples = get_few_shot_examples(full_dataset, num_examples)
        summary_df_row_idx = len(summary_df)
        df = full_dataset.sample(n=num_dataset_entries).reset_index(drop=True)

        for ablation in ablations:
            tqdm.pandas(desc=ablation['name'])
            start_time = time.time()
            df[ablation['name'] + '_tl'] = df.progress_apply(ablation['function'], axis=1)
            total_time = time.time() - start_time

            df[ablation['name'] + '_equivalence'] = df.apply(lambda row: safe_are_equivalent(row['dataset_tl'], row[ablation['name'] + '_tl']), axis=1)
            equivalence_counts = df[ablation['name'] + '_equivalence'].value_counts()

            accuracy = equivalence_counts.get(True, 0) / (num_dataset_entries - equivalence_counts.get("Invalid data entry", 0) - equivalence_counts.get("Invalid LLM formula", 0))
            validity = 1 - equivalence_counts.get("Invalid LLM formula", 0) / (num_dataset_entries - equivalence_counts.get("Invalid data entry", 0))
            inference_time = total_time / num_dataset_entries

            summary_df.loc[summary_df_row_idx,
                           [ablation['name'] + '_accuracy', ablation['name'] + '_validity', ablation['name'] + '_time']] = \
                [accuracy, validity, inference_time]

        summary_df.index.name = "Trial"
        summary_df.to_csv(f"{experiment_dir_name}/experiment_summary.csv", index=True)
        df.to_csv(f"{experiment_dir_name}/trial_{i+1}.csv", index=False)

    summary_df.index.name = "Trial"
    summary_df.to_csv(f"{experiment_dir_name}/experiment_summary.csv", index=True)
    summary_df = summary_df.astype(float)
    with open(f"{experiment_dir_name}/summary_stats.txt", "w") as f:
        f.write(summary_df.describe().round(4).to_string())

