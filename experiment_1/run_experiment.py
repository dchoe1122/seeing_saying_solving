import json
from tqdm import tqdm
import csv
import time
import pandas as pd
from llama_cpp import LlamaGrammar, Llama
import spot
import os
import re
import argparse
from openai import OpenAI
import openai

from prompt_utils import get_prompt, get_llama_bnf_spec

gemma_llm = Llama.from_pretrained(
	repo_id="google/gemma-3-27b-it-qat-q4_0-gguf",
	filename="gemma-3-27b-it-q4_0.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

openai_client = OpenAI()

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment configuration")

    parser.add_argument("--entries", type=int, required=True,
                        help="Number of dataset entries to use")
    parser.add_argument("--trials", type=int, required=True,
                        help="Number of trials to run")
    parser.add_argument("--examples", type=int, required=True,
                        help="Number of few-shot examples to include")
    parser.add_argument("--dataset_jsonl", type=str, required=True,
                        help="Path to the input JSONL dataset (e.g., navi_total_refined.jsonl)")
    return parser.parse_args()

def process_dataset(jsonl_path):
    output_filename = f"navigation_dataset.csv" 
    
    with open(jsonl_path, 'r') as json_file, open(output_filename, 'w') as csv_file:
        lines = [json.loads(line) for line in json_file]
        csv_writer = csv.DictWriter(csv_file, fieldnames=['id', 'propositions', 'nl_sentence', 'dataset_tl'])
        csv_writer.writeheader()
                
        for line in tqdm(lines, desc="Processing dataset entries", unit="entry"):
            id = line['id']
            nl_sentence = ' '.join(line['sentence'])
            logic_ltl = ''.join(line['logic_ltl'])
            logic_ltl = logic_ltl.replace("&", " & ").replace("|", " | ").replace("->", " -> ").replace("U", " U ") # add spaces around binary operators
            logic_ltl = re.sub(r"-(?!>)", "~", logic_ltl) # Change '-' negation operator to '~' to work with spot
            
            
            propositions = line['propositions']
            for prop in propositions:
                if(len(propositions[prop]['prop']) == 1 and len(propositions[prop]['prop'][0]) <= 2):
                    propositions[prop]['prop'] = '_'.join(strip_part_of_speech(word) for word in propositions[prop]['prop'][0]) # [['go_to_v', 'waste_basket_n']] => 'go_to_waste_basket'
                    propositions[prop].pop('span')
                    logic_ltl = logic_ltl.replace(prop, propositions[prop]['prop']) # F(prop_1) => F(go_to_waste_basket)
                else:
                    print("Error: more than one element")
                    breakpoint()
            propositions = [propositions[prop]['prop'] for prop in propositions]
            
            csv_writer.writerow({'id': id, 'propositions': propositions, 'nl_sentence': nl_sentence, 'dataset_tl': logic_ltl})
            
    return output_filename

def strip_part_of_speech(word):
    # 'go_to_v' => 'go_to'
    parts = word.rsplit('_', 1)
    if(parts[1] != 'v' and parts[1] != 'n'):
        print("Error: not a verb or noun")
        breakpoint()
    
    return parts[0].replace(' ','_').replace("'","")

def nl2tl_gemma(nl_sentence, propositions, few_shot_examples, grammar_constraint, grammar_prompt):
    prompt = get_prompt(propositions=propositions, task=nl_sentence, bnf_spec=get_llama_bnf_spec(propositions), few_shot=few_shot_examples, grammar_prompt=grammar_prompt)
    
    if grammar_constraint:
        tl_bnf_grammar = LlamaGrammar.from_string(get_llama_bnf_spec(propositions))
    else:
        tl_bnf_grammar = None
    
    system_prompt = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
    query = [{"role": "user", "content": [{"type": "text", "text": f"Natural Language Requirement - \"{nl_sentence}\"\nRelevant Propositions - {str(propositions)[1:-1]}"}]}]
    
    out = gemma_llm.create_chat_completion(
        messages = system_prompt + few_shot_examples + query,
        grammar=tl_bnf_grammar,
    )

    response = out['choices'][0]['message']['content']
    response = re.sub(r"-(?!>)", "_", response) # Change '-' in propositions to '_' to work with spot
    
    return response.strip()

def nl2tl_gpt4(nl_sentence, propositions, few_shot_examples, grammar_prompt):
    prompt = get_prompt(propositions=propositions, task=nl_sentence, bnf_spec=get_llama_bnf_spec(propositions), few_shot=few_shot_examples, grammar_prompt=grammar_prompt)
    
    system_prompt = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
    query = [{"role": "user", "content": [{"type": "text", "text": f"Natural Language Requirement - \"{nl_sentence}\"\nRelevant Propositions - {str(propositions)[1:-1]}"}]}]
    messages = system_prompt + few_shot_examples + query
        
    out = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages = messages,
        response_format={"type": "text"}
        )
    
    response = out.choices[0].message.content
    response = re.sub(r"-(?!>)", "_", response) # Change '-' in propositions to '_' to work with spot
    
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
        
        few_shot_examples = few_shot_examples + [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_prompt}]}                    
        ]
            
    return few_shot_examples

def safe_are_equivalent(true_ltl, llm_ltl):
    try:
        return spot.are_equivalent(true_ltl, llm_ltl)
    except Exception as e:
        try:
            spot.formula(true_ltl)
        except Exception as e:
            return "Invalid data entry"
        try:
            spot.formula(llm_ltl)
        except Exception as e:
            return "Invalid LLM formula"
        
        print([true_ltl, llm_ltl])
        return "ERROR"

if __name__ == "__main__":    
    args = parse_args()
    num_dataset_entries = args.entries
    num_trials = args.trials
    num_examples = args.examples
    
    if os.path.exists("navigation_dataset.csv"):
        dataset_filename = "navigation_dataset.csv"        
    else:
        dataset_filename = process_dataset(args.dataset_jsonl)
    full_dataset = pd.read_csv(dataset_filename)
    
    summary_df = pd.DataFrame()
    
    experiment_dir_name = f"trials_{num_dataset_entries}entries_{num_trials}trials_{num_examples}examples"
    os.makedirs(experiment_dir_name, exist_ok=True)
    
    ablations = [
        {
            'name': 'gemma_Pc',
            'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, grammar_constraint=False, grammar_prompt=True)
        },
        {
            'name': 'gemma_PC',
            'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, grammar_constraint=True, grammar_prompt=True)
        },
        {
            'name': 'gemma_pc',
            'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, grammar_constraint=False, grammar_prompt=False)
        },
        {
            'name': 'gemma_pC',
            'function': lambda row: nl2tl_gemma(row['nl_sentence'], eval(row['propositions']), few_shot_examples, grammar_constraint=True, grammar_prompt=False)
        },
        {
            'name': 'gpt4_Pc',
            'function': lambda row: nl2tl_gpt4(row['nl_sentence'], eval(row['propositions']), few_shot_examples, grammar_prompt=True)
        }
    ]

    for i in range(num_trials):
        print(f"Trial {i+1}:")
        few_shot_examples = get_few_shot_examples(full_dataset, num_examples=num_examples)
        summary_df_row_idx = len(summary_df)
        df = full_dataset.sample(n=num_dataset_entries).reset_index(drop=True)
        
        for ablation in ablations:
            tqdm.pandas(desc=ablation['name'])
            start_time = time.time()
            df[ablation['name'] + '_tl'] = df.progress_apply(ablation['function'], axis=1)
            total_time = time.time() - start_time
            
            df[ablation['name'] + '_equivalence'] = df.apply(lambda row: safe_are_equivalent(row['dataset_tl'], row[ablation['name'] + '_tl']), axis=1)
            equivalence_counts = df[ablation['name'] + '_equivalence'].value_counts()
            
            accuracy = equivalence_counts.get(True, 0) / (num_dataset_entries - equivalence_counts.get("Invalid data entry", 0))
            validity = 1 - equivalence_counts.get("Invalid LLM formula", 0) / (num_dataset_entries - equivalence_counts.get("Invalid data entry", 0))
            inference_time = total_time / num_dataset_entries
            
            summary_df.loc[summary_df_row_idx, [ablation['name'] + '_accuracy', ablation['name'] + '_validity', ablation['name'] + '_time']] = [accuracy, validity, inference_time]
            
        summary_df.index.name = "Trial"
        summary_df.to_csv(f"{experiment_dir_name}/experiment_summary.csv", index=True)
        df.to_csv(f"{experiment_dir_name}/trial_{i+1}.csv", index=False)

    summary_df.index.name = "Trial"
    summary_df.to_csv(f"{experiment_dir_name}/experiment_summary.csv", index=True)
    summary_df = summary_df.astype(float)
    with open(f"{experiment_dir_name}/summary_stats.txt", "w") as f:
        f.write(summary_df.describe().round(4).to_string())