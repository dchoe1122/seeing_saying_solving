import spot
import argparse
import pandas as pd
import subprocess
import os
import tempfile
from prompt_utils import get_llama_bnf_spec
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer

def parse_args():
    argparser = argparse.ArgumentParser(description="Failure analysis for NL to TL translation")
    argparser.add_argument("--experiment_csv", type=str, required=True, help="Path to CSV containing experiment trial results")
    return argparser.parse_args()

def validate_with_gbnf(tl_formula, propositions):
    """
    Validate a TL formula using the GBNF validator.
    Returns True if valid, False if invalid.
    """
    # Get the grammar
    grammar_content = get_llama_bnf_spec(propositions=propositions)
    
    # Transform the formula to match grammar expectations (replace _ with -)
    transformed_formula = tl_formula.replace('_', '-')
    
    # Create temporary files for grammar and input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.gbnf', delete=False) as grammar_file:
        grammar_file.write(grammar_content)
        grammar_path = grammar_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
        input_file.write(transformed_formula)
        input_path = input_file.name
    
    try:
        # Run the GBNF validator
        script_dir = os.path.dirname(os.path.abspath(__file__))
        validator_path = os.path.join(script_dir, 'bin', 'test-gbnf-validator')
        
        result = subprocess.run(
            [validator_path, grammar_path, input_path],
            capture_output=True,
            text=True
        )
        
        print(f"gbnfm out: {result.stdout}")

        # Check if the validation was successful
        is_valid = "Input string is valid according to the grammar." in result.stdout
        return is_valid
        
    except Exception as e:
        print(f"Error running GBNF validator: {e}")
        return False
    
    finally:
        # Clean up temporary files
        os.unlink(grammar_path)
        os.unlink(input_path)

def validate_with_lark(tl_formula, propositions):
    """
    Validate a TL formula using the Lark parser.
    Returns True if valid, False if invalid.
    """
    try:
        grammar = parse_ebnf(get_llama_bnf_spec(propositions=propositions))
        recognizer = StringRecognizer(grammar.grammar_encoding, grammar.symbol_table["root"])
        is_valid = recognizer._accept_prefix(tl_formula.replace('_','-'))
        return is_valid
    except Exception as e:
        print(f"Error during Lark validation: {e}")
        return False

def analyze_constrained_loss(exp_df):
    # This function is meant to analyze the loss in accuracy due to constrained generation
    gap_entries = exp_df[((exp_df['gemma_Pc_equivalence'] == "True") | (exp_df['gemma_Pc_equivalence'] == True)) & ((exp_df['gemma_PC_equivalence'] == "False") | (exp_df['gemma_PC_equivalence'] == False))]
    gap_entries = gap_entries[['nl_sentence', 'propositions', 'dataset_tl', 'gemma_Pc_tl', 'gemma_PC_tl']]
    print(f"Number of entries where constrained generation failed but unconstrained succeeded: {len(gap_entries)}")

    # Check if the unconstrained translations still meet the EBNF grammar using both Lark and GBNF validator
    for index, row in gap_entries.iterrows():
        propositions = eval(row['propositions'])
        unconstrained_tl = row['gemma_Pc_tl']
        constrained_tl = row['gemma_PC_tl']

        # Test with Lark (original method)
        try:
            grammar = parse_ebnf(get_llama_bnf_spec(propositions=propositions))
            recognizer = StringRecognizer(grammar.grammar_encoding, grammar.symbol_table["root"])
            lark_valid = recognizer._accept_prefix(unconstrained_tl.replace('_','-'))
        except Exception as e:
            print(f"Row {index}: Lark validation failed with error: {e}")
            lark_valid = False

        # Test with GBNF validator
        gbnf_valid = validate_with_gbnf(unconstrained_tl, propositions)

        validation_status = f"Lark: {'VALID' if lark_valid else 'INVALID'}, GBNF: {'VALID' if gbnf_valid else 'INVALID'}"

        print(f"Row {index}: Unconstrained translation validation ({validation_status})")
        print(f"  Constrained TL: {constrained_tl}")
        print(f"  Unconstrained TL: {unconstrained_tl}")

        if lark_valid != gbnf_valid:
            print(f"  WARNING: Validation methods disagree!")
        print()

def extract_failures_to_csv(exp_df, output_file="failures.csv"):
    """
    Extract failure cases and output them in the same format as navigation_dataset.csv
    """
    # Find all rows where at least one model failed (equivalence is False)
    equivalence_cols = [col for col in exp_df.columns if col.endswith('_equivalence')]

    # Create a mask for rows where any equivalence check failed
    failure_mask = exp_df[equivalence_cols].apply(lambda row: any(val == False or val == "False" for val in row), axis=1)

    failure_rows = exp_df[failure_mask]

    # Extract only the columns that match navigation_dataset.csv format
    output_data = failure_rows[['id', 'propositions', 'nl_sentence', 'dataset_tl']].copy()

    # Save to CSV with same format as navigation_dataset.csv
    output_data.to_csv(output_file, index=False)

    print(f"Extracted {len(output_data)} failure cases to {output_file}")
    return output_data

def generate_containment_stats(exp_df):
    containment_results = []
    ablations = [col[:-3] for col in exp_df.columns if col.endswith('_tl')]
    ablations.pop(ablations.index('dataset'))  # remove the dataset column

    for ablation in ablations:
        
        # Filter to rows where equivalence is not Invalid LLM formula or Invalid data entry for all ablations
        valid_equivalence_mask = exp_df[[f'{ablation}_equivalence'] + [f'{other}_equivalence' for other in ablations if other != ablation]].apply(
            lambda row: all(val not in ["Invalid LLM formula", "Invalid data entry"] for val in row), axis=1)
        parsable_rows = exp_df[valid_equivalence_mask]
        
        num_parsable = len(parsable_rows)

        num_llm_in_true = parsable_rows.apply(lambda row: spot.contains(spot.formula(row['dataset_tl']), spot.formula(row[f'{ablation}_tl'])), axis=1).sum()
        num_true_in_llm = parsable_rows.apply(lambda row: spot.contains(spot.formula(row[f'{ablation}_tl']), spot.formula(row['dataset_tl'])), axis=1).sum()

        containment_results.append({
            'ablation': ablation,
            'LLM_in_true': num_llm_in_true / num_parsable if num_parsable > 0 else 0,
            'true_in_LLM': num_true_in_llm / num_parsable if num_parsable > 0 else 0
        })

    #summarize containment results
    containment_df = pd.DataFrame(containment_results)
    containment_df.set_index('ablation', inplace=True)

    return containment_df

def generate_grammar_validity_stats(exp_df):
    validity_results = []
    ablations = [col[:-3] for col in exp_df.columns if col.endswith('_tl')]
    ablations.pop(ablations.index('dataset'))  # remove the dataset column

    for ablation in ablations:
        total_rows = len(exp_df)
        valid_count = exp_df.apply(lambda row: validate_with_lark(row[f'{ablation}_tl'], eval(row['propositions'])), axis=1).sum()

        validity_results.append({
            'ablation': ablation,
            'validity': valid_count / total_rows if total_rows > 0 else 0
        })

    validity_df = pd.DataFrame(validity_results)
    validity_df.set_index('ablation', inplace=True)

    return validity_df

if __name__ == "__main__":
    args = parse_args()
    experiment_csv = args.experiment_csv

    exp_df = pd.read_csv(experiment_csv)

    # Extract failures to CSV file
    extract_failures_to_csv(exp_df)

    analyze_constrained_loss(exp_df)
    containment_df = generate_containment_stats(exp_df)
    print(containment_df)

    validity_df = generate_grammar_validity_stats(exp_df)
    print(validity_df)
