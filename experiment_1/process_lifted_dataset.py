from tqdm import tqdm
import json
import csv
import argparse
import re
import random

argparser = argparse.ArgumentParser()
argparser.add_argument('--jsonl_path', type=str, required=True, help='Path to the input JSONL file')
argparser.add_argument('--output_filename', type=str, required=True, help='Path to the output CSV file')
argparser.add_argument('--train_test_split', action='store_true', help='Create separate train and test CSV files')
argparser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of data for test set (default: 0.2)')
argparser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits (default: 42)')
args = argparser.parse_args()
jsonl_path = args.jsonl_path
output_filename = args.output_filename

# Set random seed for reproducible splits
if args.train_test_split:
    random.seed(args.seed)

with open(jsonl_path, 'r') as json_file:
    lines = [json.loads(line) for line in json_file]

# Process all data first
processed_data = []
for i, line in enumerate(tqdm(lines, desc="Processing dataset entries", unit="entry")):
    id = i  # Use line number as ID since lifted_data_no_time.jsonl doesn't have an 'id' field
    nl_sentence = ' '.join(line['logic_sentence'])
    logic_ltl = ''.join(line['logic_ltl'])
    logic_ltl = logic_ltl.replace("&", " & ").replace("|", " | ").replace("->", " -> ").replace("U", " U ")
    # Add spaces around G and F if not followed by (
    logic_ltl = re.sub(r'(?<!\w)(G|F)(?!\s*\()', r' \1 ', logic_ltl)

    # Replace two or more spaces with a single space
    logic_ltl = re.sub(r'\s{2,}', ' ', logic_ltl).strip()

    # Replace '-' with '~' unless it's part of '->'
    logic_ltl = re.sub(r"-(?!>)", "~", logic_ltl)

    # Extract propositions from logic_ltl (prop_1, prop_2, etc.)
    propositions = list(set(re.findall(r'prop_\d+', logic_ltl)))
    propositions.sort()  # Sort for consistency

    processed_data.append({'id': id, 'propositions': propositions, 'nl_sentence': nl_sentence, 'dataset_tl': logic_ltl})

if args.train_test_split:
    # Shuffle and split the data
    random.shuffle(processed_data)
    split_idx = int(len(processed_data) * (1 - args.test_ratio))
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]

    # Create train and test filenames
    base_name = output_filename.rsplit('.', 1)[0]
    extension = output_filename.rsplit('.', 1)[1] if '.' in output_filename else 'csv'
    train_filename = f"{base_name}_train.{extension}"
    test_filename = f"{base_name}_test.{extension}"

    # Write train file
    with open(train_filename, 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=['id', 'propositions', 'nl_sentence', 'dataset_tl'])
        csv_writer.writeheader()
        for row in train_data:
            csv_writer.writerow(row)

    # Write test file
    with open(test_filename, 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=['id', 'propositions', 'nl_sentence', 'dataset_tl'])
        csv_writer.writeheader()
        for row in test_data:
            csv_writer.writerow(row)

    print(f"Created train file: {train_filename} ({len(train_data)} entries)")
    print(f"Created test file: {test_filename} ({len(test_data)} entries)")
else:
    # Write single output file
    with open(output_filename, 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=['id', 'propositions', 'nl_sentence', 'dataset_tl'])
        csv_writer.writeheader()
        for row in processed_data:
            csv_writer.writerow(row)