from tqdm import tqdm
import pandas as pd  # only needed if you later load it
import json
import csv
import argparse
import re

argparser = argparse.ArgumentParser()
argparser.add_argument('--jsonl_path', type=str, required=True, help='Path to the input JSONL file')
argparser.add_argument('--output_filename', type=str, required=True, help='Path to the output CSV file')
args = argparser.parse_args()
jsonl_path = args.jsonl_path
output_filename = args.output_filename


def strip_part_of_speech(word):
    parts = word.rsplit('_', 1)
    if (parts[1] not in ('v', 'n')):
        print("Error: not a verb or noun")
        breakpoint()
    return parts[0].replace(' ', '_').replace("'", "")


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

    

