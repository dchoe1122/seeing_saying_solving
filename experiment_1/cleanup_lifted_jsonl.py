import json
import os

def filter_and_replace_operators(input_file, output_file):
    replacements = {
        'globally': 'G',
        'finally': 'F',
        'or': '|',
        'and': '&',
        'imply': '->',
        'until': 'U',
        'negation': '~'
    }

    lines_kept = 0
    lines_removed = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            logic_ltl = data.get('logic_ltl', [])

            # Check if any token contains "["
            has_bracket = any('[' in str(token) for token in logic_ltl)

            if not has_bracket:
                # Replace operators in logic_ltl
                for i, token in enumerate(logic_ltl):
                    if token in replacements:
                        logic_ltl[i] = replacements[token]

                data['logic_ltl'] = logic_ltl
                outfile.write(json.dumps(data) + '\n')
                lines_kept += 1
            else:
                lines_removed += 1

    return lines_kept, lines_removed

if __name__ == "__main__":
    input_file = "lifted_data_no_time.jsonl"
    temp_file = "lifted_data_no_time_temp.jsonl"

    lines_kept, lines_removed = filter_and_replace_operators(input_file, temp_file)

    # Replace original file with updated version
    os.replace(temp_file, input_file)

    print(f"Lines kept: {lines_kept}")
    print(f"Lines removed: {lines_removed}")
    print(f"Removed lines containing '\"[' in logic_ltl field")
    print(f"Operator replacements complete:")
    print("  'globally' -> 'G'")
    print("  'finally' -> 'F'")
    print("  'or' -> '|'")
    print("  'and' -> '&'")
    print("  'imply' -> '->'")
    print("  'until' -> 'U'")
    print("  'negation' -> '~'")