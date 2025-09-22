from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from prompt_utils import get_llama_bnf_spec

propositions = ['drop_orange','pick_up_fruit']
input_text = get_llama_bnf_spec(propositions)
parsed_grammar = parse_ebnf(input_text)

start_rule_id = parsed_grammar.symbol_table["root"]
recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

# Test the grammar with a simple input.
json_input = 'G(~(drop-orange)) U pick-up-fruit'
is_accepted = recognizer._accept_prefix(json_input)
print(is_accepted)
