def get_prompt(propositions, task, bnf_spec, grammar_prompt, few_shot):
    prompt_with_grammar = f"""
You are an AI assistant specializing in formal methods and temporal logic. Your task is to translate a natural language requirement into a Temporal Logic (TL) formula, strictly adhering to the provided grammar.

You MUST generate the formula according to the following BNF-like grammar: 
```ebnf
{bnf_spec}
```

Return only the STL formula, without any additional text or explanation. The STL formula MUST adhere to the BNF grammar provided above.
    """
    
# Natural Language Requirement - "{task}"
# Relevant Propositions - {str(propositions)[1:-1]}
# Temporal Logic Specification - 
    
    prompt_without_grammar = f"""
You are an AI assistant specializing in formal methods and temporal logic. Your task is to translate a natural language requirement into a Temporal Logic (TL) formula.


Return only the STL formula, without any additional text or explanation.
    """

    if grammar_prompt:
        return prompt_with_grammar
    else:
        return prompt_without_grammar

def get_llama_bnf_spec(propositions):
    
    propositions = [p.replace("_", "-") for p in propositions]
    
    bnf_spec = \
    f"""# The root rule defines the starting point of the grammar.
# The entire output must match this rule.
root ::= ws expr ws

# An "expr" is one or more "term"s joined by binary operators.
# This structure avoids left-recursion and handles operator chaining (e.g., p & q | r).
expr ::= term (ws binary-op ws term)*

# A "term" is the fundamental, non-divisible building block.
# It can be a simple proposition, a unary operation, a negation, or a parenthesized group.
# Allow unary operators to work with both single atoms and complex expressions.
term ::= atomic-formula | unary-op ws "(" ws expr ws ")" | unary-op ws atomic-formula | "~" ws term | "(" ws expr ws ")"

# --- Base Definitions ---

predicate-name ::= {" | ".join(f'"{p}"' for p in propositions)}
# Allow propositions with or without parentheses
atomic-formula ::= predicate-name | "(" ws predicate-name ws ")"

# Defines optional whitespace (zero or more spaces, tabs, or newlines).
# This makes the grammar flexible to the LLM's output formatting.
ws ::= [ \t\n]*

binary-op ::= "&" | "|" | "->" | "U"
# '&' (and): both propositions must be true
# '|' (or): at least one predicate must be true
# '->' (implies): if first predicate is true, then second predicate must be true
# 'U' (until): first predicate must be true at least until second predicate is true

unary-op ::= "G" | "F"
# G (globally): Predicate must always be true at every timestep
# F (eventually): Predicate must be true at some time in the future"""
    return bnf_spec
