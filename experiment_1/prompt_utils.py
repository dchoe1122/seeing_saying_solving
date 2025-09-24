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

    bnf_spec = f"""root ::= ws expr ws

# The main expression rule.
expr ::= term (spaced-binary-op term)*

# REVISED: The term rule is now simpler and delegates the prefix logic.
term ::= core-term | "~" ws term

# REVISED: The core-term now explicitly lists the two alternatives,
# one with the unary operator and one without. This resolves the parsing error.
core-term ::= unary-op ws atomic-formula | atomic-formula | "(" ws expr ws ")"

# --- Base Definitions ---
predicate-name ::= {" | ".join(f'"{p}"' for p in propositions)}
atomic-formula ::= predicate-name | "(" ws predicate-name ws ")"
ws ::= [ \t\n]*
spaced-binary-op ::= ws ("&" | "|" | "->" | "U") ws
unary-op ::= "G" | "F"
"""

#     bnf_spec = \
#     f"""# The root rule defines the starting point of the grammar.
# # The entire output must match this rule.
# root ::= ws expr ws

# # MODIFIED: Removed ws from around the operator.
# expr ::= term (spaced-binary-op term)*

# # A "term" is the fundamental, non-divisible building block.
# term ::= atomic-formula | unary-op ws "(" ws expr ws ")" | unary-op ws atomic-formula | "~" ws term | "(" ws expr ws ")"

# # --- Base Definitions ---
# predicate-name ::= {" | ".join(f'"{p}"' for p in propositions)}
# atomic-formula ::= predicate-name | "(" ws predicate-name ws ")"
# ws ::= [ \t\n]*

# # OLD binary-op
# # binary-op ::= "&" | "|" | "->" | "U"

# # NEW "token-aware" rule for operators
# # This forces the model to choose the operator and its spacing as a single logical unit.
# spaced-binary-op ::= ws ("&" | "|" | "->" | "U") ws

# unary-op ::= "G" | "F"
# # G (globally): Predicate must always be true at every timestep
# # F (eventually): Predicate must be true at some time in the future"""
    return bnf_spec
