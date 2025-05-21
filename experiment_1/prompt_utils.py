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
    f"""root ::= ltl-depth-2

ltl-depth-2 ::= atomic-formula | "~(" ltl-depth-1 ")" | ltl-depth-1 binary-op ltl-depth-1 | unary-op "(" ltl-depth-1 ")"

ltl-depth-1 ::= atomic-formula | "~(" atomic-formula ")" | atomic-formula binary-op atomic-formula | unary-op "(" atomic-formula ")"

predicate-name ::= {" | ".join(f'"{p}"' for p in propositions)}
atomic-formula ::= predicate-name

binary-op ::= " & " | " | " | " -> " | " U "
# '&' (and): both propositions must be true
# '|' (or): at least one predicate must be true
# '->' (implies): if first predicate is true, then second predicate must be true 
# 'U' (until): first predicate must be true at least until second predicate is true

unary-op ::= globally | eventually
globally ::= "G"
# Predicate must always be true at every timestep

eventually ::= "F"
# Predicate must be true at some time in the future
    """
    return bnf_spec