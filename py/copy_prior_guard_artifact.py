"""Serialize a fixed prior-guard option with its discrete lexical facts."""
from run_copy_prior_guard import rule_body
from copy_longest_artifact import emit


def candidate(option,lexical):
    return (rule_body(options=(tuple(map(int,option)),))+lexical+
        '\n.decl cdecide(i:number,t:number)\ncdecide(I,T) :- pg_prediction(_,I,T).\n')
