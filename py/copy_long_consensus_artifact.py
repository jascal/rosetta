"""Serialize Datalog-selected consensus parameters into a standalone program."""
from pathlib import Path
from copy_longest_artifact import emit

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'reference/benchmarks/qwen25_native_trace_b/artifacts/behavior/circuits.dl'


def body(option):
    c,k,s=map(int,option)
    return ('.decl lg_option(c:number,minmatch:number,support:number)\n'+
        f'lg_option({c},{k},{s}).\n'+
        '.decl lc_option(c:number,minmatch:number,margin:number)\n'+
        'lc_option(C,M,4) :- lg_option(C,M,_).\n'+
        (ROOT/'dl/copy_longest_runtime.dl').read_text()+
        (ROOT/'dl/copy_long_consensus_runtime.dl').read_text())


def candidate(option):
    return body(option)+'\n.decl cdecide(i:number,t:number)\ncdecide(I,T) :- lg_prediction(_,I,T).\n'


def composite(option):
    return (BASELINE.read_text().replace('cdecide','bdecide')+'\n'+body(option)+
        '\n.decl cdecide(i:number,t:number)\ncdecide(I,T) :- bdecide(I,T).\n'
        'cdecide(I,T) :- lg_prediction(_,I,T), !bdecide(I,_).\n')
