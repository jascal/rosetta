"""Serialize Datalog-selected parameters; no model observations or decisions at runtime."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'reference/benchmarks/qwen25_native_trace_b/artifacts/behavior/circuits.dl'


def body(option):
    c,k,anchor=map(int,option)
    return ('.decl ls_option(c:number,minmatch:number,anchor:number)\n'+
        f'ls_option({c},{k},{anchor}).\n'+
        (ROOT/'dl/copy_longest_runtime.dl').read_text()+
        (ROOT/'dl/copy_longest_anchor.dl').read_text())


def candidate(option):
    return body(option)+'\n.decl cdecide(i:number,t:number)\ncdecide(I,T) :- ls_prediction(_,I,T).\n'


def composite(option):
    return (BASELINE.read_text().replace('cdecide','bdecide')+'\n'+body(option)+
        '\n.decl cdecide(i:number,t:number)\ncdecide(I,T) :- bdecide(I,T).\n'
        'cdecide(I,T) :- ls_prediction(_,I,T), !bdecide(I,_).\n')


def emit(directory,text):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'circuits.dl').write_text(text)
    (directory/'run.dl').write_text('.decl tok(i:number,p:number,t:number)\n.input tok\n'
        '.decl domain(i:number)\n.input domain\n#include "circuits.dl"\n.output cdecide\n'
        '.decl abstain(i:number)\n.output abstain\nabstain(I) :- domain(I), !cdecide(I,_).\n')
    return directory/'circuits.dl'
