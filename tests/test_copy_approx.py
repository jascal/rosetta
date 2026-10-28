"""Bounded-Hamming computation controls, independent of the LLM."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'py'))
from certificate import check

pytestmark = pytest.mark.skipif(shutil.which('souffle') is None, reason='needs souffle')


def test_approximation_handles_one_error_but_rejects_two_and_conflicts(tmp_path):
    seq = list(range(10,34))
    query = seq[:12]
    query[-1] = 100
    contexts = [seq+seq+query]
    two = query[:]
    two[-2] = 101
    contexts.append(seq+seq+two)
    conflict = seq+seq+query
    conflict[12] = 200
    contexts.append(conflict)
    contexts.append(seq+query)  # only one agreeing source
    contexts.append(seq+seq+seq[:12])  # exact matches are not this circuit's domain
    circuit = tmp_path/'circuit.dl'
    circuit.write_text('.decl ap_option(c:number,k:number,min_sources:number)\nap_option(1,12,2).\n'+
                       (ROOT/'dl/copy_approx_runtime.dl').read_text()+
                       '\n.decl cdecide(i:number,t:number)\ncdecide(I,T) :- ap_prediction(_,I,T).\n')
    verifier = tmp_path/'verify.dl'
    verifier.write_text('.decl tok(i:number,p:number,t:number)\n.input tok\n#include "circuit.dl"\n'
                        '.output cdecide\n.decl certified()\n.output certified\ncertified().\n')
    facts = {'tok':''.join(f'{i}\t{p}\t{t}\n' for i,ctx in enumerate(contexts) for p,t in enumerate(ctx))}
    x = check(verifier,circuit,facts,{})
    assert x['certified'],x.get('error')
    assert x['relations']['cdecide'] == [['0','22']]
