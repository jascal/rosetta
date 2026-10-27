"""Explicit numeric casts for the frozen memory-mixture formulas, no new choices."""
from pathlib import Path
from certificate import check as original_check,sha256
import run_qwen35_memory_kernel as study

def typed_check(verifier,candidate,*args,**kwargs):
    if Path(verifier)==study.ROOT/'dl/qwen35_memory_kernel_summary.dl':
        verifier=study.ROOT/'dl/qwen35_memory_kernel_typed.dl'
    return original_check(verifier,candidate,*args,**kwargs)
def main():
    amendment=study.read(study.OUT/'evaluation-amendment.json')
    for n,h in amendment['sources'].items():
        if sha256(study.ROOT/n)!=h:raise ValueError('amended evaluator changed')
    study.check=typed_check;study.evaluate(type('Args',(),{'out':study.OUT})())
if __name__=='__main__':main()
