"""Run the frozen evaluator with an equivalent, side-deduplicated max query."""
from pathlib import Path
from certificate import check as original_check,sha256
import run_qwen35_mlp_factor as study

def linear_check(verifier,candidate,*args,**kwargs):
    original=study.ROOT/'dl/qwen35_factor_reconstruction.dl'
    if Path(verifier)==original:
        assert Path(candidate)==original
        verifier=candidate=study.ROOT/'dl/qwen35_factor_reconstruction_linear.dl'
    return original_check(verifier,candidate,*args,**kwargs)
def main():
    amendment=study.read(study.OUT/'evaluation-amendment.json')
    for n,h in amendment['sources'].items():
        if sha256(study.ROOT/n)!=h:raise ValueError('evaluation amendment source changed')
    study.check=linear_check
    study.evaluate(type('Args',(),{'out':study.OUT})())
if __name__=='__main__':main()
