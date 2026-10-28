"""The validation seal permits new rows but forbids changing training witnesses."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check
from qwen35_reference_evidence import extract
import evaluate_qwen35_copy_inventory as evaluation


def test_seal_preserves_train_while_allowing_validation_append(tmp_path,monkeypatch):
    rows=[{'id':0,'split':'train'},{'id':1,'split':'train'},{'id':2,'split':'validation'}]
    monkeypatch.setattr(evaluation,'gate',lambda a:rows)
    monkeypatch.setattr(evaluation,'proposal_gate',lambda out:None)
    (tmp_path/'protocol.json').write_text('{}\n')
    (tmp_path/'proposals-frozen.json').write_text('{}\n')
    (tmp_path/'proposals').mkdir();(tmp_path/'proposals/run.dl').write_text('// frozen proposal\n')
    verifier=tmp_path/'verifier.dl';verifier.write_text('.decl certified()\n.output certified\ncertified().\n')
    proposal=check(verifier,verifier,{}, {},evidence_dir=tmp_path/'certificate-evidence/propose')
    (tmp_path/'proposals.json').write_text(json.dumps(proposal))
    class Scores(list):
        def tolist(self):return list(self)
    observed=[]
    for i in (0,1):
        reference=extract(Scores([0.,2.,1.]),tmp_path/'reference-evidence'/f'reference-{i:04d}.tar.gz',i)
        observed.append({'id':i,'next':1,'reference':reference})
    def save(records):(tmp_path/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    save(observed);args=SimpleNamespace(out=tmp_path)
    evaluation.seal(args);evaluation.ready(args)
    save(observed+[{'id':2,'next':2}]);evaluation.ready(args)
    changed=[dict(observed[0],next=0),observed[1],{'id':2,'next':2}]
    save(changed)
    with pytest.raises(ValueError,match='training journal changed'):evaluation.ready(args)


def test_reference_audit_rejects_a_journal_answer_without_its_witness(tmp_path):
    class Scores(list):
        def tolist(self):return list(self)
    reference=extract(Scores([0.,2.,1.]),tmp_path/'reference-evidence/ref.tar.gz',4)
    with pytest.raises(ValueError,match='journal differs'):
        evaluation.audit_record(tmp_path,{'id':4,'next':0,'reference':reference})
