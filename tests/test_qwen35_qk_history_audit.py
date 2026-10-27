"""A historical replay recognizes the pinned repair, never arbitrary edits."""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from audit_qwen35_attention_qk import checked_history
from certificate import sha256

def test_history_accepts_only_the_pinned_source_repair(tmp_path):
    root=tmp_path;out=root/'old';repair=root/'repair';window=root/'window'
    for p in (out,repair,window,root/'dl'):p.mkdir()
    def write(p,x):p.write_text(json.dumps(x))
    source=root/'dl/qwen35_attention_qk_payload.dl';source.write_text('repaired')
    other=root/'other.py';other.write_text('original')
    write(out/'closed.json',{'files':{}})
    write(out/'payload-frozen.json',{'files':{}})
    write(out/'protocol.json',{'files':{},'sources':{'dl/qwen35_attention_qk_payload.dl':'old','other.py':sha256(other)},'priors':{}})
    receipt=repair/'repair-receipt.json'
    write(receipt,{'prior_closed_sha256':sha256(out/'closed.json'),'repaired_sources':{'dl/qwen35_attention_qk_payload.dl':sha256(source)}})
    write(window/'protocol.json',{'priors':{'repair/repair-receipt.json':sha256(receipt)}})
    checked_history(root,out,repair,window)
    other.write_text('unexpected edit')
    with pytest.raises(ValueError,match='unregistered'):checked_history(root,out,repair,window)
    other.write_text('original');source.write_text('another repair')
    with pytest.raises(ValueError,match='repaired source'):checked_history(root,out,repair,window)
    source.write_text('repaired');receipt.write_text('{}')
    with pytest.raises(ValueError,match='receipt changed'):checked_history(root,out,repair,window)
