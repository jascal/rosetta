from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check
from qwen35_key_fixtures import rows,facts_for

def core(tmp_path):
    p=tmp_path/'core.dl';p.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\n'+(ROOT/'dl/copy_longest_runtime.dl').read_text());return p

@pytest.mark.parametrize('kind',['empty','zero_short','two_short','two_long','one'])
def test_exact_source_count_certificate_and_short_patch_guard(tmp_path,kind):
    seq=[] if kind=='empty' else rows(kind);c=core(tmp_path)
    m=ROOT/'dl/qwen35_attention_qk_source_counts.dl'
    f={'context':'' if kind=='empty' else '0\n','tok':''.join(f'0\t{p}\t{t}\n' for p,t in enumerate(seq))}
    x=check(m,c,f,{},evidence_dir=tmp_path/'counts');assert x['certified']==(kind=='one')
    m=ROOT/'dl/qwen35_attention_qk_payload.dl';f=facts_for(seq)
    if kind=='empty':f['context']=''
    x=check(m,c,f,{},evidence_dir=tmp_path/'payload');assert x['certified']==(kind=='one')
    if kind!='one':assert not [r for r in x.get('relations',{}).get('patch',[]) if int(r[0]) in (5,11)]
    else:assert ['0','long','1'] in check(ROOT/'dl/qwen35_attention_qk_source_counts.dl',c,{'context':'0\n','tok':f['tok']},{},evidence_dir=tmp_path/'valid-counts')['relations']['counts']
