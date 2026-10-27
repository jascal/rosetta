from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
from check_qwen35_recent_consensus import artifact
ROOT=Path(__file__).resolve().parents[1]
def test_recent_consensus_requires_two_sources_and_recent_agreement(tmp_path):
    query=list(range(100,132));source=query+[70]+list(range(200,231));decoy=query[-12:]+[80,900]
    rows=[decoy+source+source+query,source+source+decoy+query,decoy+source+query]
    circuit=tmp_path/'circuit.dl';circuit.write_text(artifact())
    x=check(ROOT/'dl/qwen35_baseline_domain.dl',circuit,{'domain':'0\n1\n2\n','tok':''.join(f'{i}\t{p}\t{t}\n' for i,row in enumerate(rows) for p,t in enumerate(row))},{},evidence_dir=tmp_path/'guard')
    assert x['certified'];assert x['relations']['firing']==[['0','70']]
    assert sorted(x['relations']['abstain'])==[['1'],['2']]
