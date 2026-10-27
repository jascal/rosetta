from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_profile_ranks_source_shift_and_uses_successor_positions(tmp_path):
    core=tmp_path/'core.dl';core.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\n'+(ROOT/'dl/copy_longest_runtime.dl').read_text())
    q=list(range(100,124));rows=[q+[70,900]+q[-12:]+[d]+q for d in (80,81)]
    f={'tok':''.join(f'{i}\t{p}\t{t}\n' for i,row in enumerate(rows) for p,t in enumerate(row)),
       'context':'0\t0\told\n1\t0\tnew\n','head':'3\t0\n3\t1\n7\t0\n',
       'attention':'0\t3\t0\t24\t0.1\n0\t3\t0\t38\t0.8\n1\t3\t0\t24\t0.8\n1\t3\t0\t38\t0.1\n0\t3\t1\t24\t0.2\n0\t3\t1\t38\t0.2\n1\t3\t1\t24\t0.3\n1\t3\t1\t38\t0.2\n0\t7\t0\t24\t0.3\n0\t7\t0\t38\t0.2\n1\t7\t0\t24\t0.1\n1\t7\t0\t38\t0.2\n'}
    m=ROOT/'dl/qwen35_attention_profile.dl';x=check(m,core,f,{},evidence_dir=tmp_path/'profile');assert x['certified']
    assert sorted(x['relations']['rank'])==[['3','0','0'],['3','1','1'],['7','0','2']]
def test_profile_plan_enumerates_all_full_heads(tmp_path):
    m=ROOT/'dl/qwen35_attention_profile_plan.dl'
    x=check(m,m,{'heads':'16\n','full_layer':'3\n7\n11\n15\n19\n23\n27\n31\n'},{},evidence_dir=tmp_path/'plan')
    assert x['certified'];assert len(x['relations']['head'])==128
