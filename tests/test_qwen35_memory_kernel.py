from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_memory_mixtures_copy_majority_and_require_all_features(tmp_path):
    core=tmp_path/'core.dl';core.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\n'+(ROOT/'dl/copy_longest_runtime.dl').read_text())
    m=ROOT/'dl/qwen35_memory_kernel_typed.dl';q=list(range(100,124));source=q+[70,901];decoy=q[-12:]+[80,902];ctx=decoy+source+source+q
    f={'tok':''.join(f'0\t{p}\t{t}\n' for p,t in enumerate(ctx)),'sample':'0\tg\n','query':'1\tg\t1\n',
       'memory':'1\t70\t603\t2.0\n1\t70\t1765\t0.0\n1\t80\t603\t-2.0\n1\t80\t1765\t0.0\n',
       'coefficient':'603\t70\t1.0\n603\t80\t-1.0\n1765\t70\t0.0\n1765\t80\t0.0\n','ref':'0\t70\n'}
    x=check(m,core,f,{},evidence_dir=tmp_path/'mixture');assert x['certified']
    assert sorted(x['relations']['preference'])==[['0','0','70'],['1','0','70'],['2','0','70']]
    assert not x['relations']['error']
    f['memory']=f['memory'].replace('1\t80\t1765\t0.0\n','')
    assert not check(m,core,f,{},evidence_dir=tmp_path/'missing')['certified']
