from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_windows_keep_absolute_positions_and_use_both_fixed_fillers(tmp_path):
    q=list(range(100,124));ctx=q+[70,900]+q[-12:]+[80]+q
    m=ROOT/'dl/qwen35_key_window_plan.dl';core=tmp_path/'core.dl';core.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\n'+(ROOT/'dl/copy_longest_runtime.dl').read_text())
    x=check(m,core,{'tok':''.join(f'0\t{p}\t{t}\n' for p,t in enumerate(ctx)),'context':f'0\t0\told\t{len(ctx)-1}\n'},{},evidence_dir=tmp_path/'plan');assert x['certified']
    assert x['relations']['source_position']==[['0','38']];assert len(x['relations']['probe'])==5
    rows={(int(a),int(p)):int(t) for a,p,t in x['relations']['probe_tok']}
    for mode in range(5):assert rows[mode,38]==80
    assert rows[0,0]==100 and rows[1,0]==0 and rows[2,0]==1
    for p in range(35,39):assert rows[1,p]==rows[2,p]==ctx[p]
