from pathlib import Path
import math,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_qk_softmax_reconstruction_and_source_only_key_substitution(tmp_path):
    m=ROOT/'dl/qwen35_attention_qk_payload.dl';core=tmp_path/'core.dl'
    core.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\n'+(ROOT/'dl/copy_longest_runtime.dl').read_text())
    q=list(range(100,124));rows=[q+[70,900]+q[-12:]+[d]+q for d in (80,81)];size=len(rows[0])
    weights={};vectors={}
    for i in (0,1):
        scores=[(1+i)*(1.0 if p==24 else (1.5 if i==0 else -1.0) if p==38 else 0.0) for p in range(size)]
        z=sum(math.exp(s) for s in scores);weights[i]=[math.exp(s)/z for s in scores]
        vectors[i]=.5*(2*weights[i][24]-weights[i][38])
    f={'shape':'1\n','context':f'0\t0\told\t{size-1}\n1\t0\tnew\t{size-1}\n',
       'tok':''.join(f'{i}\t{p}\t{t}\n' for i,row in enumerate(rows) for p,t in enumerate(row)),
       'qvec':'0\t0\t1.0\n1\t0\t2.0\n','scale':'0\t1.0\n1\t1.0\n',
       'kvec':''.join(f"{i}\t{p}\t0\t{1.0 if p==24 else (1.5 if i==0 else -1.0) if p==38 else 0.0}\n" for i in (0,1) for p in range(size)),
       'mask':''.join(f'{i}\t{p}\t0.0\n' for i in (0,1) for p in range(size)),
       'weight':''.join(f'{i}\t{p}\t{v:.17g}\n' for i in (0,1) for p,v in enumerate(weights[i])),
       'value':''.join(f'{i}\t{p}\t0\t{2.0 if p==24 else -1.0 if p==38 else 0.0}\n' for i in (0,1) for p in range(size)),
       'gate':'0\t0\t0.0\n1\t0\t0.0\n',
       'control_vector':''.join(f'{i}\t{k}\t0\t{vectors[i if k=="self" else 1-i]:.17g}\n' for i in (0,1) for k in ('self','donor'))}
    x=check(m,core,f,{},evidence_dir=tmp_path/'payload');assert x['certified'];assert len(x['relations']['run'])==12
    p={(int(a),int(j)):float(v) for a,j,v in x['relations']['probability']}
    for j in range(size):
        assert math.isclose(p[3,j],p[4,j],rel_tol=1e-12)
        assert math.isclose(p[4,j],p[5,j],rel_tol=1e-12)
    f['qvec']='0\t0\t1.0\n'
    assert not check(m,core,f,{},evidence_dir=tmp_path/'missing')['certified']
