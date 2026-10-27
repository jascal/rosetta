from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_plan_freezes_three_histories_per_group(tmp_path):
    m=ROOT/'dl/qwen35_query_prior_plan.dl'
    x=check(m,m,{'sample':'0\tg\t24\n1\tg\t24\n50\th\t32\n'},{},evidence_dir=tmp_path/'plan')
    assert x['certified'];assert len(x['relations']['query'])==6
    assert ['2','0','24','2'] in x['relations']['query']
    assert ['151','50','4','1'] in x['relations']['query']
def test_history_prior_diagnostic_counts_errors_and_abstains_on_ties(tmp_path):
    m=ROOT/'dl/qwen35_query_prior_summary.dl';core=tmp_path/'core.dl'
    core.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\n'+(ROOT/'dl/copy_longest_runtime.dl').read_text())
    query=list(range(1,25));ctx=query+[70,91,92]+query[-12:]+[80,93]+query
    f={'tok':''.join(f'0\t{p}\t{t}\n' for p,t in enumerate(ctx)),'sample':'0\tg\n','query':'0\tg\t0\n1\tg\t1\n2\tg\t2\n',
       'prior':'0\t70\t1.0\n0\t80\t2.0\n1\t70\t2.0\n1\t80\t2.0\n2\t70\t3.0\n2\t80\t2.0\n','ref':'0\t70\n'}
    x=check(m,core,f,{},evidence_dir=tmp_path/'summary');assert x['certified']
    assert ['2','0','70'] in x['relations']['guarded'];assert not x['relations']['error']
    assert ['0','1','1'] in x['relations']['priorcounts'];assert ['1','0','0'] in x['relations']['counts']
    f['prior']=f['prior'].replace('2\t80\t2.0\n','')
    assert not check(m,core,f,{},evidence_dir=tmp_path/'missing')['certified']

def test_recount_counts_instances_within_each_query_group(tmp_path):
    m=ROOT/'dl/qwen35_query_prior_counts.dl'
    f={'query':'0\tg\t0\n','preference':'0\t10\t70\n0\t11\t80\n','guarded':'0\t10\t70\n0\t11\t80\n',
       'error':'0\t11\t80\t70\n','disagreement':'0\t11\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'count');assert x['certified']
    assert x['relations']['counts']==[['0','2','1']]
    assert x['relations']['priorcounts']==[['0','2','1']]
