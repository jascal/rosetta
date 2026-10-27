from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_bisection_waits_for_both_children_then_chooses_left_success(tmp_path):
    m=ROOT/'dl/qwen35_gate_bisection.dl'
    f={'caseinfo':'0\t7\t20\n','member':''.join(f'0\t{r}\t{10+r}\t1.0\n' for r in range(4)),'ref':'0\t0\t20\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'root');assert x['certified'];assert sorted(x['relations']['pending'])==[['0','1'],['0','2']]
    f['ref']+='0\t1\t20\n'
    x=check(m,m,f,{},evidence_dir=tmp_path/'partial');assert x['relations']['pending']==[['0','2']]
    f['ref']+='0\t2\t20\n'
    x=check(m,m,f,{},evidence_dir=tmp_path/'both');assert sorted(x['relations']['pending'])==[['0','3'],['0','4']]
    f['ref']+='0\t3\t10\n0\t4\t10\n'
    x=check(m,m,f,{},evidence_dir=tmp_path/'stop');assert x['relations']['interaction_stop']==[['0','1']];assert not x['relations']['pending']
def test_bisection_reports_single_neuron_and_failed_root(tmp_path):
    m=ROOT/'dl/qwen35_gate_bisection.dl'
    f={'caseinfo':'0\t7\t20\n1\t7\t20\n','member':'0\t0\t162\t1.0\n1\t0\t381\t2.0\n','ref':'0\t0\t20\n1\t0\t10\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'leaf');assert x['certified'];assert x['relations']['leaf']==[['0','0','162']];assert x['relations']['root_failed']==[['1']]
