from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_attention_plan_and_identity_are_complete(tmp_path):
    m=ROOT/'dl/qwen35_attention_plan.dl'
    x=check(m,m,{'shape':'32\t16\t2\t4\n','full_layer':'3\n31\n'},{},evidence_dir=tmp_path/'plan');assert x['certified']
    assert len(x['relations']['identity_site'])==4;assert len(x['relations']['zero'])==8
    f={'identity_site':''.join('\t'.join(r)+'\n' for r in x['relations']['identity_site']),
       'vector':''.join(f'{l}\t{p}\t{h}\t{d}\t1.0\n' for l,p,h,d in x['relations']['identity_site'])}
    m=ROOT/'dl/qwen35_attention_identity.dl';assert check(m,m,f,{},evidence_dir=tmp_path/'identity')['certified']
    f['vector']=''
    assert not check(m,m,f,{},evidence_dir=tmp_path/'missing')['certified']
def test_site_audit_checks_identity_hash_row_mass_and_nonzero_zero_dose(tmp_path):
    m=ROOT/'dl/qwen35_attention_sites.dl'
    f={'shape':'1\t1\n','arm':'baseline\nidentity\nzero_last\n','observe':'baseline\t3\t0\nidentity\t3\t0\n',
       'vector':'baseline\t3\t1\t0\t0\t1.0\nidentity\t3\t1\t0\t0\t1.0\n',
       'attention':'baseline\t3\t1\t0\t0\t0.25\nbaseline\t3\t1\t0\t1\t0.75\nidentity\t3\t1\t0\t0\t0.25\nidentity\t3\t1\t0\t1\t0.75\n',
       'request':'zero_last\t3\t1\t0\t0\t0.0\n','applied':'zero_last\t3\t1\t0\t0\t1.0\t0.0\n',
       'hash':'baseline\ta\nidentity\ta\nzero_last\tb\n'}
    assert check(m,m,f,{},evidence_dir=tmp_path/'ok')['certified']
    f['hash']=f['hash'].replace('identity\ta','identity\tc')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'bad_identity')['certified']
    f['hash']=f['hash'].replace('identity\tc','identity\ta')
    f['attention']=f['attention'].replace('0.75','0.5')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'bad_mass')['certified']
