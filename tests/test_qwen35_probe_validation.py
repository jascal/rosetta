"""Frozen patch coverage and full-score instrument gates reject partial checks."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check


def test_identity_payload_requires_all_sites(tmp_path):
    m=ROOT/'dl/qwen35_probe_identity.dl'
    facts={'identity_site':'0\t4\t0\n0\t4\t1\n',
        'observation':'baseline\t0\t4\t0\t1.0\t2.0\t3.0\n'}
    bad=check(m,m,facts,{},evidence_dir=tmp_path/'missing')
    assert not bad['certified']
    facts['observation']+='baseline\t0\t4\t1\t2.0\t3.0\t4.0\n'
    good=check(m,m,facts,{},evidence_dir=tmp_path/'complete')
    assert good['certified'] and len(good['relations']['patch'])==2


def test_summary_checks_applied_values_and_identity_hash(tmp_path):
    m=ROOT/'dl/qwen35_probe_summary.dl'
    arms=('baseline','identity','cleanup','zero_last','direct_last')
    facts={'arm':''.join(f'{a}\n' for a in arms),'vocab':'2\n',
        'compare':'baseline\tidentity\nbaseline\tcleanup\nzero_last\tdirect_last\n',
        'observe':'baseline\t0\t4\t0\nidentity\t0\t4\t0\ncleanup\t0\t4\t0\n',
        'score':''.join(f'{a}\t0\t0.0\n{a}\t1\t1.0\n' for a in arms),
        'observation':''.join(f'{a}\t0\t4\t0\t1.0\t2.0\t3.0\n' for a in ('baseline','identity','cleanup')),
        'patch_request':'identity\t0\t4\t0\t3.0\nzero_last\t0\t4\t0\t0.0\ndirect_last\t0\t4\t0\t0.0\n',
        'applied':'identity\t0\t4\t0\t3.0\t3.0\nzero_last\t0\t4\t0\t3.0\t0.0\ndirect_last\t0\t4\t0\t3.0\t0.0\n',
        'logits_hash':''.join(f'{a}\tsame\n' for a in arms)}
    x=check(m,m,facts,{},evidence_dir=tmp_path/'good');assert x['certified'],x
    facts['applied']=facts['applied'].replace('zero_last\t0\t4\t0\t3.0\t0.0','zero_last\t0\t4\t0\t3.0\t1.0')
    bad=check(m,m,facts,{},evidence_dir=tmp_path/'bad-patch');assert not bad['certified']
    facts['logits_hash']=facts['logits_hash'].replace('identity\tsame','identity\tdifferent')
    bad=check(m,m,facts,{},evidence_dir=tmp_path/'bad-hash');assert bad['relations']['hash_bad']
