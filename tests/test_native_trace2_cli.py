"""Actual CLI workflow with a declared synthetic oracle; never the real model."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_native_trace2_collect import _oracle, _free_port
from certificate import sha256
from run_native_trace2 import ROOT


def cli(out, *args, ok=True):
    p = subprocess.run([sys.executable, str(ROOT / 'py/run_native_trace2.py'),
                        args[0], str(out), *args[1:]], capture_output=True, text=True)
    assert (p.returncode == 0) == ok, p.stdout+p.stderr
    return p


def setup(out, tmp_path, *, uniform_sign=False):
    cli(out, 'inputs', '--write')
    args = _oracle(tmp_path, _free_port())
    rows = json.loads((out/'dataset.json').read_text())
    # Test-only ground truth: exact intended copies; negatives answer token 42.
    # Unit 1 carries a perfect class signal; the other units are constant.
    table = {','.join(map(str,r['ctx'])):[r['negative'],r['expected'],uniform_sign and r['part']=='sign'] for r in rows}
    source = args.binary.read_text()
    source = source.replace('PORT = int(', 'TABLE = '+repr(table)+'\nPORT = int(')
    source = source.replace('        observations = []', '''        neg, target, uniform = TABLE[','.join(map(str,ids))]
        answer = 42 if neg else target
        observations = []''')
    source = source.replace('        # id-derived answer', '''        for o in observations:
            for channel in ('gate','value','product'):
                o[channel] = (-4.0 if neg and not uniform else 4.0) if o['neuron']==1 else 1.0
        # id-derived answer''')
    source = source.replace('"next": 900 + ids[-1]', '"next": answer')
    source = source.replace('"observations": observations}',
                            '"observations": observations, "logits": [[t,1.5] for t in probe.get("readouts",[])]}')
    args.binary.write_text(source)
    report = json.loads((args.preflight/'report.json').read_text())
    report['binary_sha256'] = sha256(args.binary)
    (args.preflight/'report.json').write_text(json.dumps(report))
    flags = ['--binary',str(args.binary),'--bundle',str(args.bundle),
             '--preflight',str(args.preflight),'--port',str(args.port)]
    return flags


def test_cli_full_workflow_preserves_failed_checks_and_runtime(tmp_path):
    out = tmp_path/'run'
    flags = setup(out, tmp_path)
    cli(out, 'collect', *flags, ok=False)  # split is required, no silent empty run
    assert not list(out.glob('*-observations.jsonl'))
    default_flags = flags.copy()
    at = default_flags.index('--preflight')
    del default_flags[at:at+2]
    refused = cli(out, 'collect', '--split', 'select', *default_flags, ok=False)
    assert 'differs from validated oracle' in refused.stderr
    cli(out, 'collect', '--split', 'select', *flags)
    assert len(json.loads((out/'select-references.json').read_text())) == 114
    cli(out, 'features')
    assert len(json.loads((out/'features.json').read_text())) == 9
    cli(out, 'collect', '--split', 'sign', *flags)
    cli(out, 'select')  # generated, real 1,344-candidate grammar must parse and rank
    selection = json.loads((out/'selection.json').read_text())
    assert not selection['relations']['degenerate']
    assert selection['relations']['admitted']
    assert 'diagnostic' in cli(out, 'freeze', ok=False).stderr
    cli(out, 'diagnostic', *flags)
    summary = json.loads((out/'diagnostic-summary.json').read_text())
    assert summary['certified']
    assert sorted(summary['relations']['summary']) == [
        ['control_zero','48','0'],['identity','48','0'],['swap','48','0'],['zero','48','0']]
    cli(out, 'freeze')
    frozen = json.loads((out/'frozen.json').read_text())
    assert frozen['artifacts']['trace']['domain']
    assert not (out/'test-references.json').exists()
    cli(out, 'collect', '--split', 'select', *flags, ok=False)
    cli(out, 'inputs', '--write', ok=False)
    cli(out, 'final', *flags)
    # Inject a wrong final reference to prove failure preservation end to end.
    refs = json.loads((out/'test-references.json').read_text())
    refs[str(frozen['artifacts']['trace']['domain'][0])] = 999999
    (out/'test-references.json').write_text(json.dumps(refs))
    cli(out, 'evaluate')
    report = json.loads((out/'report.json').read_text())
    assert not report['certificates']['trace']['firing']['certified']
    assert report['certificates']['trace']['firing']['nmiss'] >= 1
    assert report['summary']['relations']['counterexample']
    p = subprocess.run([sys.executable,str(ROOT/'py/audit_native_trace2.py'),str(out),'--write'],
                       capture_output=True,text=True)
    assert p.returncode == 0, p.stdout+p.stderr
    audit = json.loads((out/'replay-audit.json').read_text())
    assert len(audit['evidence']) >= 25
    assert all(r['verdict_preserved'] and r['all_relations_identical'] for r in audit['evidence'])
    assert all(r['predictions_match'] for r in audit['standalone'])
    # Removing all replay evidence must never yield a successful audit.
    (out/'certificate-evidence').rename(out/'hidden-evidence')
    p = subprocess.run([sys.executable,str(ROOT/'py/audit_native_trace2.py'),str(out)],capture_output=True)
    assert p.returncode != 0


def test_diagnostic_summary_rejects_missing_identity_or_extra_request(tmp_path):
    from certificate import check
    from native_trace import DL
    candidate=tmp_path/'empty.dl';candidate.write_text('')
    staged={'expected':'1\t7\t2\tidentity\n','observation':'', 'result':''}
    result=check(DL/'native_trace_diagnostic2.dl',candidate,staged,{})
    assert not result['certified'] and not result.get('error')
    staged['result']='1\t7\t2\tidentity\t4\t4\t1.5\t1.5\t2.0\t2.0\n'
    assert check(DL/'native_trace_diagnostic2.dl',candidate,staged,{})['certified']
    staged['result']+='2\t7\t2\tzero\t4\t4\t1.5\t1.5\t2.0\t2.0\n'
    assert not check(DL/'native_trace_diagnostic2.dl',candidate,staged,{})['certified']


def test_cli_degenerate_run_is_sealed_auditable_and_cannot_query_final(tmp_path):
    out = tmp_path/'run'
    flags = setup(out, tmp_path, uniform_sign=True)
    cli(out, 'collect', '--split', 'select', *flags)
    cli(out, 'features')
    cli(out, 'collect', '--split', 'sign', *flags)
    cli(out, 'select')
    assert json.loads((out/'report.json').read_text())['status'] == 'degenerate'
    assert json.loads((out/'frozen.json').read_text())['status'] == 'degenerate'
    assert not (out/'artifacts').exists()
    assert not json.loads((out/'selection.json').read_text())['relations']['score']
    cli(out, 'diagnostic', *flags, ok=False)
    cli(out, 'final', *flags, ok=False)
    assert not (out/'test-observations.jsonl').exists()
    p = subprocess.run([sys.executable,str(ROOT/'py/audit_native_trace2.py'),str(out)],
                       capture_output=True,text=True)
    assert p.returncode == 0, p.stdout+p.stderr
