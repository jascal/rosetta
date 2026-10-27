"""Model-free replay of frozen adaptive plans, all branches, and root witnesses."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_gate_bisection import ROOT,OUT,PRIOR,checked,journal,read,write
from qwen35_reference_evidence import replay_archive

def main():
    a=type('Args',(),{'out':OUT})();checked(a)
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed evidence changed')
    initial=read(OUT/'initial.json');roots={c:t for c,node,t in initial['ref']}
    for case,e in read(OUT/'root-witnesses.json').items():
        after=replay_archive(PRIOR/'reference-evidence'/e['archive'],e['sha256'],row_id=10*int(case)+3)
        if not after['certified'] or after['relations']!=e['query']['relations'] or after['relations']['ref']!=[[roots[case]]]:raise ValueError('root replay differs')
    plans={int(p.parent.name.split('-')[1]):read(p) for p in OUT.glob('round-*/plan.json')}
    obs=journal(OUT)
    for r in obs.values():
        if [str(r['case']),str(r['node'])] not in plans[r['round']]['relations']['pending']:raise ValueError('unplanned observation')
        e=r['reference'];after=replay_archive(OUT/'reference-evidence'/e['archive'],e['sha256'],row_id=100*r['case']+r['node'])
        if not after['certified'] or after['relations']!=e['query']['relations'] or after['relations']['ref']!=[[str(r['next'])]]:raise ValueError('branch reference replay differs')
    queries=[read(OUT/'inputs.json')]+list(plans.values())+[read(p) for p in OUT.glob('site-*.json')]
    for before in queries:
        after=replay(Path(before['evidence']))
        if not after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    for roundid,plan in plans.items():
        frozen=read(OUT/f'round-{roundid}/frozen.json')
        for n,h in frozen['files'].items():
            if sha256(OUT/f'round-{roundid}'/n)!=h:raise ValueError('round plan changed')
        expected={(c,node,t) for c,node,t in initial['ref']}|{(str(r['case']),str(r['node']),str(r['next'])) for r in obs.values() if r['round']<roundid}
        actual={tuple(line.split('\t')) for line in (Path(plan['evidence'])/'facts/ref.facts').read_text().splitlines()}
        if actual!=expected:raise ValueError('round used unexpected/future references')
    report={'closed_sha256':sha256(OUT/'closed.json'),'root_witnesses':len(roots),'new_reference_witnesses':len(obs),'queries':len(queries),'rounds':len(plans),'unchanged':True,'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_gate_bisection.py')}
    write(ROOT/'reference/benchmarks/qwen35_gate_bisection_replay.json',report);print(report)
if __name__=='__main__':main()
