"""Retain exact finite-domain failures for all preregistered development candidates."""
import argparse
from pathlib import Path

from certificate import check, replay
from run_copy_prefix import gate, journal, read, write, ROOT, OUT, OPTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    a = parser.parse_args()
    a.out = a.out.resolve()
    a.binary = a.bundle = None
    rows = gate(a)
    observations = journal(a.out / 'observations.jsonl')
    domains = read(a.out / 'development-domains.json')
    replayed = replay(Path(domains['evidence']))
    if replayed['relations'] != domains['relations']:
        raise ValueError('reference-free domains changed')
    reports = []
    for c, k, m in OPTIONS:
        directory = a.out / 'proposals' / str(c)
        directory.mkdir(parents=True, exist_ok=False)
        circuit = directory / 'circuits.dl'
        circuit.write_text('// OPEN: rejected development proposal, not a certified model replacement.\n'
            '.decl cp_option(c:number,k:number,minlen:number)\n' + f'cp_option({c},{k},{m}).\n' +
            (ROOT / 'dl/copy_prefix_runtime.dl').read_text() +
            '\n.decl cdecide(i:number,t:number)\ncdecide(I,T) :- cp_prediction(_,I,T).\n')
        (directory / 'run.dl').write_text('.decl tok(i:number,p:number,t:number)\n.input tok\n'
            '#include "circuits.dl"\n.output cdecide\n.decl abstain(i:number)\n.output abstain\n'
            'abstain(I) :- tok(I,_,_), !cdecide(I,_).\n')
        ids = {int(i) for option, i, _ in domains['relations']['firing'] if int(option) == c}
        subset = [r for r in rows if r['id'] in ids]
        facts = {'domain': ''.join(f'{i}\n' for i in sorted(ids)),
                 'tok': ''.join(f"{r['id']}\t{p}\t{t}\n" for r in subset for p,t in enumerate(r['ctx'])),
                 'ref': ''.join(f"{i}\t{observations[i]['response']['next']}\n" for i in sorted(ids))}
        result = check(ROOT / 'dl/equiv.dl', circuit, facts, {'ndomain': int, 'nmiss': int,
                       'nuncov': int, 'nmissing': int, 'ninvalid': int},
                       evidence_dir=a.out / f'certificate-evidence/development-equiv-{c}',
                       provenance={'source': 'pinned f32 Qwen DEVELOPMENT references; not untouched final-test evidence'})
        reports.append({'candidate': c, 'suffix': k, 'minlen': m, 'certificate': result})
        print(c, {key: result.get(key) for key in ('certified','ndomain','nmiss','nuncov')})
    write(a.out / 'development-certificates.json', {'scope': 'all six development firing domains; no final test',
                                                  'candidates': reports})


if __name__ == '__main__':
    main()
