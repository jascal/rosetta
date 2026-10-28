"""Retain exact finite-domain verdicts for all fixed development candidates."""
import argparse
from pathlib import Path

from certificate import check, replay
from copy_long_consensus_artifact import candidate
from run_copy_long_consensus import gate, journal, read, write, ROOT, OUT, OPTIONS


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
    for c, k, support in OPTIONS:
        directory = a.out / 'proposals' / str(c)
        directory.mkdir(parents=True, exist_ok=False)
        circuit = directory / 'circuits.dl'
        circuit.write_text('// Development proposal; not final-test evidence.\n'+candidate((c,k,support)))
        ids = {int(i) for option, i, _ in domains['relations']['firing'] if int(option) == c}
        subset = [r for r in rows if r['id'] in ids]
        facts = {'domain': ''.join(f'{i}\n' for i in sorted(ids)),
                 'tok': ''.join(f"{r['id']}\t{p}\t{t}\n" for r in subset for p,t in enumerate(r['ctx'])),
                 'ref': ''.join(f"{i}\t{observations[i]['response']['next']}\n" for i in sorted(ids))}
        result = check(ROOT / 'dl/equiv.dl', circuit, facts, {'ndomain': int, 'nmiss': int,
                       'nuncov': int, 'nmissing': int, 'ninvalid': int},
                       evidence_dir=a.out / f'certificate-evidence/development-equiv-{c}',
                       provenance={'source': 'pinned f32 Qwen DEVELOPMENT references; not untouched final-test evidence'})
        reports.append({'candidate': c, 'suffix': k, 'support': support, 'certificate': result})
        print(c, {key: result.get(key) for key in ('certified','ndomain','nmiss','nuncov')})
    write(a.out / 'development-certificates.json', {'scope': 'all six development firing domains; no final test',
                                                  'candidates': reports})


if __name__ == '__main__':
    main()
