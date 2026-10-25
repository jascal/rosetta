"""Post-selection descriptive Datalog strata; never changes the frozen family."""
import argparse
from pathlib import Path

from certificate import check, replay
from run_copy_longest import gate, journal, read, write, ROOT, OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    a = parser.parse_args()
    a.out = a.out.resolve()
    a.binary = a.bundle = None
    rows = gate(a)
    selected = read(a.out / 'selection.json')
    if not selected['certified']:
        raise ValueError('selector integrity failed')
    # Reuse and replay exactly the staged selector facts instead of recomputing selection inputs.
    evidence = Path(selected['evidence'])
    before = replay(evidence)
    if before['relations'] != selected['relations']:
        raise ValueError('selection replay changed')
    facts = {p.stem: p.read_text() for p in (evidence / 'facts').glob('*.facts')}
    facts['gold'] = ''.join(f"{r['id']}\t{r['expected']}\n" for r in rows)
    facts['bucket'] = ''.join(f"{r['id']}\t{k}\t{r[k]}\n" for r in rows
                               for k in ('split', 'prefix', 'length', 'order', 'gap', 'short', 'kind', 'negative'))
    verifier = a.out / 'report-verifier.dl'
    verifier.write_text((ROOT / 'dl/copy_longest_select.dl').read_text() + '\n' +
                        (ROOT / 'dl/copy_longest_strata.dl').read_text())
    result = check(verifier, a.out / 'library.dl', facts, {}, evidence_dir=a.out / 'certificate-evidence/report',
                   provenance={'source': 'descriptive development strata; not a final-test certificate'})
    write(a.out / 'report.json', result)
    if not result['certified']:
        raise ValueError(result.get('error', 'report integrity failed'))
    print('selected:', result['relations']['ls_selected'])
    print('development scores:', result['relations']['score'])
    print('strata:', result['relations']['stratum'])


if __name__ == '__main__':
    main()
