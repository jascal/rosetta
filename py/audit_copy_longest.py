"""Model-free replay of long-source development, including rejected certificates."""
import argparse
from pathlib import Path

from certificate import replay, sha256
from run_copy_longest import gate, read, write, OUT, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.binary = a.bundle = None
    gate(a)
    closed = a.out / 'closed.json'
    if closed.exists():
        locked = read(closed)
        for name, digest in locked['files'].items():
            if sha256(a.out / name) != digest:
                raise ValueError(f'closed evidence changed: {name}')
        for name, digest in locked['sources'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'closed reporting source changed: {name}')
    checks = []
    for path in sorted((a.out / 'certificate-evidence').rglob('certificate.json')):
        metadata = read(path)
        result = replay(path.parent)
        if result['certified'] != metadata['result']['certified']:
            raise ValueError(f'verdict changed: {path}')
        if result.get('error'):
            raise ValueError(result['error'])
        outputs = {p.stem: [line.split('\t') for line in p.read_text().splitlines()]
                   for p in (path.parent / 'outputs').glob('*.csv')}
        if outputs != result['relations']:
            raise ValueError(f'relations changed: {path}')
        checks.append({'evidence': str(path.parent.relative_to(a.out)), 'certified': result['certified']})
    write(a.out / 'replay-audit.json', {'scope': 'development only; no final-test certificate',
                                     'inputs_and_sources_unchanged': True, 'checks': checks})
    print(f'{len(checks)} development queries replayed with identical verdicts and relations')


if __name__ == '__main__':
    main()
