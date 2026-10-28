"""Replay head-deletion instrument queries with no model, binary or weights."""
import argparse
from pathlib import Path

from certificate import replay, sha256
from run_copy_mechanism import ROOT, read, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path,nargs='?',default=ROOT/'reference/benchmarks/qwen25_copy_heads')
    a=p.parse_args();out=a.out.resolve()
    frozen=read(out/'frozen.json')
    for n,h in frozen['files'].items():
        if sha256(out/n)!=h: raise ValueError(f'frozen input changed: {n}')
    for n,h in (frozen['sources']|frozen['historical']).items():
        if sha256(ROOT/n)!=h: raise ValueError(f'frozen source changed: {n}')
    closed=out/'closed.json'
    if closed.exists():
        for n,h in read(closed)['files'].items():
            if sha256(out/n)!=h: raise ValueError(f'closed evidence changed: {n}')
    checks=[]
    for path in sorted((out/'certificate-evidence').rglob('certificate.json')):
        before=read(path);after=replay(path.parent)
        recorded={p.stem:[line.split('\t') for line in p.read_text().splitlines()]
                  for p in (path.parent/'outputs').glob('*.csv')}
        if after['certified']!=before['result']['certified'] or after.get('error') or after['relations']!=recorded:
            raise ValueError(f'replay changed: {path}')
        checks.append({'path':str(path.parent.relative_to(out)),'certified':after['certified']})
    write(out/'replay-audit.json',{'scope':'instrument consistency, no model-equivalence claim',
        'frozen_hashes_unchanged':True,'checks':checks})
    print(len(checks),'head-deletion instrument queries replayed unchanged')


if __name__=='__main__': main()
