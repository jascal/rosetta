"""Replay all categorical readout queries without checkpoint, weights or torch."""
import argparse
from pathlib import Path
from certificate import replay,sha256
from run_copy_value_readout import OUT,gate,read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();a.out=a.out.resolve();gate(a)
    for name,digest in read(a.out/'closed.json')['files'].items():
        if sha256(a.out/name)!=digest:raise ValueError(f'closed evidence changed: {name}')
    count=0
    for path in sorted((a.out/'certificate-evidence').rglob('certificate.json')):
        before=read(path)['result'];after=replay(path.parent)
        recorded={p.stem:[line.split('\t') for line in p.read_text().splitlines()]
            for p in (path.parent/'outputs').glob('*.csv')}
        if before['certified']!=after['certified'] or before.get('error')!=after.get('error') or recorded!=after.get('relations',{}):
            raise ValueError(f'replay changed: {path}')
        count+=1
    print(count,'queries replay unchanged; frozen and closed hashes unchanged; no model loaded')


if __name__=='__main__':main()
