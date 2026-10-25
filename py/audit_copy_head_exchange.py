"""Model-free replay of head exchange, including retained preflight failures."""
import argparse
from pathlib import Path
from certificate import replay,sha256
from run_copy_head_exchange import OUT,gate,read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    a=parser.parse_args();a.out=a.out.resolve();gate(a)
    for name,digest in read(a.out/'closed.json')['files'].items():
        if sha256(a.out/name)!=digest:raise ValueError(f'closed evidence changed: {name}')
    count=0
    for path in sorted((a.out/'certificate-evidence').rglob('certificate.json')):
        before=read(path)['result'];after=replay(path.parent)
        if before['certified']!=after['certified'] or before.get('error')!=after.get('error'):
            raise ValueError(f'verdict differs: {path}')
        recorded={p.stem:[line.split('\t') for line in p.read_text().splitlines()]
            for p in (path.parent/'outputs').glob('*.csv')}
        if after.get('relations',{})!=recorded:raise ValueError(f'relations differ: {path}')
        count+=1
    print(count,'queries replay unchanged; frozen inputs, sources and closed evidence unchanged')


if __name__=='__main__':main()
