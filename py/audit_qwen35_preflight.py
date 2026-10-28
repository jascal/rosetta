"""Replay retained Qwen3.5 parity evidence without checkpoint access or torch."""
import argparse
from pathlib import Path
from certificate import replay,sha256
from run_copy_mechanism import ROOT,read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',type=Path,nargs='*')
    a=parser.parse_args()
    directories=a.directories or [ROOT/'reference/benchmarks/qwen35_stream_preflight',ROOT/'reference/benchmarks/qwen35_4b_smoke']
    probe=ROOT/'reference/benchmarks/qwen35_4b_probe_validation'
    if not a.directories and (probe/'closed.json').exists():directories.append(probe)
    count=0
    for directory in directories:
        out=directory.resolve()
        for name,digest in read(out/'closed.json')['files'].items():
            if sha256(out/name)!=digest:raise ValueError(f'closed evidence changed: {out/name}')
        manifest=out/('protocol.json' if (out/'protocol.json').exists() else 'frozen.json')
        frozen=read(manifest)
        for name,digest in (frozen['sources']|frozen.get('prior',{})).items():
            if sha256(ROOT/name)!=digest:raise ValueError(f'source changed: {name}')
        for path in sorted((out/'certificate-evidence').rglob('certificate.json')):
            before=read(path)['result'];after=replay(path.parent)
            recorded={p.stem:[line.split('\t') for line in p.read_text().splitlines()]
                for p in (path.parent/'outputs').glob('*.csv')}
            if before['certified']!=after['certified'] or before.get('error')!=after.get('error') or recorded!=after.get('relations',{}):
                raise ValueError(f'replay changed: {path}')
            count+=1
    print(count,'parity queries replay unchanged; no checkpoint, model or torch accessed')


if __name__=='__main__':main()
