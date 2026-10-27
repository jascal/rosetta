"""Training-only diagnostic: distinguish source byte copying from token identity."""
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_copy_inventory import ROOT,OUT as INVENTORY,MODEL,gate,journal,read,write
from qwen_token_bytes import vocabulary

OUT=ROOT/'reference/benchmarks/qwen35_copy_bytes'


def main():
    rows=gate(type('Args',(),{'out':INVENTORY})());observed=journal(INVENTORY)
    selected=[r for r in rows if r['split']=='train' and not r['negative']]
    if any(r['id'] not in observed for r in selected):raise ValueError('training incomplete')
    OUT.mkdir(exist_ok=False)
    tokens,special=vocabulary(MODEL/'tokenizer.json')
    positions={i:p['positions'] for p in read(INVENTORY/'pairs.json') for i in (p['a'],p['b'])}
    used={r['expected'] for r in selected}|{observed[r['id']]['next'] for r in selected}
    tails=[];outputs=[];sources=[]
    for r in selected:
        i=r['id'];prediction=observed[i]['next']
        for j,b in enumerate(tokens.get(prediction,b'')):outputs.append(f'{i}\t{j}\t{b}\n')
        for p in positions[i]:
            sources.append(f'{i}\t{p}\n')
            end=p-r['prefix']+r['length']
            tail=b''.join(tokens[t] for t in r['ctx'][p:end])
            tails.extend(f'{i}\t{p}\t{j}\t{b}\n' for j,b in enumerate(tail))
    facts={'target':''.join(f"{r['id']}\t{r['expected']}\n" for r in selected),
        'ref':''.join(f"{r['id']}\t{observed[r['id']]['next']}\n" for r in selected),
        'token_len':''.join(f'{t}\t{len(tokens[t])}\n' for t in sorted(used) if t in tokens),
        'special':''.join(f'{t}\n' for t in sorted(special&used)),
        'source':''.join(sources),'tail_byte':''.join(tails),'output_byte':''.join(outputs)}
    write(OUT/'scope.json',{'scope':'post-hoc training-only byte compatibility; no proposals or final claim',
        'positive_rows':len(selected),'validation_labels_used':False,
        'protocol_sha256':sha256(INVENTORY/'protocol.json'),'tokenizer_sha256':sha256(MODEL/'tokenizer.json'),
        'note':'Uses raw ByteLevel bytes, with no Unicode normalization or inferred source routing.'})
    module=ROOT/'dl/qwen35_copy_bytes.dl'
    x=check(module,module,facts,{},evidence_dir=OUT/'certificate-evidence',
        provenance={'source':'model output versus annotated intended-source byte continuation; training positives only'})
    write(OUT/'classification.json',x)
    again=replay(Path(x['evidence']))
    if x['certified']!=again['certified'] or x['relations']!=again['relations']:raise ValueError('replay differs')
    write(OUT/'closed.json',{'files':{str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print(x['certified'],x.get('error'),x.get('relations',{}).get('counts'))


if __name__=='__main__':main()
