"""Evaluate one fixed token-only rule on the already observed inventory."""
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_copy_inventory import ROOT,OUT as INVENTORY,gate,journal,read,write
OUT=ROOT/'reference/benchmarks/qwen35_recent_consensus_development'
def artifact():
    code='.decl lc_option(c:number,minmatch:number,margin:number)\nlc_option(0,32,4).\n'
    code+='.decl lg_option(c:number,minmatch:number,minsupport:number)\nlg_option(0,32,2).\n'
    for name in ('copy_longest_runtime.dl','copy_long_consensus_runtime.dl','qwen35_recent_consensus_runtime.dl'):
        code+=(ROOT/'dl'/name).read_text()+'\n'
    return code

def main():
    OUT.mkdir(exist_ok=False);rows=gate(type('Args',(),{'out':INVENTORY})());obs=journal(INVENTORY)
    circuit=OUT/'circuits.dl';circuit.write_text(artifact())
    domain=check(ROOT/'dl/qwen35_baseline_domain.dl',circuit,{'domain':''.join(f"{r['id']}\n" for r in rows),
        'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx']))},{},evidence_dir=OUT/'certificate-evidence/domains',provenance={'source':'fixed token-only recency/long-consensus hypothesis; known development data'})
    write(OUT/'domains.json',domain)
    ids={int(i) for i,t in domain['relations']['firing']}
    result=check(ROOT/'dl/equiv.dl',circuit,{'domain':''.join(f'{i}\n' for i in sorted(ids)),
        'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows if r['id'] in ids for p,t in enumerate(r['ctx'])),
        'ref':''.join(f"{i}\t{obs[i]['next']}\n" for i in sorted(ids))},{'ndomain':int,'nmiss':int,'nuncov':int,'nmissing':int,'ninvalid':int},
        evidence_dir=OUT/'certificate-evidence/equiv',provenance={'source':'post-hoc fixed hypothesis over known inventory firing domain; not fresh final confirmation'})
    write(OUT/'certificate.json',result)
    for before in (domain,result):
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('replay differs')
    write(OUT/'closed.json',{'inventory_closed_sha256':sha256(INVENTORY/'closed.json'),'files':{str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print({k:result[k] for k in ('certified','ndomain','nmiss','nuncov')},flush=True)
if __name__=='__main__':main()
