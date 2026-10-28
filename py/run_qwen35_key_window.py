"""One frozen competing-key window study. No search, retuning, or runtime artifact."""
import argparse,json
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_copy_inventory import ROOT,MODEL,OUT as INVENTORY,gate,read,write
PARENT=ROOT/'reference/benchmarks/qwen35_attention_qk'
REPAIR=ROOT/'reference/benchmarks/qwen35_key_window_repair'
OUT=ROOT/'reference/benchmarks/qwen35_key_window'
SOURCES=('py/run_qwen35_key_window.py','dl/qwen35_attention_qk_source_counts.dl','dl/qwen35_attention_qk_payload.dl','dl/qwen35_key_window_plan.dl','dl/qwen35_key_window_payload.dl','dl/qwen35_key_window_key_match.dl','dl/qwen35_key_window_logit_control.dl','dl/qwen35_key_window_outcome.dl','dl/qwen35_attention_factor_summary.dl','dl/copy_longest_runtime.dl','py/qwen35_attention_qk_reference.py','py/qwen35_attention_factor_reference.py','py/qwen35_attention_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','dl/qwen35_reference_argmax.dl','py/certificate.py','docs/qwen35-key-window-policy.md')
CONTROLS=('captured_self','captured_donor_short','self_full_prefix','donor_full_prefix')
def require(x,required,nonempty=(),positive=True):
    r=x.get('relations',{})
    if any(k not in r for k in required) or any(not r.get(k) for k in nonempty):raise ValueError('missing expected relation; certified bit is insufficient')
    if r.get('invalid'):raise ValueError('invalid/incomplete evidence')
    if positive and not x.get('certified'):raise ValueError('checker did not certify')
    return r

def query(a,name,module,candidate,facts,required,nonempty=(),positive=True):
    x=check(ROOT/module,candidate,facts,{},evidence_dir=a.out/f'certificate-evidence/{name}',provenance={'source':'user-scoped frozen key-window empirical test; never certified model coverage'})
    write(a.out/f'{name}.json',x);require(x,required,nonempty,positive);return x

def inputs(a):
    import torch
    if a.out.exists():raise ValueError('refusing to overwrite study')
    for n,h in read(PARENT/'closed.json')['files'].items():
        if sha256(PARENT/n)!=h:raise ValueError('frozen Q/K evidence changed')
    repair=read(REPAIR/'repair-receipt.json')
    for n,h in repair['repaired_sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError('repair receipt source changed')
    counts=read(REPAIR/'site-counts.json');contexts=read(PARENT/'contexts.json')
    require(counts,('counts','invalid','certified'),('counts','certified'))
    if {tuple(r) for r in counts['relations']['counts']}!={(str(r['id']),role,'1') for r in contexts for role in ('long','short')}:raise ValueError('strict uniqueness failed on frozen contrasts')
    a.out.mkdir(parents=True);write(a.out/'contexts.json',contexts)
    (a.out/'input-core.dl').write_bytes((PARENT/'input-core.dl').read_bytes())
    facts_path=Path(read(PARENT/'payload.json')['evidence'])/'facts'
    base={n:(facts_path/f'{n}.facts').read_text() for n in ('tok','context','shape','qvec','kvec','scale','mask','weight','value','gate')}
    plan=query(a,'plan','dl/qwen35_key_window_plan.dl',a.out/'input-core.dl',{'tok':base['tok'],'context':base['context']},('probe','probe_tok','source_position','mode','invalid','certified'),('probe','probe_tok','source_position','mode'))
    positions={int(i):int(p) for i,p in plan['relations']['source_position']};probes=[]
    for aa,ii,kk,mode,pp in plan['relations']['probe']:
        tokens=sorted((int(p),int(t)) for ar,p,t in plan['relations']['probe_tok'] if ar==aa)
        if [p for p,t in tokens]!=list(range(int(pp)+1)):raise ValueError('probe has missing absolute positions')
        probes.append({'id':int(aa),'case':int(ii),'k':int(kk),'mode':mode,'position':int(pp),'ctx':[t for p,t in tokens]})
    write(a.out/'probes.json',probes)
    expected=''.join(f'{i}\t{d}\t{v}\n' for i,p,d,v in (line.split('\t') for line in base['kvec'].splitlines()) if positions[int(i)]==int(p))
    (a.out/'expected-key.facts').write_text(expected)
    base['control_vector']='';base['prior_probability']='';reference_short={};reference_self={};control_sources={}
    pp=read(PARENT/'payload.json')['relations']['probability']
    for r in contexts:
        i=r['id'];capture=torch.load(PARENT/f'capture-{i}.pt',weights_only=True)
        base['control_vector']+=''.join(f"{6*i}\t{o['dimension']}\t{o['value']:.17g}\n" for o in capture['vectors'])
        source_arm=6*i+5;short=torch.load(PARENT/f'arm-{source_arm}.pt',weights_only=True)
        base['control_vector']+=''.join(f"{source_arm}\t{o['dimension']}\t{o['after']:.17g}\n" for o in short['applied'])
        base['prior_probability']+=''.join(f'{6*i}\t{p}\t{v}\n' for ii,p,v in (s.split('\t') for s in base['weight'].splitlines()) if int(ii)==i)
        base['prior_probability']+=''.join('\t'.join(row)+'\n' for row in pp if int(row[0])==source_arm)
        for arm,dest in ((6*i,reference_self),(source_arm,reference_short)):
            e=read(PARENT/f'reference-{arm}.json');rr=require(e['query'],('ref','invalid','certified'),('ref','certified'))
            if len(rr['ref'])!=1:raise ValueError('nonfunctional frozen reference')
            dest[str(i)]=int(rr['ref'][0][0])
        if reference_self[str(i)]!=r['expected']:raise ValueError('captured target reference differs')
        control_sources[str(i)]={'self_prior_arm':6*i,'donor_prior_arm':source_arm,'self_file':str((PARENT/f'capture-{i}.pt').relative_to(ROOT)),'donor_file':str((PARENT/f'arm-{source_arm}.pt').relative_to(ROOT))}
    write(a.out/'base-facts.json',base);write(a.out/'reference-self.json',reference_self);write(a.out/'reference-short.json',reference_short);write(a.out/'control-sources.json',control_sources)
    write(a.out/'protocol.json',{'scope':'one user-frozen empirical key-window test; not a runtime product or new certified coverage','modes':['full_prefix','window4_fill0','window4_fill1','window16_fill0','window16_fill1'],
        'capture_policy':'same six source positions, absolute positions preserved; full-prefix captures first; stop if full-prefix key match fails at1e-5+1e-5abs; then exactly the remaining24 fixed probes',
        'substitutions':'captured target replay and captured donor-short replay at distinct IDs; full-prefix same-source/donor controls; same-source and donor local windows; only one short-position key changes per head-local calculation',
        'gate':'complete prior_probability and control_vector coverage; probability/vector band1e-5+1e-5abs; whole-vocabulary control band1e-5+1e-5abs with exact argmax; all24 controls before local substitutions',
        'interpretation':'key match at1e-5 and same-source/donor-short argmax matches reported per mode; transfer without key match is negative recovery; no new widths/fillers/heads/layers or token-only extraction',
        'capture_calls':30,'substitution_calls':72,'priors':{str(p.relative_to(ROOT)):sha256(p) for p in (PARENT/'closed.json',REPAIR/'repair-receipt.json',REPAIR/'site-counts.json')},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen one test:30 captures,24 controls,48 local substitutions',flush=True)

def checked(a,payload=False):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('frozen input changed')
    for n,h in (p['sources']|p['priors']).items():
        if sha256(ROOT/n)!=h:raise ValueError('source/prior changed')
    if payload:
        for n,h in read(a.out/'payload-frozen.json')['files'].items():
            if sha256(a.out/n)!=h:raise ValueError('numerical payload/capture changed')
    return p

def key_check(a,k):
    import torch
    probes=[r for r in read(a.out/'probes.json') if r['k']==k];width=int(read(a.out/'base-facts.json')['shape'].strip());actual=''
    for r in probes:
        data=torch.load(a.out/f"probe-{r['id']}.pt",weights_only=True)
        actual+=''.join(f"{r['case']}\t{o['dimension']}\t{o['value']:.17g}\n" for o in data['key'])
    m=ROOT/'dl/qwen35_key_window_key_match.dl'
    x=query(a,f'key-match-{k}','dl/qwen35_key_window_key_match.dl',m,{'dimension':''.join(f'{d}\n' for d in range(width)),
        'context':''.join(f"{r['case']}\n" for r in probes),'expected':(a.out/'expected-key.facts').read_text(),'actual':actual},('invalid','bad','result','counts','certified'),('result','counts'),False)
    if {int(i) for i,ok in x['relations']['result']}!={r['case'] for r in probes}:raise ValueError('missing expected key-result rows')
    if not x['certified'] and not x['relations']['bad']:raise ValueError('false key certificate without a complete negative comparison')
    return x

def stop(a,reason):
    modes=read(a.out/'plan.json')['relations']['mode'];table=[]
    for k,m,w,f in sorted(modes,key=lambda r:int(r[0])):
        path=a.out/f'key-match-{k}.json';matched=None
        if path.exists():matched=int(read(path)['relations']['counts'][0][1])
        table.append({'mode':m,'key_matches':matched,'total':6,'self_argmax_matches':None,'donor_short_argmax_matches':None,'status':'negative' if matched is not None and matched<6 else 'open_control_stop'})
    write(a.out/'outcome.json',{'tag':'empirical' if table[0]['key_matches'] is not None else 'open','stopped':reason,'table':table,'new_certified_coverage':False})
    close(a);print('STOP',reason,table,flush=True)

def capture(a):
    checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_attention_qk_reference import AttentionQKReference
    torch.set_num_threads(2);model=AttentionQKReference(MODEL,vocab_chunk=1024);probes=read(a.out/'probes.json')
    for k in range(5):
        for r in sorted((r for r in probes if r['k']==k),key=lambda r:r['case']):
            path=a.out/f"probe-{r['id']}.pt"
            if path.exists():continue
            d=model.capture(r['ctx'],[(27,10)]);key=[o for o in d['rotated_keys'] if o['source']==r['position']]
            tmp=path.with_suffix('.pending');torch.save({'key':key,'logits':d['logits']},tmp);tmp.replace(path)
            print('captured',r['case'],r['mode'],flush=True)
        x=key_check(a,k)
        if k==0 and not x['certified']:stop(a,'full_prefix_key_mismatch');return
    print('all30 fixed feature probes retained',flush=True)

def prepare(a):
    checked(a)
    if (a.out/'outcome.json').exists():raise ValueError('study already stopped/closed')
    if (a.out/'payload-frozen.json').exists():checked(a,True);return
    import torch
    facts=read(a.out/'base-facts.json');facts['proxy_key']=''
    for r in read(a.out/'probes.json'):
        data=torch.load(a.out/f"probe-{r['id']}.pt",weights_only=True)
        facts['proxy_key']+=''.join(f"{r['case']}\t{r['k']}\t{o['dimension']}\t{o['value']:.17g}\n" for o in data['key'])
    m=ROOT/'dl/qwen35_key_window_payload.dl'
    x=query(a,'payload','dl/qwen35_key_window_payload.dl',a.out/'input-core.dl',facts,
        ('invalid','probability_bad','vector_bad','run','run_info','patch','probability','checked_reference','certified'),('run','run_info','patch','probability','checked_reference'),False)
    if not x['certified']:stop(a,'full_prefix_probability_or_vector_reconstruction_failed');return
    if len(x['relations']['run'])!=72 or len(x['relations']['checked_reference'])!=24:raise ValueError('missing expected control/run rows')
    write(a.out/'payload-frozen.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='payload-frozen.json'}})
    print('repaired comparison passes; all72 numerical payloads frozen',flush=True)

def full_control(a,arm,target,mode,scores):
    import torch
    prior_arm=6*target+5 if mode in ('captured_donor_short','donor_full_prefix') else 6*target
    path=PARENT/f'arm-{prior_arm}.pt' if prior_arm%6==5 else PARENT/f'capture-{target}.pt'
    prior=torch.load(path,weights_only=True)['logits'];m=ROOT/'dl/qwen35_key_window_logit_control.dl'
    x=query(a,f'control-{arm}','dl/qwen35_key_window_logit_control.dl',m,{'vocab':f'{len(scores)}\n',
        'score':''.join(f'{s}\t{t}\t{v:.17g}\n' for s,values in (('original',prior),('reconstructed',scores)) for t,v in enumerate(values.tolist()))},
        ('invalid','badargmax','badlogit','ref','certified'),('ref',),False)
    if len(x['relations']['ref'])!=2 or {s for s,t in x['relations']['ref']}!={'original','reconstructed'}:raise ValueError('missing expected whole-vocabulary argmax relation')
    return x

def run(a):
    checked(a,True)
    if (a.out/'outcome.json').exists():raise ValueError('study already closed')
    gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_attention_reference import AttentionReference
    torch.set_num_threads(2);model=AttentionReference(MODEL,vocab_chunk=1024);plan=read(a.out/'payload.json')['relations'];contexts={r['id']:r for r in read(a.out/'contexts.json')}
    arms=sorted(plan['run'],key=lambda r:(r[3] not in CONTROLS,int(r[0])));failed=False
    for j,(aa,ii,dd,mode) in enumerate(arms):
        if mode not in CONTROLS and failed:stop(a,'whole_vocabulary_control_failed');return
        arm,i=int(aa),int(ii);path=a.out/f'arm-{arm}.pt'
        if path.exists():d=torch.load(path,weights_only=True)
        else:
            patches=[(int(l),int(p),int(h),int(n),float(v)) for a,l,p,h,n,v in plan['patch'] if a==aa]
            d=model.probe(contexts[i]['ctx'],(),patches);tmp=path.with_suffix('.pending');torch.save(d,tmp);tmp.replace(path);print('substitution',j+1,'/',len(arms),mode,i,flush=True)
        if mode in CONTROLS and not full_control(a,arm,i,mode,d['logits'])['certified']:failed=True
    print('all controls passed; all fixed substitutions retained',flush=True)

def evaluate(a):
    checked(a,True)
    if (a.out/'outcome.json').exists():raise ValueError('study already closed')
    import torch
    from qwen35_reference_evidence import extract,replay_archive
    plan=read(a.out/'payload.json')['relations'];refs={};requests=''.join('\t'.join(r)+'\n' for r in plan['patch']);applied=''
    # Revalidate every whole-vocabulary control before interpreting any local hybrid.
    for aa,ii,dd,m in plan['run']:
        if m in CONTROLS:
            d=torch.load(a.out/f'arm-{aa}.pt',weights_only=True)
            if not full_control(a,int(aa),int(ii),m,d['logits'])['certified']:stop(a,'whole_vocabulary_control_failed_at_evaluation');return
    for aa,ii,dd,m in plan['run']:
        arm=int(aa);d=torch.load(a.out/f'arm-{arm}.pt',weights_only=True);path=a.out/'reference-evidence'/f'reference-{arm}.tar.gz'
        if path.exists():
            h=sha256(path);e={'archive':path.name,'sha256':h,'query':replay_archive(path,h,row_id=arm)}
        else:e=extract(d['logits'],path,arm)
        require(e['query'],('invalid','ref','certified'),('ref','certified'))
        if len(e['query']['relations']['ref'])!=1:raise ValueError('missing or nonfunctional argmax')
        write(a.out/f'reference-{arm}.json',e);refs[aa]=int(e['query']['relations']['ref'][0][0])
        applied+=''.join(f"{arm}\t{o['layer']}\t{o['position']}\t{o['head']}\t{o['dimension']}\t{o['before']:.17g}\t{o['after']:.17g}\n" for o in d['applied'])
    # Existing general FP32 patch check; reconstruction flags come from all positive controls.
    shape={'run':''.join('\t'.join(r)+'\n' for r in plan['run']),
        'expected':''.join(f"{r['id']}\t{r['expected']}\n" for r in read(a.out/'contexts.json')),
        'ref':''.join(f'{a}\t{t}\n' for a,t in refs.items()),'reconstruction':'','request':requests,'applied':applied}
    m=ROOT/'dl/qwen35_attention_factor_summary.dl';query(a,'site-check','dl/qwen35_attention_factor_summary.dl',m,shape,('invalid','counts','transferred','certified'),('counts','certified'))
    key_results=''.join(f'{i}\t{k}\t{ok}\n' for k in range(5) for i,ok in read(a.out/f'key-match-{k}.json')['relations']['result'])
    meta=plan['run_info'];donors={i:d for a,i,d,m in plan['run']};m=ROOT/'dl/qwen35_key_window_outcome.dl'
    x=query(a,'summary','dl/qwen35_key_window_outcome.dl',m,{
        'mode':''.join(f'{k}\t{label}\n' for k,label,w,f in read(a.out/'plan.json')['relations']['mode']),
        'context':''.join(f"{r['id']}\t{r['expected']}\n" for r in read(a.out/'contexts.json')),
        'donor':''.join(f'{i}\t{d}\n' for i,d in donors.items()),'meta':''.join('\t'.join(r)+'\n' for r in meta),
        'reference_short':''.join(f'{i}\t{t}\n' for i,t in read(a.out/'reference-short.json').items()),
        'ref':shape['ref'],'key_result':key_results},('invalid','counts','recovered','transfer_without_key_match','argmax_match','certified'),('counts','certified'))
    if len(x['relations']['counts'])!=5:raise ValueError('missing expected outcome rows')
    table=[{'mode':label,'total':int(n),'key_matches':int(km),'self_argmax_matches':int(sm),'donor_short_argmax_matches':int(dm),'unmatched_transfers':int(ut),'status':'empirical_recovery' if k in {r[0] for r in x['relations']['recovered']} else 'negative_recovery'} for k,label,n,km,sm,dm,ut in sorted(x['relations']['counts'],key=lambda r:int(r[0]))]
    write(a.out/'outcome.json',{'tag':'empirical','stopped':False,'table':table,'new_certified_coverage':False});close(a);print(json.dumps(table),flush=True)

def close(a):
    reports=[p for p in a.out.glob('*.json') if p.name not in ('protocol.json','outcome.json','closed.json','replay-audit.json','payload-frozen.json')]
    queries=0
    for p in reports:
        before=read(p)
        if not isinstance(before,dict) or 'evidence' not in before:continue
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
        queries+=1
    write(a.out/'replay-audit.json',{'queries':queries,'unchanged':True,'model_coverage_certificate':False})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','capture','prepare','run','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();globals()[a.phase](a)
if __name__=='__main__':main()
