"""Retain full-vocabulary parity for the memory-bounded build-time oracle."""
import json
from pathlib import Path
import resource
import shutil
import time
from certificate import check,replay,sha256
from run_copy_mechanism import ROOT,read,write

OUT=ROOT/'reference/benchmarks/qwen35_stream_preflight'
TINY=Path('/tmp/rosetta-qwen35-preflight/model')
TARGET=Path('/tmp/rosetta-qwen35-4b-checkpoint')


def main():
    import torch
    import transformers
    from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    from qwen35_stream_reference import StreamReference
    OUT.mkdir(exist_ok=False)
    for name in ('config.json','model.safetensors.index.json'):shutil.copyfile(TARGET/name,OUT/f'target-{name}')
    write(OUT/'target-revision.json',{'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        'source':'hf models info Qwen/Qwen3.5-4B --expand sha,safetensors,siblings --json'})
    ids=read(Path('/tmp/rosetta-qwen35-preflight/ids.json'))['holdout_ids']
    expected=read(Path('/tmp/rosetta-qwen35-preflight/expected.json'))
    rows=[ids[i-16:i] for i in range(16,76)];write(OUT/'contexts.json',rows)
    sources=('py/verify_qwen35_stream.py','py/qwen35_stream_reference.py','py/certificate.py','dl/qwen35_stream_parity.dl')
    write(OUT/'protocol.json',{'scope':'synthetic dense-hybrid streaming parity; actual 4B weights not evaluated',
        'contexts_sha256':sha256(OUT/'contexts.json'),'ncontexts':60,'vocab_chunk':7,
        'tolerance':{'absolute':0.000002,'relative':0.000002},'precision':'FP32','cache':False,'threads':1,
        'sources':{n:sha256(ROOT/n) for n in sources},
        'checkpoint':str(TINY),'checkpoint_files':{p.name:sha256(p) for p in TINY.iterdir() if p.is_file()}})
    torch.set_num_threads(1)
    streamed=StreamReference(TINY,vocab_chunk=7);reader=streamed.reader
    model=Qwen3_5ForCausalLM(streamed.config).float().eval()
    state={('model.'+key[len(reader.prefix):] if key.startswith(reader.prefix) else key):reader.tensor(key).float()
        for key in reader.files if key.startswith(reader.prefix) or key=='lm_head.weight'}
    model.load_state_dict(state,strict=True);del state
    ordinary=[];actual=[];timings=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            start=time.monotonic();gold=model(torch.tensor([row]),use_cache=False).logits[0,-1]
            observed=streamed.logits(row)
            ordinary.extend(f'{i}\t{t}\t{x:.17g}\n' for t,x in enumerate(gold.tolist()))
            actual.extend(f'{i}\t{t}\t{x:.17g}\n' for t,x in enumerate(observed.tolist()))
            timings.append(time.monotonic()-start)
    module=ROOT/'dl/qwen35_stream_parity.dl'
    result=check(module,module,{'domain':''.join(f'{i}\n' for i in range(60)),
        'vocab':f'{streamed.config.vocab_size}\n','expected':''.join(f'{i}\t{t}\n' for i,t in enumerate(expected)),
        'ordinary':''.join(ordinary),'streamed':''.join(actual)},{},evidence_dir=OUT/'certificate-evidence/stream-parity',
        provenance={'source':'ordinary and streamed FP32 same tiny checkpoint, complete vocabulary'})
    write(OUT/'stream-parity.json',result)
    target_config=Qwen3_5TextConfig.from_dict(read(TARGET/'config.json')['text_config'])
    target_config._attn_implementation='eager';blocks=[]
    for i in range(target_config.num_hidden_layers):
        with torch.device('meta'):block=Qwen3_5DecoderLayer(target_config,i)
        blocks.append({'layer':i,'type':target_config.layer_types[i],'fp32_parameter_bytes':sum(p.numel()*4 for p in block.parameters())})
        del block
    write(OUT/'resource-plan.json',{'actual_4b_weights_loaded':False,'block_dimensions_from_pinned_config':blocks,
        'largest_block_fp32_parameter_bytes':max(b['fp32_parameter_bytes'] for b in blocks),
        'head_chunk_fp32_bytes':1024*target_config.hidden_size*4,
        'note':'Parameter-size bound excludes activations, runtime and allocator overhead; tiny-process RSS is not 4B measured RSS.',
        'tiny_process_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'tiny_context_seconds':timings,'torch':torch.__version__,'transformers':transformers.__version__})
    again=replay(Path(result['evidence']))
    if result['relations']!=again['relations'] or result['certified']!=again['certified']:raise ValueError('replay differs')
    write(OUT/'replay-audit.json',{'queries':1,'unchanged':True})
    write(OUT/'closed.json',{'files':{str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('stream parity',result['certified'],result.get('error'),result.get('relations',{}).get('counts'))
    print('largest 4B block FP32 parameter bytes',max(b['fp32_parameter_bytes'] for b in blocks))


if __name__=='__main__':main()
