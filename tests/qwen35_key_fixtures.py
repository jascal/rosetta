"""Small complete numerical fixtures; no model or captured tensors required."""
import math

def rows(kind='one'):
    q=list(range(100,124));long=q+[70,900];short=q[-12:]+[80,901]
    blocks={'one':long+short,'zero_short':long,'two_short':long+short+short,'two_long':long+long+short}
    return blocks[kind]+q

def facts_for(sequence,window=False):
    size=len(sequence);longs=[p for p,t in enumerate(sequence) if t==70];shorts=[p for p,t in enumerate(sequence) if t==80]
    weights={};values={};vectors={};prior={}
    for i in (0,1):
        ks=[1.0 if p in longs else (1.5 if i==0 else -1.0) if p in shorts else 0.0 for p in range(size)]
        v=[2.0 if p in longs else -1.0 if p in shorts else 0.0 for p in range(size)];values[i]=v
        score=[(i+1)*x for x in ks];z=sum(math.exp(x) for x in score)
        weights[i]=[math.exp(x)/z for x in score];vectors[i]=.5*sum(w*x for w,x in zip(weights[i],v))
        prior[6*i]=weights[i]
        mixed=[1.0 if p in longs else (-1.0 if i==0 else 1.5) if p in shorts else 0.0 for p in range(size)]
        score=[(i+1)*x for x in mixed];z=sum(math.exp(x) for x in score);prior[6*i+5]=[math.exp(x)/z for x in score]
    f={'shape':'1\n','context':''.join(f'{i}\t0\t{s}\t{size-1}\n' for i,s in ((0,'old'),(1,'new'))),
       'tok':''.join(f'{i}\t{p}\t{t}\n' for i in (0,1) for p,t in enumerate(sequence)),
       'qvec':'0\t0\t1.0\n1\t0\t2.0\n','scale':'0\t1.0\n1\t1.0\n',
       'kvec':''.join(f'{i}\t{p}\t0\t{1.0 if p in longs else (1.5 if i==0 else -1.0) if p in shorts else 0.0}\n' for i in (0,1) for p in range(size)),
       'mask':''.join(f'{i}\t{p}\t0.0\n' for i in (0,1) for p in range(size)),
       'weight':''.join(f'{i}\t{p}\t{x:.17g}\n' for i in (0,1) for p,x in enumerate(weights[i])),
       'value':''.join(f'{i}\t{p}\t0\t{x:.17g}\n' for i in (0,1) for p,x in enumerate(values[i])),
       'gate':'0\t0\t0.0\n1\t0\t0.0\n'}
    if window:
        f['control_vector']=''.join(f'{a}\t0\t{.5*sum(w*x for w,x in zip(ws,values[a//6])):.17g}\n' for a,ws in prior.items())
        f['prior_probability']=''.join(f'{a}\t{p}\t{x:.17g}\n' for a,ws in prior.items() for p,x in enumerate(ws))
        f['proxy_key']=''.join(f'{i}\t{k}\t0\t{1.5 if i==0 else -1.0}\n' for i in (0,1) for k in range(5))
    else:
        f['control_vector']=''.join(f'{i}\t{k}\t0\t{vectors[i if k=="self" else 1-i]:.17g}\n' for i in (0,1) for k in ('self','donor'))
    return f
