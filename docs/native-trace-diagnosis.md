# Why the native-trace arm degenerated in run A

Status: **post-hoc analysis, `empirical`.** This was derived *after* run A
completed and after its outcome was known. It is an explanation of a result that
already stands, not a revision of it. It adds no evidence, changes no verdict,
and does not reopen `docs/native-trace-prereg.md`. Run A's result — zero
incremental certified coverage — is unchanged and remains the result.

Scope: one run, one checkpoint (Qwen2.5-0.5B-Instruct rev
`7ae557604adf67be50417f59c2c2f167def9a775`), one prior construction. Every
claim below is recomputed from the frozen evidence in
`reference/benchmarks/qwen25_native_trace/`; nothing is inferred from memory.

## The short version

Run A's trace arm did not fail because native neurons are uninformative. It
failed because the feature-selection negative class and the evaluation
population were **different populations**, and the threshold separating them was
therefore never calibrated for anything the trace arm was later asked to judge.
The prior's failure mode was off-manifold extrapolation, not weak signal.

## The two populations

**The selection population had no negative controls at all.**
`discovery.json` holds 288 rows and **zero** of them carries a `negative` key.
This was preregistered, not a slip: `docs/native-trace-prereg.md` specifies
discovery as "all 288 **positive** rows".

**The negative class is defined by rule disagreement, not by task type.**
`dl/native_trace_features.dl:19-21`:

```datalog
label(I,1) :- sample(I), cdecide(I,T), ref(I,T).
label(I,0) :- sample(I), cdecide(I,T), ref(I,R), T != R.
```

So `label(I,0)` means *the suffix-3 candidate disagrees with the model*. On a
population containing only positive copy contexts, that class is exactly the
positive contexts the rule gets wrong — recomputed from the frozen evidence, the
**7 rows** `ids 2, 4, 6, 10, 12, 230, 232`, every one of them `kind=repeat`, a
positive copy of length 8.

**The evaluation population did contain negative controls.**
`dataset.json` gives train / validation / test 114 rows each, **18 negative
controls in each**.

The prior was therefore fitted to answer *"where does the suffix-3 rule err on a
positive copy?"* and then deployed to answer *"is this a copy at all?"* Those
are different questions. The second one has an answer on negative controls; the
first one's answer carries no information about it, because the population never
contained one.

## Why that produced a uniform label rather than a weak one

`dl/native_trace_features.dl:32-35` places the decision threshold at the
midpoint of the two class means, `(A+B)/2`, with polarity toward the correct
class, and `:39-40` keeps the single highest-scoring neuron per layer/channel —
nine features total. That threshold is an interpolation between two classes drawn
from **positive copy contexts only**.

A negative control is not a hard case of either class. It is out of the
calibration set entirely, so the midpoint has no meaning for it and the vote
extrapolates. Empirically it extrapolates one way, every time: of the 18
negative controls in train, **18/18 are labelled trace-good**, alongside 96/96
positives. `trace_good` is `{0..113}` — the full split, uniformly.

This is why the run did not produce "trace arm ≈ behavior arm". It produced
*trace arm = no signal at all*.

## Why the shuffled control degenerated with it

`dl/native_trace_select.dl:52-58` adds the trace term `G - 4*B`, where `G`
counts train firings where the arm's signature holds and `B` those where it does
not. With `trace_good` universal, `B` is identically zero for every candidate and
`G` collapses to the candidate's train firing count. The score becomes the
behavior score plus a firing count — identical for the trace arm and for the
shuffled arm, because shuffling a set of all-true signatures produces another
set of all-true signatures.

Verified on the frozen `score` relation: trace and shuffled are **elementwise
identical** across all 1,090 scored candidates and propose the same top-12,
while both differ from behavior-only.

**The shuffled control did its job.** It is the reason this degeneracy is
visible at all. Without it, a uniform prior would have been indistinguishable
from a genuine but weak ranking signal.

## What this does not show

- **Nothing about native neurons in general.** The instrumentation was validated
  against upstream FP32/eager hooks before discovery (12 contexts, 108
  interventions, 525,312 coordinates, max abs error 0.000236786). That gate is
  untouched by this analysis.
- **Nothing about the Goodfire argument.** The claim under test was never
  "interpretable neurons are useless"; it was "this prior improves independently
  certified coverage for this fixed search".
- **No refutation of a trace prior in principle.** A prior fitted on a
  population that resembles the evaluation population was never tested here.
- **No change to run A.** Zero incremental certified coverage stands. All six
  `dl/equiv.dl` checks still fail and are still retained as failures.

## The transferable lesson

A prior that is uniform on the set you care about is not a weak signal — it is an
absent one, and it is *indistinguishable from its own shuffled control by
construction*. Any trace prior should therefore be checked for **label
variation on the evaluation population before it is used to rank anything**, and
that check should be a gate rather than an observation. A pre-run assertion that
the signature is non-degenerate would have caught this before collection.

Run A was faithfully executed against an unchanged preregistration whose feature
design could not have produced a discriminating prior on a negative-bearing
population. The failure is in the design, and it is diagnosable from the frozen
evidence. The correction is registered in advance in
[`native-trace-prereg2.md`](native-trace-prereg2.md).

## Reproducing these claims

No model required; all of it is retained evidence.

```bash
# selection population has no negative controls
python3 -c "import json;d=json.load(open('reference/benchmarks/qwen25_native_trace/discovery.json'));print(sum('negative' in r for r in d),'/',len(d))"

# evaluation splits do
python3 -c "import json;d=json.load(open('reference/benchmarks/qwen25_native_trace/dataset.json'));print({p:(len(v),sum(r['negative'] for r in v)) for p in ('train','validation','test') for v in [[r for r in d if r['part']==p]]})"

# the 7-instance negative class
python3 -c "import json;D='reference/benchmarks/qwen25_native_trace/';f=json.load(open(D+'discovery-predictions.json'))['relations'];r={int(k):int(v) for k,v in json.load(open(D+'discovery-references.json')).items()};c={int(a):int(b) for a,b in f['cdecide']};print(sorted(i for i in r if c[i]!=r[i]))"

# trace_good covers the negative controls
python3 -c "import json;D='reference/benchmarks/qwen25_native_trace/';t={int(x[0]) for x in json.load(open(D+'selection.json'))['relations']['trace_good']};d=json.load(open(D+'dataset.json'));n=[r['id'] for r in d if r['part']=='train' and r['negative']];print(sum(i in t for i in n),'/',len(n))"

# trace == shuffled, elementwise
python3 -c "import json;s=json.load(open('reference/benchmarks/qwen25_native_trace/selection.json'))['relations']['score'];a={};[a.setdefault(str(r[0]),{}) .__setitem__(r[1],r[2]) for r in s];print(a['1']==a['2'])"
```