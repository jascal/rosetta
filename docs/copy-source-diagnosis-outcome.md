# Source-resolved diagnosis: both routing and output transformation fail

**Empirical, historical-development diagnostics**, no new candidate rule or
final-test references. This follows the failed long-context family; it does not
reclassify any failed certificate as successful.

Datalog classified all 648 positive development contexts:

| Recorded model answer | Cases |
|---|---:|
| intended long-source successor | 526 |
| another exact suffix source's successor | 65 |
| another input token, not a suffix-source successor | 13 |
| token absent from the context | 44 |

These are observable answer categories. An answer matching a competing token
does not by itself prove that the model read that source; an absent answer
does not by itself prove that it transformed the correct source.

## Native observations

The fixed [sampling plan](copy-source-diagnosis-plan.md) chose two independent
groups per class where available, giving seven cases (one group for the
other-input class). All selected cases are from the earliest train groups at
source length 24 / prefix 12. This is a deliberately small diagnostic sample,
not a representative evaluation of the longer-prefix failures.

The pinned executable's existing `/explain` endpoint reproduced all seven
recorded model answers. Datalog checked every reported peak position/token
against the actual input. Native explanations are prefiltered: only six shown
heads and six neurons are returned. They are not complete source decompositions.

- **Correct cases 0 and 38:** heads layer16/head9 and layer16/head12 both peak
  at the intended source token and put the gold token first in their promoted
  lists (` Ralph` and ` presum`, respectively).
- **Wrong-source cases 14 and 62:** most shown peaks land on the competing
  successor (`_STOP` and ` dread`). This is consistent with wrong-source
  routing, while remaining observational.
- **Absent-output case 13:** every shown peak is at competing `_STOP`, while
  the model emits ` STOP`. The shown promotions include related stop tokens,
  but none contains the actual answer in its reported top five. This combines
  a competing-source pattern with a changed output; it does not localize the
  final token's write to one shown head.
- **Absent-output case 76:** four shown heads peak at the intended `\ty`
  source and promote actual ` y`. Layer16/head12 promotes ` y` first and the
  literal gold `\ty` third. Thus a literal source pointer can be correct while
  the observed output projection favors another token.
- **Other-input case 77:** the reported reads are mixed, including intended
  ` })}\n`, competing ` cond`, and the actual ` mutual` at another position.
  This case does not fit a single clean copy-pointer explanation.

Token strings above are decoded displays of recorded IDs, not a new string
normalization rule. Datalog operates on the actual IDs and reported positions.

## Implication for extraction

Input-shape guards and literal source equality cannot by themselves express
both observed error paths. The next candidate computation needs source routing
and the output transformation considered separately. Layer16/head9 and
layer16/head12 are concrete **hypotheses for causal testing**, not established
necessary or sufficient mechanisms. Earlier trace experiments already showed
why attribution alone must not be promoted into an extracted circuit.

A future causal plan should freeze treated and layer-matched control head sets
before interventions, preserve identity/unpatched parity, and state whether
ablation affects all positions or just the answer. A repaired output mapping
also needs a general rule derived before final references, rather than a lookup
over these seven known answers. The standalone runtime and positive incremental
finite-domain certification requirements remain unchanged and **open**.

## Evidence and validation

`reference/benchmarks/qwen25_copy_source_diagnosis/` retains the full
classification, frozen sample, raw native explanations, executable/bundle
provenance, and Datalog consistency query. `certified()` here means instrument
consistency only. There is no new `dl/equiv.dl` success.

Two synthetic tests check disjoint answer classes and correct source/position
and promotion bookkeeping, including rejection of a fabricated read token.
These tests validate the diagnostic, not an LLM mechanism.
