# Guard search control: deterministic copy oracle

**Proved on each stated finite test domain.** We ran the original six-candidate
guard family and the expanded 48-candidate family on the same seed-33
copy-control dataset. In this control, the oracle answer is defined to be the
next token after the latest matching source, so the intended rule is
deterministic by construction. Selection used only development examples.

| Search family | Selected guard `(suffix, support)` | Test firing coverage | Exact certificate on firing domain |
|---|---:|---:|---|
| Original six | `(3, 2)` | 96/144 (66.7%) | certified, 0 misses |
| Expanded 48 | `(1, 1)` | 144/144 (100%) | certified, 0 misses |

The expanded search recovers the simple copy rule on all 144 held-out positive
cases and abstains on all 16 negative controls in this authored control
domain. The original restricted grid also finds an exact subdomain, but leaves
48 positive cases uncovered. This shows
that the Datalog runtime and certificate can represent and prove the intended
copy behavior here, and that the additional candidates improve coverage when
the oracle follows the rule consistently.

This does not prove that the language is sufficient for arbitrary LLM
behavior. The MiMo oracles differ from the copy-control rule on development
cases, and the expanded family admits no rule for them. That gap is consistent
with a behavior mismatch plus incomplete candidate concepts; it does not
identify one universal cause.

Both complete benchmark reports, frozen candidates, firing domains, and
Soufflé certificate evidence are in [`search_control_original/`](search_control_original/)
and [`search_control_expanded/`](search_control_expanded/).
