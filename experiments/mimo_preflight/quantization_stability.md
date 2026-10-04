# MiMo-7B quantization stability on the frozen parity suite

**Empirical, 32-context domain.** Fieldrun f16 matched the official FP32
reference on 32/32 contexts for Base and RL. Fieldrun int8 matched Base on
32/32 and RL on 29/32. The three RL int8 top-1 flips all occur on graph
contexts; RL f16 matches the official FP32 answer on each. Therefore RL's
int8-vs-f16 top-1 difference on this suite is 3/32, concentrated in one
behavioral family. Base int8/f16 both agree with the official FP32 top-1 on
this domain.

The separate official-reference dtype sweep on eight contexts found substantial
native arithmetic sensitivity: official BF16 vs official FP32 matched 4/8;
official FP16 vs FP32 matched 6/8; the fieldrun f16 bundle matched official
FP32 on 8/8. Native BF16 is a distinct execution condition from fieldrun's
quantized bundles.

On the stratified 40-case recursive graph subset, RL int8 and f16 answers flip
on 1/40 cases; both remain far from the reachability rule on that subset
(18/40 and 19/40 mismatches, respectively). This argues against int8 alone
causing the Base/RL graph difference in that small sample.

This is not the full requested f32/f16/int8/int4 circuit-selection study.
No int4 bundle or circuit certificate by quantization is recorded. The RL
guarded-copy comparison remains int8 and should be rerun with f16 before a
strong claim about training-induced copy changes. Current artifacts and
margin data are in the [preflight folder](README.md).
