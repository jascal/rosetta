# Implementation and run notes before final evaluation

The preregistration is unchanged. Implementation fixes before the final freeze
included Datalog reserved-name corrections, reporting-only selection outputs,
and observation-cost aggregation. Initial protocol source hashes describe the
initial implementation; `frozen.json` records the actual evaluated sources.

Selection ran once for admission. A second query added reporting relations
(`arm_summary`, `trace_summary`, `rejection`, `failure_detail`, `trace_good`). Its
proposed/admitted/score/fails/support outputs match the original byte-for-byte.
Both query evidence directories are retained. The second driver invocation
refused to recreate existing artifact directories, as designed; the original
three empty-selection artifacts remain intact. No admission or ranking change
was made in response to development data.

The model-conversion executable digest was not captured at conversion time.
The converted f32 bundle, actual probe executable, checkpoint files, source
revision, and upstream comparison are pinned. This provenance limitation is
explicit in `native_trace_preflight/build.json`; numerical validation supports
this bundle, not an unrecorded converter identity.

The diagnostic selects eight correct and seven incorrect discovery cases:
only seven incorrect cases exist. All available opposite-class rank pairs are
used, giving 135 target zero patches, 135 adjacent-neuron zero controls,
135 identity patches, and 126 opposite-class swaps. No eighth incorrect case
was invented. These interventions do not affect admission.
