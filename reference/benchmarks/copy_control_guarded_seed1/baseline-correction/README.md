# Lexical baseline emitter correction

Post-test implementation correction using recorded references. This is not a fresh holdout experiment.
The original frozen artifacts and guard certificates are unchanged. The emitter now binds each instance
before computing its final token position, so mixed-length batches retain shorter contexts.

Train: 72/72 correct; exact certificate: True.
Test: 0/144 answered. See `report.json` for all counts and retained evidence.
