# CPU recurrent transducer contract

`internal/speechrecognition` reuses acoustic blocks for CTC and recurrent
transduction. Declarations supply causal spatial convolution, relative-key
projection, position limits, conditioning, LSTM bindings and greedy emission
limits. Runtime code does not dispatch on model names. Numeric storage limits
cover weights and execution state, not process RSS.

The behavioral reference is Transformers 5.16.1, Apache-2.0, copyright 2026
The HuggingFace Inc. team, using its native `nemotron3_5_asr`,
`nemotron_asr_streaming` and `parakeet` implementations. The Go implementation
re-expresses those operations; Python is not a product dependency. Source-file,
runtime-library, model, corpus and capture-script hashes are bound to the
OvergoDB documents selected by `internal/audioparity/testdata/transducer_captures.json`.
Model revision and separate weight-license bindings remain in
`internal/audioparity/testdata/registered_models.json`.

Only finite native CPU SDPA captures are admitted. The eager probe produced
nonfinite results and provides no successful parity evidence. Exact acceptance
commands and claim boundaries live in `compatibility.json`.

Streaming accepts declared mel chunks with explicit final padding, not growing
offline prefixes. Failed/finalized workspaces cannot resume. Frame advances
locate token emissions; they do not establish word alignment. Raw-wave streaming,
persisted restart and authenticated transport remain separate consumer work.
