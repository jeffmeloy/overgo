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

Waveform streaming applies declared left padding once and withholds right
padding until finalization. A raw overlap tail preserves waveform preemphasis;
the declared incomplete-hop mask applies after feature transforms. Exact native
first/following mel geometry and final padding feed the common recurrent decoder.
No preceding waveform or encoder prefix is replayed. Offline and native streaming
final text may differ; each is checked against its own captured reference.

Completed checkpoints bind the recipe and source, including their model,
frontend, tokenizer and stream-policy dependencies. They retain bounded pending
features, subsampling tails, key/value and convolution history, LSTM hidden/cell
state, cached prediction, progress and at most one incomplete UTF-8 rune.
Numerical arrays use lossless binary32 payloads. Invalid shapes, numbers,
identities, progress or final states cannot resume. Frame advances are not word
timestamps. CTC adapter training remains separate from recurrent inference.

## Live transcription protocol

The authenticated `/v1/audio/transcriptions` route accepts a native NDJSON
extension with `Content-Type: application/x-ndjson`; it is not the OpenAI
multipart request format. Multipart `stream=true` remains unsupported.

The first line supplies `model`, `source` (`audio` and stream-policy `profile`
artifact IDs), and optional `resume` (a completed cursor checkpoint ID).
Subsequent lines use `workflowruntime.AudioStreamChunk`: `sequence`, `span`,
`audio`, optional `discontinuity`, and `final`. Encoded chunks must already exist
in the artifact store, through `/artifacts/intake` or a native producer. Each
nonempty chunk must satisfy the configured transcription inspection policy;
thresholds are not silently changed for streaming. An empty final flush has no
audio artifact and an empty span at the current cursor.

SSE replies use `created`, `token`, `done`, and `error`. A token or done event
contains a text suffix, token/frame advances, output identity, admission decision
and a durable cursor checkpoint. Publish and acknowledge precede the next chunk.
Reconnect with that checkpoint and the next sequence; discontinuity resets
numerical state without rewinding source coordinates. Backpressure has no
application queue. Disconnect closes the body and releases the existing model
lease; incomplete calls earn no resumable boundary.
