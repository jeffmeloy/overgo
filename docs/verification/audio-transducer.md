# CPU recurrent transducer contract

`internal/speechrecognition` shares acoustic blocks between CTC and recurrent
transduction. Recipes declare geometry, relative positions, conditioning, LSTM
bindings and emission limits; runtime code never dispatches on model names.
Numeric budgets cover weights and execution state, not process RSS.

Reference: Transformers 5.16.1 (`nemotron3_5_asr`, `nemotron_asr_streaming`,
`parakeet`), Apache-2.0, copyright 2026 The HuggingFace Inc. team. Python is
not a product dependency. Source, library, model, corpus and capture hashes bind
the OvergoDB documents in `internal/audioparity/testdata/transducer_captures.json`;
`registered_models.json` owns revisions and separate weight licenses.

Only finite CPU SDPA captures are admitted; eager nonfinite results earn no
parity credit. `compatibility.json` owns acceptance commands and claim boundaries.

Streaming applies left padding once, right padding only at finalization,
preemphasis through retained raw overlap, and incomplete-hop masking after
feature transforms. Declared first/following mel geometry feeds the recurrent
decoder without prefix replay. Offline and streaming text have separate goldens.

Checkpoints bind recipe/source dependencies and retain bounded features,
subsampling tails, attention/convolution history, LSTM state, cached prediction,
progress and at most one incomplete UTF-8 rune. Binary32 arrays are lossless.
Invalid shapes, numbers, identities, progress or final states cannot resume.
Frame advances are not word timestamps. Recurrent training is not claimed.

## Native serving and switching

`cmd/server` accepts an active transcription model's identity, location or unique
catalog name without a GGUF inference model. An inference activation retains the
existing text-serving path. `-transcription-policy`, or `transcription_policy.json`
beside the store, must supply `memory_bytes` and `inspection`. An omitted `recipe`
binds the selected model's active recipe; a mismatched pin refuses startup.

The swap proxy owns startup, draining and cancellation. Explicit
unknown swap targets and ambiguous catalog aliases refuse. NDJSON routing reads
only the opening record. A canceled pending swap preserves active requests;
fresh children resume published checkpoints. Shared modelrecipe lifecycle
retirement and reverified predecessor activation own rollback.

## Live transcription protocol

Authenticated `/v1/audio/transcriptions` accepts `Content-Type: application/x-ndjson`;
this extension differs from OpenAI multipart. Multipart `stream=true` is unsupported.

The first line supplies `model` (recipe/model identity or bound served alias),
`source` (`audio` and stream-policy `profile`
artifact IDs), and optional `resume` (a completed cursor checkpoint ID).
Subsequent lines use `workflowruntime.AudioStreamChunk`: `sequence`, `span`,
`audio`, optional `discontinuity`, and `final`. Encoded chunks must already exist
in the artifact store, through `/artifacts/intake` or a native producer. Each
nonempty chunk must satisfy the configured transcription inspection policy;
thresholds are not silently changed for streaming. An empty final flush has no
audio artifact and an empty span at the current cursor.

SSE `created`, `token`, `done`, and `error` events carry text suffixes, token/frame
advances and output/admission/checkpoint identities. Publication precedes the
next chunk. Reconnect with the checkpoint and next sequence; discontinuity resets
numerical state, not source coordinates. No application queue is added.
Disconnect releases the model lease; incomplete calls earn no resumable boundary.
