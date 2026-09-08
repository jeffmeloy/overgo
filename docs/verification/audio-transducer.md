# CPU recognition and alignment contracts

`internal/speechrecognition` shares CTC/recurrent acoustic blocks. Recipes own
geometry, bindings and emission limits; no model-name dispatch. Numeric budgets
cover weights/execution state, not RSS.

Reference: Transformers 5.16.1, Apache-2.0, copyright 2026 The HuggingFace Inc.
team. `transducer_captures.json` binds source/library/model/corpus hashes;
`registered_models.json` owns revisions and weight licenses. Python is not a
product dependency.

Finite CPU SDPA captures alone earn parity credit. `compatibility.json` owns
acceptance commands and claim boundaries.

Streaming retains raw overlap, applies boundary padding once and masks incomplete
hops after transforms. Declared mel geometry feeds the decoder without prefix
replay. Offline and streaming goldens remain separate.

Recipe/source-bound checkpoints retain bounded features, acoustic/recurrent state,
progress and one incomplete UTF-8 rune; binary32 arrays are lossless. Invalid or
final states cannot resume. Frame advances are not word timestamps.

## Forced word alignment

`cmd/evaluate -alignment-manifest <json> -repo <store>` declares `base_recipe`,
`profile`, `memory_bytes`, `inspection`, and `inputs`. Each input binds an
`AudioPayloadReference`, an alignment `request` (transcription/half-open span),
and a `reference` alignment artifact. Audio stays in place. The report retains
every attempt; any failure exits nonzero. No activation is performed.

`AlignmentDefinition`/`TranscriptionLease.Align` reuse standalone CTC. The profile
declares pooled hop cells, whitespace-token spans and excluded blanks. Confidence
is mean target-frame probability geometrically, not correctness. Unsupported
token mappings, adapted/recurrent recipes and mismatched sources refuse.

The Apache-2.0 recurrence pins audio.cpp
`3497b7cc44753e2c141d8fe60ac42cec433e3281` and exact ties.
`ctc_alignment.json` owns oracle cases; `ami_alignment.json` owns corpus selection.
The [AMI annotation archive](https://groups.inf.ed.ac.uk/ami/download/) is CC-BY-4.0.
Its forced-alignment-derived timing is not human boundary truth. Every selected
boundary is scored; precision and exclusions remain explicit.
`OVERGO_AUDIO_ALIGNMENT_ANNOTATIONS` may locate the exact pinned archive.
No HTTP alignment, diarization, recurrent training or GPU claim follows.

## Native serving and switching

`cmd/server` accepts an active transcription model's identity, location or unique
catalog name without a GGUF inference model. An inference activation retains the
existing text-serving path. `-transcription-policy`, or `transcription_policy.json`
beside the store, must supply `memory_bytes` and `inspection`. An omitted `recipe`
binds the selected model's active recipe; a mismatched pin refuses startup.

The swap proxy owns startup, draining and cancellation; unknown/ambiguous targets
refuse. Canceled swaps preserve active requests. Fresh children resume published
checkpoints; modelrecipe retirement and reverified activation own rollback.

## Live transcription protocol

Authenticated `/v1/audio/transcriptions` accepts `Content-Type: application/x-ndjson`;
this extension differs from OpenAI multipart. Multipart `stream=true` is unsupported.

The opening line supplies `model`, `source` (`audio`/stream-policy `profile` IDs)
and optional `resume` checkpoint. Following `AudioStreamChunk` lines supply
`sequence`, `span`, `audio`, optional `discontinuity`, and `final`. Artifact-backed
chunks must satisfy the configured inspection policy. Empty final flushes have
no audio and an empty span at the cursor.

SSE `created`/`token`/`done`/`error` events carry suffixes, advances and artifact IDs.
Publication precedes the next chunk. Resume uses checkpoint/next sequence;
discontinuity resets numerical state, not coordinates. Disconnect releases the
lease; incomplete calls earn no resumable boundary.
