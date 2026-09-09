# CPU speech contracts

`internal/speechrecognition` shares acoustic blocks and `SpeechLease` across
recognition, alignment and diarization. Recipes declare tensor roles and geometry;
no model-family dispatch. Numeric arena limits are not whole-process RSS limits.

`compatibility.json` owns acceptance scope. `transducer_captures.json` binds
Transformers 5.16.1 reference identities; `registered_models.json` owns model
revisions and licenses. Python is not a product dependency.

## Streaming recognition

Raw overlap, one-time padding and post-transform incomplete-hop masking feed
incremental decoding without prefix replay. Offline and streaming have separate
goldens. Source/recipe-bound checkpoints preserve lossless binary32 state,
progress and incomplete UTF-8. Invalid/final states cannot resume; frame advances
are not word timestamps.

Authenticated `/v1/audio/transcriptions` accepts `application/x-ndjson`, an
extension distinct from OpenAI multipart; multipart `stream=true` is unsupported.
The opening line binds `model`, `source` (audio/profile IDs) and optional `resume`.
`AudioStreamChunk` lines provide sequence, span, audio, discontinuity and final.
Artifact chunks require inspection; empty final flushes have cursor-local empty
spans. SSE created/token/done/error events carry suffixes, advances and artifact
IDs. Publication precedes further input. Resume restores checkpoint/sequence;
discontinuity resets numerics, not coordinates. Disconnect releases the lease;
incomplete calls cannot establish a resumable boundary.

## Analysis evaluation

`cmd/evaluate -alignment-manifest <json> -repo <store>` accepts base_recipe,
profile, memory_bytes, inspection and inputs. Inputs bind AudioPayloadReference,
a transcription/span request and reference alignment. Standalone CTC uses pooled
hop cells, whitespace-token spans and excluded blanks. Confidence is geometric
mean target probability, not correctness. Unsupported token mappings,
adapted/recurrent recipes and source mismatches refuse. `ctc_alignment.json`
pins the Apache-2.0 audio.cpp recurrence; `ami_alignment.json` owns corpus selection.

`-diarization-manifest` accepts recipe or bound profile, memory_bytes, inspection and inputs
(audio, reference, span; optional alignment/activity IDs). Declared acoustic and
post-normalized attention operations produce recording-local speaker turns.
Float32 pre-emphasis/statistics are explicit; DFT/filterbank accumulation remains
float64. Integer-grid boundaries clip to actual audio.

Both commands retain every attempt and fail on any failed case, without activation.
Diarization reports zero-collar overlap-inclusive speaker-time error and directional
nearest-boundary distances with unmatched counts. Word composition preserves every
aligned word, timing and confidence; all overlapping labels remain, including
ambiguity or no attribution. Optional VAD limits contributing samples, not words.

`speaker_reference.json` pins the C++ capture driver and 156 captures. The
reference weights are **CC-BY-NC-4.0** and include AMI training. AMI audio/annotations
are CC-BY-4.0; word timing is forced-alignment-derived and manual segments include
pauses. Conformance is not held-out quality, commercial readiness, HTTP analysis
or GPU evidence. `OVERGO_AUDIO_SPEAKER_REFERENCE` locates the fixture;
`OVERGO_AUDIO_ALIGNMENT_ANNOTATIONS` locates its pinned archive.

## Serving and switching

`cmd/server` resolves an active transcription model by identity, location or
unique catalog name without GGUF. Inference activations retain text serving.
`-transcription-policy` or store-local `transcription_policy.json` supplies
memory_bytes and inspection; omitted recipe follows activation, mismatches refuse.
The swap proxy owns startup, draining and cancellation. Unknown/ambiguous targets
refuse; canceled swaps preserve requests. Fresh children resume checkpoints;
recipe retirement and reverified activation own rollback.
