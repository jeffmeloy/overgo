# Implementation status

This file is the live roadmap. Detailed completed-work history is archived in
[`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md). The generated model
and feature matrix is in [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md), with
[`compatibility.yaml`](compatibility.yaml) as its machine-readable source.

## Baseline

- llama.cpp: `42fc243060709331ff9b158a9ed2cbe37219ae83`
- primary host: Windows amd64
- primary device: NVIDIA compute capability 8.9
- Go: 1.26, cgo disabled
- verification: `scripts/verify.ps1`; add `-CUDA` for device integration

## Current state

| Area | State | Current boundary |
| --- | --- | --- |
| GGUF | Implemented | Bounded v2/v3 read/write, split discovery/creation/merge, hashing, streaming quantization |
| Tokenization | Implemented | BPE, SPM, WordPiece, and UGM families with pinned oracle corpora |
| Tensor graph | Implemented, expanding | Reference and CUDA execution for dense, MoE, recurrent, diffusion, encoder, and multimodal primitives |
| Model runtime | Experimental breadth | Architecture-specific metadata, catalogs, graphs, cache state, and optional real-model oracles |
| Generation | Implemented | Cached autoregressive, diffusion, encoder-decoder, embeddings, reranking, and speculative paths |
| Multimodal | Implemented, constrained | Qwen3-VL and Gemma 4 image/video; Gemma 4 audio; single-turn server media |
| Server | Implemented, expanding | Native, OpenAI Chat/Responses, Anthropic text/tools, streaming, slots, metrics, LoRA |
| CUDA | Implemented on Windows | Dynamic Driver API, cuBLAS, embedded PTX, persistent/native-quantized paths |
| Release | Implemented | Reproducible Windows-amd64 archive, SBOM, kernel ABI manifest |

## Recently completed

- Gemma 4 and Qwen3-VL projector CUDA offload.
- Encoded video input through native GIF decoding or FFmpeg.
- Ordered multi-image native, Chat Completions, and Responses prompts.
- Projection-backed multimodal input-token counting.
- Gemma 4 safetensors/ModelOpt-to-GGUF conversion.
- Cohere2-MoE, Step3.5, and HY-V3 MTP execution paths.

## Active roadmap

### Runtime architecture

1. Replace scattered architecture string tests with a central architecture
   registry and capability profile.
2. Split the flat model specification into common, attention, MoE, recurrent,
   encoder, and multimodal sub-specifications while preserving GGUF behavior.
3. Partition model graph, catalog, and weight dispatch by architecture family.
4. Add device paging, multiple sequences, and continuous-batching cache
   semantics.
5. Add padding masks for padded multi-sequence T5 batches.

### Server and multimodal

1. Bound decoded image dimensions before allocation; separate JSON and media
   request budgets.
2. Normalize native, Chat, and Responses requests through one prepared-prompt
   pipeline shared by generation and token counting.
3. Split the server implementation by protocol while retaining one handler and
   shared admission/metrics state.
4. Add media history after model-specific multi-turn template and cache
   semantics are defined.
5. Add mixed image/audio turns only where the projector prompt contract is
   validated by an upstream oracle.
6. Add token-incremental tool-call argument streaming.

### Verification and maintenance

1. Make formatting verification non-mutating and enforce it in CI.
2. Add a vet gate after isolating the intentional CUDA C-pointer diagnostic.
3. Add a CGO-enabled race job for server and inference packages.
4. Split oversized model/server/CUDA test files and centralize CUDA test setup.
5. Keep implemented claims in `compatibility.yaml`; regenerate and check
   `docs/COMPATIBILITY.md` in local verification and CI.

## Deferred or externally blocked

| Item | Reason | Resume condition |
| --- | --- | --- |
| Real-model validation for marked architectures | Local GGUF/oracle fixture unavailable | Fixture supplied or generated |
| Release signing | No user-controlled signing certificate | Certificate and signing policy supplied |
| Standard Go race detector in normal builds | Runtime contract uses `CGO_ENABLED=0` | Separate CGO-enabled verification job |
| Full repository vet | CUDA driver returns process-local C pointers as `uintptr` | Isolate or validate the interop diagnostic |
| ECMAScript regex lookaround/backreferences | Go RE2 does not implement them | Compatible bounded engine selected |
| Remote media URLs | Fetch policy and SSRF boundary undefined | Explicit allowlist, size, redirect, and timeout policy |

Blocked fixture work does not stop independent implementation work.

## Completion rules

- Unsupported combinations fail explicitly.
- Model metadata and tensor catalogs validate before execution.
- Reference semantics precede CUDA optimization.
- CPU/CUDA differentials precede real-model performance claims.
- Real-model claims name their fixture or remain marked pending.
- Compatibility changes update `compatibility.yaml` and regenerate its matrix.
- Completed detail moves to `docs/IMPLEMENTATION_LOG.md`; this file stays
  concise and current.
