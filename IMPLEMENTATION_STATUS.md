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
| Server | Implemented, expanding | Native, OpenAI Chat/Responses, Anthropic text/tools, streaming, fused batching, slots, metrics, LoRA |
| CUDA | Implemented on Windows | Dynamic Driver API, cuBLAS, embedded PTX, persistent/native-quantized paths |
| Release | Implemented | Reproducible Windows-amd64 archive, SBOM, kernel ABI manifest |

## Recently completed

- Gemma 4 and Qwen3-VL projector CUDA offload.
- Encoded video input through native GIF decoding or FFmpeg.
- Ordered multi-image native, Chat Completions, and Responses prompts.
- Projection-backed multimodal input-token counting.
- Bounded multimodal bodies, decoded media, image dimensions, and pixels.
- Shared native, Chat, and Responses prompt preparation and token counting.
- Non-mutating Go formatting verification in local and CI gates.
- Full-repository vet gate and CGO race CI for server/inference.
- Split oversized server/model/CUDA tests with shared CUDA test setup.
- Protocol-partitioned Chat, Responses, and Anthropic server implementation.
- Token-incremental JSON and Hermes tool-call argument streaming.
- Central architecture registry with family/capability profiles.
- Embedded common, attention, MoE, recurrent, encoder, and multimodal specs.
- Family-routed graph construction and tensor-catalog validation.
- Paged retained-device caches and dynamic multi-sequence cache batches.
- Fused variable-sequence CUDA graphs and continuous HTTP generation scheduling.
- Padded multi-sequence T5 encoder/decoder batches with explicit lengths.
- Gemma 4 safetensors/ModelOpt-to-GGUF conversion.
- Cohere2-MoE, Step3.5, and HY-V3 MTP execution paths.

## Active roadmap

No unblocked implementation items remain. Deferred work below requires an
external fixture, policy, certificate, or model/projector contract.

## Deferred or externally blocked

| Item | Reason | Resume condition |
| --- | --- | --- |
| Real-model validation for marked architectures | Local GGUF/oracle fixture unavailable | Fixture supplied or generated |
| Release signing | No user-controlled signing certificate | Certificate and signing policy supplied |
| ECMAScript regex lookaround/backreferences | Go RE2 does not implement them | Compatible bounded engine selected |
| Remote media URLs | Fetch policy and SSRF boundary undefined | Explicit allowlist, size, redirect, and timeout policy |
| Media history | Model/projector-specific multi-turn prompt ordering and cache semantics lack an upstream oracle | Validated multi-turn template, projector, and cache fixture supplied |
| Mixed image/audio turns | No upstream-validated projector prompt contract covers interleaved modalities | Mixed-modality projector contract and oracle fixture supplied |

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
