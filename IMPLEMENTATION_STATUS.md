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
| Multimodal | Implemented, constrained | Gemma 3n, Granite 4 Vision, Hunyuan-VL, Llama 4, MiMo-VL, PaddleOCR-VL, Qwen2-VL, dense/MoE Qwen3-VL, and Gemma 4 image/video; Gemma 4 ordered image/audio history |
| Server | Implemented, expanding | Native, OpenAI Chat/Responses, Anthropic text/tools, streaming, fused batching, slots, metrics, LoRA |
| CUDA | Implemented on Windows | Dynamic Driver API, cuBLAS, embedded PTX, persistent/native-quantized paths |
| Release | Implemented | Reproducible Windows-amd64 archive, SBOM, kernel ABI manifest |

## Recently completed

- Gemma 3n MobileNetV5 image projection with multi-scale fusion, 256 soft tokens, native history prompts, and CUDA differential.
- MiMo-VL dynamic preprocessing, GQA ViT with row/column window attention and sinks, native image prompts, and CUDA differential.
- Granite 4 Vision overview/grid tiling, SigLIP ViT, window QFormer, base/deepstack prompt contract, and CUDA differential.
- Llama 4 UHD tiling, class/learned-position ViT, two-axis vision RoPE, pixel-shuffle adapter, multi-image prompt contract, and CUDA differential.
- Hunyuan-VL dynamic image encoder, convolutional projector, multi-image prompt/grid contract, and CUDA differential.
- Gemma 4, PaddleOCR-VL, and Qwen3-VL projector CUDA offload.
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
- Pinned Gemma 4 multi-turn image history and media-signed prompt caching.
- Pinned mixed image/audio chunks and policy-bounded remote media fetching.
- Cohere2-MoE, Step3.5, and HY-V3 MTP execution paths.
- Bounded Responses continuation history with stable function-call IDs.
- Bounded Chat/Responses encoded-video input through native GIF or FFmpeg decode.
- Responses reasoning summary input/output, streaming events, and continuation replay.
- Bounded Responses inline, mapped file-ID, remote text-file, and image file-ID inputs.
- Explicit deny policy for Responses hosted, MCP, and free-form custom tools.
- Projected image/audio history combined with function tools across OpenAI protocols.
- Anthropic base64/URL/file-ID images with projection, tools, and token counting.
- Anthropic manual summarized thinking with SSE, projected images, and handler-local signed replay.
- Bounded ECMAScript lazy-GBNF triggers with lookaround and backreferences.
- Pinned Jinja globals for bounded exceptions, dates, namespaces, and ranges.
- Fused recurrent/hybrid continuous batching with named state and device forks.

## Active roadmap

Work proceeds in dependency order. A blocked item does not stop later rows.

| Priority | Item | Completion boundary |
| --- | --- | --- |
| 1 | Multimodal model gaps | Internal encoders/projectors/grid construction for models currently requiring external projected inputs |
| 2 | Speculative, audio, and adapters | Integrated draft verification, WavTokenizer waveform output, and broader aLoRA semantics |
| 3 | Platform and release breadth | Verified targets beyond the Windows-amd64 CUDA baseline |
| 4 | Pinned-upstream closure audit | Every upstream architecture, endpoint, option, and negative path implemented or explicitly classified |

## Known implementation boundaries

| Area | Current boundary |
| --- | --- |
| Responses | Bounded continuation, reasoning summaries, and typed text/image files; no PDF/non-text documents, hosted/custom tools, encrypted reasoning, or reasoning-with-tools |
| Multimodal server | Supported image/audio media may use function tools; video remains single-turn/non-mixed and cannot use tools |
| Anthropic | Manual summarized thinking uses handler-local signatures; adaptive/omitted/redacted/interleaved modes and thinking with tools are explicit exclusions |
| Architectures | Declared model graphs execute; marked real-model fixture validation remains in `compatibility.yaml` |
| Vision | Several decoder graphs require external encoders, projectors, or grid coordinates |
| Speculation | Some draft graphs leave verification and acceptance to an external coordinator |
| Audio | WavTokenizer stops at feature frames; waveform postprocessing is absent |
| Adapters | One aLoRA may be active; non-causal diffusion rejects aLoRA |
| Platforms | Primary supported release remains Windows amd64 with NVIDIA CUDA |

## Deferred or externally blocked

| Item | Reason | Resume condition |
| --- | --- | --- |
| Real-model validation for marked architectures | Local GGUF/oracle fixture unavailable | Fixture supplied or generated |
| Release signing | No user-controlled signing certificate | Certificate and signing policy supplied |
| Hosted-tool integration | No repository-wide hosted executor contract | Local bounded defaults implemented or external contract supplied |
| Real speculative model pairs | Matching target/draft fixtures unavailable | Compatible pair supplied or generated |

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
