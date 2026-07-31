# llama.cpp to Go/CUDA port plan

## 1. Executive decision

This should be built as a clean-room-compatible reimplementation, not as a mechanical translation of C++ into Go.

The recommended target is:

- all host-side runtime, model loading, tensor graph construction, scheduling, tokenization, sampling, state management, CLI, and server code written in Go;
- no cgo in the shipped executable;
- direct dynamic calls from Go to NVIDIA's Windows DLLs;
- CUDA kernels shipped as compiled PTX/cubin/fatbin assets and loaded through the CUDA Driver API;
- upstream llama.cpp used only as a pinned behavioral and performance oracle during development.

A literal "100 percent Go source" implementation is not compatible with CUDA execution. NVIDIA executes PTX/SASS kernels and exposes native driver/library DLLs. The practical and auditable definition of "entirely Go" is therefore:

> All product logic and host code is Go. The only non-Go runtime dependencies are NVIDIA's installed DLLs and generated GPU kernel binaries.

Do not make `llama.dll`, `ggml.dll`, or `ggml-cuda.dll` a permanent dependency. That would produce a Go wrapper, not a Go port.

## 2. Reviewed baseline

The review used this local source tree:

- path: `C:\Users\jeffm\llama.cpp`
- branch: `master`
- commit: `42fc243060709331ff9b158a9ed2cbe37219ae83`
- commit subject: `opencl: fix fused RMS norm mul view offset (#26085)`
- commit date: 2026-07-26
- worktree: clean

The target repository is:

- path: `C:\Users\jeffm\OneDrive\Documents\llamacpp2go`
- state: empty Git repository with no commits

The source tree is moving quickly. Every compatibility result and generated fixture must record the exact upstream commit.

## 3. What "entire llama.cpp" contains

The name llama.cpp hides several distinct products:

1. ggml tensor representation, graph construction, allocation, scheduling, and backend registry;
2. CPU and accelerator implementations of tensor operations;
3. GGUF parsing, writing, quantized data layouts, and model loading;
4. model architecture metadata, tensor naming, weight validation, and graph builders;
5. inference context, batching, KV/state caches, adapters, and state serialization;
6. vocabulary, Unicode processing, tokenizers, samplers, grammars, and chat templates;
7. public C API;
8. CLI, server, benchmarks, quantizer, perplexity, embedding, speculative decoding, and other tools;
9. Python conversion utilities and a TypeScript/Svelte server UI;
10. many non-CUDA backends.

For a CUDA-specific Go implementation, "full runtime parity" should mean items 1 through 8 for the CUDA path. Conversion scripts, the existing web UI, and non-CUDA backends should be tracked separately. Rewriting those does not unblock Go/CUDA inference and should not be on the critical path.

## 4. Review findings

### 4.1 Size and change surface

At the reviewed commit:

| Area | Approximate scale |
| --- | ---: |
| `src/`, including model implementations | 199 files, about 69k code lines |
| ggml base and all backends | 1,213 files, about 298k code lines |
| CUDA backend alone | 273 files, about 35.6k code lines |
| common CLI/server support | 66 code files, about 33k lines |
| server | 61 code files, about 24.3k lines |
| tests | 60 code files, about 37.4k lines |
| conversion and `gguf-py` | about 27.6k Python lines |

The public and semantic surface is also broad:

- 137 model architecture identifiers;
- 105 ggml operation identifiers;
- 44 ggml storage/data type identifiers;
- 87 operation cases in the main CUDA dispatch switch;
- 153 one-line public `LLAMA_API` function declarations, plus multiline declarations.

This is too large for a safe big-bang rewrite.

### 4.2 Architectural seams worth preserving

The current source has useful conceptual seams even though the Go representation should be idiomatic:

- `gguf.cpp`: container parsing and writing;
- `ggml.c`: tensor metadata and graph-building operations;
- `ggml-alloc.c`: graph allocation and liveness planning;
- `ggml-backend.cpp`: device/backend abstraction and multi-backend scheduler;
- `ggml-cuda.cu`: CUDA backend, buffer management, support checks, dispatch, and cuBLAS routing;
- `llama-model-loader.cpp`: file mapping, split files, metadata, and tensor loading;
- `llama-model.cpp`: model creation, placement, weight validation, and architecture factory;
- `llama-graph.cpp` plus `src/models/*.cpp`: reusable graph helpers and architecture-specific graphs;
- `llama-context.cpp`: batching, graph reservation, scheduling, execution, outputs, and state;
- `llama-kv-cache*.cpp` and `llama-memory*.cpp`: transformer and recurrent state;
- `llama-vocab.cpp`, `unicode*.cpp`: tokenization and text handling;
- `llama-sampler.cpp`, `llama-grammar.cpp`: decoding policy and constrained generation.

The Go port should preserve these responsibilities, but should not preserve C object layouts, manual ownership, inheritance, macro-generated APIs, or untyped operation parameter byte arrays where typed Go values are clearer.

### 4.3 CUDA is coupled to ggml internals

`ggml-cuda.dll` is not a standalone tensor library with a small stable ABI. The backend reads and writes ggml tensor structures, buffer structures, graph nodes, operation parameters, scheduler state, and backend interfaces.

Consequences:

- directly loading `ggml-cuda.dll` from a new Go tensor runtime would require reproducing fragile C/C++ layouts;
- using the public CUDA backend functions alone is insufficient to execute arbitrary Go-owned tensors;
- a permanent shim around ggml would preserve most of the C++ runtime and defeat the port;
- the final CUDA layer should call the NVIDIA Driver API and cuBLAS directly, using Go-owned descriptors.

### 4.4 Model coverage is the largest parity risk

The model directory is not 137 independent implementations of the same transformer. It includes:

- dense decoder transformers;
- mixture-of-experts models;
- encoders and embedding models;
- encoder-decoder models;
- recurrent RWKV models;
- Mamba/state-space models;
- hybrid attention/recurrent models;
- diffusion and multimodal variants;
- specialized attention, cache, speculative, and next-token mechanisms.

Porting one model file at a time will recreate duplication and take years. The Go design must identify reusable capability components, then express model families as composition and configuration.

### 4.5 Full parity is a continuing program

llama.cpp adds models, quantization layouts, graph operations, and CUDA kernels continuously. "Done" cannot mean "matches whatever is on upstream master forever."

Define releases against pinned upstream compatibility levels:

- format level: GGUF metadata and tensor layout compatibility;
- model level: named architecture and required quantization combinations;
- API level: selected llama.cpp behavior;
- server level: selected HTTP endpoints and fields;
- performance level: measured on named hardware and model fixtures.

## 5. Local machine and CUDA inventory

### Host

- OS: Windows 11 Pro 64-bit, build `10.0.26200`
- CPU: AMD Ryzen 9 9950X, 16 cores / 32 threads
- RAM: about 93.6 GiB
- Go: `go1.26.2 windows/amd64`
- current `CGO_ENABLED`: `0`

### GPU

- device: NVIDIA GeForce RTX 4090 D
- compute capability: 8.9
- VRAM: 49,140 MiB reported
- driver: 595.79
- driver DLL: `C:\Windows\System32\nvcuda.dll`

### CUDA toolkit

- toolkit: CUDA 12.9
- nvcc: 12.9.86
- toolkit root: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`
- available runtime libraries include:
  - `cudart64_12.dll`
  - `cublas64_12.dll`
  - `cublasLt64_12.dll`
  - `nvrtc64_120_0.dll`
  - `nvJitLink_120_0.dll`
- development import libraries and headers are installed.

### Native build tools

Visual Studio Community 2026 and the MSVC x64 compiler are installed. Visual Studio also contains CMake and Ninja, but `cmake`, `ninja`, `cl`, `gcc`, and `g++` are not currently on the ordinary PowerShell `PATH`.

This is sufficient for:

- compiling a pinned upstream CUDA oracle from a Developer PowerShell;
- compiling CUDA kernels for `sm_89`;
- building a no-cgo Go executable that loads NVIDIA DLLs dynamically.

No llama.cpp executables or ggml/llama shared libraries are currently on `PATH`, and no local llama.cpp build artifacts were found in the source checkout.

### Existing useful GGUF fixtures

The machine already contains models useful for later integration tests, including:

- `C:\Users\jeffm\Downloads\Qwen3.5-9B-Q8_0.gguf`
- `C:\Users\jeffm\Downloads\Zimage\ComfyUI_windows_portable\ComfyUI\models\text_encoders\Qwen3-4B-UD-Q8_K_XL.gguf`
- `C:\Users\jeffm\Downloads\Zimage\ComfyUI_windows_portable\ComfyUI\models\text_encoders\Qwen3-4B-UD-Q6_K_XL.gguf`
- `C:\Users\jeffm\Downloads\LTX2\ComfyUI_windows_portable\ComfyUI\models\text_encoders\gemma-3-12b-it-IQ4_XS.gguf`

These should not be checked into Git. Record hashes in a local test manifest and add small redistributable fixtures for CI.

## 6. Target architecture

### 6.1 Repository layout

Recommended initial layout:

```text
llamacpp2go/
  cmd/
    llama/
    server/
    bench/
    inspect-gguf/
  internal/
    cuda/
      driver/
      cublas/
      nvrtc/
      kernel/
      memory/
    tensor/
      dtype/
      graph/
      planner/
      reference/
    gguf/
    quant/
    model/
      registry/
      transformer/
      llama/
      qwen3/
    runtime/
      batch/
      cache/
      scheduler/
      session/
    vocab/
    tokenizer/
    sampler/
    grammar/
    chat/
    server/
  kernels/
    cuda/
    generated/
  testdata/
  tools/
    oracle/
    kernel-build/
    compatibility/
  go.mod
```

Use `internal/` until APIs stabilize. Prematurely exporting a llama.cpp-shaped Go API would freeze weak abstractions.

### 6.2 Core Go types

Use Go-owned, typed descriptors rather than reproducing `struct ggml_tensor`:

```go
type Tensor struct {
    Type    DType
    Shape   [4]int64
    Stride  [4]uint64
    Rank    uint8
    Storage Storage
    Offset  uint64
    Op      Op
    Inputs  []*Tensor
    Attr    any
}
```

Important rules:

- validate dimension products before multiplication;
- make views share storage explicitly;
- keep ownership on storage allocations, not individual views;
- represent operation attributes with typed structs;
- separate immutable weights from mutable activations and cache state;
- assign stable tensor IDs for tracing and differential testing;
- retain ggml's row-major and matrix multiplication semantics at the compatibility boundary, but document them in conventional mathematical terms internally.

`Attr any` is acceptable only in the first internal implementation if every operation immediately type-checks it. A generated tagged union is preferable once the operation set stabilizes.

### 6.3 Graph and memory planner

Implement:

1. DAG construction and topological ordering;
2. shape/type inference and validation;
3. capability matching against the CUDA backend;
4. liveness intervals;
5. arena allocation with alignment;
6. in-place/view alias rules;
7. stream/event dependencies;
8. graph capture eligibility;
9. trace hooks for per-node comparison.

Start with one GPU, one stream, and deterministic execution. Add multiple streams, CUDA graphs, and multi-GPU only after correctness and lifetime tests are strong.

### 6.4 CUDA binding strategy

Prefer the CUDA Driver API over the Runtime API for the core:

- load `nvcuda.dll` dynamically;
- call `cuInit`, device/context APIs, module APIs, memory APIs, stream/event APIs, and `cuLaunchKernel`;
- load `cublas64_12.dll` and `cublasLt64_12.dll` directly for dense matrix multiplication;
- use NVRTC only as a development or optional cache path, not as the sole release mechanism.

Each CUDA device should have a dedicated worker goroutine:

1. call `runtime.LockOSThread`;
2. create or retain the CUDA context on that OS thread;
3. create streams, events, and cuBLAS handles;
4. receive typed commands over a channel;
5. complete commands through events/futures;
6. destroy all resources on the same locked thread.

This avoids CUDA current-context bugs caused by Go goroutines moving between OS threads.

FFI safety rules:

- do not let asynchronous CUDA work retain pointers into movable Go memory;
- use CUDA-allocated pinned host buffers for asynchronous transfers;
- use `runtime.KeepAlive` around synchronous argument marshaling;
- keep kernel argument backing storage alive until `cuLaunchKernel` returns;
- synchronize or wait on an event before recycling pinned buffers;
- convert every CUDA/cuBLAS result into a typed Go error with the operation and device;
- probe required DLL exports and versions during initialization and fail with an actionable message.

### 6.5 Kernel strategy

Use three levels:

1. cuBLAS/cuBLASLt for FP32, FP16, BF16, and supported dense GEMM;
2. small custom kernels for elementwise, normalization, RoPE, softmax, cache movement, and indexing;
3. specialized quantized matrix-vector/matrix-matrix and fused attention kernels for performance.

For this machine, release kernel bundles should include:

- `sm_89` native cubin/SASS;
- `compute_89` PTX as a JIT fallback when practical;
- a manifest containing ABI version, kernel names, argument layouts, shared-memory rules, block constraints, source hash, nvcc version, and target architecture.

Do not call upstream C++ kernel entry points directly. Port kernel behavior in small groups and compare each group against upstream `test-backend-ops`.

Kernel source remains CUDA C++ or generated PTX and is the explicit exception to the all-Go source rule. Generated binary assets must be reproducible from source.

### 6.6 GGUF and quantized storage

Build the GGUF reader before the execution engine. It gives a low-risk way to prove format compatibility.

Requirements:

- versions and alignment rules;
- scalar, string, array, and nested metadata validation;
- split file discovery and ordering;
- tensor name, shape, type, offset, and size validation;
- bounded lengths/counts before allocation;
- overflow-safe byte-size calculations;
- file range checks before mapping or reading;
- immutable tensor descriptors;
- exact block layout definitions for every supported quantization type.

Initial type order:

1. F32, F16, BF16;
2. Q8_0;
3. Q4_0, Q4_1;
4. Q5_0, Q5_1;
5. Q2_K through Q6_K;
6. IQ and ternary families based on test-model demand;
7. remaining upstream types.

Writing and quantization are separate milestones. Reading existing GGUF files is the inference blocker.

### 6.7 Model architecture strategy

Do not mirror 137 C++ files directly. Define reusable capabilities:

- token embedding and tied/untied output;
- RMSNorm, LayerNorm, and variants;
- MHA, GQA, MQA, sliding-window, and special attention;
- RoPE variants and position scaling;
- dense FFN and activation variants;
- MoE routing and shared experts;
- encoder/decoder masks and cross-attention;
- recurrent/state-space blocks;
- architecture-specific cache/state;
- multimodal inputs;
- speculative/next-token heads.

Then port families in this order:

1. Llama-style dense decoder, using one small Llama fixture;
2. Qwen3 dense decoder, enabling use of the local Qwen3 4B models;
3. Mistral/Gemma/Phi dense variants;
4. common MoE families;
5. BERT-style encoders and embeddings;
6. T5/encoder-decoder;
7. Mamba/RWKV/delta-net/hybrid families;
8. multimodal and diffusion;
9. remaining specialized architectures.

Every architecture entry needs:

- required metadata schema;
- expected tensor names and shapes;
- supported quantization types;
- graph capability requirements;
- tokenizer requirements;
- cache/state type;
- known oracle fixtures;
- exact compatibility status.

Unsupported combinations must fail early and clearly.

### 6.8 Tokenization, sampling, and grammar

These are correctness-critical and mostly independent from CUDA. Port them in parallel with the first model graph:

- Unicode normalization and classification;
- SentencePiece, BPE, WordPiece, UGM, and model-specific pre-tokenization;
- special-token handling and byte fallback;
- detokenization and streaming UTF-8 boundaries;
- greedy, temperature, top-k, top-p, min-p, typical, penalties, mirostat, dry, infill, and sampler chains;
- deterministic seeded RNG behavior;
- grammar parsing and constrained-token filtering;
- chat-template behavior needed by the CLI/server.

Tokenizer output should be bit-exact. Floating samplers require defined tolerance or selection-equivalence tests where implementation order changes roundoff.

### 6.9 Runtime and cache

Implement runtime features in increasing complexity:

1. one sequence, token-at-a-time decode;
2. prompt batching;
3. causal KV cache;
4. context shift and cache copy/remove operations;
5. continuous batching and multiple sequences;
6. embeddings and pooling;
7. state save/restore;
8. LoRA adapters;
9. speculative decoding;
10. recurrent/hybrid memory;
11. multi-GPU split and peer transfer.

Avoid exposing cache internals in the public API. Different architecture families need different state models.

### 6.10 User-facing API

Start with idiomatic Go:

```go
model, err := model.Open(path, model.Options{Device: 0})
session, err := model.NewSession(runtime.SessionOptions{ContextSize: 8192})
tokens, err := session.Tokenize(prompt)
result, err := session.Decode(ctx, tokens)
```

Add a C-compatible facade only if a real consumer needs drop-in ABI compatibility. It cannot be exported from a no-cgo Go build on Windows without additional machinery and should not drive the internal design.

## 7. Phased implementation plan

### Phase 0: Freeze scope and build the oracle

Deliverables:

- compatibility contract defining what "entire" includes;
- pinned upstream SHA and build configuration;
- upstream CUDA build in a separate build directory;
- scripts that capture JSON/CSV oracle outputs;
- hashes and metadata for local model fixtures;
- benchmark protocol and acceptance thresholds.

Gate:

- upstream CLI, `llama-bench`, `llama-perplexity`, and `test-backend-ops` run on this GPU;
- results are reproducible from a documented Developer PowerShell.

Do not start broad porting before this gate.

### Phase 1: Go repository and CUDA smoke test

Deliverables:

- Go module with formatting, lint, unit-test, race-test, and Windows CI;
- dynamic DLL loader;
- CUDA error handling and device inventory;
- one locked-thread device worker;
- device allocation, copy, stream, event, and module wrappers;
- embedded `sm_89` test kernel performing vector addition;
- optional cuBLAS GEMM smoke test.

Gate:

- `CGO_ENABLED=0 go test ./...` passes;
- vector addition and GEMM results match CPU reference;
- repeated create/destroy and concurrent-client tests show no context/thread failures;
- leak test returns VRAM to baseline.

### Phase 2: Safe GGUF reader and inspector

Deliverables:

- metadata and tensor directory parser;
- type/block-size table;
- split-file support;
- `inspect-gguf` command;
- malformed-input and fuzz corpus;
- comparison tool against upstream GGUF output.

Gate:

- exact metadata/tensor descriptor match for small fixtures and all relevant local GGUF files;
- corrupted and truncated inputs fail without panic or excessive allocation.

### Phase 3: Tensor IR and CPU reference executor

Deliverables:

- tensor/view/storage model;
- typed operations and shape checks;
- graph ordering;
- simple arena/liveness planner;
- slow Go reference implementations for the first operation set;
- graph trace format.

Initial operation set:

- views, reshape, permute, transpose, copy, contiguous;
- add, multiply, scale;
- RMSNorm;
- softmax and masking;
- RoPE;
- get rows/embedding;
- dense matrix multiplication;
- SiLU/SwiGLU;
- basic cache writes/reads.

Gate:

- randomized operation tests match simple mathematical references;
- invalid shapes, strides, offsets, and aliases are rejected.

The reference executor is for correctness, small tests, and differential diagnosis, not production speed.

### Phase 4: CUDA executor for the first operation set

Deliverables:

- memory pool and aligned arenas;
- op capability table;
- cuBLAS/cuBLASLt dense matmul;
- custom kernels for the initial operation set;
- asynchronous execution with explicit lifetimes;
- per-node timing and trace output.

Gate:

- each operation matches the Go reference over randomized shapes and edge cases;
- differential cases derived from upstream `test-backend-ops` pass;
- Compute Sanitizer runs clean on kernel tests;
- no implicit device-wide synchronization in the steady-state token loop.

### Phase 5: First end-to-end Llama-family model

Deliverables:

- Llama metadata and weight mapping;
- dense decoder graph;
- causal KV cache;
- prompt evaluation and one-token decode;
- logits retrieval;
- F16 and Q8_0 support;
- one small, redistributable model fixture or a reproducible fixture download manifest.

Gate:

- tokenizer IDs exactly match upstream;
- selected layer tensors and final logits match within recorded tolerances;
- greedy output matches upstream for fixed prompts;
- perplexity delta is within the agreed threshold;
- model load, prompt, generation, and VRAM metrics are recorded.

This is the first meaningful product milestone.

### Phase 6: Production decode behavior

Deliverables:

- batching and micro-batching;
- multiple sequences;
- context shift and cache editing;
- complete core samplers;
- grammar constraints;
- cancellation and timeouts;
- deterministic state save/restore;
- CLI.

Gate:

- upstream test cases for batching, sampling, grammar, and state are represented as Go tests;
- long-running generation does not grow host or device memory;
- cancellation leaves the session reusable or closes it deterministically.

### Phase 7: Qwen3 and local hardware validation

Deliverables:

- Qwen3 metadata, graph, tokenizer behavior, and RoPE details;
- Q8_0 plus required K-quant support;
- local Qwen3 4B test;
- profiling and first specialized quantized matmul kernels.

Gate:

- the local Qwen3 model produces compatible greedy output and perplexity;
- generation performance is measured against the pinned upstream build;
- major regressions are attributed to named kernels or scheduler behavior.

Do not use the Qwen3.5 9B hybrid model as the first graph. It should follow Qwen3 because it exercises a more complex architecture.

### Phase 8: Server and API surface

Deliverables:

- streaming generation API;
- HTTP server with explicitly selected OpenAI-compatible endpoints;
- model/session lifecycle and admission control;
- metrics, health, structured errors, and graceful shutdown;
- server compatibility tests.

Gate:

- protocol fixtures match the chosen upstream endpoints;
- disconnect/cancel/load tests do not leak sessions or VRAM;
- overload behavior is bounded and observable.

### Phase 9: Quantization breadth and performance

Deliverables:

- remaining common GGML quantized layouts;
- specialized matvec/matmul kernels;
- fused attention;
- fused FFN operations;
- CUDA graph capture;
- pinned staging and overlap;
- benchmark dashboard.

Gate:

- every supported type has layout, dequantization, randomized op, model, and corruption tests;
- supported model/type combinations are published as a generated matrix;
- performance targets are met for prompt and generation workloads separately.

### Phase 10: Architecture family expansion

Deliverables are family-based, not file-count-based:

1. additional dense decoder variants;
2. MoE;
3. encoders/embeddings;
4. encoder-decoder;
5. recurrent and state-space;
6. hybrid;
7. multimodal/diffusion;
8. specialized remaining models.

Gate for each family:

- capability component is reusable;
- at least one small fixture and one realistic fixture;
- tokenizer, logits/perplexity, cache/state, and performance comparisons;
- unsupported members fail explicitly.

### Phase 11: Remaining tools and compatibility

Port based on demand:

- embeddings;
- perplexity;
- benchmark;
- quantizer;
- LoRA;
- speculative decoding;
- model inspection and GGUF split;
- other llama.cpp tools.

Treat conversion scripts and the web UI as separate projects. A Go rewrite is optional unless the product specifically requires it.

### Phase 12: Hardening and releases

Deliverables:

- signed/reproducible Windows releases;
- DLL discovery and compatibility diagnostics;
- kernel cache/version invalidation;
- fuzzing for GGUF, tokenizers, grammar, and server input;
- race, stress, leak, malformed-model, and out-of-memory tests;
- SBOM and license inventory;
- compatibility matrix tied to upstream SHAs;
- upgrade playbook.

Gate:

- clean install works on a machine with only the supported NVIDIA driver/toolkit prerequisites;
- failures never silently fall back to wrong results;
- release artifacts are reproducible and rollbackable.

## 8. Differential validation program

Upstream is the oracle, not an implementation dependency.

### Exact comparisons

Require exact equality for:

- GGUF metadata values and tensor descriptors;
- tensor byte offsets and computed sizes;
- tokenizer token IDs and token-piece round trips;
- integer/index operations;
- deterministic cache edits and state metadata;
- greedy token choice when logits are not tied.

### Floating comparisons

Record absolute/relative tolerances per dtype and operation. Use:

- elementwise maximum absolute and relative error;
- cosine similarity for activations;
- top-k token overlap and ranking;
- final-logit error;
- perplexity delta;
- KL divergence where useful.

Never use one broad tolerance for all kernels.

### Performance comparisons

Track:

- model load time;
- prompt tokens/second at several batch sizes;
- generation tokens/second at several context lengths;
- time to first token;
- peak and steady VRAM;
- host RAM;
- CPU utilization;
- kernel count and synchronization count per token.

Compare after correctness is established. Optimizing before trace-level parity will make failures harder to localize.

### Fixture matrix

Maintain three levels:

1. tiny generated files for unit tests and fuzzing;
2. small redistributable models for CI;
3. local realistic models for the RTX 4090-class acceptance suite.

The local Qwen3, Qwen3.5, and Gemma files are useful level-3 fixtures.

## 9. Security and correctness requirements

Treat GGUF, tokenizer data, grammar text, state files, and server input as hostile.

Mandatory rules:

- cap all file-supplied lengths and counts before allocation;
- use checked addition, multiplication, padding, and alignment;
- validate every tensor range against file size before mapping;
- validate rank, dimensions, strides, block sizes, and divisibility;
- reject unknown required metadata and unsupported types;
- never use a file-supplied token ID or tensor index before range checking;
- bound server batch, context, token, and output sizes;
- make OOM and CUDA errors explicit and recoverable where possible;
- never recycle asynchronous buffers before their event completes;
- fuzz parsers and state restoration continuously.

Go prevents many host memory errors but does not protect unsafe DLL calls, device pointers, kernel indexing, or integer arithmetic.

## 10. Versioning and upstream tracking

Create a machine-readable compatibility file, for example:

```yaml
upstream:
  commit: 42fc243060709331ff9b158a9ed2cbe37219ae83
cuda:
  toolkit: "12.9"
  architectures: ["sm_89", "compute_89"]
models:
  llama:
    status: supported
    types: [F16, Q8_0]
  qwen3:
    status: planned
```

For each upstream sync:

1. inspect changes to GGUF, type tables, ops, model registry, tokenizer, sampler, cache, and CUDA;
2. classify them as semantic, format, performance-only, or irrelevant;
3. add failing oracle cases before porting;
4. update one capability/family at a time;
5. publish compatibility against a new pinned SHA.

Avoid claiming drop-in parity with upstream `master`.

## 11. Effort and staffing reality

Reasonable order-of-magnitude estimates:

| Goal | Single experienced engineer | Small experienced team |
| --- | --- | --- |
| CUDA/no-cgo foundation plus GGUF inspector | 1-2 months | 3-6 weeks |
| One Llama-family model, limited types, CLI | 6-12 months | 3-6 months |
| Strong dense-transformer product, server, common quants | 12-24 months | 6-12 months |
| Broad llama.cpp runtime/model/tool parity | multi-year | roughly 24-48+ engineer-months |

These are not delivery promises. Specialized quantized kernels, tokenizer edge cases, model churn, and multi-GPU behavior dominate uncertainty.

The highest-leverage team composition is:

- one Go runtime/API owner;
- one CUDA/kernel/performance owner;
- one model/GGUF/tokenizer compatibility owner;
- shared testing and release ownership.

## 12. Explicit non-goals for the first release

- all 137 model architectures;
- all 44 data types;
- CPU parity or non-NVIDIA backends;
- multi-GPU;
- training/optimizer parity;
- every llama.cpp tool;
- Python converter rewrite;
- web UI rewrite;
- C ABI compatibility;
- exact reproduction of llama.cpp internal layouts.

These can be added after the first end-to-end model proves the architecture.

## 13. Go/no-go checkpoints

Stop or redesign if any checkpoint fails:

1. Direct DLL calls cannot be made stable under locked OS-thread device workers.
2. A reproducible kernel build and ABI manifest cannot be established.
3. The GGUF parser cannot match upstream safely on real files.
4. The first operation set cannot match upstream within per-op tolerances.
5. The first model cannot achieve logit/perplexity parity.
6. The CUDA design requires a growing C++ shim to make progress.
7. Performance remains far below upstream after profiling identifies no tractable kernel/scheduler gap.
8. The desired meaning of "entire" requires every model/tool/backend simultaneously.

At checkpoints 1 or 6, the honest alternative is a Go API around the upstream llama C ABI. That can be useful software, but it should be named and scoped as a binding rather than a port.

## 14. Immediate next actions

1. Approve the practical all-Go boundary: Go host code plus NVIDIA DLLs and compiled CUDA kernels.
2. Approve the first compatibility target: Llama dense decoder, F16 and Q8_0, single NVIDIA GPU, Windows amd64, CLI.
3. Build the pinned upstream CUDA oracle in a separate build directory.
4. Add the Go module and Phase 1 CUDA smoke test.
5. Add the Phase 2 GGUF inspector before model execution work.
6. Obtain or generate a small Llama-family GGUF fixture suitable for repeatable tests.
7. Record baseline results on the local Qwen3 models for the second architecture milestone.

The first implementation work should remain limited to Phases 0 through 2 until their gates pass.
