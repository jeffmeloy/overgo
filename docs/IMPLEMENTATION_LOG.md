# Implementation log

Archived compatibility and validation record through 2026-08-01. This file is
append-only historical evidence, not the live roadmap. See
[`../IMPLEMENTATION_STATUS.md`](../IMPLEMENTATION_STATUS.md) for current work
and [`COMPATIBILITY.md`](COMPATIBILITY.md) for the generated compatibility
matrix. Statements that work "remains pending" record their point in the
chronology and may be superseded by later commits and live compatibility
claims.

## Compatibility baseline

- llama.cpp commit: `42fc243060709331ff9b158a9ed2cbe37219ae83`
- initial host: Windows amd64
- initial device: NVIDIA compute capability 8.9
- cgo: disabled

## Status

| Area | State | Notes |
| --- | --- | --- |
| Repository foundation | Complete | Go module and compatibility contract created |
| CUDA driver loading | Complete | Direct `nvcuda.dll` calls, device inventory validated |
| CUDA execution | In progress | Persistent resources, cuBLAS F32, native quantized embedding/matmul/routed-expert decode, fused Qwen3-Next/Qwen3.5 recurrent kernels, and correctness-first routed MoE validated on RTX 4090 |
| GGUF format | In progress | Bounds-checked parser, automatic validated split-file loading, bounded tensor ranges, streamed device weights, and canonical streaming single-file writer validated |
| Tensor graph | In progress | Typed IR, layout transforms, head broadcasting, grouped/batched matmul, clamp, RMSNorm/affine LayerNorm, channel-group normalization, dense/depthwise same-padding Conv1D, token/learned-position embeddings, scaled and YaRN normal/NeoX RoPE, scaled multi-axis RoPE, ALiBi, causal and symmetric sliding/softcapped/gated GQA with per-head sink logits, bidirectional and causal cached T5 relative-position attention, gated linear attention and WKV6/WKV7 recurrence, legacy and absorbed MLA decomposition, DeepSeek 4 Sinkhorn hyper-connections and compressed sparse attention, softmax/sigmoid/sqrt-softplus top-k routed gated or ungated SiLU/ReLU/GELU MoE with fixed expert selection, separate or fused gate/up expert storage, split router inputs, grouped expert-bank indices, correction bias, and per-expert output scales, ReLU/GELU/xIELU/SwiGLU/squared-ReLU, SSM convolution, Mamba selective scan, fused gated delta net, reference/CUDA executors, and arena planner |
| Quantization | In progress | F32/F16/BF16/F64, I8/I16/I32/I64, Q8_0, Q2_K-Q6_K, every pinned IQ1/IQ2/IQ3/IQ4 layout, Q1_0/Q2_0, TQ1_0/TQ2_0, MXFP4/NVFP4, Q4_0/Q4_1, and Q5_0/Q5_1 decoding |
| Model runtime | In progress | Incremental dense Qwen 2/3, bounded-host/F32-preload/native-quantized Mixtral, Arctic, BailingMoE/BailingMoE2, Cohere2-MoE with single-block MTP drafting, DBRX, Deci, DOTS1, DeepSeek v1/DeepSeek2/DeepSeek 3.2/DeepSeek 4/DeepSeek2-OCR, DFlash paired-target block drafting, Eagle3 paired-target autoregressive drafting, ERNIE 4.5-MoE, Gemma3n AltUp/Laurel, Gemma4 with unified image/audio projection plus shared-context Gemma4 Assistant drafting, GLM-DSA/GLM4-MoE, Granite Hybrid/GraniteMoE, GroveMoE, Grok, Hunyuan-Dense/Hunyuan-MoE, HY-V3 with greedy/stochastic multi-head MTP drafting, Jamba, Kimi Linear KDA/no-RoPE MLA, Mamba v1/v2, Falcon-H1 parallel attention/Mamba2, RWKV6/RWKV6-Qwen2/RWKV7/ARWKV7, Mellum, MiMo2, MiniMax-M2, Mistral 3/Mistral 3 MoE/Mistral 4, PLaMo2, SmallThinker, Step3.5 with greedy/stochastic multi-head MTP drafting, Qwen2-MoE, Qwen3-MoE, Qwen3-VL-MoE, AFMoE, Laguna MoE, OLMoE, PhiMoE, EXAONE-MoE, and LLaDA-MoE, hybrid Qwen3-Next/Qwen3.5/Qwen3.5-MoE with greedy and stochastic Qwen3.5 single-block MTP drafting plus Qwen3.5 image projection, non-causal no-cache Dream, LLaDA, and RND1 MoE, hybrid LFM2/LFM2-MoE, PLM/MiniCPM3/DeepSeek2/Mistral 4 MLA, BERT/EuroBERT/Gemma Embedding/JinaBERT v2/v3 MoE/Llama Embed/ModernBERT/NeoBERT/NomicBERT/NomicBERT-MoE encoders, T5 encoder-decoder sessions, WavTokenizer semantic-token audio-feature decoding, Chameleon decoders with projected soft-token overrides, Hunyuan-VL/PaddleOCR text-coordinate decoding, Qwen2-VL and dense/MoE Qwen3-VL image/video projection and text-coordinate decoding, CogVLM text/projected-visual decoding, Apertus, Arcee, Baichuan 7B, BitNet, Bloom, CodeShell, dense Cohere2/Command R/ERNIE 4.5, Falcon, Gemma 1/2/3, GLM4, GPT-2/GPT-NeoX, Granite, InternLM2, EXAONE/EXAONE 4, XVERSE, Jais/Jais2, Maincoder, MiniCPM, MPT, Nemotron, OLMo/OLMo2/OLMoE, OpenELM, Orion, Pangu Embedded, Phi-2/Phi-3, PLaMo/PLaMo 3, dense Refact, Seed-OSS, StableLM, StarCoder/StarCoder2, SmolLM3, Talkie, and constrained Llama-family CUDA execution with serializable, prefix-editable attention/recurrent cache |
| Tokenizer and sampling | In progress | Seven tokenizer corpora match 326 upstream cases; Gemma4 raw UTF-8 BPE, BERT WordPiece, and real-model T5 UGM are validated; ordered/repeatable top-k/p, min-p, typical, top-n-sigma, XTC, penalties, DRY, infill, Mirostat v1/v2, GBNF, and JSON-Schema conversion implemented |
| CLI and server | In progress | Inspect/tokenize/block-check/generate/perplexity/embedding/benchmark/JSON-Schema CLIs plus bounded completion, streaming, embedding, literal-choice, GBNF, JSON-Schema, and native/OpenAI projected image/audio HTTP paths |
| Local verification | Complete | Unit and optional CUDA integration script |

## Validated milestones

- The August 7 semantic-fixture sweep moved compact Qwen 3.5 hybrid dimensions,
  cadence, layer inventory, and serving markers into one `modeltest` fixture.
  A shared checked artifact-identity builder replaced local helpers and ignored
  identity errors across the test tree, deleting more call-site surface than
  the two common utilities added.
- The August 7 compiled-program wave removed the pre-recipe specification read,
  sealed cache schemas into model plans, compiled fixed layer instructions with
  indexed operands, deleted `cachedBlockCatalog` and the remaining public Qwen
  GDN forwarding wrappers, and made canonical document encoding single-pass.
  A follow-up literal sweep extracted one dtype-aware checked device-value view;
  cache shift/page/compaction and packed split/selection paths no longer carry
  independent F32 stride, pointer, capacity, or shape arithmetic.
- The August 7 execution-authority follow-up made resolved programs one-shot,
  removed serving admission bypasses, required compiled layer plans, deleted
  the family-owned host Qwen graph path, compiled terminal normalization/output
  selection, and consolidated five canonical document lifecycles behind a
  typed artifact codec with net code deletion.
- The follow-on sealed `ModelPlan` policy and layer storage, compiled Gemma 4
  assistant and appended MTP layer topology, and extended the typed document
  codec to version-selected schemas. Recipe definitions/lifecycle/decisions,
  profiles and model evidence, model definitions, and versioned run records now
  retain only domain canonicalization around the shared lifecycle.
- The August 7 RepoDB adversarial follow-up hardened import identity and symlink
  containment, validated dataset split topology, removed overflow from bounded
  tensor sample indexing, and made typed profile provenance resolve through
  stored derivation evidence and lineage at publication and load boundaries.
- The August 7 RepoDB evidence wave added bounded query tooling, injected
  durable-write recovery fixtures, typed lifecycle decisions, profile v2
  per-fact provenance, environment-bound phased runs, terminal gate records,
  closure obligations, deterministic group-safe dataset memberships, raw-MAD
  regression advisories, bounded tensor measurements, and canonical JSONL
  import with source-commit/export-digest lineage.
- The August 6 RepoDB follow-through moved manifests onto shared document and
  clone contracts; centralized safe reads, alias resolution, document batches,
  and commit gates; and replaced whole recipe-catalog reconstruction with owned
  clones and direct module lookup. No production domain package imports the
  RepoDB persistence implementation.
- The August 6 artifact-ownership wave removed duplicate optional-ID and alias
  compare-and-set cloning from dataset, recipe, lifecycle, and persistence
  packages; one tested artifact helper now owns caller-pointer isolation.
- The August 6 RepoDB document-contract wave consolidated kind/media/schema
  facts, bounded identity hashing, descriptor construction, byte ownership, and
  stored-content validation across dataset, recipe, evidence, model, run, and
  evaluation documents. Domain packages retain only canonical shape policy.
- The August 6 RepoDB persistence wave replaced catalog-wide provenance scans
  with rebuildable per-artifact indexes and added immutable versioned snapshot
  segments. Snapshot payloads are canonical, bounded, SHA-256 checked, and
  atomically published; exact chain-anchor validation, tail replay, corruption
  fallback, and foreign-snapshot fallback preserve the log as authority.
- The August 6 RepoDB runtime wave added a protocol-neutral compiled-workflow
  executor with registered module adapters, typed cardinality checks, exact
  dependency delivery, and deterministic artifact-fact consolidation. Atomic
  commits retain successful, failed, and cancelled run evidence even after the
  caller context is cancelled.
- The August 6 RepoDB dataset wave added canonical version/view/split/mixture
  documents with bounded assets, selectors, projections, partitions, and
  normalized weights. Strict parsing, typed lineage, atomic publication, and
  alias resolution are covered against the real hash-chained store.
- The August 6 generic-cache-binder wave collapsed the parallel host and CUDA
  cache-input algorithms behind `layerCacheSource[T]`. Schema application and
  state construction are shared; host values and device pointers remain thin
  feed adapters covered by the existing dual-path binding fixture.
- The August 6 named-state-binding wave replaced specialized indexer,
  convolution, and SSM graph-input fields with `CacheStates[*tensor.Tensor]`.
  Host/device binders share schema-governed zero initialization, and a
  DeepSeek 3.2 fixture proves named history binding on both execution paths.
- The August 6 primary-cache-pair wave replaced positional two-element schema
  arrays with `CachePair`. Host/device preparation and cache editing now select
  explicit key/value roles, validation reports those roles, and numeric state
  input labels no longer encode the contract.
- The August 6 cache-schema-carrier wave represented primary and named shape
  contracts through `CacheState[CacheValueSchema]`. Schema modes now occupy the
  same carrier field as graph, host, and device values, eliminating the final
  schema-specific `{mode,value}` representation.
- The August 6 cache-policy wave removed the parallel cache-extent enum.
  `CachePolicy` now derives the canonical fixed/token mode in `LayerPlan`, and
  schemas, host editing, device compaction, named states, and wire validation
  consume the same typed policy without conversion or re-dispatch.
- The August 6 cache-transform wave added shared deterministic value appending
  and representation mapping to `CacheStates`. Host/T5 result materialization,
  deep cloning, and CUDA output collection now preserve names and modes through
  the common collection contract rather than local loops.
- The August 6 graph-state wave unified `DenseBlockResult` named outputs under
  `CacheStates[*tensor.Tensor]`. Each value now carries its cache mode through
  graph construction, host execution, and device retention; parallel
  token/fixed maps and their duplicate collection loops are removed.
- The August 6 cache-collection wave introduced `CacheStates[T]` with shared
  cloning and deterministic ordering. Host cache serialization/editing and
  device batching no longer maintain independent map-clone or key-sort logic;
  existing round-trip and fork-isolation fixtures cover the common operations.
- The August 6 generic-cache-state wave replaced separate host, device, and
  in-flight graph `{mode,value}` structs with `CacheState[T]`. Representation
  aliases preserve call sites while the shared carrier removes duplicated
  cache semantics and production surface.
- The August 6 cache-mode wave moved serialized fixed/token values and their
  validity/alignment rules into the model cache schema. Inference aliases the
  shared ABI type and consumes one extent-to-mode relationship while preserving
  existing payload values and API names.
- The August 6 typed-cache-key wave promoted named-state keys to
  `CacheStateName` across graph, host, device, editing, and session paths.
  Serialization converts only at the byte boundary, and a non-canonical valid
  fixture proves extension names remain round-trippable.
- The August 6 cache-ABI wave centralized every serialized named-state fact in
  the model cache schema. Graph output, host/device binding, validation,
  session paths, NextN drafting, and fixtures now consume the same DSA,
  DeepSeek 4, recurrent, position, and T5 cross-attention state names.
- The August 6 graph-weight wave replaced every execution-time requirement map
  with one ordered typed schema. Dense, encoder, MLA/DSA, Mamba, RWKV,
  DeepSeek 4, Gemma, Qwen, Kimi, LFM, and WavTokenizer builders now share
  deterministic nil-weight validation without transient map allocation.
- The August 6 metadata-field wave replaced eleven repeated map readers with
  ordered typed field schemas. SSM, RWKV, expert, MLA/indexer, DeepSeek 4,
  YaRN, xIELU, and WavTokenizer metadata now share deterministic scalar binding;
  existing model fixtures cover the migrated success and failure contracts.
- The August 6 metadata-read wave replaced all architecture-name selection in
  model metadata ingestion with resolved profile facts. Existing encoder,
  attention, MLA, recurrent, hybrid, forward, and capability policies cover
  shared schemas; a bounded bitset carries the remaining ALiBi, multimodal,
  scalar, block-count, and gating facts. Cross-policy validation rejects
  inconsistent persisted combinations before parsing.
- The August 5 profile-validation wave added one fail-closed semantic gate for
  every serialized architecture enum and policy bitset. Embedded catalogs,
  RepoDB profile documents, bound metadata, and direct plan compilation reject
  unknown values and inconsistent Qwen GDN, multi-axis rotary, or fused-QKV
  contracts before policy activation.
- The August 5 forward-policy sweep replaced the final specialized DFlash,
  Eagle3, Gemma 4 Assistant, and WavTokenizer name guards. Speculative sidecar
  compatibility now compares the complete resolved policy.
- The August 5 validation-sweep wave moved the last production `Spec`
  architecture predicate—DeepSeek 3.2 full-indexer cadence—into profile data.
- The August 5 expert-validation wave completed the 36-contract hybrid/MoE
  catalog. The shared validator now contains no architecture-name predicate;
  exact profiles own dense/MoE, routing, scaling, position, and schedule facts.
- The August 5 hybrid-validation wave moved Qwen hybrid/MoE, GroveMoE, MiMo2,
  and Step3.5 scheduling and expert selection into serialized profile policy.
- The August 5 recurrent-validation wave replaced the architecture switch and
  RWKV/Nemotron name branches with 15 serialized contracts, while naming fixed
  convolution, state-width, target-count, shift, and alignment facts.
- The August 5 MLA-validation wave moved DeepSeek 3.2/4, Kimi Linear,
  Mistral 3, and MiniCPM3 selection into typed profiles and named the remaining
  fixed family dimensions. The common MLA validator has no name dispatch.
- The August 5 attention-validation wave replaced all architecture-name
  dispatch in the shared attention validator with 32 typed profile contracts.
  Existing model-specific diagnostics and metadata bounds remain intact.
- The August 5 validation-policy wave moved base rotary and encoder-family
  invariants into serialized profile selectors. Shared BERT-family shape and
  optional-expert validators replaced six architecture-name branches.
- The August 5 runtime-policy wave moved input/logit scaling, normalization
  placement/bias/fallback, and layer-level RoPE schedules from architecture-name
  predicates into serialized typed profiles. Bound RepoDB profiles now carry
  these decisions through metadata, catalog, graph, and cache execution.
- The August 5 resolved-definition wave introduced a canonical document that
  binds model, profile, tensor-inventory, architecture, and validated `Spec`
  identities. GGUF ingestion verifies that tensor facts match the same source,
  RepoDB resolves the complete bound definition, and recipe v2 compilation
  consumes its exact policy without consulting the bootstrap architecture
  registry. Model inventory and resolved facts publish in one atomic batch;
  graph dispatch, weight catalog loading, and persistent-cache admission retain
  the bound policy. Identical relocated models retain definition identity.
- The August 5 tensor-inventory wave added one bounded, content-addressed model
  tensor schema shared by GGUF and Safetensors ingestion. Logical names,
  shapes, storage types, and payload sizes are sorted and canonical; physical
  paths, shard names, and offsets remain representation details. RepoDB lineage
  binds each inventory to its model manifest and supports exact lookup without
  reopening model payloads.
- The August 5 model-definition wave replaced the production architecture
  registry builder and its mutation helpers with a strict embedded data
  catalog. RepoDB profile publication now accepts an externally supplied
  catalog, registered aliases can supersede bootstrap facts without changing
  process-global fallback state, and activation remains profile-identity and
  parity gated. An aggregate semantic digest pins every migrated profile.
- The August 5 training-foundation wave adapted the compact `adaptive_new`
  flat optimizer core behind a dependency-free parameter plan. Named matrix
  groups derive sign or Muon updates, CPU Newton-Schulz reuses min-side Gram
  scratch, typed schedules make pre-increment semantics explicit, and
  plan/config-bound snapshots reject incompatible restores. A source-generated
  golden fixture pins mathematical parity without importing the original
  autograd, matrix, diagnostics, or CUDA packages.
- The August 4 ownership/schema consolidation wave replaced separate raw and
  F32 CUDA weight state machines with one transactional tensor owner while
  preserving their upload strategies and nil-safe public lifecycles. One
  Mamba2 tensor schema now serves Mamba2, Granite Hybrid, Falcon-H1, and
  recurrent Nemotron; dense Q/K and attention-gate stages use single typed
  applicators; and MTP families share policy admission plus appended
  dense-block coordination. Production surface fell by 4 lines; full CPU and
  CUDA verification passed on the Windows amd64 RTX 4090 D host.
- The August 4 dense-stage planning wave compiled weight requirements,
  routed/shared expert composition, Q/K normalization, query scaling,
  attention-output stages, residual/FFN flow, dense leaf selection, and Deci
  sparse selection into per-layer policies. Scheduler and dense paths now
  share one cached-block context, while remaining production leaf attention
  and RoPE executions use typed option contracts. Contract tests cover dense
  stages, expert composition, graph selection, and MoE weight requirements;
  full CPU and CUDA verification passed on the Windows amd64 RTX 4090 D host.
- The August 3 layer-graph planning wave promoted rotary, attention, and MoE
  controls into compiled `LayerPlan` values. Dense and Gemma 4 graph execution
  now share paired RoPE, planned attention, one typed routed-expert core, and
  one shared-SwiGLU composition; dense dispatch carries one typed request.
  Full CPU and CUDA verification passed on the Windows amd64 RTX 4090 D host.
- The August 3 typed-builder wave replaced the internal attention, MoE, and
  RoPE positional control streams with named option contracts. Specialized
  wrappers now declare only their active features, centralized builders own
  validation and graph attributes, and public APIs remain unchanged. Full CPU
  and CUDA verification passed on the Windows amd64 RTX 4090 D host.
- The August 3 semantic-literal wave named architecture-fixed projector/RWKV
  normalization values, quantizer near-zero thresholds, sampler/schema state
  envelopes, GGUF buffers, server and CLI defaults, completion/token bounds,
  and SSRF/media policy limits. Diagnostics now derive their numeric bounds
  from the same constants, while tensor shapes, fixture values, indexes, and
  local formula coefficients remain literal. Full CPU and CUDA verification
  passed on the Windows amd64 RTX 4090 D host.
- The August 3 projector lifecycle/geometry wave moved all twelve projector
  openers to one failure-safe GGUF transaction, moved optional bias, linear, and
  normalization operations into the shared graph runtime, and replaced three
  pixel-merge implementations plus three temporal patch-pair splitters with
  checked plans. Production surface fell by 16 lines while 98 focused ownership
  and geometry regression lines were added; full CPU and CUDA verification
  passed on the Windows amd64 RTX 4090 D host.
- The August 3 projector catalog-convergence wave removed every per-family CUDA
  opener and tensor inventory. Qwen2-VL, Qwen3-VL, CogVLM, Granite 4 Vision,
  and Gemma 4 now load validator-produced catalogs; DeepSeek-OCR v1/v2 and
  Gemma 3n load spec-owned catalogs. Gemma 4 audio remains an explicit optional
  sidecar. The wave removed 146 net code lines; full CPU and CUDA verification
  passed on the Windows amd64 RTX 4090 D host.
- The August 3 graph/catalog follow-up removed the remaining host layer graph
  family switch: the reflected layer catalog now binds every required and
  optional host field, matching device binding. Llama 4, Hunyuan-VL,
  PaddleOCR, and MiMo-VL validators now produce the ordered CUDA tensor set,
  eliminating four device-side inventories while preserving Hunyuan's
  host-reordered convolution exclusion. The wave removed 369 net code lines;
  full CPU and CUDA verification passed on the Windows amd64 RTX 4090 D host.
- The August 3 layer-catalog follow-up extended the validated mirror catalog
  across required and optional host tensor loading and device graph binding,
  replacing parallel inventories while retaining family-specific structural
  validation. Projected chunk execution now accepts one request value, and
  model loading reuses its compiled normalization plan. The wave removed 596
  net lines; full CPU and CUDA verification passed on the Windows amd64 RTX
  4090 D host.
- The August 3 policy-consolidation wave replaced separate host/device optional
  layer binders with one validated mirror catalog, including Kimi KDA fields;
  routed embedding skip, per-layer projections, attention blocks, scheduled
  temperature, and DeepSeek4 positions through one layer side-input binder;
  compiled normalization operation/placement/bias/layout and projected request
  contracts; unified draft catalog/session descriptors and tensor enumeration;
  and added cross-policy model-plan invariants. Full CPU and CUDA verification
  passed on the Windows amd64 RTX 4090 D host.
- The compatibility generator requires exact coverage of the architecture
  registry. Missing or stale model entries fail verification; the matrix now
  includes all registered decoder, encoder, hybrid, diffusion, draft, and
  multimodal families.
- The local Qwen3 4B GGUF loads a 151,936-token GPT-2/Qwen2 vocabulary and
  tokenizes representative text.
- The benchmark CLI records JSON load, TTFT, post-first-token decode,
  end-to-end, host-heap, model, and device metrics with bounded warmup/run/token
  controls. A real Qwen3 Q8 native-quant smoke run on the RTX 4090 D loaded in
  4.09 s, reached first token in 181 ms, and decoded the post-first token at
  10.5 tokens/s. A subsequent instrumented run measured 6,706,837,504
  persistent Runner-owned CUDA bytes and a 6,707,455,488-byte generation
  high-water mark.
- CUDA driver allocation accounting keeps a concurrency-safe pointer-size
  ledger for every successful `MemAlloc`/`MemFree` on the Runner's shared
  worker. Current allocation count/bytes and lifetime peak bytes therefore
  cover persistent device weights, graph arenas, feeds, and auxiliaries.
- Bounded fuzz harnesses cover GGUF parsing, tokenizer encode/decode, GBNF
  compilation, JSON-Schema conversion, sampler restoration, cache/session
  restoration, and server JSON routes. Short live campaigns completed millions
  of mutations without a panic. `scripts/fuzz-smoke.ps1` makes the eight-target
  campaign repeatable.
- A deterministic CycloneDX 1.6 SBOM inventories the Go module graph, pinned
  llama.cpp baseline, Go standard library, CUDA driver/toolkit, and SHA-256
  hashes for the versioned CUDA manifest, sources, and embedded PTX. The
  kernel-manifest verifier also checks schema/ABI metadata, the complete PTX
  entry set, and every function's ordered parameter types/alignment/array
  extents. Unit tests, verification, release creation, and CI reject stale
  provenance. The project/kernel license remains `NOASSERTION` rather than
  assuming distribution rights.
- CUDA module loading fails closed unless the embedded PTX matches the
  ABI-versioned hashes compiled into the Go host. The manifest verifier ties
  those runtime pins to the checked-in assets, providing content-based NVIDIA
  JIT-cache invalidation without a separate application cache.
- Driver-level execution telemetry counts successful custom-kernel launches,
  stream/context barriers, and H2D/D2H operations and bytes. Benchmark runs
  report deltas and per-token rates; server metrics expose monotonic totals.
- Preloaded dense generation composes embedding lookup, every layer, final
  normalization, last-token slicing, and output projection into one executor
  graph per token. KV outputs have explicit device lifetimes and identical
  RoPE attributes share one upload. On the local Qwen3 Q8 three-token
  benchmark this reduced stream synchronizations from 117 to 3 and H2D/D2H
  calls from 477/333 to 6/3, leaving only logits on the host; greedy Q8/Q6_K
  output and allocation-release invariants pass.
- Direct Qwen3.5 generation uses the same retained-output lifetime for mixed
  attention KV and convolution/SSM state. Its local three-token run uses one
  barrier and one logits copy per token; 52.69 MB of initial recurrent zeros
  are created by 48 stream-ordered device memsets instead of host transfers.
- Embeddings share one Runner implementation for mean, last-token, and
  unpooled per-token output. Normalization matches pinned
  `common_embd_normalize` (`-1`, max-absolute int16 range, and general p-norm);
  OpenAI responses can encode little-endian float32 vectors as base64. Real
  UMT5 per-token output matches its validated final hidden state.
- The release builder cross-compiles seventeen Windows-amd64 tools with
  `CGO_ENABLED=0`, `-trimpath`, no VCS stamp, and no Go build ID; it builds
  twice and requires byte-identical stored ZIPs with fixed timestamps and
  internal/external SHA-256 manifests. A real two-build check produced
  byte-identical archives; the checksum is kept outside the archive to avoid a
  self-referential manifest.
- The tokenizer matches every checked-in `ggml-vocab-qwen2` and
  `ggml-vocab-gpt-2` oracle case from the pinned llama.cpp checkout.
- BERT WordPiece matches all 46 checked-in `ggml-vocab-bert-bge` oracle cases.
  The Go path ports the pinned default/metadata-controlled NFD accent stripping
  and lowercasing, Unicode whitespace/control/mark handling,
  punctuation/ASCII-symbol/Chinese segmentation, `▁` longest-token matching,
  whole-word unknown fallback, CLS/SEP insertion, and three-pass decode-space
  cleanup. `golang.org/x/text` supplies the pinned pure-Go Unicode NFD tables.
- Llama SPM and Llama 3 BPE tokenization each match all 46 checked-in upstream
  oracle cases, including whitespace, Unicode, byte fallback, and punctuation.
- The Qwen3.5 pre-tokenizer includes combining marks in letter runs and matches
  all 46 pinned upstream `ggml-vocab-qwen35` cases. The real local 9B
  vocabulary loads and tokenizes successfully.
- Raw GGUF tensors can be streamed into persistent CUDA allocations without a
  full host-side weight copy.
- The pinned IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, and IQ3_XXS/IQ3_S
  codebooks are deterministically generated from audited `ggml-common.h`.
  Their host decoders byte-match complete 256-value C-oracle blocks, while
  native CUDA `get_rows` and `mul_mat` paths match the Go reference on the
  RTX 4090 D. The kernel manifest hashes the generated include dependency.
- Local split GGUF models opened through
  `<prefix>-00001-of-XXXXX.gguf` load as one logical file while every tensor
  retains its shard-local source, data offset, and bounds. The reader validates
  pinned `uint16` split indices/counts, the non-negative `int32` global tensor
  count, shard naming, missing parts, duplicates, aggregate limits, and closes
  all part handles together. `inspect-gguf` exposes the split count and tensor
  shard index.
- The pure-Go GGUF v2/v3 writer validates every scalar and array metadata
  representation, tensor dimensions/types/block alignment, duplicate names,
  offsets, and short payloads. It streams canonical aligned data without
  buffering weights. `gguf-merge` uses the same path to merge validated split
  sources, removes split bookkeeping, and never overwrites an existing output.
  The inverse `gguf-split` path partitions on tensor count and aligned payload
  bytes, supports a metadata-only first shard, writes pinned upstream names and
  split metadata, and cleans up only shards it created after a failure. Pinned
  `llama-gguf-hash` accepts Go writer output, and pinned
  `llama-gguf-split --merge` accepts and merges the Go-produced shard set.
- The pure-Go GGUF hasher streams raw logical tensor bytes and matches pinned
  `llama-gguf-hash` for per-tensor and concatenated-model XXH64, SHA-1,
  SHA-256, and llama.cpp-namespaced UUIDv5 output. Manifest checks preserve
  model/tensor fallback and distinct mismatch, missing-entry, unknown-format,
  and file-error exit states.
- The pure-Go model quantizer streams source blocks through the complete host
  dequantizer into pinned-layout Q1_0, Q2_0, Q2_K-Q6_K, Q4_0/Q4_1,
  Q5_0/Q5_1, Q8_0, TQ1_0/TQ2_0, IQ1_S/IQ1_M,
  IQ2_XXS/IQ2_XS/IQ2_S, IQ3_XXS/IQ3_S, IQ4_NL/IQ4_XS, or MXFP4/NVFP4
  encoders, with F32/F16/BF16 destinations as well. All 27 packed encoders,
  including internal Q8_1 and Q8_K layouts,
  byte-match exported reference routines from the pinned
  `ggml-base.dll`; multi-chunk requantization, "mostly" matrix selection,
  metadata rewriting, split input, exclusive output creation, and upstream
  `llama-gguf-hash` acceptance are covered.
- A complete small Qwen3 dense block executes on CPU and CUDA with matching
  results: Q/K/V projections, per-head Q/K RMSNorm, NeoX RoPE, causal GQA,
  attention output, residuals, and SwiGLU feed-forward.
- Layer 0 of the real local Qwen3 4B model matches the CPU reference over four
  tokens with maximum absolute error `1.43e-6`.
- The append-only KV cache matches full-forward hidden states for the checked
  two-token real-model integration case.
- Composite session state includes token history, KV tensors, deterministic RNG
  position, and adaptive Mirostat state. It validates payload bounds, sampler
  configuration, and a sampled GGUF fingerprint before restoring. A real
  Qwen3 native-Q8 save/load/continue run exactly reproduces uninterrupted
  greedy token output.
- Mirostat v1 and v2 both have deterministic adaptive implementations with
  reset and serialization coverage. DRY accepts token sequence breakers and
  expands string breakers across overlapping vocabulary-token boundaries like
  the pinned llama.cpp implementation.
- The implemented non-Mirostat stages use llama.cpp names and execute in a
  caller-selected, repeatable chain. The default is
  `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`; both
  the generation CLI and HTTP `samplers` array expose ordering. Order-sensitive
  candidate-set tests cover penalties before/after top-k, and custom chains
  participate in the resumable sampler signature without changing legacy
  default signatures.
- Top-n-sigma and XTC reproduce the pinned upstream probability-vector corpus.
  Top-p, min-p, typical, and XTC share `min_keep` behavior. XTC probability,
  threshold, and random-stream position are validated and resumable.
- The temperature stage supports the pinned dynamic-temperature entropy
  normalization, range, and exponent formula. Zero range remains byte-for-byte
  on the fixed-temperature path, and dynamic configuration is state-bound.
- Adaptive-p implements the pinned target-distance transform, always executes
  as the terminal selector, and updates its original-probability EMA only after
  token acceptance. Reset and sampler state v4 preserve its RNG and EMA;
  compatible v2/v3 state readers remain.
- Infill reproduces the pinned sorted-softmax EOG weighting, unrendered
  token-piece prefix merging, EOT/EOS fallback, and two probability thresholds.
  The runner builds its immutable descriptor from GGUF tokenizer metadata; CLI
  and HTTP ordered sampler chains populate it automatically, and its topology
  participates in the resumable sampler signature.
- Native `/infill` reproduces the pinned PSM/SPM repository-context formatter,
  3:1 prefix/suffix truncation, batch/context budgeting, BOS placement, and
  exact-token prompt handoff. The tokenizer reads all six current FIM token
  metadata keys, three deprecated aliases, and pinned text fallbacks while
  promoting auto-detected controls.
- Text-only chat formatting executes the GGUF `tokenizer.chat_template`
  through a pure-Go, in-memory-only Jinja loader with bounded source/output.
  Pinned `raise_exception`, `strftime_now`, `namespace`, and `range` globals
  are available; exception/date inputs and rendered date output are bounded.
  Qwen3, Qwen3.5, Gemma 3, and Bonsai prompts are byte-exact against pinned
  `/apply-template`, including the injected `enable_thinking` default and
  single-BOS behavior. Boundary-token native formatters remain the fallback
  for models without metadata.
- Tool-aware Jinja formatting supplies OpenAI function schemas, assistant
  calls, tool results, named `tool_use` metadata templates, and Python-style
  ordered `tojson`. Qwen3 JSON-in-XML plus Qwen3.5/Bonsai Hermes prompt and
  history vectors are byte-exact against pinned `/apply-template`. Buffered
  and streaming OpenAI chat support auto, required, none, and named choices;
  auto uses lazy delimiter-triggered GBNF, required/named use forced
  schema-derived GBNF, and parsed calls receive validated arguments, IDs, and
  `tool_calls` finish reasons. Tool-enabled SSE buffers the template output
  through validation and emits complete structured call deltas without
  leaking XML/JSON wrapper syntax.
- Lazy GBNF activation uses bounded ECMAScript Unicode trigger patterns with
  lookahead, lookbehind, backreferences, UTF-8 byte-correct capture replay,
  and caps on patterns, source, input, backtracking stack, and match time.
- A real Bonsai-27B-Q1_0 named-tool request on the RTX 4090 D generated the
  Hermes call for `weather`, parsed `{"city":"Boston"}`, assigned a call ID,
  returned null content, and finished with `tool_calls`. Disabling optional
  thinking kept the forced call to 26 output tokens.
- Ordered logit biases run before filtering and grammar acceptance, support
  additive finite adjustments and negative-infinity bans, and participate in
  the sampler signature. CLI `-ignore-eos` and HTTP `ignore_eos` ban every EOG
  token recognized by the loaded GGUF vocabulary; HTTP bias objects and pair
  arrays support numeric IDs, tokenized text keys, and `false` bans.
- Completion/chat stop strings are validated as a string or bounded array.
  The runner halts at the matching token boundary, and a cross-token output
  filter withholds possible prefixes so neither buffered nor SSE responses
  leak the matched marker. Unmatched prefixes flush at natural completion.
- Completion/chat `n` supports one through eight sequential choices. Each
  choice starts from fresh sampler, grammar, RNG, and adaptive state with a
  deterministic seed offset. Buffered and SSE responses preserve indices;
  usage counts prompt tokens once and aggregates generated choice tokens.
- The server process handles SIGINT/SIGTERM with a bounded 30-second graceful
  drain. Optional end-to-end request deadlines propagate through inference and
  return HTTP 408 on expiry; negative timeout configuration is rejected.
- Gemma turn-template formatting matches the pinned `/apply-template` oracle
  for user-only, system-prefix, and alternating conversations. It maps
  assistant to `model`, trims message bodies like the GGUF Jinja template,
  rejects role-order violations/boundary injection, and emits the final model
  generation marker. ChatML selection remains unchanged for Qwen vocabularies.
- Llama 3 header formatting matches the pinned
  `test-chat-template.cpp` vector: system/user/assistant headers, trimmed
  content, EOT delimiters, and the final assistant header. It validates role
  alternation and rejects all Llama 3 boundary tokens in message content.
- Authenticated `POST /apply-template` exposes the selected native formatter
  without inference, with bounded/strict JSON parsing and llama.cpp-compatible
  `{"prompt": ...}` output.
- Authenticated `/tokenize` and `/detokenize` endpoints use the loaded Go
  vocabulary. Tokenization accepts text or a flat mixed integer/string
  sequence, applies `add_special` only when the first component is textual,
  defaults `parse_special` to true, and can return pinned `{id,piece}` objects.
  Non-UTF-8 pieces become integer byte arrays. Detokenization concatenates
  exact pieces with special-token rendering enabled. Both reject malformed
  JSON, oversized arrays, and out-of-range IDs.
- Authenticated `GET /props` mirrors the pinned llama.cpp property envelope
  with normalized generation defaults, total slots, capability flags, model
  identity/path/file type, original GGUF chat template, BOS/EOS pieces, and a
  bounded architecture summary. Its values come from an immutable Runner
  snapshot; unsupported mutable `POST /props` is rejected with HTTP 405.
- Public `GET /models` and `/v1/models` return the pinned dual discovery
  envelope: Ollama-style `models` plus llama.cpp/OpenAI-style `data`. The
  immutable Runner snapshot supplies GGUF family, vocabulary type/count,
  train/runtime context, embedding width, quantization, overflow-safe
  parameter count, and tensor bytes while omitting the local model path.
- Authenticated native `POST /embedding` and `/embeddings` return the pinned
  non-OpenAI array with a nested pooled vector per indexed input. Both native
  and `/v1/embeddings` accept strings, exact token sequences, mixed
  token/string sequences, and bounded heterogeneous batches. Exact token IDs
  enter `EmbedTokens` directly; on real Qwen3 Q8, `[9707]` is byte-for-byte
  identical to textual `Hello` and retains unit L2 norm.
- Chat messages accept string bodies and bounded OpenAI content-part arrays.
  One to eight ordered images or one WAV-audio part in a single user turn use
  native projected generation for buffered and streaming Chat requests;
  unsupported history, mixed media, remote image URLs, tools, and multimodal
  generation options fail explicitly. Matching input-token routes run the
  projector and report the complete hard/soft-token prompt length. Public
  `/v1/health` aliases the existing health probe. Authenticated `/responses` and
  `/v1/responses` convert text or text-message inputs through the same native
  formatter and accept one to eight base64 `input_image` user parts through
  projected generation. They return pinned Responses output/usage objects.
  Streaming emits the named created/in-progress/item/content/delta/done/
  completed SSE lifecycle without a `[DONE]` marker. Flat function definitions,
  auto/none/required/named selection, replayable `function_call` and
  `function_call_output` items, schema-constrained generation, buffered
  function-call output, single/parallel call constraints, and call-complete
  argument SSE events are supported.
  `/responses/input_tokens` and `/v1/responses/input_tokens` share that
  tool-aware conversion without running inference.
- Text-only Anthropic `/v1/messages` generation supports buffered responses
  and the named message/content-block SSE lifecycle, including native stop
  reasons/sequences and cache/input/output usage without llama timing fields.
  `/v1/messages/count_tokens` supports string and multipart text system/message
  content through the identical prompt formatter. Anthropic tool definitions
  and auto/any/named selection use the same schema grammar and GGUF Jinja tool
  context as OpenAI chat. Assistant `tool_use` and user `tool_result` history,
  single/parallel call constraints, buffered tool blocks, call-complete
  streaming `input_json_delta` blocks, and tool-aware counting are supported.
- Repeatable startup LoRA paths load validated GGUF adapter pairs. Authenticated
  `/lora-adapters` reports loaded IDs, paths, and scales; POST atomically
  replaces global scales, with omitted adapters disabled. Scale changes clear
  retained prompt caches and alter serialized-session fingerprints.
- Native authenticated `POST /completion` and `/completions` reuse the same
  bounded sampler and generation core while exposing llama.cpp's `n_predict`,
  `n_cmpl`, raw-token, stop-type/word, generation-settings, prompt/token
  accounting, and timing envelope. Buffered single/multi-choice results and
  native SSE are covered; a live pinned-server differential confirmed partial
  token events, the final empty-content `stop:true` metadata event, and absence
  of OpenAI's `[DONE]` marker.
- OpenAI `/v1/completions` now shares the validated prompt parser: strings,
  exact/mixed token sequences, flat string batches, and nested heterogeneous
  batches up to 64 prompts. Exact IDs bypass tokenization, choices use stable
  prompt-major indices in buffered/SSE output, and usage counts each unique
  batch prompt once while aggregating all generated choices.
- Native prompt parsing accepts a string, an exact integer token sequence, or a
  mixed integer/string sequence. Exact IDs bypass re-tokenization in normal and
  resumable Runner generation, are cloned and range-checked, and preserve
  control/byte-token identity. A leading string segment alone applies
  configured BOS insertion. The response renders processed prompt text like
  the pinned oracle; Qwen3 Q8 exact ID `[9707]` reproduces the same
  `[9707, 27, 18, 198]` greedy sequence as textual `Hello`.
- Native multiple-prompt input supports flat string batches and heterogeneous
  batches containing nested exact/mixed token sequences, capped at 64 prompts.
  Buffered and SSE output flatten prompt-count × `n_cmpl` in prompt-major
  order with stable global indices. A pinned live oracle confirmed the order
  `Hello`, `Hello`, `Hello<`, `Hello<` for two prompts and two choices.
- Native `response_fields` projects final object, batch, and SSE metadata with
  the pinned full slash-path keys and silent omission of missing paths.
  Duplicate requests are idempotent, partial SSE token events remain intact,
  and field count/path length/component depth are bounded before generation.
- Admission uses a bounded pool of stable integer slot IDs rather than an
  anonymous semaphore. Authenticated `GET /slots` reports each slot's atomic
  busy/task state, context length, prompt processed/cache counts, raw generated
  text, zero-safe prompt/decode timing rates, and the authoritative effective
  sampler, stop, cache, and context-shift parameters passed to generation.
  `next_token` reports decoded and remaining counts; pending-token/newline flags
  are false because the synchronous Runner never retains a sampled token
  between steps.
  Completed request metrics remain available on the idle slot until its next
  acquisition.
  `fail_on_no_slot` returns HTTP 503.
  Native `id_slot` acquires the exact requested slot and returns HTTP 429 if it
  is occupied. Slot allocation is capped at 65,536 entries.
- Authenticated `/chat/completions` aliases the OpenAI-prefixed route.
  `/chat/completions/input_tokens` and `/v1/chat/completions/input_tokens`
  reuse the selected native formatter and tokenize with the same BOS/special
  policy as generation, returning the pinned `response.input_tokens` object.
  The real Qwen3 chat prompt produces an identical ID sequence through the
  generation and count paths.
- A sparse token-DFA constrains generation to exact textual alternatives,
  permits EOS only after an accepting choice, and serializes its current state
  with a topology-bound sampler signature. Repeated CLI `-grammar-choice`
  flags and the HTTP `grammar_choices` request field use the loaded model's
  tokenizer. A real Qwen3.5 run constrained `Hello` to produce `Hello there`.
- A bounded pushdown GBNF parser supports literals/escapes, Unicode character
  ranges and negation, wildcard characters, groups, alternation, named
  nonterminals, comments, right recursion, and all llama.cpp repetition
  operators. Candidate matching carries partial UTF-8 across byte tokens,
  gates EOG on root acceptance, rejects growing left recursion, and safely
  deduplicates nullable epsilon cycles. Its acceptance corpus mirrors pinned
  upstream simple/recursive/expression/repetition cases.
- Numeric `<[id]>`, vocabulary-named `<|special|>`, and inverse `!<...>` token
  terminals match the pinned upstream pushdown behavior. On Qwen3.5, both Go
  and llama.cpp constrained `Hello` to `<|im_start|><|im_end|>`; Go asserts
  generated IDs `[248045, 248046]` while omitting control pieces from decoded
  user text.
- Lazy GBNF can activate on token IDs or regex matches. It samples
  unconstrained while waiting, buffers token-piece byte spans, begins replay
  at the first participating capture group, preserves partial-token overlap,
  and serializes/restores the waiting buffer by deterministic token-history
  replay. CLI and HTTP expose triggers. A real Qwen3.5 check waits for comma,
  then constrains `Hello` to IDs `[9419, 11, 353, 1044, 248046]`
  (`Hello, I am`); the pinned llama.cpp server independently produced the
  identical `", I am"` continuation from the same prompt, grammar, and
  capture trigger.
- Sampler state v4 records variable-length accepted-token replay for GBNF and
  adaptive-p EMA state, validates it before mutation, and loads compatible
  v2/v3 state. CLI
  `-grammar`/`-grammar-file` and HTTP `grammar` fields bind source to the real
  model vocabulary. On Qwen3.5, Go and pinned llama.cpp both greedily produced
  `Return a boolean object:{"ok":true}` under the same recursive whitespace
  and boolean-alternative grammar, with identical Go token IDs asserted in
  the CUDA integration suite.
- Ordered JSON-Schema conversion matches all 70 shared cases in the pinned
  llama.cpp `test-json-schema-to-grammar.cpp` corpus plus its two C++-only
  non-capturing regular-expression cases. The fixture preserves source object
  order and covers references (including array JSON pointers and cycles),
  unions, constants/enums, bounded integers/strings/arrays, object required and
  additional-property rules, `allOf`, and anchored regular expressions. Every
  successful oracle grammar also compiles through the Go GBNF parser. The
  `json-schema-grammar` CLI accepts a file or stdin; native `/completion` and
  `/completions` compile `json_schema` with root `root` before generation and
  reject conflicts with explicit grammar controls. OpenAI `/v1/completions`
  accepts the same top-level schema. Chat completions accept top-level
  `json_schema` plus `response_format` modes `text`, `json_object`, and the
  wrapped OpenAI `json_schema.schema` form; all structured-output paths share
  the converter, compile before generation, and reject grammar conflicts and
  malformed envelopes.
- Teacher-forced perplexity uses batched CUDA output projection and stable
  log-sum-exp scoring, with optional per-token NLL output. Its `-ctx` mode
  reproduces llama.cpp's disjoint-window/latter-half convention: on the local
  Qwen3.5 fixture and a 121-token repeated-text corpus, Go reports `1.079212`
  versus the pinned CPU oracle's `1.0785` (0.066% relative difference).
- KV cache state has a versioned little-endian save/load format with strict
  rank/shape/count/layer/truncation checks. A real Qwen3 save→load→incremental
  decode matches full forward within `3.0517578125e-05` maximum absolute error.
- Cache state v2 separates active attention length from the absolute
  next-token position and loads v1 state compatibly. `ShiftCache` removes a
  dense-attention prefix without aliasing its input, retains Qwen3.5 recurrent
  state, and preserves absolute RoPE positions. Opt-in rolling context is
  available through generation/session options and the CLI/server
  `-context-shift` flag. A real Qwen3 CUDA session continued beyond an
  artificially reduced two-token context with active/history/position
  invariants intact.
- Native Q1_0/Q2_0, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1/Q8_K, Q2_K-Q6_K,
  TQ1_0/TQ2_0, IQ4_NL/IQ4_XS, and MXFP4/NVFP4 CUDA
  embedding and matrix multiplication consume persistent raw GGUF block
  layouts directly; other tensors share the F32 device bridge. Device results
  match the host decoders on synthetic packed block fixtures.
- Greedy generation on the local Qwen3 4B model matches the pinned llama.cpp
  CPU oracle for three generated tokens: prompt `Hello` produces IDs
  `[9707, 27, 18, 198]`, or `Hello<3\n`.
- The native-Q8 path completes this three-token check in about 4.6 seconds
  including `go run` startup, versus about 8.4 seconds for full F32 preload.
- Q6_K's 210-byte super-block layout is decoded on the host and consumed
  directly by native CUDA embedding/matrix kernels. The local Qwen3 4B Q6_K
  fixture produces the same `[9707, 27, 18, 198]` oracle sequence in about
  3.8 seconds including `go run`, versus about 10.5 seconds through F32 preload.
- Q2_K, Q3_K, Q4_K, and Q5_K host decoders implement the pinned 256-value
  super-block layouts with exact packed scale/min/high-bit tests.
- The HTTP surface provides health/model discovery and bounded non-streaming or
  SSE-streaming `/v1/completions`, with cancellation, admission control, and
  graceful process shutdown.
- `/metrics` emits dependency-free Prometheus text for uptime, total and active
  HTTP requests, generation attempts/errors, and generated tokens. Accounting
  is atomic across concurrent streaming and non-streaming requests. Runner
  instances also export current/peak CUDA bytes and live allocation count;
  failed optional snapshots do not fail the metrics endpoint.
- Optional bearer authentication protects generation/embedding `/v1/*` routes and the
  llama.cpp-compatible `/apply-template`, `/tokenize`, `/detokenize`, and
  `/props`, `/slots`, `/completion`, `/completions`, `/chat/completions`,
  chat token-count, `/embedding`, and `/embeddings` utility routes with
  constant-time token comparison. Secrets are loaded from
  `OVERGO_API_KEY` or `-api-key-file`; health and metrics remain available
  to orchestration probes. Native and OpenAI model-discovery routes remain
  public like the pinned server.
- `/v1/embeddings` accepts bounded string batches and returns mean-pooled,
  L2-normalized final hidden states with token usage. A real Qwen3 native-Q8
  integration check validates embedding width and unit norm.
- `/v1/chat/completions` provides non-streaming and SSE responses for ChatML
  vocabularies. Formatting validates roles, rejects boundary injection, and
  tokenizes boundary markers as control tokens; a real Qwen3 decode passes.
- Dense Llama graph construction uses consecutive-pair normal RoPE rather
  than Qwen3's split-half NeoX layout. Converted Llama 3
  `rope_freqs.weight` tensors apply one validated divisor per rotary pair on
  both the CPU reference and CUDA paths, including streamed, preloaded-F32,
  and native-quantized execution. LongRoPE short/long factors are selected
  from configured versus original context. YaRN supports interpolation,
  extrapolation, magnitude scaling, and optional per-pair factors. Metadata-driven
  linear scaling is applied to Llama normal RoPE and Qwen/Gemma NeoX RoPE and
  composes with per-pair factors.
- Dense attention Q/K/V/output and feed-forward gate/up/down projection biases
  are shape-validated, loaded through streamed, preloaded-F32, and
  native-quantized paths, and applied with broadcast additions on CPU and
  CUDA.
- Qwen3.5 prerequisite graph operations Sigmoid, numerically stable Softplus,
  and row-wise L2Norm match between the CPU reference and CUDA executors.
- Qwen3.5 projection layout extraction, rank-2 transposition, axis-zero
  concatenation, channel-wise SSM convolution, and the fused K=1 gated delta
  net match the CPU reference on CUDA for scalar/vector gates and shared Q/K
  heads.
- Multi-axis RoPE implements the pinned adjacent-pair temporal/height/width/extra
  section mapping on CPU and CUDA. Qwen3.5 also honors an explicit
  `attention.recurrent_layers` boolean array when present, with the
  full-attention interval retained as the upstream-compatible fallback.
- The local Qwen3.5 9B Q8_0 hybrid model executes all 32 layers: 24 recurrent
  gated-delta-net layers and eight gated full-attention layers. Greedy
  generation exactly matches the pinned llama.cpp oracle for three generated
  tokens: `Hello` produces IDs `[9419, 11, 353, 1044]`, or `Hello, I am`.
- The local Bonsai 27B Q1_0 Qwen3.5 hybrid executes all 64 layers with 498
  tensors retained in native one-bit CUDA storage. For two greedy tokens,
  both Go and the pinned CPU `llama-completion` oracle produce `Hello, I`;
  Go exposes exact IDs `[9419, 11, 353]`. The optional
  `OVERGO_BONSAI_MODEL` integration test preserves this differential.
- Qwen3.5 hybrid cache serialization carries fixed convolution/delta-net state
  for recurrent layers and append-only KV tensors for full-attention layers.
  A save/load session resumed after the first generated token exactly
  reproduces the oracle sequence.
- Qwen3.5-MoE reuses both hybrid cores and replaces every trunk dense FFN with
  normalized softmax top-k routed SwiGLU experts plus a sigmoid-gated shared
  SwiGLU expert. Separate and fused expert gate/up catalogs are accepted;
  metadata, strict catalog, graph semantics, and full-attention/recurrent
  reference/CUDA differentials pass. Qwen3.5 and Qwen3.5-MoE accept the pinned
  single dense NextN block after the trunk. `NewQwen35MTPSession` captures the
  target final-normalized hidden row; `AdvanceQwen35MTP` applies the pinned
  embedding/hidden RMSNorm fusion, independent MTP KV cache, gated attention,
  dense SwiGLU, optional private embedding/norm/head fallbacks, and returns an
  immutable next session for caller-controlled accept/rollback. Standalone
  MTP-only sidecars pair with an exact architecture/shape/vocabulary target;
  ordinary sidecar forward fails closed. The greedy coordinator creates
  bounded probability-filtered drafts, target-verifies one token at a time,
  commits only the accepted prefix, and replaces approximate draft hidden rows
  with target final-normalized rows. The sampled coordinator captures complete
  post-filter draft distributions, applies probability-ratio acceptance and
  positive-residual correction, and transactionally restores grammar, adaptive,
  Mirostat, and RNG state at the accepted boundary. Catalog, CPU, and CUDA differentials pass;
  the bounded resumable state carries trunk/MTP caches, pending hidden row,
  absolute positions, and independent draft/target fingerprints with
  corruption and cross-model rejection coverage. Real-model MTP validation
  awaits a local fixture.
- Qwen3-Next executes gated NeoX full attention and recurrent GDN layers with
  optimized QKV-plus-gate or legacy grouped QKVZ projection layouts. Its fused
  beta/alpha projection, adjacent value-head key-group repetition, normalized
  routed SwiGLU experts, and sigmoid-gated shared expert match the pinned
  graph. Metadata, both recurrent catalogs, host/F32/native tensor propagation,
  graph topology, and full-attention/recurrent reference/CUDA differentials
  pass. Real-model validation awaits a local fixture.
- The real Gemma 3 12B IQ4_XS file now passes strict metadata and all 48-layer
  weight catalog validation. Its graph executes with embedding scaling,
  per-head Q/K norms, alternating local/global scaled RoPE, sliding attention,
  attention/FFN post norms, GEGLU, and optional metadata-driven final-logit
  softcapping on native IQ4_XS/Q6_K CUDA weights. Softcapping is applied
  consistently to streamed, preloaded/device-cache, and perplexity logits.
  The oracle-compatible evaluator now resets BOS at every disjoint window, as
  llama.cpp does for BOS-enabled vocabularies. On a diverse 126-token probe,
  Go reports PPL `33.2276` versus the pinned CPU oracle's `33.1217` (0.32%
  relative delta). The pinned CPU server now provides raw generation IDs:
  for `Hello`, both runtimes produce generated IDs
  `[255999, 1018, 3689]`, yielding `Hello**What` after the prompt IDs.
- Gemma 2 metadata and dense graphs are supported with the upstream defaults
  for alternating 4096-token sliding attention, split-half NeoX RoPE,
  embedding and query scaling, attention-logit and final-logit softcapping,
  post-attention/post-FFN norms, and GEGLU. Attention softcapping has CPU/CUDA
  differential coverage and kernel ABI v3 validation. The tensor catalog and
  graph currently have synthetic coverage; real-model oracle validation is
  pending a local Gemma 2 fixture.
- Original Gemma metadata and dense graphs are supported with tied output
  embeddings, embedding/query scaling, split-half NeoX RoPE, and GEGLU without
  the later Gemma post norms. Metadata, tensor-catalog, and graph coverage is
  synthetic pending a local real-model fixture.
- Dense Qwen 2 metadata, tensor catalogs, and graphs are supported with
  separate Q/K/V biases, NeoX RoPE, RMSNorm, and SwiGLU. Optional
  vocabulary-wide output bias is applied before final-logit softcapping in
  streamed, preloaded/device-cache, and perplexity execution. Real-model
  differential validation remains pending a local Qwen 2 fixture.
- InternLM2 and EXAONE metadata, dense tensor catalogs, and graphs are
  supported. Both use RMSNorm, separate Q/K/V projections, and SwiGLU;
  InternLM2 uses normal consecutive-pair RoPE and requires an untied output
  projection, while EXAONE uses NeoX RoPE with optional per-pair factors and
  an optional tied output fallback. Real-model validation is pending fixtures.
- EXAONE 4 decoders support post-only attention/FFN RMSNorm, fused or separate
  QKV with optional bias, per-head Q/K RMSNorm, NeoX RoPE with optional
  factors, SwiGLU, and tied or untied output. The 64-layer cadence uses a 4096
  sliding window with RoPE on sliding layers and periodic full attention
  without RoPE. NextN/MTP layers are rejected.
- XVERSE uses the same strict RMSNorm/SwiGLU dense catalog and normal RoPE as
  InternLM2, including a required untied output projection. Metadata, catalog,
  and graph coverage is synthetic pending a real-model fixture.
- OLMo uses unweighted LayerNorm, normal RoPE, and optional separate-QKV
  clamping through the shared CPU/CUDA clamp path. OLMo2 post-normalized dense
  blocks are supported with full-projection Q/K RMSNorm, attention and FFN post
  norms, NeoX RoPE, and optional four-layer
  sliding-attention cadence. Streamed and preloaded weight adapters accept its
  intentionally absent attention/FFN pre-norm tensors. Real-model validation
  is pending a local fixture.
- SmolLM3 dense decoders support metadata-driven attention scale and the
  upstream cadence that leaves every fourth block without RoPE. Its remaining
  layers use normal consecutive-pair RoPE with the standard RMSNorm/SwiGLU
  catalog. Real-model validation is pending a local fixture.
- MiniCPM dense decoders support their backward-compatible and metadata-
  overridden embedding, residual-branch, and inverse-logit scales across host
  and retained-device inference. Standard and linear-scaled normal RoPE are
  supported, including context-selected LongRoPE factors and attention scaling,
  plus normal-layout YaRN with optional per-pair factors. Real-model validation is pending
  a local fixture.
- Granite and GraniteMoE decoders support metadata-driven embedding,
  residual-branch, attention, and inverse-logit scales, optional no-RoPE mode,
  and LongRoPE short/long factor selection. Either the Granite or GraniteMoE
  architecture may select normalized softmax top-k gated or ungated SiLU
  experts plus an optional shared SwiGLU expert.
  Granite Vision 4.1 metadata maps projected deepstack streams into decoder
  layers before attention; token-only calls leave those streams zero-filled.
  The internal projector supplies the base and mapped deepstack streams.
  Real-model validation is pending a local fixture.
- Maincoder dense decoders support normal consecutive-pair RoPE followed by
  per-head Q/K RMSNorm, preserving the upstream operation order, with the
  standard tied-output RMSNorm/SwiGLU catalog. Real-model validation is pending
  a local fixture.
- Mistral 3 decoders support normal consecutive-pair RoPE, optional
  projection/MLP biases, optional tied output, and metadata attention scale.
  Position temperature multiplies the rotated query using the pinned
  original-context step schedule. MoE variants load normalized softmax top-k
  packed SwiGLU experts through bounded-host, F32-preload, and native-quantized
  execution. Metadata, catalog, topology, and reference/CUDA differentials
  pass. LongRoPE variants use context-selected factors. YaRN combines normal
  rotary layout, optional per-pair factors, beta interpolation, attention
  scaling, and the pinned log-multiplier correction. Real-model validation awaits
  a local fixture.
- Orion dense decoders support affine LayerNorm (including attention, FFN, and
  final-output biases), split-half NeoX RoPE, parallel SwiGLU, and the
  required untied output projection. Host and preloaded-device weight adapters
  carry every norm bias. Real-model validation is pending a local fixture.
- StarCoder2 dense decoders support affine LayerNorm, split-half NeoX RoPE,
  required attention-output/up/down projection biases, sequential GELU FFNs
  without a gate tensor, and optional tied output embeddings. Real-model
  validation is pending a local fixture.
- CodeShell reuses the affine-LayerNorm, NeoX-RoPE, sequential-GELU path with
  required projection biases and an untied output projection. Its optional
  token embedding correctly falls back to the output table when absent.
  Real-model validation is pending a local fixture.
- Baichuan selects the 32-layer 7B normal-RoPE or 40-layer 13B ALiBi graph,
  with RMSNorm, parallel SwiGLU, and required untied output projection.
  Real-model validation is pending a local fixture.
- Arcee dense decoders support RMSNorm, metadata attention scaling, optional
  tied output embeddings, normal RoPE with optional per-pair factors, and the
  gate-free squared-ReLU FFN. Real-model validation is pending a local fixture.
- Nemotron dense decoders support affine LayerNorm, split-half NeoX RoPE,
  gate-free squared-ReLU FFNs, required untied output weights, and optional
  attention-output/up/down biases. Real-model validation is pending a local
  fixture.
- Jais2 dense decoders support affine LayerNorm, split-half NeoX RoPE, required
  Q/K/V/output and MLP biases, gate-free squared-ReLU FFNs, and optional tied
  output embeddings. Non-MHA metadata is rejected because upstream Jais2 bias
  tensors use the full embedding width. Real-model validation is pending.
- Original OLMo dense decoders support tensor-less LayerNorm before attention,
  FFN, and output, plus normal RoPE, SwiGLU, optional tied output, and
  metadata-driven separate-Q/K/V clamping.
- Seed-OSS dense decoders support RMSNorm, metadata attention scaling, NeoX
  RoPE, SwiGLU, and optional tied output. Its `post_attention_norm.weight` is
  mapped to the pre-FFN normalization slot matching upstream execution.
- Dense Cohere2 decoders support weight-only LayerNorm, parallel attention and
  SwiGLU residual branches, partial normal RoPE on sliding layers, periodic
  full attention without RoPE, direct logit scaling, and tied output.
- Cohere2-MoE decoder trunks support RMSNorm or weight-only LayerNorm, leading
  dense SwiGLU blocks, per-layer or periodic sliding attention, dense-prefix
  and sliding-layer normal RoPE, metadata-normalized sigmoid top-k routing, fused or
  separate expert gate/up storage, optional shared-expert half averaging,
  direct logit scaling, and optional tied output. A declared NextN block supports
  combined, trunk-only, and sidecar layouts, independent MTP state, greedy and
  stochastic verification, cache resynchronization, transactional samplers,
  and bounded model-bound serialization.
- Command R decoders support weight-only LayerNorm, parallel attention and
  SwiGLU residual branches, normal RoPE, optional direct logit scaling, tied
  output, and the 64-layer variant's per-head weight-only Q/K norms.
- Original PLaMo decoders support RMSNorm, parallel attention and SwiGLU
  residual branches, NeoX RoPE, grouped-query attention, and untied output.
- PLaMo 2 decoders support metadata-selected fused-QKV attention and recurrent
  Mamba layers, per-head Q/K RMSNorm, learned B/C/dt RMSNorm, post-mixer and
  post-FFN RMSNorm, fused gate/up SwiGLU, and serializable hybrid cache.
  Strict metadata/catalog, topology, cache, and complete CUDA differential
  tests pass; real-model validation remains pending a local fixture.
- PLaMo 3 decoders support per-layer attention/FFN widths, fused QKV,
  pre-RoPE per-head Q/K RMSNorm, NeoX RoPE with scalar or per-layer sliding
  attention, attention/FFN post-norms, fused gate/up SwiGLU, and optional tied
  output. Metadata, strict catalog, graph, and reference/CUDA block
  differentials pass; real-model validation remains pending a local fixture.
- StableLM decoders support affine LayerNorm, partial NeoX RoPE, SwiGLU,
  optional per-head weight-only Q/K norms, and both sequential-normalized and
  parallel-residual FFN layouts selected from the validated tensor catalog.
- Phi-2 decoders support affine LayerNorm, contiguous fused or separate Q/K/V
  projections with optional biases, partial NeoX RoPE, query pre-scaling,
  parallel GELU FFNs, required projection/output biases, and untied output.
- GPT-NeoX decoders support affine LayerNorm, required contiguous fused QKV
  projections and biases, partial NeoX RoPE, GELU FFNs, and metadata-selected
  parallel or sequential residual topology.
- Falcon 7B/40B decoders support affine LayerNorm, required contiguous fused
  QKV, full NeoX RoPE, parallel GELU FFNs, tied or untied output, and the 40B
  variant's optional second attention normalization.
- Dense Phi-3 decoders support RMSNorm, fused or separate QKV, query
  pre-scaling, fused gate/up SwiGLU FFNs, tied or untied output, and validated
  LongRoPE short/long factors selected from configured versus original context.
- PhiMoE reuses the Phi-3 attention path with affine RMSNorm biases, mandatory
  output/attention biases, and normalized softmax top-k packed SwiGLU experts.
- BitNet decoders support RMSNorm, attention/FFN sub-layer RMSNorms, optional
  scalar projection scales, SwiGLU, tied output, and existing ternary tensor
  decoding across streamed host and device execution paths.
- Apertus decoders support RMSNorm, fused or separate QKV with optional biases,
  per-head Q/K RMSNorm, full NeoX RoPE with ordinary or LongRoPE factors,
  metadata attention scaling, ungated xIELU FFNs with scalar-or-per-layer
  parameters, optional attention-output bias, and required untied output.
- GLM4 decoders support pre/post RMSNorm, fused or separate QKV with optional
  bias, partial normal or four-axis multimodal RoPE, projected embedding
  overrides, fused gate/up SwiGLU, and tied or untied output. NextN/MTP layers
  are rejected.
- The real UMT5 XXL encoder passes strict 24-layer T5 metadata and tensor
  catalog validation. Its SentencePiece UGM tokenizer produces
  `[23231, 3914, 332]` for `Hello world!`, exactly matching the pinned
  tokenizer oracle, and adds EOS token `1` for encoder execution.
- The T5 graph executes full bidirectional no-cache attention with per-layer
  32-bucket relative-position bias, RMSNorm, GEGLU, residuals, and final
  encoder normalization on native Q5_K/Q6_K CUDA weights. For `Hello world!`,
  the 4,096-dimensional mean-pooled vector has cosine similarity `0.9994314`,
  RMSE `0.0012743`, and a `0.17%` norm delta versus pinned llama.cpp CPU
  execution. The oracle command must specify `--attention non-causal`; its
  default for this fixture is causal.
- Full T5 encoder-decoder catalogs execute through a persistent `T5Session`.
  Decoder self K/V grows by token; encoder-derived cross K/V remains fixed in
  named cache state. Classic T5 ReLU and Flan gated-GELU FFNs, causal absolute-
  offset relative buckets, bounded-host/F32-preload paths, model-bound session
  serialization, corruption checks, cache validation, and complete CUDA block
  differentials pass. Real decoder-model validation remains pending a fixture.

## Historical runtime notes

These cumulative notes preserve the runtime boundary at each implementation
milestone. `IMPLEMENTATION_STATUS.md` and generated `docs/COMPATIBILITY.md` are
the authoritative current views; later entries below supersede earlier pending
statements.

- Preloaded dense generation retains KV state on-device between ordinary
  decoding calls and device-backed prompt reuse can require zero submissions.
  Qwen3.5 direct generation likewise retains hybrid state on-device. Editable
  sessions and streamed-weight execution use the serialized host cache.
  Ordinary rolling context shift advances attention-cache device pointers and
  shrinks logical token dimensions while preserving absolute positions and
  recurrent state. Combined immutable prompt caching plus shifting clones the
  retained device ranges before the first edit. Native completion `n_keep` performs exact prefix-preserving
  middle-range compaction in owned CUDA allocations using direct
  device-to-device range copies; zero-prefix shifts remain pointer views.
  `n_discard` accepts an explicit bounded removal size, with zero using the
  pinned half-window default. Retained CUDA caches expose fixed-width pointer
  pages rebuilt after append, trim, pointer shift, and owned compaction.
  `ContinuousBatch` owns sequence-tagged host or device caches, supports
  transactional multi-sequence append, dynamic admission/removal, host/device
  forks, context shifting, and sorted page/state snapshots.
- Native device execution currently supports Q1_0/Q2_0, Q4_0/Q4_1,
  Q5_0/Q5_1, Q8_0/Q8_1/Q8_K, Q2_K-Q6_K, TQ1_0/TQ2_0,
  every pinned IQ1/IQ2/IQ3/IQ4 layout, and MXFP4/NVFP4 matrices. This covers
  every quantized tensor type currently declared by the pinned GGML format
  map; scalar F16/BF16/F64 and integer storage still use F32 conversion.
- Default-mode tied-output logits stream bounded GGUF chunks on the CPU;
  preload modes compute logits on the GPU.
- CUDA memory metrics cover allocations owned by this runtime, not driver
  overhead or allocations made by other processes in the same device context.
- Character, token-terminal, and lazy-trigger GBNF are supported. Trigger
  patterns use bounded ECMAScript Unicode matching with lookaround and
  backreferences. General and infill sampler stages have arbitrary ordering.
- Text and function-tool GGUF Jinja templates support the pinned bounded
  exception, date, namespace, and range globals.
  Fused continuous-batching server scheduling is supported. Tool-enabled streams
  incrementally parse JSON and Hermes output while retaining complete-output
  schema validation.
- Chat generation/counting accepts string message bodies, text content-part
  arrays, tool schemas, assistant calls/results, and the implemented
  system/user/assistant/tool role families. Buffered and token-incremental SSE
  OpenAI function calls plus tool-aware token counting are supported;
  single-turn projected multi-image or single-image/audio generation is supported.
  Projected multimodal input-token counting uses the same prompt path. Media
  history, mixed media, remote images, files, and tools with media remain pending.
- OpenAI Responses generation, streaming, token counting, function
  tools/history, and single-turn projected base64 multi-image input are supported.
  Continuation IDs, hosted/custom tools, reasoning items, media history,
  mixed media, remote images, and audio/file inputs remain pending.
- Text-only Anthropic generation, streaming, counting, tool use/results, and
  tool-aware Jinja contexts are supported, including incremental tool-input
  deltas. Thinking blocks and images remain pending.
- LoRA loading, alpha/rank scaling, graph-wide tensor application, global
  control-plane scaling, prompt-cache invalidation, and session binding are
  implemented. Native per-request overrides restore global scales and isolate
  retained prompt caches by adapter signature. A single enabled aLoRA uses the
  last invocation-token match as its activation boundary; unmatched requests
  disable it and multiple simultaneous aLoRAs are rejected.
- Native completion supports strings, exact/mixed token sequences, bounded
  batches of either, and opt-in reuse from the best retained prefix meeting
  `n_cache_reuse`. It reports cached/evaluated token counts and
  separate prompt/decode timing. Context shifting accepts `n_keep`, including
  `-1` for the largest safe initial-prompt prefix, plus `n_discard`, and reports
  both in generation settings and slot state. Pure-attention prompt caches
  reuse their longest common prefix through host or zero-copy device suffix
  rollback; recurrent models retain exact-prefix-only reuse. A configurable
  bounded LRU retains independent host or CUDA prompt states. Native multimodal
  prompt objects project up to eight images or one WAV audio item after slot
  admission.
  Per-request LoRA arrays replace scales for one generation without mutating
  global control-plane state. Positive `n_probs`
  reports selected and top-N raw-logit softmax log probabilities with token
  pieces and byte arrays in buffered and SSE responses.
  `post_sampling_probs` instead reports normalized `prob`/`top_probs` from the
  sampler's actual filtered candidate set, including greedy, Mirostat, and
  adaptive paths. Native SSE
  supports the pinned 30-second default comment heartbeat, positive integer
  `sse_ping_interval` overrides, and `-1`/`0` disabling. `return_progress`
  emits exact prompt-start and prompt-complete envelopes; internal per-batch
  increments are unavailable because the Runner evaluates a prompt as one
  synchronous graph. `timings_per_token` adds measured cache/prompt/decode
  timings and zero-safe rates to each native token event. Positive
  `t_max_predict_ms` uses the Runner's graceful post-token stop hook to end on
  the first newline after the prediction budget, preserving final IDs/text;
  `0` and `-1` disable the limit. Positive `n_indent` stops when a generated
  post-newline line has insufficient leading spaces/tabs and trims the
  offending non-whitespace suffix.
- `/slots` exposes authoritative admission state, task ID while busy, context
  length, prompt/cache counters, prompt and generated text, and measured timing
  rates. Its `params` object retains the effective sampler, stop, cache, and
  context-shift settings. Its `next_token` object reports decoded/remaining
  counts and truthfully reports no pending token in the synchronous runtime.
- Llama execution covers dense and metadata-selected MoE models with optional
  projection biases, ordinary or converted Llama 3 per-pair frequency factors,
  context-selected LongRoPE factors, and normal-layout YaRN with optional
  per-pair factors. Remaining model-specific attention variants are rejected.
- The Qwen3.5 graph supports true multi-axis positions. `Generate`,
  `generate -projected-inputs`, and single-prompt native `/completion` requests
  accept caller-projected soft tokens, explicit four-axis prompt positions, and
  deepstack streams. `generate -mmproj -image` executes the Qwen3-VL
  `qwen3vl_merger` patch encoder, 24-layer bidirectional ViT, merger, exact
  image prompt, and compressed image-grid positions. A local Qwen3.5-4B F16
  run reproduced the 85-token Transformers prompt and first generated token.
  Server `-mmproj` exposes the pinned native `prompt_string` plus base64
  `multimodal_data` object through buffered and streaming completion paths;
  projection runs inside slot admission.
  Repeatable CLI `-video-frame` input adds temporal-pair preprocessing,
  timestamped video chunks, per-chunk compressed MRoPE, and odd-frame padding.
  The local 16-frame oracle matches preprocessing, full ViT/merger output,
  all 478 prompt IDs, and the first generated token. Optional `-mmproj-cuda`
  keeps the dual patch embeddings, position table, 24 ViT blocks, and merger
  resident, with fused 2D vision RoPE and temporal-group non-causal attention;
  nonzero image and multi-pair video fixtures match the CPU graph. A real
  Qwen projector GGUF is not present locally for the CUDA oracle. Encoded video
  input composites animated GIF natively and uses bounded, FPS-selected FFmpeg
  PNG streaming for MP4/WebM/MOV and other installed-codec formats. Native,
  Chat, and Responses requests accept up to eight ordered images; the projector
  merges their soft tokens and constructs per-image compressed four-axis MRoPE.
  Recurrent and hybrid continuous batches retain independent primary and named
  state within one variable-branch CUDA graph. Real-model family differentials
  remain fixture-gated.
- Gemma 4 unified image projection executes the encoder-free `gemma4uv` GGUF
  graph: pinned dynamic aspect-preserving bicubic resize, 48x48 RGB patch rows,
  affine patch LayerNorm and dense projection, factorized learned X/Y
  positions, position LayerNorm, unweighted RMSNorm, and the final 3840-wide
  projection. The local
  12B oracle matches projector probes/L2 and native Q8 generation. Pinned
  dynamic resizing retains already-budgeted images and uses 40-to-280 visual
  tokens. Mixed hard/soft input now scales only hard Gemma token rows.
  CLI and native completion server image paths select Qwen or Gemma from GGUF
  projector metadata. Multi-image server requests concatenate images-first
  soft-token blocks and preserve a distinct visual attention block per image.
  The encoder-free `gemma4ua` path pads mono 16 kHz audio
  into 640-sample rows, applies BF16-compatible unweighted RMSNorm and the
  640-to-3840 projection, renders the exact audio turn, and accepts raw F32 or
  PCM16/float32 WAV through the CLI plus WAV data URIs through native
  completion. The local projector matches all adaptive probes/L2 and all 34
  prompt IDs; F16 generation is token-exact at first token `24068` (`Music`),
  while native Q8 selects a reference top-8 token. Gemma video projection uses
  the encoder-free image path with a 70-token frame budget, timestamped prompt,
  frame-major soft tokens, and frame-block visual attention. Optional
  `-mmproj-cuda` execution keeps all Gemma image/audio weights resident on the
  selected device and runs image, video, and audio graphs with explicit BF16
  stage rounding. `gemma4-gguf-convert` now streams the adaptive
  ModelOpt checkpoint into validated BF16 language and multimodal GGUF files,
  including FP8 scale folding, patch-channel permutation, position-axis
  transposition, tokenizer metadata, and layer-scalar conversion. The converter
  emits the required proportional-RoPE factors and complete
  unified projector metadata; `-mmproj-f32` supplies a CPU-baseline artifact.
  Chat and Responses retain images at their exact formatted positions across
  user/assistant history. The pinned 12B Q8 oracle and the native implementation
  match at 213 tokens for earlier-turn media and 214 for final-turn media.
  Prompt-cache entries bind exact projected embeddings, positions, deepstack
  streams, and attention blocks so equal placeholder IDs cannot alias different
  images.
  The MTMD-style heterogeneous path now preserves ordered image/audio chunks
  across native, Chat, and Responses history. Gemma 4 image ranges remain
  bidirectional while audio ranges remain causal. Pinned llama.cpp and native
  Go both produce 221 prompt tokens for the same one-second audio plus image
  fixture in either order. Remote image/audio URLs are opt-in through the
  top-level `media_policy.yaml`; exact scheme/host/port allowlists, DNS and
  redirect revalidation, private/reserved-network rejection, independent
  connect/header/total timeouts, MIME checks, response and aggregate byte
  limits, and a fetch-concurrency bound define the SSRF boundary.
- T5 single-sequence encoder-decoder sessions are supported. `GenerateT5`
  supplies high-level sampling, stop callbacks/sequences, LoRA selection, and
  context shifting; callers can also generate incrementally with `DecodeT5`.
  Exact source-token matches reuse LoRA-isolated cached encoder output;
  prefix-only matches are re-encoded because source attention is bidirectional.
  Rectangular multi-sequence source and decoder batches carry explicit active
  lengths; padding suffixes are excluded before graph construction, so they
  cannot enter encoder attention, decoder attention, cross attention, or cache.
  Decoder relative buckets support active-range cache deletion; fixed encoder
  cross-attention K/V remains unchanged while self-attention rows compact.

## Blockers

### Implemented: DeepSeek 3.2 and GLM-DSA sparse attention

DeepSeek 3.2 executes absorbed MLA with a full lightning indexer in every main
layer. The indexer has its own serializable key cache, affine key normalization,
NeoX YaRN query/key rotation, FWHT mixing, ReLU scoring, learned head reduction,
top-k history selection, and sparse causal attention. Declared NextN/MTP layers
remain preserved but outside main-model execution. GLM-DSA uses the same sparse
path with metadata-selected full-indexer layers and graph-resident top-k reuse in
following shared-indexer layers. Reference and CUDA block oracles cover both
schedules; real-model validation awaits local GGUF fixtures.

### Implemented: DFlash paired-target drafting

DFlash extracts configured pre-layer hidden states from a distinct target
runner, projects and normalizes the concatenated features, and injects the
resulting per-layer K/V rows into a serializable draft cache. Paired decoding
uses target token embeddings and output projection with the pinned non-causal
noise-block mask. Prefix synchronization, bounded draft blocks, reference
oracles, and CUDA pipeline parity are covered. Greedy and sampled coordinators
verify target logits, transactionally restore samplers, advance accepted target
and draft caches, and resynchronize injected features from the committed prefix.
Session setup and verification now capture configured pre-layer rows during the
same cached target forward, then fuse and inject only the new rows. The explicit
`PrimeDFlash` and `SyncDFlashPrefix` helpers retain full-prefix extraction for
custom scheduler recovery.

### Implemented: WavTokenizer audio-feature decoder

WavTokenizer maps semantic token IDs to audio-feature frames through the pinned
same-padded dense/depthwise convolutions, group normalization, PosNet residual
and attention blocks, ConvNeXt GELU blocks, affine normalization, and output
projection. The dedicated `DecodeWavTokenizer` API exposes the non-logit output
contract. `DecodeWavTokenizerWaveform` ports the pinned inverse spectral
transform, Hann window, overlap-add, and envelope normalization to produce 24
kHz mono samples. Strict weight validation, reference graph tests, waveform
known answers, and CUDA parity are covered.

### Implemented: DeepSeek 4 compressed sparse/hyper-connection graph

DeepSeek 4 executes the pinned 43-layer graph with four-stream Sinkhorn
hyper-connections, grouped attention-output LoRA, attention sinks, raw sliding
windows, ratio-4 overlapping and ratio-128 attention compression, and
serializable compressor KV/score state. Ratio-4 layers reconstruct and rank
compressed history with the lightning indexer, normal RoPE, FWHT, learned head
weights, and top-k selection. Early layers use hash-derived fixed expert IDs;
later layers use learned routing bias. Both paths apply normalized
sqrt-softplus routing, native-quantized expert banks, shared SwiGLU, and the
pinned per-layer routed/shared clamps. Reference tests cover raw and both
compressed schedules; CUDA parity covers the hyper-connection bridge,
attention bridge, and native routed-expert ABI. Real-model validation awaits a
local GGUF fixture. Prefix shifts and suffix trims remain available;
non-contiguous middle-range cache deletion is rejected because compressed
block positions cannot be preserved exactly.

### Implemented: importance-weighted IQ encoders

IQ1_S, IQ1_M, IQ2_XXS, and IQ2_XS expose an explicit weighted quantization
API and byte-match the pinned GGML routines. `gguf-quantize -imatrix` reads the
pinned GGUF and legacy binary formats, normalizes per-expert sums and counts,
maps expert-specific column weights across tensor rows, and writes the pinned
file, dataset, entry-count, and chunk-count provenance keys. Missing importance
for a selected matrix is an error; token embeddings and output projections are
preserved when their optional entries are absent. Loader, expert mapping,
metadata, CLI lifecycle, deterministic hashes, and Windows DLL differential
coverage are checked in.

### Implemented: LoRA graph-wide projection application

Pinned GGUF adapters validate type, architecture, paired suffixes, base-tensor
existence, regular/grouped matrix shapes, flipped token-embedding shapes,
alpha, and optional aLoRA invocation metadata. Graph execution computes
`base*x + scale*B*(A*x)` without destructive weight merging. Exact tensor-name
binding covers dense projections, grouped expert banks, token embeddings, and
output projections in streamed, F32-preloaded, and native-quantized paths.
Multiple adapters add in stable ID order. Global scale replacement clears
prompt caches and contributes adapter content plus scales to session
fingerprints. Native request-local scales restore global state and segregate
prompt caches by adapter signature. Synthetic GGUF loader failures, reference
projection/embedding/grouped/fused-MoE oracles, server lifecycle tests, and
CUDA parity are covered. A local real-model adapter differential remains an
independent fixture follow-up. Fused MoE uses a native CUDA graph node to form
ephemeral adapted expert weights. This preserves exact request-local scaling
and quantized-base safety, but adds a full expert-bank merge pass per graph;
direct low-rank deltas inside the fused MoE kernel remain a performance-only
optimization.

### Deferred: Go vet cannot prove CUDA C-string pointer provenance

The CUDA error-name/error-string APIs return driver-owned `const char *`
values through the Windows DLL syscall boundary, whose Go representation is
necessarily `uintptr`. The bounded `readCString` conversion is annotated
`//go:nocheckptr` and live driver tests pass, but `go vet ./...` still reports
`possible misuse of unsafe.Pointer` because its analyzer cannot track pointer
provenance across `Proc.Call`. Package tests and scoped vet checks remain
enabled; replacing this ABI conversion solely to silence the analyzer would
require an extra native shim and violate the no-cgo design.

### Deferred: release signing and project license declaration

Reproducible unsigned archives are implemented. Authenticode signing requires
a user-controlled code-signing certificate/private key and timestamping
policy, neither of which is present or safe to invent. The repository owner
also has not declared a license for the Go/kernel code; SBOM and license
inventory therefore use `NOASSERTION`. Public distribution should wait for
both decisions.

### Deferred: Go race detector under the no-cgo contract

The standard Go race detector requires cgo on this host, while this port's
deployment contract requires `CGO_ENABLED=0`. `go test -race` therefore fails
before compilation. Enabling cgo solely for that check would test a different
binary contract. Deterministic concurrency/contention tests remain enabled;
race instrumentation is deferred until Go supports it for the no-cgo Windows
target or a suitable alternative is integrated.

### Supported constraint: CUDA 12.9 compiler selection

CUDA 12.9 kernel builds pass through the installed Visual Studio 2019 Build
Tools compiler selected by `scripts/build-kernels.ps1`. Visual Studio 2026 is
not a CUDA 12.9 host compiler and is intentionally bypassed. Runtime and kernel
verification pass under this supported pairing.

### Deferred: local ternary Q2_0 fixture layout mismatch

`Ternary-Bonsai-27B-Q2_0.gguf` encodes a Q2_0 tensor with an effective 17-byte
block, while the pinned llama.cpp commit defines Q2_0 as an 18-byte block. The
reader rejects the resulting non-contiguous offsets. Do not add a heuristic
layout variant without a metadata/version discriminator. Other tested F32,
BF16, Q8_0, Q6_K, IQ4_XS, and Q1_0 models parse successfully.

### Deferred: real Llama end-to-end fixture

The available local GGUF inventory has Qwen3, Qwen3.5, Gemma, T5, diffusion,
and ternary fixtures but no real Llama weight file. The pinned upstream
Llama SPM/BPE vocabulary fixtures and synthetic dense-block CPU/CUDA tests,
including converted Llama 3 frequency factors, pass. Final logits, greedy
tokens, and performance for the Llama path remain deferred until a compatible
dense Llama GGUF is available.

### Deferred: real-model infill oracle

The pinned infill probability sampler, vocabulary-piece plumbing, FIM metadata
resolution, PSM/SPM formatter, and native endpoint are implemented. None of
the local GGUF fixtures contains FIM metadata or recognized FIM control-token
spellings, so end-to-end token/output differential validation remains
deferred until a compatible model is available.

### Implemented: fused continuous-batching server scheduling

`ContinuousBatch` now supplies sequence-tagged host/device caches, page tables,
dynamic admission/removal, transactional steps, context compaction, and
host/device forks. Each device step builds variable active sequences as independent
branches of one CUDA graph submission. Retained outputs use shared reference
ownership so removing or advancing one sequence cannot invalidate sibling
caches from the same fused execution.

Recurrent and hybrid branches retain Mamba/GDN/KDA convolution and state-space
state, RWKV token/WKV state, LFM2 short-convolution state, and Falcon-H1 named
fixed state. Token-aligned named state compacts with KV rows; fixed and primary
recurrent state survives context edits. Device forks share immutable retained
outputs until copy-on-write advance or compaction.

The HTTP generation scheduler admits queued requests between token steps and
removes completed or cancelled sequences independently. Samplers, histories,
stop sequences, callbacks, and token limits remain per sequence. Preloaded
device modes activate this path when the server has multiple slots; projected
inputs, per-request LoRA, and incompatible cache policies retain the exact
single-request fallback.

### Deferred: RWKV and PLaMo2 tokenizer oracle fixtures

The pinned source contains RWKV and PLaMo2 tokenizer implementations but no
checked-in GGUF vocabulary/oracle pair for either algorithm, and the local
model inventory has neither family. WordPiece work proceeded against the
available BERT fixture. RWKV/PLaMo2 tokenization remains deferred rather than
claiming an unverified port; independent compatibility work continues.

### Implemented: pinned fused-QKV decoder families

Contiguous fused QKV projection and bias loading/slicing is shared by the dense
host, preloaded-device, and cache execution paths for every executable
architecture that declares the layout, including Phi-2, GPT-NeoX, Falcon,
BERT, and NeoBERT. Architecture gates retain each family's position encoding,
residual topology, normalization, and tensor-layout rules. GPT-J uses separate
Q/K/V projections and the standard GGUF tensor vocabulary.

BERT now adds optional sentence-A token-type row 0 and learned absolute-position
rows before affine embedding LayerNorm. Its bidirectional no-cache encoder uses
optional fused QKV and projection/MLP biases, GELU, and affine post-attention
and post-FFN LayerNorm without a decoder output projection. Metadata, strict
catalog, graph ordering, and a complete reference/CUDA block differential pass.
The optional classifier/pooler tensors remain outside the hidden-state embedding
boundary; real-model execution is pending a local weight fixture.

NeoBERT now executes fused-QKV bidirectional attention with normal RoPE,
pre-RMSNorm residual blocks, fused SwiGLU storage, and final encoder RMSNorm.
Metadata, strict catalog, graph semantics, and a complete reference/CUDA block
differential pass. Optional classifier tensors remain outside the hidden-state
embedding boundary; real-model execution is pending a local weight fixture.

Dense NomicBERT now executes optional fused QKV, bidirectional NeoX RoPE,
affine embedding/post norms, and gated SwiGLU without a decoder projection.
Metadata, strict catalog, graph semantics, and a complete reference/CUDA block
differential pass.

NomicBERT-MoE now follows its exact alternating cadence, with dense GELU blocks
and gate-free softmax top-k GELU expert banks. Metadata, strict per-layer
catalogs, graph semantics, and a complete reference/CUDA differential pass.

JinaBERT v2 now executes fixed bidirectional ALiBi, optional fused QKV and
affine Q/K norms, optional secondary attention residual normalization, and
plain GELU or separate/fused GEGLU FFNs. Metadata, strict catalog, topology
variants, and a complete reference/CUDA block differential pass.

JinaBERT v3 now executes optional fused QKV, bidirectional NeoX RoPE,
affine embedding/post norms, and dense GELU FFNs without a decoder projection.
Metadata-selected expert layers follow the pinned cadence with gate-free
softmax top-k GELU expert banks. Metadata, strict dense/MoE catalogs, graph
semantics, and reference/CUDA block differentials pass.

Llama Embed now reuses the dense and Mixtral-style Llama tensor catalogs and
normal-RoPE transformer blocks with bidirectional no-cache attention. Final
RMS-normalized hidden states form the encoder boundary; vocabulary logits are
rejected even when an optional output projection is present. Metadata, dense
and MoE catalogs, graph semantics, and a reference/CUDA differential pass.

Pangu Embedded now executes its causal RMSNorm/SwiGLU decoder and exposes the
same final normalized states used by its optional tied vocabulary projection.
Separate or fused QKV, required attention-output bias, NeoX RoPE, and optional
LongRoPE factors are covered by strict catalogs and a reference/CUDA block
differential. Real-model execution is pending a local Pangu GGUF fixture.

ModernBERT now executes embedding LayerNorm, layer-0 identity attention norm,
fused QKV, NeoX RoPE, dense-first periodic symmetric attention, and fused
GEGLU, SwiGLU, or ReGLU selected by canonical activation metadata and aliases.
Final weight-only LayerNorm exposes
hidden states without a vocabulary projection. Metadata, strict catalog,
graph variants, and a reference/CUDA differential pass; real-model execution
is pending a local ModernBERT GGUF fixture.

Gemma Embedding now executes scaled token embeddings, per-head Q/K RMSNorm,
NeoX RoPE, periodic full and symmetric local attention, branch post-RMSNorm,
and parallel GEGLU. Optional sentence-transformer dense-2/dense-3 matrices run
after pooling and before normalization on bounded-host and preloaded CUDA
paths. Metadata, strict fused/separate-QKV catalogs, graph variants, host
projection tests, and a reference/CUDA block differential pass; real-model
execution is pending a local Gemma Embedding GGUF fixture.

Bloom now applies its input embedding LayerNorm and llama.cpp-compatible ALiBi.
GPT-2 and StarCoder gather learned absolute-position rows before the first
block. MPT supports optional learned positions, ALiBi, fused-QKV clamping,
full-projection affine Q/K LayerNorm, and AWQ post-GELU activation scaling.
The activation scale uses a broadcast divide primitive covered by reference
and CUDA differential tests.
Dense Refact uses ALiBi. Qwen3-MoE loads its router and packed expert gate/up/down
tensors and executes normalized softmax top-k routing with routed weight scaling
through the bounded-host, F32-preload, and native-quantized expert paths. This
matches the complete Qwen3-MoE definition in the pinned llama.cpp source.

Mixtral now detects expert-bearing `llama` GGUF metadata without changing the
dense Llama path, loads the router and packed SwiGLU expert bank, and executes
normalized softmax top-k routing through bounded-host, F32-preload, and
native-quantized CUDA expert paths. Metadata, strict catalog, graph semantics,
and complete reference/CUDA block differential tests pass. Real-model
validation remains pending because no Mixtral GGUF is available locally.

The fused routed-expert kernel accepts F32 or any native-quantized gate/up/down
bank supported by the device decoder set. Packed blocks remain resident in
their GGUF layout and decode inside the kernel; router, activations,
accumulation, selection bias, and output remain F32. Softmax and sigmoid
routing, selected-probability normalization, and routed scaling share this
path. CUDA differentials cover classic Q4, Q8, K-quant, IQ, ternary, float4,
and small-block layouts against their dequantized F32 references.

BailingMoE now supports its dedicated BPE pre-tokenizer and loads its required output projection, standard Q/K/V attention,
packed routed experts, and always-on shared expert bank. Its normal RoPE,
optional selected-probability normalization, routed scaling, and parallel shared
SwiGLU branch execute through bounded-host, F32-preload, and native-quantized
expert paths. Metadata, strict catalog, graph semantics, and complete
reference/CUDA block differential tests pass. Real-model validation remains
pending because no BailingMoE GGUF is available locally.

DeepSeek v1 now supports its exact `deepseek-llm` BPE pre-tokenizer, including
the pinned tokenizer's restricted alphabet and punctuation ranges. The model
loads optional leading dense SwiGLU blocks followed by unnormalized softmax
top-k routed experts, routed scaling, and the always-on shared SwiGLU expert
bank. Output projection remains optional with tied token-embedding fallback.
The complete 46-case upstream tokenizer oracle, metadata, strict mixed-layer
catalog, graph semantics, and reference/CUDA block differential pass. The local
fixture inventory contains only tokenizer data, so real-model logit validation
remains pending-fixture.

Granite and GraniteMoE now load normalized softmax top-k packed experts with
either gated SwiGLU or ungated SiLU activation, plus the optional shared
SwiGLU expert. Both retain Granite's embedding, residual, attention,
inverse-logit, no-RoPE, and LongRoPE behavior. Metadata, strict catalog, graph
semantics, primitive reference coverage, and complete reference/CUDA block
differential tests pass. Real-model validation remains pending because no
Granite MoE GGUF is available locally.

DBRX now maps its `dbrx` BPE pre-tokenizer to the pinned Llama 3 segmentation,
loads required fused QKV and untied output tensors, applies its symmetric QKV
clamp before projection slicing, and executes weight-only LayerNorm, NeoX RoPE,
and normalized softmax top-k packed SwiGLU experts. Metadata, strict catalog,
graph ordering, clamp primitive coverage, and complete reference/CUDA block
differential tests pass. Real-model validation remains pending because no DBRX
GGUF is available locally.

Grok now loads optional fused or separate QKV projections, optional gated
expert and dense GEGLU branches, required attention/FFN post-norm weights, and
optional tied output projection. Its graph applies metadata/default embedding,
attention, and direct-logit scales; NeoX or YaRN RoPE; softcapped attention;
normalized softmax top-k packed GELU/GEGLU experts; and the pinned dense/MoE
blend. Reference and CUDA differential tests cover ungated-only and
gated-plus-dense blocks. Real-model validation remains pending because no Grok
GGUF is available locally.

Mellum now loads required untied output, separate QKV and per-head Q/K RMSNorm
tensors, plus normalized softmax top-k packed SwiGLU experts. Scalar or
per-layer sliding-window patterns select bounded attention; sliding layers use
unscaled NeoX RoPE while full layers retain linear or YaRN scaling. Metadata,
catalog, alternating graph, and reference/CUDA differential tests pass.
Real-model validation remains pending because no Mellum GGUF is available
locally.

SmallThinker now loads optional fused or separate QKV projections and packed
ReGLU experts, computes router logits from the pre-attention residual while
feeding normalized post-attention state to the experts, and honors metadata
selected softmax or sigmoid routing. Its dense-first periodic sliding schedule,
4096-token effective window, SWA RoPE base, and no-RoPE full-attention layers
match the pinned implementation. Metadata, strict catalog, graph semantics,
split-router primitive tests, and complete reference/CUDA block differentials
pass. Real-model validation remains pending because no SmallThinker GGUF is
available locally.

DOTS1 now loads required untied output, optional fused or separate QKV,
per-head Q/K RMSNorm, leading dense SwiGLU blocks, and later packed routed
experts. Its metadata-selected softmax or sigmoid routing supports optional
selection correction bias, selected-weight normalization, routed scaling, and
the required shared SwiGLU bank. Metadata, strict mixed-layer catalog, graph
semantics, and a complete reference/CUDA block differential pass. Real-model
validation remains pending because no DOTS1 GGUF is available locally.

MiniMax-M2 now loads required untied output, optional fused or separate QKV,
full-projection Q/K RMSNorm, partial NeoX RoPE, and packed SwiGLU experts. Its
required F32 selection correction bias works with metadata-selected softmax or
sigmoid routing, normalized selected weights, and routed scaling. Metadata,
strict catalog, graph semantics, and a complete reference/CUDA block
differential pass. Real-model validation remains pending because no MiniMax-M2
GGUF is available locally.

Deci now preserves scalar-or-array per-layer query-head, KV-head, and
feed-forward schedules. Full-attention layers support optional fused or
separate QKV, normal or LongRoPE factors, optional projection/MLP biases, and
ordinary KV caching. Projection-only linear-attention and attention-free
layers execute their pinned residual ordering, while zero-width dummy layers
remain no-ops. Non-KV layers use compact serializable sentinel caches so cached
decode and context shifting retain a uniform layer table. Metadata, strict
mixed-layer catalog, all sparse graph modes, cache editing, and a four-mode
reference/CUDA differential pass. Real-model validation remains pending
because no Deci GGUF is available locally.

BailingMoE2 adds fused QKV projection, per-head Q/K RMSNorm, NeoX RoPE, leading
dense SwiGLU blocks, optional selection correction bias, metadata-selected
softmax or sigmoid routing, configurable shared-expert width, and always-on
shared experts. Preserved NextN/MTP layers are removed from the executable block
count. Metadata, strict dense/MoE catalog, both routing graphs, and a complete
sigmoid-bias reference/CUDA block differential pass. Real-model validation is
pending because no BailingMoE2 GGUF is available locally.

Dream now has an explicit non-causal, no-cache full-sequence execution path
for bounded-host and preloaded-device weights. It returns either all hidden
states or vocabulary logits for every position, and rejects KV-cache and
autoregressive-generation entry points.

The shared diffusion runtime and CLI execute iterative mask transfer for Dream,
LLaDA, LLaDA-MoE, and RND1. They preserve the pinned timestep and block
schedules, origin/confidence/entropy/margin/random selection, top-k/top-p/token
temperature sampling, classifier-free guidance, optional Gumbel transformation,
model-metadata logit shifting, callbacks, and cancellation. Deterministic
scheduler, ranking, CFG, shift, CLI parsing, and validation tests pass. A real
diffusion GGUF/oracle remains pending because none is present locally.

Laguna now preserves scalar-or-array per-layer query/KV head metadata and its
full/SWA cadence. Full layers use NeoX YaRN while sliding layers use their
separate plain-RoPE dimensions and frequency base. Its attention output is
softplus-gated in either per-head or per-element layout; leading blocks use a
dense SwiGLU FFN and later blocks combine sigmoid top-k routed experts,
selection correction bias, optional selected-weight normalization, and one
always-on shared expert. Synthetic catalog/graph tests and a complete
reference/CUDA block differential pass. No Laguna GGUF exists in the local
fixture inventory, so real-model logits and throughput remain pending-fixture.

AFMoE now reuses the same sigmoid/correction-bias routed expert primitive and
supports its optional leading dense blocks and zero-or-more shared experts.
Its distinct graph behavior is preserved: embeddings receive MuP square-root
scaling, attention and FFN branches have both pre- and post-RMSNorm, the
attention output uses a per-element sigmoid gate before projection, and the
four-layer cadence combines periodic no-RoPE layers with interleaved sliding
attention. Synthetic metadata/catalog tests and a complete reference/CUDA
block differential pass. No AFMoE GGUF is present locally, so real-model
logits and throughput remain pending-fixture.

Qwen2-MoE now executes unnormalized softmax top-k routed experts in parallel
with its SwiGLU shared expert. The shared branch's learned scalar-per-token
sigmoid gate is represented explicitly in the graph, including its rank-one
GGUF tensor. Metadata and complete graph tests pass; real-model validation is
pending because the local fixture inventory has no Qwen2-MoE GGUF.

OLMoE now loads its packed SwiGLU expert bank, applies learned RMSNorm across
the complete Q/K projections before head reshaping, and executes unnormalized
softmax top-k routing. Metadata, strict catalog, graph-ordering, and complete
reference/CUDA block differential tests pass. Real-model validation remains
pending because no OLMoE GGUF is available locally.

PhiMoE now loads affine RMSNorm and projection/output biases, selects its
LongRoPE factor set from configured versus original context, pre-scales the
rotated query, and executes normalized softmax top-k packed SwiGLU experts.
Metadata, strict catalog, graph-semantics, and complete reference/CUDA block
differential tests pass. Real-model validation remains pending because no
PhiMoE GGUF is available locally.

EXAONE-MoE now preserves scalar-period or explicit per-layer sliding-attention
metadata, applies per-head Q/K RMSNorm and RoPE only on local layers, supports
leading dense blocks, and combines metadata-selected softmax/sigmoid routed
experts with optional selection correction bias and an always-on shared
expert. Preserved NextN/MTP layers are removed from the executable block count.
Metadata, strict catalog, graph, and complete reference/CUDA block differential
tests pass; real-model validation is pending because no EXAONE-MoE GGUF is
available locally.

RND1 now reuses the normalized softmax top-k MoE executor inside the explicit
full-sequence non-causal path. Its Q/K-normalized NeoX-RoPE attention, expert
tensors, hidden-state output, and all-position vocabulary logits run through
bounded-host and F32-preload CUDA execution; cache and autoregressive entry
points reject it. The shared iterative diffusion runtime and CLI consume those
all-position logits.

LFM2 now consumes its per-layer KV-head schedule, uses Q/K-normalized attention
on transformer layers, and runs gated channel-wise short convolution on
recurrent layers. Causal execution prepends the serialized convolution state;
explicit non-causal execution uses the pinned centered window with zero right
padding and accepts odd kernels that preserve sequence length. Its fixed
convolution window participates in host cache serialization and prefix-edit
operations; bounded-host and F32-preload CUDA execution are covered. Native
quantized input/output projections reuse the generic matmul kernels. Short
convolution coefficients remain F32 because their kernel-width dimension is
not GGUF block-quantizable. Even centered kernels are rejected because the
pinned symmetric padding rule would shorten the sequence.

LFM2-MoE composes that hybrid cache with optional leading dense SwiGLU blocks
and later normalized softmax or sigmoid top-k routed experts. Selection
correction bias is required and remains F32 while expert banks use the native
quantized device path. Metadata, mixed dense/MoE and recurrent/attention
catalogs, graph semantics, and a complete recurrent convolution-plus-MoE
reference/CUDA differential pass. Real-model validation remains pending because
no LFM2-MoE GGUF is available locally.

PLM now executes its low-rank KV-A/KV-B decomposition, weighted compressed-KV
normalization, shared positional-key expansion, partial normal RoPE, and squared-ReLU
FFN through bounded-host and F32-preload CUDA paths. Its decompressed K/V cache
is serializable and prefix-editable; native quantized MLA fusion remains
performance work.

MiniCPM3 now reuses the MLA decomposition with its query LoRA A/RMSNorm/B
chain, partial NeoX RoPE with optional short/long factors, scaled attention and
FFN residual branches, and dense SwiGLU. Embedding and inverse-logit scaling
reuse the MiniCPM path; real-model validation remains pending a local fixture.

Hunyuan-Dense now executes optional fused or separate Q/K/V projections,
XDRoPE-adjusted normal or four-axis multimodal RoPE, post-RoPE per-head Q/K
RMSNorm, projected embedding overrides, and dense SwiGLU through bounded-host
and F32-preload CUDA paths. The multimodal runner accepts distinct temporal,
height, width, and extra coordinates.

Hunyuan-VL initially landed its text decoder with optional fused or separate
Q/K/V, XDRoPE-adjusted normal or four-axis MRoPE, post-RoPE per-head Q/K
RMSNorm, dense SwiGLU, projected embedding overrides, and serializable KV
caching. Metadata, strict catalog, graph ordering, and a reference/CUDA block
differential passed. Its image tower and native grid contract were completed
later in this log; real-model validation remains fixture-gated.

CogVLM now executes text and projected-visual ranges with mandatory contiguous
fused QKV, normal RoPE with optional per-pair factors, RMSNorm, SwiGLU, optional
tied output, and serializable KV caching. Mixed prompts are split at declared
visual ranges and retain one KV cache while selecting the complete text or
visual attention/FFN bank for each chunk. Its internal projector implements
fixed-size Pillow-bicubic preprocessing, patch/class/learned-position inputs,
post-attention and post-FFN affine norms, GELU vision blocks, post-FC norm,
GELU plus gated SiLU projection, and BOI/EOI embeddings. Native question/answer
prompts place adjacent image streams after BOS. Strict metadata/catalog,
prompt/routing tests, decoder block differentials, and an end-to-end CPU/CUDA
projector differential pass; real-model validation remains fixture-gated.

ERNIE 4.5 dense and MoE decoders now execute normal-RoPE attention followed by
sequential dense or metadata-scheduled expert residuals. The MoE path supports
optional expert gates, selection correction bias, normalized softmax top-k
routing, routed-weight scaling, and an optional shared SwiGLU expert. The
catalog preserves the pinned graph's ignored attention-output-bias behavior;
metadata, mixed-layer catalogs, graph semantics, and reference/CUDA block
differentials pass. Real-model validation remains pending a local fixture.

DeepSeek2-OCR initially landed its text decoder with full split Q/K/V projections,
NeoX RoPE, leading dense SwiGLU blocks, and later fused or separate routed
experts. Softmax or sigmoid routing supports optional F32 selection bias,
normalization, scaling, and the required shared SwiGLU expert. Metadata, strict
mixed-layer catalog, graph semantics, and a reference/CUDA block differential
passed. Its DeepSeek-OCR v1 image tower and native soft-token contract were
completed later in this log; real-model validation remains fixture-gated.

PaddleOCR now reuses the ERNIE 4.5 dense catalog with its pinned adjacent-pair
four-axis MRoPE graph and optional attention-output bias. Its internal vision
tower implements bilinear dynamic resize, learned-position interpolation,
fused or separate Q/K/V attention with raster-order vision MRoPE, affine ViT
blocks, projector input normalization, padded 2x2 patch merge, and the two-layer
projector. Native single/multi-image prompts use the Paddle image markers and
compressed decoder coordinates. Reference/CUDA projector and decoder
differentials pass; real-model validation is pending a local fixture.

Qwen2-VL now executes its dense text decoder and internal image/video projector.
The visual tower implements temporal-pair patch projection, optional pre/post
LayerNorm, separate biased Q/K/V attention with visual MRoPE, affine LayerNorm,
GELU FFN blocks, and the two-layer merger. Native prompt construction supplies
ordered projected embeddings plus compressed four-axis image or temporal video
coordinates. The catalog accepts both standard and historical swapped FFN names
and defaults the spatial merge size to two when old files omit the metadata.
Host/CUDA projector differentials and decoder block differentials pass;
real-model validation is pending a local fixture.

Qwen3-VL adds mandatory per-head Q/K RMSNorm before scaled four-axis MRoPE and
vision-layer DeepStack injection over the dense text-decoder and KV-cache paths.
Its image/video tower produces the base merger output plus one projected stream
for each selected vision layer. Native prompts align those sparse visual streams
to full decoder token grids and inject them after the corresponding early text
layers. Host/CUDA tower differentials, multi-image assembly, metadata, strict
catalog, graph topology, and reference/CUDA decoder differentials pass. The
optional classification/rerank head and real-model fixture validation remain
pending.

Qwen3-VL-MoE composes the Qwen3-VL attention and image/video projector paths with packed routed
SwiGLU experts. It supports fused or separate Q/K/V, per-head Q/K RMSNorm,
scaled four-axis MRoPE, normalized softmax top-k routing, routed-weight scaling,
native-quantized experts, internal base/deepstack vision projection, prompt-aligned
DeepStack injection, and serializable KV caching. The optional
classification/rerank head and real-model fixture validation remain pending.
Metadata, strict catalog, graph topology, and reference/CUDA block differential
tests pass; real-model validation is pending a local fixture.

Chameleon now loads per-head affine Q/K LayerNorm weights and optional biases,
honors standard or sandwich RMS-normalization topology, uses normal RoPE, and
suppresses image-code logits 4 through 8195 for text output. Causal decoder
entry points accept caller-provided embedding overrides before learned
positions/scaling/embedding normalization and return an ordinary continuable
KV cache. This is a complete projected-soft-token decoder boundary. Image
preprocessing, a Chameleon vision encoder/projector, and image-token generation
remain deferred because neither the pinned source nor the local model inventory
provides a validated Chameleon vision GGUF/oracle fixture; external encoders can
already enter through the override API.

Talkie now executes its normalized token-embedding skip across every decoder
layer, unweighted RMSNorm before attention and FFN, post-RoPE learned query
gain, unweighted key RMSNorm, causal NeoX attention, parallel SwiGLU, per-layer
scalar skip injection, final unweighted RMSNorm, untied output projection, and
direct logit scaling. Bounded-host, F32-preload, native-quantized, and retained
device-cache paths share the graph. Metadata, strict catalog, cache topology,
and complete reference/CUDA block differentials pass; real-model validation is
pending because no Talkie GGUF is present locally.

Llama 4 now executes separate-Q/K/V grouped-query attention with normal RoPE,
post-RoPE unweighted Q/K RMSNorm where required, chunk-aligned causal attention,
and position-dependent query temperature on periodic full-attention layers.
Its metadata-selected expert layers use unnormalized sigmoid top-k routed
SwiGLU experts plus the dense shared SwiGLU expert; intervening layers remain
dense. Bounded-host, F32-preload, native-quantized expert, and retained-device
cache paths share the graph. GPT-4o/Llama-4 BPE pre-tokenization, strict mixed
catalogs, chunk-mask reference/CUDA parity, and a complete Llama 4 MoE block
differential pass. Real-model validation remains pending a local GGUF fixture.
Its multimodal encoder/projector was completed later in this log.

GPT-J now executes separate-Q/K/V attention with partial interleaved normal
RoPE, affine LayerNorm, a shared pre-attention normalized input, parallel
attention and GELU-New FFN branches, required FFN biases, and an untied biased
output projection. Metadata and rotary-width validation, strict catalog tests,
graph topology checks, and a complete reference/CUDA block differential pass;
real-model validation remains pending a local GGUF fixture. The implementation
uses the stable GGUF tensor contract because the pinned tree no longer carries
an executable GPT-J graph.

GPT-OSS/OpenAI-MoE now executes alternating standard sliding and full causal
attention with per-head sinks, separate RoPE bases, biased output projection,
and residual post-attention RMSNorm. Every layer uses biased routed experts
with top-k selection on raw router logits, softmax over selected logits, and
the clamped OpenAI SwiGLU formula. F32 and native-quantized expert storage,
including MXFP4, share the reference/CUDA graph. Strict metadata/catalog tests,
formula-level routing tests, and a complete CUDA block differential pass;
real-model validation remains pending a local GPT-OSS GGUF fixture.

Eagle3 now uses an explicit paired target/draft session. Three configured target
hidden streams pass through the draft feature encoder; target embeddings and
output projection may be shared. Autoregressive draft advance retains its own
KV cache. Bounded greedy and sampled coordinators verify against the target,
advance accepted target/draft caches, restore or commit sampler checkpoints,
and resynchronize the recurrent feature from committed target-layer inputs.
The target forward now captures configured pre-layer inputs alongside cache
advancement, so setup and verification fuse only newly committed rows. Strict
paired-contract, reference, CUDA pipeline, and coordinator validation pass;
real-model pair validation remains pending local fixtures.

Gemma4 Assistant now uses an explicit shared-target-context session. Each step
combines the target token embedding and pending target hidden state, selects the
target model's final sliding/global K/V caches, runs fixed-position query-only
attention, and returns logits plus the recurrent next hidden state. Bounded
greedy and sampled coordinators now verify proposals against the target, advance
only the accepted target prefix plus correction, restore sampler checkpoints on
failure, commit the correction, and replace recurrent draft state with the true
target hidden row. Strict paired-contract, reference, CUDA pipeline, and
coordinator validation pass; real-model pair validation remains pending local
fixtures. Projected-session setup also accepts the Gemma4 image/audio embedding
overrides and bidirectional media blocks, preserving the media-aware target
prefix cache before text-only draft continuation.

Gemma3n now executes four-stream AltUp prediction/correction, magnitude-matched
projection and unembedding, Laurel low-rank residuals, first-ten-layer Gaussian
activation sparsity, projected per-layer token inputs, and query-only suffix
layers sharing sliding/global K/V from layers 18/19. Cross-stream reductions
remain bounded host operations; attention and dense projections use the normal
reference/CUDA and preloaded-weight paths. Metadata, strict 30/35-layer catalog,
AltUp formula/layout, cache topology, and CUDA active-stage differential tests
pass. The MobileNetV5 projector executes fixed bicubic preprocessing, edge and
universal inverted-residual blocks, downsampled multi-query attention,
multi-scale fusion, average pooling, spatial normalization, and fixed 256-token
soft projection. Native multi-image/history prompts preserve raw image
embeddings through Gemma3n input scaling. Strict synthetic catalogs, pooling
oracles, and full CPU/CUDA projector differentials pass. The pinned mtmd path
explicitly skips Gemma3n audio execution; implementing it requires a new
upstream execution contract and matching GGUF/oracle fixture.

Falcon-H1 now executes parallel NeoX-RoPE GQA and Mamba2 in every layer, sums
both mixer outputs into one residual, then applies the parallel SwiGLU FFN.
Named fixed cache states retain convolution and SSM tensors beside token-growing
attention K/V. Separate/fused QKV, optional attention/SSM/FFN biases, optional
SSM grouped RMSNorm, bounded-host, F32-preload, native-quantized, serialization,
cache editing, topology, and full CUDA block differential tests pass; real-model
validation remains pending a local Falcon-H1 GGUF fixture.

DeepSeek2 now executes dense-only and routed variants with direct or LoRA query projection, compressed KV RMSNorm,
legacy decompressed MLA, and modern absorbed MLA with a one-head compressed KV
cache. Per-head grouped matmul handles K absorption and post-attention V
expansion for F32 and native-quantized weights. YaRN magnitude-derived
attention scaling, optional position temperature, dense-leading SwiGLU, later
softmax or sigmoid routed experts, optional correction bias, fused or separate
expert gate/up storage, and the shared SwiGLU expert match the pinned graph.
Strict metadata/catalog, topology, reference, F32 CUDA, and quantized grouped-
matmul tests pass; real-model validation remains pending a local GGUF fixture.

Mistral 4 now uses its native `mistral4.*` GGUF metadata with the same loader,
tensor catalog, and graph inherited from DeepSeek2 in the pinned upstream.
Direct/LoRA query projection, legacy and absorbed MLA, YaRN, attention
temperature, dense-leading or routed/shared-expert FFN, native-quantized
weights, and both cache layouts follow the already verified DeepSeek2 paths.
Architecture-specific spec, catalog, topology, and cache tests pass;
real-model validation remains pending a local Mistral 4 GGUF fixture.

Mamba v1 now executes pure recurrent decoder layers with pre-RMSNorm, split
input/gate projection, cached channel convolution, selective state-space scan,
D residual skip, SiLU gate, and output projection. FalconMamba's optional
unweighted dt/B/C RMSNorm is supported. Bounded-host, F32-preload, and
native-quantized weight paths share serializable convolution and SSM state.
Strict metadata/catalog, cache topology, reference formulas, and complete CUDA
block differentials pass; real-model validation remains pending a local GGUF
fixture.

Mamba2 now executes its enlarged input projection, grouped B/C convolution and
scan state, scalar-per-head A/D coefficients, SiLU z gate, and grouped learned
RMSNorm before output projection. The scan primitive accepts both Mamba v1's
per-state A and Mamba2's scalar A without changing cache serialization.
Bounded-host, F32-preload, native-quantized, topology, formula, cache, and full
CUDA block differential tests pass; real-model validation remains pending a
local GGUF fixture.

RWKV6-Qwen2 now executes cached normalized token shift, five-way low-rank time
mixing, optional biased R/K/V projections, grouped-KV gated linear attention,
sigmoid output gating, sequential SwiGLU, and optional periodic residual
rescaling. The graph preserves native-quantized projection matrices and stores
token-shift plus WKV state in the serializable two-tensor recurrent cache.
Strict metadata/catalog, topology, cache, reference formulas, and full CUDA
block differential tests pass; real-model validation remains pending a local
GGUF fixture.

RWKV6 now executes affine-normalized dual token shifts, five-way low-rank time
mixing, classic time-first WKV6 recurrence, per-head group normalization, SiLU
output gating, squared-ReLU channel mixing, and periodic residual rescaling.
Fused and legacy separate time-mix lerp catalogs are supported. The two shift
vectors pack into one serializable cache tensor beside WKV state. Strict
metadata/catalog, topology, cache, reference formulas, and full CUDA block
differential tests pass; real-model validation remains pending a local GGUF
fixture.

RWKV7 and ARWKV7 now execute vector-valued decay recurrence, learned
in-context rates, normalized/corrected keys, per-head output correction, and
the transient first-layer value residual shared across later layers. RWKV7
uses affine normalization, dual token shifts, gated time mix, time group norm,
and squared-ReLU channel mix; ARWKV7 uses RMSNorm, one token shift, optional
gate/group norm, and SwiGLU. Only recurrent shifts and WKV state are serialized;
the cross-layer value is rebuilt per forward call. Strict metadata/catalog,
topology, cache, formula, and CUDA kernel differential tests pass; real-model
validation remains pending local GGUF fixtures.

Jamba now executes metadata-selected Mamba v1 and no-RoPE GQA layers in one
hybrid cache. Its recurrent layers apply learned dt/B/C RMSNorm and every layer
then applies either dense SwiGLU or optional softmax routed-MoE FFN with the
pinned non-renormalized selected weights. Strict mixed catalogs, attention and
recurrent topology, cache validation, and a complete recurrent-MoE CUDA block
differential pass; real-model validation remains pending a local GGUF fixture.

Granite Hybrid now executes metadata-selected Mamba2 and GQA layers in one
hybrid cache. It preserves optional RoPE, attention, embedding, residual, and
inverse-logit scaling; optional convolution and dense projection biases; gated
or ungated softmax experts; and optional shared SwiGLU experts. Strict metadata,
mixed catalog, topology, cache, and complete CUDA block differential tests pass;
real-model validation remains pending a local GGUF fixture.

Nemotron-H and Nemotron-H-MoE now execute metadata-selected no-RoPE GQA,
Mamba2, and standalone FFN layers in one hybrid cache. Dense layers use
squared ReLU. MoE layers use sigmoid top-k squared-ReLU experts with correction
bias, optional latent down/up projections, optional selected-weight
normalization, routed scaling, and the shared squared-ReLU expert. The expert
primitive permits distinct router and expert widths across F32 and native
quantized storage. Strict metadata, three-way catalogs, graph topology, cache,
attention/recurrent/MoE CUDA differentials, and native-quantized expert
differentials pass; real-model validation remains pending a local GGUF fixture.

Step3.5 now preserves and loads every declared NextN block with its full
per-layer head, sliding-attention, RoPE, gate, dense/MoE, correction-bias,
shared-expert, and SwiGLU-clamp schedule. The target exposes pre-output-norm
hidden rows. Each trained draft head catches up independently over the prompt
and rebuilds the growing speculative prefix, matching the pinned multi-head
chain. Greedy and stochastic coordinators perform target verification, cache
resynchronization, probability-ratio acceptance, positive-residual correction,
and transactional sampler updates. Bounded serialized state retains trunk and
per-head caches, draft tokens/hidden rows, absolute positions, and model
binding. Metadata, catalog, graph, state, reference, and CUDA suites pass;
real-model validation remains pending a local Step3.5 MTP GGUF fixture.

HY-V3 now preserves and executes its appended full-decoder NextN heads. It uses
the same independently caught-up, growing-prefix head coordinator as Step3.5
while preserving HY-V3's post-final-norm target and chained draft hidden-state
contract. Greedy and stochastic verification, target/head-cache resync,
transactional sampler updates, bounded state serialization, native-quantized
preload, and optional fixture integration are covered. Real-model validation
remains pending a local HY-V3 MTP GGUF fixture.

Cohere2-MoE now preserves and executes its appended NextN block using the
pinned normalized token/hidden fusion, full-attention RoPE block, parallel
attention/FFN residual, routed experts, optional shared-expert averaging, and
direct logit scale. Combined and paired-sidecar runners support greedy and
stochastic verification, positive-residual correction, transactional sampler
updates, cache resynchronization, bounded state serialization, native-quantized
preload, and optional fixture integration. Real-model validation remains
pending a local Cohere2-MoE MTP GGUF fixture.

Server media ingestion now uses a separate bounded body envelope for
media-capable native, Chat, and Responses routes. Base64 decoding enforces
per-image and aggregate decoded-byte limits. Image headers are decoded first,
with per-dimension, per-image-pixel, and aggregate-pixel limits enforced before
full allocation. Invalid media geometry returns a request error before
generation. Unit coverage includes ordinary/media body-budget separation,
dimension bombs, per-image pixel overflow, aggregate pixel overflow, and the
normal ordered multi-image paths.

Native completion, Chat, and Responses prompts now converge on one prepared
representation containing final text, exact token IDs, projected inputs, and
media state. Chat and Responses token-count endpoints reuse the same message
normalization, tool selection, formatting, tokenization, and media projection
as generation. Generation receives the prepared token IDs directly, preventing
usage drift from a second tokenizer pass. Cross-route tests compare reported
input counts, generation usage, and the exact IDs delivered to the runner.

Formatting verification is now read-only. Local verification and the
cross-platform CI matrix run `gofmt -l` and fail with the complete unformatted
file list; neither gate rewrites the worktree.

The CUDA driver now copies driver-owned error strings through `lstrlenA` and
`RtlMoveMemory`, eliminating the process-pointer arithmetic that blocked vet.
Local and CI verification enforce `go vet ./...`. A separate Ubuntu CGO job
runs the race detector over server and inference packages without changing the
normal cgo-free runtime contract.

The largest server, model specification/catalog/graph, and CUDA executor test
files are split into bounded source units without changing package boundaries.
CUDA integration setup is centralized in `internal/cuda/testutil`; 137 device
tests now share one environment gate and skip contract.

Responses resources now use a disabled-by-default top-level YAML contract.
Mapped file IDs resolve only through canonicalized allowlisted roots and eager
MIME/byte validation. Typed inline text, policy-allowed remote text URLs, and
mapped image file IDs share existing request, SSRF, redirect, timeout, and
projection bounds. Unsupported binary documents fail before formatting.
Hosted, MCP, and free-form custom Responses tools now use an explicit deny
policy. Non-deny configuration fails at startup until a bounded external
executor exists; request errors identify the policy category.
OpenAI Chat/Responses and Anthropic now route function-tool schemas through
the full-history media formatter. Anthropic base64, allowlisted URL, and mapped
file-ID image blocks use the shared projector, projected cache signatures, and
input-token path. Video/tool combinations remain explicitly rejected.
Anthropic manual summarized thinking now uses the pinned budget rules, emits
buffered and named-SSE thinking/signature blocks, works with projected images,
and replays only byte-exact blocks signed by the same handler. Local signatures
are integrity tokens, not Anthropic-portable encrypted reasoning. Adaptive,
omitted, redacted, interleaved, and thinking/tool modes fail explicitly.

OpenAI Chat, OpenAI Responses, and Anthropic request/response types, parsing,
buffered generation, streaming, tools, and token-count handlers now live in
protocol-specific source files. The shared `Handler`, admission slots, metrics,
sampling, prompt preparation, native routes, and error transport remain in the
server core.

Tool-call streams now parse model output incrementally at the inference
boundary. JSON-in-XML and Hermes parameter-tag templates emit function starts
and argument fragments as generation pieces arrive across OpenAI Chat,
OpenAI Responses, and Anthropic protocols. The complete parser still validates
the final function names, JSON, schemas, and call sequence before completion.

Architecture support now has one registry covering every accepted GGUF
architecture. Profiles expose primary attention, MoE, recurrent, hybrid,
encoder, encoder-decoder, diffusion, and draft families plus orthogonal
position, normalization, multimodal, and execution capabilities. Spec parsing
and shared inference predicates use the profiles instead of duplicated family
lists.

Model metadata is grouped into embedded common, attention, MoE, recurrent,
encoder, and multimodal sub-specifications. Promoted field access preserves
runtime behavior, while keyed internal construction names the owning sub-spec
explicitly. The complete model, inference, server, and CUDA executor suites
pass the migration.

Graph construction now enters through an architecture-family dispatcher with
separate recurrent/hybrid, MoE/MLA/DSA, and attention paths. Tensor catalog
validation uses matching attention, MoE, recurrent, hybrid, encoder,
encoder-decoder, diffusion, and draft entry points over the shared strict
shape/name validation core. Family routing is sourced only from the central
architecture profile.

Compiled model plans now select composed dense or per-layer cached graph
execution. The decision combines graph family, AltUp capability, every block
and attention policy, and attention/sentinel cache schemas. The runner's
architecture-name exclusion chain is removed while dense, MoE, Deci sentinel,
MLA/DSA, recurrent, hybrid, encoder, and Gemma 3n policies remain pinned by
registry-wide tests.

Architecture profiles now also select the public inference forward route:
cached causal, non-causal, DFlash/Eagle3/Gemma 4 Assistant dedicated session,
WavTokenizer audio decoder, T5 encoder, or T5 encoder-decoder session. Spec-
selected non-causal execution is compiled into the prepared model snapshot;
the runner no longer owns the corresponding architecture-name dispatch chain.
Embedding overrides, cache entry points, T5 sessions, audio decoding, and
generation use the same compiled route.

Final normalization is now one profile contract covering absent, model,
encoder, decoder, and token-embedding tensor ownership. BERT-style embedding
and post-norm layout has an explicit capability rather than repeated five-name
predicates. Host and retained-device execution share one output-norm graph
builder, while weight loading uses the profile's tensor name and absence rule.
The architecture registry's policy setters share one checked mutation loop;
graph and weight construction consume their existing profile snapshot directly.
Ten legacy string-to-profile adapters were removed from production. T5,
diffusion, and speculative entry guards likewise reuse prepared forward,
family, and draft policies.

Draft profiles now expose shared bounded-head and single-head predicates used
by MTP graph builders, weight discovery, and inference session validation.
DeepSeek2-compatible metadata and tensor layout is a separate capability from
DeepSeek2 graph behavior, allowing GLM-DSA to share catalog/spec rules without
inheriting incompatible dense-block requirements. The remaining DeepSeek,
GELU, and squared-ReLU string lookup adapters were removed; graph and catalog
code consume the cached layout, attention, and feed-forward policies directly.

Retained CUDA KV caches now maintain configurable token-page tables. Each page
contains layer-specific device pointer views with bounded token extents;
append, suffix trim, zero-copy shift, and owned range compaction rebuild the
table. The continuous-batch session layers sequence ownership on those caches,
commits multi-sequence steps transactionally, permits sequences to enter and
leave between steps, clones host/device branches, and exposes stable page snapshots.
Preloaded device batches now place all active variable-length sequences in one
graph execution. The server advances that active set token by token, admitting
new work between steps and preserving independent stop and cancellation state.

T5 now accepts padded rectangular source and decoder batches with explicit
per-row lengths. Each active prefix is evaluated as an independent session
inside one batch transaction, which provides the required padding mask
semantics without placing padding tokens into any attention graph or cache.

LFM2 centered short convolution now accepts even kernels with asymmetric
left/right padding while retaining same-length output and recurrent-state
updates. Refact now admits gated and ungated top-k expert tensors. DeepSeek 4
cache state now carries explicit token positions, preserving middle-edit gaps
through raw attention, compressed reconstruction, serialization, and append.

GLM4 and EXAONE 4 NextN tails now load as executable full decoder blocks with normalized
token/target-hidden fusion, private-or-shared heads, independent KV state,
bounded greedy drafting, and target verification/resync.
EXAONE-MoE uses the same runtime with its declared dense NextN tail, overriding
the routed-expert trunk schedule only for the appended draft block.
GLM4-MoE uses the same runtime while retaining its routed-expert tail schedule.
BailingMoE2 and MiMo2 add their required tail output norm; BailingMoE2 retains
its routed/shared-expert tail and MiMo2 admits the checkpoint-selected dense or
routed tail branch.
DeepSeek 3.2 executes its MLA/DSA NextN tail with an independent growing
lightning-indexer key cache alongside the draft KV cache.
GLM-DSA retains the trunk's final full-indexer top-k selection as transient
session metadata and reuses it in the shared-indexer MTP tail.

Hunyuan-VL now executes its dynamic Pillow-bicubic image preprocessing,
interpolated learned-position ViT, RMS-normalized convolutional spatial merger,
row-newline prefix projection, begin/end embeddings, and final RMSNorm. Native
single/multi-image prompts interleave text and media, preserve history, and emit
the decoder's time/column/row/image axes for begin, content, newline, and end
tokens. Tiny catalog, raster-order, prompt-position, and CPU/CUDA differential
tests pass; real-model validation remains fixture-gated.

Llama 4 now executes its llava-UHD refined-grid and overview preprocessing,
class/learned-position vision transformer, width/height rotary attention,
pixel-shuffle adapter MLP, and final projection. Native single/multi-image
prompts use the image boundary and slot tokens, preserve history, and retain
refined-tile-before-overview ordering. Strict catalog, preprocessing, prompt,
two-axis RoPE, and CPU/CUDA differential tests pass; real-model validation
remains fixture-gated.

Granite 4 Vision now executes Pillow-bicubic aspect-fit preprocessing with an
overview followed by selected grid tiles, learned-position SigLIP vision
blocks, and one window QFormer per selected feature layer. QFormer streams
support average-pool or spatial-offset downsampling, learned image/query
positions, self/cross attention, exact GELU-ERF FFNs, raster reconstruction,
and per-tile newline embeddings. Native single/multi-image prompts retain the
leading image marker, map later marker tokens to the base stream, and carry
aligned deepstack streams into configured Granite decoder layers. Strict
metadata/catalog checks, preprocessing and prompt tests, plus a nonzero
multi-window CPU/CUDA differential pass; real-model validation remains
fixture-gated.

MiMo-VL now executes dynamic Pillow-bicubic preprocessing, split temporal-pair
patch projection, grouped-query vision attention, two-axis RoPE, alternating
row/column symmetric windows with per-head sink logits, SwiGLU blocks, affine
post normalization, and the 2x2 GELU merger. Native single/multi-image prompts
use the pinned MiMo system turn and vision markers while history prompts retain
caller formatting. Strict metadata/catalog checks, order round trips, prompt
coverage, reference/CUDA sink-window differentials, and a nonzero end-to-end
CPU/CUDA differential pass; real-model validation remains fixture-gated.

DeepSeek-OCR now executes its dynamic Pillow-bicubic local-grid and overview
preprocessing, SAM patch tower with local/global decomposed-relative attention,
CLIP QuickGELU tower, concatenated feature projection, row-newline assembly,
and trailing view separator. Native single/multi-image prompts preserve history
and bind each produced soft token to the decoder. Strict metadata/catalog and
preprocessing/prompt tests, tensor primitive parity, plus an end-to-end CPU/CUDA
differential pass; real-model validation remains fixture-gated.

DeepSeek-OCR-2 reuses the SAM tower with 768-pixel independent local tiles and
the 1024-pixel overview, then concatenates size-specific learned queries into
its GQA Qwen2 encoder. Image prefixes attend bidirectionally while resampler
queries attend the complete image prefix and their causal query prefix. Only
query outputs enter the final projection; the overview receives the trailing
view separator. Tiny overview/grid/prompt contracts and a nonzero end-to-end
CPU/CUDA differential pass; real-model validation remains fixture-gated.

## Shared runtime follow-on

Model catalog loading now uses ordered tensor-requirement schemas for common
MTP, shared-expert, draft, DeepSeek 4, Kimi Linear, T5 cross-attention, and MLA
families. Required and optional bindings share one path, preserve declared
validation order, and bind value or pointer destinations without reflection.

Inference host/device graph setup now shares one runtime for inputs, weights,
layer bindings, and execution. Qwen3.5 and Cohere2-MoE share a typed single-head
MTP advance transaction; NextN uses the same graph runtime while retaining DSA
indexer and auxiliary-state policy. Qwen3-VL, MiMo-VL, Hunyuan-VL, and
PaddleOCR-VL now execute one projector graph on reference or CUDA backends.

Server endpoints now share exact method guards and strict one-document YAML
decoding. OpenAI Completions, Chat, Responses, and Anthropic generation paths share
stop filtering, generated/completion counts, final flush, and finish-reason
selection while protocol-specific text, reasoning, and tool events remain in
their protocol sinks. Prompt-cache selection shares one LRU promotion helper,
and CLI model flags/open options now cover benchmark, diffusion, embedding,
rerank, generation, perplexity, and server commands with command-specific
names and defaults.

## Consolidation closure

Architecture profiles now own mandatory-output, classifier-head, bias-free
projection, and appended-draft-block decisions previously repeated as family
name chains. The model catalog has no inline tensor-binding maps: ordered
schemas cover required, optional, and F32-constrained bindings, and test
fixtures derive names and shapes from those schemas. RWKV catalog dispatch now
lives in its family unit.

The shared inference graph runtime now drives MTP, Eagle3 feature fusion and
draft steps, DFlash feature fusion and cache injection, and Gemma 4 Assistant.
Hunyuan-VL, MiMo-VL, and PaddleOCR-VL retain only their reference/CUDA graph
paths; the retired handwritten host graphs and family-specific host/device
binding helpers were removed.

## Consolidation expansion

The model-owned F32 device-layer binder now serves inference directly; the
second 500-line inference mapping was removed. CUDA validation covers the
shared binder. The inference graph runtime also owns Gemma 3n attention/FFN
stages, embedding projection, classifier ranking, perplexity output
projection, and Step3.5/HY-V3 multi-head execution.

State-space catalog dispatch for Jamba, PLaMo2, Mamba, Falcon-H1, Mamba2,
Qwen GDN, Granite Hybrid, and LFM2 moved to a dedicated family unit. The
Mamba catalog test constructs its fixture from the production tensor schema.
Fused-QKV policy moved into architecture capabilities. Speculative families
share sampled-limit, sampler-pair, and detached last-hidden helpers. Retired
Qwen3-VL host attention and unused multi-axis-position helpers were deleted.

## Policy, plan, and execution convergence

Architecture profiles now expose typed normalization, position, residual,
attention, and feed-forward policies. A per-layer plan combines those policies
with graph family, catalog family, cache extent, recurrence, sliding attention,
RoPE, multi-axis position, and KV requirements; graph dispatch consumes that
plan instead of rediscovering family semantics.

Layer cache schemas describe standard KV, MLA/DSA, recurrent, hybrid, named,
cross-attention, indexer, and fixed-state storage. Host validation and device
allocation, shifting, compaction, and batch input construction now consume the
same schema. Weight loading separates standard attention, dense FFN, and MoE
selection catalogs while retaining common ordered binding and validation.

Standalone graph leaves now use the shared inference graph runtime for host and
device feeds, weights, bindings, execution, and output reads. Core layer-loop
executors remain the only direct graph builders. Single-head MTP session state
uses one bounded codec, and speculative families share generic greedy draft and
verification coordinators with model-specific advance and resynchronization
callbacks.

Image prompt execution now follows one plan: validate sources, encode planned
items, render family control syntax, tokenize variable placeholder runs, merge
deepstack streams, map embeddings, generate multi-axis positions, and construct
attention blocks. Granite 4, Llama 4, Hunyuan-VL, PaddleOCR, MiMo-VL, Qwen2-VL,
Qwen3-VL, and Gemma 4 use that lifecycle for native and history prompts.

## Shared utility substrate

Checked size arithmetic and alignment now share one overflow-aware package
across GGUF, quantization, caches, media, CUDA, and tensor planning. Generation,
T5, and MTP session formats retain their wire layouts while using one bounded
binary encoder/decoder with consistent truncation, trailing-data, and allocation
guards.

Server and CLI JSON inputs now use one strict single-document decoder with
unknown-field and body-limit enforcement. Inference and projector runtimes share
host/device feed construction and execution plumbing; model and projector tensor
catalogs share neutral requirement validation while retaining their local binding
policies.

Reference tensor values now provide checked clone and row-slice operations.
Audio float32/WAV conversion and image cloning share media primitives, and common
GGUF/WAV test fixtures moved to a test utility package. Redundant alignment,
maximum-integer, state-copy, media-copy, and fixture helpers were removed.

## Repository-wide utility sweep

The bounded state codec now owns sampler, KV-cache, single-head MTP,
multi-head Step3.5/HY-V3 MTP, T5, and generation session envelopes. Existing
magic/version layouts remain byte-compatible, including legacy sampler and
cache readers; common cursor bounds replace format-local offset arithmetic.

All production slice copies use `slices.Clone`, and shallow map copies use
`maps.Clone`. One dtype implementation now owns BF16 storage, expansion, and
rounding for quantization, conversion, reference execution, and Gemma 4
projection. CUDA driver host views replace executor/kernel scalar-to-byte
helpers, and normal/speculative sampling share one adaptive-p transformation.

Every multimodal runner delegates resource closure to one generic transaction
and host tensor pairs to one loader. CLI commands share error exits, compact or
pretty JSON emission, and generated-artifact check/update/output behavior.
Reusable WAV fixtures cover PCM16 and float32 without local RIFF builders.

## Compiled execution contracts

Model opening now compiles an immutable `ModelPlan` with one block policy and
cache policy per layer. Graph dispatch no longer routes on architecture names,
and host cache-input construction materializes fixed, token, and named states
from the same cache schema used by validation and device execution. This
removed the parallel RWKV, Mamba, Kimi, Falcon-H1, and hybrid cache setup chain
from the core runner.

CUDA execution now supports reusable `CompiledGraph` values containing
validated topological order, BLAS requirements, and arena liveness plans.
Host, device-feed, and retained-output entry points share one compiled
execution transaction. A typed tensor operation catalog owns operation names,
diagnostic classes, and declared reference/CUDA coverage. Server protocols now
share optional model-selector enforcement.

## Cache-plan convergence

Host cache validation now delegates all primary and named-state shapes, finite
value checks, extents, and strict-state membership to the layer cache schema.
Only cross-layer and position-order semantics remain local. Range removal uses
the same schema to clone fixed recurrent state and compact token state, including
runners without weight-derived recurrence metadata.

Qwen GDN device initialization now uses the common cache-input builder instead
of a second recurrent-shape allocator. Architecture profiles own shared-KV,
AltUp, per-layer-embedding, and embedding-skip capabilities; `LayerPlan` pins
shared-cache ownership and source layers before host or CUDA graph execution.

## Prepared model ownership

Runner storage now separates loaded GGUF, compiled plans, tensor catalogs,
tokenizer data, CUDA executors, device weights, output data, configuration, and
model fingerprints from mutable LoRA scales and retained prompt caches. Close
releases request state before prepared resources through independent ownership
methods.

Prompt-cache reset is one lifecycle operation shared by explicit clearing,
LoRA-scale changes, and runner closure. Explicit clearing preserves all prepared
assets and returns retained-device release failures instead of discarding them.

## Protocol execution convergence

Chat Completions, Responses, and Anthropic Messages now share bounded generation
token selection, formatter/tokenizer capability checks, tool-grammar setup, slot
admission plus projected prompt preparation, input-token response construction,
and inference option assembly. Projected Anthropic prompts now use the same
exact-media prompt-cache contract as Chat and Responses.

One incremental tool-delta stream now owns fallback buffering, parser dispatch,
function names, argument accumulation, and index validation for all three SSE
protocols. Protocol encoders retain their distinct event and response shapes.

## Projected and side-input plans

Architecture profiles now declare projected embedding order, deepstack mode,
bidirectional attention-block support, cross-layer auxiliary flow, and scheduled
attention-temperature behavior. `LayerPlan` resolves Granite/Qwen3-VL stream
placement, RWKV value-residual carry, GLM-DSA top-k carry, and per-layer
temperature feeds before graph execution. Host, preloaded, and retained-device
paths consume the same compiled contracts.

Post-norm and pre-FFN tensor namespaces use narrow typed catalogs with preserved
BERT biases and Grok fallback handling. Qwen3.5, Step3.5, HY-V3, NextN, and
Cohere2-MoE MTP wrappers share one family descriptor for eligibility,
normalization, hidden carry, logit scaling, and diagnostics; block math remains
family-local.

## Working rules

- A blocked task is recorded here and deferred while independent work continues.
- Unsupported model/type combinations fail explicitly.
- Upstream llama.cpp is an oracle and test dependency, not a shipped runtime
  dependency.
- Correctness gates precede performance work.
