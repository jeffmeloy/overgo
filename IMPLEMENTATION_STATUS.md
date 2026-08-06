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
| Safetensors | Implemented ingestion | Bounded Hugging Face config, shard index, lazy tensor catalog, and repository inspection; Llama, Qwen2, and Qwen 3.5 text/recurrent/MTP metadata, catalogs, and payload streams bridge to GGUF runtime contracts |
| Tokenization | Implemented | BPE, SPM, WordPiece, and UGM families with pinned oracle corpora |
| Tensor graph | Implemented, expanding | Reference and CUDA execution for dense, MoE, recurrent, diffusion, encoder, and multimodal primitives |
| Model runtime | Experimental breadth | Architecture-specific metadata, catalogs, graphs, cache state, and optional real-model oracles |
| Generation | Implemented | Cached autoregressive, diffusion, encoder-decoder, embeddings, reranking, and speculative paths |
| Multimodal | Implemented, constrained | CogVLM, DeepSeek-OCR, Gemma 3n, Granite 4 Vision, Hunyuan-VL, Llama 4, MiMo-VL, PaddleOCR-VL, Qwen2-VL, dense/MoE Qwen3-VL, and Gemma 4 image/video; Gemma 4 ordered image/audio history |
| Server | Implemented, expanding | Native, OpenAI Chat/Responses, Anthropic text/tools, streaming, fused batching, slots, metrics, LoRA |
| RepoDB | Artifact, policy, workflow, and evidence foundations implemented | Content-addressed manifests/profile/tensor-inventory/model-definition documents, data-backed bootstrap profiles, external profile-catalog publication, typed lineage/locations/recipe DAGs, generation/embedding/rerank/projection/training plans, run/evaluation evidence, parity-gated runtime policy injection, atomic hash-chained commits, strict replay, and fail-closed single-writer storage |
| Training | Optimizer core implemented | Validated flat parameter plans, adaptive/Muon CPU updates, explicit schedules, and identity-bound state restore; backward graphs and mutable model master weights remain pending |
| CUDA | Implemented on Windows | Dynamic Driver API, cuBLAS, embedded PTX, persistent/native-quantized paths |
| Release | Implemented | Reproducible Windows-amd64 archive, SBOM, kernel ABI manifest |

## Recently completed

- Generic `CacheState[T]` now carries mode plus value for host tensors,
  CUDA-resident tensors, and in-flight device graph tensors. Existing aliases
  preserve API vocabulary while removing three duplicated state structs and
  keeping representation choice outside cache semantics.
- The model cache schema now owns explicit serialized fixed/token mode values,
  mode validation, token-alignment semantics, and extent conversion. Inference
  retains source-compatible aliases, eliminating its duplicate enum and inline
  schema-to-wire branching without changing cache payloads.
- `CacheStateName` now types named-state keys end to end across model schemas,
  graph results, host caches, CUDA-retained caches, range editing, sessions,
  and serialization. Decoding still accepts validated extension names, while
  internal APIs can no longer mix arbitrary strings with cache contracts.
- Cache schemas now own the serialized names for DSA indexer, DeepSeek 4
  compression, recurrent convolution/SSM, position, and T5 cross-attention
  states. Graph builders, host/device cache binding, validation, sessions, and
  fixtures consume the canonical ABI facts instead of repeating literals.
- Ordered graph-weight schemas now replace per-family requirement maps across
  dense, encoder, MLA/DSA, recurrent, state-space, multimodal, and specialized
  graph builders. Conditional contracts compose through one typed API, retain
  deterministic first-error ordering, and avoid transient map allocation.
- Ordered typed metadata field schemas now replace repeated map-based scalar
  decoding across SSM, RWKV, expert, MLA/indexer, DeepSeek 4, YaRN, xIELU, and
  WavTokenizer ingestion. Required fields retain deterministic first-error
  ordering, and the shared reader reduces production surface and allocations.
- Model metadata ingestion now consumes typed profile selectors and a bounded
  metadata-read fact set. Architecture-core, expert, position, draft, family,
  and runtime parsing contain no architecture-name dispatch; persisted RoPE,
  ALiBi, multimodal, and special-scalar facts are validated before binding.
- DFlash, Eagle3, Gemma 4 Assistant, and WavTokenizer executable guards now use typed forward policies; speculative target compatibility requires exact resolved-profile equality rather than matching architecture text. Remaining production architecture comparisons are identity/envelope checks only.
- DeepSeek 3.2 full-indexer scheduling moved into typed layer cadence policy. Production `Spec` and every shared metadata validator now contain no architecture-name predicates; exact resolved profiles own all migrated selection facts.
- All hybrid/MoE validation now selects one of 36 typed profile contracts. Dense/MoE variants, shared-expert layouts, routing/scaling, LongRoPE/YaRN, sliding attention, and hybrid convolution schedules retain family diagnostics without architecture-name dispatch in the shared validator.
- Qwen3-Next/Qwen3.5, Qwen3-MoE/VL/RND1, GroveMoE, MiMo2, and Step3.5 hybrid/draft invariants now select typed profile contracts. Shared hybrid scheduling and expert checks no longer identify these families by architecture name, and rotary alignment relationships use named constants.
- Recurrent validation now selects typed WavTokenizer, draft, Mamba, RWKV, Jamba, Granite Hybrid, PLaMo2, and Nemotron-H contracts from profiles. Shared recurrent validation contains no architecture-name dispatch; convolution widths, state multipliers, target cardinalities, token shifts, and rotary alignment are named facts.
- MLA/DSA validation now uses typed DeepSeek 3.2, DeepSeek 4, Kimi Linear, Mistral 3, and MiniCPM3 profile contracts. The common MLA validator contains no architecture-name dispatch, and previously scattered model dimensions are named profile facts or validation constants.
- Attention-family invariants now select 32 typed profile contracts covering per-layer head layouts, Gemma/Cohere/Phi encoder variants, rotary dimensions, sliding attention, MRoPE, fused expert metadata, and architecture-specific projection constraints. The shared validator contains no architecture-name dispatch.
- Base rotary requirements and BERT/JinaBERT/NeoBERT/NomicBERT encoder invariants now dispatch through typed validation profiles rather than architecture names. Shared encoder-shape and optional-expert predicates reduce duplicated validation surface while preserving model-specific errors.
- Embedding and logit scale direction, normalization placement/bias/fallback, and layer-level RoPE schedules now live in typed runtime profile policies. Runtime methods consume the exact bound policy without architecture-name predicates, and the semantic profile digest pins the migrated RepoDB facts.
- Canonical model-definition documents now bind a model manifest, exact profile, tensor inventory, architecture, and fully validated runtime `Spec`. GGUF ingestion produces relocation-stable definitions; one atomic batch publishes the physical inventory and all resolved facts, runtime dispatch preserves the bound policy, and recipe v2 compiles without architecture-registry lookup.
- Validated GGUF and Safetensors ingestion now emits one canonical tensor-inventory document containing ordered logical names, shapes, storage types, and payload sizes. RepoDB binds the inventory to its model manifest through typed lineage, preserves identity across relocation, and resolves the unique inventory without reopening tensor payloads.
- Architecture policy facts now live in a strict data catalog instead of a 700-line Go registry builder. The embedded catalog supplies only the bootstrap fallback; RepoDB can publish replacement profile catalogs without mutating it, while recipe activation still requires exact profile identity and parity evidence. A semantic digest pins all bootstrap profile identities across the migration.
- The compact `adaptive_new` optimizer core now runs independently over flat FP32 weights and gradients: named complete parameter plans derive sign/Muon policy, CPU Newton-Schulz owns one bounded scratch set, learning-rate semantics are typed, and snapshots bind exact plan/config identity. Golden parity covers the source optimizer without retaining its autograd, matrix, telemetry, or CUDA dependencies.
- Generation, embedding, rerank, image/audio/video projection, and training orchestration now use canonical dependency-bound recipe DAGs, a shared typed module catalog, and deterministic topological plans; immutable run/evaluation records capture queryable provenance and serve as lifecycle promotion evidence.
- Recipe v2 identities now hash canonical typed role/slot dependencies for models, profiles, tokenizers, projectors, adapters, datasets, and checkpoints; publication records dependency lineage, profile recipes no longer write mutable side aliases, and strict canonical v1 reads preserve existing logs.
- Architecture profiles now persist as strict content-addressed RepoDB facts; candidate recipes bind an exact profile, registry/profile plan parity produces typed evidence, and only matching evidence unlocks active runtime policy injection through metadata, catalog, and graph planning.
- Recipe lifecycle now records content-addressed candidate/validated/active/refused/superseded evidence, promotes status and active selection through compare-and-set aliases, atomically supersedes an older active recipe, and resolves active inference recipes into existing model plans.
- Typed recipe DAGs now validate module task/placement policies, port schemas/cardinality, producer uniqueness, cycles, and bidirectional reachability; canonical recipe documents persist through generic content records, and the first binding compiles inference recipes through existing model/layer plans.
- Canonical model manifests now bind ordered typed components without physical paths; GGUF single/split and Hugging Face Safetensors adapters reuse validated loader inventories, while RepoDB tracks relocatable file/directory availability independently.
- RepoDB now starts from storage-neutral artifact contracts: kind-qualified SHA-256 identities, immutable descriptors, acyclic typed lineage, compare-and-set aliases, canonical idempotent batches, and a versioned CRC32C/hash-chained store with strict corruption handling and torn-tail recovery.
- Safetensors repository ingestion now shares one bounded shard/index reader across inspection and Gemma 4 conversion; 14 local model repositories pass header-only validation.
- Standard Llama and Qwen2 Safetensors names/shapes translate to GGUF runtime catalogs; Qwen2.5, Carbon, and MiniCPM5 local repositories pass full spec/weight validation, with shared streaming F16/BF16 vector promotion.
- Qwen 3.5 Safetensors text, recurrent, and MTP tensors translate without repacking; the local 4B repository passes full runtime spec/weight validation, with its singleton convolution axis normalized and vision tensors kept outside the language catalog.
- Token-row validation and positions, fixed-run audio/video prompt assembly, incremental tool-delta routing, and SSE/JSON transport now use shared inference, projector, and protocol components.
- Raw and F32 CUDA weights now share one transactional tensor owner; Mamba2-family recurrent tensors share one requirement schema; Q/K and attention-gate stages use single typed applicators; and MTP families share admission plus appended dense-block coordination.
- Dense weight requirements, routed/shared expert composition, Q/K preprocessing, query scaling, attention-output stages, residual flow, and dense leaf selection now compile into `LayerPlan`; scheduler dispatch and dense execution share one typed cached-block context, and remaining production leaf attention/RoPE calls use typed option contracts.
- `LayerPlan` now compiles detailed rotary, attention, and routed-expert graph controls; dense/Gemma 4 execution consumes the plans, paired RoPE and shared experts have one implementation, and production model MoE construction converges on one typed core.
- Typed internal option contracts now own attention, MoE, and RoPE graph construction; specialized wrappers no longer forward positional nil/zero/boolean control streams or mutate completed graph nodes.
- Named semantic constants now own projector/RWKV normalization contracts, quantizer thresholds, serialization/schema envelopes, GGUF buffers, server/CLI defaults, media security bounds, and derived validation diagnostics; fixture data and mathematical shapes remain literal.
- Projector opening now uses one failure-safe GGUF ownership transaction; shared graph math owns optional bias, linear, and normalization operations; checked geometry plans own pixel merging and paired temporal patch splitting.
- All projector CUDA initialization now consumes validator- or spec-owned tensor catalogs through one generic loader; per-family CUDA tensor inventories and wrapper types are removed.
- Host layer graph construction now binds every required and optional weight through the shared mirror catalog; Llama 4, Hunyuan-VL, PaddleOCR, and MiMo-VL shape validation now produces the authoritative ordered CUDA tensor catalog.
- Required and optional layer tensors now share one reflected catalog for host loading and device graph binding; projected chunk execution carries one compiled request value and normalization plans are resolved once per model load.
- Host and device layer graphs now bind mirrored optional weights through one drift-checked field catalog; Kimi KDA device fields are included automatically.
- One plan-driven side-input binder now owns embedding skip, projected per-layer input, attention blocks, scheduled temperature, and DeepSeek4 positions across graph paths.
- Normalization, projected requests, and speculative families now compile complete execution descriptors; model-plan invariants reject contradictory cross-policy state before execution.
- Chat, Responses, and Anthropic now share token bounds, tool grammar, prompt preparation/counting, projected generation options, and incremental tool-call state.
- Runner ownership now separates prepared model/CUDA assets from mutable LoRA and retained prompt state; cleanup follows the same boundary.
- Cache validation and range editing now consume one compiled schema; typed profiles/plans own shared-KV, AltUp, per-layer-embedding, and embedding-skip behavior.
- Model opening now compiles immutable per-layer block/cache policies; host and device cache setup plus graph dispatch consume the same plan.
- CUDA graphs expose reusable compiled topology, BLAS selection, and arena plans; tensor operation identity and backend coverage share one typed catalog.
- Server protocols share optional model-selection enforcement, and compatibility claims cover the compiled plan/graph/catalog contracts.
- Sampler, KV-cache, and all MTP/session persistence now share bounded state encoding, checked sizing, legacy-version reads, and trailing-data enforcement.
- Standard slice/map cloning replaced 140 handwritten copy idioms; BF16 conversion, CUDA host scalar views, and adaptive-p transforms now each have one implementation.
- Projector resource closure and host tensor-pair loading are shared across all multimodal runners; commands share error exits, JSON output, and generated-artifact workflows.
- Narrow shared packages now own checked arithmetic/alignment, bounded state envelopes, strict JSON decoding, graph feeds, tensor catalog validation, reference-value copies/slices, media codecs, and reusable fixtures.
- Generation, T5, and MTP state persistence share one bounded codec; server and CLI JSON inputs share one strict single-document policy.
- Inference and projector execution share a graph-feed substrate; model and projector catalogs share neutral tensor-requirement validation.
- Typed architecture policies and per-layer plans now drive normalization, position, residual, attention, FFN, graph-family, cache, and weight-catalog dispatch.
- Compiled model plans select composed dense or layered cached graphs from typed family, capability, block, attention, and cache contracts; runner architecture exclusion chains are removed.
- Architecture profiles now select cached, non-causal, draft-session, audio-decoder, T5-encoder, or T5-session public forward routes.
- Output-normalization ownership and BERT norm layout are profile policies; host/device paths share one norm graph builder and weight loading uses the same tensor-name contract.
- Architecture registry updates share one checked mutator; runtime graph/weight paths consume prepared profiles directly, with legacy string-to-profile wrappers removed from production.
- Draft profiles own bounded single/multi-head eligibility across graph, weight, and session paths; DeepSeek2-compatible metadata/tensor layout is distinct from DeepSeek2 graph behavior.
- Projected embeddings, deepstack placement, bidirectional attention blocks, RWKV/DSA auxiliary flow, and attention-temperature feeds now compile into profile/layer policies; norm namespaces and MTP wrappers consume typed catalogs.
- Declarative layer cache schemas now define host validation, device allocation, shifting, compaction, and recurrent-state shapes.
- Attention, FFN, and MoE tensor catalogs are separated from orchestration; shared catalog bindings retain ordered validation.
- Shared graph execution now covers the remaining standalone inference leaves; only core cached/non-cached layer loops construct graphs directly.
- Speculative families share bounded session codecs plus generic greedy drafting and verification coordinators.
- One image-prompt planner now serves Granite 4, Llama 4, Hunyuan-VL, PaddleOCR, MiMo-VL, Qwen2-VL, Qwen3-VL, and Gemma 4, including history, deepstack, positions, and attention blocks.
- Model-owned device-layer binding now serves every inference path; the duplicate 500-line inference binder was removed after CPU/CUDA parity verification.
- Shared graph execution expanded to Gemma 3n, embedding, reranking, perplexity, and multi-head MTP; state-space catalogs split by family with schema-derived Mamba fixtures.
- Speculative coordinators now share sampling limits, sampler-pair validation, and detached last-hidden state handling.
- Complete ordered tensor schemas across the model catalog, including optional/F32 constraints and schema-derived fixtures; RWKV dispatch split into its family unit.
- Shared inference graph runtime extended from MTP to Eagle3, DFlash, and Gemma 4 Assistant; retired Hunyuan/MiMo/Paddle host-only projector graphs removed.
- Architecture profiles now own mandatory-output, classifier-head, bias-free projection, and appended-draft-block policy.
- Ordered tensor-requirement schemas with deterministic catalog binding across MTP, shared-expert, draft, MLA/indexer, DeepSeek 4, Kimi Linear, and encoder-decoder weights.
- Shared inference graph runtime plus a typed single-head MTP advance adapter for Qwen3.5, Cohere2-MoE, and NextN host/device execution.
- One reference/CUDA projector graph for Qwen3-VL, MiMo-VL, Hunyuan-VL, and PaddleOCR-VL; the Qwen3-VL handwritten CPU graph was removed.
- Shared server method guards, strict single-document YAML decoding, and protocol-neutral generation pumping for stop filtering, token accounting, flush, and finish reasons.
- Shared prompt-cache LRU promotion and configurable CLI model flags/open options across benchmark, diffusion, embedding, rerank, generation, perplexity, and server commands.
- Shared projector CUDA/catalog/graph runtimes, prompt assembly, MTP session/sampling transactions, quant dispatch, server SSE/admission, cache planning, CLI model flags, and test fixtures; 3,000+ duplicate lines removed.
- Single-pass cached target-layer input capture with incremental Eagle3 and DFlash feature resynchronization.
- Gemma 4 Assistant projected image/audio target-prefix sessions.
- Exact compatibility-manifest coverage for every registered architecture, enforced by the generator.
- WavTokenizer pinned inverse-spectral waveform synthesis with 24 kHz mono output.
- Gemma 4 real image preprocessing, full projector output, and prompt-token oracles pass against the local BF16 projector fixture.
- DFlash greedy/sampled block drafting with target verification, sampler rollback, cache advancement, and feature-cache resynchronization.
- Eagle3 greedy/sampled drafting with target verification, sampler rollback, target-cache advancement, and feature resynchronization.
- Gemma 4 Assistant greedy/sampled drafting with target verification, sampler rollback, cache advancement, and hidden-state resynchronization.
- DeepSeek-OCR v1/v2 SAM image projection with dynamic local tiles, CLIP or masked Qwen2 towers, native history prompts, and CUDA differentials.
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

## Closure roadmap

Pinned internal execution work is complete for the current baseline. Remaining
rows require an upstream contract, external artifact, or additional platform.

| Item | State | Completion boundary |
| --- | --- | --- |
| Pinned architecture registry | Complete | Exact 135/135 compatibility-manifest coverage enforced in CI |
| Speculative execution | Complete | Greedy/sampled verification and cache resync for supported draft families |
| WavTokenizer output | Complete | Feature decoder plus pinned 24 kHz waveform postprocessor |
| Multimodal gaps | Classified | Only components absent from pinned upstream execution remain |
| Platform and release breadth | External | Verified targets beyond Windows-amd64 CUDA require those environments |

## Known implementation boundaries

| Area | Current boundary |
| --- | --- |
| Responses | Bounded continuation, reasoning summaries, and typed text/image files; no PDF/non-text documents, hosted/custom tools, encrypted reasoning, or reasoning-with-tools |
| Multimodal server | Supported image/audio media may use function tools; video remains single-turn/non-mixed and cannot use tools |
| Anthropic | Manual summarized thinking uses handler-local signatures; adaptive/omitted/redacted/interleaved modes and thinking with tools are explicit exclusions |
| Architectures | Declared model graphs execute; marked real-model fixture validation remains in `compatibility.yaml` |
| Vision | Chameleon accepts projected soft tokens; its encoder/projector and image-token generator are absent from the pinned upstream execution path |
| Speculation | Integrated greedy/sampled coordinators; real target/draft pair validation remains fixture-gated |
| Audio | WavTokenizer waveform synthesis complete; Gemma 3n audio remains absent from pinned mtmd execution |
| Adapters | One aLoRA may be active, matching pinned server policy; non-causal diffusion rejects aLoRA |
| Platforms | Primary supported release remains Windows amd64 with NVIDIA CUDA |

## Deferred or externally blocked

| Item | Reason | Resume condition |
| --- | --- | --- |
| Real-model validation for marked architectures | Remaining rows lack paired model and oracle artifacts; Gemma 4 projector/prompt validation is complete, but language-model generation remains pending | Supply or generate the paired artifacts named by each pending row |
| Release signing | No user-controlled signing certificate | Certificate and signing policy supplied |
| Hosted-tool integration | No repository-wide hosted executor contract | Local bounded defaults implemented or external contract supplied |
| Real speculative model pairs | Matching target/draft fixtures unavailable | Compatible pair supplied or generated |
| Gemma 3n audio encoder | Pinned mtmd explicitly skips Gemma 3n audio execution | Upstream executable contract plus GGUF/oracle fixture supplied |
| Chameleon vision path | Pinned source has decoder-only projected-token execution | Upstream projector/image-token contract plus GGUF/oracle fixture supplied |
| Optional classifier/rerank heads | Corresponding heads are absent from pinned model graphs | Upstream executable graph and fixture supplied |
| Non-Windows release validation | Required host/device environments are unavailable in this worktree | Target CI runners or machines supplied |

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
