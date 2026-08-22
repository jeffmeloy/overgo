# Overgo technical architecture and incorporation assessment

Research snapshot: commit `e743f7e7839ca2a970ccd92476673493d5cdc631`, version `0.1.1`.

## Purpose and method

This is a code-level assessment of what Overgo does and which of its technical ideas are candidates for incorporation into another system. It is not an implementation plan and does not treat README claims as proof. The trace followed the CLI and HTTP entry points into artifact identity, RepoDB, recipe compilation and activation, model loading, tensor planning, host/CUDA execution, training, evaluation, and repository verification. Tests, compatibility declarations, and CI configuration were then used to estimate evidence strength and maturity.

Repository scale at this snapshot:

- 909 non-test Go files and 739 Go test files.
- 66 command directories and 97 top-level internal package directories.
- 48 cataloged feature claims in `compatibility.json`.
- One non-standard-library Go dependency (`regexp2`), with CUDA accessed directly through the Windows driver ABI.

## Executive finding

Overgo is best understood as two systems joined together:

1. A broad, correctness-first local model runtime with GGUF/safetensors ingestion, host and CUDA tensor execution, token generation, embeddings, reranking, multimodal projection, training workflows, evaluation suites, and compatible HTTP APIs.
2. A content-addressed control plane that makes exact artifacts, compiled recipes, evidence, lifecycle decisions, and lineage authoritative over model names or mutable paths.

The second system is the more differentiated and more transferable contribution. The runtime has substantial breadth and unusually explicit contracts, but its own compatibility data still marks every one of 136 model architecture entries as `experimental`. Most architecture entries lack named real-model fixtures. It should therefore be treated as a research runtime and validation workbench, not as drop-in evidence of broad production compatibility.

The most valuable ideas to incorporate are:

- content-addressed model and workflow authority;
- typed, deterministic recipe compilation;
- evidence-gated activation with compare-and-swap aliases;
- exact checkpoint/resume authority;
- evaluation plans that bind code, environment, data, model, recipe, and scorer;
- a small neutral tensor IR compiled to both a reference backend and CUDA;
- bounded parsers and explicit resource policies at trust boundaries.

The least portable parts are the Windows-only CUDA ABI layer, the embedded kernel set optimized for one recorded compute capability, and the full architecture compatibility surface.

## System model

```text
Physical files / datasets / source configs
                 |
                 v
       content-addressed artifacts
       manifests + tensor inventories
                 |
                 v
 RepoDB append-only facts and lineage <---- run/evaluation/gate evidence
                 |
                 v
 typed recipe definition --validate--> deterministic compiled program
                 |                              |
      evidence-gated active alias               |
                 |                              |
                 +--------------+---------------+
                                v
                    resolved model definition
                    + compiled model plan
                                |
              +-----------------+-----------------+
              |                                   |
        host reference                     CUDA executor
        tensor execution           locked worker + PTX + graph cache
              |                                   |
              +-----------------+-----------------+
                                v
             CLI / OpenAI / Anthropic / llama.cpp-like HTTP
              generation, embedding, rerank, media, workflows
```

The key architectural rule is that a path supplies bytes, not authority. Authority comes from a recipe activated for the exact content identity of those bytes.

## 1. Content identity and RepoDB

### Artifact identity

`internal/artifact/id.go` defines typed IDs of the form `kind:sha256:<digest>`. Kinds cover models, tensor sets, tokenizers, projectors, adapters, datasets and shards, checkpoints, recipes, outputs, evidence, profiles, runs, evaluations, tensor inventories, and model definitions. JSON documents are canonicalized and hashed through the same identity system.

An artifact descriptor states identity, size, media type, and schema. Content, manifests, aliases, locations, and lineage are distinct facts (`internal/artifact/schema.go`). That separation is useful:

- identity is immutable and content-derived;
- a manifest can represent a multi-file logical artifact;
- locations can change without changing identity;
- aliases provide controlled mutability;
- lineage records why one artifact depends on or derives from another.

Document codecs validate that stored bytes reproduce the claimed identity and satisfy the expected media type, schema, and size bounds (`internal/artifact/document.go`).

### What RepoDB actually is

RepoDB is not a relational database or remote artifact service. It is a local append-only ledger in `repodb.log`, plus rebuilt in-memory indexes (`internal/repodb/store.go`, `internal/repodb/log.go`).

Each frame has CRC32C integrity and participates in a SHA-256 commit hash chain. Opening the store replays the log, truncates a torn final frame, and rejects checksum or chain corruption. Commits are normalized before append and are durable-synced. A writer lock permits one writer; read-only processes can replay the store independently.

Commit validation is stronger than a plain metadata store:

- descriptors, content, and manifests cannot conflict with existing facts;
- lineage endpoints must exist and lineage cycles are rejected;
- alias updates use compare-and-swap through an expected previous target;
- idempotency keys may repeat only with identical payloads;
- facts are canonically sorted so equivalent batches have stable encoding.

The design is appropriate for a local audit ledger and reproducibility authority. It is not, without another service layer, a multi-host coordination database, object store, query engine, or high-write-throughput event system. Inline log frames are bounded to 64 MiB, while large physical artifacts are normally represented by identity and location rather than embedded bytes.

## 2. Recipes are the execution control plane

### Schema and type system

Recipe v2 (`internal/recipe/schema.go`) defines 13 task labels: inference, generation, embedding, rerank, projection, training, forecast, tabular, seq2seq, speech, image generation, video generation, and VQA.

A recipe contains:

- the exact model artifact ID;
- typed dependencies with roles and slots;
- nodes bound to immutable module contracts;
- typed edges, inputs, and outputs;
- host, device, or hybrid placement;
- session lifetime policy;
- weight residency policy.

Residency values include streaming, host cache, full device F32, native device storage, BF16 variants, hybrid-native, and host-reference. Dependencies distinguish model, model definition, profile, tokenizer, projector, adapter, dataset, checkpoint, objective, precision, placement, memory, optimizer, evaluator, promotion policy, and other authorities.

### Deterministic compilation

`internal/recipe/definition.go` canonicalizes dependency, node, edge, input, and output order before computing the recipe ID. It requires a model dependency in slot zero and rejects duplicates.

`internal/recipe/validate.go` then validates the graph against an immutable module catalog:

- task and placement must be admitted by the module contract;
- port types and cardinality must match;
- one-producer rules and required inputs are enforced;
- cycles are rejected;
- every relevant node must be reachable in the forward and backward graph sense.

`internal/recipe/program.go` emits a deterministic topological sequence of stages. At execution time, `internal/workflowruntime` binds each module ID to an adapter and checks every input and output against the recipe data kind. It commits a terminal run record even when execution fails or the request is cancelled.

One important boundary: the generic workflow runtime explicitly refuses training programs as orchestration-only. Native training is executed by `internal/trainingworkflow`, which interprets the compiled training recipe and binds it to the dense training implementation. Thus the recipe type system is shared, but there is not one universal executor for all task kinds.

### Activation and mutable state

Recipe lifecycle is candidate -> validated -> active, with refused and superseded states (`internal/modelrecipe/lifecycle.go`). The only mutable serving pointer is an alias scoped by exact model ID and task. Alias movement is compare-and-swap.

Activation is not a flag flip. `ActivateCapability` requires a successful recipe-bound gate and run, creates a typed acceptance decision with an evidence tier, and only then advances the active alias. Reverification can attach newer proof to an already active definition. Retirement requires failed proof plus a typed refusal decision.

This is the strongest incorporation candidate: it converts “the configured model” from a mutable name/path into an exact, reviewable tuple of bytes, architecture policy, execution policy, and evidence.

## 3. Exact model loading and inference

### Admission path

Normal CLI and server model opening goes through `internal/clioptions/model.go` and `modelrecipe.ResolveActiveGGUF`, not directly from a file path to a runner.

The resolver:

1. Opens the GGUF, including local split files.
2. Hashes/inventories it to obtain the exact model and tensor-inventory identities.
3. Resolves the active inference recipe for that exact model ID.
4. Loads the recipe's model definition, architecture profile, and tensor inventory.
5. Re-reads the file's model spec and refuses if it differs from the stored definition or inventory.
6. Reads weights and compiles a model plan.
7. Seals the loaded program so ownership of the underlying file/weights is transferred once into `inference.OpenWithProgram`.

Runtime flags can choose device ordinal and related operating parameters, but they cannot silently replace the recipe-owned residency mode.

### Model compilation

`internal/model` converts GGUF metadata and weight catalogs into a `ModelPlan`. The plan fixes per-layer operator instructions, attention/cache policy, residual and normalization behavior, projections, and terminal behavior. A separate forward program selects cached causal, bidirectional/encoder, encoder-decoder, audio, diffusion, or session-oriented execution.

`internal/model/graph_dispatch.go` dispatches compiled operators rather than switching on public model names at each layer. The operator set includes ordinary and modified causal attention, cross/relative attention, mixture-of-experts routing, selective scan, gated delta, short convolution, and RWKV/WKV-style recurrent operations. Architecture-specific reading still exists in the profile/spec compiler, but the execution boundary is a neutral layer program.

### Runner capabilities

The `inference.Runner` owns tokenizer, model plan, weights/residency, optional adapters and output bias, cache state, and host/CUDA executors. Its surface includes:

- greedy and sampled text generation;
- embeddings and token embeddings;
- pair reranking;
- causal, recurrent, hybrid, bidirectional, encoder-decoder, audio, and diffusion forward modes;
- LoRA and invocation-activated adapter handling;
- grammar-constrained generation;
- speculative and multi-token-prediction paths;
- prompt cache, serializable session/cache state, cache editing and shifting;
- multimodal embedding overrides and position inputs;
- continuous batching with paged device cache.

The ordinary `Generate` method takes the runner mutex for the whole generation. When server concurrency is configured above one and the runner supports it, the server constructs a `ContinuousGenerator`, which owns multi-sequence scheduling instead of issuing concurrent calls through the ordinary runner path. The server default remains one admitted generation request.

### Residency behavior

`inference.OpenWithProgram` can keep weights in host reference form, preload F32, retain native quantized/F16/BF16/FP8 weights, decode native weights to BF16, cache on host, or stream weights. Hybrid-native attempts device retention and can fall back to streaming after a measured CUDA out-of-memory condition, releasing partial allocations first.

That fallback is operationally useful, but it also means performance is recipe- and hardware-dependent. Evidence should record the realized lane, not infer it only from the requested residency policy.

## 4. Tensor IR and CUDA execution

### Neutral tensor graph

`internal/tensor` is a compact graph IR. Tensor nodes have stable IDs, shapes, storage types/views, operations, attributes, and input edges. Topological compilation checks cycles, duplicate IDs, operation attributes, and storage-view validity.

The same graph concepts feed a host reference executor and the CUDA executor. This makes differential tests possible and provides a correctness oracle for new kernels. It is an attractive pattern to incorporate even if the specific operator set is not reused.

### CUDA architecture

The production CUDA backend is deliberately cgo-free and Windows-specific:

- `internal/cuda/driver/driver_windows.go` dynamically loads `nvcuda.dll` and resolves the CUDA Driver API through `syscall`.
- `internal/cuda/cublas/cublas_windows.go` similarly binds cuBLAS.
- Go objects passed through DLL calls are pinned with `runtime.Pinner` where required.
- `internal/cuda/device/worker.go` creates a context and stream on a goroutine locked to one OS thread and serializes all work for that context.
- Embedded PTX is validated against a generated kernel ABI/content manifest before module use.

`internal/cuda/executor` compiles tensor topology, memory planning, kernel selection, and graph rewrites. It uses a device arena and reusable power-of-two buffer pool. Recognized subgraphs are fused (for example weighted RMS normalization, activated gates, GELU, projection-add, append, attention, and argmax paths).

Repeated decode traces can be captured as CUDA graphs. A four-entry LRU cache matches a compiled graph plus its indexed pointer replay frame, prefers graph-exec update, and otherwise instantiates a new graph. This avoids reissuing the same per-token kernel sequence from Go in steady state.

### Portability boundary

The checked-in compatibility data records toolkit 12.9, driver 595.79, compute capability 8.9, and kernel targets `sm_89`/`compute_89`. Unsupported-platform files let the repository compile elsewhere, and CI cross-builds Linux, but the actual CUDA driver backend and release artifact target Windows amd64. Incorporating this backend means accepting a Windows/NVIDIA-specific maintenance surface or replacing the ABI and kernel-distribution layer.

## 5. File formats and model intake

### GGUF

`internal/gguf` provides bounded parsing, random-access tensor views, metadata/tensor validation, split-file loading, hashing, writing, split/merge, quantization, and importance-matrix support. Parser options bound strings, arrays, metadata, tensors, alignment, and split count. Split models must be opened through the first conventionally named shard and are cross-checked for declared counts.

### Safetensors and PyTorch input

`internal/safetensors` opens single or indexed sharded repositories with explicit limits on index/header size, shard count, tensor count, rank, and name size. It validates dtype, byte range, overlaps, and the index-to-shard catalog while preserving random-access payload views.

`internal/modelartifact` builds content-addressed multi-file manifests and typed tensor inventories. Safetensors facts include names, shapes, storage, and byte counts. A PyTorch zip reader catalogs `.pt`/`.pth` tensors for intake, but this is not evidence of general PyTorch execution.

`cmd/hf-gguf-convert` converts supported Hugging Face checkpoint directories to GGUF and can separately emit a multimodal projector. Optional recording commits extracted source configuration, bound to the converted model identity. Missing source declarations remain absent rather than being guessed.

These parser and inventory packages are relatively separable incorporation candidates, especially their bounded-input and exact-inventory patterns.

## 6. Serving surface

`cmd/server` opens one exact active model recipe and optionally an active projector session. It then exposes a single HTTP handler with an embedded web UI and operational endpoints.

Protocol coverage includes:

- OpenAI-style completions, chat completions, embeddings, Responses, image generation, and audio speech;
- Anthropic-style messages and token counting;
- llama.cpp-like native completion, tokenization, detokenization, infill, slots, metrics, health, and model metadata;
- reranking aliases;
- optional asynchronous training, model-building, and evaluation workspaces;
- read-only dataset/run browsing and bounded analysis tools.

The protocol layer supports streaming, function/tool-call formatting, grammar and JSON-schema constraints, multimodal prompt history, request-scoped LoRA scales, prompt caching, response continuation state, and token accounting.

Security and resource boundaries are explicit but local-runtime oriented:

- default listen address is loopback;
- bearer authentication is optional, read from a file or environment variable;
- request and media sizes, dimensions, pixels, token counts, embedding inputs, stored responses, concurrency, and timeouts are bounded;
- remote media is disabled unless an explicit scheme/host/port/network/redirect/MIME/size policy admits it;
- Responses file IDs and external tool types are denied unless a policy/resolver is installed.

This is a compatibility server, not a full multi-tenant service. There is no built-in identity provider, tenant isolation, quota service, distributed scheduler, or durable response database. The in-memory response history and local RepoDB assumptions should be preserved only for a trusted single-host deployment.

## 7. Native training

### Supported mainline workflow

`cmd/train` runs recipe-bound training from a safetensors model directory. `internal/trainingworkflow/workflow.go` resolves the active training recipe for the exact model identity and rejects a caller-supplied recipe if it is not the active one.

Three dense objectives are recognized by exact compiled module sequences:

- token prediction: batch -> forward -> backward -> Muon;
- DPO: preference batch -> policy/reference scoring -> DPO loss -> backward -> Muon;
- GRPO: rollout batch -> policy scoring -> GRPO loss -> backward -> Muon.

Token prediction can use host reference or CUDA-resident execution and optionally freeze tied lexical weights on the CUDA lane. In the traced mainline workflow, DPO and GRPO execute on host. Other modality-training commands are specialized probes/workflows and should not be conflated with general support in `cmd/train`.

Optimizer policy comes from the recipe and is compiled against the actual parameter count and Muon parameter grouping. Dataset bytes are converted to a deterministic stream; token, preference, or rollout batches advance an exact stream identity and position.

### Checkpoint and resume contract

A checkpoint contains:

- run-plan and training-program IDs;
- base model and exact tensor-set identity of saved weights;
- dataset, split, processor, and stream identity/position;
- optimizer plan identity, step, momentum state, and parameter count;
- named RNG algorithm/seed/counter state for data and augmentation;
- projector/codec IDs and full lineage;
- an assertion that checkpoint publication occurs at an accumulation boundary.

Publication writes to a temporary sibling directory and atomically renames it only after weights and checkpoint JSON are complete. Loading re-hashes the weights. Resume is refused unless checkpoint, compiled program, dataset/split, stream identity, processors, projectors, and codecs all agree (`internal/trainingprogram/checkpoint.go`). This is a particularly strong reusable design.

The checkpoint directory itself is not automatically committed into RepoDB by this workflow; it is a filesystem publication whose document and lineage are content-addressed. A production integration would likely add explicit artifact upload/cataloging after atomic publication.

## 8. Model construction

`cmd/model-build` is not a general architecture search or large-model builder. It runs a small corpus-derived scratch-model campaign (`internal/modelbuilder/scratch.go`): initialize a deterministic construction, train it for a bounded number of steps, evaluate causal loss, record run/evaluation/evidence artifacts, and issue a promote/refuse decision based on whether validation loss improved.

The reusable part is its five-phase state machine (`internal/workflowruntime/model_builder.go`). Every phase must preserve recipe, dataset, and construction authority while adding the expected checkpoint, run, evaluation, evidence, and decision IDs. The specific scratch backend should be viewed as a demonstrator of the workflow contract.

## 9. Evaluation and evidence

`cmd/evaluate` reads a strict manifest, catalogs local benchmark sources into RepoDB, and launches one worker process per model. A worker opens the exact active GGUF program once and evaluates its listed suites as a resident campaign.

Compiled evaluation plans bind:

- model-definition ID;
- runtime-recipe ID;
- dataset and split IDs;
- case profile and scorer IDs;
- isolated or resident execution policy;
- Git commit (SHA-1 or SHA-256 form);
- environment-evidence ID.

Suite implementations include exact generation, multiple choice, generated-answer scoring, MMLU-Pro, grouped choice, probability-mass scoring, structured generation, and instruction-rule evaluation. Shard results are published under plan-specific aliases so exact completed work can be reused. Campaign reports depend on all shard reports.

A successful campaign publishes a run, metric record, report, evaluator, acceptance contract, and a consolidated evidence document. Evidence validation reloads every stored authority, rechecks identities and lineage, confirms the run succeeded under the expected code/environment, verifies report/shard presence, and confirms metric names, units, directions, and acceptance shape. Acceptance here is primarily a metric-contract check; threshold or comparative promotion logic belongs to higher-level admission records.

Evaluator promotion has an additional useful safeguard: a proposed evaluator can be tested against a ranked set of known model outcomes and must preserve strict dominance ordering.

## 10. Repository verification and change control

The repository contains a sizeable internal change-control system beyond ordinary CI:

- `cmd/gate` derives impacted test scope, enforces staged-path scope and plan binding, runs formatting/vet/build/test/manifest/SBOM/claim/document checks, performs the commit, and records a typed gate/run result in RepoDB;
- `cmd/compatibility -check` verifies that claim evidence points at the expected source/test content identities;
- generated kernel manifests bind PTX/source hashes, complete entry sets, ABI parameter order, launch conventions, and shared-memory conventions;
- the release builder removes Go build IDs/VCS stamps, uses fixed zip timestamps, builds twice, and compares outputs;
- a deterministic CycloneDX SBOM and license inventory are CI-verified;
- fuzz targets cover GGUF, tokenizer, grammars/schema conversion, sampling state, cache/session state, and server JSON.

Standard CI runs the hermetic test lane on Windows and Ubuntu, vet (including tagged compilation lanes), Linux cross-build, SBOM/kernel/compatibility checks, and race tests for server/inference. Real GPU tests are a manually dispatched self-hosted Windows job, not part of every push. Release output is Windows amd64 and is not code-signed; `compatibility.json` records signing as blocked because no certificate is controlled by the project.

This control-plane code is ambitious and valuable, but it is tightly coupled to this repository's planning, test-scope, and commit conventions. Incorporate the evidence model and invariants before considering the gate command itself.

## 11. Evidence-based maturity assessment

The repository is unusually honest about different proof levels:

- 48 feature claims are marked implemented.
- 35 are `contract-tested`.
- 10 are `device-gated`.
- 3 are `pinned-oracle`.
- all 136 model architecture records are `experimental`.
- four architecture records name a `validated_fixture` (`gemma3`, `qwen3`, `qwen35`, and `t5encoder`); Qwen 3.5 also names separate multimodal and video fixtures.
- 109 architecture records explicitly declare real-model validation as `pending-fixture`.

These counts are not contradictory. A feature such as “compiled model plan” can be thoroughly contract-tested while a particular architecture/quantization/model combination remains unvalidated. The architecture registry demonstrates breadth of parsers and operator composition; it does not establish broad numerical parity.

The CUDA differential tests are extensive, and device-gated adaptive parity fixtures add meaningful evidence. Still, the manual GPU lane, one recorded compute capability, small number of named model fixtures, and experimental status across the matrix are the dominant readiness constraints.

## 12. Incorporation recommendations

### Tier 1: incorporate the concepts now

1. **Typed content IDs and separate mutable aliases.** Preserve the distinction among immutable identity, physical location, manifest composition, and active name.
2. **Canonical recipe compilation.** Hash normalized definitions, validate typed ports and graph reachability, and execute only compiled programs from an immutable module catalog.
3. **Evidence-gated activation.** Make promotion an atomic alias transition that references exact successful gate/run evidence and an explicit decision.
4. **Exact evaluation plans.** Bind model, recipe, dataset split, scorer, code revision, environment, and lifecycle into one plan ID before evaluation begins.
5. **Exact resume authority.** Persist data-stream position, RNG counters, optimizer-plan identity, processors, codecs/projectors, and weight identity; fail closed on mismatch.
6. **Bounded parsers and policies.** Adopt explicit allocation/count/range limits for model formats and allowlist-based remote resource access.

These ideas are backend-agnostic and solve common reproducibility and governance failures without requiring Overgo's model kernels.

### Tier 2: prototype behind adapters

1. **Neutral tensor/layer programs.** Use a small typed IR with a slow reference backend and an optimized backend. Validate new graph rewrites differentially.
2. **Resident session director.** Reuse the bounded capacity, bounded waiting, exact component lifetime, and idle eviction model, but connect it to the target system's scheduler and observability.
3. **Run/evaluation evidence documents.** Map them into the target metadata store rather than adopting local RepoDB unchanged.
4. **GGUF/safetensors inventory logic.** Reuse or port after adversarial parser review and fixture testing against the target model catalog.
5. **CUDA graph replay and arena planning.** Prototype only on target GPUs and workloads; measure graph-cache hit rate, memory fragmentation, fallback behavior, and numerical parity.

### Tier 3: do not adopt wholesale yet

1. **The full architecture registry.** Require a per-model qualification matrix with exact artifact IDs, numerical criteria, generation outputs, performance, and target hardware.
2. **The Windows CUDA ABI layer.** Adopt only if Windows/NVIDIA/cgo-free deployment is itself a requirement.
3. **Mainline RL training.** Host-only DPO/GRPO and the narrow dense model loader need independent scale and correctness evaluation.
4. **The complete gate/plan automation.** Its assumptions are repository-specific and much broader than the portable evidence primitives.
5. **The compatibility server as a multi-tenant service.** Put it behind a hardened service boundary or add identity, quotas, durable state, distributed scheduling, and tenant isolation.

## 13. Proposed technical experiments before incorporation

1. Select two target models and two target GPUs. Catalog exact model IDs and run host-vs-CUDA logits, greedy-token, cache-edit, and long-context comparisons.
2. Exercise each intended residency mode under constrained VRAM and verify that measured lane/fallback evidence is recorded correctly.
3. Kill training immediately before and after checkpoint rename; verify no partial target is accepted and exact resume reproduces the uninterrupted next step.
4. Corrupt and tear RepoDB frames, race alias promotion, and validate backup/restore on a realistically sized ledger.
5. Re-run evaluation after changing code commit, environment, recipe, one dataset case, and scorer policy; every change should produce a distinct plan and prevent stale-result reuse.
6. Fuzz GGUF and safetensors intake with the target system's resource limits and run a separate unsafe/ABI review of Windows syscall boundaries.
7. Load-test continuous batching, prompt-cache forks, cancellation, and cache shifting; report latency percentiles and memory, copy, launch, and synchronization counters.
8. Validate API compatibility against the actual client SDKs in scope. Endpoint resemblance should not be treated as full behavioral compatibility.

## Bottom line

Overgo's most compelling contribution is not the number of model families it names. It is the attempt to make every execution and promotion explainable as an immutable chain:

`bytes -> inventory -> model definition -> compiled recipe -> run -> evaluation -> decision -> active alias`.

That chain is technically substantive and worth incorporating in stages. The model runtime should be evaluated as an interchangeable execution backend behind that control plane, with qualification performed per exact model, quantization, task, residency policy, GPU, and evidence tier.
