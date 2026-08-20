# Overgo

Overgo is a native Go model-engineering platform for creating, training,
evaluating, composing, serving, and analyzing models across text, image,
audio, video, time-series, and tabular workloads on consumer NVIDIA hardware.

Instead of implementing a separate executor for every model family, Overgo
compiles typed recipes from shared model, tensor, training, media, and device
components. Architecture profiles describe model structure. Recipes bind exact
content-addressed artifacts to execution topology, placement, residency,
processors, objectives, and evidence.

The same runtime powers native training and evaluation, OpenAI-, Anthropic-,
and llama.cpp-compatible APIs, command-line workflows, and an integrated model
engineering workbench. RepoDB records artifact identity, lineage, runs,
verification, promotion, and rollback so that catalog support, implemented
capability, artifact verification, and production activation remain explicit
and separate.

Overgo uses selected formats, numerical semantics, quantization behavior, and
kernel concepts from llama.cpp, but its overall architecture is broader. It
adds recipe compilation, common multimodal execution, native training,
evaluation campaigns, model construction, evidence-bound activation, and one
GUI spanning the model lifecycle.

**Current host:** Windows amd64, Go 1.26, NVIDIA CUDA driver,
`CGO_ENABLED=0`. `github.com/dlclark/regexp2/v2` is the sole third-party Go
runtime dependency. Public interfaces may change while the platform is under
active development.

## 1. What Overgo is

Overgo is one runtime for the work that normally crosses several model tools:

| Area | Capabilities |
| --- | --- |
| Model intake | Inspect GGUF and safetensors, convert Hugging Face repositories, split and merge GGUF files, quantize weights, inventory tensors, and assign content identity |
| Model definition | Bind immutable architecture profiles, tensor requirements, model definitions, and capability recipes |
| Model execution | Run dense, MoE, recurrent, hybrid, encoder, encoder-decoder, diffusion, embedding, reranking, speculative, and constrained-generation programs |
| Modalities | Process text, image, audio, and video inputs; generate text, images, video, and speech; forecast series; predict tabular values; run VQA and OCR workflows |
| Model creation | Construct deterministic scratch models, initialize parameter manifests, build adapters, propose components, and preserve derived-model lineage |
| Training | Compile objectives, stream datasets, execute Muon updates on host or CUDA, checkpoint, resume exactly, and record training evidence |
| Preference optimization | Run DPO and GRPO with shared scoring, VJPs, Muon updates, checkpoints, evaluation, and GUI reporting |
| Evaluation | Compile suites, run campaigns, inspect failures, compare metrics, and bind results to model, recipe, data, code, and environment identities |
| Serving | Expose native llama.cpp-style, OpenAI-compatible, and Anthropic-compatible HTTP APIs with streaming, tools, structured output, media, batching, and caches |
| Workbench | Use one embedded GUI for chat, generation, runtime telemetry, datasets, training, model building, export, evaluations, artifacts, and model analysis |
| Evidence | Record artifact identities, lineage, runs, recipes, gates, evaluations, promotion, rollback, and release checks in RepoDB and generated compatibility records |

Overgo is designed around compositional capability. A model does not require a
new top-level executor merely because it has a new brand or architecture name.
If its tensor layout and behavior can be expressed with existing profiles,
operators, processors, and recipe modules, it can reuse the same compiled
runtime paths. A model-specific implementation is needed only when the model
introduces a materially new tensor convention, operator, processor, codec,
numerical rule, or execution topology.

This makes two claims deliberately different:

- **The code can represent and execute a capability** through tested generic
  components and a valid recipe.
- **A particular model artifact has been verified** with its exact bytes,
  recipe, inputs, environment, and expected result.

The first scales by composition. The second remains artifact-specific and is
recorded through RepoDB.

![Overgo platform architecture and model lifecycle](docs/assets/overgo-platform-architecture.png)

## 2. How Overgo works

### Shared components instead of model-family executors

Overgo factors model behavior into reusable owners:

- architecture profiles describe dimensions, tensor layout, attention,
  recurrence, experts, normalization, position encoding, and operator policy;
- tensor and graph packages describe computation independently of model names;
- host operators provide numerical references;
- CUDA operators execute compiled graphs and training programs;
- processors, projectors, tokenizers, schedulers, and codecs handle typed data;
- training programs own objectives, backward traversal, parameter groups, and
  optimizer order;
- runtime adapters bind compiled recipe modules to implementations;
- RepoDB owns durable identity, lineage, evidence, and activation decisions.

Runtime code consumes compiled plans. It does not rediscover model behavior
from a model-family switch during execution.

### Profiles, definitions, and recipes

A runnable capability is assembled from immutable records:

1. **Model artifact:** content-hashed model bytes and known locations.
2. **Architecture profile:** model structure and operator policy with fact
   provenance.
3. **Model definition:** the exact model, profile, tensor inventory,
   dimensions, and validated tensor facts.
4. **Capability recipe:** typed modules for one task, including dependencies,
   inputs, outputs, ordering, placement, session lifetime, and residency.
5. **Compiled program:** validated stages and indexed bindings consumed by the
   runtime.
6. **Run and evaluation records:** exact inputs, outputs, environment, outcome,
   and evidence associated with execution.

Recipe compilation rejects unknown modules, invalid ordering, incompatible
data kinds, unbound ports, incorrect cardinality, graph cycles, unreachable
nodes, missing dependencies, and placement or residency conflicts.

### Execution and evidence loop

```text
model bytes + tensor facts + architecture profile
                         |
                         v
               typed capability recipe
                         |
                         v
                  compiled program
                         |
                         v
          reusable host/device model session
                         |
                         v
             shared Go and CUDA operators
                         |
                         v
          output artifacts + run observations
                         |
                         v
            evaluation + gate + decision
                         |
                         v
              promoted active recipe
```

Inference and production commands resolve the active recipe associated with a
model artifact's content hash. They do not select a different residency mode or
model-family path from convenience flags. Activating a candidate requires
successful evidence tied to the exact model and recipe identities. Changing
model bytes, tensor facts, profiles, or recipe content creates a new identity
and requires new evidence.

RepoDB can retain several candidates for the same model and task. Only a
promoted active recipe is used by production entry points. Missing evidence is
an error rather than a reason to fall back to another execution path.

## 3. System architecture

### Typed execution runtime

Recipes are executable data rather than descriptive labels. A versioned recipe
names one task, its artifact dependencies, typed modules, host/device
placement, ports, ordered edges, inputs, outputs, residency, and session
lifetime.

Supported task classes include inference, token generation, embedding,
reranking, projection, training, forecasting, tabular prediction,
sequence-to-sequence generation, speech synthesis, image generation, video
generation, and VQA.

Opening a model builds one indexed weight catalog. Compiled requirement schemas
validate alternative shapes, storage, and cross-tensor relations. Layer
programs carry indexed bindings so execution does not repeatedly search tensor
names or reapply family policy.

### Host and CUDA execution

The runtime remains no-cgo. It loads installed NVIDIA DLLs through the Windows
ABI and maintains each CUDA driver context on a worker locked to one operating-
system thread. Go memory is pinned while its address crosses a DLL boundary,
and device copies are validated against live driver allocations.

Compiled graphs reuse validated topology, BLAS selection, memory-arena plans,
indexed operands, fusion descriptors, and replay state. CUDA and PTX assets in
the kernel manifest are the only non-Go runtime components maintained in the
repository.

Preloaded decoding keeps weights, KV state, recurrent state, and graph outputs
on the GPU. Only sampling inputs return to Go. Media sessions retain branch
graphs, conditioning, prefix KV, hidden state, and latent state across
generation steps where the recipe permits it.

Sessions are keyed by model artifact, recipe, compiled resources, device, and
execution policy. Session and component lifetimes are explicit recipe
properties. Capacity-bound sessions may be reused; request-bound components
are retired after their lease. Separate session locks allow independent models
or capabilities to execute concurrently.

### Artifacts and RepoDB

RepoDB is a hash-chained artifact catalog and evidence ledger. It stores:

- content-addressed descriptors and optional payloads;
- model, tokenizer, projector, dataset, checkpoint, output, report, and
  evidence artifacts;
- construction, derivation, evaluation, and production lineage;
- aliases with compare-and-set activation semantics;
- artifact locations without copying large model or dataset bytes into Git;
- run, gate, evaluation, finding, decision, promotion, and rollback records.

RepoDB rejects conflicting artifact facts, repeated batch keys with different
content, invalid aliases, unknown lineage endpoints, and lineage cycles.
Execution records are committed even when an operation is cancelled or fails,
so the absence of an output does not erase the attempted run.

### Dataset and training architecture

Dataset documents are immutable, content-addressed records:

| Document | Meaning |
| --- | --- |
| `version` | Named external assets, artifact identities, and record counts |
| `view` | A source dataset plus an immutable selector or selected fields |
| `split` | Named partition views over one source dataset |
| `mixture` | Canonically ordered datasets with normalized integer weights |

The training materializer resolves documents, selectors, memberships,
processors, and external assets; validates sizes and content; removes exact
duplicates; and constructs a deterministic stream. Dataset identity, mixture
order, shuffle order, epoch, seed, and stream position are part of the resume
contract.

A training objective binds objective type, input and output modalities,
dataset and split, processors, optional projectors or codecs, loss, evaluation,
and evidence. A compiled `TrainingProgram` defines ordered forward, backward,
and Muon operations plus the exact trainable parameter plan. A
`TrainingRunPlan` adds model construction, checkpoint, data stream, precision,
placement, memory, evaluation, checkpointing, and promotion policy.

Resume is refused when the program, model, dataset, split, stream, processor,
projector, codec, or other authority differs from the checkpoint.

### Repository map

| Path | Purpose |
| --- | --- |
| `cmd/` | Thin command entry points for user workflows, verification, and development automation |
| `internal/model`, `internal/inference` | Model profiles, compiled programs, inference runners, caches, and session behavior |
| `internal/recipe`, `internal/modelrecipe` | Typed capability schemas, compilation, lifecycle, and model bindings |
| `internal/workflowrecipe`, `internal/workflowruntime` | Cross-capability recipe modules and execution |
| `internal/cuda`, `kernels/` | CUDA driver interface, executor, generated bindings, and manifested assets |
| `internal/tensor`, `internal/graphruntime` | Typed tensor graphs, planners, host references, and compiled graph execution |
| `internal/projector`, `internal/media` | Image, audio, and video decoding and projection |
| `internal/latentimage`, `internal/latentvideo`, `internal/oscillatorimage` | Generative image and video programs |
| `internal/densecausal`, `internal/hybridtrain`, `internal/adaptertrain` | Shared training and backward implementations |
| `internal/optimizer` | Muon parameter plans, state, host reference, and CUDA updates |
| `internal/dataset`, `internal/trainingdata` | Dataset documents, splits, mixtures, deterministic streams, and typed processors |
| `internal/trainingprogram`, `internal/trainingworkflow` | Compiled objectives, run plans, DPO/GRPO, checkpointing, and resume |
| `internal/scratchmodel`, `internal/modelbuilder`, `internal/composition` | Model construction, component synthesis, and derived lineage |
| `internal/evaluation` | Compiled evaluation suites, records, reports, and comparisons |
| `internal/artifact`, `internal/repodb`, `internal/runrecord` | Identity, storage, lineage, runs, observations, and decisions |
| `internal/server`, `internal/server/webui` | HTTP protocols and the embedded model engineering workbench |
| `compatibility.json` | Machine-checked feature and model claims |
| `docs/plan.json` | Current unfinished campaign work and verification commands |

## 4. Model and modality capabilities

### Model formats and construction

Overgo can inspect and validate GGUF and safetensors repositories, stream
Hugging Face conversion, merge and split GGUF files, quantize tensors, compute
content identity, inspect model metadata, and characterize tensor statistics.

It can also create models rather than only load them. Model-builder workflows
support deterministic scratch construction, parameter manifests,
initialization profiles, adapters, grafts, component proposals, derived-model
lineage, evaluation, and promotion decisions.

### Runtime model classes

The architecture catalog covers shared execution policies for representative
families in these groups:

| Runtime group | Representative profiles |
| --- | --- |
| Dense causal decoders | Llama, Gemma, Qwen, Mistral, GPT-2/J/NeoX, MPT, Phi, Falcon, Cohere, Granite, OLMo, StableLM, StarCoder |
| Mixture-of-experts | DeepSeek, Qwen MoE, DBRX, Arctic, Grok, Granite MoE, Hunyuan MoE, Exaone MoE, Ernie MoE |
| Recurrent and hybrid | Mamba, Mamba2, RWKV6/7, Jamba, Qwen3Next/Qwen3.5, Kimi Linear, GraniteHybrid, Falcon-H1, LFM2 |
| Encoders and embeddings | BERT, T5 encoder, ModernBERT, NeoBERT, Nomic BERT, Jina BERT, Gemma and Llama embedding profiles |
| Encoder-decoder and structured tasks | T5, Needle sequence-to-sequence, TimesFM forecasting, TabFM prediction, OCR, and reranking paths |
| Multimodal language | Gemma3/3n/4, Qwen2-VL/Qwen3-VL, Hunyuan VL, CogVLM, Chameleon, DeepSeek OCR, PaddleOCR |
| Diffusion and discrete generation | LLaDA, LLaDA-MoE, Dream, RND1, SimpleDiffusion, and latent or oscillator image/video programs |
| Media systems | Pocket-TTS, Krea and SenseNova image generation, Wan and LiveEdit video, Un-0 oscillator image/video |

Catalog entries express profiles and requirements. They do not claim that
every matching checkpoint has been installed and executed. Artifact-specific
status is generated in `docs/COMPATIBILITY.md`.

### Inference and structured generation

The common inference runtime supports dense, MoE, recurrent, hybrid, encoder,
encoder-decoder, diffusion-text, embedding, reranking, speculative, and
constrained token generation. Sampling includes common probability filters,
penalties, grammar and JSON-schema constraints, prompt caching, context
editing, and continuous multi-sequence execution where admitted by the active
recipe.

### Multimodal input

Typed processors and projectors accept image, audio, and video inputs,
including ordered mixed-media conversation history. Local media is bounded by
size and geometry. Remote media is disabled unless an explicit policy enables
allowed schemes, hosts, ports, redirects, timeouts, concurrency, private
network access, and response sizes.

### Image and video generation

Image generation uses typed conditioning, compiled denoising or integration
programs, resident CUDA sessions where supported, and durable output artifact
publication. Implemented routes include latent diffusion, routed image
transformers, and oscillator-based generation.

Video generation uses typed conditioning, temporal programs, resident latent
state, VAE encoding and decoding, and GIF or externally encoded publication.
Implemented routes include oscillator video, Wan-style latent video, and
LiveEdit-style source-conditioned generation.

### Speech, forecasting, tabular, and sequence-to-sequence

Pocket-TTS workflows provide text tokenization, latent generation, decoding,
and waveform publication. TimesFM-style programs produce quantile forecasts.
Tabular ICL programs produce row predictions. Encoder-decoder programs support
grounded sequence generation and structured tool-call output.

### HTTP protocols

The server implements:

- OpenAI-compatible completions, chat completions, embeddings, Responses,
  image generation, speech, reranking, input-token counting, streaming,
  structured output, and function tools;
- Anthropic-compatible Messages and token counting;
- llama.cpp-style completion, infill, embedding, tokenization,
  detokenization, template application, slots, LoRA controls, model
  properties, health, metrics, and discovery;
- native generation, training, model-builder, export, evaluation, artifact,
  dataset, run, recipe, operation, telemetry, and analysis endpoints.

Unsupported projectors, executors, tools, or recipe capabilities return an
explicit refusal rather than substituting an unrelated path.

## 5. Model creation, training, and composition

### Model creation routes

Overgo supports three construction routes:

1. Import or convert existing weights, derive a tensor inventory, bind an
   architecture profile, and compile task recipes.
2. Construct a model deterministically from dataset-derived topology,
   parameter manifests, initialization profiles, and named random streams.
3. Derive a model from adapters, grafts, or component proposals while
   retaining parent, construction, evidence, and decision lineage.

The model-builder workflow is available through `cmd/model-build` and the web
workbench. Import and conversion remain explicit format workflows. Serving a
newly constructed model requires an export form compatible with a serving
runtime profile.

### Shared Muon training

Muon is the production optimizer. Matrix, vector, and scalar groups use one
compiled parameter plan. Host code provides the numerical reference; resident
CUDA implementations provide device execution and checkpoint state.

Shared training infrastructure is used for dense causal, scratch,
encoder-decoder, speech, diffusion-image, adapter, recurrent, and hybrid model
paths where corresponding forward and VJP components exist. Each compiled
program identifies its objective and trainable parameter authority.

Checkpoints bind:

- weights and Muon momentum;
- update step and random-number stream counters;
- dataset identity and stream position;
- processor, projector, and codec identities;
- compiled training program and run-plan identities;
- model, parent, and RepoDB lineage;
- evaluation and promotion policy where applicable.

Publishing is atomic and refuses an existing target. Resume into a new target
requires exact authority agreement.

### Preference optimization and RL

DPO and GRPO are native objectives using the same recipe, data, execution,
optimizer, checkpoint, resume, evaluation, evidence, and GUI contracts as
other training workflows.

DPO validates a shared prompt prefix, derives completion masks, scores chosen
and rejected responses through policy and frozen reference models, computes
the relative margin and loss, executes score VJPs, and applies Muon updates.

GRPO reads candidate groups with evaluator-bound rewards, centers and
RMS-normalizes rewards within each group, scores completions, accumulates VJPs,
and applies the common update path. Equal-reward groups produce zero objective
gradient without an artificial threshold.

PPO, online environment rollout collection, reward-model training, and online
policy serving are not currently implemented.

### Component composition

Composition records describe model components, their capabilities, compatible
ports, required bridge operations, construction parents, and evidence. A
composition must compile through the same resource, placement, residency, and
session authorities as ordinary execution. Derived models retain acyclic
lineage to their parents and governing facts.

Recursive or self-improving trials do not authorize their own promotion.
Admission, development data, promotion data, evaluator, evaluation run,
decision authority, and rollback target are separate recorded facts.

## 6. Evaluation system

Evaluation is a native runtime subsystem, not a collection of external scripts
or a report imported after execution. Suites, cases, scorers, execution policy,
acceptance contracts, evaluators, reports, and results have typed,
content-addressed identities. Evaluation runs use the same model sessions,
recipes, artifact store, operation manager, CLI, HTTP server, and GUI as other
Overgo workflows.

### Compiled suites and plans

Suite JSON is strictly decoded and compiled before model work begins. The
compiler validates the suite kind, cases, scorer configuration, grouping,
metric contract, and referenced data. It then binds the compiled suite to an
evaluation plan containing:

- the exact model definition and runtime recipe;
- dataset and immutable split identities;
- case-profile and scorer identities;
- isolated or resident execution lifecycle;
- source commit and execution environment.

Changing any of those authorities produces a different evaluation-plan
identity. Results from different plans therefore cannot be treated as the same
experiment merely because they share a display name.

The campaign runner supports these compiled suite classes:

| Suite class | Evaluation behavior |
| --- | --- |
| Exact generation | Generate from fixed cases and require the declared exact result |
| Multiple choice | Score candidate continuations and report aggregate and grouped accuracy |
| Generated answer | Generate free-form answers and apply the suite's answer scorer |
| MMLU-Pro | Score multiple-choice cases with category-level metrics |
| Grouped choice | Evaluate demonstrated choices and report named-group accuracy |
| Probability mass | Measure the probability mass assigned to declared positive outcomes |
| Structured generation | Generate typed structured output and score validity and expected content by group |
| Instruction rules | Evaluate strict and loose prompt- and instruction-level compliance |

Evaluation packages also provide typed target scoring used by training and
model-building workflows:

- normalized text targets with exact, named, or verifier-backed scoring;
- numeric targets with explicit assumptions, tolerances, and per-record
  results;
- media targets with named oracles, verifiers, and artifact observations;
- preference targets for chosen and rejected outputs;
- supervised-fine-tuning views bound to immutable dataset splits.

These scorers are reusable components. A new model using an existing scorer
does not require a second implementation of the evaluation logic.

### Metrics, acceptance, and evaluators

Every compiled suite creates an acceptance contract before execution. A metric
contract names the metric, optional unit, and whether it must be minimized or
maximized. Reports cannot silently add, remove, rename, or reverse metrics
after a run.

An evaluator binds an evaluation plan to its acceptance contract. Evaluator
promotion can be tested against a sealed set of known outcomes: the candidate
must rank those outcomes consistently with every metric's declared direction,
and the promotion decision must name the same evaluator and sealed evidence.
This prevents a scorer from being adopted merely because it favors the current
candidate.

Supported reports include overall metrics, group or category metrics,
per-record observations, explicit failures, and typed output artifacts. Failed
records remain inspectable rather than disappearing into an aggregate score.

### Campaign execution and publication

An evaluation campaign executes one or more compiled suites against the same
bound model runtime. Successful execution publishes:

1. the compiled evaluation plan;
2. the suite report and its individual observations;
3. a bound run record with code, environment, timing, inputs, and outputs;
4. an evaluation record containing the declared metrics;
5. an evidence document connecting the plan, acceptance policy, evaluator,
   report, run, and evaluation.

Failed and cancelled campaigns still publish terminal run evidence. Cancellation
uses a context-independent finalization path so the cancelled request does not
erase its own execution history.

Exact-generation suites can be evaluated in deterministic shards and combined
without changing the governing plan. Campaign history is queryable by model,
and two evaluations can be compared metric by metric with an explicit
baseline, current value, delta, direction, and improvement result.

### Evaluation interfaces

Use `cmd/evaluate` for compiled suite execution and `cmd/eval-lane` for the
repository's evidence-aware evaluation lane. Start the server with repeatable
`-evaluation-suite <suite.json>` arguments and bind source identity with
`-evaluation-commit <commit>`.

The HTTP workspace exposes capability discovery, asynchronous campaign
execution, history, reports, failures, and comparison. The Evaluations tab uses
those same endpoints to select one or more suites, report progress, cancel
execution, inspect records and failures, browse model history, and compare two
runs against an explicit baseline.

## 7. Evidence and lifecycle management

### Evidence-bound lifecycle

RepoDB separates several lifecycle states:

```text
candidate recipe
      |
      v
compiled and contract-checked
      |
      v
artifact-specific run and evaluation
      |
      v
gate and independent decision
      |
      v
promoted active recipe
      |
      +----> observation and regression evidence
      |
      +----> rollback to recorded prior recipe
```

Activation is compare-and-set against the expected current binding. A failed
candidate does not overwrite an active recipe. A rollback names the prior
artifact and governing evidence instead of reconstructing historical state
from mutable configuration.

### Development and release evidence

The repository's Go automation connects code changes to the current campaign
plan and affected tests. The gate can derive package ownership from imports and
`go:embed` files, check formatting and vetting, build commands, verify generated
compatibility and SBOM data, inspect kernel manifests, and record results in
RepoDB.

CI, release, and local automation share the same short hermetic test owner.
Tests excluded from short mode are listed and are not counted as passes. Model,
GPU, smoke, and race lanes remain explicit commands because their prerequisites
and evidence scopes differ.

Independent SQA records can bind separate developer and reviewer worktrees,
candidate and evaluator revisions, finding resolutions, target commits, and
results. A human or external authority decides whether to merge or activate a
candidate.

## 8. Capability and verification levels

Overgo uses explicit status terms:

| Level | Meaning |
| --- | --- |
| **Cataloged** | Metadata, tensor requirements, and graph policy compile. This does not claim that every matching checkpoint runs correctly. |
| **Implemented** | Source code and a test or verification command establish a checked component or contract claim. |
| **Verified** | A named fixture or model artifact passed its stated contract, reference-output, numerical, or device test. |
| **Promoted** | A recipe tied to an exact artifact has successful gate and run evidence and may be selected by production entry points. |

Each claim is scoped to its named artifact, input, execution lifecycle,
precision, environment, and verification level. A profile may be cataloged
without installed model weights. A generic component may be implemented and
tested without rerunning every model that composes it. A model requires new
artifact-specific evidence before promotion even when all of its components
are already verified independently.

Representative testing is organized around genuinely different behavior:
operators, tensor conventions, topology classes, quantization formats,
processors, codecs, schedulers, and cross-component boundaries. Branded models
that reduce to the same proven components do not require duplicate unit tests,
but their exact artifacts still require deployment evidence.

Known open areas are maintained in `docs/plan.json` and the generated
compatibility matrix. They include model artifacts that are not locally
available, incomplete full-stack training for some large profiles, remaining
modality-pair evidence, incomplete comparable memory measurements, and
capabilities whose source or reference artifacts are absent.

## 9. Quick start

### Requirements

- Windows amd64
- Go 1.26
- an NVIDIA CUDA driver visible to the installed DLL loader
- model artifacts compatible with a compiled Overgo profile and recipe

FFmpeg is optional for encoded video other than native GIF. Select it with
`-ffmpeg`, `OVERGO_FFMPEG`, `PATH`, or a detected Windows installation.

### Configure data roots

Model and workflow commands resolve data in this order:

1. `OVERGO_DATA_ROOT`, containing `repodb-store/`, `models/`, `datasets/`, and
   `checkpoints/`;
2. machine-local `local-models.json` in the working directory;
3. those four directories directly under the working directory.

Example `local-models.json`:

```json
{
  "store": "D:/overgo-data/repodb-store",
  "models": "D:/overgo-data/models",
  "datasets": "D:/overgo-data/datasets",
  "checkpoints": "D:/overgo-data/checkpoints"
}
```

Large model, dataset, and checkpoint bytes stay outside Git. RepoDB records
their identities and locations.

### Verify the checkout

```bash
go run ./cmd/cuda-info
go run ./cmd/kernel-manifest
go run ./cmd/compatibility -check
go run ./cmd/test-lane ./...
```

The common lane is hermetic and reports classified short-mode skips. Real GPU
and model-artifact checks are separate:

```bash
go run ./cmd/device-lane
go run ./cmd/smoke-lane
go run ./cmd/race-lane
```

A required artifact or device result reported as `UNAVAILABLE` does not count
as a pass for a claim that requires it.

### Inspect a model

```bash
go run ./cmd/inspect-gguf -metadata -tensors D:/models/model.gguf
go run ./cmd/model-info D:/models/model.gguf
go run ./cmd/recipe status -task inference D:/models/model.gguf
```

### Verify and activate a recipe

Candidate verification accepts typed JSON. `-input -` reads standard input and
avoids shell quoting limits:

```bash
echo '{"text":"Weather in San Francisco?","max_tokens":64}' | \
  go run ./cmd/recipe verify -task seq2seq -input - needle
```

Verification prints recipe-bound gate and run identities. Activation consumes
those immutable records:

```bash
go run ./cmd/recipe activate \
  -task inference \
  -reason "validated artifact and execution policy" \
  -gate evidence:sha256:<gate-id> \
  -run-id run:sha256:<run-id> \
  D:/models/model.gguf
```

Generate through the active recipe:

```bash
go run ./cmd/generate -n 32 D:/models/model.gguf "Hello"
```

Placement, residency, and quantized execution come from the recipe rather than
generation flags.

### Start the server and GUI

```bash
go run ./cmd/server \
  -listen 127.0.0.1:8080 \
  D:/models/model.gguf
```

Open `http://127.0.0.1:8080/`. Add `-mmproj <projector.gguf>
-mmproj-cuda` for supported image, audio, or video requests. Set
`OVERGO_API_KEY` or use `-api-key-file` to protect model execution and
state-changing endpoints.

Enable additional workspaces when their dependencies are configured:

```bash
go run ./cmd/server \
  -listen 127.0.0.1:8080 \
  -training \
  -model-builder \
  -evaluation-suite D:/overgo-data/evaluations/core.json \
  D:/models/model.gguf
```

### Train and resume

Train a compatible dense safetensors model:

```bash
go run ./cmd/train \
  -model D:/models/carbon \
  -dataset D:/datasets/train.txt \
  -out D:/checkpoints/carbon-run \
  -steps 4 -freeze-lexical
```

Resume into a new atomic checkpoint target:

```bash
go run ./cmd/train \
  -resume D:/checkpoints/carbon-run \
  -dataset D:/datasets/train.txt \
  -out D:/checkpoints/carbon-run-2 \
  -steps 4 -freeze-lexical
```

Run `go run ./cmd/<name> -h` for command flags. Subcommand help follows the
subcommand, for example `go run ./cmd/recipe status -h`.

## 10. CLI catalog and commands

### Model inspection, formats, and conversion

| Command | Purpose |
| --- | --- |
| `cmd/inspect-gguf` | Inspect GGUF metadata, tensors, and storage |
| `cmd/inspect-safetensors` | Inspect safetensors repositories and tensor catalogs |
| `cmd/model-info` | Report compiled model and architecture information |
| `cmd/model-characterize` | Compute bounded tensor and model characterization records |
| `cmd/hf-gguf-convert` | Stream supported Hugging Face repositories into GGUF |
| `cmd/gemma4-gguf-convert` | Convert Gemma 4 model and multimodal artifacts |
| `cmd/gguf-hash` | Compute GGUF content identity |
| `cmd/gguf-merge` | Merge validated split GGUF artifacts |
| `cmd/gguf-split` | Split GGUF artifacts while preserving validated structure |
| `cmd/gguf-quantize` | Quantize GGUF tensors through manifested encoders |
| `cmd/tokenize` | Inspect tokenizer encoding and decoding |
| `cmd/json-schema-grammar` | Compile JSON Schema into constrained-generation grammar |
| `cmd/gen-iq-tables` | Generate checked integer-quantization tables |

### Inference, generation, and serving

| Command | Purpose |
| --- | --- |
| `cmd/generate` | Run text or admitted multimodal generation through an active recipe |
| `cmd/server` | Start HTTP protocols and the embedded workbench |
| `cmd/embedding` | Produce encoder embeddings with pooling and normalization |
| `cmd/rerank` | Score and rank query-document pairs |
| `cmd/diffusion` | Run supported diffusion-text programs |
| `cmd/latentvideo-run` | Execute resident latent-video generation and verification |
| `cmd/perplexity` | Measure next-token or disjoint-window perplexity |

### Recipes, models, training, and composition

| Command | Purpose |
| --- | --- |
| `cmd/recipe` | Inspect, verify, activate, and execute capability recipes |
| `cmd/train` | Run compiled Muon, DPO, or GRPO training with checkpoint resume |
| `cmd/model-build` | Construct, evaluate, and record new models |
| `cmd/adapter-train-probe` | Verify adapter training paths |
| `cmd/flow-organ-train-probe` | Verify flow-head component training |
| `cmd/mixture-train-probe` | Verify mixture and routed-component training |
| `cmd/graft-probe` | Verify component graft and synthesis paths |
| `cmd/oscillatorimage-train-probe` | Verify oscillator-image training |
| `cmd/seriesforecast-train-probe` | Verify forecasting training |
| `cmd/tabular-train-probe` | Verify tabular training |
| `cmd/controller-action` | Compile and execute repository-controller actions |

### Evaluation, evidence, and repository data

| Command | Purpose |
| --- | --- |
| `cmd/evaluate` | Execute compiled evaluation suites and write records |
| `cmd/eval-lane` | Run the repository evaluation lane |
| `cmd/benchmark` | Record elapsed time, throughput, memory, launches, synchronization, and transfers |
| `cmd/perf-sweep` | Execute bounded performance sweeps |
| `cmd/compatibility` | Check claims or regenerate evidence identities and the compatibility matrix |
| `cmd/repodb-query` | Query artifacts, recipes, runs, evaluations, findings, and decisions |
| `cmd/repodb-import` | Import validated external records into RepoDB |
| `cmd/repodb-backup` | Create and verify RepoDB backups |
| `cmd/finding` | Create and inspect structured review findings |
| `cmd/advisories` | Inspect repository and dependency advisories |
| `cmd/sbom` | Generate or verify the CycloneDX software bill of materials |

### CUDA, kernels, parity, and diagnostics

| Command | Purpose |
| --- | --- |
| `cmd/cuda-info` | Inspect CUDA driver and device capabilities |
| `cmd/cuda-smoke` | Run focused CUDA smoke execution |
| `cmd/device-lane` | Run real-device verification selected by repository evidence |
| `cmd/block-check` | Compare a model block between host reference and CUDA execution |
| `cmd/build-kernels` | Build maintained CUDA assets |
| `cmd/kernel-bindings` | Generate or verify Go bindings for kernel interfaces |
| `cmd/kernel-manifest` | Verify the allowed CUDA and PTX asset manifest |
| `cmd/sealed-authority` | Verify compiled runtime authority boundaries |
| `cmd/vqaparity` | Run VQA reference and parity checks |
| `cmd/sensenovaparity` | Run SenseNova image-generation parity checks |
| `cmd/smoke-lane` | Run model and capability smoke checks |
| `cmd/race-lane` | Run race and synchronization verification |

### Planning, gates, and release automation

| Command | Purpose |
| --- | --- |
| `cmd/plan` | Inspect and verify the current campaign plan and worktree context |
| `cmd/gate` | Validate a planned change, commit it, and record evidence |
| `cmd/guard` | Protect shell execution and durable data roots |
| `cmd/test-lane` | Run the classified hermetic Go test lane |
| `cmd/release` | Build releases and verify reproducibility |
| `cmd/closure-scan` | Scan code and evidence closure requirements |
| `cmd/admission` | Evaluate bounded admission records |
| `cmd/loop` | Run the repository's plan-driven iteration loop |
| `cmd/loophook` | Execute validated loop lifecycle hooks |

The gate associates staged paths with the current plan step, derives affected
tests, checks structural and generated records, and writes RepoDB preparation
and result evidence. If result recording fails after a Git commit, use:

```bash
go run ./cmd/gate -watchdog
go run ./cmd/gate -reconcile
```

Additional release checks:

```bash
go vet ./...
go vet -tags modeltest ./...
go vet -tags integration ./...
go run ./cmd/release -out dist -verify-reproducible
```

## 11. GUI features and usage

The server embeds a thin HTML, CSS, and JavaScript workbench at
`http://127.0.0.1:8080/`. The browser owns presentation state only. Models,
recipes, datasets, operations, sessions, runs, and artifacts remain owned by
the Go server and RepoDB.

The workbench is organized into four sections and eighteen functional tabs:

| Section | Tabs | Primary use |
| --- | --- | --- |
| Inference | Chat, Generate, Runtime, Activity | Run streaming conversations and recipe-driven generation; inspect sessions, throughput, transfers, and serving evidence |
| Datasets | Datasets | Filter registered datasets and inspect bounded typed previews |
| Training | Train, Model Builder, Export, Runs | Train or construct models, export artifacts, control operations, inspect checkpoints, and compare run evidence |
| Workbench | Recipe, Artifacts, Evaluations, Model, Vocabulary, Logit lens, Hidden states, Attention, Tensors | Inspect compiled authority, outputs, evaluation campaigns, model structure, tokens, activations, attention, and tensor statistics |

### Capability-driven forms

Generate, Train, Model Builder, and Export read their available operations and
typed controls from the server. Forms therefore follow active recipes and
configured workspaces instead of embedding a browser-side model catalog.

If a capability is unavailable, the tab reports the missing recipe, dataset,
projector, model-builder, training, evaluation, or export authority. It does
not choose a fallback model or execution policy.

### Common operation lifecycle

Long-running workflows share one lifecycle:

1. Read admitted capabilities and controls from the server.
2. Submit a typed request and receive an operation identity.
3. Poll common status, progress, metrics, and outputs.
4. Cancel through the same operation manager when necessary.
5. Inspect the resulting run, checkpoint, report, trace, media, and lineage
   artifacts.

Runs display outcome, recipe and commit identity, wall time, major phases,
inputs, outputs, and linked artifacts. Training plots include DPO or GRPO
measurements when those objectives are active. A selected run can become an
explicit comparison baseline.

### Chat and generation

Chat supports system prompts, multi-turn history, streaming, stop, reset,
Markdown rendering, token counting, prompt-cache reporting, and response
copying. During execution it reports context use, cached and generated tokens,
prefill and decode throughput, and elapsed time.

Generate builds its controls from active generation recipes and can publish
text, image, audio, or video artifacts when those tasks are configured.

### Runtime and activity

Runtime shows active model sessions, slot state, task, device, context use,
prompt and cached tokens, generated tokens, throughput, and elapsed time.
Activity reads bounded serving observations from RepoDB, including model and
recipe identity, outcome, duration, token counts, and host/device transfer
measurements. Raw request and response bodies are not displayed.

### Datasets, training, and model building

Dataset filtering covers name, family, modality, and language. Bounded previews
show processor-produced roles, modalities, text, encodings, and byte counts
without loading the complete dataset into the browser.

Training resolves active recipes and exposes only their admitted dataset,
checkpoint, optimizer, objective, and output controls. Model Builder runs
corpus-derived construction through the same operation and evidence contracts.
Export publishes compatible durable forms without moving model-family policy
into the UI.

### Evaluations and artifacts

Evaluations selects configured suites, runs campaigns, displays progress and
failures, browses per-model history, and compares metrics with an explicit
baseline. Artifacts supports kind filtering, paging, producer identity, and
inline display of image, audio, and video outputs.

### Model analysis

The workbench includes:

- model architecture and execution dimensions;
- paged vocabulary and token search;
- logit probabilities, selected-token probability, and entropy;
- hidden-state distance, nearest-neighbor, and nonmetric MDS views;
- exact captured Q/K attention replay where the active policy permits it;
- bounded tensor location, scale, L-moment, sparsity, energy-entropy, and
  effective-rank analysis;
- nearest-tensor lookup over comparable catalog measurements.

Tensor similarity is descriptive analysis. It does not infer semantic
interchangeability or authorize component composition.

## 12. Compatibility and benchmark references

Model support, verification, and performance claims are artifact-specific and
change as recipes and evidence are recorded. The README intentionally does not
duplicate the full result ledger.

- [Compatibility matrix](docs/COMPATIBILITY.md) contains model profiles,
  capabilities, verification levels, known gaps, and reproduction commands.
- [Machine-readable compatibility data](compatibility.json) contains checked
  claims, evidence tiers, source identities, and verification commands.
- RepoDB run and evaluation records contain exact artifact, recipe, input,
  environment, output, timing, memory, transfer, and decision provenance.
- [Current plan](docs/plan.json) contains unfinished campaign work and its
  verification commands.
- [RepoDB import contract](docs/REPODB_IMPORT.md) describes validated store
  import behavior.
- [Iteration doctrine](skill.md) describes architecture, porting,
  verification, and development workflow rules.
- [SBOM](SBOM.cdx.json) records dependency and binary provenance.

Cross-system performance comparisons are valid only for the stated artifact,
input, seed, precision, expected output, loading and cache state, measurement
scope, and hardware environment. Consult the retained run records rather than
treating a summary number as a portable guarantee.
