# Overgo

Overgo is a no-cgo Go system for importing, defining, creating,
training, evaluating, serving, and composing models on consumer NVIDIA
hardware. It supports text, image, audio, video, time-series, tabular, speech,
embedding, reranking, and preference-training workflows through one typed
runtime.

It uses selected formats, semantics, and kernel behavior from llama.cpp.
Adaptive_new supplies reference implementations and performance baselines.
Overgo implements compiled recipes, runtime execution, Muon training,
verification records, and artifact lifecycle management.

**Current host:** Windows amd64, Go 1.26, NVIDIA CUDA driver, `CGO_ENABLED=0`.
`github.com/dlclark/regexp2/v2` is the sole third-party Go runtime dependency.
Public interfaces may change while the runtime is under active development.

## Feature overview

| Workflow | Implemented capability |
| --- | --- |
| Model import | GGUF and safetensors inspection, Hugging Face conversion, split/merge, quantization, tensor inventory, and content-hash identity |
| Model definition | Immutable architecture profiles, tensor inventories, model definitions, capability recipes, and compiled execution plans |
| New model creation | Deterministic scratch-model construction, initialized parameter manifests, adapters, component proposals, and derived-model lineage |
| Inference | Dense, MoE, recurrent, hybrid, encoder, encoder-decoder, diffusion-text, embedding, reranking, speculative, and constrained token generation |
| Modalities | Text input/output; image, audio, and video projection; image and video generation; speech synthesis; forecasting; tabular prediction; VQA and OCR paths |
| Training | Shared dataset streams, compiled objectives, Muon optimization, host reference execution, CUDA-resident execution, checkpoints, exact resume, and evaluation records |
| Preference training | Native DPO data preparation, policy/reference scoring, loss and gradients, Muon updates, checkpoint resume, run evidence, and GUI plots |
| Serving | Native llama.cpp-style, OpenAI-compatible, and Anthropic-compatible HTTP APIs with streaming, tools, structured output, media, batching, and caches |
| GUI | Inference, recipe-driven generation, datasets, DPO training, export, operations, runs, artifacts, recipes, and model/tensor/state/attention analysis |
| Evidence | RepoDB identities, lineage, run records, evaluation records, recipe promotion, rollback records, compatibility claims, and reproducible release checks |

Artifact-specific results and verification levels are listed in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## System contract

Overgo uses one Go runtime rather than separate model-specific executors.

- Model and processor metadata comes from GGUF, safetensors, or RepoDB
  artifacts.
- Typed recipes select capabilities, modules, and execution policy.
- Compiled programs seal model topology before execution.
- Shared Go/CUDA operators execute the program.
- RepoDB records identity, lineage, verification, recipe activation, and
  rejected activation attempts.
- Git records chronology; plan and state files contain current work only.

Inference commands do not select device residency from command flags. They
resolve an active recipe tied to the model artifact's content hash from RepoDB.
Activating a candidate recipe requires a successful gate and run for that
recipe. Missing verification is an error; the runtime does not silently use a
different recipe.

The generated [compatibility matrix](docs/COMPATIBILITY.md) is the source of
truth for model and feature support. Status terms have these meanings:

- **Cataloged:** metadata, tensor requirements, and graph policy compile; this
  does not claim every matching checkpoint runs correctly.
- **Implemented:** a checked claim identifies its source code and a test command
  that can fail.
- **Verified:** the named fixture or artifact passed the stated contract,
  reference-output, or device test.
- **Promoted:** a recipe tied to a specific artifact has a successful gate and
  run and may be activated for production use.

Each claim applies only to the named artifact, input, execution lifecycle, and
verification level. A catalog entry without verification makes no
artifact-specific execution claim.

## Architecture

```text
model bytes + artifact/profile metadata
                 |
                 v
        typed capability recipe
                 |
                 v
          compiled program
                 |
                 v
   reusable model/device session
                 |
                 v
       shared Go/CUDA operators
                 |
                 v
 typed output artifact + run and evaluation records
```

Placement chooses an executor; it cannot change modality or select a different
operator implementation. Prompt templates, normalization, token budgets,
schedulers, and codecs come from validated artifact and profile metadata.

Opening a model builds one indexed weight catalog. A compiled requirement schema
validates alternative shapes, storage, and cross-tensor relations. Layer
programs carry indexed bindings; execution does not rediscover family policy.

The CUDA runtime loads installed NVIDIA DLLs directly. Go memory is pinned while
its address crosses a DLL boundary, and every device copy range must belong to a
live driver allocation. A worker locked to one operating-system thread maintains
each driver context. Compiled graphs reuse
validated topology, BLAS selection, memory-arena plans, indexed operands, and
replay state. Graph rewrites compile typed fusion launch descriptors before
execution. CUDA and PTX assets listed in the kernel manifest are the only non-Go
runtime components maintained in this repository.

Preloaded decoding keeps model weights, KV state, recurrent state, and graph
outputs on the GPU. Only values required for token sampling return to Go. Media
generation sessions keep compiled branch graphs, prefix KV, and hidden state on
the GPU across denoising steps, then release host weights and tensor mappings
after upload. Sessions are cached by model artifact, recipe, device, and
execution policy. A separate lock for each cached session allows different
sessions to execute concurrently.

## Recipe schemes

Recipes are executable data, not labels. A versioned recipe names one task,
artifact dependencies, typed modules, host/device placement, data ports,
ordered edges, inputs, and outputs. The compiler rejects unknown modules,
invalid ordering, incompatible data kinds, missing dependencies, and placement
or residency conflicts.

Supported recipe tasks are inference, token generation, embedding, reranking,
projection, training, forecasting, tabular prediction, sequence-to-sequence
generation, speech synthesis, image generation, video generation, and VQA.

### Model recipe scheme

A runnable model has four related records:

1. **Model artifact:** content-hashed model bytes and locations.
2. **Model profile:** immutable architecture and operator policy with fact
   provenance.
3. **Model definition:** exact binding of the model artifact, profile, tensor
   inventory, architecture name, dimensions, and validated tensor facts.
4. **Capability recipe:** ordered modules for one task, including placement,
   session lifetime, residency, processors, projectors, adapters, or other
   model dependencies.

Compilation produces an indexed model plan and, for inference, a decode plan.
The plan identity includes the model, profile, definition, recipe, recipe
version, placement, residency, and runtime. Runtime code consumes this plan;
it does not choose behavior from a model-family switch.

RepoDB may store several candidate recipes for the same artifact and task.
Only an active recipe can run through production entry points. Activation
requires a successful gate and run tied to the exact recipe and model identity.
Changing model bytes, profile facts, tensor inventory, or recipe content creates
a different identity and requires new evidence.

### Dataset recipe scheme

Dataset documents are also immutable, content-addressed records. Four document
types describe data without copying dataset bytes into Git:

| Dataset document | Meaning |
| --- | --- |
| `version` | Named external assets, artifact identities, and record counts |
| `view` | A source dataset plus an immutable selector and/or selected fields |
| `split` | Named partition views over one source dataset |
| `mixture` | Canonically ordered datasets with normalized integer weights |

Group-aware split plans assign all records from one group to the same partition
using a seed and immutable membership records. RepoDB stores document content,
lineage, aliases, and file locations. The training materializer resolves the
documents, selectors, split membership, processors, and assets; validates file
sizes and content; removes exact duplicates; and builds a deterministic stream.
Weighted member order, shuffle order, epoch, seed, and stream position are part
of the resumable data contract.

### Training recipe scheme

A training objective record binds objective type, input/output modalities,
dataset and split, processors, optional projectors or codecs, loss, evaluation,
and evidence. A compiled `TrainingProgram` defines ordered forward, backward,
and Muon update operations plus the exact trainable parameter plan. A
`TrainingRunPlan` adds the model or scratch construction, initial checkpoint,
dataset stream, precision, placement, memory, evaluation, checkpoint, and
promotion policies.

The runtime rejects a resume checkpoint when its program, dataset, split,
stream, processor, projector, codec, or model authority differs. DPO adds a
frozen reference model, chosen/rejected preference batches, and a positive
objective scale to the same plan and checkpoint scheme.

## Model and runtime coverage

The architecture catalog currently contains 136 profiles. Profiles describe
tensor layout and operator policy. Artifact tests and run records are tracked
separately. Representative groups are:

| Runtime group | Representative catalog profiles and capabilities |
| --- | --- |
| Dense causal decoders | Llama, Gemma, Qwen, Mistral, GPT-2/J/NeoX, MPT, Phi, Falcon, Cohere, Granite, OLMo, StableLM, StarCoder |
| Mixture-of-experts | DeepSeek, Qwen MoE, DBRX, Arctic, Grok, Granite MoE, Hunyuan MoE, Exaone MoE, Ernie MoE |
| Recurrent and hybrid | Mamba, Mamba2, RWKV6/7, Jamba, Qwen3Next/Qwen3.5, Kimi Linear, GraniteHybrid, Falcon-H1, LFM2 |
| Encoders and embeddings | BERT, T5 encoder, ModernBERT, NeoBERT, Nomic BERT, Jina BERT, Gemma embedding, Llama embedding |
| Encoder-decoder and structured tasks | T5, Needle sequence-to-sequence, TimesFM forecasting, TabFM tabular prediction, OCR and reranking paths |
| Multimodal language | Gemma3/3n/4, Qwen2-VL/Qwen3-VL, Hunyuan VL, CogVLM, Chameleon, DeepSeek OCR, PaddleOCR |
| Diffusion and discrete generation | LLaDA, LLaDA-MoE, Dream, RND1, SimpleDiffusion, latent and oscillator image/video programs |
| Media systems | Pocket-TTS speech, Krea and SenseNova image generation, Wan and LiveEdit video, Un-0 oscillator image/video |

Use `go run ./cmd/compatibility` or
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for artifact-specific status.

## Capability status

| Area | Current implementation | Remaining verification or implementation |
| --- | --- | --- |
| Text inference | Dense, MoE, recurrent, hybrid, encoder, encoder-decoder, diffusion-text, and speculative components | Artifact-specific verification is recorded in the compatibility matrix. |
| Quantized execution | GGUF parsing and conversion plus native quantized weights and experts selected by recipe | Exact data-type and model-family coverage is generated in `docs/COMPATIBILITY.md`. |
| Serving | Native llama.cpp-style endpoints; OpenAI Chat/Completions/Embeddings/Responses; Anthropic Messages; built-in inference, RepoDB browsing, and model-analysis UI | One contract matrix verifies streaming, structured tools, embeddings, ordered media, response continuation, prompt-cache reuse and context-window editing, cancellation, and unsupported-operation errors. Model output quality requires recipe-specific tests. |
| Multimodal input | Image, audio, and video projection; size-limited local and allowlisted remote media; mixed-media conversation history | E4B, Gemma/Qwen, RxBrain, and Unlimited OCR have tests using model files. Many catalog entries have only synthetic-fixture tests. |
| Image generation | Typed conditioning, CUDA-resident denoising, and PNG artifact output | Krea has verified 2048-pixel execution. SenseNova executes the native 1536x2720, 50-step infographic request through the production recipe and passes a reviewed spatial, contrast, edge, and palette signature. Un-0 publishes an exact retained artifact. SimpleDiffusion uses a geometry-keyed resident CUDA recipe; its real seed-7, two-step, 64x64 generation is 15-18x faster warm than the host in repeated tests, with one decoded color channel differing by one byte from the retained PNG. SenseNova image editing remains blocked by the absent source PNG. |
| Video generation | Typed oscillator, Wan, and LiveEdit recipes; encoded artifact publication; CUDA-resident Wan denoising and VAE encoding/decoding; retained LiveEdit text projection, cumulative attention history, and source-latent reuse | Un-0 publishes six real-artifact frames as a 64x64 GIF byte-identical to adaptive_new. Wan has verified full-clip execution. LiveEdit executes all 30 blocks. Its full 81-frame edit matches adaptive and Python output quality, uses 15.624 GiB peak device memory, takes 81.6-81.9 s cold, and takes 51.0-51.5 s when the same source latent is resident. The production recipe publishes GIF; the performance gate streams MP4. |
| Speech, forecast, table, seq2seq | Shared runtime and recipe components. Pocket-TTS verifies waveform output and trains its real backbone+flow parameter set through compiled Muon on native generated codec latents. TimesFM verifies exact forecasts and a held-out Supernova baseline. Needle verifies exact numeric parity, grounded text-to-tool-call JSON, and real GSM8K training through the common dataset stream, compiled training program, and device Muon. | Needle retains BF16 matrices and measures 66.758-67.363 MiB across matched cold processes versus adaptive_new's 120.918-121.328 MiB. Shared reverse traversal trains both final norms, all eight decoder self/cross-attention pairs, all 12 encoder self-attention blocks, and the tied source/target/output embedding. A fixed 4-train/4-held-out GSM8K gate improves both aggregate losses. Pocket-TTS corpus audio encoding and held-out training evidence remain open. Comparable process peak measurements remain open for the other capabilities. |
| Training | Shared dataset streaming for dense, scratch, seq2seq, speech, and diffusion-image Muon trainers. Every compiled program identifies its objective. RepoDB documents bind objective kind, corpus, split, processors, projectors/codecs, loss, evaluation, and evidence. | Frozen-lexical Carbon and the recorded scratch configuration outperform their references. Qwen3.5-4B recurrent layer 0 and the Gemma E4B layer-0 adapter train from real artifacts. Pocket-TTS latent-sequence and SimpleDiffusion flow-matching trainers use the shared program order. Eight adaptive objective contracts are represented; a missing program binding is refused. A Git-pinned workflow corpus trains three scratch-controller seeds and records held-out evaluations plus an external promotion/rollback decision. A live CUDA measurement compiles a forced-cap controller Tier-1 schedule that reduces physical reservation from 10 MiB to 8 MiB; streamed execution remains open. Real-record evidence approves text-to-text, image-to-text, and audio-to-text; the other 33 single-modality pairs remain refused. Complete model stacks, production FNS/forecast/OCR/image-latent/distillation executors, and checkpoint adoption by every trainer remain open. |
| New model creation | Content-addressed model definitions, deterministic scratch construction, initialized parameters, shared Muon training, evaluation, evidence, and promotion decisions use one model-builder workflow. The workflow is available through `cmd/model-build` and the web workbench. | Import and conversion remain separate CLI workflows. Serving a newly constructed scratch model requires a compatible export target. |
| Preference training and RL | DPO and grouped relative policy optimization (GRPO) use active recipes, shared sequence scoring and VJPs, Muon updates, exact resume, checkpoints, RepoDB traces and decisions, and GUI plots. GRPO rewards name their evaluator evidence. | PPO, online environment rollout collection, reward-model training, and online policy serving are not implemented. |
| Web workbench | Embedded thin client for chat, recipe-driven generation, runtime state, datasets, model building, DPO and GRPO training, export, operations, runs, recipes, artifacts, vocabulary, logits, hidden states, attention, and tensor analysis. | Tabs report unavailable server capabilities directly. Training requires `-training`, RepoDB, an active training recipe, dataset and model locations, and checkpoint storage. Model building requires `-model-builder`. |

Known gaps include the SenseNova image-edit source oracle, LiveEdit cold-request
leadership and recipe-configured MP4 publication, exact full-sequence Unlimited OCR comparison, comparable
peak-memory measurements, checkpoint adoption outside dense training, production activation of
scratch-built controllers, complete real-model Qwen3.5 and E4B/Gemma4 stack training. Un-0
training refuses execution until a recipe supplies real class/image data; the
former synthetic constant-target trainer was deleted.

## Recorded performance

Snapshot reviewed 2026-08-14. These are recorded single-machine results, not
portable guarantees. The authoritative protocols and artifact identities live
in the RepoDB store's committed run and generation records.

| Workload | Reference | Overgo | Verdict |
| --- | ---: | ---: | --- |
| Qwen3.5-4B decode | adaptive 16.76 ms/token | 10.96 ms/token; about 2.2 GB more peak memory | Faster token generation; higher peak memory use |
| Qwen3.5-4B image/video | Python reference image and 16-frame video | Exact prompt IDs, MRoPE, and first tokens; CUDA projector tests pass | Matches the tested real inputs; comparable peak-memory measurement is not complete |
| Gemma4 12B with image, audio, and video inputs | adaptive_new provides reference results for image and audio. It accepts video input but does not provide a reference result for video. | With the same model files, the image test generates the same first token. The audio test generates a first token accepted by the reference result. Mixed image-then-audio and audio-then-image inputs preserve their order and generate the expected token IDs, 107 and 108. A two-frame video preserves frame order through text generation. | No end-to-end video output comparison or comparable peak-memory measurement |
| RxBrain VQA | adaptive 18.4-23.1 s / 11.97 GB | Fresh device sessions 9.529-10.273 s; 3.940 GiB approximate device peak | Exact answer; faster execution and lower measured peak memory |
| Unlimited OCR | Native BF16 image/text reference output | Exact 277-token prompt and 200-token output prefix; 273 projected tokens; complete 29-row output differs by one coordinate/text edit; native 35-gram/128-window policy | Real-model OCR comparison passes at the stated tolerance; exact full sequence and comparable peak-memory measurement are not complete |
| Pocket-TTS speech | Adaptive real-model generation fixture | Compiled recipe plus matching latent/EOS/PCM/WAV output; 24 kHz mono; 0.48-0.53 s repeated synthesis; 0.540-0.543 GiB peak heap | Output matches and repeated execution is faster; comparable process peak-memory measurement is not complete |
| Carbon-500M causal Muon | adaptive 6.816 s loop / 11.47 GiB | 1.510-1.550 s / 4.985 GiB; matching loss trajectory | At least 77.3% faster training loop and 56.5% lower peak memory |
| Qwen2.5-0.5B causal Muon | adaptive 505 ms retained warm step | 469-480 ms retained warm execution; 552-753 ms cold execution; 6.592 GiB peak | At least 5.0% faster retained warm execution; lifecycle phases gated separately |
| Qwen3.5-4B recurrent-layer Muon | adaptive_new has no training oracle | Real GGUF layer 0, 112,885,760 matrix and 38,080 vector parameters; all five token IDs from a recorded prompt; loss 13.185789 to 11.006020 in two resident steps; 1.552 s training loop; 1.775 GiB peak | Overgo-only layer capability; no parity or full-model claim |
| Gemma E4B per-layer adapter Muon | adaptive_new has no E4B adapter-training trajectory | Real GGUF layer 0, 1,313,280 parameters; fingerprinted real text rows and CUDA image/audio tower outputs; losses 14.315110, 16.071185, and 16.074883; 68.989 ms for three Muon updates; exact checkpoint resume | Overgo-only adapter capability; no full-model or cross-repository trajectory claim |
| Corpus-derived scratch causal | adaptive 221.4-231.8 ms / 3.13-3.18 MB peak | 58.0-65.4 ms matching steps / 1.15-1.17 MB combined peak; process, driver, model, PTX/program, first-run, and repeated-run phases measured separately | At least 3.38 times faster and 62% lower peak memory for the recorded configuration |
| Krea 2048 image | Python 97.5-135.4 s; adaptive 158.144 s / 33.47 GB | 63.510 s / 32.732 GB; MAE 0.03910 | Faster execution and lower peak memory within the stated quality tolerance |
| Wan video | Resident Python 463.4 s; adaptive 795.1 s | 370.27 s; stage peaks 7.875/10.330 GB | Faster than both references; a matching repeat run is not complete |
| SenseNova 1536x2720 image | adaptive exact Go replay 206.010 s / 18.162 GiB | Two cold 50-step production runs: 136.603-136.829 s / 17.069 GiB; both reproduce the reviewed PNG and pass its spatial, contrast, edge, and palette signature | 33.6% lower wall and 6.0% lower peak; image editing remains blocked by the absent source PNG |
| Gemma E4B image tower | adaptive 14.56 s focused test | 1.65 s including model load; 0.210-0.215 s with the model already loaded | Selected intermediate values match; comparable peak-memory measurement is not complete |
| Gemma E4B audio tower | adaptive 1.03 s focused test | 0.129 s with the model already loaded | Numerical output matches; comparable load-state and peak-memory measurements are not complete |

Cross-repository comparisons are valid only for the stated artifact, input,
seed, numeric precision, expected output, model loading and cache state, and an
otherwise idle GPU. A host implementation and a GPU implementation are not
described as directly comparable unless these conditions match.

## Automation and verification records

The automation is implemented in Go. Every commit must reference the current
plan step. `cmd/plan -context` reports the Git commit, branch, worktree, role,
current plan step, modified paths, and any unfinished RepoDB gate record. It
does not rank work or assign tasks.

`cmd/gate` validates and commits changes:

- staged paths must match the current plan step;
- structural metrics identify large changed functions and duplicate code for
  review;
- Go imports and files referenced by `go:embed` determine affected tests;
- changed files and claims determine whether kernel-manifest, compatibility,
  SBOM, hard-coded-constant, and CUDA checks run;
- race and model smoke tests remain separate commands;
- skipped tests, missing prerequisites, and fixture results that do not satisfy
  a claim are reported explicitly;
- RepoDB preparation, heartbeat, result, and reconciliation records identify
  the plan, environment, paths, Git commit, and result.

Shared command execution preserves exit status and limits captured output
without shell wrappers. CI, release, and local gates use the same Go test
classification. If RepoDB result recording fails after the Git commit,
`cmd/gate` reports failure and writes a local recovery record. The command
`cmd/gate -watchdog` identifies this condition, and `cmd/gate -reconcile`
records only the previously validated batch.

Independent SQA records identify the developer and reviewer, their separate
clean worktrees, the evaluator revision, candidate commit, finding resolutions,
target commit, and result. A human or external process decides whether to
activate or merge the candidate. Automatic review scheduling, resource locking,
and merging are not implemented.

## Requirements and data roots

Required:

- Windows amd64;
- Go 1.26;
- an NVIDIA CUDA driver visible to the installed DLL loader;
- model artifacts compatible with a compiled Overgo recipe.

FFmpeg is optional for encoded video other than native GIF. Select it with
`-ffmpeg`, `OVERGO_FFMPEG`, `PATH`, or the detected Windows installation.

Model, capability, and verification commands use this data-root resolution
order:

1. `OVERGO_DATA_ROOT`, containing `repodb-store/`, `models/`, `datasets/`, and
   `checkpoints/`;
2. machine-local `local-models.json` in the working directory;
3. those four directories under the working directory.

Example `local-models.json`:

```json
{
  "store": "D:/overgo-data/repodb-store",
  "models": "D:/overgo-data/models",
  "datasets": "D:/overgo-data/datasets",
  "checkpoints": "D:/overgo-data/checkpoints"
}
```

The file is machine-local. Model, dataset, and checkpoint bytes stay outside
Git; RepoDB records their identities and locations.

## Quick start

Verify the checkout and CUDA installation:

```bash
go run ./cmd/cuda-info
go run ./cmd/kernel-manifest
go run ./cmd/compatibility -check
go test ./...
```

`go test ./...` does not require model files. Tests that require model files or
a GPU are separate commands. If a required prerequisite is missing, the test
reports `UNAVAILABLE` and fails instead of passing silently.

Inspect a model:

```bash
go run ./cmd/inspect-gguf -metadata -tensors D:/models/model.gguf
go run ./cmd/model-info D:/models/model.gguf
```

Check its active inference recipe:

```bash
go run ./cmd/recipe status -task inference D:/models/model.gguf
```

Verify a candidate with a real task input. `-input -` reads JSON from standard
input and avoids shell quoting limits:

```bash
echo '{"text":"Weather in San Francisco?","max_tokens":64}' | \
  go run ./cmd/recipe verify -task seq2seq -input - needle
```

Verification requires committed Go source, then prints the generated output and
a recipe-bound gate and run ID. Activation consumes those immutable IDs:

```bash
go run ./cmd/recipe activate \
  -task inference \
  -reason "validated artifact and execution policy" \
  -gate evidence:sha256:<gate-id> \
  -run-id run:sha256:<run-id> \
  D:/models/model.gguf
```

`recipe run` executes only an active recipe. The Needle 26M recipe has been
verified, activated, and replayed with grounded text-to-tool-call JSON. Its
activation records prove the stated execution path; matched process-memory
evidence and an approved training objective remain separate work.

Generate after activation:

```bash
go run ./cmd/generate -n 32 D:/models/model.gguf "Hello"
```

Device residency and quantized execution are specified by the active recipe.
Historical `-preload`, `-native-quant`, and `-native-q8` generation flags are
not part of the current CLI contract.

## Multimodal processing

The generation CLI accepts typed local media through a projector GGUF:

```bash
go run ./cmd/generate \
  -mmproj D:/models/mmproj.gguf -mmproj-cuda \
  -image D:/inputs/image.png -n 32 \
  D:/models/model.gguf "Describe this image."

go run ./cmd/generate \
  -mmproj D:/models/mmproj.gguf -mmproj-cuda \
  -video D:/inputs/clip.mp4 -video-fps 2 -video-max-frames 32 -n 32 \
  D:/models/model.gguf "Describe this video."

go run ./cmd/generate \
  -mmproj D:/models/gemma4-mmproj.gguf -mmproj-cuda \
  -audio D:/inputs/audio.wav -n 32 \
  D:/models/gemma4.gguf "Transcribe or describe this audio."
```

Model metadata selects supported projector behavior. The compatibility matrix
states the verification level. Image workflows produce typed PNG artifacts.
Runtime output checks dimensions and finite numeric values. CUDA verification
tests, rather than production execution, apply image-quality thresholds so that
valid flat or low-contrast images are accepted.

Remote media is disabled by the shipped `media_policy.json`. Enabling it
requires explicit scheme, host, port, network, redirect, MIME, concurrency,
timeout, and byte policies. Responses file IDs are separately controlled by
`resource_policy.json`.

## HTTP server

Start the server after recipe activation:

```bash
go run ./cmd/server \
  -listen 127.0.0.1:8080 \
  D:/models/model.gguf
```

Add `-mmproj <projector.gguf> -mmproj-cuda` for supported image/audio/video
requests. `-max-concurrent N` enables bounded continuous batching when the
active recipe supplies device-resident execution.

Supported protocol endpoints:

- OpenAI-compatible completions, chat, embeddings, Responses, rerank, input
  token counts, streaming, JSON Schema/GBNF, and function tools;
- Anthropic-compatible `/v1/messages` and token counting;
- llama.cpp-style completion, infill, embedding, tokenize/detokenize,
  apply-template, slots, LoRA adapters, and properties;
- public health, metrics, and model discovery;
- a built-in web UI on otherwise unmatched GET routes. It includes chat,
  read-only dataset/run browsing, model/vocabulary/logit/hidden-state analysis,
  bounded distribution-free tensor characterization and rank-based nearest-shape
  search, and exact host-replayed attention heatmaps for plain-causal policies
  without sinks, windows, softcap, or ALiBi.

Set `OVERGO_API_KEY` or `-api-key-file` to protect generation and endpoints that
run or modify models. Health, metrics, and model discovery remain public.
Request sizes, media geometry, generation length, batching, caches, and stored
response history have configured limits.

`TestAdaptiveServingContractMatrix` is the server compatibility gate. It covers
native, OpenAI, and Anthropic streaming; structured tool calls; exact token and
text embedding inputs; ordered image, audio, and video projection; stored
response continuation; prompt-cache reuse plus `n_keep`/`n_discard` context
editing; client and timeout cancellation; and explicit refusal when an
unconfigured projector or tool executor is required.

## Web workbench

The server embeds a thin HTML, CSS, and JavaScript client at
`http://127.0.0.1:8080/`. The browser holds presentation state only. Models,
recipes, datasets, operations, runs, and artifacts remain server-owned.

The workbench is organized into four sections:

| Section | Functions |
| --- | --- |
| Inference | Multi-turn chat, recipe-driven generation controls, token streaming, cancellation, and live runtime/cache/session status |
| Datasets | RepoDB dataset browsing, version and lineage detail, bounded record preview, and processor-aware values |
| Training | Recipe-driven DPO controls, resumable checkpoint selection, operation progress, cancellation, export, run history, and checkpoint comparison |
| Workbench | Active recipe inspection; artifact gallery; model and tensor inventory; vocabulary; logit lens; hidden-state layout; attention heatmaps; tensor statistics and similar-shape search |

Generation, training, and export forms are built from the server's typed
capability declarations. The client does not contain model-family forms or
execution switches. Each operation reports state, progress, metrics, run
identity, output artifacts, and failure information through common operation
endpoints.

Start the server with training controls enabled:

```bash
go run ./cmd/server \
  -listen 127.0.0.1:8080 \
  -training \
  D:/models/model.gguf
```

Training mode resolves the active training recipe from RepoDB and requires the
configured dataset and checkpoint roots. The current native GUI training
workspace runs DPO and exposes dataset, resume checkpoint, output name, steps,
learning rate, momentum, and DPO scale. The Runs view plots DPO loss, policy and
reference margins, relative margin, gradient L2, update L2, learning rate, and
chosen/rejected token counts. It links each plot to its run, trace, and
checkpoint artifacts and can compare the final observations from two
checkpoints.

## Training

Muon is the only optimizer used by production training. Matrix, vector, and
scalar groups share one compiled parameter plan; SGD and sign-update fallbacks
were deleted. Host code provides the numerical reference. CUDA Newton-Schulz
and resident updates provide the verified GPU implementation.

Train a dense safetensors causal model:

```bash
go run ./cmd/train \
  -model D:/models/carbon \
  -dataset D:/datasets/train.txt \
  -out D:/checkpoints/carbon-run \
  -steps 4 -freeze-lexical
```

`-freeze-lexical` requires resident CUDA training. `-host` forces the host
reference and cannot be combined with it. A non-positive `-lr` derives
`n_params^-1/2`.

The frozen-lexical Carbon and Qwen2.5-0.5B configurations are verified
production training configurations. The CLI atomically publishes a new directory and
refuses an existing target. `checkpoint.json` binds weights, Muon momentum and
step, data and augmentation RNG counters, dataset stream position, processor,
projector and codec identities, compiled run/program identities, and RepoDB
lineage parents. Resume into another new checkpoint with:

```bash
go run ./cmd/train \
  -resume D:/checkpoints/carbon-run \
  -dataset D:/datasets/train.txt \
  -out D:/checkpoints/carbon-run-2 \
  -steps 4 -freeze-lexical
```

Qwen3.5-4B recurrent layer 0 also trains from the pinned real GGUF through an
ordered `TrainingProgram`. The evidence uses all five token IDs from the recorded
"The capital of France is" prompt, trains 112,885,760 matrix and 38,080 vector
parameters, and lowers the two-step loss from 13.185789 to 11.006020. This proves
one real recurrent layer, not complete 32-layer training. adaptive_new commit
`214950b3b` provides no Qwen3.5 training oracle, so this is an Overgo-only
capability rather than a parity result.

Gemma E4B layer 0 trains its artifact-declared per-layer residual adapter through
the same compiled program and Muon owner. The acceptance run fingerprints the
14.94 GB language GGUF, 956 MB projector, and adaptive vision/audio inputs;
projects real image and audio representations; and applies text, image, and
audio updates to 1,313,280 adapter parameters. The three update losses are
14.315110, 16.071185, and 16.074883 with 68.989 ms combined update wall. An
interrupted run publishes the shared checkpoint and resumes to bit-identical
weights and Muon state. The model plan also verifies the native sliding, full,
and shared-KV schedule. This is selected-adapter training, not complete
42-layer training. The artifact declares no AltUp or Laurel components.

### Preference optimization and RL

Overgo implements DPO and GRPO as native training objectives. Both use the
same dataset, recipe, execution, optimizer, checkpoint, resume, evidence, and
GUI contracts as other training workflows.

The DPO path performs these steps:

1. Read chosen and rejected responses from the compiled preference dataset.
2. Verify their common prompt prefix and derive completion masks.
3. Score both responses with the trainable policy and frozen reference model.
4. Compute the relative policy/reference margin, DPO loss, and score gradients.
5. Run the model VJP and update policy parameters through Muon.
6. Store checkpoints, traces, run records, and recipe identities for exact
   resume and comparison.

GRPO reads candidate groups with prompt, completion, reward, and evaluator
evidence fields. It centers and RMS-normalizes rewards within each group,
scores each completion, accumulates score VJPs, and applies the shared Muon
update. Equal-reward groups produce zero objective gradients without an
artificial threshold.

Run either objective from `cmd/train` with a positive `-objective-scale`, or
use the Training section of the web workbench. DPO also requires `-reference`.
The active recipe selects the objective. PPO, online rollout collection,
reward-model training, and online policy serving are not implemented.

### Dataset processing

`internal/trainingdata` provides shared dataset processing for dense, scratch,
and diffusion-image training:

- resolves RepoDB dataset versions, views, mixtures, asset locations, and
  immutable split memberships;
- builds line-offset indexes, keeps one cached open file per artifact, and reads
  record bytes only when needed;
- applies exact content deduplication before sampling;
- uses the processor profiles and typed input and target modalities specified by
  a compiled `TrainingRunPlan`;
- decodes normalized image and rate-bearing audio values through shared typed
  processors;
- produces deterministic weighted order from dataset identity, seed, member,
  epoch, and record identity;
- snapshots `{stream identity, position}` for exact order resume;
- packs by example and byte bounds, forms microbatches, and decodes with bounded
  parallel workers while preserving order and rolling back on failure.

`cmd/train` currently processes one UTF-8 file as an in-memory document. Direct
RepoDB dataset and split selection, and atomic storage of the stream position
with optimizer and random-number-generator state, are not implemented.

`internal/trainingprogram` stores each approved training objective as a RepoDB
profile. The profile binds objective kind, one ordered input/output modality
pair, real dataset and split, processor, optional projector and codec, loss,
native evaluation metric, and evidence. Every `TrainingProgram` carries the
same objective kind; repository-aware run-plan compilation rejects a mismatch.
The contract matrix covers adaptive FNS, latent L2, latent-sequence L2,
forecast, OCR token prediction, flow matching, image-latent, and logit
distillation semantics. Missing shared programs compile to refused rows.

Pocket-TTS latent-sequence training and SimpleDiffusion flow-matching training
use the common forward/backward/Muon program order. This is execution evidence
for those two objectives, not for all eight contracts. FNS, forecast, OCR,
image-latent, and distillation still require production model bindings and
model-quality evidence. The modality matrix remains separate: only
text-to-text, image-to-text, and audio-to-text have approved real-record E4B
evidence; the other 33 pairs remain refused.

### New model creation and scratch training

Overgo supports three model-creation routes:

1. Import or convert checkpoint weights, derive their tensor inventory, bind an
   architecture profile, and compile task recipes.
2. Construct a model deterministically from a dataset-derived topology,
   parameter manifest, initialization profile, and named random streams.
3. Create a derived model from adapters, grafts, or component proposals while
   retaining parent, evidence, and construction lineage.

The import and conversion route is available through commands. Scratch and
derived construction use package APIs and recorded verification workflows.

`internal/scratchmodel` implements adaptive_new's corpus-derived causal
controller through Overgo's training interfaces. A versioned derivation profile
and immutable corpus metadata deterministically produce train, validation, and
test membership; a rune tokenizer; context length and topology; learning and
initialization settings; a parameter manifest; tied-weight rules; and one flat
initialized parameter allocation. Content hashes identify the dataset, split,
profiles, tokenizer, manifest, initialized model, recipe, and named
random-number streams.

Construction compiles one ordered `TrainingProgram` and shared Muon parameter
plan. Production execution uses the shared tensor forward/VJP and optimizer
implementations. The adaptive pointer-autodiff implementation is used only in
tests. Host and CUDA-resident trainers consume the same dataset stream, keep
weights and momentum on the GPU, and match the recorded adaptive_new reference
loss sequence. For the recorded small configuration, Overgo measured
58.0-65.4 ms versus adaptive_new's 221.4-231.8 ms, and 1.15-1.17 MB versus
3.13-3.18 MB combined peak memory.

Scratch construction is implemented as an internal package but is not exposed
as a complete CLI workflow. `internal/controllertrain` compiles typed
text/image/audio/video component and workflow actions from Git-pinned records
into immutable RepoDB corpus and external-holdout artifacts. Its CUDA gate trains
three scratch seeds through `scratchmodel.ResidentTrainer`, publishes run and
evaluation lineage, and keeps promotion plus rollback in an external evidence
document. The current eight-case holdout improves from 12.5% action accuracy to
37.5-50.0% and from 12.5-25.0% modality accuracy to 37.5-50.0%; all three
held-out causal losses improve. This proves the lifecycle, not a production-ready
controller. Fractale, Carbon, and Qwen2.5 comparisons are explicitly refused
until a compatible controller scorer is available.

Recursive trials have a separate authority boundary. `trainingprogram` compiles
non-authorizing corpus, recipe, evaluator, and component-composition proposals.
`runrecord` requires a separate admission authority, distinct development and
promotion splits, an independent evaluator and decider, a bound evaluation run,
and rollback to the parent. The decision aggregate names the child, parent model,
data, splits, recipe, code, evaluator, run, evaluation, and authorities. The
child retains acyclic construction lineage to the parent and governing facts.
No descendant improvement has been demonstrated yet. CLI activation, exact
checkpoint/resume, and a larger repository-task corpus remain open.

See [docs/plan.json](docs/plan.json) for the open campaign rows.

## Commands

Core user commands:

| Command | Purpose |
| --- | --- |
| `cmd/generate` | Text and multimodal generation through an active recipe |
| `cmd/server` | Native, OpenAI-compatible, and Anthropic-compatible HTTP serving |
| `cmd/train` | Dense safetensors Muon and DPO training with checkpoint resume |
| `cmd/recipe` | Activate, run, and inspect capability recipes |
| `cmd/benchmark` | JSON elapsed-time, throughput, memory, launch, synchronization, and transfer metrics |
| `cmd/embedding` | Encoder embeddings with selectable pooling/normalization |
| `cmd/rerank` | Qwen3/Qwen3-VL pair scoring |
| `cmd/diffusion` | Dream, LLaDA, LLaDA-MoE, and RND1 diffusion-text generation |
| `cmd/latentvideo-run` | CUDA-resident Wan denoising, causal VAE decoding, and full-clip verification |
| `cmd/perplexity` | Next-token or disjoint-window perplexity |

Model and format tools:

| Command | Purpose |
| --- | --- |
| `cmd/inspect-gguf`, `cmd/inspect-safetensors` | Metadata, tensor, and runtime-catalog inspection |
| `cmd/hf-gguf-convert`, `cmd/gemma4-gguf-convert` | Streaming Hugging Face conversion |
| `cmd/gguf-merge`, `cmd/gguf-split` | Validated split-model conversion |
| `cmd/gguf-quantize` | GGUF quantization through manifest-specified encoders |
| `cmd/tokenize`, `cmd/json-schema-grammar` | Token and constrained-generation tooling |
| `cmd/cuda-info`, `cmd/cuda-smoke` | Driver/device inspection and smoke execution |
| `cmd/compatibility` | Check claims or refresh verification record identities and the generated matrix |
| `cmd/repodb-query` | Query typed artifacts, runs, evaluations, findings, and decisions |

Run `go run ./cmd/<name> -h` for standard command flags. `cmd/recipe` uses
`activate`, `run`, and `status` before its options; for example,
`go run ./cmd/recipe status -h`.

## Development and gates

`docs/plan.json` contains only unfinished work. Every commit must name its
current plan step and pass through `cmd/gate`. Direct commits that bypass the
gate are rejected.

```bash
go run ./cmd/plan -next
go run ./cmd/plan -verify
go run ./cmd/gate \
  -message-file commit-message.txt \
  -paths internal/foo/bar.go,internal/foo/bar_test.go \
  -plan item-id/step-id
```

The gate derives affected tests from Go imports and files referenced by
`go:embed`. Depending on the changed files and claims, it checks formatting,
vet, builds, the kernel manifest, the SBOM, compatibility claims, hard-coded
constants, and CUDA results. Every result lists the checks that ran and those
that did not. A required result reported as `UNAVAILABLE` fails the claim. CI,
release, and local gates use the same test classification through
`go run ./cmd/test-lane ./...`. Tests excluded by short mode are listed and are
not counted as passing.

Before creating a Git commit, the gate records the planned commit in RepoDB. It
then adds the gate result to that record. If RepoDB cannot record the result
after the Git commit, the gate fails and writes a deterministic local recovery
record:

```bash
go run ./cmd/gate -watchdog
go run ./cmd/gate -reconcile
```

`-watchdog` reports the gate state as `running`, `stale`, `finalized`,
`record_debt`, or `absent`. `-reconcile` records only the previously validated
batch; it never creates another Git commit.

Additional checks:

```bash
go run ./cmd/device-lane
go run ./cmd/smoke-lane
go run ./cmd/race-lane
go run ./cmd/release -out dist -verify-reproducible
```

The runtime remains no-cgo. `cmd/race-lane` may enable cgo only for Go's
test-only race instrumentation; CUDA race/synchronization checks use
`compute-sanitizer`.

`cmd/guard` protects shell execution. Automation cannot delete, move, or change
permissions for model, dataset, or checkpoint stores.

## Repository map

| Path | Purpose |
| --- | --- |
| `cmd/` | Thin command entry points |
| `internal/model`, `internal/inference` | Compiled model programs, runners, cache/session behavior |
| `internal/modelartifact`, `internal/tensorstats` | Bounded model inventories; L-moment, energy, rank-neighbor, and small-matrix spectral characterization |
| `internal/recipe`, `internal/modelrecipe`, `internal/workflowrecipe`, `internal/workflowruntime` | Typed capability definitions, compilation, lifecycle, and execution |
| `internal/cuda`, `kernels/` | CUDA driver interface, executor, generated interfaces, and manifested CUDA assets |
| `internal/projector`, `internal/latentimage`, `internal/latentvideo` | Multimodal projection and media generation |
| `internal/optimizer`, `internal/densecausal`, `internal/hybridtrain`, `internal/adaptertrain` | Muon and training implementations |
| `internal/dataset`, `internal/trainingdata` | Immutable dataset recipes, splits, mixtures, materialization, and deterministic streams |
| `internal/trainingprogram`, `internal/trainingworkflow` | Compiled objectives, run plans, DPO execution, checkpointing, and resume |
| `internal/scratchmodel`, `internal/controllertrain` | Corpus-derived model construction and repository-workflow training |
| `internal/artifact`, `internal/repodb`, `internal/runrecord` | Identity, lineage, decisions, runs, and verification records |
| `internal/server`, `internal/server/webui` | HTTP contracts and embedded thin-client console/workbench |
| `compatibility.json` | Machine-checked feature and model claims |
| `docs/plan.json` | Current work items and their verification commands |

## Reference documentation

- [Compatibility matrix](docs/COMPATIBILITY.md): generated model and feature
  status with verification levels and commands.
- [Plan](docs/plan.json): the single dispatching authority for all open
  campaign work.
- [Iteration doctrine](skill.md): architecture, verification, porting, and
  development workflow rules.
- [RepoDB import](docs/REPODB_IMPORT.md): store import contract.
- [SBOM.cdx.json](SBOM.cdx.json): dependency and binary provenance.
