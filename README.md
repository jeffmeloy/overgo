# Overgo

Overgo is an experimental, no-cgo Go system for model inference, training,
evaluation, and composition on consumer NVIDIA hardware.

It uses selected formats, semantics, and kernel behavior from llama.cpp.
Adaptive_new supplies reference implementations and performance baselines.
Overgo implements compiled recipes, runtime execution, Muon training,
verification records, and artifact lifecycle management.

**Current host:** Windows amd64, Go 1.26, NVIDIA CUDA driver, `CGO_ENABLED=0`.
`github.com/dlclark/regexp2/v2` is the sole third-party Go runtime dependency.
The implementation and public interfaces remain experimental.

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
verification level. A catalog entry without verification remains experimental.

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

The CUDA runtime loads installed NVIDIA DLLs directly. A worker locked to one
operating-system thread maintains each driver context. Compiled graphs reuse
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

## Capability status

| Area | Current implementation | Remaining verification or implementation |
| --- | --- | --- |
| Text inference | Dense, MoE, recurrent, hybrid, encoder, encoder-decoder, diffusion-text, and speculative components | Each model artifact requires its own verification. |
| Quantized execution | GGUF parsing and conversion plus native quantized weights and experts selected by recipe | Exact data-type and model-family coverage is generated in `docs/COMPATIBILITY.md`. |
| Serving | Native llama.cpp-style endpoints; OpenAI Chat/Completions/Embeddings/Responses; Anthropic Messages; built-in inference, RepoDB browsing, and model-analysis UI | Protocol behavior has automated contract tests. Model output quality requires recipe-specific tests. |
| Multimodal input | Image, audio, and video projection; size-limited local and allowlisted remote media; mixed-media conversation history | E4B, Gemma/Qwen, RxBrain, and Unlimited OCR have tests using model files. Many catalog entries have only synthetic-fixture tests. |
| Image generation | Typed conditioning, CUDA-resident denoising, and PNG artifact output | Krea has verified 2048-pixel execution. SenseNova has a CUDA-tested 256-pixel text-to-PNG recipe. Full-size and image-edit tests are not complete. |
| Video generation | CUDA-resident Wan denoising and VAE decoding; LiveEdit schedule, checkpoint selection, reference trajectory, and shared text conditioning | Wan has verified full-clip execution. LiveEdit source encoding, resident denoising, production activation, and performance comparison are not complete. |
| Speech, forecast, table, seq2seq | Shared runtime and recipe components. Pocket-TTS verifies waveform output. TimesFM verifies exact forecasts and a held-out Supernova baseline. Needle verifies exact numeric parity, grounded text-to-tool-call JSON, and real GSM8K decoder-boundary training through the common dataset stream, compiled training program, and device Muon. | Needle retains BF16 matrices and measures 66.758-67.363 MiB across matched cold processes versus adaptive_new's 120.918-121.328 MiB. Its final cross-attention gate and output projection train with held-out evidence; Q/K/V, remaining decoder, and encoder gradients remain open. Comparable process peak measurements remain open for the other capabilities. |
| Training | Shared dataset streaming for dense, scratch, and diffusion-image Muon trainers; scratch construction uses shared tensor VJP and resident CUDA/Muon sessions | Frozen-lexical Carbon and the recorded scratch configuration outperform their references. RepoDB dataset selection from the CLI, complete-state resume, and general multimodal objectives are not implemented. |

Known gaps include full-size SenseNova image and edit tests, the remaining
LiveEdit CUDA pipeline, exact full-sequence Unlimited OCR comparison, comparable
peak-memory measurements, complete-state resume, publication and activation of
scratch-built controllers, and real-model Qwen3.5/E4B/Gemma4 training.

## Recorded performance

Snapshot reviewed 2026-08-14. These are recorded single-machine results, not
portable guarantees. The authoritative protocol, artifact identities, quality
checks, and unresolved limitations are in the
[adaptive_new parity report](docs/adaptive_new_parity_report.md).

| Workload | Reference | Overgo | Verdict |
| --- | ---: | ---: | --- |
| Qwen3.5-4B decode | adaptive 16.76 ms/token | 10.96 ms/token; about 2.2 GB more peak memory | Faster token generation; higher peak memory use |
| Qwen3.5-4B image/video | Python reference image and 16-frame video | Exact prompt IDs, MRoPE, and first tokens; CUDA projector tests pass | Matches the tested real inputs; comparable peak-memory measurement is not complete |
| Gemma4 12B with image, audio, and video inputs | adaptive_new provides reference results for image and audio. It accepts video input but does not provide a reference result for video. | With the same model files, the image test generates the same first token. The audio test generates a first token accepted by the reference result. Mixed image-then-audio and audio-then-image inputs preserve their order and generate the expected token IDs, 107 and 108. A two-frame video preserves frame order through text generation. | No end-to-end video output comparison or comparable peak-memory measurement |
| RxBrain VQA | adaptive 18.4-23.1 s / 11.97 GB | Fresh device sessions 9.529-10.273 s; 3.940 GiB approximate device peak | Exact answer; faster execution and lower measured peak memory |
| Unlimited OCR | Native BF16 image/text reference output | Exact 277-token prompt and 200-token output prefix; 273 projected tokens; complete 29-row output differs by one coordinate/text edit; native 35-gram/128-window policy | Real-model OCR comparison passes at the stated tolerance; exact full sequence and comparable peak-memory measurement are not complete |
| Pocket-TTS speech | Adaptive real-model generation fixture | Compiled recipe plus matching latent/EOS/PCM/WAV output; 24 kHz mono; 0.48-0.53 s repeated synthesis; 0.540-0.543 GiB peak heap | Output matches and repeated execution is faster; comparable process peak-memory measurement is not complete |
| Carbon-500M causal Muon | adaptive 6.816 s loop / 11.47 GiB | 5.05-5.13 s / 4.620 GiB; matching loss trajectory | About 25% faster training loop and 60% lower peak memory |
| Corpus-derived scratch causal | adaptive 221.4-231.8 ms / 3.13-3.18 MB peak | 58.0-65.4 ms matching steps / 1.15-1.17 MB combined peak; process, driver, model, PTX/program, first-run, and repeated-run phases measured separately | At least 3.38 times faster and 62% lower peak memory for the recorded configuration |
| Krea 2048 image | Python 97.5-135.4 s; adaptive 158.144 s / 33.47 GB | 63.510 s / 32.732 GB; MAE 0.03910 | Faster execution and lower peak memory within the stated quality tolerance |
| Wan video | Resident Python 463.4 s; adaptive 795.1 s | 370.27 s; stage peaks 7.875/10.330 GB | Faster than both references; a matching repeat run is not complete |
| SenseNova generation core | adaptive reusable generation core 6.88 s | Compiled 256px, two-step recipe: 13.56 s total; generation core first run 3.26-7.37 s, repeated run 3.24-3.32 s / 0.708 GiB; exact PNG `d439b8ce...` | Repeated generation-core execution is faster; the production text-to-image path passes its CUDA test; full-size and edit tests are not complete |
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
verified, activated at the experimental tier, and replayed with grounded
text-to-tool-call JSON. Experimental means the real execution path is proven;
matched process-memory evidence and an approved training objective remain open.

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

The frozen-lexical Carbon configuration is the currently verified production
training configuration. The CLI writes model weights, `config.json`, and
`tokenizer.json`. It does not yet write one atomic checkpoint containing the
optimizer, random-number-generator state, and dataset position required for
complete resume.

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
- produces deterministic weighted order from dataset identity, seed, member,
  epoch, and record identity;
- snapshots `{stream identity, position}` for exact order resume;
- packs by example and byte bounds, forms microbatches, and decodes with bounded
  parallel workers while preserving order and rolling back on failure.

`cmd/train` currently processes one UTF-8 file as an in-memory document. Direct
RepoDB dataset and split selection, and atomic storage of the stream position
with optimizer and random-number-generator state, are not implemented.

### From-scratch model construction and training

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
as a complete CLI workflow. RepoDB publication of the initialized model, run,
and checkpoint; atomic resume; general multimodal objectives; evaluation and
activation of a held-out controller; and descendant improvement are not
implemented.

See the [training plan](docs/training_plan.md) for the exact boundary.

## Commands

Core user commands:

| Command | Purpose |
| --- | --- |
| `cmd/generate` | Text and multimodal generation through an active recipe |
| `cmd/server` | Native, OpenAI-compatible, and Anthropic-compatible HTTP serving |
| `cmd/train` | Dense safetensors Muon training |
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
| `internal/recipe`, `internal/modelrecipe`, `internal/workflowruntime` | Typed capability definitions, lifecycle, and execution |
| `internal/cuda`, `kernels/` | CUDA driver interface, executor, generated interfaces, and manifested CUDA assets |
| `internal/projector`, `internal/latentimage`, `internal/latentvideo` | Multimodal projection and media generation |
| `internal/optimizer`, `internal/densecausal`, `internal/hybridtrain` | Muon and training implementations |
| `internal/trainingprogram`, `internal/scratchmodel` | Compiled training plans and corpus-derived scratch construction and execution |
| `internal/artifact`, `internal/repodb`, `internal/runrecord` | Identity, lineage, decisions, runs, and verification records |
| `internal/server`, `internal/server/webui` | HTTP contracts and embedded thin-client console/workbench |
| `compatibility.json` | Machine-checked feature and model claims |
| `docs/plan.json` | Current work items and their verification commands |

## Reference documentation

- [Compatibility matrix](docs/COMPATIBILITY.md): generated model and feature
  status with verification levels and commands.
- [Adaptive_new parity report](docs/adaptive_new_parity_report.md): capability,
  quality, wall, peak-memory, and remaining-gap assessment.
- [Training plan](docs/training_plan.md): scratch-controller-first Muon design
  and current implementation boundary.
- [RepoDB](docs/REPODB.md): artifact identity, lineage, and store contracts.
- [Merge floor](docs/MERGE_FLOOR_PLAN.md): automation implementation status.
- [Iteration doctrine](skill.md): architecture, verification, porting, and
  development workflow rules.
- [LICENSES.md](LICENSES.md) and [SBOM.cdx.json](SBOM.cdx.json): dependency and
  binary provenance.
