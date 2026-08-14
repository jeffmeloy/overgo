# Overgo

Overgo is an experimental, no-cgo Go system for model inference, training,
evaluation, and composition on consumer NVIDIA hardware.

It inherits pinned formats, semantics, and kernel behavior from llama.cpp.
Adaptive_new supplies capability oracles and performance baselines. Overgo owns
the compiled recipes, runtime, Muon training path, evidence, and artifact
lifecycle.

**Current host:** Windows amd64, Go 1.26, NVIDIA CUDA driver, `CGO_ENABLED=0`.
`github.com/dlclark/regexp2/v2` is the sole third-party Go runtime dependency.
The implementation and public interfaces remain experimental.

## System contract

Overgo is one Go-owned runtime, not a family of model-specific executors.

- Model and processor facts come from GGUF, safetensors, or RepoDB artifacts.
- Typed recipes select capabilities, modules, and execution policy.
- Compiled programs seal model topology before execution.
- Neutral Go/CUDA operators execute the program.
- RepoDB records identity, lineage, verification, promotion, and refusal.
- Git records chronology; plan and state files contain current work only.

Inference commands do not infer residency from command flags. They resolve an
active, identity-bound recipe from RepoDB. Candidate activation requires a
successful recipe-bound gate and run; missing evidence is an error, never a
fallback.

The generated [compatibility matrix](docs/COMPATIBILITY.md) is the model and
feature authority. Status terms are strict:

- **Cataloged:** metadata, tensor requirements, and graph policy compile; this
  does not claim every matching checkpoint runs correctly.
- **Implemented:** a checked claim names live source and a failable command.
- **Verified:** the named fixture/artifact passed its stated contract, fixture,
  pinned-oracle, or device evidence.
- **Promoted:** an identity-bound recipe has a successful gate/run pair.

Claims do not generalize beyond their named artifact, input, lifecycle, and
evidence tier. Unverified catalog rows remain experimental.

## Architecture

```text
model bytes + artifact/profile facts
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
      neutral Go/CUDA operators
                 |
                 v
 typed output artifact + run/evaluation evidence
```

Placement chooses an executor; it cannot change modality or select a different
operator implementation. Prompt templates, normalization, token budgets,
schedulers, and codecs come from validated artifact/profile facts.

Model opening builds one indexed weight catalog. A compiled requirement schema
validates alternative shapes, storage, and cross-tensor relations. Layer
programs carry indexed bindings; execution does not rediscover family policy.

The CUDA runtime loads installed NVIDIA DLLs directly. A locked-thread worker
owns each driver context. Compiled graphs reuse validated topology, BLAS
selection, arena plans, indexed operands, and replay frames. Graph rewrites
compile typed fusion launch descriptors before execution. Manifest-pinned
CUDA/PTX assets are the only project-owned non-Go runtime components.

Preloaded decoding retains model weights, KV state, recurrent state, and graph
outputs on device; only sampling boundaries cross to Go. Media generation
sessions retain compiled branch graphs, prefix KV, and hidden state across
denoise steps, then release host weights and binding maps after device upload.
Capability sessions are cached by model artifact, recipe, device, and execution
policy. Per-entry leases allow unrelated keys to execute concurrently.

## Capability status

| Area | Current implementation | Promotion boundary |
| --- | --- | --- |
| Text inference | Dense, MoE, recurrent, hybrid, encoder, encoder-decoder, diffusion-text, and speculative components | Real-artifact evidence remains model-specific |
| Quantized execution | GGUF parsing/conversion plus native quantized weights and experts through recipe-selected residency | Exact type/family coverage is generated in `docs/COMPATIBILITY.md` |
| Serving | Native llama.cpp-style routes; OpenAI Chat/Completions/Embeddings/Responses; Anthropic Messages; built-in inference, RepoDB browse, and model-analysis UI | Protocol features are contract-tested; model quality remains recipe-specific |
| Multimodal input | Typed image, audio, and video projection; bounded local/allowlisted remote media; mixed-media history | E4B, Gemma/Qwen, RxBrain, and Unlimited OCR have real evidence; many catalog rows remain fixture-only |
| Image generation | Typed conditioning, resident CUDA denoise, PNG artifact publication | Krea has retained 2048 evidence; SenseNova has a device-gated text request-to-PNG recipe at the pinned 256px case; full-size and edit evidence remain open |
| Video generation | Resident Wan denoise and CUDA VAE decode; LiveEdit schedule, checkpoint binding, neutral trajectory, and shared text conditioning | Wan has full-clip evidence; LiveEdit source encode, retained denoise integration, production activation, and output leadership remain open |
| Speech, forecast, table, seq2seq | Shared runtime/recipe components exist | Pocket-TTS has real recipe, latent/EOS, PCM/WAV, channel/rate, wall, and heap evidence; other promotions vary |
| Training | Muon-only dense/hybrid primitives plus corpus-derived scratch construction through shared tensor VJP and resident CUDA/Muon sessions | Frozen-lexical Carbon and the pinned scratch profile lead; complete-state resume and universal objectives remain open |

Explicit gaps include full-size SenseNova image/edit evidence, the remaining
LiveEdit device pipeline, Unlimited OCR exact full-sequence numerics and matched peak
evidence, complete-state resume, scratch publication/controller promotion, and real-model Qwen3.5/E4B/Gemma4 training.

## Recorded performance

Snapshot reviewed 2026-08-14. These are retained single-machine results, not
portable guarantees. The authoritative protocol, artifact identities, quality
checks, and open caveats are in the
[adaptive_new parity report](docs/adaptive_new_parity_report.md).

| Workload | Reference | Overgo | Verdict |
| --- | ---: | ---: | --- |
| Qwen3.5-4B decode | adaptive 16.76 ms/token | 10.96 ms/token; about 2.2 GB more peak | Wall lead, memory loss |
| Qwen3.5-4B image/video | Python image and 16-frame video goldens | Exact prompt IDs, MRoPE, and first tokens; CUDA projector probes pass | Real input parity; peak open |
| Gemma4 12B image/audio/video | Adaptive image/audio oracles; declared video route | Image exact token, audio top-set token, ordered mixed image/audio tokens 107/108, and ordered real-frame video through language | Video output oracle and peaks open |
| RxBrain VQA | adaptive 18.4-23.1 s / 11.97 GB | Fresh device sessions 9.529-10.273 s; 3.940 GiB coarse device peak | Exact answer; wall and peak lead |
| Unlimited OCR | Native BF16 image/text golden | Exact 277-token prompt and 200-token output prefix; 273 projected tokens; complete 29-row output within one coordinate/text edit; native 35-gram/128-window policy | Real production OCR parity; exact full sequence and peak open |
| Pocket-TTS speech | Adaptive real-model generation fixture | Compiled recipe plus full latent/EOS/PCM/WAV oracle; 24 kHz mono; 0.48-0.53 s warm matched synthesis; 0.540-0.543 GiB peak heap | Output parity and warm-wall lead; matched process peak open |
| Carbon-500M causal Muon | adaptive 6.816 s loop / 11.47 GiB | 5.05-5.13 s / 4.620 GiB; matched trajectory | About 25% loop-wall and 60% peak lead |
| Corpus-derived scratch causal | adaptive 221.4-231.8 ms / 3.13-3.18 MB peak | 58.0-65.4 ms matched steps / 1.15-1.17 MB combined peak; process, driver, model, PTX/program, cold, and warm phases separately gated | At least 3.38x wall and 62% peak lead on pinned profile |
| Krea 2048 image | Python 97.5-135.4 s; adaptive 158.144 s / 33.47 GB | 63.510 s / 32.732 GB; MAE 0.03910 | Bounded-quality wall/peak lead |
| Wan video | retained Python 463.4 s; adaptive 795.1 s | 370.27 s; stage peaks 7.875/10.330 GB | Lead vs retained references; matched rerun open |
| SenseNova generation core | adaptive reusable body 6.88 s | compiled 256px/2-step recipe 13.56 s total; body cold 3.26-7.37 s, warm 3.24-3.32 s / 0.708 GiB; exact PNG `d439b8ce...` | Warm-body lead; production text-to-image route gated; full-size/edit evidence open |
| Gemma E4B image tower | adaptive 14.56 s focused test | 1.65 s load+run; 0.210-0.215 s resident body | Sampled-stage parity; peak open |
| Gemma E4B audio tower | adaptive 1.03 s focused test | 0.129 s resident body | Numerical parity; matched lifecycle/peak open |

Cross-repo comparisons are valid only for the stated artifact, input, seed,
precision, output contract, lifecycle, and uncontended device. A faster host-vs-
device comparison is not labeled a like-for-like runtime win.

## Requirements and data roots

Required:

- Windows amd64;
- Go 1.26;
- an NVIDIA CUDA driver visible to the installed DLL loader;
- model artifacts compatible with a compiled Overgo recipe.

FFmpeg is optional for encoded video other than native GIF. Select it with
`-ffmpeg`, `OVERGO_FFMPEG`, `PATH`, or the detected Windows installation.

Model, capability, and evidence commands share one data-root resolution order:

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

`go test ./...` is model-free. Real-artifact/device lanes are explicit; missing
required prerequisites are UNAVAILABLE and fail rather than silently passing.

Inspect a model:

```bash
go run ./cmd/inspect-gguf -metadata -tensors D:/models/model.gguf
go run ./cmd/model-info D:/models/model.gguf
```

Check its active inference recipe:

```bash
go run ./cmd/recipe status -task inference D:/models/model.gguf
```

Activation consumes an existing successful verifier gate/run pair:

```bash
go run ./cmd/recipe activate \
  -task inference \
  -reason "validated artifact and execution policy" \
  -gate evidence:sha256:<gate-id> \
  -run-id run:sha256:<run-id> \
  D:/models/model.gguf
```

Generate after activation:

```bash
go run ./cmd/generate -n 32 D:/models/model.gguf "Hello"
```

Residency and quantized execution come from the active recipe. Historical
`-preload`, `-native-quant`, and `-native-q8` generation flags are not part of
the current CLI contract.

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

Supported projector behavior is metadata-selected and evidence-tiered in the
compatibility matrix. Image workflows publish typed PNG artifacts. Runtime
publication validates dimensions and finiteness; image-quality thresholds
belong to evidence gates, so valid flat or low-contrast images are not rejected
by production execution.

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

Primary protocol surfaces:

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

Set `OVERGO_API_KEY` or `-api-key-file` to protect generation and model-action
routes. Health, metrics, and model discovery remain public. Request sizes,
media geometry, generation length, batching, caches, and retained response
history are bounded.

## Training

Muon is the sole production optimizer authority. Matrix, vector, and scalar
groups share one compiled parameter plan; SGD and sign-update fallbacks were
deleted. Host code is the numerical reference. CUDA Newton-Schulz and resident
updates provide the promoted device path.

Train a dense safetensors causal model:

```bash
go run ./cmd/train \
  -model D:/models/carbon \
  -text D:/datasets/train.txt \
  -out D:/checkpoints/carbon-run \
  -steps 4 -freeze-lexical
```

`-freeze-lexical` requires resident CUDA training. `-host` forces the host
reference and cannot be combined with it. A non-positive `-lr` derives
`n_params^-1/2`.

Current production evidence is the frozen-lexical Carbon route. The CLI writes
model weights plus `config.json` and `tokenizer.json`; it is not yet an atomic,
complete-state resume containing optimizer, RNG, and data cursor. Internal
scratch construction compiles corpus-derived topology, a flat initialized slab,
shared tensor forward/VJP, and host/resident Muon programs with matched evidence.
`cmd/train` does not yet publish that construction as a complete model artifact
or expose scratch construction as its production CLI path.
See the [training plan](docs/training_plan.md) for the exact boundary.

## Commands

Core user commands:

| Command | Purpose |
| --- | --- |
| `cmd/generate` | Text and multimodal generation through an active recipe |
| `cmd/server` | Native, OpenAI-compatible, and Anthropic-compatible HTTP serving |
| `cmd/train` | Dense safetensors Muon training |
| `cmd/recipe` | Activate, run, and inspect capability recipes |
| `cmd/benchmark` | JSON wall, throughput, memory, launch, sync, and transfer metrics |
| `cmd/embedding` | Encoder embeddings with selectable pooling/normalization |
| `cmd/rerank` | Qwen3/Qwen3-VL pair scoring |
| `cmd/diffusion` | Dream, LLaDA, LLaDA-MoE, and RND1 diffusion-text generation |
| `cmd/latentvideo-run` | Resident Wan denoise, causal VAE decode, and full-clip evidence |
| `cmd/perplexity` | Next-token or disjoint-window perplexity |

Model and format tools:

| Command | Purpose |
| --- | --- |
| `cmd/inspect-gguf`, `cmd/inspect-safetensors` | Metadata, tensor, and runtime-catalog inspection |
| `cmd/hf-gguf-convert`, `cmd/gemma4-gguf-convert` | Streaming Hugging Face conversion |
| `cmd/gguf-merge`, `cmd/gguf-split` | Validated split-model conversion |
| `cmd/gguf-quantize` | GGUF quantization through pinned encoders |
| `cmd/tokenize`, `cmd/json-schema-grammar` | Token and constrained-generation tooling |
| `cmd/cuda-info`, `cmd/cuda-smoke` | Driver/device inspection and smoke execution |
| `cmd/compatibility` | Check claims or refresh canonical evidence identities and the generated matrix |
| `cmd/repodb-query` | Query typed artifacts, runs, evaluations, findings, and decisions |

Run `go run ./cmd/<name> -h` for standard command flags. `cmd/recipe` uses
`activate`, `run`, and `status` before its options; for example,
`go run ./cmd/recipe status -h`.

## Development and gates

`docs/plan.json` contains open work only. Every commit is bound to its current
step and goes through `cmd/gate`; raw campaign commits are refused.

```bash
go run ./cmd/plan -next
go run ./cmd/plan -verify
go run ./cmd/gate \
  -message-file commit-message.txt \
  -paths internal/foo/bar.go,internal/foo/bar_test.go \
  -plan item-id/step-id
```

The gate derives affected tests from the import graph and compiler-resolved
`go:embed` ownership, then conditionally checks formatting, vet, build, kernel
manifest, SBOM, compatibility claims, magic closures, and device evidence.
Every result states what ran and what did not. UNAVAILABLE evidence fails a
claim that requires it. CI and release use the same Go-owned classifier through
`go run ./cmd/test-lane ./...`; classified short exclusions remain visible and
are not credited as passing tests.

Before Git can advance, the gate commits a typed preparation to RepoDB. It then
finalizes that lifecycle with the gate result. A post-commit RepoDB failure is a
failing gate with a deterministic local reconciliation payload:

```bash
go run ./cmd/gate -watchdog
go run ./cmd/gate -reconcile
```

`-watchdog` classifies the Go heartbeat as `running`, `stale`, `finalized`,
`record_debt`, or `absent`. `-reconcile` replays only the validated batch bound
to the authoritative preparation; it never creates another Git commit.

Additional lanes:

```bash
go run ./cmd/device-lane
go run ./cmd/smoke-lane
go run ./cmd/race-lane
go run ./cmd/release -out dist -verify-reproducible
```

The runtime remains no-cgo. `cmd/race-lane` may enable cgo only for Go's
test-only race instrumentation; CUDA race/synchronization checks use
`compute-sanitizer`.

`cmd/guard` protects shell execution. Model, dataset, and checkpoint stores are
read-only to automation for delete, move, and permission-changing operations.

## Repository map

| Path | Owner |
| --- | --- |
| `cmd/` | Thin executable composition roots |
| `internal/model`, `internal/inference` | Compiled model programs, runners, cache/session behavior |
| `internal/modelartifact`, `internal/tensorstats` | Bounded model inventories; L-moment, energy, rank-neighbor, and small-matrix spectral characterization |
| `internal/recipe`, `internal/modelrecipe`, `internal/workflowruntime` | Typed capability definitions, lifecycle, and execution |
| `internal/cuda`, `kernels/` | Driver binding, executor, generated bindings, manifested CUDA assets |
| `internal/projector`, `internal/latentimage`, `internal/latentvideo` | Multimodal projection and media generation |
| `internal/optimizer`, `internal/densecausal`, `internal/hybridtrain` | Muon and training implementations |
| `internal/trainingprogram`, `internal/scratchmodel` | Compiled training authority and corpus-derived scratch construction/execution |
| `internal/artifact`, `internal/repodb`, `internal/runrecord` | Identity, lineage, decisions, runs, and evidence |
| `internal/server`, `internal/server/webui` | HTTP contracts and embedded thin-client console/workbench |
| `compatibility.json` | Machine-checked feature and model claims |
| `docs/plan.json` | Current gate-executable work |

## Authoritative documentation

- [Compatibility matrix](docs/COMPATIBILITY.md): generated model and feature
  status with evidence tiers and verification commands.
- [Adaptive_new parity report](docs/adaptive_new_parity_report.md): capability,
  quality, wall, peak-memory, and remaining-gap assessment.
- [Training plan](docs/training_plan.md): scratch-controller-first Muon design
  and current implementation boundary.
- [RepoDB](docs/REPODB.md): artifact identity, lineage, and store contracts.
- [Merge floor](docs/MERGE_FLOOR_PLAN.md): automation-floor component status.
- [Iteration doctrine](skill.md): architecture, evidence, porting, and agent
  operating rules.
- [LICENSES.md](LICENSES.md) and [SBOM.cdx.json](SBOM.cdx.json): dependency and
  binary provenance.
