# Overgo

Overgo is an experimental, no-cgo Go system for model inference, training,
evaluation, and composition on consumer NVIDIA hardware.

It inherits pinned formats, semantics, and kernel behavior from llama.cpp.
Adaptive_new supplies capability oracles and performance baselines. Overgo owns
the compiled recipes, runtime, Muon training path, evidence, and artifact
lifecycle.

**Current host:** Windows amd64, Go 1.26, NVIDIA CUDA driver,
`CGO_ENABLED=0`. `github.com/dlclark/regexp2/v2` is the sole third-party Go
runtime dependency. The implementation and public interfaces remain
experimental.

## Contents

- [System contract](#system-contract)
- [Architecture](#architecture)
- [Capability status](#capability-status)
- [Recorded performance](#recorded-performance)
- [Requirements and data roots](#requirements-and-data-roots)
- [Quick start](#quick-start)
- [Multimodal processing](#multimodal-processing)
- [HTTP server](#http-server)
- [Training](#training)
- [Commands](#commands)
- [Development and gates](#development-and-gates)
- [Repository map](#repository-map)
- [Authoritative documentation](#authoritative-documentation)

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
feature authority. Its evidence tiers matter: implemented, synthetic-fixture,
real-artifact parity, and promoted performance are different claims.

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

The CUDA runtime loads installed NVIDIA DLLs directly. A locked-thread worker
owns each driver context. Compiled graphs reuse validated topology, BLAS
selection, and arena plans. Manifest-pinned CUDA/PTX assets are the only
project-owned non-Go runtime components.

Preloaded decoding retains model weights, KV state, recurrent state, and graph
outputs on device; only sampling boundaries cross to Go. Media generation
sessions retain compiled branch graphs, prefix KV, and hidden state across
denoise steps. Capability sessions are cached by model artifact, recipe,
device, and execution policy. Per-entry leases allow unrelated keys to execute
concurrently.

## Capability status

| Area | Current implementation | Promotion boundary |
| --- | --- | --- |
| Text inference | Dense, MoE, recurrent, hybrid, encoder, encoder-decoder, diffusion-text, and speculative components | Real-artifact evidence remains model-specific |
| Quantized execution | GGUF parsing/conversion plus native quantized weights and experts through recipe-selected residency | Exact type/family coverage is generated in `docs/COMPATIBILITY.md` |
| Serving | Native llama.cpp-style routes plus OpenAI Chat/Completions/Embeddings/Responses and Anthropic Messages | Protocol features are contract-tested; model quality remains recipe-specific |
| Multimodal input | Typed image, audio, and video projection; bounded local/allowlisted remote media; mixed-media history | E4B image/audio and selected Gemma/Qwen projectors have real evidence; many catalog rows remain fixture-only |
| Image generation | Typed conditioning, resident CUDA denoise, PNG artifact publication | Krea promoted with bounded pixel evidence; SenseNova full decode/publication remains open |
| Video generation | Resident Wan denoise and CUDA VAE decode | Fresh Overgo result; retained Python reference still needs a same-revision rerun |
| Speech, forecast, table, seq2seq | Shared runtime/recipe components exist | Promotion varies; consult the parity report and compatibility tiers |
| Training | Muon-only dense and hybrid primitives; resident CUDA dense path | Frozen-lexical Carbon is promoted; compiled model construction and universal training are not complete |

Explicit gaps include production SenseNova image/edit publication, LiveEdit,
Unlimited OCR parity, Pocket-TTS output-quality evidence, complete-state
training resume, model-from-scratch construction, and real-model
Qwen3.5/E4B/Gemma4 training.

## Recorded performance

Snapshot reviewed 2026-08-13. These are retained single-machine results, not
portable guarantees. The authoritative protocol, artifact identities, quality
checks, and open caveats are in the
[adaptive_new parity report](docs/adaptive_new_parity_report.md).

| Workload | Reference | Overgo | Verdict |
| --- | ---: | ---: | --- |
| Qwen3.5-4B decode | adaptive 16.76 ms/token | 10.96 ms/token; about 2.2 GB more peak | Wall lead, memory loss |
| RxBrain VQA | adaptive 18.4-23.1 s / 11.97 GB | 9.5 s / 10.59 GB | Wall and peak lead |
| Carbon-500M causal Muon | adaptive 6.816 s loop / 11.47 GiB | 5.05-5.13 s / 4.620 GiB; matched trajectory | About 25% loop-wall and 60% peak lead |
| Krea 2048 image | Python 97.5-135.4 s; adaptive 158.144 s / 33.47 GB | 63.510 s / 32.732 GB; MAE 0.03910 | Bounded-quality wall/peak lead |
| Wan video | retained Python 463.4 s; adaptive 795.1 s | 370.27 s; stage peaks 7.875/10.330 GB | Lead vs retained references; matched rerun open |
| SenseNova generation core | adaptive reusable body 6.88 s | cold 3.65-4.00 s; warm 3.66-4.02 s / 0.710 GiB | 41.6% warm-body lead; full route open |
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

`go test ./...` expects the configured evidence artifacts referenced by the
parity matrix. Missing required evidence is UNAVAILABLE and fails rather than
silently passing.

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
- a small built-in web UI on otherwise unmatched GET routes.

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
complete-state resume containing optimizer, RNG, and data cursor. The planned
`TrainingRunPlan`, `ScratchConstruction`, and model-level `TrainingProgram` remain
design contracts; `cmd/train` cannot yet derive and initialize a new model from a
dataset.
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
| `cmd/latentvideo-run` | Wan latent-video generation and optional VAE decode |
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

The gate derives affected tests from the import graph and conditionally checks
formatting, vet, build, kernel manifest, SBOM, compatibility claims, magic
closures, and device evidence. Every result states what ran and what did not.
UNAVAILABLE evidence fails a claim that requires it.

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
| `internal/recipe`, `internal/modelrecipe`, `internal/workflowruntime` | Typed capability definitions, lifecycle, and execution |
| `internal/cuda`, `kernels/` | Driver binding, executor, generated bindings, manifested CUDA assets |
| `internal/projector`, `internal/latentimage`, `internal/latentvideo` | Multimodal projection and media generation |
| `internal/optimizer`, `internal/densecausal`, `internal/hybridtrain` | Muon and training implementations |
| `internal/artifact`, `internal/repodb`, `internal/runrecord` | Identity, lineage, decisions, runs, and evidence |
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
