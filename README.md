# Overgo

Overgo runs, trains, evaluates, and composes AI models in Go and CUDA. Its
browser workbench, command-line tools, and HTTP APIs share execution components
and an artifact store. Recipes bind models, tasks, and execution policy;
OvergoDB retains inputs, results, and lineage.

The goal is recursive self-improvement: measured outcomes improve model
capability and the process that develops it. An improvement is one a user sees
on the [scorecard](#measured-improvement). Operators set goals, budgets, and
stop conditions; agents propose changes; executable policy checks admission,
verification, and activation.

[Quick start](#quick-start) · [Capabilities](#capabilities) ·
[Measured improvement](#measured-improvement) · [Training](#training) ·
[Development](#development-and-verification) · [References](#references)

## Quick start

The runtime target is Windows amd64 with Go 1.26. GPU execution requires an
NVIDIA CUDA driver; bundled kernels target compute capability 8.9 or newer (see
[kernels](kernels/README.md)).

```powershell
.\overgo_gui.bat                          # build, start, open the workbench
.\overgo_gui.bat "D:\models\model.gguf"   # start on a supported GGUF model
.\overgo_gui.bat "" -training -model-builder
```

The launcher builds the server and the model-swap proxy and opens
[the workbench](http://localhost:8080/). The model picker switches the served
model live; Library registers local models or configures a hosted provider.
Flags after the model reach every served child, so enabled workspaces survive
model switches. Conversation fine-tuning and `cmd/train` routes need no recipe;
other training and construction need their registered inputs and recipes.

To serve one model directly:

```powershell
go run ./cmd/server -listen 127.0.0.1:8080 D:/models/model.gguf
```

A model runs from its own files whether or not anything has verified it: the
prototype its architecture selects, sized by its metadata or `config.json`, and
the template and sampling they declare. The catalog, each response and each
attempt's record say whether it is verified. Directories run for the `llama`,
`qwen2` and `qwen3_5` model types; other architectures run from a GGUF file.

The server binds to loopback by default; see [SECURITY.md](SECURITY.md) for
authentication and network settings.

## Capabilities

Available operations depend on the active model, recipe, backend, and
workspace. The [compatibility matrix](docs/COMPATIBILITY.md), the model catalog
(`go run ./cmd/validate -models -repo <store>`), and the
[training report](docs/TRAINING_COMPATIBILITY.md) record verification scope and
separate bounded smoke runs from broader claims.

| Area | Operations |
| --- | --- |
| Serving | Chat, generation, embeddings, reranking, sequence scoring, streaming, constrained output, prompt caching, live model switching |
| Media and structured data | Image and video generation, video editing, speech synthesis, transcription, vision QA, time series, tabular execution |
| Model lifecycle | GGUF and safetensors intake, Hugging Face integration, conversion, quantization, inspection, construction, adapters, composition |
| Training | Token prediction, DPO, GRPO, host and CUDA execution, Muon, checkpoint and resume |
| Evaluation | Benchmarks, reference comparisons, long-context and retrieval checks, ablations, quality, throughput, and memory measurements |
| Agents and workflows | Registered tools, typed workflows, authorization, receipts, schedules, peers, recovery |

The server embeds the workbench; no client build is needed. Native,
OpenAI-compatible, and Anthropic-compatible APIs share one server; the
[API manifest](docs/api_manifest.json) inventories commands, routes,
authentication, and document contracts.

## Architecture and state

Recipes declare architecture, tensors, components, and execution policy; shared
inference and training components execute them. Profiles cover dense,
mixture-of-experts, recurrent, hybrid, encoder, diffusion, and multimodal
programs. Pinned references are [llama.cpp](https://github.com/ggml-org/llama.cpp),
audio.cpp, and [Transformers](https://github.com/huggingface/transformers);
provenance and license records accompany ported capabilities.

OvergoDB retains artifacts, recipes, runs, measurements, decisions,
checkpoints, and active versions. Content identities bind results to their
inputs for reuse, comparison, resume, and rollback. Evaluation retains
responses and protocols, so a compatible run carries forward instead of
repeating.

![Overgo architecture and improvement loop](docs/assets/overgo-platform-technical-architecture.png)

[Editable SVG](docs/assets/overgo-platform-technical-architecture.svg) ·
[Figure definition](docs/assets/overgo_graphic.json)

## Measured improvement

`go run ./cmd/validate -scorecard` prints the user scorecard for every
activated model and task: time to first output, throughput, held-out quality,
failures, load time, peak device memory, and setup steps. Each value is read
from the store, or listed as a gap with its reason.

One improvement cycle (`go run ./cmd/loop -improve`) first runs the checks its
measurements depend on: each measurement's preflight, which names any measurement
whose prerequisites are missing, and the longer test runs, which stop the cycle
if one fails. It then runs the validations
the scorecard waits on (long-form guards, serving, media, training and projection
proofs, held-out benchmarks that replay their newest record), publishes the
scorecard, and files each decline and each open gap as a plan row. There is no
passed state: progress is each scorecard cell compared with its previous
measurement.

### Served models

Every model with an active recipe, as the record `scorecard-20261028T150740Z`
measured it: one measurement run on the reference PC (Ryzen 9 9950X, RTX 4090 D
with 48 GiB, Windows 11), every row served and scored on this machine. Modality
is the kind of data a model works in; input → output is what a request gives it
and what it answers. Speed is reported warm, as prompt processing and token
generation on a model already loaded, and never the first request after a load;
the measured figures follow the table. Peak device memory
is the most the measured attempt held on the card; "host" marks a model that
runs on the CPU. The full scorecard, with failure rate, load time and setup
steps, is `go run ./cmd/validate -scorecard`; beyond these models, 138
architecture profiles are registered (`go run ./cmd/validate -models -repo <store>`).

| Model | Modality | Input → output | Held-out quality | Peak device memory |
| --- | --- | --- | --- | --- |
| Qwen3.8 Flash-Next (80B MoE, UD-IQ4_XS) | text, vision-language | text → text; image → text | MMLU-Pro 0.66 on 1,209 cases; image questions 0.47 on 64 | 46.0 GiB |
| Qwen3.8-27B (Q6_K) | text, vision-language, video-language | text → text; image or video → text | MMLU-Pro 0.60; image and video questions 0.92 on 64 | 22.3 GiB |
| Qwen3.5-9B (Q8_0) | text | text → text | MMLU-Pro 0.49 | 9.0 GiB |
| Qwen3.5-4B | text | text → text | MMLU-Pro 0.49; 0.75 on 40 questions when left to think | 8.9 GiB |
| gemma-4-12B-it (fp8, native) | text | text → text | MMLU-Pro 0.49 | 14.3 GiB |
| gemma-4-12B-it (fp8 unpacked to bf16) | text | text → text | MMLU-Pro 0.49 | 22.2 GiB |
| gemma-4-E4B-it | text, vision-language, audio-language | text → text; audio or image → text | MMLU-Pro 0.44; 1.9% word error rate on 32 clips | 13.9 GiB |
| Bonsai-27B (Q1_0) | text | text → text | MMLU-Pro 0.42 | 4.2 GiB |
| MiniCPM5-1B | text | text → text | MMLU-Pro 0.30 | 2.0 GiB |
| Qwen2.5-0.5B (f16) | text | text → text | MMLU-Pro 0.17 | 0.9 GiB |
| Unlimited-OCR (bf16) | text | text → text | BBH 0.30, within noise of a constant answer | 5.5 GiB |
| Carbon-500M (f16) | genomic | DNA sequence → DNA sequence | perplexity 910 on 128 sequences (a draft model; not what it is for) | 1.0 GiB |
| Fractale-350M-base | text | text → text, carrying notes between chunks | carried gain 5.6 on 37 documents | host |
| needle | function calling | text → function call | 0.255 on 400 calls | host |
| Hy-Embodied-RxBrain-1.0 | vision-language | image + question → answer | 0.63 on 64 questions (chance 0.25) | 3.5 GiB |
| Krea-2-Turbo | text-to-image | text → image | 0.94 on 415 prompts | 25.0 GiB |
| SenseNova-U1-8B-MoT-Infographic-V3 | text-to-image | text → image (infographics) | 100 of 100, the suite's ceiling | 15.4 GiB |
| SimpleDiffusion-TensorProductAttentionRope | image generation | noise → image, no caption | no caption suite applies | 0.5 GiB |
| MOSS-Image-SFT | text-to-image | text → image; [trains adapters](docs/MOSS_IMAGE.md) | 0.91 on 415 questions about 64 captions | 13.0 GiB |
| Un-0 | image generation, video generation | class → 8×8 picture or 12-frame clip (a 120 kB toy model) | no caption suite applies | host |
| Wan2.1-T2V 1.3B | text-to-video | text → video | 0.91 on 128 clips | 3.6 GiB |
| LiveEdit | video editing | video + instruction → video | 0.84 on 64 clips | 3.5 GiB |
| Wan2.1 VAE | video autoencoder | video frame → frame (reconstruction) | 41.4 dB PSNR on 128 | 4.4 GiB |
| ViTXL_n08 | image autoencoder | image → image (reconstruction) | 20.5 dB PSNR on 128 | host |
| webssl-dino300m-full2b-224 | image embedding | image → embedding | retrieval 0.46 on 483 | host |
| webssl-dino-to-rae-bridge | image autoencoder | image → image through a fitted bridge | 18.2 dB PSNR on 128 | host |
| granite-speech 470M | speech recognition | speech → text | 0.9% word error rate on 32 clips | 1.0 GiB |
| nemotron-3.5-asr-streaming-0.6b | speech recognition | speech → text, streaming | 3.1% word error rate on 32 clips | host |
| pocket-tts | speech synthesis | text → speech | 2.2% word error rate, re-transcribed, on 64 texts | host |
| timesfm-2.5-200m | time series | numeric series → forecast | mean absolute error 2.24 on 11 series | host |
| timesfm-3.0 | time series | numeric series → forecast | mean absolute error 2.18 on 30 series | host |
| tabfm-1.0.0 | tabular | table → class | 0.72 on 7 tables, within noise of a constant answer | host |

Training recipes with a completed proof on this machine: Qwen2.5-0.5B (token
accuracy 0.508 against its base 0.500 on 24 held-out cases), Qwen3.5-4B,
Qwen3.5-9B, Hy-Embodied-RxBrain, LiveEdit, Wan2.1, tabfm (0.729 against 0.724)
and the dino-to-RAE bridge; see the
[training report](docs/TRAINING_COMPATIBILITY.md).

### Speed

Prompt processing (pp) is a fresh prompt's tokens over the time to its first
token, read on a model already loaded after one unmeasured prompt of that size,
so it includes each request's fixed costs; token generation (tg) is generated
tokens over decode time on a request again on the loaded model.

<!-- speed table: go run ./cmd/validate -readme writes it from the newest scorecard record -->
| Model | pp, 4,096-token prompt | pp, 16,384-token prompt | tg |
| --- | --- | --- | --- |
| Bonsai-27B-Q1_0 | 1,990 tok/s (2.06 s) | 1,613 tok/s (10.16 s) | 58 tok/s |
| Carbon-500M | 57,983 tok/s (0.07 s) | — | 504 tok/s |
| MiniCPM5-1B | 40,583 tok/s (0.10 s) | 18,640 tok/s (0.88 s) | 354 tok/s |
| Qwen2.5-0.5B | 62,203 tok/s (0.07 s) | 36,876 tok/s (0.44 s) | 528 tok/s |
| Qwen3.5-4B | 10,495 tok/s (0.39 s) | 6,172 tok/s (2.65 s) | 99 tok/s |
| Qwen3.5-9B | 5,880 tok/s (0.70 s) | 4,343 tok/s (3.77 s) | 97 tok/s |
| Qwen3.8 Flash Next | 2,898 tok/s (1.41 s) | 3,464 tok/s (4.73 s) | 168 tok/s |
| Qwen3.8-27B | 1,940 tok/s (2.11 s) | 1,591 tok/s (10.30 s) | 49 tok/s |
| Unlimited-OCR-bf16 | 23,405 tok/s (0.18 s) | 18,891 tok/s (0.87 s) | 459 tok/s |
| gemma-4-12B-it-fp8 | 3,149 tok/s (1.30 s) | 2,817 tok/s (5.82 s) | 36 tok/s |
| gemma-4-12B-it-fp8-native | 3,661 tok/s (1.12 s) | 3,208 tok/s (5.11 s) | 49 tok/s |
| gemma-4-E4B-it | 9,081 tok/s (0.45 s) | 7,523 tok/s (2.18 s) | 69 tok/s |
| qwen25-05b-judged-proof | — | — | 335 tok/s |
| qwen25-05b-order-25 | — | — | 335 tok/s |
<!-- end of the speed table -->

Strata, on the same PC and model, processes Flash-Next's 4,096-token prompt at
1,850 tok/s and its 16,384-token prompt at 3,290 tok/s, and generates 135 tok/s
on its own bench's short chat and 145 tok/s on agent turns. A conversation of 237,290
tokens is read in 2m16s, every later question answered in about 0.8 s, and a
question that branches off an earlier turn of such a conversation is answered
from the context already read, in about 1.3 s.

### Sample outputs

Each sample is an output of the attempts the scorecard reads, exported from
the store into [assets](assets) by
`go run ./cmd/validate -export-samples -media-scope measured -repo <store>` and
named by the SHA-256 of its bytes, so it is checkable against the run that
produced it. The time and memory beside it are the scorecard's: the time is the
median of the newest attempts at a whole request and the memory the newest's peak
on the card, so neither need be that output's own.

**Krea-2-Turbo, text → image.** "a red fox licking a vanilla ice cream cone in
snow", 512 × 512, 4 steps, seed 2026: 13.9 s, 25.1 GiB.

![Krea-2-Turbo: a red fox licking a vanilla ice cream cone in snow](assets/4f08c0fe12295fba0593c4932420b9d49defe20ce73a00b3b59c223f4f1840c3.png)

**MOSS-Image-SFT, text → image.** "A small red sailboat on a quiet lake at
sunrise.", 1024 × 1024, 50 steps, guidance 3, seed 42: 95 s, 13.0 GiB.

![MOSS-Image-SFT: a small red sailboat on a quiet lake at sunrise](assets/cfab6e10e0551187ce87aec12beab4f438ad063b2b202def0d6e8bf5856903d8.png)

**SenseNova-U1-8B-MoT-Infographic-V3, text → image.** "a minimal infographic
of three rising bars labeled A B C", 512 × 512, 4 steps, guidance 4.5, seed
2026: 13.1 s, 15.5 GiB. A 1536 × 2720 infographic in 50 steps from the frozen
visual-review case:
[image](assets/5f3a56dff6b7a4a9e96d3d129082fdc349717357ff48d54454ff9535fb0cce68.png).

![SenseNova: a minimal infographic of three rising bars labeled A B C](assets/2ff397c2af4142c28e0de7b09e510422e2bcd6a09731f3659d76055bb8d3e9fd.png)

**SimpleDiffusion-TensorProductAttentionRope, noise → image.** 256 × 256 in 8
steps, seed 2026: 0.92 s, 0.6 GiB.

![SimpleDiffusion: a 256 by 256 sample](assets/cd3f5feb951d9413f07060683035b12baaa909f67dead4b698c0fac24a629220.png)

**Wan2.1-T2V 1.3B, text → video.** The measured request is a small one: "a red
fox licking a vanilla ice cream cone on a snowy pine forest trail", 21 frames
at 32 × 16, 2 steps: 13.0 s, 3.6 GiB
([clip](assets/e6e246d2f322356bf07a83e137a2c4341a97eed8dd45b3f188736ba92e248865.gif)).
The frozen visual-review case, 81 frames at 832 × 480 in 50 steps, took 6 min
14 s when it was drawn at commit fcda318af (2026-09-12); its device memory was
not recorded.

![Wan2.1-T2V: 81 frames at 832 by 480](assets/d5c72f13e0bf07a1fb8268917b4da15f53e1343bfcdb3093a309591bb08f8ff9.gif)

**LiveEdit, video + instruction → video.** 21 source frames at 32 × 16 through
four denoising steps, seed 7: 9.5 s, 3.5 GiB
([clip](assets/5b1c2046f691ac277efc9b08f4f6fe0f35225e021a82afc8006fedebad127d4b.gif)).

**pocket-tts, text → speech.** "Once upon a time, in a quiet valley, a small
fox learned to sing.", voice alba: 5.4 s of 24 kHz audio in 2.6 s on the host
([WAV](assets/e845e24c17cae1270136a1aac2feb680c79e81c842a243ef2e69def47db8bc4a.wav)).

**granite-speech 470M and nemotron, speech → text.** The same held-out
LibriSpeech clip. granite, 2.8 s and 1.1 GiB: "my 1st and principal reason was
that they enforced beyond all resistance on any student who might attempt to
copy them this method of laying portions of distinct hue side by side".
nemotron, 4.3 s on the host: "My first and principal reason was that they
enforced beyond all resistance on any student who might attempt to copy them
this method of laying portions of distinct hue side by side."

**Hy-Embodied-RxBrain-1.0, image + question → answer.** On the model's own
demonstration picture, "What objects are on the stovetop, and where is the
green toy?" → "The stovetop holds a metal pot on the left burner", the
request's 12-token budget, in 9.8 s and 3.5 GiB.

**Un-0, class → picture or clip.** Class 1, seed 2026: an 8 × 8
[picture](assets/1995cda0e2c6ed9fb394c00c7baa68b208b3d8a6a604923588289d1c5950b556.png)
in 12 ms and a 12-frame 16 × 16
[clip](assets/db326dae32fc3850abcf31985b0516513f3a59ca27276de82eccbb66806da36d.gif)
in 29 ms, on the host.

## Training

One Muon optimizer handles matrices, vectors, scalars, embeddings,
normalization parameters, and adapters. The recipe binds the
[optimizer policy](internal/trainingprogram/optimizer_policy.json); the count
$P$ of parameters in the run's plan, a frozen embedding table among them,
determines its settings:

- **Learning rate:** $\eta=P^{-0.6}$, constant, for a model that already
  learned; $P^{-0.5}$, decaying linearly, for weights trained from
  initialization.
- **Momentum:** $\mu=(N-1)/(N+1)=29/31$ with $N=30$ effective samples, from the
  rule of thumb of Cochran's confidence-interval criterion.
- **Update scale:** $\eta\sqrt{\max(m,n)}\sqrt{1-\mu^2}$ per $m\times n$ group.

Muon combines gradients with momentum, normalizes the Nesterov direction, and
orthogonalizes by Newton–Schulz iteration. On CUDA a resident trainer's math
stays on the device and only scalars cross to the host, and a flat or streamed
step copies its weights to the device and back. Optimizer state keeps the resolved
settings, update count, and momentum. Suitability is assessed per task through
training evidence.

`cmd/train` is the one entry point. A registered recipe trains a checkpoint;
`-objective <alias>` trains a text model on a registered objective's training
split and publishes the model and the held-out verdict that binds it; `-route
<trainer>` runs a bounded family trainer by name. A training recipe activates on
the verification its first successful run records.

```powershell
go run ./cmd/train -route oscillatorimage -model Un-0 -steps 2
```

See the [optimizer](internal/hostoptimizer/optimizer.go) and
[training compatibility](docs/TRAINING_COMPATIBILITY.md) for implementation and
recorded runs.

## Agent tools

A [tool manual](internal/agenttool/manual.go) declares a tool's arguments,
effect class, and transport: Go functions, HTTP endpoints, fixed programs
without a shell, which receive only the variables their manual names over a
small base, MCP tools, sequenced JSON streams.
[cmd/agent-tool](cmd/agent-tool/main.go) publishes manuals to the store. The
[agent loop](internal/agentloop/coordinator.go) checks registration and
execution bounds; mutations need a completed inspection and exact approval and
record a durable receipt first. The serving executor refuses every destination
[outside public unicast](internal/agenttool/security.go), the list the media
fetcher uses: loopback, private, link-local, multicast, carrier-grade NAT,
documentation and reserved ranges. The operator executor admits them.

## Development and verification

[skill.md](skill.md) holds the development principles; [docs/plan.json](docs/plan.json)
holds priorities, dependencies, owners, and acceptance.

1. Take the next row: `go run ./cmd/plan -next`.
2. Land it: `go run ./cmd/loop -land <item>/<step> -message-file -`. The
   [small gate](internal/landing/landing.go) (its package doc lists the stages)
   checks the tree and commits with the step removed from the plan.
3. The improvement cycle measures the scorecard and files each decline as a row.
4. Scorecard gaps become the next rows.

The gate judges a commit from the tree alone: unit, golden, hermetic, and
small-device checks. Real-model and live-store suites belong to the cycle.
Priorities are fewer components, facts classified once through existing owners,
and less repeated work; rules that matter are enforced by the gate. The
[loop driver](cmd/loop/main.go) supervises workers under configured limits and
operator stops.

Local feedback on affected packages:

```powershell
go run ./cmd/validate -tests ./internal/recipe
```

Device and browser checks use `cmd/device-lane` and `cmd/webui-lane`; installed
models are checked with `go run ./cmd/validate -smoke`.

## References

- [API manifest](docs/api_manifest.json) — commands, routes, typed contracts
- Benchmark report — `go run ./cmd/benchmark-report -repo <store>` prints evaluation and performance results
- Model catalog — `go run ./cmd/validate -models -repo <store>`
- Media report — `go run ./cmd/validate -media markdown -repo <store>` (`-media json` for the typed projection)
- [Training compatibility](docs/TRAINING_COMPATIBILITY.md) — training evidence and scope
- [Compatibility matrix](docs/COMPATIBILITY.md) — features and verification references
- [Development plan](docs/plan.json) — current work and completion checks
- [License](LICENSE) and [software bill of materials](SBOM.cdx.json)
