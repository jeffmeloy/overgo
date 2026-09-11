# Overgo

Overgo runs, trains, evaluates, and composes AI models in Go and CUDA.
Its browser workbench, command-line tools, and HTTP APIs share one execution
system and one artifact store.

The intent is recursive self-improvement (RSI): use measured outcomes to
improve both task capability and the process that proposes, executes, and
evaluates later work. Operators set goals, budgets, and constraints. Humans
or models propose changes; executable policy controls admission and activation.

![Overgo recursive self-improvement: propose, admit, realize, evaluate, decide, and observe, with measured feedback into later iterations](docs/assets/overgo-platform-technical-architecture.png)

[Editable figure definition](docs/assets/overgo_graphic.json)

## The improvement loop

Each iteration follows the same cycle:

1. **Propose** a falsifiable change and its expected benefit.
2. **Admit** it after checking dependencies, authority, and resource limits.
3. **Realize** the candidate through training, composition, or a code,
   recipe, data, or policy change.
4. **Evaluate** it on fixed tasks against a matched baseline, measuring
   quality and resource cost and using ablations to isolate its effect.
5. **Decide** whether to promote it, reject it, or request further work.
6. **Observe** outcomes and regressions, with quarantine and rollback when
   required.

The recursive step is the feedback into later iterations. Results can change
models and adapters, prompts and recipes, data and retrieval, routing, and
the policies used to plan, execute, and evaluate work. Rejected candidates
and failed runs remain part of that record.

Budgets, stop conditions, and recorded operator decisions bound the loop.
The [development plan](docs/plan.json) defines the current work.

## Capabilities

| Area | Implementation |
| --- | --- |
| Serving | Chat and text generation, embeddings, reranking, sequence scoring, streaming, constrained output, prompt caching, and live model switching |
| Media and structured data | Image generation, speech and transcription, video generation and editing, vision QA, time-series, and tabular execution paths |
| Model lifecycle | GGUF and safetensors intake, Hugging Face integration, conversion, quantization, inspection, model construction, adapters, and composition |
| Training | Token prediction, DPO, GRPO, host and CUDA execution, Muon optimization, and checkpoint/resume |
| Evaluation | Benchmark suites, reference comparisons, long-context checks, retrieval evaluation, ablations, and quality, throughput, and memory measurements |
| Agents and workflows | Registered tools, typed workflow graphs, authorization and receipts, scheduled jobs, remote peers, and recovery after interruption |

Architecture profiles bind model tensors to shared execution primitives for
dense, mixture-of-experts, recurrent, hybrid, encoder, diffusion, and
multimodal programs.

Training takes a model, data, objective, and budget. The trainer derives
optimizer settings and uses Muon across trainable parameter geometries.
Checkpoints retain the model, optimizer, random state, and data position
needed to resume the same run.

## Recipes and durable state

A **recipe** binds an exact model and task to its components and execution
policy. Validation and activation are separate steps; serving resolves an
active recipe for the requested capability. The workbench derives its
available modes from these declarations.

**OvergoDB** records content identities, lineage, recipes, runs, tool receipts,
measurements, decisions, checkpoints, and active versions. Its journal,
snapshots, and backup and recovery tools preserve the state needed to compare
attempts, resume work, and restore a predecessor.

Agent tools use registered, typed manuals that declare their arguments,
effects, and transport. Mutation authorization includes prior inspection,
applicable approvals, and a durable receipt before execution. Human-directed
work, agents, and automation use the same backend controls.

## Workbench and APIs

The server embeds a browser workbench with no separate client build step.
Use it to chat and attach media, select models, manage the model library,
run training and evaluations, inspect agent sessions, and review operations
and their records.

Local models and declared remote providers appear in the model catalog.
Remote providers use a hosted chat relay; local models run through the
Go and CUDA runtime.

HTTP interfaces include native routes and OpenAI- and Anthropic-compatible
surfaces. The [API manifest](docs/API_MANIFEST.md) lists commands, routes,
authentication requirements, and document contracts.

## Quick start

The current runtime target is Windows amd64 with Go 1.26 and an NVIDIA CUDA
driver. Bundled kernels target compute capability 8.9 or newer; see the
[kernel documentation](kernels/README.md) for build details.

From the repository directory, launch the workbench:

```powershell
.\overgo_gui.bat
```

The launcher builds the server and model-swap proxy, opens the browser, and
lets you select a model from the catalog. Use the Library to register local
models or declare a remote provider.

To launch with a supported GGUF model:

```powershell
.\overgo_gui.bat "D:\models\model.gguf"
```

Or run the server directly:

```powershell
go run ./cmd/server -listen 127.0.0.1:8080 D:/models/model.gguf
```

Open [the workbench](http://127.0.0.1:8080/). The server binds to loopback by
default. Authentication and network configuration are described in the
[security policy](SECURITY.md).

## Verification

Run the compatibility check and hermetic test lane:

```powershell
go run ./cmd/compatibility -check
go run ./cmd/test-lane ./...
```

Device, installed-model, race, and browser checks have separate lanes:
`cmd/device-lane`, `cmd/smoke-lane`, `cmd/race-lane`, and `cmd/webui-lane`.

Repository changes use the plan and commit gate to select checks, record
results, and advance work. See the [development workflow](skill.md).

## Technical references

- [API manifest](docs/API_MANIFEST.md) — commands, routes, and typed contracts
- [Benchmark report](docs/BENCHMARK.md) — evaluation and performance results
- [Model catalog](docs/model_compatibility.json) — prototypes and model-specific records
- [Media report](docs/MEDIA_REPORT.md) — media recipes, runs, and outputs
- [Training compatibility](docs/TRAINING_COMPATIBILITY.md) — training paths and recorded runs
- [Compatibility matrix](docs/COMPATIBILITY.md) — features and their verification references
- [Development plan](docs/plan.json) — current work and completion checks
- [License](LICENSE) and [software bill of materials](SBOM.cdx.json)

Current release: **v0.1.1**.
