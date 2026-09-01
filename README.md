# Overgo

Overgo is an operating recursive self-improvement (RSI) system platform for
model-driven software agents.

It has two permanent uses:

- provide the deterministic framework needed to build RSI automation;
- expose the same model, agent, training, evaluation, and automation framework
  to people for useful work through the operator workbench, APIs, and command
  line.

The central design rule is that model cognition proposes work while
deterministic code controls state, authority, execution, verification, and
recovery. A model may interpret a problem or generate a candidate patch, but
acceptance still depends on measured evidence and executable policy.

A second design rule governs how models enter the system. Overgo owns no
model-family executors. It ports llama.cpp behavior -- container formats,
numerical rules, quantization, and kernels -- into generic Go and CUDA
components, and expresses each model family as a typed prototype definition in
OvergoDB: the tensor layout, execution modes, operators, and numerical features
that family needs. Specific checkpoints are verified as instances inside a
prototype. Adding a model is adding a prototype definition and a recipe, not a
new code path -- model identity lives in the store as data, and a reviewed
census (docs/family_branch_baseline.json, enforced by the architecture
family-branch test) pins the small residue of family-named branches that
remain in shared execution code: the GGUF tokenizer-algorithm kinds and the
llama-compatibility loader. That census only shrinks; a new model
expressible through existing primitives introduces no family-named branch.

An operator can steer the framework by supplying goals, constraints,
priorities, and decisions about what to investigate next. The graphical
workbench is an external interface for steering and observation; it is not the
RSI controller. The RSI path adds model-directed steering from durable
measurements and state alongside human-directed work. Both paths use the same
deterministic control plane to admit, evaluate, activate, reject, or recover
work.

Overgo was built by the loop it describes. Every commit in its history was
dispatched by `cmd/plan`, gated by `cmd/gate`, and carries the trailers that
record it -- the plan step it closed, the code and recipe manifests it
observed, and the falsifiable check that passed (`Overgo-Plan-Item`,
`Overgo-Code-Manifest`, `Overgo-Verify`) -- so the repository is the loop's
output, not a description of a loop that has yet to run.

What that demonstrates is bounded, and the boundary is the point. Supervised,
system-level RSI is not a future goal here: it is how this repository exists --
an external model proposes and executes work, the deterministic control plane
admits, gates, and records it, and an operator steers goals and priorities. The
recursive loop is implemented end to end: measurement, controlled experiments,
evidence-gated policy promotion, typed steering proposals, and a driver that
consumes them under mechanical stop conditions. What remains open is narrower
and specific -- unattended operation at scale, improvement of the proposing
model's own cognition, and measured effectiveness against competing strategies
(see What remains before fully autonomous RSI).

![Overgo technical architecture](docs/assets/overgo-platform-technical-architecture.png)

[Structured figure definition](docs/assets/overgo_graphic.json)

## Objectives

Reliable RSI requires more than repeatedly asking a model to modify its own
code. The surrounding system must make each attempt reproducible, bounded, and
falsifiable. Human-directed use needs the same properties so useful work can be
inspected, resumed, compared, and trusted.

```text
steering: operator or model
      -> typed task and bounded context
      -> deterministic admission and execution
      -> independent tests and evaluations
      -> evidence-and-policy decision
      -> activate, reject, or roll back
      -> durable measurements and state
      -> next steering proposal
```

The same mechanism must also evaluate changes to the automation itself. An
improved prompt, planning policy, retrieval method, gate selector, or repair
strategy is only an improvement when repeated measurements show a better
result under the same constraints.

## What exists

Overgo already combines the following components in one codebase:

| Area | Current implementation |
| --- | --- |
| Model execution | Shared Go and CUDA execution for dense, mixture-of-experts, recurrent, hybrid, encoder, diffusion, and multimodal programs |
| Modalities | Text, image, audio, video, time-series, and tabular input or output paths |
| Model lifecycle | GGUF and safetensors intake, conversion, quantization, model construction, training, evaluation, serving, and rollback |
| Agent definitions | Immutable bindings among prompts, model configurations, tools, datasets, automations, and policies |
| Workflow control | Typed dependency graphs, bounded admission, resource placement, scheduled execution, restart recovery, and remote peers |
| Mutation safety | Exact tool identity, persistent executable policy, inspection before mutation, argument-bound approval, and a durable receipt before a side effect |
| Reproducibility | Content identities, provenance, run records, stage receipts, checkpoints, exact resume checks, and versioned activation |
| Verification | Plan-driven gates with manifest-derived check selection, structural source analysis, host/device comparisons, model-specific evidence, and compatibility records |
| Benchmarks and evaluation | Store-derived benchmark catalogs, native lm_eval-family suites, per-model prompt templates, domain-routed evaluation, and measured capability claims |
| Human-directed work | External workbench for goals, chat and media, agent sessions, model and data operations, measurement review, intervention, and rollback |
| RSI steering | Typed falsifiable proposals through deterministic admission, consumed by the driver under budget, saturation, and operator-stop conditions; shared with human-directed work |
| Tool calling | UTCP-style manuals in the store: each tool declares its effect class, typed arguments, and native transport; unregistered tools are not callable |
| Interfaces | Command line, HTTP APIs, scheduled jobs, and an external operator workbench over shared backend state; the whole public surface is enumerated in the generated [API manifest](docs/API_MANIFEST.md) |

## Capabilities

The following sections describe the implemented mechanisms. They do not imply
that every mechanism is available for every model. A capability becomes
servable only when the store contains an active recipe for the exact model,
task, runtime policy, and required components. Compatibility records identify
the artifacts and environments for which verification evidence exists.

### Serving and generation

- Text completion, chat completion, infill, embedding, reranking,
  tokenization, detokenization, perplexity measurement, and sequence scoring.
- OpenAI-compatible HTTP surfaces for Completions, Chat Completions,
  Responses, Embeddings, image generation, video generation and editing, and
  speech generation.
- An Anthropic Messages-compatible surface with token counting, system
  messages, image input, tool use, streaming events, and bounded local
  reasoning output.
- Buffered and streaming generation, stop sequences, grammar-constrained
  sampling, JSON Schema grammar compilation, log probabilities, prompt-state
  reuse, context shifting, and supported speculative decoding paths.
- Runtime LoRA scale control, multimodal prompt projection, concurrent request
  admission, continuous multi-request scheduling, model properties, device
  statistics, and live model replacement through the model-swap proxy.
- Recipe-selected image, video, video-editing, speech, time-series, tabular,
  encoder, decoder, recurrent, hybrid, mixture-of-experts, and diffusion
  execution paths.

Authentication and method requirements are defined per route. The generated
[API manifest](docs/API_MANIFEST.md#routes) is the authoritative route list.

### Model and artifact lifecycle

- GGUF inspection, hashing, splitting, merging, conversion, quantization, and
  importance-matrix processing.
- Safetensors inspection, bounded repository loading, conversion, and
  checkpoint publication.
- Hugging Face model and dataset search, verified download, repository intake,
  and conversion into registered local artifacts.
- Model characterization, tensor inventory, vocabulary inspection, hidden
  state and attention analysis, model construction, and active-recipe
  inspection.
- Scratch-model and corpus-derived model construction, including compiled
  construction plans, model-builder workflows, and registered output
  artifacts.
- LoRA extraction, model grafting, component alignment, interface-adapter
  training, model composition, candidate evaluation, activation, rollback,
  and supersession.
- CUDA device inspection, kernel compilation, generated host bindings, kernel
  ABI manifests, embedded PTX validation, and source-to-binary identity checks.
- Architecture profiles that compile tensor namespaces, operator sequences,
  numerical policies, cache behavior, execution modes, and runtime limits into
  a model program. A checkpoint that fits existing compiled behavior requires
  a prototype and recipe; a new numerical mechanism requires a corresponding
  generic operator or policy implementation and new verification evidence.

### Training and adaptation

- Native token-prediction training, direct preference optimization (DPO), and
  group relative policy optimization (GRPO).
- Host-reference and supported CUDA-resident forward, backward, loss, and
  optimizer execution.
- Compiled Muon optimizer plans with host and CUDA implementations, parameter
  grouping, portable optimizer state, and exact resume validation.
- Dataset streaming, deterministic batch selection, bounded sequence plans,
  frozen lexical or component surfaces, and adapter-only training with donor
  models held immutable.
- Exact checkpoints that bind model identity, recipe, objective, optimizer,
  random-number state, data-stream position, and parent lineage.
- Supervised training sessions with resource bounds, phase measurements,
  pre-training and post-training evaluation brackets, health classification,
  MoE router observations, evidence-derived routing targets and controllers,
  and operator intervention records.
- Dedicated probes and workflows for dense causal, mixture-of-experts,
  hybrid, image, video, modality-transformer, time-series, and tabular
  training paths.

The supported training combinations and their evidence are listed in
[Training compatibility](docs/TRAINING_COMPATIBILITY.md).

### Evaluation, routing, and live safety

- Store-derived suites for exact answers, multiple choice, grouped choice,
  generated answers, structured generation, instruction rules, probability
  mass, sequence likelihood, preference targets, numeric targets, retrieval,
  and media outputs.
- Native evaluators and prompt construction for MMLU, MMLU-Pro, TruthfulQA,
  IFEval, BBH, MuSR, and other registered benchmark families.
- Per-model prompt templates, evaluation-domain declarations, group-safe data
  splits, resource measurements, failure records, comparison reports, and
  campaign history.
- Capability-gap derivation, grounded retrieval replay, candidate attribution,
  evidence coverage checks, route selection, and counterfactual replay of
  alternative routing rules.
- Media-quality, offline-artifact, and composite-generation evaluation with
  production probes, candidate attribution, and ablation-bound promotion
  evidence.
- Deterministic rollout cohorts, reproducible evidence projections,
  distribution-free live-safety windows, circuit-breaker escalation,
  quarantine, atomic rollback, and evidence-bound quarantine reentry.
- Multidimensional improvement decisions that prohibit promotion when any
  required fitness dimension regresses, even if an aggregate score improves.

### Data and retrieval

- Typed dataset registration, inventory, aliases, content identities,
  provenance, availability checks, and deterministic train, validation, and
  test splits.
- Bounded readers for the supported Hugging Face Arrow IPC and Parquet
  subsets, plus benchmark-specific normalization and import paths.
- Text, image, audio, preference, rollout, interaction, capability-episode,
  and domain-specific training records.
- Incremental selection, streaming transformations, resumable materialization,
  benchmark catalog construction, and repository-backed dataset preview.
- Text, image, audio, preference, grouped-rollout, and multimodal training
  materializers that publish exact source and stream-position evidence.
- Agent retrieval with persisted indexes, embedding and reranking providers,
  retrieval receipts, held-out retrieval judgments, and quality evaluation.

### Agents, tools, workflows, and automation

- Immutable agent definitions that bind a prompt, model recipe, exact tool
  manuals, datasets, capability bundles, automations, and policies.
- UTCP-style tool manuals with typed arguments, effect classification,
  built-in or native HTTP streaming transports, executable policy, and exact
  registered identities.
- Inspection-before-mutation admission, action-bound approvals, durable
  pre-execution receipts, mutation checkpoints, bounded sessions, restart
  recovery, and interaction replay.
- Typed workflow graphs with compiled dependencies, bounded parallel ready
  sets, stage adapters, run receipts, cancellation, failure publication, and
  deterministic recovery of completed stages.
- Closed-world tool workflows that bind registered manuals, authorization
  evidence, typed inputs, execution receipts, and retained results to one
  compiled workflow authority.
- Scheduled automation, authenticated webhook intake, input binding, delivery
  tools, execution history, and recovery after interruption.
- Coalesced wake signals, late-stimulus reconciliation, incremental context
  handoff, and persistent obligations that reconstruct required follow-up work
  after interruption or restart.
- Remote peers, capability publication, placement decisions, delegated agent
  execution, remote stage adapters, capacity observations, and peer lifecycle
  reconciliation.
- Operator-visible operation queues, dependency graphs, timelines, decisions,
  cancellation, waiting, and recovery actions.

### Storage, provenance, and recovery

OvergoDB is the canonical transaction authority for artifacts and operational
evidence. It provides:

- kind-qualified SHA-256 content identities, immutable blobs, typed document
  contracts, manifests, lineage, causality, locations, and compare-and-set
  aliases;
- atomic batches, idempotent batch keys, store-head preconditions, a
  hash-chained journal, checksummed frames, writer locking, and torn-tail
  recovery;
- bounded projections and queries, query cursors, alias history, read-only
  refresh, per-projection checkpoints, snapshots, segmented journals, and
  deterministic replay;
- verified backup, import, compaction, retention, rebuild, repair, parity
  checking, and migration drills; and
- operational records for runs, evaluations, approvals, receipts, policies,
  rollouts, findings, advisories, and compatibility evidence.

The `overgodb-query`, `overgodb-backup`, `overgodb-import`,
`overgodb-compact`, `overgodb-rebuild`, `overgodb-repair`, and `store-check`
commands expose the principal administrative operations.

### Development, verification, and release

- A canonical dependency-aware plan, bounded work leases, explicit workspace
  claims, and deterministic dispatch through `cmd/plan` and `cmd/loop`.
- A commit gate that binds changes to the current plan step, derives test scope
  from source and manifest impact, executes the required checks, constructs the
  accepted Git commit, and publishes the result to OvergoDB.
- Structural source analysis, ownership checks, clone and closure censuses,
  modernization censuses, staged-surface records, architecture ratchets,
  generated API records, and compatibility evidence checks.
- Closure scanning, unclassified-surface reporting, deterministic remediation,
  reactivation checks, and restoration of displaced alias authority.
- Separate hermetic, race, browser, CUDA device, installed-model smoke, and
  performance lanes. An unavailable required environment is reported as
  unavailable and is not treated as successful evidence.
- Reproducible Windows release archives, version checks, kernel ABI and source
  verification, generated bindings, a CycloneDX software bill of materials,
  and content hashes for release artifacts.

## Capability status and evidence

Overgo distinguishes implementation from evidence and activation:

| State | Meaning |
| --- | --- |
| Compiled mechanism | The source contains the generic operator, format, workflow, or policy implementation. This alone is not a claim about a particular model artifact. |
| Candidate recipe | A content-addressed recipe binds an exact model and its dependencies to compiled behavior, but it is not available for serving. |
| Validated or verified recipe | The recipe passed its required structural or execution checks and remains non-serving unless separately activated. |
| Active recipe | A versioned activation makes the exact recipe available for its declared task and retains its predecessor for rollback where applicable. |
| Verified evidence tier | A candidate execution produced the required typed artifact. |
| Parity evidence tier | Output matched the declared reference implementation under the recorded comparison. |
| Production evidence tier | Production-shaped workloads and the recorded environment support the claim. |

Absence of a model, dataset, device, or required reference is reported as
unavailable rather than successful. The compatibility reports bind each claim
to an exact artifact, recipe, source revision, environment, and verifier.

## Operator workbench

The graphical workbench is the human interface to the same control plane the
automation uses. It is a self-contained browser client embedded in the server
binary: `cmd/server` serves it at the listen address under a same-origin
content-security policy, with no external assets or build step. The browser
projects backend records and submits bounded actions; it does not own
execution state or participate in the autonomous decision loop. Every panel
reads the same store the command line and the automation read, so a result
shown in the workbench is the same durable record a gate or a script would
see.

### Surfaces

- **Chat and media.** Multimodal conversation against the served model:
  text, image, and audio attachments flow through the same bounded media
  policy the APIs enforce, projected media and token counts survive in
  earlier turns of the formatted history, and conversations continue across
  server restarts because the interaction record — not the browser — owns
  the state. Dedicated image, video, and speech generation workspaces drive
  the corresponding model recipes.
- **Agent sessions.** Agent definitions bind a prompt, model configuration,
  tools, and policies immutably; sessions execute with native tool calling
  against the typed tool catalog. Layered authority maps bound what each
  session may touch, and every past interaction replays from its durable
  record for inspection.
- **Approvals inbox.** Exceptional mutations — actions outside a session's
  standing authority — queue as argument-bound approval requests. The
  operator sees the exact tool identity and arguments that will run; a
  durable receipt precedes the side effect, and nothing executes on a stale
  approval.
- **Models and data.** The model catalog is derived from the store: every
  locally present model with its active recipe, verified capability tier,
  and measured evidence — benchmark throughput and evaluation scores appear
  beside each entry in the picker, so choosing a model is choosing from
  evidence rather than filenames. Dataset browsing reads the active catalog
  without payload access, and the model builder and recipe inspector expose
  construction and activation records.
- **Analysis.** Structure, tensor, attention, logit, hidden-state, and
  vocabulary inspectors over the loaded artifact. The attention view replays
  bounded plain-causal attention on the host and refuses compiled score
  policies it cannot reproduce exactly — the displayed weights are recomputed
  evidence, not a screenshot of runtime state.
- **Training.** Session-supervised training: a session is leased, observed,
  and recorded; the observer captures a pre-training evaluation bracket on
  admission and publishes the post-training deltas on finish, so every
  session's measured effect is attributed to it in the store.
- **Evaluation.** Suites derive from the benchmark catalog in the store —
  no suite files — and route by each model's declared evaluation domains,
  so a DNA model never meets English multiple choice. Results publish back
  through the campaign ledger and reappear as the evidence beside the model
  in the picker.
- **Workflows and runtime.** Typed workflow graphs with bounded admission,
  scheduled jobs, run browsing over the store's phased run records, media
  compositions, live runtime activity, and remote peers for distributed
  execution.
- **Artifacts and provenance.** The artifact gallery and the provenance
  records behind every displayed result, down to the content identities a
  claim cites.

### Usage

Launch the server with a model (see Quick start) and open the listen
address; the workbench is the default page. A typical serving session:
pick a model in the catalog — the evidence line under each entry shows its
measured throughput and evaluation scores — then chat, attach media, or
open a generation workspace. A typical measurement session: open the
evaluation workspace, run the store-derived suites for the served model,
and watch the results land in the picker as published evidence. A typical
training session: start a supervised session from the training workspace
and read its bracket when it finishes — the pre/post deltas are the
session's measured effect, not an impression.

Interventions follow the same shape everywhere: inspect the record first,
act through a bounded control, and find the durable receipt in the store
afterward. Steering from the workbench — goals, priorities, interventions,
approvals, rollback — enters through the same bounded interface the RSI
path uses. Nothing is reachable from the browser that is not equally
reachable, and equally checked, from the deterministic control plane.

## Tool calling: UTCP, not MCP

Agent tool calling follows the UTCP philosophy: describe each tool once, in a
typed manual, and invoke it over its own native transport — no
protocol-translation server sits between the agent and the tool, and there is
deliberately no MCP bridge in the tree.

A manual is a store document (`overgo/agent-tool-manual/v1`) declaring the
tool's name, its **effect class** — `inspection` reads state, `mutation`
changes it and is permitted only after a prior inspection — its typed arguments, and its
transport binding: a built-in Go function or an http-json-stream endpoint.
Manuals are published to OvergoDB under registered aliases; orchestration
resolves tools from the store, never from code alone, so an unregistered tool
is not callable and a mutation without a durable receipt does not execute.

```bash
go run ./cmd/agent-tool -manuals tools.json            # register manuals
go run ./cmd/agent-tool -resolve <name>                # resolve one registered manual
```

Agent sessions in the workbench execute against this same catalog: the server
derives each session's tool-manual set and layered authority map server-side,
exceptional mutations queue as argument-bound approvals, and every invocation
lands as a durable record. The `agent-tool-manuals` protocol row in the
[API manifest](docs/API_MANIFEST.md) is the machine-readable statement of this
contract; capability proxies that expose model recipes as agent tools are
staged surface awaiting the delegation campaign
([staged surface](docs/staged_surface.json)).

## Current state

The deterministic harness is substantially implemented, but evidence maturity
varies by capability.

| Requirement | State |
| --- | --- |
| Exact identities for code, models, data, tools, policies, and runs | Implemented |
| Bounded and recoverable workflow execution | Implemented |
| Durable authorization and mutation receipts | Implemented |
| Independent host, CUDA, integration, and model verification lanes | Implemented; artifact coverage varies |
| Verification derived from exact code manifests, fail-closed on unproven independence | Implemented |
| Exact training checkpoints and fail-closed resume | Implemented for supported training paths |
| Agent and automation lifecycle management | Implemented |
| Measurement of automation effectiveness | Implemented: every gate run emits a typed attempt record, success and failure |
| Durable cross-run measurement history for steering | Implemented: attempt history is store-queried and aggregated per step and strategy |
| Controlled comparison of competing automation strategies | Implemented: isolated-worktree experiments from one baseline, judged by each step's own verify |
| Evidence-conditioned promotion of automation policies | Implemented: declared to active on recorded evidence and measured wins, rollback retained |
| Model-proposed steering from accumulated evidence | Implemented: typed falsifiable proposals through deterministic admission; history-free proposals are refused |
| Closed recursive policy-improvement loop | Implemented: the driver consumes admitted proposals under budget, saturation, and operator-stop conditions; live unattended campaigns remain to accumulate evidence |
| Evidence-derived model routing | Implemented: typed routing decisions derived from admitted evaluation evidence, every selection recorded, candidate routing rules measured by counterfactual replay over recorded decisions |
| Deterministic rollout of promoted candidates | Implemented: content-addressed rollout plans, deterministic cohort assignment that is identical across retry, replay, and restart, reproducible projection reports, promotion only after the required evidence is present |
| Live-safety containment | Implemented: comparison windows derived from each metric's own history without distributional assumptions, staged circuit-breaker escalation ending in operator stop, atomic rollback with recorded cause, and quarantine reentry that requires evidence committed after the quarantine |
| Prototype recipe lifecycle with rollback | Implemented: candidate, validated, active, refused, superseded, and rollback transitions; a superseded recipe returns to service only through a new verified activation |
| Composition improvement driver | Implemented: exact retrieval over normalized component descriptors, seam alignment residual audited against measured bridge results, ranked candidate enumeration, adapter training with both models frozen, multidimensional fitness selection without averaging, promotion through the existing ablation-controlled check, and run bounds computed from measured history; live composite campaigns remain to accumulate evidence |

Architecture support and artifact verification are separate claims. Shared
components can express more model types than are installed and tested on the
current hardware. Compatibility records therefore identify the exact artifact,
configuration, environment, and verification command behind each claim.

## The recursive loop, as implemented

The mechanisms the loop needs now exist end to end, each deterministic and
store-recorded:

- **Measurement.** Every gate run — success and failure alike — emits a typed
  attempt record binding the plan step it served, the strategy that produced
  it, the manifest selection it observed, the change size, wall time, and
  outcome. Attempt history is queryable across runs, strategies, and code
  revisions, aggregated per step, never scraped from logs.
- **Controlled experiments.** Competing worker strategies run the same plan
  step in isolated worktrees from the same baseline commit; each trial is
  judged by the step's own machine verify, selection is verify outcome first
  and measured cost second, and failed trials persist as durable
  counterexamples.
- **Policy promotion.** Automation policies move declared, experimental,
  verified, active — exactly as model recipes promote. Each promotion demands
  recorded evidence: an experiment for experimental, repeated measured wins
  for verified, and a measured non-regression over the incumbent for active,
  with the displaced incumbent retained as the rollback target.
- **Steering proposals.** A model — or a human — proposes the next task as a
  typed document: goal, predicted benefit, predicted cost, explicit
  uncertainty, a falsifiable check, and references into recorded measurement
  history. Deterministic admission refuses a proposal that cites nothing or
  cites measurements nobody recorded, and converts an admitted proposal into
  a plan row whose verify is the falsifiable check and whose rationale
  carries the prediction, so outcome is comparable against what was promised.
- **Closure.** When the plan drains, the driver consumes the next admitted
  proposal instead of stopping, under three mechanical stop conditions: the
  invocation budget, saturation — consecutive proposal rows ending without
  their falsifiable check passing — and the explicit recorded operator stop.
  Proposals never execute anything; every row still advances only through
  the gate.
- **Routing.** Which model serves a task is a recorded decision. Candidates
  are admitted only through verified evaluation evidence measured on the
  same data split. The quality threshold is the current model's own
  measured value, and one registered rule picks the cheapest candidate
  that meets it. A new routing rule is first replayed against the recorded
  decisions to measure what it would have changed; only then can it serve
  live traffic.
- **Rollout and containment.** A promoted candidate is activated for
  everyone only after a rollout: an immutable plan assigns each workload
  to the baseline or the candidate by a hash of stable identities, so the
  assignment is the same across retries and restarts. Reports over the
  rollout are reproducible: reading the same store state twice produces
  the same bytes. Promotion requires the observation count the plan
  demands, a committed rollback authority, and the plan's baseline still
  in service. If the live metrics regress past bounds computed from the
  metric's own history, a circuit breaker escalates step by step —
  advisory, finding, quarantine, operator stop — and rollback restores the
  previous model in one step with the causing evidence recorded. A
  quarantined candidate can only return with new evidence committed after
  the quarantine; resubmitting old evidence does nothing.
- **Composition.** The model-improvement driver follows the same structure
  as the policy loop. Improvement targets come from admitted evaluation
  results. Donor components are found through an exact index over
  normalized component descriptors. For each candidate seam, a linear
  least-squares fit over recorded activations measures how well a simple
  adapter could connect the two models; that residual is checked against
  measured bridge results before it is trusted. Realization initializes
  the adapter from that fit and trains only the adapter — both models
  stay frozen, and the result proves their bytes did not change. A
  composite is selected only if no fitness dimension got worse and at
  least one got better; averaging across dimensions is not allowed.
  Promotion requires ablation evidence through the existing promotion
  check: the composite must lose its measured gains when the donor input
  is dropped or shuffled. The run stops at an attempt budget and failure streak
  computed from its own history. Hit rate, fitness per compute, and
  adapter training cost are aggregated into a learning curve, which is
  the record the autonomy ratchet reads.

## What remains before autonomous RSI

The sole live backlog and dependency queue is [docs/plan.json](docs/plan.json).
Imported branch plans, design records, staged-surface inventory, and historical
Git snapshots are evidence only; distinct requirements from them are mapped
into that canonical plan instead of maintained as parallel checklists.

Human steering remains a supported operating mode throughout: human and model
steering submit goals through the same bounded interface, and deterministic
policy retains admission, evaluation, activation, rollback, resource budgets,
saturation detection, and stop conditions.

## Scope

Overgo includes a broad model-engineering runtime because an RSI harness must
be able to execute and evaluate the systems it changes. The same inference,
training, evaluation, multimodal processing, agent, automation, and distributed
execution capabilities are also exposed for direct human use. These are two
consumers of one control and evidence system, not separate platforms.

The current host target is Windows amd64 with Go 1.26 and an NVIDIA CUDA
driver. The runtime uses the Windows ABI without cgo. Public interfaces may
change while the control and evidence model is tightened.

Current release designation: **v0.1.1**.

## Quick start

Requirements:

- Windows amd64
- Go 1.26
- NVIDIA CUDA driver
- A model artifact supported by an available configuration

Start the operator workbench:

```bat
overgo_gui.bat "D:\models\model.gguf"
```

Or start the server directly:

```bash
go run ./cmd/server -listen 127.0.0.1:8080 D:/models/model.gguf
```

Open `http://127.0.0.1:8080/`.

Run the hermetic verification lane:

```bash
go run ./cmd/compatibility -check
go run ./cmd/test-lane ./...
```

Real-device and installed-model checks are separate:

```bash
go run ./cmd/device-lane
go run ./cmd/smoke-lane
go run ./cmd/race-lane
```

An unavailable model or device is not counted as a successful verification.

## References

- [API manifest](docs/API_MANIFEST.md) — the generated public surface: every
  command binary, every typed store document contract, the UTCP tool
  protocol, and content-digested automation authorities; canonical JSON at
  [docs/api_manifest.json](docs/api_manifest.json); regenerate with
  `go run ./cmd/api-manifest -update`, verify with `-check`
- [Media capability report](docs/MEDIA_REPORT.md) — every media activation
  with its measured verifier run, phase decomposition, peak device bytes,
  and generated samples beside the requests that produced them; regenerate
  with `go run ./cmd/compatibility -update-media`
- [Model prototypes with verified models, nested](docs/model_compatibility.json) —
  structured JSON: each specific model's inference and training verification
  summaries inside the prototype it belongs to; regenerate with
  `go run ./cmd/compatibility -update-models`
- [Compatibility and verified capabilities](docs/COMPATIBILITY.md)
- [Training compatibility](docs/TRAINING_COMPATIBILITY.md)
- [Machine-readable compatibility data](compatibility.json)
- [Staged surface declarations](docs/staged_surface.json) — reviewed exported
  symbols whose production consumers are deliberately deferred, each with the
  trigger that retires it
- [Current development plan](docs/plan.json)
- [Development and verification doctrine](skill.md)
- [Software bill of materials](SBOM.cdx.json)

Completed development history is retained in Git. The active plan is intended
to contain future work and verification requirements rather than duplicate the
commit history.
