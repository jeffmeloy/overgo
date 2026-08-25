# Overgo

Overgo is an experimental systems platform for reliable recursive
self-improvement (RSI) of model-driven software agents.

The central design rule is that model cognition proposes work while
deterministic code controls state, authority, execution, verification, and
recovery. A model may interpret a problem or generate a candidate patch, but
acceptance still depends on measured evidence and executable policy.

At present, an operator supplies much of the steering: goals, constraints,
priorities, and decisions about what to investigate next. The graphical
workbench is an external interface for that steering and for observing the
system; it is not the RSI controller. The target is to let a model propose the
next steering action from durable measurements and state while the same
deterministic control plane continues to admit, evaluate, activate, or reject
it.

Overgo is not a complete RSI system. It currently provides much of the model,
agent, automation, and evidence infrastructure required to build and study one.

![Overgo system architecture](docs/assets/overgo-platform-architecture-scientific.png)

[Structured figure definition](docs/assets/overgo_graphic.json)

## Objective

Reliable RSI requires more than repeatedly asking a model to modify its own
code. The surrounding system must make each attempt reproducible, bounded, and
falsifiable.

```text
steering: operator now, model target
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
| Verification | Plan-driven gates, structural source analysis, host/device comparisons, model-specific evidence, and compatibility records |
| Operator steering | External workbench for goals, measurement review, intervention, exceptional approvals, and rollback |
| Interfaces | Command line, HTTP APIs, scheduled jobs, and an external operator workbench over shared backend state |

The workbench exposes chat and media, agent sessions, workflow graphs,
provenance, models and datasets, training, evaluation, artifacts, runtime
activity, and exceptional approvals. It projects backend records and submits
bounded actions; the browser does not own execution state or participate in
the autonomous decision loop.

## Current state

The deterministic harness is substantially implemented, but evidence maturity
varies by capability.

| Requirement | State |
| --- | --- |
| Exact identities for code, models, data, tools, policies, and runs | Implemented |
| Bounded and recoverable workflow execution | Implemented |
| Durable authorization and mutation receipts | Implemented |
| Independent host, CUDA, integration, and model verification lanes | Implemented; artifact coverage varies |
| Exact training checkpoints and fail-closed resume | Implemented for supported training paths |
| Agent and automation lifecycle management | Implemented |
| Measurement of automation effectiveness | Partial |
| Durable cross-run measurement history for steering | Partial |
| Controlled comparison of competing automation strategies | Not complete |
| Evidence-gated promotion of automation policies | Not complete |
| Model-proposed steering from accumulated evidence | Not complete |
| Closed recursive policy-improvement loop | Not complete |

Architecture support and artifact verification are separate claims. Shared
components can express more model types than are installed and tested on the
current hardware. Compatibility records therefore identify the exact artifact,
configuration, environment, and verification command behind each claim.

## What remains before RSI

### 1. Close the harness

- Keep one deterministic authority for planning, execution, policy decisions,
  evidence, and recovery.
- Bound retrieval construction and search as well as request execution.
- Derive verification from changed symbols and interfaces, with run-by-default
  behavior when independence cannot be proven.
- Maintain false-negative corpora for gate selection and mutation tests for the
  automation itself.
- Finish race, restart, browser, device, and release verification across the
  integrated agent and automation paths.

### 2. Measure the automation

Every attempt needs an exact identity for its model, prompt, context,
retrieval, tools, strategy, source state, resource budget, and expected result.
The durable evidence store must compare prediction with outcome: defects
found, tests added, regressions introduced, code removed, coverage changed,
elapsed time, and compute used. Measurements must remain queryable across runs,
strategies, and code revisions rather than existing only as logs.

### 3. Run controlled strategy experiments

Competing strategies should operate in isolated worktrees against the same
task and baseline. Independent evaluations should select among the candidates.
Failures and counterexamples must remain durable inputs so later attempts do
not repeat known mistakes.

### 4. Promote better policies

Planning, retrieval, repair, verification, and stopping policies need a staged
lifecycle: declared, experimentally useful, repeatedly verified, and active.
Promotion must reference measured evidence and retain a rollback target.

### 5. Transfer steering, then close the recursive loop

The operator currently interprets results and chooses the next goal. A model
can take over that steering only when the measurement history can support
comparable retrieval, predicted benefit and cost, explicit uncertainty, and a
falsifiable next experiment. The model then proposes goals and constraints
through the same bounded steering interface; deterministic policy retains
admission, evaluation, activation, rollback, resource budgets, saturation
detection, and stop conditions.

## Scope

Overgo includes a broad model-engineering runtime because an RSI harness must
be able to execute and evaluate the systems it changes. Inference, training,
evaluation, multimodal processing, agents, automation, and distributed
execution are supporting experimental infrastructure rather than separate
product claims.

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

- [Compatibility and verified capabilities](docs/COMPATIBILITY.md)
- [Training compatibility](docs/TRAINING_COMPATIBILITY.md)
- [Machine-readable compatibility data](compatibility.json)
- [Current development plan](docs/plan.json)
- [Development and verification doctrine](skill.md)
- [Software bill of materials](SBOM.cdx.json)

Completed development history is retained in Git. The active plan is intended
to contain future work and verification requirements rather than duplicate the
commit history.
