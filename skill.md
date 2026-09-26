---
name: overgo-iteration
description: Guide Overgo development toward recursive self-improvement, robustness, and efficiency. Use for implementation and explicit continue/resume/loop requests; keep autonomy within user scope.
---

# Intent

Build one Go-native system to serve, train, evaluate, and compose models on
consumer hardware. Use measured outcomes to improve capability and the process
that proposes, executes, and evaluates work.

Humans or models propose. Operators set goals, constraints, budgets, and stops.
Executable policy admits work and controls activation. OvergoDB retains state
and results. Build durable observation-to-proposal feedback through the existing
driver; track implementation in the plan.

## Goals

| Goal | Improve |
| --- | --- |
| Robustness | Correctness, data preservation, predictable failure, exact resume, cancellation, recovery, fewer interventions |
| Efficiency | Cost per useful outcome, capacity adaptation, less repeated work and maintenance |
| Capability | Quality, task and model coverage, training results on fixed external workloads |

Optimize together. Preserve required correctness and quality. Prefer simpler
implementations with clearer execution and recovery.

## Fewer components

The repository should shrink while its capability grows.

- Make each change remove something: one change that retires two mechanisms
  beats two changes. Fold parallel commands, tables, and paths into one owner
  (per-family trainers become `cmd/train -route` entries; lanes, parity tools,
  and file tools follow).
- Refactor the pattern that causes growth, not only its latest instance.
- Enforce by construction. A rule that matters becomes code the gate runs; this
  document states intent and never substitutes for a check. A ledger that
  records objects exists only until the code makes invalid objects impossible
  to add.
- Classify once, at the owner, from typed facts. No regex over structured data;
  no second copy of a classification.
- Capabilities are not dead code. A trainer, server path, or converter that
  only tests reach is unwired: give it an entry point. Product capability
  shrinks by deduplication; tooling that only checks or records the repo may
  also be deleted.

## Autonomous feedback

Robustness governs autonomy; efficiency guides selection. Stay within operator
scope and resource ceilings. Learn from successful, rejected, and failed work.

- Coalesce runs, regressions, recovery failures, and recurring costs into durable
  follow-up obligations. Reconsider at meaningful boundaries; respect active
  work and dependencies. Do not wait for an empty plan or another prompt.
- Retrieve objective, measurements, prior attempts, owner, and remaining budget.
  Propose a causal hypothesis, benefit, affected components, cost, acceptance
  comparison, and rollback. A targeted measurement or no action is valid.
- Rank eligible candidates by recorded outcomes and total expected cost; retain
  bounded exploration. Feed rejection into the next decision. Give missing
  resources explicit resumption conditions; continue independent eligible work.
- Persist observation position, candidate disposition, outstanding checks, and
  cumulative budget. Resume unfinished transitions without duplicating proposals,
  acquisitions, or budget allowances.
- Compare predictions with results. Improve retrieval, prompts, ranking, and
  experiment selection through the same cycle. Freeze each experiment's
  acceptance; evaluate policy changes against an independent task set and judge.

## Working loop

1. Read `git status --short`, the plan row, owning code, and stored results.
   Re-read after interruption or concurrent edits; preserve others' work.
2. Follow user scope. For authorized campaigns, take the next row from
   `go run ./cmd/plan -next`; respect dependencies, lane ownership, acceptance,
   and stops.
3. State change, expected outcome, and deciding comparison. Probe only
   uncertainty that could change the design.
4. Fix the cause in its owner. Migrate callers; remove displaced implementation.
5. Run required checks, retain results, resolve failures. Preserve unfinished
   obligations across checkpoints, retries, and restart.
6. Land with `go run ./cmd/loop -land <item>/<step> -message-file -`, the
   message on standard input stating `Cause:` and `Predicted effect:`. It
   claims, preflights, gates, confirms from Git, and awaits the deferred lanes;
   the gate owns acceptance and removes the completed row. No raw commits.
   A lane merge is `git merge --no-ff --no-commit <branch>`, then
   `go run ./cmd/gate -merge -plan-projection first-parent-target
   -merge-source-store <lane store> -plan merge-<head12>/do -message-file <f>`.

Continue/resume/loop requests return to dispatch after each landing. Landings do
not complete campaigns. A user prompt is answered, then work continues; only an
explicit user stop (`go run ./cmd/plan -stop user-stop:<detail>`) pauses. Use
plan stops for required external prerequisites or irreversible actions.

### Plan and SQA

[docs/sqa_findings.json](docs/sqa_findings.json) is an independent review. After
the final `docs/plan.json` edit, the implementation agent replaces its
`implementation_review`, bound to the saved plan (`agent`, `reviewed_at`,
`findings_reviewed_at`, `plan_path`, `plan_sha256`), with one comment per open
or deferred finding: `finding_id`, `disposition` (`planned`, `deferred`,
`declined`, `disputed`, `resolved`), a concise `comment`, and `plan_steps` or
`evidence`. Deferral names its reconsideration condition; resolution names its
evidence.

The lead owns adoption, scope, and sequence; SQA rankings are advisory. Answer a
high-priority `user_finding` substantively. Declining an approach does not
refute its evidence. The reviewer never authors the responses. Keep one current
file; re-read before writing and preserve the other writer's fields.

### Campaign supervision

`go run ./cmd/loop` supervises unattended work from its machine-local
configuration (`.overgo-runtime/loop.json`); single dispatch and gate commands
start no supervision. The launcher owns one worktree's process lock and passes a
fresh session to workers; never copy its token into unrelated processes. Worker
exit triggers deferred-validation recovery, then redispatch. Owner PID:
`.overgo-runtime/loop_supervisor.json`. A locator without its live lock
requires restart.

## Robustness

- Bind source, model, data, recipe, policy, and environment identities exactly.
  Reuse/resume only matching relevant inputs and contracts.
- Persist results and obligations. Make publication/retries idempotent; recover
  from durable state. Retrying a failed-result write cannot turn failure into pass.
- Use one resource/concurrency owner. Bound workers; handle cancellation, errors,
  process death, and release. Use existing OS locks; never infer abandonment
  from elapsed time or steal another process's lock.
- Keep model, dataset, and source checkpoints immutable. Publish through lifecycle
  owners. Validate relocation and rollback before retiring originals; preserve
  content identity and provenance.
- Separate execution, verification, activation. Retain rollback predecessors.
  Test failure/recovery alongside success; expose missing, invalid, partial state.
- Honor command guards and mutation authorization. Preserve evidence and
  concurrent work. No silent resets, semantic changes, or weakened acceptance
  to obtain a pass; a failing check is fixed, never relabeled.

## Efficiency

- Measure whole attempts: proposal, failure, wait, acquisition, verification,
  recovery, finalization. Separate elapsed time from summed parallel durations.
  Count repeated loads, executed/reused checks, outstanding work, and interventions
  where material.
- Reuse valid artifacts, analyses, checkpoints, and independent check results.
  Reacquire only invalidated work. Selection and reuse share resolved dependencies.
- Drive selection, ordering, reuse, recovery, and reporting from shared typed
  facts: AST/types, parsed commands, typed results, declared capabilities.
  Normalize external text at its boundary; retain unknowns, independent
  selection reasons, and domain acceptance.
- Feed true dependencies, effects, resource demand, reuse conditions, and measured
  cost into the existing scheduler. Run cheap informative checks first where
  dependencies permit; dispatch independent work within capacity; resume waiting
  work on state change. Classification must reduce total work.
- Scope verification through dependency owners. Unknown dependencies retain
  required broader scope. Verify policy changes against existing acceptance
  before adopting reductions.
- Share resources where capacity and ownership permit. Serialize conflicting
  mutations and measurements requiring isolation. CPU work holds no GPU claim.
  Add concurrency, pooling, caching, or fusion only for measured benefit.
- Extend existing workflows; add mechanisms only for demonstrated gaps with
  justified ongoing cost. Remove unused flags, wrappers, state, exports, and
  callers. Inconclusive optimization requires a new hypothesis or stop; never
  rerun merely to obtain a favorable result.

## Scale to available compute

- Resolve usable RAM, per-device VRAM, effective CPU capacity, and GPU capabilities
  through resource owners. Include current load, reservations, operator limits.
  Installed capacity is not an execution budget; avoid machine-specific defaults.
- Derive placement, batches, chunks, workers, residency, and transfers from
  workload dimensions, measurements, and availability. Include transient/retained
  allocations, transfers, contention. Justify reserves from measurements or
  declared limits; arbitrary RAM/VRAM fractions remain magic numbers.
- Scale down through supported streaming, tiling, bounded concurrency; scale up
  for measured benefit. Recheck admission capacity; replan through its owner.
  Preserve numerical contracts, checkpoint identity, cumulative budget; record
  resolved choices. If nothing fits, retain a resumable wait or report the limit.
- Verify constrained and larger profiles, contention, and cancellation.

## Derived values and justified assumptions

- Eliminate arbitrary literals in implementation, automation, and evaluation.
  Derive dimensions, thresholds, tolerances, sample requirements, optimizer
  settings, timeouts, and resource choices from declarations, observations, or
  mathematical constraints. Naming, configuring, or fitting a value does not
  justify it; derive it or remove the mechanism needing it.
- Retain exact identities, external format/ABI requirements, and justified policy.
  Record source, scope, validation in the owning contract. Record necessary
  conventions/unresolved assumptions with rationale and reconsideration condition
  in existing decision/closure authority. Keep caller controls on goals, data,
  and resource limits.
- Minimize distribution, shape, and geometry assumptions. Derive dimensions and
  sequence lengths; validate layouts. Justify Gaussianity, independence,
  stationarity, finite variance, and sample-size rules for actual observations.
  Check estimator and uncertainty assumptions, including robust/nonparametric ones.
- Justify Euclidean distance, linear interpolation, inner products, and isotropic
  noise from the model, representation, and task. Vectors do not imply Euclidean
  geometry. Use supported metrics/operations; compare consequential alternatives.
  Preserve model-defined operations; expose their assumptions.

## Implementation

- Go owns runtime, orchestration, and tooling; no runtime cgo, no Python in the
  tree. CUDA enters through the kernel manifest. Scratch lives in `tmp/`,
  executables in `bin/`.
- Port through shared typed contracts and recipe-selected consumers. Prefer
  verified implementations: llama.cpp for pinned behavior; adaptive_new for
  source capabilities. Preserve commits, licenses, artifact identities, outputs.
  Model-family facts belong in artifacts/recipes, never in model-specific code.
- Keep owners cohesive, errors/cancellation clear, hot-path buffers reusable.
  Add interfaces/abstractions for real consumers. Apply `gofmt`, `go vet`, and
  required checks to code changes; affected reference comparisons to numerical
  or gradient changes.

## Verification

- Freeze acceptance before candidates. Evaluation-policy changes cannot redefine
  their own judge. Compare identical tasks, artifacts, protocols, total budgets;
  use ablations for attribution. Aggregate gains cannot hide required regressions.
- Exercise relevant shape, distribution, geometry, resource variations. Check
  declared assumptions/invariances. Retain negative/inconclusive results; report
  denominators, exclusions, scope, resolved compute environment.
- Test observable contracts and failure modes with controlled inputs. Missing,
  skipped, incomplete, zero-match checks earn no pass. Resolve prerequisites or
  report exact outstanding work.
- Prove real models one at a time, smallest first. Parity goldens keep the
  upstream output itself in Git with its provenance (repository, commit,
  checkpoint, input); comparing overgo with its own earlier output stacks
  tolerances.
- Use `cmd/test-lane`, `cmd/device-lane`, `cmd/smoke-lane`, `cmd/race-lane`,
  `cmd/webui-lane`. The gate owns change-specific selection, compatibility,
  manifests, and required checks.

## Sources of truth

| Surface | Owner |
| --- | --- |
| [skill.md](skill.md) | Intent and operating principles; sole root agent-instruction document |
| [docs/plan.json](docs/plan.json) | Priorities, dependencies, lanes, acceptance, outstanding work |
| OvergoDB | Artifacts, lineage, runs, decisions, findings, receipts; query via `cmd/overgodb-query` |
| [compatibility.json](compatibility.json) | Model/capability verification contracts |
| [kernels/manifest.json](kernels/manifest.json) | Kernel ABI, source/binary identities |
| [API manifest](docs/api_manifest.json) | Public commands, routes, document contracts |

Use existing owners; avoid parallel plans, ledgers, schedulers, recovery paths.
The active plan retains campaign constraints. Durable contracts live in docs,
measurements in the store, development history in Git.

Report outcome, material measurements, checks, and remaining obligations.
