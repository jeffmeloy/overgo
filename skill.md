---
name: overgo-iteration
description: Guide Overgo implementation and repository doctrine toward recursive self-improvement, robustness, and efficiency. Use for development work and explicit continue/resume/loop requests; autonomous continuation stays within the user's requested scope.
---

# Intent

Build one Go-native system that serves, trains, evaluates, and composes models
on consumer hardware. Recursive self-improvement means using measured outcomes
to improve both task capability and the mechanisms that propose, execute, and
evaluate later work.

Humans or models propose changes. Operators set goals, constraints, budgets,
and stop conditions. Executable policy admits work and controls activation;
OvergoDB retains the state and results needed for the next iteration.

Build toward autonomous selection and execution of the next useful experiment
within that operator-defined scope. Make observation-to-proposal feedback a
durable part of the existing driver, with implementation tracked in the plan.

## Goals

| Goal | What to improve |
| --- | --- |
| Robustness | Correct results, preserved data, predictable failure behavior, exact resume, reliable cancellation, and recovery with fewer operator interventions |
| Efficiency | Time and resources per useful outcome, adapting to available RAM, VRAM, CPU and GPU capacity while reducing repeated work and maintenance cost |
| Capability | Quality, supported tasks, model coverage, and training results on fixed external workloads |

Evaluate these goals together. Preserve required correctness and quality while
reducing cost. Prefer a simpler implementation when it retains behavior and
makes execution or recovery easier to understand.

## Autonomous feedback

Use outcomes to identify improvements in capability, robustness, and efficiency,
then generate the next proposal. Identify the limiting capability, recurring
failure, or avoidable cost and explain how the proposed change addresses it.
Apply this reasoning to both the model system and its improvement process;
carry lessons from successful, rejected, and failed attempts into the next choice.

Robustness determines how independently the loop can operate. Efficiency guides
what it attempts. Increase autonomy as recovery becomes reliable and useful
outcomes require less compute, repeated work, and operator intervention;
remain within operator-set scope and resource ceilings.

- Turn completed runs, regressions, recovery failures, and recurring costs into
  durable follow-up obligations. Coalesce related observations and reconsider
  the next action at meaningful boundaries, while respecting plan dependencies
  and active work. Proposal generation should not depend on an empty plan or
  another human prompt.
- Assemble focused context from the objective, relevant measurements, prior
  attempts, existing implementation owners, and remaining budget. Generate
  candidates with a causal hypothesis, expected benefit, affected components,
  estimated cost, acceptance comparison, and rollback. A targeted measurement
  or a decision to take no action is also a valid outcome.
- Apply existing admission rules and select eligible candidates using recorded
  outcomes and total expected cost. Retain a bounded exploration allowance for
  unfamiliar approaches. Rejection feeds the next decision; unavailable
  resources become waiting obligations with explicit resumption conditions,
  while independent eligible work continues.
- Persist the observation position, candidate disposition, outstanding checks,
  and cumulative budget. Resume the unfinished transition after interruption
  without duplicating proposals, completed acquisitions, or budget allowances.
- Compare predicted benefit and cost with actual results. Improve retrieval,
  prompts, candidate ranking, and experiment selection through the same cycle.
  Keep each experiment's acceptance criteria fixed and evaluate changes to the
  evaluation policy against an independently retained task set and judge.

Connect existing loop, run-record, candidate, and plan owners. Validate a
complete observation-to-next-proposal cycle, including rejection, duplicate
events, restart, result reuse, and budget exhaustion. Compare task improvement,
total cost, recovery success, repeated acquisition, and operator interventions
on the same workloads and budgets.

## Working loop

1. Read `git status --short`, the relevant plan row, the owning code, and its
   stored results. Re-read touched files after interruption or concurrent work;
   preserve changes made by others.
2. Follow the user's task. During an authorized campaign, obtain the dispatched
   step with `go run ./cmd/plan -prompt` and respect its dependencies, lane
   ownership, acceptance criteria, and stop conditions.
3. State the change, the outcome it should improve, and the comparison that
   will decide it. Use a bounded probe when uncertainty could change the design.
4. Fix the cause in its existing owner. Make a coherent change, migrate affected
   callers, and remove displaced implementation as part of the same work.
5. Run the required checks, preserve their results, and resolve failures.
   Checkpoints retain unfinished obligations across retries and restart.
6. Commit campaign work through
   `go run ./cmd/gate -plan <item>/<step> -message-file <file> -paths <csv>`.
   The gate owns acceptance and plan advancement; do not bypass it with a raw
   commit. Record the reason, measured results, and remaining work.

Under a continue/resume/loop request, return to dispatch after a completed
slice. A checkpoint alone does not complete the campaign. Respect operator
stops and declared budgets; use the plan's stop mechanism for a required
external prerequisite or irreversible action. A bounded user task ends when
that task is complete.

## Robustness

- Preserve exact identities for source, models, data, recipes, policies, and
  execution environments. Reuse or resume only when the relevant inputs and
  contracts still match.
- Persist completed work and outstanding obligations. Make publication and
  retries idempotent; recover from durable state after interruption. A failed
  result remains a failed result, even when recording it requires a retry.
- Give resource ownership and concurrency one implementation. Bound worker
  lifetimes; handle cancellation, errors, process death, and release. Use the
  existing OS lock owner for contention; do not infer abandonment from elapsed
  time or steal another process's lock.
- Keep model, dataset, and source checkpoint artifacts immutable. Publish new
  outputs through their lifecycle owners. Validate relocation and rollback
  before retiring an original; preserve content identity and provenance.
- Keep execution, verification, and activation distinct. Retain predecessors
  for rollback and test failure and recovery paths alongside successful runs.
  Missing inputs, invalid state, and partial execution must remain explicit.
- Use the repository's command guard and mutation authorization. Repair must
  preserve evidence and concurrent work; it must not silently reset state,
  change semantics, or weaken acceptance to obtain a pass.

## Efficiency

- Measure complete attempts, including proposal generation, failures, waiting,
  acquisition, verification, recovery, and finalization. Report elapsed time
  separately from summed parallel durations. Count repeated model loads,
  executed and reused checks, outstanding work, and operator interventions
  when they drive the cost.
- Reuse valid artifacts, analyses, checkpoints, and independent check results.
  Reacquire only what changed inputs or a demonstrated defect invalidates.
  Check selection and reuse must follow the same resolved dependencies.
- Run cheap checks before expensive work. Scope verification through existing
  dependency owners; unresolved dependencies retain the broader required
  scope. Changes to verification policy must pass the existing acceptance
  criteria before reduced scope is adopted.
- Share resources when independent work fits and ownership permits it.
  Serialize conflicting mutations and measurements that need isolation.
  CPU work holds no GPU reservation. Add concurrency, pooling, caching, or
  fusion for measured benefit under the resource policy below.
- Repetition triggers an ownership review. Extend an existing command or
  workflow when that removes recurring work. Add a new mechanism only for a
  demonstrated gap, with a comparison of its benefit and ongoing cost.
- Compare simplification, reuse, and direct measurement before adding
  abstractions. Remove unused flags, wrappers, state, and exports with their
  callers. An inconclusive optimization needs a new hypothesis or a stop,
  not repeated runs until a favorable result appears.

## Scale to available compute

Resolve usable host RAM, per-device VRAM, effective CPU capacity, and GPU
compute capabilities through the existing resource owners. Account for current
load, other reservations, and operator limits; installed capacity alone is not
an execution budget. Avoid assumptions about a particular machine, core count,
GPU count, or memory size.

Derive placement, batch and chunk sizes, worker counts, cache residency, and
transfer plans from actual workload dimensions, measured costs, and available
resources. Include transient and retained allocations, host/device transfers,
and contention. Justify reserve margins from measurements or declared limits;
an arbitrary fraction of RAM or VRAM is still a magic number.

Scale down through supported streaming, tiling, and bounded concurrency; scale
up when additional resources improve measured outcomes. Recheck capacity at
admission and replan through the existing owner when conditions change. Preserve
the numerical contract, checkpoint identity, and cumulative budget; record the
resolved execution choices. If no supported plan fits, retain a resumable
waiting obligation or report the limit. Verify constrained and larger resource
profiles, including contention and cancellation.

## Derived values and justified assumptions

Actively eliminate magic numbers and unnecessary literals in implementation,
automation, and evaluation. Derive dimensions, thresholds, tolerances, sample
requirements, optimizer settings, timeouts, and resource choices from artifact
declarations, runtime observations, or mathematical constraints. A named
constant, configuration flag, or fitted correction does not by itself resolve
an arbitrary assumption; derive the quantity or remove the mechanism needing it.

Retain literal values when they express exact mathematical identities, external
format or ABI requirements, or an explicitly justified policy. Record their
source, applicable scope, and validation in the owning contract. Necessary
conventions and unresolved assumptions belong in the existing decision or
closure authority with a reason and a condition for revisiting them. Keep
caller controls focused on goals, data, and resource limits.

Prefer methods with minimal assumptions about data shape, distribution, and
geometry. Derive tensor dimensions and sequence lengths from declarations and
inputs; validate required layouts instead of embedding model-specific shapes.
Do not assume Gaussianity, independence, stationarity, finite variance, or a
particular sample-size rule without justification for the actual observations.
Choose estimators and uncertainty methods whose assumptions fit the data;
robust or nonparametric methods still require their own assumptions to be checked.

Treat Euclidean distance, linear interpolation, inner-product similarity, and
isotropic noise as modeling choices that need justification from the declared
model or the representation and task. Storing data in vectors does not establish
Euclidean geometry. Use the metric and operations the domain supports; compare
alternatives where the choice affects results. Preserve model-defined numerical
operations while making their assumptions explicit.

## Implementation

Go owns runtime behavior and orchestration; CUDA kernels enter through the
kernel manifest. Keep runtime code free of cgo. Use Go or bash for repository
automation and keep development scratch under `tmp/`; `bin/` is for executables.

Port capabilities through shared typed contracts and recipe-selected consumers.
Start from an existing verified implementation when available: llama.cpp
provides pinned behavioral references; adaptive_new provides source capabilities.
Preserve source commits, licenses, artifact identities, and reference outputs.
Keep model-family facts in artifacts and recipes rather than new executor branches.

Use small cohesive owners, clear error and cancellation paths, and reusable
hot-path buffers. Add interfaces and shared abstractions where real consumers
need them. Apply `gofmt`, `go vet`, and the required repository checks to code
changes. Numerical or gradient changes require the affected reference comparisons.

## Verification

Freeze acceptance inputs and criteria before candidate generation. A candidate
that changes evaluation policy cannot redefine its own judge. Compare the same
tasks, artifacts, protocols, and total budgets; use ablations where needed to
attribute a gain. An aggregate improvement cannot hide a required regression.

Exercise relevant variations in shape, distribution, geometry, and resource
availability. Check assumptions and invariances against the declared contract.
Keep negative and inconclusive results; report denominators, exclusions,
measurement scope, and the resolved compute environment.

Tests exercise observable contracts and failure modes using controlled inputs.
Missing, skipped, incomplete, and zero-match checks do not count as passes.
Resolve required prerequisites or report the exact outstanding work.

Use the existing lanes: `cmd/test-lane`, `cmd/device-lane`, `cmd/smoke-lane`,
`cmd/race-lane`, and `cmd/webui-lane`. The gate owns change-specific selection,
compatibility, manifest, and other required checks.

## Sources of truth

| Surface | Owner |
| --- | --- |
| [skill.md](skill.md) | Repository intent and operating principles; the sole root agent-instruction document |
| [docs/plan.json](docs/plan.json) | Current priorities, dependencies, lane ownership, acceptance, and outstanding work |
| OvergoDB | Artifacts, lineage, runs, decisions, findings, and receipts; query through `cmd/overgodb-query` |
| [compatibility.json](compatibility.json) | Model and capability verification contracts |
| [kernels/manifest.json](kernels/manifest.json) | Kernel ABI and source/binary identities |
| [API manifest](docs/api_manifest.json) | Public commands, routes, and document contracts |

Use existing owners rather than parallel plans, ledgers, schedulers, or
recovery mechanisms. Campaign-specific constraints remain in the active plan;
this skill does not override them. Keep durable technical contracts in docs,
measurements in the store, and development history in Git.

Communicate concisely: outcome, material measurements, checks run, and any
remaining obligation. Keep instructions focused on decisions that improve
capability, robustness, or efficiency.
