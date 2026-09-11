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

## Goals

| Goal | What to improve |
| --- | --- |
| Robustness | Correct results, preserved data, predictable failure behavior, exact resume, reliable cancellation, and recovery with fewer operator interventions |
| Efficiency | Time and resources per useful outcome: elapsed time, compute, peak and retained memory, I/O, repeated acquisition, and verification and maintenance cost |
| Capability | Quality, supported tasks, model coverage, and training results on fixed external workloads |

Evaluate these goals together. Preserve required correctness and quality while
reducing cost. Prefer a simpler implementation when it retains behavior and
makes execution or recovery easier to understand.

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

- Measure complete attempts, including failures, waiting, acquisition,
  verification, and finalization. Report elapsed time separately from summed
  parallel durations. Count repeated model loads, executed and reused checks,
  outstanding work, and operator interventions when they drive the cost.
- Reuse valid artifacts, analyses, checkpoints, and independent check results.
  Reacquire only what changed inputs or a demonstrated defect invalidates.
  Check selection and reuse must follow the same resolved dependencies.
- Run cheap checks before expensive work. Scope verification through existing
  dependency owners; unresolved dependencies retain the broader required
  scope. Changes to verification policy must pass the existing acceptance
  criteria before reduced scope is adopted.
- Share resources when independent work fits and ownership permits it.
  Serialize conflicting mutations and measurements that need isolation.
  CPU work holds no GPU reservation; derive capacity from the actual device
  and workload. Add concurrency, pooling, caching, or fusion for measured benefit.
- Repetition triggers an ownership review. Extend an existing command or
  workflow when that removes recurring work. Add a new mechanism only for a
  demonstrated gap, with a comparison of its benefit and ongoing cost.
- Compare simplification, reuse, and direct measurement before adding
  abstractions. Remove unused flags, wrappers, state, and exports with their
  callers. An inconclusive optimization needs a new hypothesis or a stop,
  not repeated runs until a favorable result appears.

## Implementation

Go owns runtime behavior and orchestration; CUDA kernels enter through the
kernel manifest. Keep runtime code free of cgo. Use Go or bash for repository
automation and keep development scratch under `tmp/`; `bin/` is for executables.

Port capabilities through shared typed contracts and recipe-selected consumers.
Start from an existing verified implementation when available: llama.cpp
provides pinned behavioral references; adaptive_new provides source capabilities.
Preserve source commits, licenses, artifact identities, and reference outputs.
Keep model-family facts in artifacts and recipes rather than new executor branches.

Derive model geometry, resource choices, optimizer settings, and thresholds
from artifact declarations, measurements, or mathematical constraints. Keep
caller controls focused on goals, data, and resource limits. Record necessary
conventions and unresolved assumptions in the existing decision or closure
authority, with a reason and a condition for revisiting them.

Use small cohesive owners, clear error and cancellation paths, and reusable
hot-path buffers. Add interfaces and shared abstractions where real consumers
need them. Apply `gofmt`, `go vet`, and the required repository checks to code
changes. Numerical or gradient changes require the affected reference comparisons.

## Verification

Freeze acceptance inputs and criteria before candidate generation. A candidate
that changes evaluation policy cannot redefine its own judge. Compare the same
tasks, artifacts, protocols, and total budgets; use ablations where needed to
attribute a gain. An aggregate improvement cannot hide a required regression.

Choose summaries that fit the data and sample size. Do not assume Gaussian,
independent, or stationary observations without justification. Keep negative
and inconclusive results; report denominators, exclusions, and measurement scope.

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
| [API manifest](docs/API_MANIFEST.md) | Public commands, routes, and document contracts |

Use existing owners rather than parallel plans, ledgers, schedulers, or
recovery mechanisms. Campaign-specific constraints remain in the active plan;
this skill does not override them. Keep durable technical contracts in docs,
measurements in the store, and development history in Git.

Communicate concisely: outcome, material measurements, checks run, and any
remaining obligation. Keep instructions focused on decisions that improve
capability, robustness, or efficiency.
