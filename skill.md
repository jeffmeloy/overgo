---
name: overgo-iteration
description: Operating principles for Overgo development: ambition, recursive self-improvement, robustness, efficiency. Use for implementation and explicit continue/resume/loop requests; stay within user scope.
---

# Intent

One Go-native system serves, trains, evaluates, composes models on consumer
hardware.

- Purpose: recursive self-improvement through own inference and training,
  unattended.
- Goal: improvement a user sees. Scorecard (`go run ./cmd/validate -scorecard`):
  `first-output`, `throughput`, `held-out-quality`, `failure-rate`, `load-time`,
  `peak-device-memory`, `setup-steps`.
- Loop: measure, find gap, train or change, judge on held-out data, promote or
  revert. A row that neither closes nor speeds the loop, nor serves inference or
  training, is not filed.
- Progress = promotions, not landings. `go run ./cmd/loop -improve`: measure,
  record the scorecard as held only if held-out quality holds against the newest
  held record, file gaps as rows. `master` takes landings; local `main` is only
  GitHub staging, which nothing reads; push only when asked.
- Humans or models propose; operators set goals, budgets, stops; executable policy
  admits and activates; OvergoDB holds state; the plan holds implementation.
- Be ambitious. Aim each change at capability, performance, development
  velocity or memory use, and take the bold change over the cautious increment: a
  rewritten slow path, a deleted subsystem, a new kernel, a different data layout.
  The framework catches regressions and broken code: the gate (format, vet, rules,
  invariants, tests, lanes), the held-out scorecard and the cycle. Make the
  change, write the test that would have caught its failure, and let the
  framework judge. Do not shrink a change to what can be checked by hand, hedge it
  with flags or shims, or defer it for fear of breaking what the framework
  checks. A refused landing is information, not failure.
- Ambition governs the code, not the operator's boundaries: scope, budgets,
  stops, pushes, downloads and security settings stay theirs (Robustness).
- Robustness, efficiency, capability together. Keep correctness and quality;
  prefer the simpler implementation: ambition is the size of the gain, not the
  size of the change.

## Fewer components

Repository shrinks while capability grows.

- Every change removes something; net production lines flat or falling unless the
  row declares growth; growth that buys capability, speed or memory is declared
  and welcome. One change retiring two mechanisms beats two changes.
- Subtraction first: delete before building; build for the case that exists,
  generalize at the second; existing tools and signals before new ones.
- Consolidate: add a mode or consumer to an existing owner before a new owner.
  Refactor the growth pattern, not its latest instance.
- Enforce by construction: a rule that matters is code the gate runs. A ledger of
  objects lives only until invalid objects are impossible.
- Classify once, at the owner, from typed facts. No regex over structured data.
- Capabilities are not dead code. A path only tests reach is unwired: add its
  route, lane, or command. Delete one only if its input chain is half missing, it
  is record ceremony, it costs now, and a rebuild would redesign it.
- No markdown behavior control: ideas become Go tests or gate rules.

## Autonomous feedback

- Stay inside operator scope and resource ceilings. Robustness governs autonomy;
  efficiency guides selection.
- Fold runs, regressions, failures, recurring costs into durable obligations.
- Propose with measurement, prior attempts, owner, budget: hypothesis, benefit,
  cost, acceptance comparison, rollback. A measurement or no action is valid.
- Rank by recorded outcomes and expected cost; bounded exploration; rejection
  feeds the next decision. Missing resource: explicit resume condition; continue
  independent work.
- Persist position, dispositions, outstanding checks, budget; resume without
  duplicates.
- Compare predictions with results. Freeze acceptance first. Judge policy changes
  on an independent task set.

## Working loop

1. Read `git status --short`, plan row, owner code, stored results. Re-read after
   interruption; preserve others' work.
2. User scope first. Campaign: `go run ./cmd/plan -next`; honor dependencies,
   lanes, acceptance, stops.
3. State change, expected effect, deciding comparison. Probe only uncertainty that
   could change the design.
4. Fix the cause in its owner; migrate callers; delete displaced code.
5. Step verify fails before, passes after. Preflight first:
   `go run ./cmd/gate -plan <item>/<step> -message-file - -preflight` runs the
   gate's cheap stages (format, fix, vet, rules, invariant tests, verify), commits
   nothing. Also `go run ./cmd/validate -claims` after hand edits.
6. Land: `go run ./cmd/loop -land <item>/<step> -message-file -`; message on stdin
   with `Cause:` and `Predicted effect:` naming scorecard metrics (worsening in
   words). Gate writes numbers. Run unpiped; confirm `git log -1` shows
   `Landed-Row:`. No raw commits. Merge: `git merge --no-ff --no-commit <branch>`,
   resolve, land.
7. Open decision: take the recommended default, record `plan -answer`, continue.

- Landing returns to dispatch. Only an explicit user stop pauses
  (`go run ./cmd/plan -stop user-stop:<detail>`; lift `-resume-stop`).
- Prompt: answer, continue. Question gets an answer; instruction gets work.
- One lander on master; others review. Never touch another session's work.
- `go run ./cmd/loop` supervises unattended work (`.overgo-runtime/loop.json`); one
  lock per worktree; never copy its session token.
- Off-scope finding: `go run ./cmd/finding`.

### Gate and cycle

- Gate judges the commit from the tree alone: unit, golden, hermetic, small
  device. Store writes never change its verdict.
- Cycle (`validate -run`) judges models: real-model, live-store, device suites, by
  build tag. Promotion needs them at the promoted commit. Phases: cells, publish,
  lanes; an interrupted cycle resumes at the first unpassed phase when
  `go run ./cmd/loop -improve` runs again.
- A hand step is a system defect: change the system so it does not recur.
- Landing cost = lanes reached + inference-surface guard expiry + pinned-media
  reconciliation. Batch surface moves; land before an evaluation run.
- Native `go test` selects. No time budgets, ratchets, second selector. Pin
  boundaries as table rows.

### Plan and SQA

- `docs/plan.json`: sole dispatch authority. Rationale lines: Serves, Disappears,
  Evidence, Batch.
- `docs/sqa_findings.json`: independent, scheduled, advisory. Lead owns adoption
  and sequence. P1 on the next promotion's path: fix first. Answer high-priority
  `user_finding` entries. Declining an approach does not refute its evidence.
- `implementation_review`: one comment per open finding (`disposition`,
  `comment`, `plan_steps` or `evidence`). Reviewer never writes responses. Re-read
  before writing; keep the other writer's fields.

## Evidence

- Bind source, model, data, recipe, policy, environment identities exactly; reuse
  and resume only on matching inputs and contracts.
- One run serves every obligation with matching identity and contract: one
  record, many readers. A run never judges its own output; its evidence is
  complete by its own contract.
- Unknown provenance keeps its value, earns no compatible evidence.
- Replay carries only bit-identical outputs; evaluate decides.
- Compare identical tasks, artifacts, protocols, budgets. Aggregate gains never
  hide required regressions.

## Robustness

- Persist results and obligations. Idempotent publish and retry; recover from
  durable state. A retried failed write never becomes a pass.
- One resource owner. Bound workers; handle cancellation, error, death, release.
  OS locks only; never infer abandonment from elapsed time or steal a lock.
- Model, dataset, checkpoint inputs immutable; publish through lifecycle owners;
  validate rollback before retiring originals.
- Separate execution, verification, activation; keep rollback predecessors; test
  failure and recovery beside success; expose missing, invalid, partial state.
- Honor guards and authorization. No silent resets, semantic changes, weakened
  acceptance. A failing check is fixed, never relabeled.
- No clock decides failure; signals do. Short timing: `processmeasure.Counter`.
- Never change OS security settings (Defender, UAC); report and ask.

## Efficiency

- Measure whole attempts (proposal, failure, wait, acquisition, verification,
  recovery); elapsed apart from summed parallel time; count repeated loads.
- Reuse valid artifacts, checkpoints, check results; reacquire only what is
  invalidated.
- Selection, ordering, reuse, reporting from typed facts (AST, types, parsed
  commands, declared capabilities). Normalize external text at the boundary.
- Scope verification through dependency owners; unknown dependency keeps broader
  scope.
- Serialize conflicting mutations and isolated measurements; CPU work holds no GPU
  claim. Concurrency, caching, fusion only for measured benefit.
- Remove unused flags, wrappers, state, exports. Inconclusive optimization: new
  hypothesis or stop, never a rerun for a favorable result.
- Cheapest method first: reuse data and checkpoints; post-hoc eval, then free
  training signals, then per-layer norms, then per-step decompositions.

## Compute and derived values

- Resolve RAM, VRAM, CPU, GPU capability through resource owners with load,
  reservations, limits. Installed capacity is not a budget; no machine-specific
  defaults.
- Derive placement, batch, chunk, workers, residency from workload, measurement,
  availability. Scale down by streaming, tiling, bounded concurrency; up for
  measured benefit. Preserve numerical contracts and checkpoint identity. Nothing
  fits: resumable wait or reported limit.
- No arbitrary literals: a number other than 0 or 1 is a named constant with a
  reason (gate rule). Derive thresholds, tolerances, sample sizes, optimizer
  settings from declarations, observations, or math, else delete the mechanism.
  Keep exact identities, external formats, justified policy; record source, scope,
  reconsideration condition in the owning contract.
- Justify distribution, geometry, metric assumptions (Gaussian, independent,
  Euclidean, linear interpolation) per model and task.

## Training

- Muon only; rates derived. Policy `internal/trainingprogram/optimizer_policy.json`:
  rate = P^exponent (-0.6 trained, -0.5 from initialization), momentum from N=30.
  Pass `BaseLearningRate: 0`.
- Optimizer math on device; only scalars cross to host.
- `cmd/train` is the one entry (`-route`, `-objective`). `-freeze-lexical` for
  dense tied-embedding models.
- Quantized formats: native fast path, guarded by held-out quality, over output
  identity with bf16.

## Implementation

- Go owns runtime, orchestration, tooling. No runtime cgo; no Python in the tree
  (goldens are data from the Python original). CUDA through the kernel manifest.
  Scratch in `tmp/`, executables in `bin/`.
- Models, datasets, store, `build/` live in `OVERGO_DATA_ROOT`, outside the module.
- Port through typed contracts and recipe-selected consumers. llama.cpp for pinned
  behavior; `adaptive_new` read-only. Keep commits, licenses, artifact identities.
  Model-family facts live in artifacts and recipes, never model-specific code.
- Cohesive owners; clear errors and cancellation; reusable hot-path buffers;
  interfaces for real consumers. `gofmt`, `go vet`; reference comparison for
  numerical or gradient changes.
- Structured edits: JSON via `cmd/plan` or a Go program; code via exact edits.

## Verification

- Test observable contracts and failure modes with controlled inputs. Missing,
  skipped, incomplete, zero-match checks earn no pass.
- The framework is the safety net, so a bold change ships with the check that
  would have caught its failure. Where the framework has no such check, add it
  (a Go test, an invariant row, a gate rule) instead of reducing the change.
- Vary shape, distribution, resource. Keep negative and inconclusive results;
  report denominators, exclusions, scope, environment.
- Real models one at a time, smallest first. Parity goldens keep upstream output
  with provenance (repository, commit, checkpoint, input); comparing overgo to its
  own earlier output stacks tolerances.
- Tools: `cmd/validate` (`-scorecard`, `-run`, `-claims`, `-tests`, `-smoke`,
  `-parity`, `-receipt`), `cmd/device-lane`, `cmd/webui-lane`.

## Sources of truth

| Surface | Owner |
| --- | --- |
| [skill.md](skill.md) | Intent and principles; sole root agent-instruction document |
| [docs/plan.json](docs/plan.json) | Doctrine, priorities, dependencies, acceptance, open work |
| OvergoDB | Artifacts, lineage, runs, decisions, findings, receipts (`go run ./cmd/overgodb query`) |
| [compatibility.json](compatibility.json) | Model and capability verification contracts |
| [kernels/manifest.json](kernels/manifest.json) | Kernel ABI, source and binary identities |
| [API manifest](docs/api_manifest.json) | Public commands, routes, document contracts |

Existing owners only; no parallel plans, ledgers, schedulers, recovery paths.
Contracts in docs, measurements in the store, history in Git.

Report outcome, material measurements, checks, remaining obligations.
