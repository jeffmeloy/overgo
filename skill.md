---
name: overgo-iteration
description: Autonomous iteration doctrine for overgo. Use when working in this repo under an explicit continue/resume/loop directive, closing an implementation slice, updating repo doctrine/state, or deciding the next autonomous action. skill.md is the only top-level markdown agent instruction surface.
---

# Laconic mode

- Fewest words the subject allows. No preamble, restating, closing summary, or
  follow-up offers. Lead with the number/verdict/decision; add reasoning only
  when it changes what the user does.
- Keep every distinction/measurement/check that changes the action; drop the
  rest and all hedging.
- Brevity never overrides rigor. Numbers stay quantitative with uncertainties;
  distinct facts stay distinct; honest "unknown" beats a tidy false claim.
  Compression drops words, never conclusions. Correctness needs length => take
  it, not one line more.
- Formal artifacts keep their own conventions; laconic governs chat, not format.

# Overgo Iteration

## Operating procedure (non-negotiable)

The loop is cognitive offloading, not control. A repo this size plus continuous
iteration exceeds what any agent can hold at once. Self-managing the goal, the
next action, the commit path, and where code lives eats the exact budget you
need for the actual problem — and that starvation IS the over-analysis, the
forgotten goal, the invented fork. Handing that bookkeeping to the loop frees
your reasoning for the hard part: following it makes you more capable, not less.
Every turn, run the loop — do not reinvent it, improve it in-flight, or fall
back to the manual path:

1. `go run ./cmd/plan -prompt` => the ONE dispatched step. Do exactly that step.
2. Need an existing capability? `docs/MAP.md` => the owner. Do not re-derive.
3. Commit ONLY via `go run ./cmd/gate -plan <item>/<step> -message-file <f>
   -paths <csv>`; then `go run ./cmd/plan -advance <item> <step>`.
4. Step wrong / blocked / you disagree => STOP and tell the user
   (`cmd/plan -stop`). Never substitute your own work for the dispatched step;
   never end a turn on a summary or a "should I continue?".

Going off-script IS the failure mode. When in doubt, offload to the tool, not to
your own reasoning.

## Mission

- One Go-native system: serve, train, evaluate, compose models on consumer
  hardware. Every claim evidence-bound, every constant derived.
- Two inheritances, kept distinct. llama.cpp: pinned behavior (formats,
  semantics, kernels; upstream oracle commit is the conscience). adaptive_new:
  philosophy (derivation, evidence, refusal) + proven capabilities, ported
  through neutral contracts.
- User levers: `dataset`, `cpu_threads`, `memory`. Else derived, deleted, or an
  open magic-ledger row.
- Improvement = external capability delta: recipe coverage, quality, throughput,
  peak memory, training curves, multimodal behavior, llama.cpp parity on the
  same artifact. A greener dashboard is not progress.
- Tightness is the product. Same capability + smaller/clearer wins. Prefer exact
  simplifications. New constant/selector/distribution assumption => derive, or
  record the decision + trigger.

## Navigation and Decisiveness

- Reuse before rediscovery. `docs/MAP.md` maps capability => existing owner;
  read it before grepping "does X exist" or writing a helper. Missing row => add
  it when found. The recurring failure is reinventing or re-deriving code that
  already exists.
- Ground once, then act. Task within a stated recommendation and its owner
  exists => execute. Do not re-verify a settled direction, invent alternatives,
  or re-ask what the user already decided.
- Anchor to the plan, not to memory. `go run ./cmd/plan -prompt` (or `-next`) is
  the task of record every turn; long context degrades recall, the plan does
  not. Dispatch = execute; never open a turn with a self-chosen meta-question.
- Use the automation. `cmd/{plan,gate,loophook}` + `scripts/*.sh` own the loop
  (dispatch, commit, doctrine, guard). Invoke them; do not hand-roll the
  disciplined path. Guard-blocked `git commit` is the design, not friction —
  gate is the path.
- No manufactured blockers. RepoDB catalogs models AND datasets; gate/plan/guard
  is the sanctioned path, not an obstacle to route around. A path that feels
  blocked => re-read `docs/MAP.md` before proposing any fork.

## Priority order (imperatives in conflict)

`correctness > reuse-existing > decisiveness > tightness > loop-continuation`.
Lower never overrides higher: "never stop" never licenses churn; tightness never
licenses reinventing an existing owner.

## Scope and Precedence

- Direct user request outranks the loop. Review/explanation/status/scoped edit =
  finished when answered; a complete turn, not a stop to defend. Never let loop
  pressure turn a bounded request into a campaign.
- Full loop applies only under a live continue/resume/loop directive, at a
  committed-slice close, or when no narrower user task is active.
- Co-implementer rule: user and concurrent lanes are parallel implementers.
  After any user message/interruption/long tool run: `git status --short`,
  re-read touched files before judging or editing. Tree already holds the fix =>
  verify it, record the stale finding; never race a duplicate.

## Authoritative Surfaces

One owner per fact. Store is system of record; files only where a file is the
natural contract.

- `skill.md` — doctrine and agent behavior; sole root markdown instruction
  surface. Campaign doctrine cited from here, never re-authored elsewhere.
- `docs/plan.json` — live open-work surface; `cmd/plan` dispatches. Open work
  only, never a completion ledger; every step has a non-vacuous verify.
- RepoDB store — artifacts, lineage, profiles, recipes, decision events
  (promotion/refusal + reason + decider commit), runs (phased, env-bound),
  evaluations, advisories, magic rows, findings, gate outcomes. Query via
  `cmd/repodb-query`; never grep the binary log.
- `compatibility.json` — machine-checked claims + evidence pointers + evidence
  tier (`implemented` != `oracle-backed`). No claim => no capability; stale
  evidence fails the gate.
- `kernels/manifest.json` — kernel ABI authority: hashes, layouts, bindings.
  Kernels enter/change only through it.
- `docs/` — durable design/contract records only; `MERGE_FLOOR_PLAN.md` tracks
  automation-floor component status. Chronology lives in Git; the commit body is
  the ledger (Why / Evidence / Next).

## Automation Doctrine

Gate holds itself to the code's standard.

- Derived constants only. Every threshold/timeout/budget/tolerance/retry is (a)
  computed from store evidence, (b) a math fact, or (c) a typed decision +
  justification + reopen trigger. Magic scan flags unexplained automation
  literals.
- Distribution-free calibration. Threshold = empirical quantile of the
  statistic's own history; alarm budget = recorded decision. Median/MAD/
  quantiles/envelopes only. Two-window confirmation escalates advisory→finding.
  First run calibrates, second enforces; no history => advisory only.
- Computed scope, never listed. Test scope = import-graph reverse cone; CUDA
  lane = manifest diff ∩ executor cone; smoke matrix = store query; claim
  re-verify = evidence-path intersection. Hand-maintained matrices are process
  magics.
- Evidence over green. Every run lands as a phased, env-bound store run.
  UNAVAILABLE never passes. Every green output ends with an honesty line naming
  what did NOT run (steps run/skipped, derived scope, unavailable/claims/magics
  counts as lanes land). Silence about what didn't run ships regressions.
- Commit mechanics (incident-proven): `-message-file` only; status mirror keyed
  by result id + commit, written once from the store record (store
  authoritative, mirror advisory); exit codes read unpiped; staged-scope
  canonical checks only; dirty path outside the plan refuses the commit.
- Session safety first: corpus-tested destructive-command guard before every
  Bash/PowerShell call. Model/dataset/checkpoint stores guard-enforced read-only
  to automation (deny delete/move/chmod); provenance references bytes by
  location, never by copy.

## Operating Loop

1. Ground: `git status --short`; read open findings + `docs/plan.json`; re-rank
   from the current tree.
2. One behavioral change per slice — unless mechanically inseparable, or a
   same-transformation batch that provably leaves parity evidence unchanged (one
   gate run).
3. Prefer deletion/derivation over new paths. Compatibility removal is atomic:
   migrate every caller, use compile/tests to find misses, delete the old path
   same slice. No aliases, no fallbacks. Exception: externally immutable
   format/interface, explicit + evidence-backed.
4. Bounded probe before implementation when uncertainty blocks a decision. No
   bounded probe => rerank, not a closure essay.
5. Commit body carries the narrative: Why (one paragraph), Evidence (gate
   commands + numbers), Next. Ports cite the adaptive_new sha(s) that proved it.
6. Commit through the gate; never raw `git commit` during a campaign.
7. Review batched + risk-scoped (see Governance). After a slice lands, pick the
   next action fresh; the previous rank-1 is a candidate, not a default.

### Port-First (standing)

The campaign PORTS adaptive_new; never from scratch. Every capability/perf slice
starts from adaptive_new's verified impl: read end-to-end, port
structure/kernels/math, verify against its goldens. Novel design only after the
port matches, and only as a measured improvement on top. Reinventing what
adaptive ships is the named failure mode.

Media generation uses the model-native Python path as its golden and must beat
its wall and peak memory without failing the same artifact-quality assertions.
Head-to-head runs use the same artifact and real catalog inputs.

### Continuation

Session ends only for three user-owned reasons: user said stop; irreversible
action needs confirmation; a prerequisite only the user can supply is missing.
Nothing else is a stop — not a checkpoint, summary, context-depth judgment, or
"fresh context for delicate work." Delicate work => smaller slices + harder
verification, not deferral. Repo evidence can decide a fork => decide it, report
what changed. Wakes/task-notifications/user-messages are events into a
continuous dispatcher, never boundaries; "end of turn" is transport, not a unit
of work.

Every event: for each free capacity (CPU, or GPU VRAM that fits an open step),
dispatch the highest open step that fits, or write one line why none fits. GPU
is a 48 GB pool, not a lane — pack VRAM-fitting work concurrently (correctness
dumps, small serves, kernel builds, recons, host ports). Serialize onto one GPU
task only for: (a) clean-timing perf measurement, (b) a job needing most of the
card, (c) file/gate overlap. Yield only when every capacity is occupied or has a
written blocked-line — a scheduler yield, not an ending; nothing gets wrapped up.

Delegation contract: a subagent's prompt requires verification run to completion
before its final message; "suite still running, will finalize later" is
malformed. Long suites => detach with a watcher, wait on it in the same turn.
Stall = no transcript growth + idle GPU ~15 min => checkpoint demand; no reply
by next wake => TaskStop + salvage its tree (`scripts/stall_check.sh`).

### Testing Lanes

- Fast correctness: `go test ./...` — model-free, per code slice; gated scope
  from the import graph.
- Claims / manifest / SBOM: gate steps, scope-derived (claims by evidence-path
  ∩, manifest by kernel-owning paths, SBOM by dependency-owning paths).
- Device: `go run ./cmd/device-lane`; gate routes by kernel/CUDA-cone paths.
  Missing prereqs = UNAVAILABLE => FAIL a change needing device evidence. Never
  passing, never silently skipped.
- Model smoke: `go run ./cmd/smoke-lane` — store-derived matrix over the
  servable predicate; every serve records a run + evaluation vs its own recipe;
  absent models fail the lane.
- Race: `go run ./cmd/race-lane` — host goroutine races via the Go race detector
  (ThreadSanitizer, `CGO_ENABLED=1`, test-only, never in the shipped binary) over
  the goroutine-bearing packages; device/kernel races via CUDA
  `compute-sanitizer` (racecheck + synccheck) wrapping a compiled device test
  binary. Complementary — `-race` can't see kernels, the sanitizer can't see
  goroutines. Missing C compiler, GPU, or compute-sanitizer = UNAVAILABLE =>
  FAIL, never skipped.
- Magic scan: gate step over the commit's constants vs the closure ledger.

## Capability-Grounded Governance

- Eligibility precedes score. Admissible only for: repo integrity/data loss, a
  wrong landed result, an active capability blocker, or measured recurring tax
  with a plausible amortization window. Else = accepted residual + reopen
  trigger.
- Every permanent guard records incident class, recurrence, recurring gate cost,
  expected payback, retirement trigger. Incidents alone insufficient; lifetime
  tax is part of the decision.
- Frontier interrupt rights: an open finding on a thesis-path surface aged past
  its own evidence cadence outranks eligible governance.
- External intake leads. Rank from recipe coverage, end-to-end results, quality,
  throughput, VRAM, llama.cpp parity on shared artifacts. Diff review is
  secondary intake, never the sole defect source.
- Full adversarial review batches at a port wave, merge window, changed shared
  boundary, or capability milestone; per-slice review is risk-scoped, may find
  nothing. Governance-vs-capability commit ratio publishes in the gate summary,
  quantile-alarmed like any longitudinal metric.

## Porting Discipline

Transfer unit = a verified processing capability, never a file/package:

    adaptive behavior + evidence -> neutral typed contract -> reference (CPU)
    -> CUDA -> recipe-selected consumer -> parity/perf evidence -> no compat path

- No permanent cross-repo dependency. Imports arrive as neutral JSONL via
  `repodb-import`; code arrives as behavior re-expressed through overgo owners.
- No extmodel package copies. Neutral leaf packages (own oracle, no extmodel
  imports, no family names) may move as-is.
- No adaptive family names in executors; the family lives in the recipe scope,
  never the identifier.
- Port every caller the capability needs immediately; delete any comparison
  adapter the same wave.
- Closures ride along: each port lists its magic rows; each closure ports with
  it or visibly reopens as a tracked row. A ported shape without its closure
  silently reopens an expensive question.
- Preserve source commit, artifact identity, fixture hash, numerical evidence;
  record license/provenance — overgo is clean-room compatible.
- Named failure mode: a second adaptive runtime inside overgo. Endpoint =
  capability through overgo's graph/recipes/runtime; adaptive orchestration
  discarded.

## Scoring

Rank by external capability delta per wall-hour, after the eligibility bar.
Positive: changed behavior on a shared artifact/benchmark, model/scale evidence,
measured frontier option, measured unblock, net magic closure, wall removed.
Negative: recurring governance tax, new magic cost, doctrine-surface cost.
Consistency-only movement = zero. Process work scores only vs measured wall
saved, deleted surface, measured prevented recurrence, or an unlocked probe —
pre-credit nothing. Multi-regime evidence outranks single-corpus. A run of
rejected same-axis probes closes the class => pivot. Verification cost is wall
time.

## Magic Discipline

Magic = any hard-coded value/threshold/distribution assumption/hidden shape
assumption/fixed horizon that could be derived. Closure uses understanding to
simplify/generalize/delete the handle — never a better story for a retained
number.

- Production path: named const in the owning package + store magic row (tier,
  understanding, closure path, re-eval trigger, owner surface, pinning fixture) +
  same-commit closure plan.
- Closure paths: math fact; runtime/data-derived adaptation; mechanism
  replacement; consolidation; a simplification making the constant irrelevant.
- Forbidden: keeping a magic because it works; swapping one magic for a fitted
  rectifier; fixed arbitrary quantiles as derivations; probe-only constants
  exempt once they affect decisions. Automation constants are magics too.

## Distribution Discipline

Distribution assumptions are structural magics. Default: non-Gaussian,
non-stationary, heavy-tailed, non-IID until proven otherwise.

- Prefer L-moments/L-scale over variance absent a finite-variance guarantee;
  median/MAD for decision diagnostics; rank views under monotone heavy tails;
  bootstrap or explicit uncertainty at small n.
- Never silently assume Gaussian residuals, IID, symmetry, finite fourth
  moments, stationarity, or Wald/normal CIs where robust intervals exist.
- n<=3: only bankable comparison is envelope separation (non-overlapping
  min-max). Needs a distribution to see the effect => add replicates or don't
  decide.
- Scale source "sd" must defend why variance is meaningful; L-scale/MAD/
  envelope/sign-test need no defense. Small-sample floors are named store-owned
  conventions, not per-tool literals.

## Proxy Discipline

Before proxying a runtime quantity: name the proxy, the canonical measurement,
the measured bias on a bounded audit sample; choose one — bias is mechanism
(derive the rectifier), bias is noise (use as-is), or no derivable structure
(verification-only). A fitted rectifier without derivation is a relocated magic.
Proxy fails => close its class, don't try a sibling metric.

## Evidence Standards

- Match evidence scope to claim scope. Closure verdicts cite operator/object/
  scale; reuse only when all three match.
- Robust primary summaries unless bounded by construction. Best-value-only wins
  are directional, not magnitude proof.
- Preserve null results that close a mechanism family. Don't escalate compute on
  a saturated lane without a new mechanism.
- Findings persist while live as store documents (owner surface, evidence,
  closure path, failable check — a `go test` check asserts `--- PASS`; `ok`
  passes on SKIP). Close only with implemented fix + failable check; refute only
  with direct counter-evidence.
- A/B wins elect incumbents, not truths: scale-local until re-defended at a
  second scale point; unchallenged != confirmed.
- Pareto canvass before adding a mechanism: vs subtraction, direct measurement,
  the cheapest existing lever. Two paths, one claim => keep the fewer operations.

## Probes

Run only when it can change a decision.

- Name the decision, current classification, the one the probe could move it to.
  Same next action either way => no probe.
- Numeric ship/kill threshold, written before the run. No threshold => no probe.
- Yield gate ordering: the measurement deciding whether a lane survives runs at
  the earliest step that can produce it. Gate fails => lane closes, no
  downstream surface built.
- Screen before build: oracle/ceiling screen with existing levers; oracle miss
  => skip the realistic build.
- Physics first: name the removed quantity (cycles/allocations/transfers/
  latency) in measured units before optimizing. Sample % alone = unidentified
  mechanism.

## Code Hygiene

Go is the runtime surface. No cgo in the runtime, ever; C ABIs via loaded DLLs;
kernels via the manifest. Scripts are bash or Go, never PowerShell. (Test-only
race instrumentation is the sole cgo exception — see Testing Lanes; it never
enters a shipped binary.) Go is simple — write the obvious code and match these
forms:

```go
// error: wrap once at the boundary that acts; context in the message; never log+return
if err != nil {
    return fmt.Errorf("load %s: %w", path, err)
}
// context: first param, never a struct field, never Background below an assembly root
func Decode(ctx context.Context, r io.Reader) (Frame, error) { ... }
// zero value usable, else reject in the constructor — no partial structs, no builder state
var buf bytes.Buffer            // ready at zero
// interface: smallest, defined at the consumer, unexported; not exported for tests/maybe-callers
type sink interface{ write([]byte) error }
// name = behavior/math, not checkpoint/family; the family lives in the recipe scope
func decodeHybrid(...)          // yes
func qwen35Decode(...)          // no
// hot path: reuse buffers/plans sized from derived counts; no per-token maps/reflection/boxing
out := plan.scratch[:0]
```

- Simplicity is a constraint: one control path, early returns, minimal nesting.
  An abstraction must delete repeated policy/mechanics for >=2 real consumers; a
  one-consumer generic helper stays local or dies. No "utility" dumping grounds.
- Packages own cohesive capabilities, not horizontal utilities; names short,
  concrete, non-redundant. Dependencies point at stable contracts; composition
  roots bind implementations. Composition over inheritance frameworks.
- Concrete types by default; add an interface only after a second implementation
  is real. Exported API is permanent cost — export only stable cross-package
  contracts; migrate all callers + delete displaced compat APIs the same slice.
- Concurrency has one owner: bounded goroutine lifetime; detach under lock,
  release outside it; no callbacks while locked; prove shutdown/error/cancel.
  Synchronous until concurrency removes measured waiting. Measure before
  pooling/fusion/cache complexity.
- No sentinel success, no panic-recovery as validation, no topology-changing
  fallback. Typed/sentinel errors only when callers branch.
- Model facts come from the artifact's declaration or the store, never computed
  in shared code. Comments telegraphic + load-bearing (contract/invariant/
  non-obvious why); never narrate syntax. Go initialism/receiver/package
  conventions throughout.
- Tests assert observable contracts + failure modes: table-driven, named
  fixtures. Don't copy the algorithm under test, inspect impl trivia, depend on
  ambient state, or make literals a second config surface.
- Autograd/tape-topology change => grad-parity fixture over every touched
  gradient, split to localize.
- One-contract anti-drift: a mechanism proven in one consumer promotes to the
  shared contract next slice, or banks a consolidation row + trigger. "Next
  architecture needs a new file or a policy row?" => drift. A call-site literal
  restating a config fact is a magic even when correct.
- Every slice: `gofmt`, focused tests, `go test ./...`, `go vet ./...`; add
  race/fuzz/bench/CUDA/integration when risk requires. A green gate never
  excuses an unnecessary API or path.
- Review order: correctness + ownership; API/zero-value/error/context;
  concurrency + cleanup; allocations; naming + comments; tests; delta. Prefer the
  change that deletes authority/branches/state/ops/bytes while preserving
  behavior.
- Delete dead flags/tools/fixtures once the verdict is recorded; no behavior
  change in a cleanup commit. Don't revert user changes unless asked.

## Documentation Hygiene

- One fact, one owner. Live docs = current decisions + invariants; chronology in
  Git; evidence in the store.
- Telegraphic; delete stale docs rather than improve them; live guidance never
  references deleted surfaces.
- Docs grow in a slice => compress or delete equal/larger stale prose, unless the
  growth is machine-readable state.

## Research Intake

Per paper: extract mechanism, assumptions, scale; ground against the nearest
existing primitive with a falsifier. exploitable=yes is a rank input, not build
order. Shared vocabulary without a code transformation is rationale only.
Reusable doctrine lands here; priorities land in the plan.

## Final Gate

Before any final answer during a campaign: gate green with its honesty line
printed, a required long-running command still running, or the turn is a
bounded/status request answered. A checkpoint is not a stopping point; a stop
names something only the user can do.
