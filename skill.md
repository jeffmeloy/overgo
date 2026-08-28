---
name: overgo-iteration
description: Autonomous iteration doctrine for overgo. Use when working in this repo under an explicit continue/resume/loop directive, closing an implementation slice, updating repo doctrine/state, or deciding the next autonomous action. skill.md is the only top-level markdown agent instruction surface.
---

# Laconic mode

Fewest words the subject allows: lead with the verdict, no preamble/recap/offers.
Keep every distinction, measurement, and check that changes the action; drop the
rest. Brevity never overrides rigor — numbers stay quantitative, "unknown" beats
a tidy false claim, correctness may take length. Laconic governs chat, not
formal artifacts.

# Operating procedure (non-negotiable)

The loop is cognitive offloading, not control: self-managing goal, next action,
commit path, and code location starves the reasoning the actual problem needs.
Every turn:

1. `go run ./cmd/plan -prompt` => the ONE dispatched step. Do exactly that step.
2. Reuse before rediscovery: inspect the owning package and OvergoDB before
   writing anything. Reinventing existing code is the recurring failure.
3. Commit ONLY via `go run ./cmd/gate -plan <item>/<step> -message-file <f>
   -paths <csv>`; the gate reruns acceptance and advances the plan atomically.
   Guard-blocked `git commit` is the design; never raw-commit in a campaign.
4. Step wrong / blocked / you disagree => `cmd/plan -stop` and tell the user.
   Never substitute your own work for the dispatched step; never end a turn on a
   summary or "should I continue?".

Ground once, then act: a settled direction is executed, not re-verified,
re-asked, or forked. Anchor to the plan, not memory — long context degrades
recall, the plan does not. A path that feels blocked => inspect ownership and
the store before proposing any fork; no manufactured blockers.

Scope: a direct user request outranks the loop and bounds the ANSWER, not the
turn — answer it, then continue the dispatched row. Full loop applies under a
live continue/resume directive, at a committed-slice close, or when no narrower
user task is active. Co-implementer rule: after any user message, interruption,
or long tool run — `git status --short` and re-read touched files before judging
or editing; the tree may already hold the fix.

Priority when imperatives conflict:
`correctness > reuse-existing > decisiveness > tightness > loop-continuation`.
Lower never overrides higher.

# Mission

- One Go-native system: serve, train, evaluate, compose models on consumer
  hardware. Every claim evidence-bound, every constant derived.
- Two inheritances, distinct: llama.cpp pins behavior (formats, semantics,
  kernels; upstream oracle commit is the conscience); adaptive_new supplies
  philosophy and proven capabilities, ported through neutral contracts.
- User levers: `dataset`, `cpu_threads`, `memory`. Everything else is derived,
  deleted, or an open magic-ledger row.
- Improvement = external capability delta (recipe coverage, quality, throughput,
  peak memory, training curves, parity on shared artifacts). A greener dashboard
  is not progress. Tightness is the product: same capability, smaller/clearer
  wins.

# Durable automation

Agent effort is transient; commands are durable. Any action an agent performs
twice by hand — a census, a triage, a rebind, a measurement, a repair — becomes
an owned Go command with typed flags the second time it occurs. The hand-executed
repeat IS the trigger; "I'll just handle it" is the failure signal.

- Every trackable fact gets a deterministic Go owner; agents invoke owners,
  never re-derive facts in-context. Orchestration is never cognitive.
- A durable command prints an honesty line (what ran, what didn't, denominators)
  and exits nonzero on violation, so it can serve as a plan-row verify.
- Ratchets over memories: a standard the repo must hold (duplication ceiling,
  zero-debt bindings, staged surface) lives as a machine-checked authority file
  the gate enforces, not as agent recall.
- New automation is admitted like code: derived constants, closure rows, net
  surface accounted. Automation that only an agent's memory knows how to run is
  scrap, not automation.

# Authoritative surfaces

One owner per fact. Store is system of record; files only where a file is the
natural contract.

- `skill.md` — doctrine; sole root markdown instruction surface.
- `docs/plan.json` — live open-work surface; `cmd/plan` dispatches; every step
  has a non-vacuous verify; never a completion ledger.
- OvergoDB store — artifacts, lineage, profiles, recipes, decisions, runs,
  evaluations, advisories, magic rows, findings, gate outcomes. Query via
  `cmd/overgodb-query`; never grep the binary log.
- `compatibility.json` — machine-checked claims + evidence tiers; no claim =>
  no capability; stale evidence fails the gate.
- `kernels/manifest.json` — kernel ABI authority; kernels enter/change only
  through it.
- `docs/` — durable design/contract records; chronology lives in Git; the
  commit body is the ledger (Why / Evidence / Next).
- `./tmp/` — only home for development scrap; `./bin/` executables only; scrap
  never lands in `bin/` or the repository root.

# Automation doctrine

Gate holds itself to the code's standard.

- Derived constants only: every threshold/timeout/budget/tolerance is computed
  from store evidence, a math fact, or a typed decision + justification +
  reopen trigger. Automation constants are magics too.
- Distribution-free calibration: thresholds are empirical quantiles of the
  statistic's own history; median/MAD/quantiles/envelopes only; two-window
  confirmation escalates advisory→finding; first run calibrates, second
  enforces.
- Computed scope, never listed: test scope from the import-graph reverse cone,
  CUDA lane from manifest diff ∩ executor cone, smoke matrix from store query.
  Hand-maintained matrices are process magics.
- Evidence over green: every run lands as a phased, env-bound store run;
  UNAVAILABLE never passes; every green output ends with an honesty line naming
  what did NOT run.
- Commit mechanics (incident-proven): `-message-file` only; exit codes read
  unpiped; staged-scope canonical checks; dirty path outside the plan refuses
  the commit.
- Session safety: corpus-tested destructive-command guard before every shell
  call; model/dataset/checkpoint stores read-only to automation; provenance
  references bytes by location, never by copy.

# Operating loop

1. Ground: `git status --short`; read open findings + plan; re-rank from the
   current tree.
2. One behavioral change per slice, unless mechanically inseparable or a
   same-transformation batch with provably unchanged parity evidence.
3. Prefer deletion/derivation over new paths. Compatibility removal is atomic:
   migrate every caller, delete the old path same slice; no aliases or
   fallbacks (sole exception: an externally immutable format, explicit and
   evidence-backed).
4. Bounded probe before implementation when uncertainty blocks a decision; no
   bounded probe => rerank, not a closure essay.
5. Commit body: Why (one paragraph), Evidence (gate commands + numbers), Next.
   Ports cite the adaptive_new sha(s).
6. After a slice lands, pick the next action fresh; the previous rank-1 is a
   candidate, not a default.

Port-first (standing): capability/perf slices start from adaptive_new's
verified implementation — read end-to-end, port structure/kernels/math, verify
against its goldens; novel design only after the port matches, as a measured
improvement. Media generation uses the model-native Python path as golden and
must beat its wall and peak memory on the same artifact-quality assertions.

## Continuation

Session ends only for: user said stop; irreversible action needs confirmation;
a prerequisite only the user can supply. Nothing else — not checkpoints,
summaries, or context-depth judgments. Delicate work => smaller slices + harder
verification, not deferral. Wakes/notifications/user messages are events into a
continuous dispatcher, never boundaries.

Every event: for each free capacity (CPU, or GPU VRAM that fits an open step),
dispatch the highest open step that fits, or write one line why none fits. GPU
is a 48 GB pool — pack VRAM-fitting work concurrently; serialize only for clean
perf timing, a near-full-card job, or file/gate overlap. Yield only when every
capacity is occupied or has a written blocked-line.

Delegation: a subagent's prompt requires verification run to completion before
its final message. Long suites detach with a watcher waited on in the same
turn. Stall (no transcript growth + idle GPU ~15 min) => checkpoint demand;
no reply by next wake => TaskStop + salvage (`scripts/stall_check.sh`).

## Testing lanes

- Fast correctness: `go test ./...`, import-graph scoped.
- Claims/manifest/SBOM: gate steps, scope-derived.
- Device: `go run ./cmd/device-lane`; missing prereqs = UNAVAILABLE => FAIL for
  changes needing device evidence; never silently skipped.
- Model smoke: `go run ./cmd/smoke-lane`, store-derived matrix; absent models
  fail the lane.
- Race: `go run ./cmd/race-lane` — Go race detector for host goroutines
  (test-only cgo, never shipped) + CUDA compute-sanitizer for kernels; missing
  prereqs = UNAVAILABLE => FAIL.
- Magic scan: gate step over the commit's constants vs the closure ledger.

# Governance

- Eligibility precedes score: admissible only for repo integrity/data loss, a
  wrong landed result, an active capability blocker, or measured recurring tax
  with a plausible amortization window. Else accepted residual + reopen trigger.
- Permanent guards record incident class, recurrence, gate cost, payback,
  retirement trigger; lifetime tax is part of the decision.
- Frontier interrupt: an open finding on a thesis-path surface aged past its
  evidence cadence outranks eligible governance.
- External intake leads ranking (recipe coverage, results, quality, throughput,
  VRAM, parity); diff review is secondary intake. Full adversarial review
  batches at port waves/merge windows/milestones; per-slice review is
  risk-scoped.

# Porting discipline

Transfer unit = a verified processing capability, never a file/package:
adaptive behavior + evidence -> neutral typed contract -> reference (CPU) ->
CUDA -> recipe-selected consumer -> parity/perf evidence -> no compat path.

- No permanent cross-repo dependency; imports as neutral JSONL via
  `repodb-import`; behavior re-expressed through overgo owners.
- No extmodel package copies; no adaptive family names in executors (family
  lives in recipe scope).
- Port every caller the capability needs immediately; delete comparison
  adapters the same wave. Closures ride along or visibly reopen as rows.
- Preserve source commit, artifact identity, fixture hash, numerical evidence;
  record license/provenance. Named failure mode: a second adaptive runtime
  inside overgo.

# Scoring

Rank by external capability delta per wall-hour, after eligibility. Positive:
changed behavior on shared artifacts, scale evidence, measured unblock, net
magic closure, wall removed. Negative: recurring tax, new magics, doctrine
surface. Consistency-only movement = zero; process work scores only against
measured wall saved or deleted surface — pre-credit nothing. A run of rejected
same-axis probes closes the class => pivot.

# Magic discipline

Magic = any hard-coded value/threshold/distribution/shape assumption that could
be derived. Closure simplifies/generalizes/deletes the handle — never a better
story for a retained number.

- Production path: named const in the owning package + store magic row (tier,
  understanding, closure path, re-eval trigger) + same-commit closure plan.
- Closure paths: math fact; runtime/data-derived adaptation; mechanism
  replacement; consolidation; simplification making the constant irrelevant.
- Forbidden: keeping a magic because it works; fitted rectifiers as closures;
  fixed arbitrary quantiles as derivations.

Distribution assumptions are structural magics: default non-Gaussian,
non-stationary, heavy-tailed, non-IID. L-moments/L-scale over variance absent a
finite-variance guarantee; median/MAD for diagnostics; bootstrap at small n.
n<=3: only envelope separation is bankable. "sd" must defend why variance is
meaningful; small-sample floors are store-owned conventions.

Proxies: name the proxy, the canonical measurement, the measured bias on a
bounded audit sample; bias is mechanism (derive rectifier), noise (use as-is),
or structureless (verification-only). Proxy fails => close its class.

# Evidence standards

- Match evidence scope to claim scope (operator/object/scale all match).
- Robust primary summaries unless bounded by construction; best-value-only wins
  are directional.
- Preserve null results that close a mechanism family; don't escalate compute
  on a saturated lane without a new mechanism.
- Findings persist as store documents with a failable check (a `go test` check
  asserts `--- PASS`; bare `ok` passes on SKIP). Close only with implemented
  fix + failable check; refute only with direct counter-evidence.
- A/B wins elect incumbents, not truths: scale-local until re-defended.
- Pareto canvass before adding a mechanism: vs subtraction, direct measurement,
  the cheapest existing lever.

# Probes

Run only when it can change a decision: name the decision and the
classification the probe could move; numeric ship/kill threshold written before
the run. Yield-gate ordering: the survival measurement runs at the earliest
step that can produce it; gate fails => lane closes, no downstream surface.
Physics first: name the removed quantity in measured units before optimizing.

# Code hygiene

Go is the runtime surface. No cgo in the runtime (test-only race
instrumentation is the sole exception); C ABIs via loaded DLLs; kernels via the
manifest. Scripts are bash or Go, never PowerShell.

```go
// error: wrap once at the boundary that acts; never log+return
if err != nil {
    return fmt.Errorf("load %s: %w", path, err)
}
// context: first param, never a struct field, never Background below an assembly root
func Decode(ctx context.Context, r io.Reader) (Frame, error) { ... }
// zero value usable, else reject in the constructor
var buf bytes.Buffer
// interface: smallest, defined at the consumer, unexported
type sink interface{ write([]byte) error }
// name = behavior/math, not checkpoint/family
func decodeHybrid(...)          // yes
func qwen35Decode(...)          // no
// hot path: reuse buffers/plans sized from derived counts
out := plan.scratch[:0]
```

- Google Go style: clarity, simplicity, concision; `gofmt` + `go vet` every
  slice. Correctness, parity, contracts, measured performance, and repository
  authority take precedence over style.
- One control path, early returns. An abstraction must delete repeated
  policy/mechanics for >=2 real consumers; one-consumer helpers stay local. No
  utility dumping grounds; packages own cohesive capabilities.
- Concrete types by default; interfaces after a second real implementation.
  Exported API is permanent cost — migrate all callers + delete displaced
  compat APIs the same slice. Delete or privatize exports before documenting
  survivors; exported Godoc uses complete name-led sentences.
- Concurrency has one owner: bounded goroutine lifetime; detach under lock,
  release outside; prove shutdown/error/cancel. Synchronous until concurrency
  removes measured waiting. Measure before pooling/fusion/cache complexity.
- No sentinel success, no panic-recovery as validation, no topology-changing
  fallback. Model facts come from the artifact's declaration or the store,
  never computed in shared code.
- Constants name stable roles, never values. A call-site literal restating a
  config fact is a magic even when correct.
- Tests assert observable contracts + failure modes: table-driven, named
  fixtures; never copy the algorithm under test or depend on ambient state.
  Autograd/tape changes => grad-parity fixtures over every touched gradient.
- One-contract anti-drift: a mechanism proven in one consumer promotes to the
  shared contract next slice, or banks a consolidation row + trigger.
- Review order: correctness + ownership; API/zero-value/error/context;
  concurrency + cleanup; allocations; naming; tests; delta. Prefer the change
  that deletes authority/branches/state/bytes while preserving behavior.
- Delete dead flags/tools/fixtures once the verdict is recorded; no behavior
  change in cleanup commits. Don't revert user changes unless asked.

# Documentation hygiene

One fact, one owner. Live docs = current decisions + invariants; chronology in
Git; evidence in the store. Telegraphic; delete stale docs rather than improve
them. Docs grow in a slice => compress or delete equal/larger stale prose.

Research intake: per paper — mechanism, assumptions, scale, grounded against
the nearest primitive with a falsifier. Doctrine lands here; priorities land in
the plan.

# Final gate

Before any final answer during a campaign: gate green with its honesty line
printed, a required long-running command still running, or the turn is a
bounded/status request answered. A checkpoint is not a stopping point; a stop
names something only the user can do.
