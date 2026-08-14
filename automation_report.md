# Overgo automation report

Status: living assessment for `codex/overgo_automation` on 2026-08-14.

## Executive assessment

Overgo already has a strong verification control plane. It can bind a commit to
one declared plan step, derive affected Go tests from the import graph, reject
vacuous test evidence, verify generated manifests and claims, record structured
run evidence, exercise CUDA-specific lanes, and produce reproducible release
archives. The automation is unusually good at saying what did *not* run.

The system does not yet have an autonomous orchestration control plane. That is
intentional: the owner currently assigns worktrees, resolves conflicts, allocates
the shared GPU, integrates branches, and decides promotion. The next automation
phase should strengthen observation, isolation, independent SQA, and merge
eligibility without prematurely taking scheduling authority away from the owner.

Independent admission now validates immutable developer/SQA identities, clean
separate worktrees, frozen evaluators, exact target heads, complete findings,
and approved verdicts from RepoDB. The remaining governance risk is orchestration:
review packet creation/priority and manual worktree/resource state still live
largely in the owner's head.

## Incremental implementation status

- Current `master` was refreshed into the automation lane at `fa67865`; the
  original merge at `57a5749` passed the repository gate, including the device
  lane.
- `cmd/plan -context` now emits one deterministic task context containing exact
  Git/worktree identity, explicit role, one current task, normalized dirt, and
  conservative evidence debt (`e5fe81f`).
- The first evidence-honesty slice landed at `9626b9f`:
  CI and release use a Go-owned hermetic test lane, unclassified skips and
  unavailable evidence fail that lane, classified short exclusions are reported
  but not credited, loop orphan detection covers all Git-visible work, and
  changed embedded assets inherit their package tests from compiler-resolved
  `go:embed` metadata.

- The gate-lifecycle slice landed at `fc822ac`. A typed
  preparation now lands in RepoDB before Git can advance; finalization closes it
  atomically with result evidence. Post-commit record failure is non-green and
  leaves a validated reconciliation batch. Retry reuse is bound to the typed
  environment identity, a Go heartbeat supports stale detection, and automation
  context derives record debt from RepoDB rather than the advisory status mirror.
- Immutable developer, SQA, worktree, evaluator, candidate, finding, and verdict
  identities landed at `98fd512` as the evidence substrate for review admission.
- The initial `internal/gatecontrol` extraction (`7a8774a`) improved ownership
  locally but increased total production Go. It was removed after measuring the
  full delta: lifecycle, retry, and environment mechanics each have one command
  consumer and therefore remain local. Shared domain behavior stays in
  `runrecord`, `artifact`, and `repodb`.
- Strict evidence-document construction is shared across review, lifecycle, and
  environment records (`1f88d21`); the obsolete advisory-mirror classifier is
  deleted.

The next highest-priority gap is enforceable independence between developer and
SQA identities, including target-head review admission and immutable findings.

### Tightening measurements

The fixed baseline is `fa67865`; measurements use committed
`git diff --no-renames --numstat` over Go files, with `_test.go` reported
separately. The first three passes had actually grown production Go by 109 lines
and tests by 177 lines. The correction removes the one-consumer `gatecontrol`
abstraction while preserving its behavior.

| Surface vs `fa67865` | Net change |
| --- | ---: |
| Production Go | -52 lines |
| Go tests | +45 lines |
| Total Go | -7 lines |
| `internal/plan/context.go` | -48 lines |
| Run-record document family | -9 production lines |
| `cmd/gate/main.go` | -12 lines |

The result has no `gatecontrol` package or exported automation API. Common code
is retained only where multiple real consumers exist: record codecs and lineage
use the document-family primitive; gate and plan use `internal/jsonfile` for
strict decoding and canonical writes. Command-specific coordination stays in
its command.

## Operating boundary

Current division of responsibility:

| Responsibility | Current owner | Desired near-term state |
| --- | --- | --- |
| Select and assign worktree tasks | Human owner | Human, assisted by ranked candidates and resource estimates |
| Allocate CPU, RAM, and GPU capacity | Human owner | Human-approved reservations with measured estimates |
| Implement a capability slice | Development lane | Unchanged |
| Adversarially review a candidate | Informal/separately requested lane | Distinct, recorded SQA identity and clean worktree |
| Determine merge eligibility | Human plus gate output | Machine-generated eligibility packet; human decision |
| Promote model/runtime evidence | Human owner | External authority over sealed evidence |
| Recover abandoned or conflicting lanes | Human owner | Detect and recommend; do not mutate automatically yet |

This boundary is appropriate for the current embryo stage. Automation should
first make the owner's decisions cheaper, better grounded, and auditable. Full
dispatch autonomy should be earned later from measured scheduling and recovery
performance.

## Automation implementation doctrine

Automation exists to remove recurring cognitive work from the model. A proposed
mechanism is valuable when it converts a repeated interpretation, calculation,
scope decision, or bookkeeping obligation into deterministic, inspectable, and
tested behavior. Automation that merely adds another state surface for the model
to reconcile is negative value.

Implementation follows Overgo's Go-code doctrine:

- Go owns automation policy, state transitions, derivations, validation, and
  reusable mechanisms whenever feasible.
- Search for and extend the existing common Go owner before creating a new
  package or command-local implementation. One fact and one behavior have one
  owner.
- Commands should be thin adapters over testable `internal` packages: parse
  inputs, call the common mechanism, render a typed result, and select an exit
  code.
- Bash is limited to unavoidable hook and external-tool boundaries. It must not
  become a second policy implementation. Repository automation never uses
  PowerShell.
- Python is reserved for externally native reference/oracle generation or a
  dependency that cannot reasonably be expressed through the Go runtime. It is
  not an alternate orchestration or state-management surface.
- Prefer structured, stable machine output with a concise human rendering. Later
  automation and models should consume the same typed result rather than parse
  narrative text.
- Derive scope and constants from Go/package graphs, manifests, RepoDB evidence,
  and typed decisions. Do not replace model guesswork with hand-maintained path
  tables or unexplained automation literals.
- Every extracted mechanism carries focused tests, including the incident or
  repeated model failure that justified extracting it.
- Deletion is part of the transfer: once common Go code owns a behavior, remove
  duplicate shell logic, prompt instructions, and command-local variants in the
  same slice where feasible.

The acceptance question for each automation slice is: **what deterministic Go
result can the next model invocation trust without reconstructing the work?**

## AST-driven structural profile

Overgo should track code structure as a versioned evidence vector, not optimize
one scalar quality score. A compact `internal/repoanalysis` substrate will own
baseline/candidate file inventories, worktree overlays, content identities,
generated/test classification, and lazy parse-once Go ASTs. It is a data plane,
not a plugin framework: analyzers remain normal typed Go functions with no
registration system, policy DSL, scheduler, or independent state.

Normalized dirty-state facts move first because plan context, loophook, and gate
already consume them. The AST snapshot waits until code profiling lands, when
the new profile and existing `internal/closurescan` become two real consumers;
closure scanning's old walk/parser is deleted in that slice. The profile is
deliberately small:

- normalized syntax mass, separated into production and test code;
- duplicate excess tokens, clone count, longest clone, and cross-package clone
  mass, with deterministic source spans for every reported cluster;
- function syntax size and branch-count distributions plus the largest outliers;
- exported declaration counts and package import-edge counts.

Clone detection will retain operators, types, calls, and selectors while
canonicalizing local bindings and literal values. A rolling fingerprint over
normalized syntax will find exact and renamed statement sequences, extend and
coalesce overlapping matches, and ignore short boilerplate. Canonical generated
files are excluded; tests are never mixed with production results. Parse errors
are unavailable evidence rather than zero measurements.

The comparison unit is the candidate versus its merge base, including staged,
unstaged, and untracked candidate content when used before commit. Files shared
by both trees are recognized by content identity and analyzed once. One inventory
walk and one parse per unique blob feed all requested analyzers; deterministic
parse-count tests make reuse observable. Output is a concise summary plus a
complete versioned RepoDB document, and changed clone clusters and structural
outliers are attributed back to candidate spans.

Broader AST checks migrate onto this substrate only when a focused slice deletes
their old walker or demonstrates a measured wall/allocation improvement. The
substrate does not accumulate speculative facts for hypothetical analyzers, and
it gains no persistent cache until repository-scale benchmarks show that the
in-process content-identity reuse is insufficient.

This profile is a review instrument, not a ratchet:

- there is no composite score, ceiling, growth budget, exception workflow, or
  suppression ledger;
- the gate requires honest profile evidence, never a favorable metric direction;
- independent SQA acknowledges the largest deltas and decides whether they are
  missing abstractions, intentional parallel structure, or incidental syntax;
- reduced duplication is not credited when it merely concentrates branches,
  expands API, weakens tests, or creates a one-consumer abstraction;
- no shared abstraction is extracted without two real consumers.

The first implementation must be calibrated against small checked-in clone
fixtures and several known repository examples. If the highest-ranked findings
are not useful, the detector is deleted rather than surrounded with tuning and
waiver machinery.

## Capability inventory

### Plan and loop control

- `cmd/plan` owns a compact, open-work-only queue and selects the first open
  step. It can add work, define verification, generate a task prompt, verify,
  advance, compact, and record one of three recognized stop reasons.
- `cmd/loophook` and the thin shell adapters inject doctrine at session start,
  arm a post-commit dispatch marker, and block a turn end that would leave
  uncommitted Go work or a committed-but-undispatched boundary.
- The gate requires every normal and merge commit to name the current plan
  item/step. Off-plan commits are refused inside the configured harness.

### Commit gate

`cmd/gate` owns the repository's strongest automation path:

1. Verify plan binding and exact changed-path scope.
2. Check formatting for changed Go files.
3. Run repository vet/build work when Go ownership is touched.
4. Derive direct and transitive dependent test packages from the Go import
   graph.
5. Reject skips and unavailable prerequisites for direct evidence; report
   dependent fixture gaps without crediting them.
6. Verify kernel manifest, SBOM, compatibility claims, and magic-ledger coverage
   when their owning paths are touched.
7. Route CUDA-cone changes to the device lane.
8. Commit through a message file and write a structured RepoDB gate record.

Successful expensive tree-dependent steps are cached against a hash of HEAD,
staged changes, unstaged changes, and planned untracked content. The final
honesty section lists skipped, reused, unavailable, and advisory evidence.

### Safety guard

The PreToolUse guard carries useful incident lineage. It detects common forms of
force deletion, protected data-path mutation, dangerous worktree removal, raw
commit bypass, nested shell wrappers, and source-writing heredocs. It explicitly
describes itself as accident prevention rather than an adversarial security
boundary, which is the correct claim.

### Evidence and provenance

- RepoDB provides content identities, atomic batches, lineage, relations,
  snapshots, read-only discovery, and compare-and-set authority.
- Gate, smoke, benchmark, evaluation, advisory, and promotion records share the
  run-record model.
- Compatibility claims carry explicit evidence tiers and verification targets.
- Kernel manifests bind generated PTX, CUDA sources, entry points, and argument
  layouts.
- The SBOM generator binds modules and important generated/kernel files by hash.
- Closure scanning and the magic ledger expose unexplained constants and
  capability-transfer closure rather than hiding them in prose.

### Verification lanes

- The device lane treats a missing CUDA device or driver as unavailable and
  non-passing.
- The race lane combines Go's host race detector with CUDA racecheck and
  synccheck; missing tools fail as unavailable.
- The smoke lane derives its matrix from RepoDB's servable predicate and records
  each real serve against its active recipe.
- Benchmarking emits structured wall, memory, token, and execution metrics.
- Advisory generation compares recorded observations and can escalate repeated
  regressions into findings.

### CI and release

- GitHub CI runs formatting, tests, vet, SBOM, kernel-manifest, and compatibility
  checks on Windows and Linux, plus a focused Linux race job.
- The release workflow builds a Windows archive twice and requires byte-for-byte
  reproducibility before publishing the archive and checksum artifact.

## adaptive_new automation transfer review

Review baseline: adaptive_new commit `214950b3b` plus a read-only inspection of
its concurrent working tree on 2026-08-14. The working tree contained unrelated
edits, so no adaptive_new files were changed. Focused tests for `cmd/state`,
`cmd/commit-gate`, `internal/gateorch`, `cmd/hygiene`, `cmd/check-doc-drift`,
`cmd/check-magic-drift`, and `cmd/gate-watchdog` pass. `cmd/state -validate`
reports no state or process invariant violations.

Adaptive_new contains valuable operational DNA, but its automation is also an
example of the cognitive surface Overgo must avoid recreating. `cmd/state` alone
has roughly 6,900 production lines across 66 files, backed by about 4,800 test
lines. Its commit gate adds roughly 1,300 production and 1,000 test lines. The
right transfer unit is one proven mechanism re-expressed through an existing
Overgo Go owner, followed by deletion of any superseded prompt or script logic.

### Transfer decisions

| Adaptive_new mechanism | Decision | Overgo form | Important correction |
| --- | --- | --- | --- |
| Machine-readable dispatch context | Port early | Extend `internal/plan` and `cmd/plan` with one typed JSON context containing HEAD, worktree, current step, dirty scope, evidence debt, and role | Expose exactly one current-task identity; adaptive_new emits both state rank-1 and plan live-rank-1 |
| Git-derived implementation -> SQA -> priority phase | Port with stronger authority | Derive phase from candidate/review records and Git, enforce it in `cmd/gate` | A distinct findings commit is not a distinct reviewer; bind developer and SQA identities |
| Immutable cycle event -> deterministic state projection | Port the pattern | RepoDB run/review/decision events own history; generate any human view | Do not add mutable `state.json` and `cycle.json` as competing authorities |
| Structured findings and failable closure checks | Port | RepoDB finding documents with severity, status, owner, evidence, closure/refutation check, decision, and reopen trigger | Keep checks typed where possible; do not grow prose-ledger records into mini reports |
| Typed gate step catalog and lifecycle | Port | Extend `internal/runrecord` and `cmd/gate` with run ID, PID/process identity, step state, terminal reason, and prepared/finalized record debt | RepoDB remains authoritative; status files are disposable projections |
| CPU/GPU/file-progress watchdog | Port after lifecycle state | Common Go watchdog driven by typed gate status and measured step history | Derive deadlines from Overgo evidence; do not copy adaptive_new's fixed 600/3600 second defaults |
| Build-constraint-aware CUDA routing | Port the compiler-derived part | Parse Go build constraints and manifest ownership; retain conservative fallback and standing census tests | Reject adaptive_new's family/file-name routing tables as a long-term owner |
| Hygiene coverage plan with selected/skipped reasons | Port the contract | Gate emits required, selected, skipped, unavailable, and why for every impact class | Derive coverage from owners/manifests/imports rather than a large ordered string catalog |
| Gate timing baselines, replay, alarm-rate and injected-regression checks | Adapt | Reuse Overgo run records; add walk-forward false-alarm and injected-power reports before enforcing thresholds | Replace fixed MAD ladders and overlapping windows with calibrated Overgo statistics |
| Process-guard registry | Adapt by derivation | Gate/hook step catalogs generate the guard inventory and verify configured hooks resolve | Do not maintain a second manual JSON statement of code that already owns the guards |
| Doc/schema drift checks | Port selectively | Verify live links, schema-field coverage, generated/snapshot status, skill anchors, and deleted-code references | Prefer deleting stale documents over expanding a large regex rule corpus |
| Resumable close-cycle transaction | Port the idempotence properties | Prepared/final gate records and deterministic reconciliation make interrupted closes resumable | Do not recreate a monolithic close-cycle CLI around mutable docs |
| Run-drought and repeated-axis detectors | Defer as advisory | Add only when their output changes an owner decision and can be derived from RepoDB history | They must never become a governance loop that consumes more attention than it saves |
| `y*p*unblock/wall` priority scoring | Retain as human judgment for now | Optional recommendation evidence later | Inputs are subjective estimates; encoding the formula does not make the ranking objective |
| Monolithic `cmd/state` surface | Reject | Small capabilities in existing Go owners with typed composition | One command with dozens of modes becomes a second cognitive system the model must reconstruct |

### High-value lessons

#### One grounding payload materially reduces model work

Adaptive_new's `-dispatch-context` combines HEAD, dirty paths, workflow phase,
state summary, current priority, plan anchor, and last cycle into JSON. This is
the clearest direct example of the desired cognitive-load transfer: the model no
longer needs to run several commands, reconcile outputs, and infer the legal
phase. Overgo should make this the first implementation slice.

The adaptive_new payload also demonstrates the failure mode. Its current output
names `media-execution-maturity` in the state rank-1 while separately naming
`controller-training-system` as plan live-rank-1. Validation deliberately allows
any active plan row because parallel lanes made exact equality oscillate. That
choice is reasonable internally, but presenting both as rank-1 pushes ambiguity
back onto the consumer. Overgo should distinguish `current_task` from
`campaign_focus`, or omit the latter from task dispatch.

#### Workflow phase ordering is proven; actor independence is not

Adaptive_new derives workflow phase from Git history instead of storing a phase
marker. After an implementation commit, another implementation is refused until
a distinct findings update lands, followed by a fresh priority update. Open SQA
findings can also require reviewed commit, prototype worktree, branch, prototype
commit, and paired checks.

This is worth porting, but no field establishes who performed development versus
review. Overgo's version should retain Git-derived ordering and add immutable
actor/run identities, candidate bytes, evaluator identities, and a rule that the
SQA actor cannot modify the candidate being judged.

#### Gate lifecycle and watchdog semantics are mature

`internal/gateorch` declares step IDs, dependencies, optional conditions,
mutation classes, evidence owners, lifecycle states, run IDs, refusal records,
and terminal outcomes. The watchdog waits until protection is observably ready,
tracks process-tree CPU, GPU utilization and artifact progress, preserves run
identity across malformed status reads, and stamps abandoned or killed runs
loudly. Tests cover concurrent-run refusal, stale status, early watchdog exit,
and terminal persistence.

This is a stronger basis than Overgo's current process-name probe and shell
stall check. The transfer should use Overgo's RepoDB/run-record model and should
solve gate-record debt at the same time.

#### Impact classification contains both a model and an anti-model

Adaptive_new correctly extracted one common Go owner for impact classification
and parses actual Go build constraints to find CUDA-only files. It also carries
large regex, filename, package, model-area, and test-prefix tables accumulated
from incidents. Those tables are useful evidence of what goes wrong, not the
desired Overgo architecture.

Port the constraint parser, additive conservative fallback, and standing census
test. Derive the remaining ownership from Go imports/embeds, kernel manifests,
recipe relations, generated-file provenance, and declared external boundaries.

#### Finding and drift discipline contains useful falsifiers

Adaptive_new's finding validator captures several strong rules:

- A finding title names the defect; drifting measurements live in evidence.
- Trend claims cite deltas, not only endpoints.
- Checks identify checkout-specific prerequisites and exact paths.
- Go-test closures must prove a named test ran rather than accept package-level
  success.
- Claims against a gate first run the actual gate and cite its output.
- Differenced measurements state instrument noise first.
- Closure proposals are not treated as validated plans.

These rules belong in typed finding validation and SQA templates. Their lengthy
historical explanations should remain in Git or incident tests rather than being
copied into Overgo's live documentation.

#### Guard registries are useful only when mechanically cross-checked

Adaptive_new validates that hook commands resolve, registered blocking guards
have an executable nonzero path, and every configured hook guard appears in the
registry. That closes the common gap where a document claims a guard exists but
the hook is missing. Overgo can go one step further: generate the inventory from
the Go gate/guard catalogs and compare it with installed hook configuration,
leaving no manually duplicated command list.

### Transfer order

1. Typed automation context with one current-task identity.
2. Unified evidence honesty across CI, gate, loop dirtiness, and non-Go owners.
3. Typed gate lifecycle, environment-bound retry, heartbeat, and record-debt
   reconciliation.
4. Git/RepoDB-derived review phase with independently identified SQA.
5. Advisory worktree leases, resource estimates, and target-head merge
   eligibility while the owner continues to dispatch manually.

This order first reduces immediate model reconstruction, then makes evidence and
long-running execution trustworthy, and only then adds coordination state.

## Strengths

### Evidence honesty

The strongest recurring design choice is that absence is named. Device and race
lanes fail unavailable, the commit gate parses structured Go test output, plan
verification looks for known vacuity markers, and the gate prints skipped work.
This directly addresses the common failure mode where a green process merely
means the meaningful test never executed.

### Derived scope instead of hand-maintained mappings

Affected Go tests come from the import graph. Smoke coverage comes from the
servable predicate. Kernel ABI checks come from manifests. Compatibility pages
come from structured claims. These are durable derivations and align with the
repository's anti-magic philosophy.

### Plan-bound change control

The plan, verification command, changed path set, commit, and gate record form a
traceable chain. The gate refuses staged paths outside the declared slice and
reports unrelated worktree dirt without sweeping it into the commit. This is
well suited to parallel worktrees.

### Incident-driven hardening

Automation comments name the failures that shaped each control: destructive
worktree removal, shell-eaten commit prose, stale running markers, staged-path
leakage, skipped oracles, and retry-loop tax. This is valuable operational memory
when it remains attached to the mechanism rather than duplicated in status docs.

### Go-native ownership

Core policy lives in testable Go packages. Shell is generally a small hook
adapter. This reduces platform-specific orchestration drift and keeps policy in
the same build, test, and review system as the runtime.

## Weaknesses and failure modes

### A6: Plan state cannot describe manual orchestration decisions

The FIFO plan intentionally leaves scheduling to the owner, but it also lacks a
place to record dependencies, conflicting surfaces, worktree assignment,
resource estimates, or active leases. The human scheduler therefore carries
important transient state mentally.

Required direction: keep dispatch manual while adding advisory task metadata and
RepoDB leases. Record `worktree`, `role`, `depends_on`, `conflicts_with`,
`cpu_threads`, `host_ram_gib`, `vram_gib`, `gpu_exclusive`, and required evidence
lanes. The initial automation should report conflicts and fit; it should not
assign work without owner approval.

### A8: Statistical guarantees exceed current calibration depth

Advisories permit very small baselines, and repeated confirmation windows can
overlap. The implementation is useful as a regression signal, but the terms
"distribution-free" and "independent windows" currently claim more than the
mechanism guarantees.

Required direction: define rank/conformal or permutation calibration, derive
minimum history from the alarm budget, require non-overlapping confirmation
windows, and account for sequential testing and regime changes. Until then,
label the output directional/advisory.

### A9: Hook and guard enforcement is harness-local and fail-open

The guard and loop are wired through `.claude/settings.json`; the shell guard
allows execution when its binary is absent. Raw Git and other clients can bypass
the controls. This matches the guard's stated accident-prevention scope, but it
is not repository-wide enforcement.

Required direction: retain fail-open interactive safety, but add repository-side
verification for integration and protected branches. Report whether hooks were
active in each run record rather than assuming they were.

### A10: Lane and release coverage remains uneven

- The device lane runs the full device set; manifest-scoped selection remains a
  documented target.
- `cmd/device-lane` has no direct unit tests around selection or reporting.
- Hosted CI has no real-GPU lane and the race workflow covers only server and
  inference packages.
- The smoke lane returns success for an empty servable matrix while saying the
  result is "not green"; callers relying on exit status cannot distinguish it.
- Release output still includes historical documents that conflict with the
  doctrine that Git owns chronology.
- The current baseline SBOM is stale, and project-owned source remains
  `NOASSERTION` in `LICENSES.md`.

Required direction: make lane outcomes typed (`pass`, `fail`, `unavailable`,
`empty`) and ensure callers declare which outcomes they accept. Separate durable
release contracts from historical assessment documents.

### A11: Override and stop controls are too narrow for later autonomy

`plan -advance -force` prints a reason but does not create a durable structured
override record. The three valid stop reasons omit evidence corruption,
evaluator contamination, worktree collision, device instability, and loss of
rollback guarantees.

Required direction: record overrides as immutable evidence and add typed
containment stops. These should stop the affected lane or promotion, not
necessarily the whole manually managed campaign.

## Prioritized roadmap

### Phase 1: establish independent development and SQA

1. Enforce distinct identities and clean worktrees for protected surfaces.
2. Freeze evaluators and holdouts before the SQA run.
3. Require target-head findings disposition, rerun evidence, and an independent
   verdict while keeping promotion under owner/external authority.

### Phase 2: make manual orchestration observable

1. Add advisory resource, dependency, conflict, worktree, and role metadata.
2. Add compare-and-set worktree leases in RepoDB.
3. Produce an owner dashboard/report showing active lanes, GPU/RAM reservations,
   stale leases, merge conflicts, evidence status, and suggested next candidates.
4. Add target-head merge eligibility packets and rerun required gates after
   integration.

No automatic assignment or lane termination is required in this phase.

### Phase 3: calibrate scheduling before delegating it

1. Record predicted and actual CPU, RAM, VRAM, wall time, and interference.
2. Measure packing quality, collision rate, abandonment rate, and recovery cost.
3. Let automation recommend assignments and compare them with owner choices.
4. Delegate only low-risk scheduling classes whose recommendations demonstrate
   sustained benefit and reliable recovery.

### Phase 4: bounded autonomous operation

Autonomous dispatch is appropriate only after leases, independent SQA, sealed
promotion evidence, containment stops, rollback, and scheduler calibration are
all enforced. The owner should then move from being the dispatch loop to being
the objective, exception, and promotion authority.

## Suggested success measures

- Zero CI-green runs with required skipped or unavailable evidence.
- Zero turn ends with unreported meaningful worktree dirt.
- Every gated commit has either a finalized RepoDB gate record or visible debt.
- Every protected candidate has distinct developer and SQA identities.
- Every merge-eligible packet is evaluated at the target head.
- Resource predictions include error bounds and improve against recorded actuals.
- No autonomous scheduling class is enabled without measured recovery behavior.
- The owner can understand active work, evidence gaps, and resource use from one
  concise view without surrendering control.

## Immediate baseline findings

- Focused automation packages pass their tests at this snapshot:
  `cmd/loophook`, `cmd/plan`, `cmd/gate`, `internal/guard`,
  `internal/testevidence`, `internal/runrecord`, and `internal/repodb`.
- Kernel-manifest and compatibility checks pass.
- `go run ./cmd/sbom -check` fails because `SBOM.cdx.json` is stale.
- Project-owned Go and CUDA sources have no declared distribution license.
- The scheduler remains intentionally human-operated; this report does not
  classify that boundary itself as a defect.
