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
phase should strengthen observation, isolation, and independent SQA without
prematurely taking scheduling authority away from the owner. Merge eligibility
remains deferred until an integration consumer needs it.

Independent admission now validates immutable developer/SQA identities, clean
separate worktrees, frozen evaluators, exact target heads, complete findings,
and approved verdicts from RepoDB. Git and RepoDB now derive review priority,
and owner-authored leases expose worktree/resource state. Assignment, integration,
and promotion appropriately remain owner decisions.

## Incremental implementation status

- Current `master` is an ancestor of this branch; the latest merge gate passed
  scope, profile, formatting, vet, build, impacted tests, and evidence routing.
- `cmd/plan -context` emits one deterministic task identity with Git/worktree
  state, role, normalized dirt, and evidence debt. Plan validation now refuses
  any open step without a runnable verifier.
- Gate lifecycle is typed and RepoDB-backed: preparation precedes Git mutation,
  finalization is atomic, retry identity includes the environment, and record
  debt is explicit and reconcilable.
- Independent review admission binds immutable developer, SQA, worktree,
  evaluator, candidate, finding, and verdict identities. Owner-authored leases
  expose resource and conflict state without taking dispatch authority.
- Structural profiling reports production/test/validator mass, exact-clone
  excess, exports, and imports against `HEAD`. Ranking is advisory; parity
  evidence and semantic ownership remain authoritative.
- An empty smoke matrix is a typed nonzero `empty` outcome. CI/release test
  evidence rejects unclassified skips and unavailable required prerequisites.
- The measured one-consumer `gatecontrol` abstraction remains deleted.
  Multi-consumer evidence and JSON mechanics use existing common owners;
  command-local coordination stays local.

Open automation work is now calibration, repository-side enforcement telemetry,
lane outcome completeness, durable override/containment records, and measured
scheduler recommendations. Completed implementation history remains in Git.

## Operating boundary

Current division of responsibility:

| Responsibility | Current owner | Desired near-term state |
| --- | --- | --- |
| Select and assign worktree tasks | Human owner | Human, assisted by ranked candidates and resource estimates |
| Allocate CPU, RAM, and GPU capacity | Human owner | Human-approved reservations with measured estimates |
| Implement a capability slice | Development lane | Unchanged |
| Adversarially review a candidate | Informal/separately requested lane | Distinct, recorded SQA identity and clean worktree |
| Determine merge eligibility | Human plus gate output | Defer a machine packet until an integration consumer exists; human decision |
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

Overgo tracks code structure as a versioned evidence vector, not one quality
score. `internal/repoanalysis` owns content-identified, generated/test-aware,
parse-once Go snapshots; `internal/codeprofile` and `internal/closurescan` are
its two consumers. The profile records production/test AST mass, function node
and branch counts, exported declarations, import edges, and normalized exact
whole-function clone groups. Operators, selectors, and control structure remain
significant; local identifiers and literal values are canonicalized. Tests and
production are never grouped together.

The gate stores the complete profile and reports candidate-versus-`HEAD` deltas
for production/test files and nodes, validator-shaped function mass, clone
excess, clone/function counts, exports, and imports. Changed-path candidates and
exact clones are ranked separately as production, validator, or test signals. A
validator signal is deliberately narrow and inspectable: a non-test `validate*`
function returning `error`. These are inspection order, not defect verdicts.
There is no composite score, ceiling, growth budget, exception workflow,
suppression ledger, or mandatory favorable direction.

The effective loop is advisory and deletion-led: rank candidates; inspect
semantic ownership and numerical contracts; migrate every caller; delete the
displaced path; rerun the profile; use parity tests and gates as the behavioral
authority. Exact-clone similarity never proves numerical equivalence. A lower
duplicate count never offsets unreported production, test, export, or AST
growth.

The gate flags the adverse pattern "duplication fell while production surface
grew" and states that the reduction does not offset growth. It also requires
the reviewer to inspect semantic ownership and numerical contracts, migrate
callers, delete displaced paths, and obtain parity evidence. The profile remains
targeting/accountability automation, never an autonomous refactoring verdict.
Partial statement clones remain deferred until whole-function focus demonstrably
misses recurring hygiene problems.

## Capability inventory

### Plan and loop control

- `cmd/plan` owns a compact, open-work-only queue and selects the first open
  step. It can add work, define verification, generate a task prompt, verify,
  advance, compact, and record one of three recognized stop reasons. The shared
  validator refuses an open step without a failable verifier; blocked steps name
  their unavailable prerequisite instead.
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

- Smoke, device, race, and release share `runrecord.LaneOutcome`:
  `pass`, `fail`, `unavailable`, and `empty`. Typed non-passing results
  retain their outcome through wrapped command errors.
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
phase. Overgo's `cmd/plan -context` now emits the narrower, single-task form.

The adaptive_new payload also demonstrates the failure mode. Its current output
names `media-execution-maturity` in the state rank-1 while separately naming
`controller-training-system` as plan live-rank-1. Validation deliberately allows
any active plan row because parallel lanes made exact equality oscillate. That
choice is reasonable internally, but presenting both as rank-1 pushes ambiguity
back onto the consumer. Overgo exposes one `current_task`; campaign context
does not compete with dispatch identity.

#### Workflow ordering needs immutable actor identity

Adaptive_new derives workflow phase from Git history instead of storing a phase
marker. After an implementation commit, another implementation is refused until
a distinct findings update lands, followed by a fresh priority update. Open SQA
findings can also require reviewed commit, prototype worktree, branch, prototype
commit, and paired checks.

Adaptive_new did not establish who performed development versus review. Overgo
retains Git-derived ordering and binds immutable actor/run, candidate, evaluator,
finding, and verdict identities; review admission rejects candidate mutation.

#### Gate lifecycle and watchdog semantics are mature

`internal/gateorch` declares step IDs, dependencies, optional conditions,
mutation classes, evidence owners, lifecycle states, run IDs, refusal records,
and terminal outcomes. The watchdog waits until protection is observably ready,
tracks process-tree CPU, GPU utilization and artifact progress, preserves run
identity across malformed status reads, and stamps abandoned or killed runs
loudly. Tests cover concurrent-run refusal, stale status, early watchdog exit,
and terminal persistence.

Overgo transferred the lifecycle properties through RepoDB/run-record evidence,
environment-bound retries, typed heartbeats, and explicit gate-record debt.
Resource-progress watchdog calibration remains open.

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

### Transfer status

The first transfer wave is complete: one task context, evidence honesty, typed
gate lifecycle and reconciliation, independent SQA identity, and advisory
worktree/resource leases all have exercised Go consumers. Remaining
adaptive_new mechanisms are admitted only when they delete model work through an
existing Overgo owner; no monolithic state controller is planned.

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

### A6: Manual orchestration is observable only at the exercised lease boundary

Owner-authored RepoDB leases now record task, worktree, branch, role, target
head, conflicts, CPU/RAM/VRAM request, GPU exclusivity, and expiry. One CAS alias
owns each worktree; `cmd/plan -lease-report` reports active reservations and
collisions without assigning, terminating, merging, or promoting work.
`cmd/plan -record-lease-outcome` now binds predicted and actual CPU, RAM, VRAM,
wall, overlap interference, collision, abandonment, and recovery measurements
to an exercised lease. Deterministic evaluation reports finite-sample maximum
errors, resource packing totals, collision/abandonment/recovery counts, and
componentwise recommendation wins against owner-approved reservations; it does
not assign work or claim statistical confidence.

Dependency, evidence-lane, and merge-eligibility APIs were removed because they
had no production consumer. Add them only with the integration path that reads
them and with owner authority explicit in its output. Until then Git, admitted
review evidence, and the owner remain the merge decision.

### A8: Advisory calibration remains directional

Advisories now derive minimum history from the recorded alarm budget, calibrate
only on disjoint blocks, and hold the latest block out. They label empirical
thresholds as directional, disclaim false-alarm guarantees, and require
non-overlapping windows before opening a finding. This removes the prior
overclaim that median/MAD alone made the result distribution-free.

Repeated sequential looks and regime changes remain outside that empirical
per-look calibration. The command reports this limitation explicitly and does
not turn directional evidence into an enforcement decision.

### A9: Hook and guard enforcement is harness-local and fail-open

The guard and loop are wired through `.claude/settings.json`; the shell guard
allows execution when its binary is absent. Raw Git and other clients can bypass
the controls. This matches the guard's stated accident-prevention scope, but it
is not repository-wide enforcement.

The gate now verifies `.github/protection.json` against the configured hooks
and pull-request workflow, and records those repository facts in its immutable
step evidence. Parent-harness activation and host branch enforcement remain
external facts: the gate records activation as unobserved instead of inferring
it from configuration. Retain fail-open interactive safety and report whether
hooks were active in each run record rather than assuming they were.

### A10: Lane and release coverage remains uneven

- The gate passes changed paths to the device lane, which deterministically
  selects the affected internal packages plus the CUDA smoke probe. An empty
  path set retains the explicit full-device mode; direct tests pin both routing
  and reporting.
- A manual real-GPU workflow targets a labelled self-hosted Windows runner and
  invokes the same typed device lane; missing hardware is a non-passing
  `unavailable` outcome. Runner registration and host branch enforcement remain
  external owner infrastructure. The race workflow still covers only server
  and inference packages.
- Smoke distinguishes `empty` from success; smoke, device, race, and release
  now share one typed result contract.
- Release output contains durable compatibility, import, licensing, and SBOM
  contracts; chronological assessments remain repository history. The SBOM is
  regenerated from the current dependency graph. Project-owned source remains
  `NOASSERTION` pending the owner's license decision.

Required direction: expand race coverage only from measured concurrency risk.

### A11: Override and stop controls are too narrow for later autonomy

`plan -advance -force` now records immutable override evidence before mutating
the plan. `plan -contain` records a lane-scoped event for evidence corruption,
evaluator contamination, worktree collision, device instability, or unavailable
rollback. These events contain the affected lane or promotion and do not
masquerade as a stop for the whole manually managed campaign.

## Prioritized roadmap

### P0: enforce trustworthy task and evidence contracts

1. Keep the open-step verifier invariant enforced by `internal/plan`.
2. Calibrate advisory false-alarm behavior against non-overlapping history;
   retain explicitly directional labels until that evidence exists.

### P1: close repository and containment gaps

1. Record hook/guard activation in run evidence and verify protected integration
   paths repository-side while retaining fail-open interactive safety.
2. Record forced advances as immutable override events.
3. Add typed lane-scoped containment reasons for evidence corruption, evaluator
   contamination, worktree collision, device instability, and lost rollback.
4. Regenerate the SBOM, resolve source-license authority, and remove historical
   documents from release payloads.

### P2: calibrate recommendations under human dispatch

1. Record predicted and actual CPU, RAM, VRAM, wall time, and interference on
   exercised leases.
2. Measure packing quality, collision rate, abandonment, and recovery cost.
3. Compare deterministic recommendations with owner choices; add dependency or
   merge-eligibility packets only with their first integration consumer.

### P3: bound autonomy by measured recovery

`internal/plan.AssessAutonomy` marks a scheduling class eligible only when every
observed comparison is a recommendation win, collisions and abandonments are
absent, developer and SQA evidence are distinct, and sealed promotion,
lane-scoped containment, rollback, and measured recovery evidence are present.
It does not dispatch or promote; the owner remains objective, exception, and
promotion authority.

## Suggested success measures

- Zero CI-green runs with required skipped or unavailable evidence.
- Zero turn ends with unreported meaningful worktree dirt.
- Every gated commit has either a finalized RepoDB gate record or visible debt.
- Every protected candidate has distinct developer and SQA identities.
- Any future merge-eligibility packet is evaluated at the target head and is
  introduced with its integration consumer.
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
- Smoke's empty-matrix contract is tested and exits nonzero with
  `outcome=empty`.
- Every current open plan step has a nonempty verifier; validation now rejects
  regressions before dispatch or save.
- Project-owned Go and CUDA sources have no declared distribution license.
- The scheduler remains intentionally human-operated; this report does not
  classify that boundary itself as a defect.
