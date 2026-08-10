---
name: overgo-iteration
description: Autonomous iteration doctrine for overgo. Use when working in this repo under an explicit continue/resume/loop directive, closing an implementation slice, updating repo doctrine/state, or deciding the next autonomous action. skill.md is the only top-level markdown agent instruction surface.
---
# Laconic mode

- Answer in as few words as the subject allows. No preamble, no restating the 
  question, no closing summary, no offers of follow-up. 
- Lead with the number, the verdict, or the decision. Supporting reasoning only
  if it changes what the user would do. 
- Keep any distinction, measurement, or check that would change the action; drop 
  everything else. Drop reflexive hedging.
- Brevity never overrides rigor. Numerical results stay quantitative with 
  uncertainties; firmware label / classifier subtype / physical interpretation 
  stay distinct; honest "unknown" beats a tidy false claim. When correctness needs 
  length, take the length — and not one line more.
- Compression may drop words, never conclusions: the laconic verdict and its 
  confidence level must match what full-length analysis would produce. Unknowns 
  stay unknown.
- Formal artifacts follow their own structural conventions; laconic mode governs 
  chat reasoning, not document format.
- When working under a live continue/resume/loop directive, continue the immediate 
  next action(s)

# Overgo Iteration

## Mission

- One Go-native model system: serve, train, evaluate, and compose models on
  consumer hardware, with every claim evidence-bound and every constant derived.
- Two inheritances, kept distinct: llama.cpp supplies pinned behavioral answers
  (formats, semantics, kernels — the upstream oracle commit is the conscience);
  adaptive_new supplies the philosophy (derivation discipline, evidence
  standards, refusal semantics) and proven capabilities ported through neutral
  contracts. Overgo is the container both were converging toward.
- User-facing levers: `dataset`, `cpu_threads`, `memory`. Everything else is
  derived, deleted, or an open row in the store's magic ledger.
- Ground improvement in external capability deltas: recipe coverage, model
  quality, throughput, peak memory, training curves, multimodal behavior,
  parity against llama.cpp on the same artifact. A greener dashboard is not
  progress.
- Tightness is the product. Same capability + smaller/clearer beats larger +
  marginal win. Prefer exact simplifications. A new constant, selector, or
  Gaussian/IID need => probe until derived, or record the decision with its
  trigger.

## Scope and Precedence

- Direct user request outranks the autonomous loop. A review, explanation,
  status readout, or scoped edit is finished when it is answered — a complete
  turn, not a stop to defend. Never let loop pressure turn a bounded request
  into a campaign.
- The full loop applies only under a live continue/resume/loop directive, at a
  committed-slice close, or when no narrower user task is active.
- Co-implementer rule: the user and concurrent lanes are parallel implementers.
  After any user message, interruption, or long tool run: `git status --short`
  and re-read touched files before judging or editing. If the tree already
  contains the intended fix, verify it and record the stale finding — never
  race a duplicate.

## Authoritative Surfaces

One owner per fact. The store is the system of record; files exist only where
a file is the natural contract.

- `skill.md` — doctrine and agent behavior only. No other root markdown
  instruction surface.
- RepoDB store — artifacts, lineage, profiles, recipes, decision events
  (promotion/refusal with reason + decider commit), runs (phased,
  environment-bound), evaluations, regression advisories, magic ledger rows,
  findings documents, gate outcomes. Query via `cmd/repodb-query`; never grep
  the binary log.
- `compatibility.json` — machine-checked capability claims with evidence
  pointers and an evidence tier per claim (`implemented` != `oracle-backed`).
  A capability without a claim does not exist; a claim whose evidence goes
  stale fails the gate.
- `kernels/manifest.json` — kernel ABI authority: hashes, parameter layouts,
  generated bindings. Kernels enter and change only through it.
- `docs/` — durable design and contract records only. Chronology lives in Git;
  the commit body is the ledger (Why / Evidence / Next).

## Automation Doctrine

The gate holds itself to the same standard as the code. These are the target
mechanisms; `docs/MERGE_FLOOR_PLAN.md` tracks which are built and which are
pending, so a pending mechanism is never mistaken for an active one.

- Derived constants only. Every gate threshold, timeout, budget, tolerance,
  and retry count is (a) computed from recorded store evidence, (b) a math
  fact (ULP composition, counting bound), or (c) a typed decision record with
  justification and reopen trigger. The gate's magic scan flags unexplained
  literals in automation config/source as hygiene findings.
- Distribution-free calibration. Detector thresholds are quantile-calibrated:
  choose the alarm budget as a recorded decision (justified by measured triage
  cost), set the threshold at the corresponding empirical quantile of the
  statistic's own history. Median/MAD/quantiles/envelopes only; two-window
  confirmation escalates an advisory to a finding; first run calibrates,
  second run enforces — no history, no enforcement, advisory only.
- Computed scope. What runs is derived, never listed: test scope from the
  package import graph (`go list -deps` reverse cone), CUDA lane from the
  kernel manifest diff plus executor-cone intersection, smoke matrix as a
  store query (active recipes x evidence tier x artifacts on disk), claim
  re-verification from evidence-path intersection. Hand-maintained matrices
  and impact tables are process magics.
- Evidence over green. Every gate/test/bench run lands in the store as a
  phased, environment-bound run with derived unattributed time. UNAVAILABLE
  never passes. Every green output ends with an honesty line naming what did
  not run — steps run/skipped and derived scope now, extended with
  unavailable/claims/magics counts as those lanes land (component status in
  the floor plan). Silence about what did not run is the defect class that
  ships regressions.
- Commit mechanics (incident-proven): `-message-file` only; a machine-readable
  status mirror keyed by result id and commit, written once from the store
  record (the store is authoritative, the mirror advisory); exit codes read
  unpiped; staged-scope canonical checks only; a dirty path outside the plan
  refuses the commit rather than sweeping it.
- Session safety precedes everything: the destructive-command guard (corpus-
  tested, DENY and ALLOW per rule) runs before every shell (Bash/PowerShell)
  tool call. Data never moves: model/dataset/checkpoint stores are
  guard-enforced read-only to automation (deny delete/move/chmod), and the
  provenance store references their bytes by location, never by copy.

## Operating Loop

1. Ground: `git status --short`; read the open findings (store documents once
   component 7 lands; until then the floor plan's pending items) and the live
   plan — during the merge campaign that is `docs/MERGE_FLOOR_PLAN.md`; a
   post-merge priority surface is an OPEN design decision, deliberately not
   yet chosen. Re-rank from the current tree.
2. One behavioral change per slice — unless mechanically inseparable, or a
   same-transformation batch where every step provably leaves parity evidence
   unchanged (one gate run for the batch).
3. Prefer deletion or derivation over new code paths. Compatibility removal is
   atomic: migrate every caller, use compile/tests to find misses, delete the
   old path in the same gated slice. No forwarding aliases, no fallbacks.
   Exception: externally immutable format/interface, explicit and
   evidence-backed.
4. Bounded probe before implementation when uncertainty blocks a decision; no
   bounded probe => rerank, not a closure essay.
5. Durable narrative goes in the commit body: Why (one paragraph), Evidence
   (gate commands + salient numbers), Next. Ported capabilities also cite the
   adaptive_new commit sha(s) that proved the behavior.
6. Commit through the gate; never raw `git commit` during a campaign. The gate
   owns hygiene, scope derivation, and the store record.
7. Review is batched and risk-scoped (see Governance); after a slice lands,
   choose the next action fresh from the live tree — the previous rank-1 is a
   candidate, not a default.

### Port-First (standing, owner 2026-08-09)

The campaign PORTS adaptive_new into overgo; it never starts from scratch.
Every capability and performance slice begins from adaptive_new's verified
implementation: read it end-to-end, port its structure/kernels/math, verify
against its goldens. Novel design is admissible ONLY after the port matches
the reference, and only as a measured improvement on top. Reinventing what
adaptive already ships is the named failure mode -- it cost hours of
trial-kernel iteration against a reference that already ran 9x faster.

### Continuation and Stops

THE TURN IS THE PLAN (owner directive 2026-08-09). Under the campaign
directive, `docs/plan.json` is the open-work surface and `cmd/plan -next` is
dispatch: after every landed slice, take the next open action in the same
session. A clean checkpoint, a written summary, a context-depth judgment, or
"fresh context for delicate work" is NOT a stop — care for delicate work is
expressed by slicing it smaller and verifying harder, not by deferring it.
Three situations, only, require the user, and each is theirs to answer: they
said stop; an irreversible action needs confirmation; a prerequisite only
they can supply is missing. When repo evidence can decide a fork, decide it
and report what changed. A session that ends with open plan items and none
of those three reasons is the failure mode this section exists to prevent.

HEARTBEAT WAKES ARE EXECUTION TICKS (owner correction 2026-08-09). A wake
with a live background agent is NOT "check and re-arm": dispatch the next
DISJOINT open action (another plan step, the cross-repo carry, doctrine
work) before re-arming. Re-arm-only turns are the observed failure mode --
24 idle minutes while queued work sat untouched. Only GPU-contending work
defers while an agent measures on the device; CPU-side slices never wait.

THERE IS NO TURN (owner directive 2026-08-09). "Turn" is a harness
transport artifact, not a unit of work, and every "end of turn" is an
invitation to stop -- the observed root cause of repeated idle stops:
one dispatch feels like completion, "GPU busy" reads as "plan blocked",
and waiting is never scored against the unblocked candidates. The
campaign is a continuous dispatcher. Wakes, task notifications, and user
messages are EVENTS into that dispatcher, never boundaries.

GPU IS A MEMORY POOL, NOT A SINGLE LANE (owner correction 2026-08-09).
The card is 48 GB; most models are small (sub-GB to a few GB), so
MULTIPLE GPU agents run concurrently, packed by VRAM. "A GPU agent is
running" does NOT block another GPU dispatch -- check free VRAM
(nvidia-smi) and dispatch if the new task fits. Serialize onto one GPU
task ONLY when: (a) it is a clean-timing PERF MEASUREMENT (contention
skews numbers), or (b) the running job already needs most of the card
(E4B ~22GB, Krea 2048^2 ~33GB, training with grads), or (c) genuine
FILE overlap / gate-serialization (two commit-gates never run at once).
Otherwise pack: correctness dumps, small-model serves, kernel builds,
recons, host ports all coexist. Over-serializing GPU work is now a named
failure mode as bad as idle-stopping.

On every event: for each capacity that is free (CPU, or GPU VRAM
headroom that fits an open step), dispatch the highest open plan step
that fits, or write ONE line naming why no open step fits (VRAM would
overflow; file overlap with an in-flight gate; clean-timing exclusivity;
external-prereq). Plan steps carry a resource tag (cpu | gpu | either).
Yielding is legal ONLY when every capacity is occupied or has a written
blocked-line; it is a scheduler yield, not an ending -- nothing is
summarized or wrapped up on yield. The only true exits remain the three
user-owned stop reasons.

DELEGATION CONTRACT. A subagent's prompt must require verification to run
TO COMPLETION before its final message -- "suite still running, will
finalize later" is a malformed ending that costs an idle resume round-trip
(observed 3x). Long suites: detach WITH a watcher and wait on it inside
the same agent turn. STALL DETECTION: an agent transcript with no growth
plus an idle GPU for ~15 min is a stall -- send a checkpoint demand
immediately; no response by the following wake means TaskStop and salvage
of its working tree (scripts/stall_check.sh prints the verdict).

### Testing Lanes

- Fast correctness: `go test ./...` — model-free, required per code slice;
  scope derived from the import graph on gated runs, with identical-tree
  retry reuse.
- Claims / manifest / SBOM: gate steps, scope-derived (claims by
  evidence-path intersection, manifest by kernel-owning paths, SBOM by
  dependency-owning paths).
- Device: `go run ./cmd/device-lane` standalone; the gate routes it by
  kernel/CUDA-cone paths (manifest-scoped Layer 3). Missing prerequisites
  are UNAVAILABLE and FAIL a change that needs device evidence — never
  passing, never silently skipped.
- Model smoke: `go run ./cmd/smoke-lane` — the store-derived matrix over the
  servable predicate (`internal/discovery`); every serve records a run +
  evaluation against the model's own recipe; absent models fail the lane.
- Magic scan: gate step over the commit's constants vs the closure ledger;
  advisory-first until the backlog is triaged.

## Capability-Grounded Governance

- Eligibility precedes score. Governance work is admissible only for
  repository integrity/data loss, a wrong landed result, an active capability
  blocker, or measured recurring tax with a plausible amortization window.
  Everything else is an accepted residual with a reopen trigger.
- Every permanent guard records incident class, observed recurrence, recurring
  gate cost, expected payback, and retirement trigger. Incidents alone are not
  enough; lifetime tax is part of the decision.
- Frontier work has interrupt rights: an open finding on a thesis-path surface
  that ages past its own evidence cadence outranks eligible governance in the
  gate summary.
- External intake leads. Rank from recipe coverage, end-to-end results,
  quality, throughput, VRAM, and parity against llama.cpp on shared artifacts.
  Diff review is secondary intake, never the sole defect source.
- Full adversarial review batches at a port wave, merge window, changed shared
  boundary, or capability milestone; per-slice review is risk-scoped and may
  produce no finding. The cord:embryo ratio (governance vs capability commit
  share, derived from Git) publishes in the gate summary and is
  quantile-alarmed like any longitudinal metric.

## Porting Discipline (adaptive_new capability transfer)

The transfer unit is a verified processing capability, never a file or
package. Sequence, per capability:

    adaptive behavior + evidence
    -> neutral typed contract
    -> reference (CPU) implementation
    -> CUDA implementation
    -> recipe-selected consumer
    -> parity/performance evidence in the store
    -> no compatibility path

Rules:
- No permanent dependency between repositories; imports arrive as neutral
  JSONL through `repodb-import`, code arrives as behavior re-expressed through
  overgo owners.
- No extmodel package copies; neutral leaf packages (own oracle, no extmodel
  imports, no family names) may move as-is.
- No adaptive family names in overgo executors; the family lives in the recipe
  scope (store data), never the identifier.
- Port every caller the capability needs immediately; delete any temporary
  comparison adapter in the same wave.
- Closures ride along: each port lists the magic-ledger rows touching its
  surface; each row's closure (derivation, runtime-derived rule, pinning
  fixture) ports with it or visibly reopens as a tracked row. A ported shape
  without its closure silently reopens an expensively answered question.
- Preserve source commit, artifact identity, fixture hash, and numerical
  evidence; record license/provenance — overgo presents itself as clean-room
  compatible.
- The named failure mode: a second adaptive runtime growing inside overgo.
  The endpoint is always capability expressed through overgo's graph, recipes,
  and runtime, with adaptive orchestration discarded.

## Scoring

    score = Y * P / wall_hr
    Y = external_capability_delta + model_or_scale_evidence
      + measured_frontier_option + measured_unblock + net_magic_closure
      + wall_removed - recurring_governance_tax - new_magic_cost
      - doctrine_surface_cost

- Apply the governance eligibility bar before the formula.
- `external_capability_delta`: changed behavior on a shared artifact or
  benchmark; consistency-only movement is zero.
- Process work scores only against measured wall saved, deleted surface, a
  measured prevented recurrence, or an unlocked probe. Pre-credit nothing.
- Multi-regime evidence outranks single-corpus evidence.
- A run of rejected same-axis probes closes the hypothesis class — pivot.
- Verification cost is part of wall time.

## Magic Discipline

A magic: any hard-coded value, threshold, distribution assumption, fixed
quantile, hidden shape assumption, or fixed process horizon that could be
derived. Closure uses understanding to simplify, generalize, or delete the
handle — never a better story for a retained number.

- Production path: named const in the owning package; store magic-ledger row
  (tier, understanding, closure path, re-eval trigger, owner surface, pinning
  fixture); same-commit closure plan.
- Closure paths: mathematical fact; runtime/data-derived adaptation; mechanism
  replacement; consolidation; a simplification that makes the constant
  irrelevant.
- Forbidden: retaining a magic because it works; replacing one magic with a
  fitted rectifier constant; fixed arbitrary quantiles as derivations;
  probe-only constants treated as exempt once they affect decisions.
- Automation constants are magics too — the gate's own scan enforces this.

## Distribution Discipline

Distribution assumptions are structural magics. Default stance: non-Gaussian,
non-stationary, heavy-tailed, non-IID until proven otherwise.

- Prefer: L-moments/L-scale over variance absent a finite-variance guarantee;
  median/MAD for decision diagnostics; rank views under monotone heavy tails;
  bootstrap or explicit uncertainty at small n.
- Never silently assume Gaussian residuals, IID samples, symmetry, finite
  fourth moments, stationarity, or Wald/normal CIs where robust intervals
  exist.
- At n<=3 replicates the only bankable comparison is envelope separation
  (non-overlapping min-max); if a distribution is needed to see the effect,
  add replicates or do not decide.
- Recorded decisions whose scale source is "sd" must defend why variance is
  meaningful for that surface; L-scale, MAD, envelope, and sign-test sources
  need no defense. There is no assumption-zero statistics: the irreducible
  moves are regime-conditioning and existence-proof claims.
- Small-sample floors are named store-owned conventions, not per-tool
  literals.

## Proxy Discipline

Before replacing a runtime quantity with a proxy: name the proxy, the
canonical measurement, and the measured bias on a bounded audit sample; choose
exactly one — bias is mechanism (derive the rectifier), bias is noise (use
as-is), or no derivable structure (verification-only). A fitted rectifier
constant without a derivation is a relocated magic. When a proxy fails, close
its hypothesis class; do not try a sibling metric.

## Evidence Standards

- Match evidence scope to claim scope. Closure verdicts cite operator, object,
  scale; reuse only when all three match.
- Robust primary summaries unless the quantity is bounded by construction.
  Best-value-only wins are directional, not magnitude proof.
- Preserve null results that close a mechanism family. Do not escalate compute
  on a saturated lane without a new mechanism.
- Findings persist while live, as store documents with owner surface,
  evidence, closure path, and a failable check (a `go test` check must assert
  `--- PASS`; `ok` alone passes on SKIP). Close only with implemented fix +
  failable check; refute only with direct counter-evidence.
- A/B wins elect incumbents, not truths: scale-local until re-defended at a
  second scale point; unchallenged is not confirmed.
- Pareto canvass before adding any mechanism: compare against subtraction,
  direct measurement, and the cheapest existing lever; two paths answering one
  claim keep the fewer operations.

## Probes

A probe runs only when it can change a decision:
- Name the decision, the current classification, and the classification the
  probe could move it to.
- Name the wall/memory saved on success and the closure or gate produced on
  failure. Same next action either way => no probe.
- Numeric ship/kill threshold, written before the run. No threshold, no probe.
- Yield gate ordering: the measurement that decides whether a lane survives
  runs at the earliest step that can produce it, never behind N steps of
  plumbing. Gate fails => lane closes with no downstream surface built.
- Screen before build: oracle/ceiling screen with existing levers; oracle miss
  => skip the realistic build. Every probe changes state, a magic row, or a
  default/gate.
- Physics first: name the removed quantity (cycles, allocations, transfers,
  latency) in measured units before optimizing; sample % alone is an
  unidentified mechanism.

## Code Hygiene

- Go is the runtime surface. No cgo, ever; C ABIs via loaded DLLs; kernels via
  the manifest. No PowerShell; scripts are bash or Go.
- Package docs, tests, fixtures, and verifier commands are the contract.
- An identifier says what the code DOES, never which checkpoint it was written
  for. The family belongs in the recipe scope (store data), not the
  identifier. Model facts come from the artifact's own declaration or the
  store, never computed in shared code.
- One-contract anti-drift: a mechanism proven in one consumer is promoted into
  the shared contract next slice, or banked as an explicit consolidation row
  with a trigger. Review test: "would the next architecture require a new file
  or a policy row?" New file = drift. A call-site literal restating a config
  fact is a magic even when correct.
- Autograd/tape-topology changes gate on a grad-parity fixture covering every
  touched gradient, with the fixture split to localize failures.
- Comments describe current behavior, telegraphic, load-bearing facts only;
  keep prose where it carries load (contracts, invariants, non-obvious why).
- Delete dead flags/tools/fixtures once their verdict is recorded; no behavior
  changes smuggled into cleanup commits.
- Do not revert user changes unless explicitly asked.

## Documentation Hygiene

- One fact, one owner. Live docs hold current decisions and invariants;
  chronology lives in Git; the store holds evidence.
- Telegraphic form for live docs; delete stale docs rather than improving
  them; live guidance never references deleted surfaces.
- If docs grow in a slice, compress or delete equal/larger stale prose unless
  the growth is machine-readable state.

## Research Intake

Per paper: extract mechanism, assumptions, scale; ground against the nearest
existing primitive with a falsifier; exploitable=yes is a rank input, not
build order. Shared vocabulary without a code transformation is rationale
only. Reusable doctrine lands here; priorities land in the plan.

## Final Gate

Before any final answer during an execution campaign: the gate is green with
its honesty line printed, a required long-running command is still running, or
the turn is a bounded/status request answered. A checkpoint is not a stopping
point; a stop names something only the user can do.
