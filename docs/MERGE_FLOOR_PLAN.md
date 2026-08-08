# adaptive_new -> overgo merge floor plan

The floor is the automation and provenance substrate that must exist in overgo
BEFORE any adaptive_new capability ports land. Capability arrives through
whatever verification exists when it commits; building the floor first makes
every port gate-grade from its first commit, and inverts adaptive_new's
history (capability first, rigor retrofitted at ~0.8 governance commits per
capability commit) into rigor-first with capability arriving already audited.

Direction and rules are owned by [`skill.md`](../skill.md) (Porting
Discipline, Automation Doctrine). This document tracks the floor's components,
their status, and the acceptance bar. Chronology lives in Git; this file holds
current state only.

## Sequencing principle

    floor  ->  one small pilot port  ->  calibrate  ->  full-rate organ transfer

Two dependency directions, each with a named failure mode:

- Some floor pieces MUST precede ports: guard (session safety before heavy
  agent work near model stores), gate (so parity claims are trustworthy), the
  magic ledger (so the per-port closure audit is a query, not a hand-catch),
  evidence tiers (so ported breadth cannot inherit evidence it lacks).
- Some floor pieces CANNOT precede ports: the statistical layer's thresholds
  derive from recorded run history. Building them before runs exist means
  asserting their constants -- the exact violation they exist to enforce. They
  calibrate WITH the wave-1 pilot, which is their first corpus.

## Component status

| # | Component | Status | Owner surface |
|---|-----------|--------|---------------|
| 0 | Destructive-command guard | DONE `df84648` | `internal/guard`, `cmd/guard`, `scripts/guard.sh`, `.claude/settings.json` |
| 1 | Commit gate | DONE `df84648` | `cmd/gate` |
| 2 | Claim evidence tiers | DONE `df84648` | `cmd/compatibility` |
| 3 | Magic-ledger export contract | DONE (overgo side) | `internal/repodbimport`, `cmd/repodb-import`; producer `cmd/repodb-export` in adaptive_new |
| 4 | Magic ledger resident in store | DONE (59 rows) | overgo RepoDB store |
| 5 | Closure scan / native magic baseline | PENDING | `cmd/closure-scan` (to build) |
| 6 | Statistical / longitudinal layer | DEFERRED to wave 1 | `internal/runrecord` advisories + gate wiring |
| 7 | Exporter durable commit + remaining scopes | PARTIAL | `cmd/repodb-export` (adaptive_new) |
| 8 | PowerShell retirement (Go/bash automation) | PARTIAL | `scripts/*.ps1`, README, docs |

## Built components

### 0. Destructive-command guard (DONE)

PreToolUse safety engine, ported from adaptive_new at its current hardening
(quoted-target and executable-substitution rules). Adaptations: `OVERGO_COMMIT_GATE`
exemption env; `repodb-store` added to protected paths; adaptive-specific
rules (record-cycle prose) dropped. One rule the lineage guard lacked:
`find -delete` / `-exec <delete-verb>` over a protected path (the delete
target is the substituted `{}`, so no verb-scoped rule sees it).

Corpus: 42 cases, DENY and ALLOW per rule, non-vacuity enforced by the test
(a one-directional rule fails the suite). Fail-open on unparseable input; the
corpus, not the binary, is where strictness lives.

Incident lineage covered: 300 GB force-delete, 170 GB worktree-junction
delete, backtick command substitution eating commit prose, heredoc backslash
mangling, raw `git commit` bypassing the gate.

### 1. Commit gate (DONE)

`cmd/gate -message-file <path> -paths <csv>`. Steps, in order:

1. **scope** -- staged path outside `-paths` REFUSES (the commit ships the
   whole index regardless of pathspec); unplanned dirty paths are reported in
   the honesty line, never swept; planned paths must exist or be tracked
   deletions.
2. **fmt / vet** -- scoped to changed `.go`; skipped (honestly) when none.
3. **build** -- `go build ./...`.
4. **test** -- scope DERIVED from the import graph: packages owning changed
   files plus every package whose transitive deps include one (`go list -deps`
   reverse cone). No hand-listed impact table. First floor commit resolved 4
   packages; the codec fix resolved 34 -- both derived, not declared.
5. **manifest / sbom / claims** -- kernel ABI regenerate-and-diff, SBOM check,
   compatibility claim verification.
6. **commit** -- scoped `git add -- <paths>`, then `git commit -F` with
   `OVERGO_COMMIT_GATE=1` set so the guard stands down for exactly that commit.

Every run lands as a `runrecord.GateRecord` in the store (authoritative;
controlled phase vocabulary) with an advisory `bin/gate_status.json` mirror.
A killed gate cannot leave a false "running" state -- the store record is
written once, atomically, at the end. Green output ends with the honesty line:
steps run/skipped, unplanned dirt, derived test scope. Bootstrap was
self-hosted -- the gate's first commit was the gate itself.

Incident-proven mechanics carried verbatim: `-message-file` only (shell-parsed
prose loses backticked text to command substitution); staged-scope canonical
checks (tree-global canonicalization wedged cross-lane in adaptive_new); the
store record replaces the stale-status-file failure mode.

### 2. Claim evidence tiers (DONE)

`cmd/compatibility -check` now REQUIRES `real_model_validation` or a validated
fixture for any model claiming `implemented`; models without oracle evidence
are `experimental`. Enforcement over data that already existed -- the current
manifest passes honestly (34 oracle-backed implemented, 135 experimental).
The rule exists so inherited breadth can never claim evidence it does not
carry.

### 3-4. Magic ledger: export contract + residency (DONE)

Two-stage, no permanent cross-repo dependency:

- **Producer** `cmd/repodb-export` (adaptive_new): emits `MAGICS.json` as the
  neutral JSONL contract (`docs/REPODB_IMPORT.md`). Wire records mirror the
  importer's typed structs field-for-field (canonicality is a byte compare);
  closure documents mirror `internal/closureledger.Document` (tier -> status
  via `validTierStatus`, values canonicalized by UseNumber round-trip, texts
  CR-scrubbed); owner files export as hashed artifacts; the `MAGICS.json`
  snapshot is every row's pinning fixture until per-row fixtures upgrade.
- **Consumer** `cmd/repodb-import` (overgo): resolves logical names to
  content-addressed IDs, canonicalizes each known document through its OWNING
  codec (`DocumentCodec.Normalize` -- the fix from `ced810c`; a generic
  map-marshal orders keys alphabetically while codec-canonical form is struct
  order, which had silently broken the typed-document path for ALL 17 media
  types), commits one atomic batch, and links every artifact to source
  evidence carrying the producer commit and the stream SHA-256.

Result: 59 closure documents resident, verified by `repodb-query`. The
per-port magic audit is now a store query (`magics touching surface X, closure
status`) instead of a JSON grep.

IMPORTANT distinction: the 59 rows are adaptive_new's, owner surfaces pointing
at adaptive_new paths. They are the AUDIT REFERENCE for ports, not overgo's own
magic state. See component 5.

## Pending components

### 5. Closure scan / native magic baseline (PENDING -- recommended next)

Overgo has ZERO native closure documents; its own constants (kernel launch
widths, cache band multipliers, graph fusion ladders, gate constants, ported
optimizer iters) are uncatalogued. A `MAGICS.json` file is the WRONG shape --
it reintroduces the side-file the store-resident ledger replaced (a second
owner, drift bait, contra skill.md). The right form is a scan tool that emits
overgo's own constants AS closure documents in the store.

This tool IS Automation Doctrine Layer 6 (the gate's magic scan): it does
double duty -- produces the baseline catalog AND becomes the enforcement that
flags uncatalogued literals landing in future commits.

Design requirements:
- Tier at emit time (candidate-magic / math-fact / decision-record); a raw
  literal scan of ~145K lines drowns in loop indices and buffer sizes.
  Full-list discipline: one ranked surface, not per-file noise.
- Distinguish overgo-native rows (local owner surface, no adaptive_new source
  lineage) from the imported audit-reference rows.
- Emit through `closureledger.Normalize` so authored rows are canonical by
  construction.
- Full catalog is ongoing triage; the tool is the deliverable.

### 6. Statistical / longitudinal layer (DEFERRED to wave 1)

Quantile-calibrated alarms, timing/memory budgets, two-window regression
confirmation, cord:embryo instrumentation. CANNOT precede runs (first run
calibrates, second enforces). Builds against the wave-1 pilot's own run
history as its calibration corpus. Distribution-free by mandate: median/MAD/
quantiles/envelopes, alarm budget as a recorded decision, no Gaussian/IID.

### 7. Exporter durable commit + remaining scopes (PARTIAL)

`cmd/repodb-export` is proven (produced the stream that imported cleanly) but
UNCOMMITTED in adaptive_new -- caught behind that repo's unrelated
`adversarial_sqa` phase over a media merge. Low-stakes bookkeeping; lands when
that phase clears naturally or on explicit direction. Do NOT review the
co-implementer's media merge just to unblock it (inverts priority, risks a
duplicate review).

Remaining export scopes, same wire pattern, ordered by port need:
- **Profiles** with per-field provenance -> needed before model ports.
- **Measurements** (phased, environment-bound runs) -> seeds the statistical
  layer's history and the external scoreboard.
- **Dataset** split/mixture/dedup facts -> needed before the training wave.
- **Findings** -> the live adversarial register moves as store documents.

### 8. PowerShell retirement (PARTIAL)

Doctrine (skill.md Code Hygiene) is no PowerShell -- scripts are bash or Go.
This is not cosmetic: the guard BLOCKS `powershell`/`pwsh` invocation, so the
three shipped `.ps1` scripts are a live contradiction -- the repo forbids
running the workflow its own README documents, and merge-week muscle memory
must not form around commands the guard rejects.

State and disposition:
- `verify.ps1` -- ~80% SUPERSEDED by `cmd/gate` (fmt/vet/build/test/sbom/
  manifest/claims). The only unique remainder is the CUDA/device lane
  (`cuda-info`, `cuda-smoke`, `internal/cuda/...` integration tests) and the
  full unscoped `go test ./...`. Disposition: give the CUDA lane a Go/bash
  home (the manifest-routed device lane of Automation Doctrine Layer 3), THEN
  delete `verify.ps1` and repoint README. Do not delete before the CUDA lane
  has a home -- that loses the device-test invocation.
- `build-kernels.ps1` -- nvcc PTX/cubin build. Port to a `cmd/build-kernels`
  Go tool (or thin bash) that shells nvcc; the ABI manifest already owns the
  provenance side, this is just the invocation.
- `fuzz-smoke.ps1` -- fuzz harness. Port to bash or a Go cmd.
- Doc/example references (README, `docs/IMPLEMENTATION_LOG.md`,
  `docs/REPODB_IMPORT.md`): repoint to the bash/Go equivalents. The
  `REPODB_IMPORT.md` example is already converted.

Sequence: it rides component 1 (gate) already having absorbed the hygiene
chain, so the remaining work is the CUDA lane home + two script ports + doc
repointing. Retiring `verify.ps1` is the last step, gated on the CUDA lane.

## The porting discipline this floor enables

Per capability (skill.md Porting Discipline):

    adaptive behavior + evidence
    -> neutral typed contract
    -> reference (CPU) implementation
    -> CUDA implementation
    -> recipe-selected consumer
    -> parity/performance evidence in the store
    -> no compatibility path

The floor makes each step enforceable: the guard protects the working tree,
the gate records parity evidence as store runs, the evidence tiers keep the
capability's claim honest, and the magic ledger lets the per-port closure
audit run as a query -- each ported row either carries its closure
(derivation, runtime-derived rule, pinning fixture) or visibly reopens as a
tracked row. A shape ported without its closure silently reopens an
expensively answered question (the `schedule.go` num_steps case).

## Acceptance bar: "floor complete"

- [x] Guard blocks the incident classes; corpus non-vacuous.
- [x] Gate records every outcome to the store; scope derived; honesty line.
- [x] Evidence tiers enforced; manifest honest.
- [x] Magic ledger resident and queryable.
- [ ] Overgo's own magics catalogued as native closure documents (component 5).
- [ ] Profiles + measurements exported (component 7) -- gates wave-1 model port.
- [ ] Statistical layer calibrated against wave-1 runs (component 6).
- [ ] PowerShell scripts retired to Go/bash; guard-vs-shipped-PS contradiction
      resolved (component 8).

First four are met -- the floor is load-bearing for the wave-1 pilot (dense
Qwen recipe parity) NOW. Components 5-7 complete it in parallel with that
pilot; none blocks starting the pilot, and the pilot's runs are what
component 6 needs to exist.

## Wave-1 pilot (first port, calibration corpus)

Dense Qwen recipe compiled into overgo's `ModelPlan`, active-recipe authority
required (already native: `e1fc61f`), token/cache/CUDA parity proven and
recorded to the store. Small enough to be safe under the bare floor, real
enough to generate the run history that calibrates component 6. Inference and
training organs at full rate follow only after it defends.
