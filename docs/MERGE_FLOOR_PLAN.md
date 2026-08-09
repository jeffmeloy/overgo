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
| 8 | PowerShell retirement (Go/bash automation) | DONE | `cmd/device-lane`, `cmd/build-kernels`, `scripts/fuzz-smoke.sh` |
| 9 | Data organization (models/datasets/checkpoints) | PARTIAL | data roots, `internal/artifact` locations, guard |

## Built components

### 0. Destructive-command guard (DONE)

PreToolUse safety engine, ported from adaptive_new at its current hardening
(quoted-target and executable-substitution rules). Adaptations: `OVERGO_COMMIT_GATE`
exemption env; `repodb-store` added to protected paths; adaptive-specific
rules (record-cycle prose) dropped. One rule the lineage guard lacked:
`find -delete` / `-exec <delete-verb>` over a protected path (the delete
target is the substituted `{}`, so no verb-scoped rule sees it).

Corpus: 44 cases (grown with each new rule; the count in this doc is
informational — the test owns it), DENY and ALLOW per rule, non-vacuity
enforced by the test (a one-directional rule fails the suite). Fail-open on unparseable input; the
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

### 8. PowerShell retirement (DONE)

Doctrine (skill.md Code Hygiene) is no PowerShell -- scripts are bash or Go.
The three `.ps1` scripts were a live contradiction (the guard blocks
`powershell`/`pwsh` while README documented running them). Resolved:

- `cmd/device-lane` (Go) owns the CUDA/device lane: `cuda-info` as the
  availability probe (failure = UNAVAILABLE, loudly, never green),
  `cuda-smoke`, then the env-gated CUDA integration tests. Faithful port of
  verify.ps1's unique remainder; the manifest-scoped form remains the Layer-3
  target and the lane says so in its honesty line.
- `cmd/build-kernels` (Go) owns kernel builds: nvcc PTX for the pinned device
  class (`compute_89`, the compatibility-baseline device -- a baseline
  decision, not a flag), runtime SHA-256 pin refresh in the validation
  source, then kernel-manifest update + verify so provenance and bytes agree.
- `scripts/fuzz-smoke.sh` (bash) owns the never-panic fuzz sweep, duration
  per target as the argument.
- `verify.ps1`, `build-kernels.ps1`, `fuzz-smoke.ps1` deleted; README
  converted to bash/Go throughout (gate for commits, device-lane and
  fuzz-smoke as standalone lanes). `docs/IMPLEMENTATION_LOG.md` retains
  historical `.ps1` mentions as frozen chronology, which is allowed -- live
  guidance no longer references deleted surfaces.

### 9. Data organization (PARTIAL)

Data is where this project's irreversible mistakes live (both historical
catastrophes were data-path errors). The floor's stance is mechanical, not
disciplinary.

**Core principle -- the store references bulk by location, never copies bytes.**
`repodbimport.importFile` hashes a file for content identity and records a
`LocationFile` at its resolved absolute path; the bytes never enter the store.
So models, datasets, and checkpoints stay physically put while the store holds
their content ID + location. "Data never moves, code comes to the data" is a
property of the design. The store is provenance and orchestration, not a blob
store.

**Three data roots, distinct lifecycles -- kept separate on purpose:**

| Root | Contents | Provenance | Reproducible |
|------|----------|-----------|--------------|
| `models/` | vendor/downloaded inputs (~700 GB) | download source | No -- irreplaceable |
| `datasets/` | training data (raw + canonical content store) | manifest/source | Partial |
| `checkpoints/` | trained outputs | base + data + recipe + run | Yes -- regenerable |

Downloaded and trained models have OPPOSITE reproducibility, so they must not
share a directory -- mixing them loses track of what is irreplaceable versus
what can be pruned and regenerated. Both repos already separate them
(`KindModel` vs `KindCheckpoint` in the store; adaptive_new's `checkpoints/`
distinct from `models/`).

**Checkpoints are lineage-bearing artifacts.** A trained model is a
`KindCheckpoint` artifact whose `depends-on` edges name its base model,
dataset (with the persisted split), and training recipe, plus a link to the
producing run record. "What produced this, from what, at which commit, with
what eval" is a store query -- the capability-evidence contract made
structural. Retention policy falls out: full lineage + passing eval =>
reproducible => prunable; missing lineage => irreplaceable => protected.
Promotion to serving is an alias flip to the content ID, not a byte copy (the
E4B name-binding failure cannot recur because recipes bind IDs, not names).

**Guard coverage (DONE this component):** the destructive-command guard now
protects `checkpoints/` alongside `models/`, `datasets/`, `docs/repodb/`, and
`repodb-store`; `.gitignore` covers all three data roots. Trained checkpoints
represent real GPU-hours and are not all reproducible until training is
deterministic, so leaving them unguarded was the exact hole class behind the
prior data-loss incidents.

**Remaining (PENDING):**
- Single data-root config contract: one owner (`OVERGO_DATA_ROOT` env or the
  gitignored `local-models.yaml`) naming where the three roots live, read by
  the importer AND the runtime. Today `generate` takes ad-hoc path flags and
  `server` a model-id -- no unified root, which is how smoke-matrix staleness
  (E4B) started. Establish before the wave-1 model port.
- Discovery as a store query: a servable model = artifact present at its
  location with an active oracle-backed recipe -- the same query that drives
  the smoke matrix and evidence tier. Unifies discovery with the tier.
- Junction discipline (unchanged doctrine): data stays at its canonical home;
  overgo references by absolute path via the config, never junction-copies
  into a worktree. `dir /AL /S` before any recursive delete.
- GGUF first-class: a model/checkpoint that is one `.gguf` is hashed as the
  artifact; no sidecars written next to it.

**Merge rule (spans components 3-4 and 7):** ingest FACTS, reference BULK.
adaptive_new's recipes, measurements, magics, and findings import into the
store; its weights, dataset payloads, the measurements-log content, and
`docs/repodb/content` are referenced by location, never copied. The importer
already does the right thing (hash + locate); the discipline is never pointing
an ingest at the byte payloads.

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
- [ ] Profiles + measurements exported (component 7) -- gates the model
      capability waves (TimesFM onward), NOT the wave-1 pilot, which runs on
      overgo's own safetensors/catalog path.
- [ ] Statistical layer calibrated against wave-1 runs (component 6).
- [x] PowerShell scripts retired to Go/bash; guard-vs-shipped-PS contradiction
      resolved (component 8).
- [x] Guard + gitignore cover all three data roots incl. checkpoints (component 9).
- [ ] Single data-root config contract; discovery as a store query (component 9).

First four are met -- the floor is load-bearing for the wave-1 pilot (dense
Qwen recipe parity) NOW. Components 5-7 complete it in parallel with that
pilot; none blocks starting the pilot, and the pilot's runs are what
component 6 needs to exist.

## Model ladder (owner directive 2026-08-08)

After the floor: merge adaptive_new's models ONE AT A TIME, smallest to
largest, verifying the inference AND training capabilities of each before the
next -- the ladder is the port order, and each rung is its own gated,
evidence-recorded slice.

Per-rung acceptance:
- Inference: the model serves through overgo recipe authority; parity evidence
  recorded to the store (vs adaptive_new goldens where they exist, vs the
  pinned upstream oracle where they apply); its compatibility claim carries
  the honest evidence tier.
- Training: whatever adaptive_new could do with this model (fine-tune, LoRA
  adapt, probe-train) is demonstrated through overgo's training path, with
  the magic-ledger closure audit run on every surface the rung touches.
- Performance (owner directive 2026-08-08): overgo's wall time AND peak
  memory for the rung's operations must be AT OR BELOW adaptive_new's
  recorded numbers for the same artifact and operation -- the match-or-beat
  doctrine applied per rung, measured through the store's phased
  environment-bound runs against adaptive_new's exported measurement
  baselines (component 7's measurements scope is what supplies them). A rung
  that regresses time or memory does not pass; the regression becomes the
  rung's blocking finding. The comparison itself follows the no-magic and
  distribution disciplines (owner directive 2026-08-08): at n<=3 replicates
  the only bankable verdict is envelope separation (non-overlapping min-max);
  beyond that, median/MAD -- never mean/sd without a recorded defense; any
  tolerance or noise floor in the comparison is derived from the measurements
  themselves or is a recorded decision with its trigger, never an asserted
  constant; and the full accounting (both sides' runs, the derived floor, the
  verdict) lands in the store, not in prose.
- Weights never move: each rung's artifacts resolve via the data-root
  contract (local-models.yaml pointing at adaptive_new's model home).

Ladder, smallest -> largest (adaptive_new models/ + checkpoints/; sizes
approximate; re-derive exact order from disk at each rung):

    timesfm-200m -> Fractale-350M -> Carbon-500M -> Qwen2.5-0.5B -> tabfm
    -> needle -> pocket-tts -> Un-0 -> SimpleDiffusion -> MiniCPM5-1B
    -> Wan2.1-1.3B -> Qwen3.5-4B -> gemma-4-E4B -> RxBrain
    -> SenseNova-U1-8B -> Krea-2-Turbo -> gemma-4-12B-fp8 -> Qwen3.5-9B

The first rung doubles as the calibration corpus for component 6 (its runs
are the first store history the statistical layer calibrates against). The
dense-Qwen recipe parity check remains the wave-1 *mechanism* pilot -- it
proves the recipe/plan/parity machinery on overgo's best-supported family
before the ladder starts consuming it rung by rung.
