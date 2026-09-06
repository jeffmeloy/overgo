# Integrating the professional GUI lane into master

Rehearsed 2026-09-06 with `git merge-tree --write-tree master
professional_overgo_gui` (no worktree, nothing written): the merge is
clean. This record is the sequence the owner runs from the master
worktree once it is clean; the lane itself never writes there.

## What the rehearsal found

| Fact | Value |
|------|-------|
| Merge base | `27f3944b47af` (master's head; master has not moved since the lane's last catch-up merge) |
| Lane head at the rehearsal | `bba28535` plus this record's commit |
| Merged tree | `a7574809bf6b6968e421c807c0115472bcad3580`, no conflicting path |
| Paths the merge changes | 123 (5737 insertions, 798 deletions) |
| New packages | `internal/apimanifest` (admission), `internal/providerintake`, `internal/remoteprovider`, `internal/remoterelay` with `relaytest`, `cmd/remote-provider`; extensions in `cmd/evaluate`, `cmd/server`, `internal/modelrecipe`, `internal/runrecord`, `internal/server` |

The master worktree at the time of the rehearsal held 22 staged paths of
another lane's gate transaction (`internal/gate/batch_cost*.go` and its
generated documents). The integration waits for that transaction to land
or be reverted; it is never merged over.

## What the integration must not take from the lane

- `docs/plan.json`: the lane's plan never merges (lane merge protocol).
  Master keeps its own file; the lane's rows landed through the gate and
  live in Git's structured trailers. The lane plan names its lane and
  marks master's retained rows master's; none of that belongs on master.
- `docs/plan_stop.json`: untracked on both sides; nothing to take.

## What the integration regenerates on master

Generated documents differ on the lane because they were regenerated
over the lane tree: `compatibility.json`, `docs/COMPATIBILITY.md`,
`docs/api_manifest.json`, `docs/API_MANIFEST.md`,
`docs/modern_go_baseline.json`, `docs/modern_go_census.json`,
`docs/harness_surface_baseline.json`, `docs/staged_surface.json`,
`docs/structure_budgets.json`. Each is regenerated over the merged tree
rather than taken from either side (see the sequence). The structure
budgets file carries the lane's reviewed exception text for the server's
42 internal imports; the merged tree stays at 42.

## What master's owner accepts

The contract extensions are tabled in `ANALYSIS.md` ("Hosted-evaluation
contract extensions"), each with its owning package, its pin and the
acceptance it asks. Two require a change on master's side:

- `modality-verification` verifies request the E4B acceptance through
  the data root today. After the merge the accepted-E4B test requests it
  explicitly: the verify sets
  `OVERGO_E4B_VALIDATION=docs/verification/e4b-validation.json` beside
  `OVERGO_DATA_ROOT`, or the test skips and the verify reads as vacuous.
- The credential-less admission lives in `internal/apimanifest` and the
  swap proxy applies it before any mutation of its own; the server's
  behaviour is unchanged and its cross-origin test still passes.

## The sequence from the master worktree

1. Land or revert the staged transaction so `git status` is clean.
2. `git merge --no-ff --no-commit professional_overgo_gui`
3. `git checkout master -- docs/plan.json` and
   `git update-index --clear-resolve-undo`, so master's plan stays.
4. Regenerate over the merged tree: `go run ./cmd/modern-census
   -lower-baseline` then `-publish-census`; `go run ./cmd/compatibility
   -refresh-identities` then `-check`; `go run ./cmd/api-manifest
   -update` last; republish `docs/harness_surface_baseline.json` from the
   merged tree if preflight reports the ceiling; `go run ./cmd/preflight`.
5. Add master's merge row: `go run ./cmd/plan -add -title "Merge
   professional_overgo_gui at <12hex>" -vcmd "go build ./..." -before
   <first open id> merge-<12hex>`, and whitelist `merge-<12hex>/do` in
   `internal/plan/campaign_structure_test.go`.
6. Import the lane store's closure documents before gating:
   `go run ./cmd/closure-scan -import-store
   C:/Users/jeffm/professional_overgo_gui/overgodb-store`, then
   `-import-store overgodb-store -review-callsites`.
7. Gate the merge: `go run ./cmd/gate -merge -plan merge-<12hex>/do
   -message-file <msg> -plan-projection first-parent-target
   -merge-source-store C:/Users/jeffm/professional_overgo_gui/overgodb-store`.
8. Rebuild `bin/gate.exe`, `bin/plan.exe` and `bin/loophook.exe` on
   master (the plan package changed), and update the
   `modality-verification` verifies as above.
