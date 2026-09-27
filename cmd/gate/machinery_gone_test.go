package main

import (
	"os"
	"path/filepath"
	"testing"
)

// requireGone fails for each repository path that still exists.
func requireGone(t *testing.T, paths ...string) {
	t.Helper()
	for _, path := range paths {
		if _, err := os.Stat(filepath.Join("..", "..", filepath.FromSlash(path))); !os.IsNotExist(err) {
			t.Errorf("%s still exists (%v)", path, err)
		}
	}
}

// TestOldGateGone holds the landing path to the small gate: the old gate's
// package and its command modes are deleted.
func TestOldGateGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/gate")
}

// TestSelectionAnalyzersGone holds test selection to go test's own cache:
// the code manifest, its impact analysis and the check-planning package are
// deleted.
func TestSelectionAnalyzersGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/codemanifest", "internal/automationcheck", "cmd/code-manifest")
}

// TestCompletionAuthorityGone holds the plan to plain data: the completion
// authority, its transitions and its proof horizon are deleted, and git is
// the record of what landed.
func TestCompletionAuthorityGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/plan/completion.go", "internal/plan/projected_merge_authority.go",
		"internal/plan/completion_transitions.go", "internal/plan/proof_horizon.go", "docs/plan_proof_horizon.json")
}

// TestClosureLedgerGone holds the inline-number rule to the small gate's
// diff check: the closure ledger, its scanner, the code profile and clone
// census, and their baselines are deleted.
func TestClosureLedgerGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/closurescan", "internal/closureledger", "cmd/closure-scan",
		"internal/codeprofile", "cmd/clone-census", "docs/clone_baseline.json",
		"docs/staged_surface.json", "docs/harness_surface_baseline.json")
}

// TestCensusesGone holds the tree rules to what a landing runs: the modern-Go
// census gives way to go fix and the gate's legacy-form rule, and the
// ownership, family-branch, store-access, structure-budget and harness
// censuses are deleted with their commands and documents.
func TestCensusesGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/repoanalysis/modern_go_census.go", "internal/repoanalysis/modern_go_catalog.json",
		"internal/repoanalysis/ownership_census.go", "internal/repoanalysis/family_branch_census.go",
		"internal/repoanalysis/store_access_census.go", "internal/repoanalysis/structure_budgets.go",
		"internal/repoanalysis/harness.go", "cmd/modern-census", "cmd/rsi-census",
		"docs/modern_go_baseline.json", "docs/rsi_ownership_census.json", "docs/structure_budgets.json")
}

// TestGateRecordsGone holds the store to result records: the old gate's
// lane obligations, selection causes and suite costs, the telemetry class
// that retained them, and the plan history that read them are deleted.
func TestGateRecordsGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/runrecord/lane_obligation.go", "internal/runrecord/selection_cause.go",
		"internal/runrecord/suite_cost.go", "internal/storepolicy/telemetry.go", "cmd/plan/history_phases.go")
}

// TestPlanOrchestrationGone holds the plan tool to plain data: the leases,
// exploration budgets, experiments, locality schedules, efficiency judging
// and automation context it once carried are deleted.
func TestPlanOrchestrationGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "cmd/plan/orchestration.go", "internal/plan/exploration.go", "internal/plan/locality_scheduler.go",
		"internal/plan/scheduler.go", "internal/plan/legacy_lease.go", "internal/plan/context.go",
		"internal/runrecord/review.go", "internal/runrecord/attempt_history.go")
}

// TestPrepareMergeGone holds merges to git: a merge is git merge --no-ff
// --no-commit and a landing, so the prepare-merge flow, its source-store
// resolution, the lane store mirror and the semantic plan merge are deleted.
func TestPrepareMergeGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "cmd/plan/sync.go", "cmd/plan/source_store.go", "internal/lanestore",
		"internal/plan/sync.go", "internal/plan/merge_projection.go")
}

// TestDispatchClaimsGone holds dispatch to the plan: a row is the first ready
// one in prerequisite order, so the claim release command and the readiness
// snapshot are deleted and prompting writes nothing.
func TestDispatchClaimsGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/dispatchreadiness", "cmd/plan/dispatch_claim.go", "cmd/plan/release_by_item_test.go",
		"internal/plan/campaign_test.go")
}

// TestOldGateMachineryGone closes the sweep the tests above make: the last
// hand-run ratchet, the coverage-lanes report and its reviewed floors, goes
// with the rest.
func TestOldGateMachineryGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "cmd/coverage-lanes", "docs/coverage_lanes_baseline.json")
}
