// Package repoanalysis_test holds the ratchets that need both the analysis
// owner and its closure-census consumer in one test binary.
package repoanalysis_test

import (
	"path/filepath"
	"testing"

	"overgo/internal/closurescan"
	"overgo/internal/jsonfile"
	"overgo/internal/repoanalysis"
)

// TestStructureBudgetRatchets holds the live tree to its reviewed size and
// coupling budgets: every production file within the shared line ceiling or
// its named exception, every package within the internal-import ceiling,
// every command package within its file spread, the agent-harness ownership
// audit clean, and the harness surface free of regressions against its
// stored baseline. A budget widens only by a reviewed manifest edit.
func TestStructureBudgetRatchets(t *testing.T) {
	root := filepath.Join("..", "..")
	budgets, err := repoanalysis.LoadStructureBudgets(filepath.Join(root, filepath.FromSlash(repoanalysis.StructureBudgetsFile)))
	if err != nil {
		t.Fatal(err)
	}
	if budgets.MaxStagedPathsPerCommit <= 0 {
		t.Fatal("staged-path budget must be declared")
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	for _, finding := range repoanalysis.MeasureStructureBudgets(snapshot, budgets) {
		t.Errorf("budget exceedance: %s %s = %d over limit %d",
			finding.Budget, finding.Subject, finding.Value, finding.Limit)
	}
	consolidation, err := repoanalysis.AuditAgentHarnessConsolidation(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, finding := range consolidation.Findings {
		t.Errorf("harness consolidation finding: %s %s expected %s got %s at %s",
			finding.Kind, finding.Symbol, finding.Expected, finding.Actual, finding.File)
	}
	var baseline closurescan.HarnessSurface
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs", "harness_surface_baseline.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	surface, err := closurescan.BuildAgentHarnessSurface(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, regression := range closurescan.AgentHarnessSurfaceRegressions(baseline, surface) {
		t.Errorf("harness surface regression: %s %d -> %d %s",
			regression.Metric, regression.Base, regression.Value, regression.Detail)
	}
}
