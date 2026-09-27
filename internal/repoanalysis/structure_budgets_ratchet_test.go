// Package repoanalysis_test holds the live-tree checks of the repository's
// architecture rules.
package repoanalysis_test

import (
	"path/filepath"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/repoanalysis"
)

// TestStructureBudgetRatchets holds the live tree to its reviewed size and
// coupling budgets: every production file within the shared line ceiling or
// its named exception, every package within the internal-import ceiling,
// every command package within its file spread, and the agent-harness
// ownership audit clean. A budget widens only by a reviewed manifest edit.
func TestStructureBudgetRatchets(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	budgets, err := repoanalysis.LoadStructureBudgets(filepath.Join(root, filepath.FromSlash(repoanalysis.StructureBudgetsFile)))
	if err != nil {
		t.Fatal(err)
	}
	if budgets.MaxStagedPathsPerCommit <= 0 {
		t.Fatal("staged-path budget must be declared")
	}
	snapshot, err := gosyntax.DiscoverGo(root, "internal", "cmd")
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
}
