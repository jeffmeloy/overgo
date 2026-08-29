package runrecord

import (
	"testing"

	"overgo/internal/artifact/repositorytest"
)

// TestSemanticTransitionUsesOneCommit proves a human-decision
// admission -- decision document, lineage, and resolution alias
// together -- lands as exactly one atomic batch.
func TestSemanticTransitionUsesOneCommit(t *testing.T) {
	store, request, decision := humanDecisionFixture(t)
	counting := &repositorytest.CountingRepository{Repository: store}
	if err := PublishHumanDecision(t.Context(), counting, request, decision); err != nil {
		t.Fatal(err)
	}
	if counting.Commits != 1 {
		t.Fatalf("decision publication used %d commits, want 1", counting.Commits)
	}
}
