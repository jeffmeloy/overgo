package main

import "testing"

// TestAPlanEditRidesWithTheNextLanding holds the stop gate's orphan rule to work a landing would lose
// (plan-rides/no-orphan): an edit of the plan a turn made is no orphaned work, as a plan review rides with the next
// landing, while an edit of code beside it still is.
func TestAPlanEditRidesWithTheNextLanding(t *testing.T) {
	t.Parallel()
	planned := dirtyFact{Path: "docs/plan.json", WorktreeStatus: "M", WorkIdentity: "after"}
	coded := dirtyFact{Path: "internal/x/x.go", WorktreeStatus: "M", WorkIdentity: "after"}
	if created := turnCreatedDirt(nil, []dirtyFact{planned}); len(created) != 0 {
		t.Fatalf("a turn that edited only the plan left %v orphaned; want nothing, the plan riding with the next landing", created)
	}
	if created := turnCreatedDirt(nil, []dirtyFact{planned, coded}); len(created) != 1 || created[0].Path != coded.Path {
		t.Fatalf("a turn that edited the plan and code left %v orphaned; want the code alone", created)
	}
}
