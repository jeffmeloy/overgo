package plan

import (
	"testing"
)

func frontierTestPlan() Plan {
	return Plan{Campaign: "frontier", Items: []Item{
		{ID: "alpha", Title: "alpha", Status: StatusOpen, Steps: []Step{
			{ID: "one", Title: "one", Status: StatusOpen, Verify: "go test ./internal/plan"},
			{ID: "two", Title: "two", Status: StatusOpen, Verify: "go test ./internal/plan",
				DependsOn: []string{"alpha/one"}},
		}},
		{ID: "beta", Title: "beta", Status: StatusOpen, Steps: []Step{
			{ID: "solo", Title: "solo", Status: StatusOpen, Verify: "go test ./internal/plan"},
			{ID: "after", Title: "after", Status: StatusOpen, Verify: "go test ./internal/plan",
				DependsOn: []string{"gamma/pruned"}},
		}},
	}}
}

// TestReadyFrontierMembership pins the complete ordered ready frontier: a
// step waits on a prerequisite the plan still holds unfinished, a
// prerequisite no longer in the plan is settled, and Current stays the first
// member of the frontier. Lease-isolation is the reporting concern of
// frontierLeaseDiagnostics, not a frontier refusal.
func TestReadyFrontierMembership(t *testing.T) {
	t.Parallel()
	document := frontierTestPlan()
	frontier, err := ReadyFrontier(document)
	if err != nil {
		t.Fatal(err)
	}
	if len(frontier) != 3 || frontier[0].String() != "alpha/one" || frontier[1].String() != "beta/solo" || frontier[2].String() != "beta/after" {
		t.Fatalf("ready frontier = %v", frontier)
	}
	item, step, ok := Current(document, "")
	if !ok || item.ID+"/"+step.ID != frontier[0].String() {
		t.Fatalf("Current %s/%s is not the first frontier row", item.ID, step.ID)
	}
}
