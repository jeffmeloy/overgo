package plan

import (
	"strings"
	"testing"
)

func frontierTestPlan() Plan {
	return Plan{Campaign: "frontier", Doctrine: "frontier doctrine", Items: []Item{
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

// TestMergeRetiresPrunedRows holds the plan merge to plain data: a step or a
// whole item one side dropped and the other left unchanged is retired, and a
// drop beside a concurrent edit of the same row is refused.
func TestMergeRetiresPrunedRows(t *testing.T) {
	t.Parallel()
	base, local := frontierTestPlan(), frontierTestPlan()
	prunedStep := frontierTestPlan()
	prunedStep.Items[0].Steps = prunedStep.Items[0].Steps[1:]
	merged, err := MergeDocuments(base, local, prunedStep)
	if err != nil || len(merged.Items[0].Steps) != 1 || merged.Items[0].Steps[0].ID != "two" {
		t.Fatalf("step retirement = %+v, %v", merged.Items, err)
	}
	prunedItem := frontierTestPlan()
	prunedItem.Items = prunedItem.Items[1:]
	merged, err = MergeDocuments(base, local, prunedItem)
	if err != nil || len(merged.Items) != 1 || merged.Items[0].ID != "beta" {
		t.Fatalf("item retirement = %+v, %v", merged.Items, err)
	}
	edited := frontierTestPlan()
	edited.Items[0].Steps[0].Rationale = "concurrent local edit"
	if _, err := MergeDocuments(base, edited, prunedStep); err == nil || !strings.Contains(err.Error(), "beside concurrent edits") {
		t.Fatalf("a drop beside a concurrent edit = %v", err)
	}
}

// TestMergePreservesDependencyOrder pins that an accepted pruning keeps the
// dependency graph intact: retained steps keep their references to the pruned
// row, document order survives the merge, and the dependent leads the merged
// frontier once its producer has left the plan.
func TestMergePreservesDependencyOrder(t *testing.T) {
	t.Parallel()
	base := frontierTestPlan()
	local := frontierTestPlan()
	upstream := frontierTestPlan()
	upstream.Items[0].Steps = upstream.Items[0].Steps[1:]

	merged, err := MergeDocuments(base, local, upstream)
	if err != nil {
		t.Fatal(err)
	}
	if merged.Items[0].ID != "alpha" || merged.Items[1].ID != "beta" {
		t.Fatalf("merge reordered items: %+v", merged.Items)
	}
	dependent := merged.Items[0].Steps[0]
	if dependent.ID != "two" || len(dependent.DependsOn) != 1 || dependent.DependsOn[0] != "alpha/one" {
		t.Fatalf("merge lost the dependency on the pruned row: %+v", dependent)
	}
	frontier, err := ReadyFrontier(merged)
	if err != nil {
		t.Fatal(err)
	}
	if len(frontier) == 0 || frontier[0].String() != "alpha/two" {
		t.Fatalf("proven producer did not unblock the dependent in order: %v", frontier)
	}
}
