package plan

import (
	"slices"
	"testing"

	"overgo/internal/worklease"
)

// TestDispatchVisitsPrerequisitesFirst orders ready rows so a blocked row's
// prerequisites come before its later siblings, and dispatches the first.
func TestDispatchVisitsPrerequisitesFirst(t *testing.T) {
	t.Parallel()
	document := Plan{Items: []Item{
		{ID: "publication", Status: StatusOpen, Steps: []Step{
			{ID: "memoize", Status: StatusOpen, Verify: "go test ./x", DependsOn: []string{"publication/measure", "selection/source-binding"}},
			{ID: "measure", Status: StatusOpen, Verify: "go test ./x"},
			{ID: "tooling", Status: StatusOpen, Verify: "go test ./x"},
		}},
		{ID: "selection", Status: StatusOpen, Steps: []Step{{ID: "source-binding", Status: StatusOpen, Verify: "go test ./x"}}},
	}}
	ordered, err := dispatchPriority(document, worklease.UnassignedRole)
	if err != nil {
		t.Fatal(err)
	}
	want := []Ref{{Item: "publication", Step: "measure"}, {Item: "selection", Step: "source-binding"}, {Item: "publication", Step: "tooling"}}
	if !slices.Equal(ordered, want) {
		t.Fatalf("priority = %v, want %v", ordered, want)
	}
	dispatch, err := selectDispatch(document, worklease.UnassignedRole, "")
	if err != nil || dispatch.Item != "publication" || dispatch.Step != "measure" {
		t.Fatalf("dispatch = %+v, %v", dispatch, err)
	}
	if requested, err := selectDispatch(document, worklease.UnassignedRole, "publication/memoize"); err != nil || requested.Waiting == "" {
		t.Fatalf("a blocked requested row dispatched: %+v, %v", requested, err)
	}
}
