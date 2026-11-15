package plan

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestDispatchVisitsPrerequisitesFirst orders ready rows so a blocked row's
// prerequisites come before its later siblings, and dispatches the first.
func TestDispatchVisitsPrerequisitesFirst(t *testing.T) {
	t.Parallel()
	document := Plan{Items: []Item{
		{ID: "publication", Steps: []Step{
			{ID: "memoize", Verify: "go test ./x", DependsOn: []string{"publication/measure", "selection/source-binding"}},
			{ID: "measure", Verify: "go test ./x"},
			{ID: "tooling", Verify: "go test ./x"},
		}},
		{ID: "selection", Steps: []Step{{ID: "source-binding", Verify: "go test ./x"}}},
	}}
	ordered := testcheck.Must(dispatchPriority(document))(t)
	want := []Ref{{Item: "publication", Step: "measure"}, {Item: "selection", Step: "source-binding"}, {Item: "publication", Step: "tooling"}}
	if !slices.Equal(ordered, want) {
		t.Fatalf("priority = %v, want %v", ordered, want)
	}
	dispatch, err := selectDispatch(document, "")
	if err != nil || dispatch.Item != "publication" || dispatch.Step != "measure" {
		t.Fatalf("dispatch = %+v, %v", dispatch, err)
	}
	if requested, err := selectDispatch(document, "publication/memoize"); err != nil || requested.Waiting == "" {
		t.Fatalf("a blocked requested row dispatched: %+v, %v", requested, err)
	}
}
