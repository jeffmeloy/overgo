package main

import (
	"bytes"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestARunWhosePlanNamesNoPlaceKeepsNothing holds the runner to keeping results only where its plan says
// (lane-record/do, lanes-record): a plan that names no place keeps nothing, and one told where keeps its lane run whole
// there, a pass at the commit it ran at. The runner's own test, whose seven fixture lanes went through it, had written
// them into the trunk's record, which the status then printed.
func TestARunWhosePlanNamesNoPlaceKeepsNothing(t *testing.T) {
	t.Parallel()
	root, head := fixtureRepository(t)
	lane := Plan{Suites: []SelectedSuite{{Lane: "passes", Arguments: []string{"env", "GOFLAGS"}}}}
	var out bytes.Buffer
	testcheck.NoError(t, executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lane, scopeLanes))
	if kept := plan.ReadTestResults(root); len(kept) != 0 {
		t.Fatalf("a run whose plan names no place kept %v", kept)
	}
	lane.LanesKept = t.TempDir()
	testcheck.NoError(t, executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lane, scopeLanes))
	if result := plan.ReadTestResults(lane.LanesKept)[testKey("", "passes")]; result.Failed || result.Commit != head {
		t.Fatalf("a run told where to keep its lanes keeps %+v; want the passing lane at %s", result, head)
	}
}
