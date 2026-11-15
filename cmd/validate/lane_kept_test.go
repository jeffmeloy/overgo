package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestARunWhosePlanNamesNoPlaceKeepsNothing holds the runner to keeping lanes' ends only where its plan says
// (lane-record/do): a plan that names no place keeps nothing, and one told where keeps its lane there. The runner's
// own test, whose seven fixture lanes went through it, had written them into the trunk's record, which the status then
// printed.
func TestARunWhosePlanNamesNoPlaceKeepsNothing(t *testing.T) {
	t.Parallel()
	root, head := fixtureRepository(t)
	lane := Plan{Suites: []SelectedSuite{{Lane: "passes", Arguments: []string{"env", "GOFLAGS"}}}}
	var out bytes.Buffer
	testcheck.NoError(t, executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lane, scopeLanes))
	if _, err := os.Stat(filepath.Join(root, ".overgo-runtime", "lane_runs.json")); err == nil {
		t.Fatalf("a run whose plan names no place kept its lane: %v", plan.ReadLaneRuns(root))
	}
	lane.LanesKept = t.TempDir()
	testcheck.NoError(t, executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lane, scopeLanes))
	if record := plan.ReadLaneRuns(lane.LanesKept)["passes"]; !record.Last.Passed || record.Last.Commit != head {
		t.Fatalf("a run told where to keep its lanes keeps %+v; want the passing lane at %s", record, head)
	}
}
