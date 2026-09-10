package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestPrepareMergeLaneProjection pins the lane case of prepare-merge: a
// first-parent-target merge into a plan that names its lane is the lane
// case and nothing else is; the lane's merge row lands at the top owned by
// the lane and proven by the build, while a target's merge row stays
// unowned and proven by the compatibility check; and the finalize command
// still names the live source store.
func TestPrepareMergeLaneProjection(t *testing.T) {
	item := func(id string) plan.Item {
		return plan.Item{ID: id, Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./..."}}}
	}
	laned := plan.Plan{Campaign: "campaign", Doctrine: "doctrine", Lane: "hatchet", Items: []plan.Item{item("lane-row")}}
	target := plan.Plan{Campaign: "campaign", Doctrine: "doctrine", Items: []plan.Item{item("target-row")}}
	if !laneMerge(plan.MergeProjectionFirstParentTarget, laned) || laneMerge(plan.MergeProjectionSemanticUnion, laned) || laneMerge(plan.MergeProjectionFirstParentTarget, target) {
		t.Fatal("the lane case is a first-parent-target merge into a laned plan, and nothing else")
	}
	merged, err := insertMergeRow(laned, "merge-0123456789ab", "Merge master at 0123456789ab", laned.Lane, true)
	if err != nil {
		t.Fatal(err)
	}
	row := merged.Items[0]
	if row.ID != "merge-0123456789ab" || row.Owner != "hatchet" || len(row.Steps) != 1 || row.Steps[0].Verify != "go build ./..." || merged.Items[1].ID != "lane-row" {
		t.Fatalf("lane merge row = %+v", row)
	}
	authority := mustTestCompletionAuthority(t, merged)
	if dispatch := plan.DispatchOf(merged, plan.UnassignedRole, authority); dispatch.Item != "merge-0123456789ab" || dispatch.Step != "do" {
		t.Fatalf("the lane dispatches %+v, want its merge row", dispatch)
	}
	merged, err = insertMergeRow(target, "merge-0123456789ab", "Merge branch at 0123456789ab", "", false)
	if err != nil {
		t.Fatal(err)
	}
	if row := merged.Items[0]; row.Owner != "" || row.Steps[0].Verify != "go run ./cmd/compatibility -check" {
		t.Fatalf("target merge row = %+v", row)
	}
	arguments, err := mergeFinalizeProjectionArguments(plan.MergeProjectionFirstParentTarget, `C:\master\overgodb-store`)
	if err != nil || !strings.Contains(arguments, "-merge-source-store") {
		t.Fatalf("finalize arguments = %q, %v", arguments, err)
	}
}
