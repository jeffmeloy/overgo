package plan

import (
	"slices"
	"testing"
)

// TestMergeDeclaresItsOwnRow lets a merge land with the row that judges it:
// the row first appears in the merge, unawaited by the closeout (plan
// -prepare-merge inserts it so), and completing it leaves the first-parent
// plan exactly as it was. An awaited row, a second new row, or a change to an
// existing row all adopt rows from outside the target plan.
func TestMergeDeclaresItsOwnRow(t *testing.T) {
	t.Parallel()
	local := Plan{Items: []Item{
		{ID: "work", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./a"}}},
		{ID: "closeout", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./b", DependsOn: []string{"work/do"}}}},
	}}
	declare := func(awaited bool, ids ...string) Plan {
		preAdvance := local
		preAdvance.Items = slices.Clone(local.Items)
		closeout := preAdvance.Items[1]
		closeout.Steps = slices.Clone(closeout.Steps)
		for _, id := range ids {
			preAdvance.Items = slices.Insert(preAdvance.Items, 0, Item{ID: id, Status: StatusOpen, Budget: Budget{Reason: "the lane's code"},
				Steps: []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./lane"}}})
			if awaited {
				closeout.Steps[0].DependsOn = append(slices.Clone(closeout.Steps[0].DependsOn), id+"/do")
			}
		}
		preAdvance.Items[len(preAdvance.Items)-1] = closeout
		return preAdvance
	}
	owned := func(preAdvance Plan) error {
		t.Helper()
		child, err := advancePlan(preAdvance, "merge-lane", "do", false)
		if err != nil {
			t.Fatal(err)
		}
		return verifyFirstParentTargetPlanOwnership(local, preAdvance, child)
	}
	if err := owned(declare(false, "merge-lane")); err != nil {
		t.Fatalf("a merge declaring its own row was refused: %v", err)
	}
	if owned(declare(true, "merge-lane")) == nil {
		t.Fatal("a closeout dependency on the merge row survived its completion unnoticed")
	}
	if owned(declare(false, "merge-lane", "lane-feature")) == nil {
		t.Fatal("a merge adopted a second row from outside the target plan")
	}
	changed := declare(false, "merge-lane")
	changed.Items[1].Steps = []Step{{ID: "do", Status: StatusOpen, Verify: "go test ./changed"}}
	if owned(changed) == nil {
		t.Fatal("a merge changed an existing target row")
	}
}
