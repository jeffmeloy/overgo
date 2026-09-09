package plan

import "testing"

// TestLaneOwnedDispatch pins the lane boundary: a plan naming its lane
// dispatches the lane's rows, then unowned rows, and never another
// lane's; an explicit role still takes precedence over the plan's lane.
func TestLaneOwnedDispatch(t *testing.T) {
	document := Plan{Lane: "gui", Items: []Item{
		{ID: "master-only", Owner: "master", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen}}},
		{ID: "shared", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen}}},
		{ID: "gui-own", Owner: "gui", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen}}},
	}}
	authority := testCompletionAuthority(t, document)
	if item, _, ok := Current(document, UnassignedRole, authority); !ok || item.ID != "gui-own" {
		t.Fatalf("lane dispatch = %s, ok=%v", item.ID, ok)
	}
	if item, _, ok := Current(document, "master", authority); !ok || item.ID != "master-only" {
		t.Fatalf("explicit role dispatch = %s, ok=%v", item.ID, ok)
	}
	retained := Plan{Lane: "gui", Items: []Item{
		{ID: "master-only", Owner: "master", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen}}},
	}}
	if item, _, ok := Current(retained, UnassignedRole, testCompletionAuthority(t, retained)); ok {
		t.Fatalf("another lane's retained row dispatched: %s", item.ID)
	}
	if err := Validate(Plan{Lane: "bad\nlane", Items: retained.Items}); err == nil {
		t.Fatal("an invalid lane name validated")
	}
}
