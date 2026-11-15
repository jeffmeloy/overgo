package plan

import (
	"testing"

	"overgo/internal/worklease"
)

// TestLaneOwnedDispatch pins the lane boundary: a plan naming its lane is that lane's alone, so it gives its rows to
// the lane's role and to no role, and none to another lane's role, which a landing of that role is refused by too.
func TestLaneOwnedDispatch(t *testing.T) {
	t.Parallel()
	document := Plan{Lane: "gui", Items: []Item{
		{ID: "gui-own", Steps: []Step{{ID: "do", Verify: "go test ./..."}}},
	}}
	for _, role := range []string{"gui", worklease.UnassignedRole} {
		if other := otherLane(document, role); other != "" {
			t.Fatalf("the lane's plan refuses role %q: %s", role, other)
		}
	}
	if other := otherLane(document, "master"); other == "" {
		t.Fatal("another lane's role is given the plan's rows")
	}
	if err := admits(document, "master", true, Ref{Item: "gui-own", Step: "do"}); err == nil {
		t.Fatal("another lane's role lands the plan's row")
	}
	if err := admits(document, "gui", true, Ref{Item: "gui-own", Step: "do"}); err != nil {
		t.Fatalf("the lane's role is refused its own row: %v", err)
	}
	if err := Validate(Plan{Lane: "bad\nlane", Items: document.Items}); err == nil {
		t.Fatal("an invalid lane name validated")
	}
}
