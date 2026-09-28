package plan

import (
	"strings"
	"testing"
)

// TestDispatchPassesRowsAwaitingADecision holds dispatch to the owner's
// open decisions: a row an unanswered decision blocks is passed over for
// the next ready row, an answer releases it, and when every open row waits
// on a decision dispatch says so instead of offering one.
func TestDispatchPassesRowsAwaitingADecision(t *testing.T) {
	t.Parallel()
	row := func(id string) Item {
		return Item{ID: id, Title: id, Status: StatusOpen, Steps: []Step{{ID: "do", Title: id, Status: StatusOpen, Verify: "true"}}}
	}
	decision := Decision{ID: "choose", Question: "Which?", Options: []string{"a", "b"}, Recommendation: "a", Blocks: []string{"first"}, Asked: "2026-09-28"}
	document := Plan{Campaign: "fixture", Doctrine: "fixture", Items: []Item{row("first"), row("second")}, Decisions: []Decision{decision}}
	if dispatch, err := selectDispatch(document, "", ""); err != nil || dispatch.Item != "second" {
		t.Fatalf("with first awaiting a decision, dispatch = %+v, %v; want second", dispatch, err)
	}
	document.Decisions[0].Answer = "a"
	if dispatch, err := selectDispatch(document, "", ""); err != nil || dispatch.Item != "first" {
		t.Fatalf("with the decision answered, dispatch = %+v, %v; want first", dispatch, err)
	}
	document.Decisions[0].Answer = ""
	document.Items = document.Items[:1]
	if dispatch, err := selectDispatch(document, "", ""); err != nil || dispatch.Item != "" || !strings.Contains(dispatch.Waiting, "owner decisions") {
		t.Fatalf("with every row awaiting a decision, dispatch = %+v, %v; want it to wait on the decision", dispatch, err)
	}
}
