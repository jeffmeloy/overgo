package plan

import (
	"strings"
	"testing"
)

// TestAnswerTakesOneOfTheOptions holds an answer to the decision's own
// options, given once: an option outside them, an unknown decision and a
// second answer are refused, and the original plan is left as it was.
func TestAnswerTakesOneOfTheOptions(t *testing.T) {
	t.Parallel()
	document := Plan{Decisions: []Decision{{ID: "choose", Question: "Which?", Options: []string{"a", "b"}, Recommendation: "a", Asked: "2026-09-28"}}}
	if _, err := Answer(document, "choose", "c"); err == nil {
		t.Fatal("an answer outside the options was recorded")
	}
	if _, err := Answer(document, "other", "a"); err == nil {
		t.Fatal("an answer to an unknown decision was recorded")
	}
	answered, err := Answer(document, "choose", "b")
	if err != nil || answered.Decisions[0].Answer != "b" || document.Decisions[0].Answer != "" {
		t.Fatalf("answered = %+v, original = %+v, %v", answered.Decisions, document.Decisions, err)
	}
	if _, err := Answer(answered, "choose", "a"); err == nil {
		t.Fatal("a second answer was recorded")
	}
}

// TestADecisionBlocksNamedItems holds what a decision blocks to item names:
// an item that has landed and left the plan is still a name, and a blank
// entry, which blocks nothing while saying it blocks something, is refused.
func TestADecisionBlocksNamedItems(t *testing.T) {
	t.Parallel()
	decision := Decision{ID: "choose", Question: "Which?", Options: []string{"a", "b"}, Recommendation: "a", Asked: "2026-09-28"}
	for _, blocks := range [][]string{nil, {"a-row-that-landed"}} {
		decision.Blocks = blocks
		if err := Validate(Plan{Campaign: "fixture", Decisions: []Decision{decision}}); err != nil {
			t.Fatalf("a decision blocking %q was refused: %v", blocks, err)
		}
	}
	for _, blocks := range [][]string{{""}, {"a-row", ""}, {"an/item/path"}} {
		decision.Blocks = blocks
		if err := Validate(Plan{Campaign: "fixture", Decisions: []Decision{decision}}); err == nil {
			t.Fatalf("a decision blocking %q was accepted", blocks)
		}
	}
}

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
	document := Plan{Campaign: "fixture", Items: []Item{row("first"), row("second")}, Decisions: []Decision{decision}}
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
