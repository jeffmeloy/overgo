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

// decisionFixture is a plan of two rows, the first blocked by an unanswered decision whose recommendation is a.
func decisionFixture() Plan {
	row := func(id string) Item {
		return Item{ID: id, Title: id, Status: StatusOpen, Steps: []Step{{ID: "do", Title: id, Status: StatusOpen, Verify: "true"}}}
	}
	decision := Decision{ID: "choose", Question: "Which?", Options: []string{"a", "b"}, Recommendation: "a", Blocks: []string{"first"}, Asked: "2026-09-28"}
	return Plan{Campaign: "fixture", Items: []Item{row("first"), row("second")}, Decisions: []Decision{decision}}
}

// TestDispatchPassesRowsAwaitingADecision holds dispatch to the open
// decisions: a row an unanswered decision blocks is passed over for the
// next ready row, and an answer releases it.
func TestDispatchPassesRowsAwaitingADecision(t *testing.T) {
	t.Parallel()
	document := decisionFixture()
	if dispatch, err := selectDispatch(document, "", ""); err != nil || dispatch.Item != "second" {
		t.Fatalf("with first awaiting a decision, dispatch = %+v, %v; want second", dispatch, err)
	}
	document.Decisions[0].Answer = "a"
	if dispatch, err := selectDispatch(document, "", ""); err != nil || dispatch.Item != "first" || len(dispatch.Decided) != 0 {
		t.Fatalf("with the decision answered, dispatch = %+v, %v; want first, no decision taken by default", dispatch, err)
	}
}

// TestAQuestionNeverHoldsTheLoop holds dispatch to never waiting on a question: when every open row waits on an
// unanswered decision, the row goes on under the decision's recommended option, and dispatch says so; the plan it
// read keeps the decision open, so an answer given before then is the one the row goes on under.
func TestAQuestionNeverHoldsTheLoop(t *testing.T) {
	t.Parallel()
	document := decisionFixture()
	document.Items = document.Items[:1]
	dispatch, err := selectDispatch(document, "", "")
	if err != nil || dispatch.Item != "first" || dispatch.Waiting != "" || len(dispatch.Decided) != 1 || dispatch.Decided[0] != "choose: a" ||
		!strings.Contains(dispatch.Line, "taken at the recommended option: choose: a") {
		t.Fatalf("with every row awaiting a decision, dispatch = %+v, %v; want first under the recommended option, said so", dispatch, err)
	}
	if document.Decisions[0].Answer != "" {
		t.Fatal("dispatch wrote an answer into the plan it read")
	}
}
