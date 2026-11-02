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

// TestARowRunsOnlyUnderTheAnswerItWasWrittenFor: a row was written for its decision's recommendation. Answered that way it
// runs, and answered another way it is passed over, reported, and cannot land, until it is rewritten or retired.
func TestARowRunsOnlyUnderTheAnswerItWasWrittenFor(t *testing.T) {
	t.Parallel()
	document := decisionFixture()
	document.Decisions[0].Answer = "b"
	if dispatch, err := selectDispatch(document, "", ""); err != nil || dispatch.Item != "second" {
		t.Fatalf("answered another way, dispatch = %+v, %v; want second", dispatch, err)
	}
	if _, err := Advance(document, "first", "do"); err == nil {
		t.Fatal("a row answered another way landed")
	}
	document.Items = document.Items[:1]
	dispatch, err := selectDispatch(document, "", "")
	if err != nil || dispatch.Item != "" || !strings.Contains(dispatch.Waiting, "first is held by decision choose") {
		t.Fatalf("with only that row left, dispatch = %+v, %v; want it reported", dispatch, err)
	}
	document.Decisions[0].Answer = "a"
	if dispatch, err := selectDispatch(document, "", ""); err != nil || dispatch.Item != "first" {
		t.Fatalf("answered as written, dispatch = %+v, %v; want first", dispatch, err)
	}
	if _, err := Advance(document, "first", "do"); err != nil {
		t.Fatal(err)
	}
}

// TestADoctrineDecisionIsNeverTakenByRecommendation: a decision that changes a standing rule holds its rows for the owner,
// though nothing else is ready, and they cannot land; an unanswered ordinary decision goes on as it does.
func TestADoctrineDecisionIsNeverTakenByRecommendation(t *testing.T) {
	t.Parallel()
	document, err := MarkDoctrine(decisionFixture(), "choose")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := MarkDoctrine(document, "unknown"); err == nil {
		t.Fatal("an unknown decision was marked")
	}
	document.Items = document.Items[:1]
	dispatch, err := selectDispatch(document, "", "")
	if err != nil || dispatch.Item != "" || len(dispatch.Decided) != 0 || !strings.Contains(dispatch.Waiting, "waits for the owner's answer") {
		t.Fatalf("with a doctrine decision unanswered, dispatch = %+v, %v; want the row reported as waiting", dispatch, err)
	}
	if _, err := Advance(document, "first", "do"); err == nil {
		t.Fatal("a row a doctrine decision holds landed")
	}
	if ordinary := decisionFixture(); ordinary.Decisions[0].Doctrine {
		t.Fatal("the fixture's decision is doctrine")
	} else if _, err := Advance(ordinary, "first", "do"); err != nil {
		t.Fatalf("a row under an unanswered ordinary decision could not land: %v", err)
	}
	answered, err := Answer(document, "choose", "a")
	if err != nil {
		t.Fatal(err)
	}
	if dispatch, err := selectDispatch(answered, "", ""); err != nil || dispatch.Item != "first" {
		t.Fatalf("answered, dispatch = %+v, %v; want first", dispatch, err)
	}
}

// TestPromptAndNextAgreeOnAnAwaitingRow: a row that goes on by recommendation is dispatched whether it is asked for by name
// or found next, and the note names only the decisions that block it.
func TestPromptAndNextAgreeOnAnAwaitingRow(t *testing.T) {
	t.Parallel()
	document := decisionFixture()
	document.Items = document.Items[:1]
	other := document.Decisions[0]
	other.ID, other.Blocks = "unrelated", []string{"another-row"}
	document.Decisions = append(document.Decisions, other)
	next, err := selectDispatch(document, "", "")
	if err != nil {
		t.Fatal(err)
	}
	named, err := selectDispatch(document, "", "first/do")
	if err != nil {
		t.Fatal(err)
	}
	if next.Item != "first" || named.Item != "first" || named.Waiting != "" {
		t.Fatalf("next = %+v, by name = %+v", next, named)
	}
	if len(next.Decided) != 1 || next.Decided[0] != "choose: a" || len(named.Decided) != 1 {
		t.Fatalf("next decided %v, by name %v; want only the decision that blocks the row", next.Decided, named.Decided)
	}
}
