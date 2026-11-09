package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestAnOwnersAnswerMayBeNoneOfTheOptions: plan -answer records the owner's answer in their own words when it is none
// of the decision's options, as the owner set aside gemma-12b-judge's choice between converting a model and labelling
// cases by hand (2026-11-09). The decision leaves the open questions and its answer waits for the next landing's
// message, as an option's does. An answer in the owner's words that is an option, an answer given both ways or neither,
// and a second answer are refused, the plan as it was.
func TestAnOwnersAnswerMayBeNoneOfTheOptions(t *testing.T) {
	t.Parallel()
	held := plan.Plan{
		Items: []plan.Item{{ID: "row", Title: "Row", Status: plan.StatusOpen, Steps: []plan.Step{openStep("do")}}},
		Decisions: []plan.Decision{{ID: "choose", Question: "Convert the model, or label cases by hand?", Options: []string{"convert", "label"},
			Recommendation: "convert", Blocks: []string{"row"}, Asked: "2026-11-09"}},
	}
	refusing, written := editFixture(t, held)
	const usage = "plan -answer -reason <an answer that is none of its options> <decision-id>"
	requireRefused(t, refusing, written, usage, "-answer", "choose")
	requireRefused(t, refusing, written, usage, "-answer", "-title", "convert", "-reason", "Neither.", "choose")
	requireRefused(t, refusing, written, "give it as the option it is", "-answer", "-reason", "label", "choose")

	root, _ := editFixture(t, held)
	words := "Neither: read its vision tower from the published tensors."
	output, err := planCommand(t, root, "-answer", "-reason", words, "choose")
	if err != nil || !strings.Contains(output, "answered choose: "+words) {
		t.Fatalf("plan -answer in the owner's words: %v\n%s", err, output)
	}
	after := editedPlan(t, root)
	if len(after.Decisions) != 0 || len(after.Answered) != 1 || after.Answered[0].Answer != words {
		t.Fatalf("after the answer the plan holds open %+v and answered %+v, want it answered in the owner's words", after.Decisions, after.Answered)
	}
	if output, err := planCommand(t, root, "-answer", "-reason", "Convert it after all.", "choose"); err == nil {
		t.Fatalf("a second answer was recorded:\n%s", output)
	}
	requireNoCheckRan(t, root)
}
