package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/worklease"
)

// TestAPlanEditNamesTheStepDispatchGives holds the next step a plan edit prints to the one -next gives (dispatch-next/do):
// on 2026-11-15 a doctrine decision held single-source and -next gave identity-gate/do, while each -move printed
// "next: single-source / skill-source", the first open step. A row a decision holds is passed over in the edit's line
// as in dispatch.
func TestAPlanEditNamesTheStepDispatchGives(t *testing.T) {
	t.Parallel()
	document := plan.Plan{Campaign: "fixture", Items: []plan.Item{
		{ID: "held", Title: "held", Steps: []plan.Step{openStep("do")}},
		{ID: "ready", Title: "ready", Steps: []plan.Step{openStep("do")}},
	}, Decisions: []plan.Decision{{ID: "form", Question: "which form?", Options: []string{"one", "other"}, Recommendation: "one",
		Blocks: []string{"held"}, Asked: "2026-11-15", Doctrine: true}}}
	if line, open := nextAction(document, worklease.UnassignedRole); !open || !strings.HasPrefix(line, "ready / do") {
		t.Fatalf("with held waiting on a doctrine decision a plan edit names %q, want ready / do", line)
	}
}
