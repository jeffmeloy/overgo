package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestASetCheckSaysTheTrunksCheckJudgesItsLanding holds what plan -setverify says (single-source/check-said): on
// 2026-11-15 finding-fields' check moved to another package with -setverify and its landing was refused at form, since
// a landing runs the check the trunk's plan holds for its step (SQA-378), which nothing said. A check set over another
// the trunk holds says the trunk's and that a change of the plan alone lands the new one first; the trunk's own check
// set again says nothing more.
func TestASetCheckSaysTheTrunksCheckJudgesItsLanding(t *testing.T) {
	t.Parallel()
	root, _ := editFixture(t, plan.Plan{Campaign: "fixture", Items: []plan.Item{
		{ID: "first", Title: "first", Steps: []plan.Step{openStep("do")}},
		{ID: "row", Title: "row", Steps: []plan.Step{{ID: "do", Title: "row", Verify: "go test ./x"}}},
	}})
	for _, arguments := range [][]string{{"init", "-q"}, {"add", "-A"},
		{"-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "plan"}} {
		testcheck.Must(commandInput(root, nil, "git", arguments...))(t)
	}
	said := testcheck.Must(planCommand(t, root, "-setverify", "-vcmd", "go test ./y", "row", "do"))(t)
	if !strings.Contains(said, "runs the trunk's check, go test ./x,") || !strings.Contains(said, "a change of the plan alone") {
		t.Fatalf("a check set over the trunk's says:\n%s\nwant the trunk's check and that a change of the plan alone lands it first", said)
	}
	if said := testcheck.Must(planCommand(t, root, "-setverify", "-vcmd", "go test ./x", "row", "do"))(t); strings.Contains(said, "trunk's check") {
		t.Fatalf("the trunk's own check set again says:\n%s", said)
	}
}
