package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"overgo/internal/plan"
)

// TestFrontierReportsConditionOutcomes pins: the frontier report evaluates
// each ready row's conditions against recorded facts and names the verdict
// with its reason, names every blocked row with the dependency state that
// holds it, and counts only proceeding rows as dispatchable.
func TestFrontierReportsConditionOutcomes(t *testing.T) {
	document := plan.Plan{Items: []plan.Item{
		{ID: "guard", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "check", Status: plan.StatusOpen, Verify: "go test ./guard"}}},
		{ID: "big", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "pass", Status: plan.StatusOpen, Verify: "go test ./big", DependsOn: []string{"guard/check"}, AcceptsRefusal: []string{"guard/check"},
			Conditions: &plan.StepConditions{SkipIf: []plan.OutcomeCondition{{
				Parent: "guard/check", Field: "disposition", Operator: "eq", Value: json.RawMessage(`"skip"`),
			}}},
		}}},
		{ID: "strict", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "pass", Status: plan.StatusOpen, Verify: "go test ./strict", DependsOn: []string{"guard/check"},
		}}},
		{ID: "release", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "publish", Status: plan.StatusOpen, Verify: "go test ./release",
			Conditions: &plan.StepConditions{WaitFor: []plan.WaitCondition{{Event: "owner-signoff"}}},
		}}},
	}}
	refused, err := plan.Refuse(document, "guard", "check", plan.DispositionSkip, "the 12B guard failed", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	root := initializePlanTestRepository(t, refused)
	var output bytes.Buffer
	if err := printReadyFrontier(root, refused, &output); err != nil {
		t.Fatal(err)
	}
	report := output.String()
	for _, want := range []string{
		"big/pass\nrelease/publish\n",
		"condition: big/pass skip guard/check.disposition eq \"skip\"",
		"condition: release/publish wait event owner-signoff not recorded",
		"blocked: strict/pass waits on guard/check skipped: the 12B guard failed",
		"frontier: 2 dispatchable row(s), 0 proceeding",
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("frontier report lacks %q:\n%s", want, report)
		}
	}
}
