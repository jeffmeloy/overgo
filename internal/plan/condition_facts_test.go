package plan

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestConditionFactsFromDocumentAndAuthority pins: completed references
// expose their snapshot outcomes, retained rows expose recorded outcomes and
// refusals expose disposition and reason; blocked rows name each unsatisfied
// dependency by state.
func TestConditionFactsFromDocumentAndAuthority(t *testing.T) {
	document := Plan{Items: []Item{
		{ID: "guard", Status: StatusOpen, Steps: []Step{{ID: "check", Status: StatusOpen, Verify: "go test ./guard"}}},
		{ID: "measure", Status: StatusOpen, Steps: []Step{{
			ID: "small", Status: StatusOpen, Verify: "go test ./small", Outcome: json.RawMessage(`{"pass_rate": 0.4}`),
		}}},
		{ID: "strict", Status: StatusOpen, Steps: []Step{{
			ID: "pass", Status: StatusOpen, Verify: "go test ./strict", DependsOn: []string{"guard/check", "measure/small", "gone/row", "lost/row"},
		}}},
	}}
	refusedPlan, err := Refuse(document, "guard", "check", DispositionSkip, "guard failed", time.Date(2026, 9, 6, 0, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	authority := testCompletionAuthority(t, refusedPlan, "gone/row")
	authority.completedReferences["gone/row"] = completionEvidence{outcome: `{"accuracy": 0.9, "model": "9b"}`}
	facts := DocumentConditionFacts(refusedPlan, authority)
	if facts.Outcomes["gone/row"]["accuracy"] != 0.9 || facts.Outcomes["gone/row"]["model"] != "9b" {
		t.Fatalf("completed outcome = %v", facts.Outcomes["gone/row"])
	}
	if facts.Outcomes["measure/small"]["pass_rate"] != 0.4 {
		t.Fatalf("retained outcome = %v", facts.Outcomes["measure/small"])
	}
	if facts.Outcomes["guard/check"]["disposition"] != "skip" || facts.Outcomes["guard/check"]["reason"] != "guard failed" {
		t.Fatalf("refusal facts = %v", facts.Outcomes["guard/check"])
	}
	frontier, err := ReadyFrontier(refusedPlan, authority)
	if err != nil {
		t.Fatal(err)
	}
	blocked := BlockedRows(refusedPlan, frontier, authority)
	report := FormatBlocked(blocked)
	for _, want := range []string{
		"blocked: strict/pass waits on guard/check skipped: guard failed; measure/small open; lost/row absent without completion evidence",
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("blocked report lacks %q:\n%s", want, report)
		}
	}
	if strings.Contains(report, "gone/row") || strings.Contains(report, "measure/small waits") {
		t.Fatalf("blocked report names a satisfied dependency or a ready row:\n%s", report)
	}
}
