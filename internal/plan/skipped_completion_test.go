package plan

import (
	"strings"
	"testing"
	"time"
)

// refusalPlan: a guard row, a strict dependent that needs the guard's
// success, and a lenient dependent that accepts the guard's refusal.
func refusalPlan() Plan {
	return Plan{Items: []Item{
		{ID: "guard", Status: StatusOpen, Steps: []Step{{ID: "check", Status: StatusOpen, Verify: "go test ./guard"}}},
		{ID: "strict", Status: StatusOpen, Steps: []Step{{
			ID: "pass", Status: StatusOpen, Verify: "go test ./strict", DependsOn: []string{"guard/check"},
		}}},
		{ID: "lenient", Status: StatusOpen, Steps: []Step{{
			ID: "pass", Status: StatusOpen, Verify: "go test ./lenient", DependsOn: []string{"guard/check"}, AcceptsRefusal: []string{"guard/check"},
		}}},
	}}
}

// TestSkippedAndCancelledRowAuthority pins: a skip or cancel is recorded on
// the row with its reason and the row stays in the plan; a dependent that
// accepts the refusal dispatches, a dependent that requires success does
// not and the frontier names why; a refusal needs a reason and a refusal
// disposition; the schema refuses a refused status without its record and
// an accepted refusal that is not a dependency.
func TestSkippedAndCancelledRowAuthority(t *testing.T) {
	now := time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)
	for _, disposition := range []Disposition{DispositionSkip, DispositionCancel} {
		t.Run(string(disposition), func(t *testing.T) {
			refusedPlan, err := Refuse(refusalPlan(), "guard", "check", disposition, "the 12B guard failed", now)
			if err != nil {
				t.Fatal(err)
			}
			guard := refusedPlan.Items[0].Steps[0]
			if guard.Status != refusalStatus(disposition) || guard.Refusal == nil || guard.Refusal.Reason != "the 12B guard failed" || guard.Refusal.Recorded != "2026-09-06T12:00:00Z" {
				t.Fatalf("refused guard = %+v", guard)
			}
			authority := testCompletionAuthority(t, refusedPlan)
			frontier, err := ReadyFrontier(refusedPlan, authority)
			if err != nil || len(frontier) != 1 || frontier[0].String() != "lenient/pass" {
				t.Fatalf("frontier = %v, %v; want the lenient dependent only", frontier, err)
			}
			item, step, ok := Current(refusedPlan, UnassignedRole, authority)
			if !ok || item.ID != "lenient" || step.ID != "pass" {
				t.Fatalf("dispatch = %s/%s ok=%t; want lenient/pass", item.ID, step.ID, ok)
			}
			strict := refusedPlan.Items[1].Steps[0]
			refusedLines := RefusedDependencies(refusedPlan, strict)
			if len(refusedLines) != 1 || !strings.Contains(refusedLines[0], "guard/check "+string(disposition)+": the 12B guard failed") {
				t.Fatalf("strict refused dependencies = %v", refusedLines)
			}
			if lines := RefusedDependencies(refusedPlan, refusedPlan.Items[2].Steps[0]); len(lines) != 0 {
				t.Fatalf("lenient dependent reports refusals: %v", lines)
			}
			if _, err := Advance(refusedPlan, "guard", "check"); err == nil {
				t.Fatal("a refused row advanced as done")
			}
			if _, err := Refuse(refusedPlan, "guard", "check", disposition, "again", now); err == nil {
				t.Fatal("a refused row was refused twice")
			}
		})
	}

	if _, err := Refuse(refusalPlan(), "guard", "check", DispositionProceed, "reason", now); err == nil {
		t.Fatal("proceed was recorded as a refusal")
	}
	if _, err := Refuse(refusalPlan(), "guard", "check", DispositionSkip, "  ", now); err == nil {
		t.Fatal("a refusal without a reason was recorded")
	}
	bare := refusalPlan()
	bare.Items[0].Steps[0].Status = StatusSkipped
	if err := Validate(bare); err == nil || !strings.Contains(err.Error(), "refusal record") {
		t.Fatalf("skipped status without a record validated: %v", err)
	}
	stray := refusalPlan()
	stray.Items[1].Steps[0].AcceptsRefusal = []string{"lenient/pass"}
	if err := Validate(stray); err == nil || !strings.Contains(err.Error(), "not a declared dependency") {
		t.Fatalf("accepted refusal of a non-dependency validated: %v", err)
	}
	// Both dependents wait while the guard is open: a refusal is never inferred.
	open := refusalPlan()
	frontier, err := ReadyFrontier(open, testCompletionAuthority(t, open))
	if err != nil || len(frontier) != 1 || frontier[0].String() != "guard/check" {
		t.Fatalf("open frontier = %v, %v", frontier, err)
	}
}
