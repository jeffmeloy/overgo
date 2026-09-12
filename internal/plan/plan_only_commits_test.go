package plan

import (
	"strings"
	"testing"
)

// TestPlanOnlyCommitsAreMergesOnly pins the owner's rule of 2026-09-10: a
// lane commit whose planned paths carry only the plan is refused unless it
// is a merge, an implementation commit carries its re-planning in the same
// plan path, and master's plan, which declares no lane, is untouched. The
// rule applies independently of a lane's doctrine wording.
func TestPlanOnlyCommitsAreMergesOnly(t *testing.T) {
	lane := Plan{Lane: "hatchet"}
	if err := PlanOnlyCommitRefusal(lane, "gate-wall/validate-first", []string{Path}, false); err == nil || !strings.Contains(err.Error(), "plan-only lane commit") {
		t.Fatalf("plan-only lane commit = %v, want the refusal", err)
	}
	for name, paths := range map[string][]string{
		"implementation with re-plan": {"internal/gate/verification.go", Path},
		"implementation alone":        {"internal/gate/verification.go"},
	} {
		if err := PlanOnlyCommitRefusal(lane, "gate-wall/validate-first", paths, false); err != nil {
			t.Fatalf("%s = %v, want admission", name, err)
		}
	}
	if err := PlanOnlyCommitRefusal(lane, MergeItemPrefix+"2ef953fab346/do", []string{Path}, false); err != nil {
		t.Fatalf("merge row = %v, want admission", err)
	}
	if err := PlanOnlyCommitRefusal(lane, "gate-wall/validate-first", []string{Path}, true); err != nil {
		t.Fatalf("merge gate = %v, want admission", err)
	}
	if err := PlanOnlyCommitRefusal(Plan{}, "modality-verification/do", []string{Path}, false); err != nil {
		t.Fatalf("master's plan = %v, want admission", err)
	}
	for _, document := range []Plan{
		{Lane: "audio"},
		{Lane: "audio", Doctrine: "Keep the campaign focused on audio capabilities."},
		{Lane: "hatchet", Doctrine: "only a merge lands a plan-only commit"},
	} {
		if err := PlanOnlyCommitRefusal(document, "implementation/do", []string{Path}, false); err == nil {
			t.Fatalf("lane %q doctrine %q admitted a plan-only commit", document.Lane, document.Doctrine)
		}
	}
}
