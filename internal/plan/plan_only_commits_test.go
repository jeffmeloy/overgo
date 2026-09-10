package plan

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestPlanOnlyCommitsAreMergesOnly pins the owner's rule of 2026-09-10: a
// lane commit whose planned paths carry only the plan is refused unless it
// is a merge, an implementation commit carries its re-planning in the same
// plan path, and master's plan, which declares no lane, is untouched. The
// lane's live doctrine states the rule.
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
	document, err := Load(filepath.Join(testutil.RepoRoot(t), filepath.FromSlash(Path)))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "" && !strings.Contains(document.Doctrine, "only a merge lands a plan-only commit") {
		t.Fatal("the lane doctrine omits the plan-only commit rule")
	}
}
