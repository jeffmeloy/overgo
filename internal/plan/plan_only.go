package plan

import (
	"fmt"
	"strings"
)

// MergeItemPrefix names the items a merge gate synthesizes; a merge row
// carries no code of its own, so it is the one plan-only lane commit.
const MergeItemPrefix = "merge-"

// PlanOnlyCommitRefusal refuses a lane commit whose planned paths carry
// nothing but the plan: routine re-planning rides in the implementation
// commit's plan path, so a lane cycle lands one commit, and only a merge
// lands a plan-only commit. Master's plan declares no lane and is untouched;
// a commit with any path beside the plan is an implementation commit.
func PlanOnlyCommitRefusal(document Plan, ref string, paths []string, merge bool) error {
	if document.Lane == "" || merge || len(paths) == 0 {
		return nil
	}
	item, _, _ := strings.Cut(ref, "/")
	if strings.HasPrefix(item, MergeItemPrefix) {
		return nil
	}
	for _, path := range paths {
		if path != Path {
			return nil
		}
	}
	return fmt.Errorf("plan: %s is a plan-only lane commit; re-plan in the implementation commit's %s path, since only a merge lands a plan-only commit", ref, Path)
}
