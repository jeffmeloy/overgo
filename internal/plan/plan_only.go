package plan

import (
	"fmt"
	"slices"
	"strings"
)

// PlanOnlyCommitRefusal refuses a lane commit whose planned paths carry
// nothing but the plan: routine re-planning rides in the implementation
// commit's plan path, so a lane cycle lands one commit, and only a merge
// lands a plan-only commit. Master's plan declares no lane and is untouched;
// a commit with any path beside the plan is an implementation commit.
func PlanOnlyCommitRefusal(document Plan, ref string, paths []string, merge bool) error {
	item, _, _ := strings.Cut(ref, "/")
	merging := merge || slices.ContainsFunc(document.Items, func(row Item) bool { return row.ID == item && row.Class == ClassMerge })
	if document.Lane == "" || merging || len(paths) == 0 || slices.ContainsFunc(paths, func(path string) bool { return path != Path }) {
		return nil
	}
	return fmt.Errorf("plan: %s is a plan-only lane commit; re-plan in the implementation commit's %s path, since only a merge lands a plan-only commit", ref, Path)
}
