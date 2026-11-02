package landing

import (
	"fmt"
	"slices"

	"overgo/internal/plan"
)

// The trailers a commit states a row's altered terms with.
const (
	changedCheckTrailer = "Changed-Check"
	changedCountTrailer = "Changed-Count"
)

// termsChanged writes each row's check and count a change alters, old and new, so a weakened check or a raised count reads
// in the commit: a plan-only commit is judged by the plan it carries, and a change may raise its own count. A row the change
// adds or removes has no term to set beside its own.
func termsChanged(before, after plan.Plan) []string {
	var changed []string
	for _, item := range after.Items {
		at := slices.IndexFunc(before.Items, func(held plan.Item) bool { return held.ID == item.ID })
		if at < 0 {
			continue
		}
		held := before.Items[at]
		if held.Budget.Nodes != item.Budget.Nodes {
			changed = append(changed, fmt.Sprintf("%s: %s: %d -> %d: %q", changedCountTrailer, item.ID, held.Budget.Nodes, item.Budget.Nodes, item.Budget.Reason))
		}
		for _, step := range item.Steps {
			index := slices.IndexFunc(held.Steps, func(old plan.Step) bool { return old.ID == step.ID })
			if index >= 0 && held.Steps[index].Verify != step.Verify {
				changed = append(changed, fmt.Sprintf("%s: %s/%s: %q -> %q", changedCheckTrailer, item.ID, step.ID, held.Steps[index].Verify, step.Verify))
			}
		}
	}
	return changed
}
