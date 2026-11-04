package landing

import (
	"fmt"
	"slices"

	"overgo/internal/plan"
)

// The trailers a commit states a row's altered terms with, and a step it removes without landing it.
const (
	changedCheckTrailer = "Changed-Check"
	changedCountTrailer = "Changed-Count"
	retiredRowTrailer   = "Retired-Row"
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

// rowChanges are the steps a change removes from the plan without landing them, as item/step, and the checks it puts in
// place of a step's check. A removed step is named in the commit and in its message, and a replacing check must fail on
// the commit the change builds on, as an added step's does, so neither a deletion nor a weaker check passes unseen.
func rowChanges(before, after plan.Plan) (removed, replaced []string) {
	for _, held := range before.Items {
		var kept []plan.Step
		if at := slices.IndexFunc(after.Items, func(item plan.Item) bool { return item.ID == held.ID }); at >= 0 {
			kept = after.Items[at].Steps
		}
		for _, step := range held.Steps {
			switch index := slices.IndexFunc(kept, func(next plan.Step) bool { return next.ID == step.ID }); {
			case index < 0:
				removed = append(removed, held.ID+"/"+step.ID)
			case kept[index].Verify != step.Verify:
				replaced = append(replaced, kept[index].Verify)
			}
		}
	}
	return removed, replaced
}
