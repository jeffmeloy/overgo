package landing

import (
	"fmt"
	"slices"
	"strings"

	"overgo/internal/plan"
)

// inScope refuses a change that writes outside the files its landed steps list: on Nov 7 a helper's side effect
// rewrote the trunk's hook setting, and a weaker model wanders further. A listed directory holds everything under it,
// tests and fixtures among them; the plan and the findings register are every change's to write, and so is what the
// repository's generators derive from it, which the landing writes itself; a step that lists nothing keeps the
// admission every change has. The list is the terms' the change started with, so a change cannot widen its own.
func (l *landing) inScope(named []ref) error {
	var listed []string
	for _, row := range named {
		step, _ := plan.ExactStep(l.terms, row.item, row.step)
		if len(step.Files) == 0 {
			return nil
		}
		listed = append(listed, step.Files...)
	}
	derived := []string{plan.Path, findingsRegister}
	for _, generator := range l.generators() {
		derived = append(derived, generator.writes...)
	}
	outside := slices.DeleteFunc(slices.Clone(l.written), func(path string) bool {
		return slices.Contains(derived, path) || plan.Lists(listed, path)
	})
	if len(outside) == 0 {
		return nil
	}
	return fmt.Errorf("land: the change writes %s, outside the files %s lists (%s); edit: restore each from HEAD, or remove it if new, and file that work as a step of its own",
		strings.Join(outside, ", "), l.Ref, strings.Join(listed, ", "))
}
