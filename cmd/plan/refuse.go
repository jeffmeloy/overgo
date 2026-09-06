package main

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/plan"
)

// refuseRow records a skip or cancel disposition with its reason on the
// named open row under the plan authority lock; the row stays in the plan
// as the refusal record and dependents resolve by their declarations.
func refuseRow(root, reference, disposition, reason, role string) error {
	itemID, stepID, bound := strings.Cut(reference, "/")
	if !bound || itemID == "" || stepID == "" {
		return errors.New("usage: plan -refuse <item>/<step> -disposition skip|cancel -reason <text>")
	}
	return withPlanMutation(root, false, func(document plan.Plan) error {
		updated, err := plan.Refuse(document, itemID, stepID, plan.Disposition(disposition), reason, time.Now())
		if err != nil {
			return err
		}
		completions, err := resolveCompletionAuthority(root, updated)
		if err != nil {
			return err
		}
		if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), updated); err != nil {
			return err
		}
		action, _ := nextAction(updated, role, completions)
		fmt.Printf("refused %s (%s: %s); next: %s\n", reference, disposition, strings.TrimSpace(reason), action)
		for _, line := range heldDependents(updated, reference) {
			fmt.Println(line)
		}
		return nil
	})
}

// heldDependents: every open row the refusal of reference now holds, with
// the refusal named; a row that accepts the refusal is not held.
func heldDependents(document plan.Plan, reference string) []string {
	var lines []string
	for _, item := range document.Items {
		for _, step := range item.Steps {
			if step.Status != plan.StatusOpen {
				continue
			}
			for _, refusal := range plan.RefusedDependencies(document, step) {
				if strings.HasPrefix(refusal, reference+" ") {
					lines = append(lines, fmt.Sprintf("holds %s/%s: %s", item.ID, step.ID, refusal))
				}
			}
		}
	}
	return lines
}
