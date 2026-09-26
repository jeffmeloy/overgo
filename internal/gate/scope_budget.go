package gate

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/codeprofile"
	"overgo/internal/plan"
)

// row loads this gate's row as the plan declares it, and that plan; a gate
// with no row, or a row the plan no longer holds, declares nothing: its zero
// budget maintains.
func (g *gateContext) row() (plan.Item, plan.Plan, error) {
	document, err := plan.Load(filepath.Join(g.repo, filepath.FromSlash(plan.Path)))
	item, _, _ := strings.Cut(g.planRef, "/")
	if index := slices.IndexFunc(document.Items, func(row plan.Item) bool { return row.ID == item }); err == nil && index >= 0 {
		return document.Items[index], document, nil
	}
	return plan.Item{}, document, err
}

// admitScopeBudget refuses production-node growth the row never declared.
// The refusal names the two ways on, both of which go through the plan, so
// a pivot is a recorded re-scope and never rides the row in flight. A merge
// carries rows that were each held at their own landing, and a gate with no
// row has nothing to outgrow.
func (g *gateContext) admitScopeBudget(base, candidate codeprofile.Profile) error {
	if g.planRef == "" || g.mergeBefore != nil {
		return nil
	}
	row, _, err := g.row()
	if err != nil {
		return err
	}
	grown := candidate.Runtime.Nodes + candidate.Automation.Nodes - base.Runtime.Nodes - base.Automation.Nodes
	return scopeBudgetAdmission(g.planRef, row.Budget, grown)
}

// scopeBudgetAdmission admits growth a row declared: a declared reason admits
// the growth the gate measures, and a declared cap -- negative for a row that
// must shrink -- bounds it. An undeclared row maintains.
func scopeBudgetAdmission(reference string, budget plan.Budget, grown int) error {
	if grown <= budget.Nodes || budget.Nodes == 0 && budget.Reason != "" {
		return nil
	}
	declared := "no declared reason"
	if budget.Nodes != 0 {
		declared = fmt.Sprintf("a declared cap of %d", budget.Nodes)
	}
	item, _, _ := strings.Cut(reference, "/")
	return fmt.Errorf("scope budget: production nodes grew %+d with %s; declare why the row grows through the plan (`go run ./cmd/plan -budget -reason <why> %s`) or split it", grown, declared, item)
}

// openPaydown names the open row this gate's row declared as paying for a
// harness baseline raise, empty when it declared none or that row has closed.
func (g *gateContext) openPaydown() (string, error) {
	if g.planRef == "" {
		return "", nil
	}
	row, document, err := g.row()
	open := func(paying plan.Item) bool {
		return paying.ID == row.Budget.Paydown && paying.Status == plan.StatusOpen
	}
	if err != nil || !slices.ContainsFunc(document.Items, open) {
		return "", err
	}
	return row.Budget.Paydown, nil
}
