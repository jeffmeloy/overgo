package gate

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/codeprofile"
	"overgo/internal/plan"
)

// rowBudget loads the scope this gate's row declared through the plan and
// the plan it was declared in. Undeclared is the zero budget: maintain.
func (g *gateContext) rowBudget() (plan.Budget, plan.Plan, error) {
	document, err := plan.Load(filepath.Join(g.repo, filepath.FromSlash(plan.Path)))
	if err != nil {
		return plan.Budget{}, plan.Plan{}, err
	}
	item, _, _ := strings.Cut(g.planRef, "/")
	index := slices.IndexFunc(document.Items, func(row plan.Item) bool { return row.ID == item })
	if index < 0 || document.Items[index].Budget == nil {
		return plan.Budget{}, document, nil
	}
	return *document.Items[index].Budget, document, nil
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
	budget, _, err := g.rowBudget()
	if err != nil {
		return err
	}
	grown := candidate.Runtime.Nodes + candidate.Automation.Nodes - base.Runtime.Nodes - base.Automation.Nodes
	return scopeBudgetAdmission(g.planRef, budget, grown)
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
	budget, document, err := g.rowBudget()
	open := func(row plan.Item) bool { return row.ID == budget.Paydown && row.Status == plan.StatusOpen }
	if err != nil || !slices.ContainsFunc(document.Items, open) {
		return "", err
	}
	return budget.Paydown, nil
}
