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
	return scopeBudgetAdmission(g.planRef, budget.Nodes, grown)
}

func scopeBudgetAdmission(reference string, budget, grown int) error {
	if grown <= budget {
		return nil
	}
	item, _, _ := strings.Cut(reference, "/")
	return fmt.Errorf("scope budget: production nodes grew %+d against a declared budget of %d; re-budget the row through the plan (`go run ./cmd/plan -budget -nodes <n> -reason <why> %s`) or split it", grown, budget, item)
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
