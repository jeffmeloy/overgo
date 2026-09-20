package gate

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestScopeBudgetForcesReplan holds a landing to the scope its row declared
// through the plan: an undeclared row maintains, so any growth is refused
// while a shrinking change is not; a declared budget admits growth up to it
// and no further; every refusal names re-budgeting through the plan or
// splitting; and a harness raise finds its paydown row only while that row
// is open.
func TestScopeBudgetForcesReplan(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name          string
		budget, grown int
		refused       string
	}{
		{name: "an undeclared row that shrinks", grown: -40},
		{name: "an undeclared row that holds", grown: 0},
		{name: "an undeclared row that grows", grown: 1, refused: "declared budget of 0"},
		{name: "growth up to the budget", budget: 100, grown: 100},
		{name: "growth over the budget", budget: 100, grown: 101, refused: "declared budget of 100"},
		{name: "a paydown row that does not shrink", budget: -50, grown: -49, refused: "declared budget of -50"},
	} {
		err := scopeBudgetAdmission("the-row/do", test.budget, test.grown)
		switch {
		case test.refused == "" && err != nil:
			t.Fatalf("%s was refused: %v", test.name, err)
		case test.refused != "" && (err == nil || !strings.Contains(err.Error(), test.refused) ||
			!strings.Contains(err.Error(), "-budget") || !strings.Contains(err.Error(), "the-row") || !strings.Contains(err.Error(), "split")):
			t.Fatalf("%s = %v, want a refusal naming %q, the re-budget command and the split", test.name, err, test.refused)
		}
	}

	repo := t.TempDir()
	publish := func(paydownStatus string) {
		t.Helper()
		document := plan.Plan{Items: []plan.Item{
			{ID: "the-row", Status: plan.StatusOpen, Budget: &plan.Budget{Nodes: 44, Paydown: "the-paydown", Reason: "schema field"}},
			{ID: "the-paydown", Status: paydownStatus},
		}}
		path := filepath.Join(repo, filepath.FromSlash(plan.Path))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, mustGateValue(json.Marshal(document)), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for status, want := range map[string]string{plan.StatusOpen: "the-paydown", "done": ""} {
		publish(status)
		for reference, paydown := range map[string]string{"the-row/do": want, "the-paydown/do": "", "absent/do": ""} {
			g := &gateContext{repo: repo, planRef: reference}
			if got, err := g.openPaydown(); err != nil || got != paydown {
				t.Fatalf("paydown of %s while the paydown row is %s = %q, %v; want %q", reference, status, got, err, paydown)
			}
		}
	}
	if budget, _, err := (&gateContext{repo: repo, planRef: "the-row/do"}).rowBudget(); err != nil || budget.Nodes != 44 {
		t.Fatalf("declared budget = %+v, %v", budget, err)
	}
}
