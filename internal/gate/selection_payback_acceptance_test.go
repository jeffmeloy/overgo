package gate

import (
	"slices"
	"testing"
)

// TestSelectionPaybackAcceptance is the first-reduction payback decision: the
// work-lease-leaf cut is adopted only because a plan-internal change avoids real
// complete-group execution, not merely because it names fewer packages. It
// compares the expensive dependent group a change inside internal/plan selects
// against the group a change to the work-lease leaf those packages consume
// selects, and requires the difference to be concrete: agentloop, a work-lease
// consumer that never reaches the plan, drops out of the plan change while a
// change to the leaf still runs it, and the plan change's complete group is
// strictly smaller, so the avoided work is genuine and no seeded regression in
// the leaf escapes the selection that pays the cut back.
func TestSelectionPaybackAcceptance(t *testing.T) {
	t.Parallel()
	live := liveRepositoryFixture(t)
	// A work-lease-only consumer: it reaches the leaf through work leases, not
	// through the plan the changed files own, so a plan change avoids it while a
	// change to the leaf still runs it.
	const saved = "overgo/internal/agentloop"

	var planSelected, planDependent []string
	live.plan(t, []string{"internal/plan/frontier.go"}, func(g *gateContext) {
		scope, err := g.deriveTestScope()
		if err != nil {
			t.Fatal(err)
		}
		planSelected, planDependent = scope.selected(), slices.Clone(scope.dependent)
	})

	var leafSelected, leafDependent []string
	live.plan(t, []string{"internal/worklease/worklease.go"}, func(g *gateContext) {
		scope, err := g.deriveTestScope()
		if err != nil {
			t.Fatal(err)
		}
		leafSelected, leafDependent = scope.selected(), slices.Clone(scope.dependent)
	})

	// The consumer is avoided by the plan change and still run by a change to the
	// leaf, so its exclusion is saved execution rather than a lost regression path.
	if slices.Contains(planSelected, saved) {
		t.Errorf("plan change still selects %s; the reduction saves no work", saved)
	}
	if !slices.Contains(leafSelected, saved) {
		t.Errorf("leaf change does not run %s; a regression there would escape the selection", saved)
	}

	// Compare the actual expensive group before and after: a package-count
	// reduction without a smaller complete group is not payback. The plan change's
	// dependent group is strictly smaller than the leaf change's.
	if len(planDependent) >= len(leafDependent) {
		t.Fatalf("no complete-group payback: plan dependents %d not fewer than leaf dependents %d", len(planDependent), len(leafDependent))
	}
}
