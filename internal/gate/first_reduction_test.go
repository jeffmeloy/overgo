package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/automationcheck"
)

// firstReductionCeiling bounds the complete group a plan-internal change may
// select after the work-lease leaf took the operation runtime's only plan
// import. Measured 2026-09-14: the plan's production compile closure fell
// from 82 dependents to 7; the retained gate of c578c882, a plan change
// before the cut, ran 88 dependents in the complete group, and the same
// kind of change after the cut ran 14. The packages that still reach the
// plan do so through a named command: agenttool runs ./cmd/plan, and the
// server and the browser lane compile agenttool.
const firstReductionCeiling = 20

// TestFirstReductionAcceptance holds the first real reduction on the live
// repository and on a fixture: a change inside internal/plan no longer
// selects the operation runtime or the library intake, because they consume
// work leases from the worklease leaf instead of the plan, and the device
// lane leaves the closure; a change to that leaf still reaches them; and a
// seeded regression in the leaf is caught by the consumer the reduction
// leaves selected.
func TestFirstReductionAcceptance(t *testing.T) {
	t.Parallel()
	t.Run("fixture", firstReductionFixture)
	live := liveRepositoryFixture(t)
	isolated := []string{"overgo/internal/operation", "overgo/internal/libraryintake"}
	live.plan(t, []string{"internal/plan/frontier.go"}, func(g *gateContext) {
		scope, err := g.deriveTestScope()
		if err != nil {
			t.Fatal(err)
		}
		selected := scope.selected()
		if !slices.Contains(selected, "overgo/internal/gate") {
			t.Fatalf("plan change lost its reader internal/gate: %v", selected)
		}
		for _, name := range isolated {
			if slices.Contains(selected, name) {
				t.Errorf("plan change still selects %s, which consumes work leases without the plan", name)
			}
		}
		if len(scope.dependent) >= firstReductionCeiling {
			t.Errorf("plan change selects %d dependents, ceiling %d: %v", len(scope.dependent), firstReductionCeiling, scope.dependent)
		}
		checks := ownedChecks(g)
		resolver, err := g.dependencyResolver()
		if err != nil {
			t.Fatal(err)
		}
		impact := automationcheck.OwnershipByDependency(checks, []string{"internal/plan"}, resolver)
		if reason, excluded := impact.ExclusionReason("device"); !excluded || !strings.Contains(reason, "closure") {
			t.Errorf("plan change reached the device lane: excluded=%v reason=%s", excluded, reason)
		}
		// The browser lane stays: its server compiles agenttool, which runs
		// ./cmd/plan by name, and a named command is a real reach.
		reason, excluded := impact.ExclusionReason(automationcheck.WebUICheckName)
		t.Logf("plan change: %s excluded=%v reason=%s", automationcheck.WebUICheckName, excluded, reason)
		// The leaf that carries the leases is the boundary: a change to it
		// reaches every consumer the plan change left out and keeps the lane.
		relevant := automationcheck.OwnershipByDependency(checks, []string{"internal/worklease"}, resolver)
		if _, excluded := relevant.ExclusionReason(automationcheck.WebUICheckName); excluded {
			t.Error("work-lease change excluded the browser lane whose server consumes leases")
		}
		t.Logf("plan change: direct=%d uncertain=%d dependent=%d excluded=%d; dependents=%v", len(scope.direct), len(scope.uncertain), len(scope.dependent), scope.excluded, scope.dependent)
	})
	live.plan(t, []string{"internal/worklease/worklease.go"}, func(g *gateContext) {
		scope, err := g.deriveTestScope()
		if err != nil {
			t.Fatal(err)
		}
		selected := scope.selected()
		for _, name := range isolated {
			if !slices.Contains(selected, name) {
				t.Errorf("work-lease change no longer reaches %s", name)
			}
		}
		t.Logf("work-lease change: direct=%d uncertain=%d dependent=%d excluded=%d", len(scope.direct), len(scope.uncertain), len(scope.dependent), scope.excluded)
	})
}

// ownedChecks lists the pipeline checks that declare package ownership.
func ownedChecks(g *gateContext) []automationcheck.Check {
	var checks []automationcheck.Check
	for _, check := range g.pipelineChecks() {
		if check.Descriptor.Ownership.Fact != "" {
			checks = append(checks, check)
		}
	}
	return checks
}

// firstReductionFixture: lease is a leaf both plan and runtime import; a
// plan change selects plan's importers and not runtime; a seeded regression
// in lease selects runtime and fails its test.
func firstReductionFixture(t *testing.T) {
	g := scopeCompilerFixture(t)
	write := func(name, content string) {
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/lease/lease.go", "package lease\nfunc Owner() int { return 1 }\n")
	write("internal/runtime/runtime.go", "package runtime\nimport \"overgo/internal/lease\"\nfunc Value() int { return lease.Owner() }\n")
	write("internal/runtime/runtime_test.go", "package runtime\nimport \"testing\"\nfunc TestValue(t *testing.T) { if Value() != 1 { t.Fatal(Value()) } }\n")
	write("internal/dispatch/dispatch.go", "package dispatch\nimport (\"overgo/internal/lease\"; \"overgo/internal/plan\")\nfunc Value() int { return lease.Owner() + plan.Value }\n")
	write("internal/dispatch/dispatch_test.go", "package dispatch\nimport \"testing\"\nfunc TestValue(t *testing.T) { if Value() != 2 { t.Fatal(Value()) } }\n")
	runGitFixture(t, g.repo, "add", ".")
	g.packageGraph = nil
	g.paths = []string{"internal/plan/plan.go"}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	selected := scope.selected()
	if !slices.Contains(selected, "overgo/internal/dispatch") || slices.Contains(selected, "overgo/internal/runtime") {
		t.Fatalf("plan change: dispatch selected=%v runtime selected=%v", slices.Contains(selected, "overgo/internal/dispatch"), slices.Contains(selected, "overgo/internal/runtime"))
	}
	write("internal/lease/lease.go", "package lease\nfunc Owner() int { return 2 }\n")
	g.packageGraph = nil
	g.paths = []string{"internal/lease/lease.go"}
	scope, err = g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	if selected := scope.selected(); !slices.Contains(selected, "overgo/internal/runtime") || !slices.Contains(selected, "overgo/internal/dispatch") {
		t.Fatalf("lease change omitted a consumer: %v", selected)
	}
	if output, err := commandEnvironment(g.repo, os.Environ(), "go", "test", "./internal/runtime", "-count=1"); err == nil {
		t.Fatalf("seeded lease regression escaped the selected consumer: %s", output)
	}
}
