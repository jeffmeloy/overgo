package gate

import (
	"slices"
	"testing"
)

// TestRuntimeCallIsolationAcceptance holds the runtime-input side of the
// work-lease cut on the live repository: internal/plan/plan.go is the reader
// of docs/plan.json, so a change to it selects every package that compiles
// the plan; the operation runtime consumes leases from the worklease leaf
// and is isolated from the reader, while the gate, which compiles the plan,
// is the reachable reader the change still selects. A change to the leaf
// itself reaches the runtime, so the isolation is a measured boundary and
// not an exemption.
func TestRuntimeCallIsolationAcceptance(t *testing.T) {
	t.Parallel()
	live := liveRepositoryFixture(t)
	const isolated, reader = "overgo/internal/operation", "overgo/internal/gate"
	live.plan(t, []string{"internal/plan/plan.go"}, func(g *gateContext) {
		scope, err := g.deriveTestScope()
		if err != nil {
			t.Fatal(err)
		}
		selected := scope.selected()
		if !slices.Contains(selected, reader) {
			t.Fatalf("plan reader change lost its importer %s: %v", reader, selected)
		}
		if slices.Contains(selected, isolated) {
			t.Fatalf("plan reader change selected the isolated consumer %s", isolated)
		}
		t.Logf("plan reader change: direct=%d uncertain=%d dependent=%d excluded=%d", len(scope.direct), len(scope.uncertain), len(scope.dependent), scope.excluded)
	})
	live.plan(t, []string{"internal/worklease/worklease.go"}, func(g *gateContext) {
		scope, err := g.deriveTestScope()
		if err != nil {
			t.Fatal(err)
		}
		if selected := scope.selected(); !slices.Contains(selected, isolated) || !slices.Contains(selected, reader) {
			t.Fatalf("work-lease change omitted a consumer: %v", selected)
		}
	})
}
