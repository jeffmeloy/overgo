package main

import (
	"testing"

	"overgo/internal/plan"
)

// TestEnforceAdvanceGate pins that a step cannot be advanced without its verify
// passing -- the machine-checked "done" that replaces self-declared prose. The
// no-verify and force paths return before invoking a shell, so this is portable.
func TestEnforceAdvanceGate(t *testing.T) {
	// No verify command defined + no force: refused.
	if err := gateAdvance(plan.Item{ID: "i"}, plan.Step{ID: "s"}, ""); err == nil {
		t.Fatal("advance with no verify and no force must be refused")
	}
	// -force overrides (the loud escape for genuinely-manual steps).
	if err := gateAdvance(plan.Item{ID: "i"}, plan.Step{ID: "s"}, "manual: owner sign-off"); err != nil {
		t.Fatalf("-force must override the verify gate: %v", err)
	}
}
