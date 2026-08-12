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

// TestEnforceAddInsertsTask pins the mechanical task-injection: -add creates a
// top-priority (or -before) open item with a "do" step + verify, and the new item
// becomes the current step -- no hand-editing plan.json.
func TestEnforceAddInsertsTask(t *testing.T) {
	base := plan.Plan{Items: []plan.Item{
		{ID: "a", Status: "open", Steps: []plan.Step{{ID: "s", Status: "open"}}},
		{ID: "b", Status: "open", Steps: []plan.Step{{ID: "s", Status: "open"}}},
	}}
	top, err := insertItem(base, "new", "do the thing", "", "go test ./...")
	if err != nil {
		t.Fatal(err)
	}
	if top.Items[0].ID != "new" || top.Items[0].Steps[0].ID != "do" || top.Items[0].Steps[0].Verify != "go test ./..." {
		t.Fatalf("insert-at-top produced %+v", top.Items[0])
	}
	if it, st, ok := plan.Current(top); !ok || it.ID != "new" || st.ID != "do" {
		t.Fatalf("new item must be the current step, got %s/%s", it.ID, st.ID)
	}
	mid, err := insertItem(base, "x", "t", "b", "")
	if err != nil {
		t.Fatal(err)
	}
	if mid.Items[1].ID != "x" {
		t.Fatalf("insert -before b should land at index 1, got %s", mid.Items[1].ID)
	}
	if _, err := insertItem(base, "a", "t", "", ""); err == nil {
		t.Fatal("duplicate id must be rejected")
	}
	if _, err := insertItem(base, "z", "t", "nope", ""); err == nil {
		t.Fatal("unknown -before must be rejected")
	}
}
