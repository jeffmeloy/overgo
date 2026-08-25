package main

import (
	"os"
	"path/filepath"
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

// TestAdvanceRemovesCompletedStep pins the plan contract at the CLI:
// an advanced step leaves the saved plan, which then holds only the
// remaining open work.
func TestAdvanceRemovesCompletedStep(t *testing.T) {
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })
	document := plan.Plan{Items: []plan.Item{{
		ID: "item", Status: "open", Steps: []plan.Step{
			{ID: "first", Status: "open", Verify: "go test ./..."},
			{ID: "second", Status: "open", Verify: "go test ./..."},
		},
	}}}
	if err := advanceStep(document, "item", "first", "test-verified", plan.UnassignedRole, nil); err != nil {
		t.Fatal(err)
	}
	saved, err := plan.Load("")
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.Validate(saved); err != nil {
		t.Fatal(err)
	}
	if len(saved.Items) != 1 || len(saved.Items[0].Steps) != 1 ||
		saved.Items[0].Steps[0].ID != "second" {
		t.Fatalf("advanced plan = %+v", saved)
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
	if it, st, ok := plan.Current(top, plan.UnassignedRole); !ok || it.ID != "new" || st.ID != "do" {
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

// TestEnforceSetVerify pins the pure verify-assignment: it replaces the named
// step's verify (closing the hand-edit-plan.json gap) and rejects an absent
// item or step.
func TestEnforceSetVerify(t *testing.T) {
	base := plan.Plan{Items: []plan.Item{
		{ID: "a", Status: "open", Steps: []plan.Step{{ID: "s1", Status: "open"}, {ID: "s2", Status: "open"}}},
	}}
	got, err := assignVerify(base, "a", "s2", "  go test ./x  ")
	if err != nil {
		t.Fatal(err)
	}
	if got.Items[0].Steps[1].Verify != "go test ./x" {
		t.Fatalf("verify not set/trimmed: %q", got.Items[0].Steps[1].Verify)
	}
	if got.Items[0].Steps[0].Verify != "" {
		t.Fatal("sibling step's verify must be untouched")
	}
	if _, err := assignVerify(base, "a", "nope", "x"); err == nil {
		t.Fatal("unknown step must be rejected")
	}
	if _, err := assignVerify(base, "nope", "s1", "x"); err == nil {
		t.Fatal("unknown item must be rejected")
	}
}

// TestEnforceStopReason pins that only the three legitimate stop reasons are
// accepted -- a manufactured "checkpoint"/"should I continue?" is refused, so the
// stop-gate can tell a real stop from an invented one.
func TestEnforceStopReason(t *testing.T) {
	for _, r := range []string{"user-stop", "user-stop: they said wait", "irreversible: needs confirm", "external-prereq: model missing"} {
		if err := validateStop(r); err != nil {
			t.Fatalf("valid stop %q rejected: %v", r, err)
		}
	}
	for _, r := range []string{"", "checkpoint", "should I continue?", "done for now", "milestone", "picking up fresh"} {
		if err := validateStop(r); err == nil {
			t.Fatalf("manufactured stop %q must be REFUSED", r)
		}
	}
	// Self-pacing dressed as an external prerequisite is the recorded drift
	// mode (2026-08-18: a stage deferred as "best started with fresh
	// context"); the category demands something verifiably external.
	for _, r := range []string{
		"external-prereq: stage 3 best started with fresh context",
		"external-prereq: continuing next session",
		"external-prereq: nothing in particular",
		"external-prereq: taking a break at this milestone",
	} {
		if err := validateStop(r); err == nil {
			t.Fatalf("self-pacing stop %q must be REFUSED", r)
		}
	}
	for _, r := range []string{
		"external-prereq: gate running in background (bq7kh24rf)",
		"external-prereq: owner must provision the candidate principal",
		"external-prereq: tightening-lane merge pending",
	} {
		if err := validateStop(r); err != nil {
			t.Fatalf("genuinely external stop %q rejected: %v", r, err)
		}
	}
}
