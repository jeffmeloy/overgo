package main

import (
	"os"
	"path/filepath"
	"strings"
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

func TestTrainingPlanMatchesImplementation(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(filepath.FromSlash(path))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	document := read("../../docs/training_plan.md")
	production := read("../../cmd/train/run_cuda_windows.go")
	resident := read("../../internal/densecausal/train_device_resident_cuda_windows.go")
	backward := read("../../internal/densecausal/device_backward_cuda_windows.go")

	if !strings.Contains(production, ".TrainDeviceFull(") || strings.Contains(production, ".TrainDeviceResident(") {
		t.Fatal("production training reachability changed; update training_plan.md")
	}
	if !strings.Contains(resident, "func (m *Model) TrainDeviceResident(") || !strings.Contains(backward, "func (m *Model) deviceLayerBackward(") {
		t.Fatal("documented resident/backward implementation is absent")
	}
	for _, fact := range []string{
		"Device backward has landed",
		"`cmd/train` still calls `TrainDeviceFull`",
		"`TrainDeviceResident` is not yet production-reachable",
		"Exact checkpoint/resume is not implemented",
		"Compiled multimodal training authority is not implemented",
		"No real checkpoint-backed model has completed",
	} {
		if !strings.Contains(document, fact) {
			t.Errorf("training plan omits current fact %q", fact)
		}
	}
	for _, stale := range []string{"backward is host-only today", "already resident-trainable", "open, done or blocked"} {
		if strings.Contains(document, stale) {
			t.Errorf("training plan retains stale claim %q", stale)
		}
	}
}

func TestPlanContainsOpenWorkOnly(t *testing.T) {
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
			{ID: "first", Status: "open"},
			{ID: "second", Status: "open"},
		},
	}}}
	if err := advanceStep(document, "item", "first", "test-verified"); err != nil {
		t.Fatal(err)
	}
	saved, err := plan.Load("")
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.ValidateOpenWork(saved); err != nil {
		t.Fatal(err)
	}
	if len(saved.Items) != 1 || len(saved.Items[0].Steps) != 1 || saved.Items[0].Steps[0].ID != "second" {
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
}
