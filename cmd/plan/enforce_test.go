package main

import (
	"testing"

	"overgo/internal/plan"
)

// TestEnforceAddInsertsTask pins the mechanical task-injection: -add creates a
// top-priority (or -before) open item with a "do" step + verify, and the new item
// becomes the current step -- no hand-editing plan.json.
func TestEnforceAddInsertsTask(t *testing.T) {
	base := plan.Plan{Items: []plan.Item{
		{ID: "a", Status: "open", Steps: []plan.Step{{ID: "s", Status: "open", Verify: "go test ./..."}}},
		{ID: "b", Status: "open", Steps: []plan.Step{{ID: "s", Status: "open", Verify: "go test ./..."}}},
	}}
	top, err := insertItem(base, "new", "do the thing", "", "go test ./...")
	if err != nil {
		t.Fatal(err)
	}
	if top.Items[0].ID != "new" || top.Items[0].Steps[0].ID != "do" || top.Items[0].Steps[0].Verify != "go test ./..." {
		t.Fatalf("insert-at-top produced %+v", top.Items[0])
	}
	authority := mustTestCompletionAuthority(t, top)
	if it, st, ok := plan.Current(top, plan.UnassignedRole, authority); !ok || it.ID != "new" || st.ID != "do" {
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

// TestPrunedDependencyRequiresGatedCompletion pins the CLI selector as a
// consumer of the same fail-closed authority as the plan package.
func TestPrunedDependencyRequiresGatedCompletion(t *testing.T) {
	document := plan.Plan{Items: []plan.Item{{
		ID: "dependent", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Status: plan.StatusOpen, Verify: "go test ./...",
			DependsOn: []string{"missing/do"},
		}},
	}}}
	if _, err := testCompletionAuthority(t, document); err == nil {
		t.Fatal("unknown pruned dependency produced completion authority")
	}
	if action, open := nextAction(document, plan.UnassignedRole, plan.CompletionAuthority{}); open {
		t.Fatalf("unknown pruned dependency dispatched as %q", action)
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
