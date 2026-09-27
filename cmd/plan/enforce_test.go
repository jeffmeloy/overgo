package main

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/worklease"
)

// TestEnforceAddInsertsTask pins the mechanical task-injection: -add creates a
// top-priority (or -before) open item with a "do" step + verify, and the new item
// becomes the current step -- no hand-editing plan.json.
func TestEnforceAddInsertsTask(t *testing.T) {
	t.Parallel()
	base := plan.Plan{Items: []plan.Item{
		{ID: "a", Status: "open", Steps: []plan.Step{{ID: "s", Status: "open", Verify: "go test ./..."}}},
		{ID: "b", Status: "open", Steps: []plan.Step{{ID: "s", Status: "open", Verify: "go test ./..."}}},
	}}
	top, err := insertItem(base, "new", "do the thing", "", "", "go test ./...")
	if err != nil {
		t.Fatal(err)
	}
	if top.Items[0].ID != "new" || top.Items[0].Steps[0].ID != "do" || top.Items[0].Steps[0].Verify != "go test ./..." {
		t.Fatalf("insert-at-top produced %+v", top.Items[0])
	}
	if it, st, ok := plan.Current(top, worklease.UnassignedRole); !ok || it.ID != "new" || st.ID != "do" {
		t.Fatalf("new item must be the current step, got %s/%s", it.ID, st.ID)
	}
	mid, err := insertItem(base, "x", "t", "", "b", "")
	if err != nil {
		t.Fatal(err)
	}
	if mid.Items[1].ID != "x" {
		t.Fatalf("insert -before b should land at index 1, got %s", mid.Items[1].ID)
	}
	if _, err := insertItem(base, "a", "t", "", "", ""); err == nil {
		t.Fatal("duplicate id must be rejected")
	}
	if _, err := insertItem(base, "z", "t", "", "nope", ""); err == nil {
		t.Fatal("unknown -before must be rejected")
	}
}

// TestAddOwnsCloseoutDependency holds -add to the structure policy: the closeout
// waits for the new row and what it already waited for, the caller's plan is
// untouched, and a plan with no closeout still takes a row or refuses -before.
func TestAddOwnsCloseoutDependency(t *testing.T) {
	t.Parallel()
	waits := []plan.Step{{ID: "do", Status: "open", DependsOn: []string{"earlier/do"}}}
	base := plan.Plan{Items: []plan.Item{{ID: "earlier", Status: "open"}, {ID: "campaign-closeout", Status: "open", Steps: waits}}}
	added, err := insertItem(base, "new-row", "title", "", "campaign-closeout", "go test ./...")
	if err != nil || !slices.Equal(added.Items[2].Steps[0].DependsOn, []string{"earlier/do", "new-row/do"}) {
		t.Fatalf("closeout after adding a row = %+v, %v", added.Items, err)
	}
	if len(base.Items) != 2 || !slices.Equal(waits[0].DependsOn, []string{"earlier/do"}) {
		t.Fatalf("adding a row rewrote the caller's plan: %+v", base.Items)
	}
	grown, err := insertItem(plan.Plan{}, "lone", "title", "", "", "")
	if _, refused := insertItem(plan.Plan{}, "lone", "title", "", "absent", ""); err != nil || len(grown.Items) != 1 || refused == nil {
		t.Fatalf("an empty plan took a row: %v; refused -before an absent row: %v", err, refused)
	}
}

// TestRelocateItemReranksWithoutLoss pins the mechanical re-rank: -move
// takes an existing item to the top or before another, keeps every
// other item in order, and refuses unknown ids.
func TestRelocateItemReranksWithoutLoss(t *testing.T) {
	t.Parallel()
	step := []plan.Step{{ID: "s", Status: "open", Verify: "go test ./..."}}
	base := plan.Plan{Items: []plan.Item{
		{ID: "a", Status: "open", Steps: step}, {ID: "b", Status: "open", Steps: step}, {ID: "c", Status: "open", Steps: step},
	}}
	order := func(document plan.Plan) string {
		var ids strings.Builder
		for _, it := range document.Items {
			ids.WriteString(it.ID)
		}
		return ids.String()
	}
	top, err := relocateItem(base, "c", "")
	if err != nil || order(top) != "cab" {
		t.Fatalf("move to top = %s, %v", order(top), err)
	}
	before, err := relocateItem(base, "a", "c")
	if err != nil || order(before) != "bac" {
		t.Fatalf("move before c = %s, %v", order(before), err)
	}
	same, err := relocateItem(base, "b", "b")
	if err != nil || order(same) != "abc" {
		t.Fatalf("move before itself = %s, %v", order(same), err)
	}
	if _, err := relocateItem(base, "zz", ""); err == nil {
		t.Fatal("unknown item must be rejected")
	}
	if _, err := relocateItem(base, "a", "nope"); err == nil {
		t.Fatal("unknown -before must be rejected")
	}
	// -retitle re-scopes the item and its single "do" step together, and
	// leaves a multi-step item's steps as they are.
	single := plan.Plan{Items: []plan.Item{
		{ID: "one", Title: "old", Status: "open", Steps: []plan.Step{{ID: "do", Title: "old", Status: "open"}}},
		{ID: "two", Title: "old", Status: "open", Steps: []plan.Step{{ID: "x", Title: "x", Status: "open"}, {ID: "y", Title: "y", Status: "open"}}},
	}}
	scoped, err := rescopeItem(single, "one", " new scope ")
	if err != nil || scoped.Items[0].Title != "new scope" || scoped.Items[0].Steps[0].Title != "new scope" {
		t.Fatalf("retitle single-step item = %+v, %v", scoped.Items[0], err)
	}
	scoped, err = rescopeItem(single, "two", "new scope")
	if err != nil || scoped.Items[1].Title != "new scope" || scoped.Items[1].Steps[0].Title != "x" {
		t.Fatalf("retitle multi-step item = %+v, %v", scoped.Items[1], err)
	}
	if _, err := rescopeItem(single, "one", "  "); err == nil {
		t.Fatal("empty title must be rejected")
	}
	if _, err := rescopeItem(single, "zz", "t"); err == nil {
		t.Fatal("unknown item must be rejected")
	}
}

// TestEnforceSetVerify pins the pure verify-assignment: it replaces the named
// step's verify (closing the hand-edit-plan.json gap) and rejects an absent
// item or step.
func TestEnforceSetVerify(t *testing.T) {
	t.Parallel()
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
	t.Parallel()
	for _, r := range []string{"user-stop", "user-stop: they said wait", "irreversible: needs confirm", "external-prereq: model missing"} {
		if _, _, err := plan.ParseStopReason(r); err != nil {
			t.Fatalf("valid stop %q rejected: %v", r, err)
		}
	}
	for _, r := range []string{"", "checkpoint", "should I continue?", "done for now", "milestone", "picking up fresh"} {
		if _, _, err := plan.ParseStopReason(r); err == nil {
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
		if _, _, err := plan.ParseStopReason(r); err == nil {
			t.Fatalf("self-pacing stop %q must be REFUSED", r)
		}
	}
	for _, r := range []string{
		"external-prereq: gate running in background (bq7kh24rf)",
		"external-prereq: owner must provision the candidate principal",
		"external-prereq: tightening-lane merge pending",
	} {
		if _, _, err := plan.ParseStopReason(r); err != nil {
			t.Fatalf("genuinely external stop %q rejected: %v", r, err)
		}
	}
}
