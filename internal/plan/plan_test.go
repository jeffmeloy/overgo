package plan

import "testing"

// TestEnforceCurrentFirstOpenStep pins the shared dispatch rule.
func TestEnforceCurrentFirstOpenStep(t *testing.T) {
	p := Plan{Items: []Item{
		{ID: "b", Status: "open", Steps: []Step{
			{ID: "s2", Status: "open"},
		}},
		{ID: "c", Status: "open", Steps: []Step{{ID: "s1", Status: "open"}}},
	}}
	it, st, ok := Current(p)
	if !ok || it.ID != "b" || st.ID != "s2" {
		t.Fatalf("Current = %s/%s ok=%v, want b/s2 ok=true", it.ID, st.ID, ok)
	}

	if _, _, ok := Current(Plan{}); ok {
		t.Fatal("Current on an empty plan must return ok=false")
	}

	// An open item with no open step is itself the action (sentinel step ".").
	openNoStep := Plan{Items: []Item{{ID: "x", Status: "open"}}}
	if it, st, ok := Current(openNoStep); !ok || it.ID != "x" || st.ID != "." {
		t.Fatalf("Current(open item, no steps) = %s/%s ok=%v, want x/. ok=true", it.ID, st.ID, ok)
	}
}

func TestPlanContainsOpenWorkOnly(t *testing.T) {
	valid := Plan{Items: []Item{
		{ID: "open", Status: "open", Steps: []Step{{ID: "work", Status: "open", Verify: "go test ./..."}}},
		{ID: "blocked", Status: "blocked-external-prereq", Steps: []Step{{ID: "wait", Status: "blocked-external-prereq"}}},
	}}
	if err := ValidateOpenWork(valid); err != nil {
		t.Fatal(err)
	}
	legacy := Plan{Items: []Item{
		{ID: "done", Status: "done", Steps: []Step{{ID: "old", Status: "done"}}},
		{ID: "mixed", Status: "open", Steps: []Step{{ID: "old", Status: "done"}, {ID: "next", Status: "partial", Verify: "go test ./..."}}},
	}}
	if err := ValidateOpenWork(legacy); err == nil {
		t.Fatal("completion ledger passed live-plan validation")
	}
	compacted := Compact(legacy)
	if err := ValidateOpenWork(compacted); err != nil {
		t.Fatal(err)
	}
	if len(compacted.Items) != 1 || len(compacted.Items[0].Steps) != 1 || compacted.Items[0].Steps[0].Status != "open" {
		t.Fatalf("compacted plan = %+v", compacted)
	}
}

func TestOpenStepRequiresVerifier(t *testing.T) {
	document := Plan{Items: []Item{{
		ID: "item", Status: "open", Steps: []Step{{ID: "work", Status: "open"}},
	}}}
	if err := ValidateOpenWork(document); err == nil {
		t.Fatal("open step without a verifier passed validation")
	}
	document.Items[0].Steps[0].Status = "blocked-external-prereq"
	if err := ValidateOpenWork(document); err != nil {
		t.Fatalf("blocked step should name its blocker without a runnable verifier: %v", err)
	}
}

func TestCompactPropagatesBlockedStatus(t *testing.T) {
	document := Plan{Items: []Item{{
		ID: "roadmap", Status: "open", Steps: []Step{{ID: "license", Status: "blocked-external-prereq"}},
	}}}
	compacted := Compact(document)
	if compacted.Items[0].Status != "blocked-external-prereq" {
		t.Fatalf("blocked-only item status = %q", compacted.Items[0].Status)
	}
	if item, _, ok := Current(compacted); ok && item.ID == "roadmap" {
		t.Fatal("blocked-only item remained dispatchable")
	}
}
