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
		{ID: "open", Status: "open", Steps: []Step{{ID: "work", Status: "open"}}},
		{ID: "blocked", Status: "blocked-external-prereq", Steps: []Step{{ID: "wait", Status: "blocked-external-prereq"}}},
	}}
	if err := ValidateOpenWork(valid); err != nil {
		t.Fatal(err)
	}
	legacy := Plan{Items: []Item{
		{ID: "done", Status: "done", Steps: []Step{{ID: "old", Status: "done"}}},
		{ID: "mixed", Status: "open", Steps: []Step{{ID: "old", Status: "done"}, {ID: "next", Status: "partial"}}},
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
