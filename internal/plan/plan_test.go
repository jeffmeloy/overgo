package plan

import "testing"

// TestEnforceCurrentFirstOpenStep pins the queue rule shared by cmd/plan and
// cmd/gate: the first step of the first item, and ok=false for an empty queue.
func TestEnforceCurrentFirstOpenStep(t *testing.T) {
	p := Plan{Items: []Item{
		{ID: "b", Status: "open", Steps: []Step{{ID: "s2", Status: "open"}}},
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
	valid := Plan{Items: []Item{{
		ID: "item", Status: "open",
		Steps: []Step{{ID: "step", Status: "open"}},
	}}}
	if err := ValidateOpenOnly(valid); err != nil {
		t.Fatalf("open plan rejected: %v", err)
	}
	for name, invalid := range map[string]Plan{
		"completed item": {Items: []Item{{ID: "item", Status: "done"}}},
		"blocked item":   {Items: []Item{{ID: "item", Status: "blocked-external-prereq"}}},
		"completed step": {Items: []Item{{ID: "item", Status: "open", Steps: []Step{{ID: "step", Status: "done"}}}}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateOpenOnly(invalid); err == nil {
				t.Fatal("historical row accepted")
			}
		})
	}
}
