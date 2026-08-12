package plan

import "testing"

// TestEnforceCurrentFirstOpenStep pins the "current step" rule that both cmd/plan
// and cmd/gate depend on: the first open step of the first open item, and ok=false
// only when every item is done.
func TestEnforceCurrentFirstOpenStep(t *testing.T) {
	p := Plan{Items: []Item{
		{ID: "a", Status: "done", Steps: []Step{{ID: "s1", Status: "done"}}},
		{ID: "b", Status: "open", Steps: []Step{
			{ID: "s1", Status: "done"},
			{ID: "s2", Status: "open"},
		}},
		{ID: "c", Status: "open", Steps: []Step{{ID: "s1", Status: "open"}}},
	}}
	it, st, ok := Current(p)
	if !ok || it.ID != "b" || st.ID != "s2" {
		t.Fatalf("Current = %s/%s ok=%v, want b/s2 ok=true", it.ID, st.ID, ok)
	}

	done := Plan{Items: []Item{{ID: "a", Status: "done", Steps: []Step{{ID: "s1", Status: "done"}}}}}
	if _, _, ok := Current(done); ok {
		t.Fatal("Current on an all-done plan must return ok=false")
	}

	// An open item with no open step is itself the action (sentinel step ".").
	openNoStep := Plan{Items: []Item{{ID: "x", Status: "open"}}}
	if it, st, ok := Current(openNoStep); !ok || it.ID != "x" || st.ID != "." {
		t.Fatalf("Current(open item, no steps) = %s/%s ok=%v, want x/. ok=true", it.ID, st.ID, ok)
	}
}
