package plan

import "testing"

// TestEnforceCurrentFirstOpenStep pins the shared dispatch rule.
func TestEnforceCurrentFirstOpenStep(t *testing.T) {
	p := Plan{Items: []Item{
		{ID: "a", Status: StatusDone, Steps: []Step{{ID: "s1", Status: StatusDone}}},
		{ID: "b", Status: "open", Steps: []Step{
			{ID: "s1", Status: StatusDone},
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

func TestPlanRetainsCompletionState(t *testing.T) {
	valid := Plan{Items: []Item{
		{ID: "open", Status: "open", Steps: []Step{{ID: "work", Status: "open", Verify: "go test ./..."}}},
		{ID: "blocked", Status: "blocked-external-prereq", Steps: []Step{{ID: "wait", Status: "blocked-external-prereq"}}},
		{ID: "done", Status: StatusDone, Steps: []Step{{ID: "old", Status: StatusDone, Verify: "go test ./..."}}},
	}}
	if err := Validate(valid); err != nil {
		t.Fatal(err)
	}
	invalid := Plan{Items: []Item{
		{ID: "mixed", Status: StatusDone, Steps: []Step{{ID: "next", Status: StatusOpen, Verify: "go test ./..."}}},
	}}
	if err := Validate(invalid); err == nil {
		t.Fatal("done item with open work passed validation")
	}
}

func TestOpenStepRequiresVerifier(t *testing.T) {
	document := Plan{Items: []Item{{
		ID: "item", Status: "open", Steps: []Step{{ID: "work", Status: "open"}},
	}}}
	if err := Validate(document); err == nil {
		t.Fatal("open step without a verifier passed validation")
	}
	document.Items[0].Steps[0].Status = "blocked-external-prereq"
	if err := Validate(document); err != nil {
		t.Fatalf("blocked step should name its blocker without a runnable verifier: %v", err)
	}
}

func TestAdvanceMarksRowsDone(t *testing.T) {
	document := Plan{Items: []Item{{
		ID: "item", Status: StatusOpen, Steps: []Step{
			{ID: "first", Status: StatusOpen, Verify: "go test ./..."},
			{ID: "second", Status: StatusOpen, Verify: "go test ./..."},
		},
	}}}
	advanced, err := Advance(document, "item", "first")
	if err != nil {
		t.Fatal(err)
	}
	if len(advanced.Items[0].Steps) != len(document.Items[0].Steps) || advanced.Items[0].Steps[0].Status != StatusDone {
		t.Fatalf("first advance removed history: %+v", advanced.Items[0])
	}
	if _, step, ok := Current(advanced); !ok || step.ID != "second" {
		t.Fatalf("current after first advance = %s, open=%v", step.ID, ok)
	}
	advanced, err = Advance(advanced, "item", "second")
	if err != nil {
		t.Fatal(err)
	}
	if advanced.Items[0].Status != StatusDone || advanced.Items[0].Steps[1].Status != StatusDone {
		t.Fatalf("final advance = %+v", advanced.Items[0])
	}
	if _, _, ok := Current(advanced); ok {
		t.Fatal("completed plan remained dispatchable")
	}
}
