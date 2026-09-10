package main

import (
	"encoding/json"
	"testing"

	"overgo/internal/plan"
)

// TestDispatchJSONMatchesProse pins the dispatch as data: the JSON the plan
// tool prints for -next -json names the row the prose line names, carries
// the step's verify, and a complete plan is one field, not a sentence to
// parse.
func TestDispatchJSONMatchesProse(t *testing.T) {
	document := plan.Plan{Items: []plan.Item{{
		ID: "item", Title: "item title", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Title: "make the bounded change", Status: plan.StatusOpen, Verify: "go test ./x -run '^TestX$'",
		}},
	}}}
	authority := mustTestCompletionAuthority(t, document)
	dispatch := plan.DispatchOf(document, plan.UnassignedRole, authority)
	encoded, err := json.Marshal(dispatch)
	if err != nil {
		t.Fatal(err)
	}
	var decoded plan.Dispatch
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	prose, open := nextAction(document, plan.UnassignedRole, authority)
	if !open || decoded.Complete || decoded.Line != prose || decoded.Item != "item" || decoded.Step != "do" || decoded.Verify != "go test ./x -run '^TestX$'" {
		t.Fatalf("dispatch %s does not match the prose %q", encoded, prose)
	}
	done := plan.DispatchOf(plan.Plan{}, plan.UnassignedRole, mustTestCompletionAuthority(t, plan.Plan{}))
	if !done.Complete || done.Item != "" || done.Line != "plan complete: every item is done" {
		t.Fatalf("complete dispatch = %+v", done)
	}
}
