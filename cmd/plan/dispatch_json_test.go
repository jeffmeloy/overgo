package main

import (
	"encoding/json"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
	"overgo/internal/worklease"
)

// TestDispatchJSONMatchesProse pins the dispatch as data: the JSON the plan
// tool prints for -next -json names the row the prose line names, carries
// the step's verify, and a complete plan is one field, not a sentence to
// parse.
func TestDispatchJSONMatchesProse(t *testing.T) {
	t.Parallel()
	document := plan.Plan{Items: []plan.Item{{
		ID: "item", Title: "item title", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Title: "make the bounded change", Status: plan.StatusOpen, Verify: "go test ./x -run '^TestX$'",
		}},
	}}}
	dispatch := plan.DispatchOf(document, worklease.UnassignedRole)
	encoded, err := json.Marshal(dispatch)
	testcheck.NoError(t, err)
	var decoded plan.Dispatch
	testcheck.NoError(t, json.Unmarshal(encoded, &decoded))
	prose, open := nextAction(document, worklease.UnassignedRole)
	if !open || decoded.Complete || decoded.Line != prose || decoded.Item != "item" || decoded.Step != "do" || decoded.Verify != "go test ./x -run '^TestX$'" {
		t.Fatalf("dispatch %s does not match the prose %q", encoded, prose)
	}
	done := plan.DispatchOf(plan.Plan{}, worklease.UnassignedRole)
	if !done.Complete || done.Item != "" || done.Line != "plan complete: every item is done" {
		t.Fatalf("complete dispatch = %+v", done)
	}
}
