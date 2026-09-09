package main

import (
	"testing"

	"overgo/internal/plan"
)

// TestAssignOwnerAndLane pins the pure cores: an owner lands on the named
// open item only, a lane lands on the plan, and blanks or unknown ids are
// refused.
func TestAssignOwnerAndLane(t *testing.T) {
	document := plan.Plan{Items: []plan.Item{
		{ID: "one", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "true"}}},
		{ID: "two", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "true"}}},
	}}
	owned, err := assignOwner(document, "two", " master ")
	if err != nil || owned.Items[1].Owner != "master" || owned.Items[0].Owner != "" {
		t.Fatalf("assign = %+v, %v", owned.Items, err)
	}
	if _, err := assignOwner(document, "zz", "master"); err == nil {
		t.Fatal("unknown item assigned")
	}
	if _, err := assignOwner(document, "one", " "); err == nil {
		t.Fatal("blank owner assigned")
	}
	laned, err := setPlanLane(document, " gui ")
	if err != nil || laned.Lane != "gui" {
		t.Fatalf("lane = %q, %v", laned.Lane, err)
	}
	if _, err := setPlanLane(document, ""); err == nil {
		t.Fatal("blank lane set")
	}
}
