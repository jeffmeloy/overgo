package main

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestFrontierReportsLaneAssignments pins: when the plan declares lanes,
// the frontier report charges every ready row's declared cost on a lane
// or names why it waits, in frontier order.
func TestFrontierReportsLaneAssignments(t *testing.T) {
	document := plan.Plan{
		Lanes: []plan.LaneCapacity{
			{Lane: "gpu", Labels: []string{"cuda"}, Capacity: plan.SlotCost{Device: 1, Host: 2}},
			{Lane: "host", Capacity: plan.SlotCost{Host: 4}},
		},
		Items: []plan.Item{
			{ID: "train", Status: plan.StatusOpen, Steps: []plan.Step{{
				ID: "run", Status: plan.StatusOpen, Verify: "go test ./train",
				SlotCost: &plan.SlotCost{Device: 1, Host: 1}, Labels: []plan.LabelRequirement{{Label: "cuda", Required: true}},
			}}},
			{ID: "eval", Status: plan.StatusOpen, Steps: []plan.Step{{
				ID: "run", Status: plan.StatusOpen, Verify: "go test ./eval",
				SlotCost: &plan.SlotCost{Device: 1}, Labels: []plan.LabelRequirement{{Label: "cuda", Required: true}},
			}}},
			{ID: "docs", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "build", Status: plan.StatusOpen, Verify: "go test ./docs", SlotCost: &plan.SlotCost{Host: 3}}}},
		},
	}
	root := initializePlanTestRepository(t, document)
	var output bytes.Buffer
	if err := printReadyFrontier(root, document, &output); err != nil {
		t.Fatal(err)
	}
	report := output.String()
	for _, want := range []string{
		"lane: train/run -> gpu",
		"lane: eval/run waits: cost device=1 host=0 exceeds the free capacity of every labelled lane",
		"lane: docs/build -> host",
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("frontier report lacks %q:\n%s", want, report)
		}
	}
}
