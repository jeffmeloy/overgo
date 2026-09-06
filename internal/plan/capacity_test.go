package plan

import (
	"strings"
	"testing"
)

// TestSlotCostAndLabelAssignment pins: a row is charged on the first lane
// holding every required label with free capacity for its cost; charges
// accumulate so a second device row waits on capacity; a required label no
// lane holds leaves the row waiting with that reason; reservations charged
// before the pass count; invalid costs, labels and lanes are refused; the
// report names each verdict.
func TestSlotCostAndLabelAssignment(t *testing.T) {
	lanes := []LaneCapacity{
		{Lane: "gpu", Labels: []string{"cuda"}, Capacity: SlotCost{Device: 1, Host: 4}},
		{Lane: "host", Capacity: SlotCost{Host: 8}},
	}
	demands := []SlotDemand{
		{Row: Ref{Item: "train", Step: "run"}, Cost: SlotCost{Device: 1, Host: 1}, Labels: []LabelRequirement{{Label: "cuda", Required: true}}},
		{Row: Ref{Item: "eval", Step: "run"}, Cost: SlotCost{Device: 1}, Labels: []LabelRequirement{{Label: "cuda", Required: true}}},
		{Row: Ref{Item: "docs", Step: "build"}, Cost: SlotCost{Host: 2}},
		{Row: Ref{Item: "tpu", Step: "run"}, Cost: SlotCost{Device: 1}, Labels: []LabelRequirement{{Label: "tpu", Required: true}}},
		{Row: Ref{Item: "idle", Step: "noop"}},
	}
	assignments, err := AssignSlots(lanes, nil, demands)
	if err != nil {
		t.Fatal(err)
	}
	want := []SlotAssignment{
		{Row: demands[0].Row, Lane: "gpu"},
		{Row: demands[1].Row, Waits: "cost device=1 host=0 exceeds the free capacity of every labelled lane"},
		{Row: demands[2].Row, Lane: "gpu"},
		{Row: demands[3].Row, Waits: "required label tpu is held by no lane"},
		{Row: demands[4].Row, Lane: "gpu"},
	}
	for index := range want {
		if assignments[index] != want[index] {
			t.Fatalf("assignment %d = %+v, want %+v", index, assignments[index], want[index])
		}
	}
	report := FormatSlotAssignments(assignments)
	if !strings.Contains(report, "lane: train/run -> gpu") || !strings.Contains(report, "lane: eval/run waits: cost device=1") {
		t.Fatalf("report:\n%s", report)
	}

	// Units already charged on the gpu lane push the host-only row to the host lane.
	charged, err := AssignSlots(lanes, map[string]SlotCost{"gpu": {Device: 1, Host: 3}}, demands[2:3])
	if err != nil || charged[0].Lane != "host" {
		t.Fatalf("charged assignment = %+v, %v", charged, err)
	}
	for name, invalid := range map[string]func() error{
		"negative cost": func() error {
			_, err := AssignSlots(lanes, nil, []SlotDemand{{Row: Ref{Item: "a", Step: "b"}, Cost: SlotCost{Device: -1}}})
			return err
		},
		"duplicate label": func() error {
			_, err := AssignSlots(lanes, nil, []SlotDemand{{Row: Ref{Item: "a", Step: "b"}, Labels: []LabelRequirement{{Label: "cuda"}, {Label: "cuda"}}}})
			return err
		},
		"duplicate lane": func() error {
			_, err := AssignSlots([]LaneCapacity{{Lane: "gpu"}, {Lane: "gpu"}}, nil, nil)
			return err
		},
		"upper-case label": func() error {
			_, err := AssignSlots([]LaneCapacity{{Lane: "gpu", Labels: []string{"CUDA"}}}, nil, nil)
			return err
		},
	} {
		if invalid() == nil {
			t.Fatalf("%s was accepted", name)
		}
	}
}
