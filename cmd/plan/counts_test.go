package main

import (
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestOnlyAConsolidationTakesACount holds plan's edits to counts on consolidations alone (consolidation-counts/do): a
// feature row's count asked nothing of a landing, which holds only a consolidation to its count, and since 10 Nov 59
// of 111 landings added more than their row's. A step added listing its files, and a row added so, carry no count;
// -setcount refuses a feature row and sets a consolidation's.
func TestOnlyAConsolidationTakesACount(t *testing.T) {
	t.Parallel()
	held := plan.Plan{Items: []plan.Item{
		{ID: "row", Title: "Row", Steps: []plan.Step{openStep("first")}},
		{ID: "shrink", Title: "Shrink", Class: plan.ClassConsolidation, Steps: []plan.Step{openStep("do")}},
	}}
	root, _ := editFixture(t, held)
	if output, err := planCommand(t, root, "-addstep", "-title", "Second", "-vcmd", "true", "-files", "internal/x", "-reason", "The second step.", "row", "second"); err != nil {
		t.Fatalf("plan -addstep: %v\n%s", err, output)
	}
	if output, err := planCommand(t, root, "-add", "-title", "Fresh", "-vcmd", "true", "-files", "cmd/z", "fresh"); err != nil {
		t.Fatalf("plan -add: %v\n%s", err, output)
	}
	edited := editedPlan(t, root)
	if fresh, row := edited.Items[0], edited.Items[1]; fresh.Budget != (plan.Budget{}) || row.Budget != (plan.Budget{}) || fresh.Steps[0].Files[0] != "cmd/z" {
		t.Fatalf("rows filed with files carry counts %+v and %+v; want none", fresh.Budget, row.Budget)
	}
	if output, err := planCommand(t, root, "-setcount", "-nodes", "40", "-reason", "It grows.", "row"); err == nil || !strings.Contains(output+err.Error(), "only a consolidation carries a count") {
		t.Fatalf("plan -setcount on a feature row gave %v\n%s; want it refused", err, output)
	}
	if output, err := planCommand(t, root, "-setcount", "-nodes", "-40", "-reason", "It must shrink.", "shrink"); err != nil {
		t.Fatalf("plan -setcount on a consolidation: %v\n%s", err, output)
	}
	if shrink := editedPlan(t, root).Items[2].Budget; shrink != (plan.Budget{Nodes: -40, Reason: "It must shrink."}) {
		t.Fatalf("the consolidation's count is %+v, want -40 and its reason", shrink)
	}
}
