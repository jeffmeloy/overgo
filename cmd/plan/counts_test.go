package main

import (
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestANewStepsCountComesFromItsDirectoriesHistory holds plan -addstep and -add to the history of the directories a
// new step lists: on 2026-11-08 the lead raised all 24 media rows about threefold by hand, since counts filed by hand
// ran far under what landed. In a repository whose landings in internal/x counted 100, 300 and 500 nodes, a step added
// listing internal/x adds the median, 300, to its row's count, with the session's figure written beside it, and a row
// added listing a directory no landing changed is counted by the session's figure.
func TestANewStepsCountComesFromItsDirectoriesHistory(t *testing.T) {
	t.Parallel()
	held := plan.Plan{Items: []plan.Item{{ID: "row", Title: "Row", Status: plan.StatusOpen, Budget: plan.Budget{Nodes: 50, Reason: "Filed."}, Steps: []plan.Step{openStep("first")}}}}
	root, _ := editFixture(t, held)
	git := func(arguments ...string) {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", append([]string{"-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid"}, arguments...)...)
		command.Dir = root
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v %s", arguments, err, output)
		}
	}
	git("init", "-q")
	for _, landed := range []struct{ nodes, path string }{{"+100", "internal/x/a.go"}, {"+500", "internal/x/b.go"}, {"+300", "internal/x/c.go"}, {"+9000", "internal/y/y.go"}} {
		git("commit", "-q", "--allow-empty", "-m", "row/do: a change\n\nChange:\n- files: "+landed.path+"\n\nMeasured by the gate: rules 1s (production "+landed.nodes+" nodes of the 700 its row may add)")
	}
	output, err := planCommand(t, root, "-addstep", "-title", "Second", "-vcmd", "true", "-files", "internal/x", "-nodes", "120", "-reason", "The second step.", "row", "second")
	if err != nil {
		t.Fatalf("plan -addstep: %v\n%s", err, output)
	}
	row := editedPlan(t, root).Items[0]
	if row.Budget.Nodes != 350 || !strings.Contains(row.Budget.Reason, "Step second: +300 (300, the median of what 3 landings that changed internal/x counted; the session's figure 120)") {
		t.Fatalf("the row counts %d nodes for %q, want 50 and the median 300", row.Budget.Nodes, row.Budget.Reason)
	}
	if output, err := planCommand(t, root, "-add", "-title", "Fresh", "-vcmd", "true", "-files", "cmd/z", "-nodes", "120", "fresh"); err != nil {
		t.Fatalf("plan -add: %v\n%s", err, output)
	}
	fresh := editedPlan(t, root).Items[0]
	if fresh.ID != "fresh" || fresh.Budget.Nodes != 120 || !strings.Contains(fresh.Budget.Reason, "no landing changed cmd/z; the session's figure 120") || fresh.Steps[0].Files[0] != "cmd/z" {
		t.Fatalf("the added row is %+v", fresh)
	}
}
