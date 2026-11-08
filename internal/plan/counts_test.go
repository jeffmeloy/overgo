package plan

import (
	"slices"
	"strings"
	"testing"
)

// TestANewStepsCountComesFromItsDirectoriesHistory holds a new step's count to what landings in its directories
// counted (cmd/plan holds the command's side): a landing's message gives the paths it changed and the production nodes
// its rules stage counted, a message that states either not gives no count, and the count of a step listing a
// directory is the median of the landings that changed a path in it.
func TestANewStepsCountComesFromItsDirectoriesHistory(t *testing.T) {
	t.Parallel()
	landed := func(nodes, path string) string {
		return "row/do: a change\n\nChange:\n- files: " + path + ", docs/plan.json\n\nMeasured by the gate: vet 1s, rules 1s (production " +
			nodes + " nodes of the 700 its row may add, test code +9 nodes), test 1s\n\nLanded-Row: row/do\n"
	}
	count, read := ReadLandedCount(landed("+253", "internal/x/x.go"))
	if !read || count.Nodes != 253 || !slices.Equal(count.Paths, []string{"internal/x/x.go", "docs/plan.json"}) {
		t.Fatalf("the landing's message reads as %+v, %t", count, read)
	}
	if _, read := ReadLandedCount("sqa: a review's register\n\n- files: docs/sqa_findings.json\n"); read {
		t.Fatal("a message with no counted nodes gave a count")
	}
	var history []LandedCount
	for _, message := range []string{landed("+100", "internal/x/a.go"), landed("+500", "internal/x/b.go"), landed("+300", "internal/x/c.go"),
		landed("+9000", "internal/y/y.go"), landed("-40", "internal/x/d.go")} {
		landing, _ := ReadLandedCount(message)
		history = append(history, landing)
	}
	if nodes, landings := HistoryCount(history, []string{"internal/x"}); nodes != 300 || landings != 4 {
		t.Fatalf("a step in internal/x is counted %d from %d landings, want the median 300 of 4", nodes, landings)
	}
	if nodes, landings := HistoryCount(history, []string{"cmd/z"}); nodes != 0 || landings != 0 {
		t.Fatalf("a step no landing touched is counted %d from %d landings", nodes, landings)
	}
	if reason := HistoryReason(300, 4, 120, []string{"internal/x"}); !strings.Contains(reason, "median of what 4 landings that changed internal/x") || !strings.Contains(reason, "the session's figure 120") {
		t.Fatalf("the reason %q does not say where the count came from and the session's figure", reason)
	}
}
