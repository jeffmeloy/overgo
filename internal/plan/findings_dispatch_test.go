package plan

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAnUnansweredP1IsDispatchedFirst holds every dispatch to the findings no row names, where only an empty plan's
// had read them, so none was dispatched from 11 November while the plan held rows (owner, 2026-11-13): with an open row
// and a register holding a P1 and a P2 no row names, the dispatch puts filing the P1 first and names the P2 by id
// beside the step; with a row naming the P1, the step comes first again, the P2 still named; and a P2 the lead's
// triage declined is no longer named.
func TestAnUnansweredP1IsDispatchedFirst(t *testing.T) {
	t.Parallel()
	repository := planRepository(t, Plan{Campaign: "fixture", Items: []Item{{ID: "row", Title: "Row",
		Steps: []Step{{ID: "do", Title: "Do", Verify: "true"}}}}})
	register := func(triage string) {
		t.Helper()
		testcheck.NoError(t, os.WriteFile(filepath.Join(repository, filepath.FromSlash(FindingsRegister)), []byte(`{"findings": [
			{"id": "SQA-7", "severity": "P1", "actual": "an urgent defect"},
			{"id": "SQA-10", "severity": "P2", "actual": "a second-rank defect"}],
			"master_lead_triage": {"findings": {`+triage+`}}}`), 0o644))
	}
	line := func() string {
		t.Helper()
		return testcheck.Must(ResolveDispatch(t.Context(), repository, DispatchRequest{Mode: ExecutionInteractive}))(t).Line
	}
	answered := func(id, action, actual string) string {
		return `"` + id + `": {"action": "` + action + `", "answered": "` + openFinding{Actual: actual}.answeredAs() + `"}`
	}
	urgent, second := answered("SQA-7", "implemented", "an urgent defect"), answered("SQA-10", "declined", "a second-rank defect")
	register("")
	if got := line(); !strings.HasPrefix(got, "first file as a row the P1 finding no row names, SQA-7 [answer ") || !strings.Contains(got, "row / do") || !strings.Contains(got, "SQA-10") {
		t.Fatalf("with an unanswered P1 the dispatch reads %q, want filing SQA-7 first, then row/do, SQA-10 named", got)
	}
	register(urgent)
	if got := line(); !strings.HasPrefix(got, "row / do") || !strings.Contains(got, "SQA-10") || strings.Contains(got, "SQA-7") {
		t.Fatalf("with the P1 answered the dispatch reads %q, want row/do first and SQA-10 alone named", got)
	}
	register(urgent + ", " + second)
	if got := line(); strings.Contains(got, "SQA-") {
		t.Fatalf("with both findings answered, one declined, the dispatch reads %q, want neither named", got)
	}
}
