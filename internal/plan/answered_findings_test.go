package plan

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestARewrittenFindingIsUnansweredAgain holds the fall-through to findings the lead has not answered as they now read
// (triage-current). An answer deferring a finding to the next run's record (SQA-266), or marking it implemented to wait
// for the review's recheck (2026-11-08), keeps it from being filed again while it names the finding's text; a finding
// with no answer is filed. Rewritten by the review after its answer, as nine were up to efa59bd59 that sat unread since,
// it is filed again with the text a new answer names it by, and an answer naming no text answers nothing.
func TestARewrittenFindingIsUnansweredAgain(t *testing.T) {
	t.Parallel()
	answer := func(action, actual string) string {
		return fmt.Sprintf(`{"action": %q, "answered": %q}`, action, openFinding{Actual: actual}.answeredAs())
	}
	filed := func(actual, triage string) []string {
		t.Helper()
		register := fmt.Sprintf(`{"findings": [
			{"id": "SQA-1", "severity": "P2", "actual": %q},
			{"id": "SQA-2", "severity": "P2", "actual": "not answered yet"}],
			"master_lead_triage": {"findings": {%s}}}`, actual, triage)
		return testcheck.Must(FindingsToFile(Plan{Campaign: "fixture"}, []byte(register)))(t)
	}
	unanswered := "SQA-2 [answer " + openFinding{Actual: "not answered yet"}.answeredAs() + "] (P2): not answered yet"
	for _, action := range []string{triageDeferred, triageImplemented, triageDeclined} {
		if got := filed("as first written", `"SQA-1": `+answer(action, "as first written")); !slices.Equal(got, []string{unanswered}) {
			t.Errorf("a finding answered %s: filed %q, want only the unanswered %q", action, got, unanswered)
		}
	}
	rewritten := "SQA-1 [answer " + openFinding{Actual: "as the review rewrote it"}.answeredAs() + "] (P2): as the review rewrote it"
	if got := filed("as the review rewrote it", `"SQA-1": `+answer(triageImplemented, "as first written")); !slices.Equal(got, []string{rewritten, unanswered}) {
		t.Errorf("a finding rewritten after its answer: filed %q, want it again, %q, beside %q", got, rewritten, unanswered)
	}
	if got := filed("as the review rewrote it", `"SQA-1": `+answer(triageImplemented, "as the review rewrote it")); !slices.Equal(got, []string{unanswered}) {
		t.Errorf("a rewritten finding answered again: filed %q, want only %q", got, unanswered)
	}
	if got := filed("as first written", `"SQA-1": {"action": "implemented"}`); len(got) != 2 {
		t.Errorf("an answer naming no text: filed %q, want both findings", got)
	}
}
