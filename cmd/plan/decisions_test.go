package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"overgo/internal/plan"
)

// TestDecisionsListPendingOwnerDecisions holds plan -decisions to the owner's
// open questions: an unanswered decision is listed with its wait, its
// options (the recommended one marked) and the rows it blocks; an answered
// one is not; and the plan refuses a decision whose recommendation is not
// one of its options.
func TestDecisionsListPendingOwnerDecisions(t *testing.T) {
	t.Parallel()
	document := plan.Plan{Decisions: []plan.Decision{
		{ID: "download", Question: "Download the model?", Options: []string{"yes", "no"}, Recommendation: "yes", Blocks: []string{"row"}, Asked: "2026-09-24"},
		{ID: "settled", Question: "Re-verify?", Options: []string{"re-verify", "nothing"}, Recommendation: "re-verify", Asked: "2026-09-21", Answer: "re-verify"},
	}}
	var output bytes.Buffer
	printDecisions(&output, document, time.Date(2026, time.September, 28, 12, 0, 0, 0, time.UTC))
	if want := "download (asked 2026-09-24, 4 days): Download the model?\n  * yes\n    no\n  blocks: row\n"; output.String() != want {
		t.Fatalf("plan -decisions printed %q, want %q", output.String(), want)
	}
	unrecommended := plan.Plan{Decisions: []plan.Decision{
		{ID: "x", Question: "Which?", Options: []string{"a", "b"}, Recommendation: "c", Asked: "2026-09-28"},
	}}
	if err := plan.Validate(unrecommended); err == nil || !strings.Contains(err.Error(), "plan decision") {
		t.Fatalf("a recommendation outside the options validated: %v", err)
	}
}
