package plan

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestAFindingDeferredToTheRunsRecordLetsTheRunStart holds the fall-through to findings a row can act on now (SQA-266):
// a finding whose deciding evidence comes only from the next measurement run, deferred to it in the lead's triage,
// held that run back until a row named it, and a row filed for it kept the plan from emptying. A register with one open
// deferred P2 finding and one open untriaged P2 finding names only the untriaged one.
func TestAFindingDeferredToTheRunsRecordLetsTheRunStart(t *testing.T) {
	t.Parallel()
	register := []byte(`{
  "findings": [
    {"id": "SQA-1", "status": "open", "severity": "P2", "title": "decided by the run's record"},
    {"id": "SQA-2", "status": "open", "severity": "P2", "title": "a fault a row can fix now"}
  ],
  "master_lead_triage": {"findings": {"SQA-1": {"action": "deferred"}}}
}`)
	filed := testcheck.Must(FindingsToFile(Plan{Campaign: "fixture"}, register))(t)
	if want := []string{"SQA-2 (P2): a fault a row can fix now"}; !slices.Equal(filed, want) {
		t.Fatalf("the findings to file are %q, want %q: the deferred one waits for the run's record", filed, want)
	}
}
