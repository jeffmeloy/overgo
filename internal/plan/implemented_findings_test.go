package plan

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestAFindingTheLeadImplementedWaitsForItsReview holds the fall-through to findings the lead has not answered: on
// 2026-11-08 the rows filed from it landed and dispatch at once named the same 13 findings to file again, since they
// stay open until the review closes them. A register with one open P2 finding the lead's triage marks implemented and
// one open untriaged P2 finding names only the untriaged one.
func TestAFindingTheLeadImplementedWaitsForItsReview(t *testing.T) {
	t.Parallel()
	register := []byte(`{
  "findings": [
    {"id": "SQA-1", "status": "open", "severity": "P2", "title": "answered, awaiting the reviewer"},
    {"id": "SQA-2", "status": "open", "severity": "P2", "title": "not answered yet"}
  ],
  "master_lead_triage": {"findings": {"SQA-1": {"action": "implemented", "rows": ["row/do"]}}}
}`)
	filed := testcheck.Must(FindingsToFile(Plan{Campaign: "fixture"}, register))(t)
	if want := []string{"SQA-2 (P2): not answered yet"}; !slices.Equal(filed, want) {
		t.Fatalf("the findings to file are %q, want %q: the implemented one waits for its review", filed, want)
	}
}
