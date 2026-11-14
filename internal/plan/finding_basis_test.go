package plan

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestAFindingToFileSaysWhatEvidenceItRestsOn holds dispatch's line for a finding to its evidence basis beside its
// severity (findings-basis/print): the lead weighs a finding by what it rests on, observed, shown from source or
// possible, before answering it, and the register says it of each. A finding the register gives no basis is named as
// before, and a finding a change touches is named with its basis too.
func TestAFindingToFileSaysWhatEvidenceItRestsOn(t *testing.T) {
	t.Parallel()
	register := []byte(`{
  "findings": [
    {"id": "SQA-1", "status": "open", "severity": "P2", "evidence_basis": "shown_from_source", "title": "read from the code", "sources": [{"path": "p/p.go"}]},
    {"id": "SQA-2", "status": "open", "severity": "P2", "evidence_basis": "observed", "title": "seen in a run"},
    {"id": "SQA-3", "status": "open", "severity": "P2", "title": "no basis given"}
  ]
}`)
	filed := testcheck.Must(FindingsToFile(Plan{Campaign: "fixture"}, register))(t)
	want := []string{"SQA-1 (P2, shown_from_source): read from the code", "SQA-2 (P2, observed): seen in a run", "SQA-3 (P2): no basis given"}
	if !slices.Equal(filed, want) {
		t.Fatalf("the findings to file are %q, want %q", filed, want)
	}
	touching := testcheck.Must(FindingsTouching(register, []string{"p/p.go"}))(t)
	if want := []string{"SQA-1 (P2, shown_from_source): read from the code; cites p/p.go"}; !slices.Equal(touching, want) {
		t.Fatalf("the findings the change touches are %q, want %q", touching, want)
	}
}
