package plan

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestDispatchFallsThroughToFindingsThenTheRun holds what an empty plan's work is, as the week to Nov 7 showed: on
// Nov 1 every open row belonged to another role and the lead sat 6.3 hours. The register's open P1 findings, then its
// P2 ones, each no row names, are filed as rows before the measurement run; a P3 is not, nor a closed finding, nor one
// a row names, SQA-33 not named by SQA-334; and an empty plan's line says which to file. With every one named, the
// measurement run is next.
func TestDispatchFallsThroughToFindingsThenTheRun(t *testing.T) {
	t.Parallel()
	register := []byte(`{"branch": "master", "findings": [
		{"id": "SQA-10", "status": "open", "severity": "P2", "title": "a second-rank defect", "sources": []},
		{"id": "SQA-33", "status": "open", "severity": "P2", "title": "a defect a row names"},
		{"id": "SQA-334", "status": "open", "severity": "P3", "title": "a low-rank tidy"},
		{"id": "SQA-5", "status": "closed", "severity": "P1", "title": "a defect already closed"},
		{"id": "SQA-7", "status": "open", "severity": "P1", "title": "an urgent defect"}]}`)
	document := Plan{Campaign: "fixture", Items: []Item{{ID: "row", Title: "Row", Status: StatusOpen,
		Steps: []Step{{ID: "do", Title: "Do", Status: StatusOpen, Rationale: "SQA-33 asks for it, and SQA-334 is beside it."}}}}}
	filed := testcheck.Must(FindingsToFile(document, register))(t)
	if want := []string{"SQA-7 (P1): an urgent defect", "SQA-10 (P2): a second-rank defect"}; !slices.Equal(filed, want) {
		t.Fatalf("the findings to file = %v, want %v", filed, want)
	}
	unnamed := Plan{Campaign: "fixture"}
	if filed := testcheck.Must(FindingsToFile(unnamed, register))(t); len(filed) != 3 || !strings.HasPrefix(filed[2], "SQA-33 ") {
		t.Fatalf("with no row naming them, the findings to file = %v, want the P1 and both P2s", filed)
	}
	line := dispatchLine(Dispatch{Complete: true, Findings: filed})
	if !strings.Contains(line, "before the measurement run") || !strings.Contains(line, "SQA-7 (P1)") {
		t.Fatalf("an empty plan's line = %q, want the findings to file before the measurement run", line)
	}
	document.Items[0].Steps[0].Rationale += " SQA-7 and SQA-10 too."
	if filed := testcheck.Must(FindingsToFile(document, register))(t); len(filed) != 0 {
		t.Fatalf("with every P1 and P2 named, the findings to file = %v, want none, the measurement run next", filed)
	}
	if line := dispatchLine(Dispatch{Complete: true}); line != "plan complete: every item is done" {
		t.Fatalf("an empty plan with nothing to file reads %q", line)
	}
}
