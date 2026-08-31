package main

import "testing"

// TestFamilyFilterOutcome pins the empty-selection contract: a model
// whose declared domains exclude the requested family skips by its own
// declaration, an undeclared model's empty selection names a family the
// catalog cannot serve, and any selected suite is unconditional success.
func TestFamilyFilterOutcome(t *testing.T) {
	if err := familyFilterOutcome(1, false, nil, "mmlu"); err != nil {
		t.Fatalf("selected suites = %v, want success", err)
	}
	if err := familyFilterOutcome(0, true, []string{"dna"}, "mmlu"); err != nil {
		t.Fatalf("declared-domain exclusion = %v, want a named skip", err)
	}
	if err := familyFilterOutcome(0, false, nil, "mmlu"); err == nil {
		t.Fatal("undeclared empty selection succeeded; an unserved family must refuse")
	}
}
