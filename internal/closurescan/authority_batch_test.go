package closurescan

import (
	"strings"
	"testing"

	"overgo/internal/closureledger"
)

func TestPermanentAuthorityReportsAllMissingPolicies(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/policy.go": "package policy\nconst RequestLimit = 8\nconst RetryLimit = 9\n",
	})
	candidates, err := ScanSnapshot(snapshot, nil, CandidateConstants)
	if err != nil || len(candidates) != 2 {
		t.Fatalf("candidates=%v err=%v", candidates, err)
	}
	var documents []closureledger.Document
	for _, candidate := range candidates {
		documents = append(documents, candidateDocument(t, candidate))
	}
	_, err = ValidatePermanentAuthority(snapshot, nil)
	if err == nil {
		t.Fatal("uncatalogued policies passed")
	}
	for _, candidate := range candidates {
		if !strings.Contains(err.Error(), "uncatalogued production policy "+candidate.Name+" at internal/policy.go:") {
			t.Errorf("missing independent obligation %s in %v", candidate.Name, err)
		}
	}
	_, err = ValidatePermanentAuthority(snapshot, documents[:1])
	if err == nil || strings.Contains(err.Error(), candidates[0].Name) || !strings.Contains(err.Error(), candidates[1].Name) {
		t.Fatalf("partial repair diagnostic: %v", err)
	}
	if report, err := ValidatePermanentAuthority(snapshot, documents); err != nil || report.ClassifiedSites != len(candidates) {
		t.Fatalf("complete repair: %+v, %v", report, err)
	}
}
