package closurescan

import (
	"errors"
	"strings"
	"testing"

	"overgo/internal/closureledger"
)

func TestPermanentAuthorityReportsMixedFindings(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/active.go":  "package policy\nconst ActiveLimit = 8\n",
		"internal/removed.go": "package policy\nconst RemovedLimit = 9\n",
		"internal/open.go":    "package policy\nconst OpenLimit = 10\n",
	})
	candidates, err := ScanSnapshot(snapshot, nil, CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	var documents []closureledger.Document
	for _, candidate := range candidates {
		document := candidateDocument(t, candidate)
		if candidate.Name == "OpenLimit" {
			document.Status = closureledger.StatusOpen
		}
		documents = append(documents, document)
	}
	changed, err := snapshot.Overlay(map[string][]byte{
		"internal/active.go":      []byte("package policy\nconst ActiveLimit = 11\n"),
		"internal/removed.go":     nil,
		"internal/new.go":         []byte("package policy\nconst NewLimit = 12\n"),
		"internal/policy_test.go": []byte("package policy\nconst expectedOpenLimit = 10\nfunc verify() { if OpenLimit != expectedOpenLimit { panic(\"policy\") } }\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := ValidatePermanentAuthority(changed, documents)
	if err == nil || report != (AuthorityReport{}) {
		t.Fatalf("mixed debt credited: %+v %v", report, err)
	}
	for _, want := range []string{"stale active binding", "RemovedLimit", "open closure row OpenLimit", "uncatalogued production policy ActiveLimit", "uncatalogued production policy NewLimit", "test policy copy expectedOpenLimit"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("missing %q: %v", want, err)
		}
	}
	missing, ok := errors.AsType[*UncataloguedPolicyError](err)
	if !ok || len(missing.Sites) != 2 {
		t.Fatalf("typed proposal obligations lost: %v", err)
	}
	candidates, err = ScanSnapshot(changed, nil, CandidateConstants)
	if err != nil {
		t.Fatal(err)
	}
	documents = nil
	for _, candidate := range candidates {
		documents = append(documents, candidateDocument(t, candidate))
	}
	if _, err := ValidatePermanentAuthority(changed, documents); err == nil || !strings.Contains(err.Error(), "test policy copy") || strings.Contains(err.Error(), "stale active binding") {
		t.Fatalf("partial repair lost the remaining obligation: %v", err)
	}
	repaired, err := changed.Overlay(map[string][]byte{"internal/policy_test.go": nil})
	if err != nil {
		t.Fatal(err)
	}
	if report, err := ValidatePermanentAuthority(repaired, documents); err != nil || report.ClassifiedSites != len(candidates) {
		t.Fatalf("complete repair: %+v %v", report, err)
	}
}

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
