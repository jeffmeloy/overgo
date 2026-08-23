package closurescan

import (
	"bytes"
	"fmt"

	"overgo/internal/closureledger"
	"overgo/internal/repoanalysis"
)

// AuthorityReport summarizes permanent literal authority.
type AuthorityReport struct {
	ProductionSites int
	ClassifiedSites int
	TestSites       int
}

// ValidatePermanentAuthority rejects all literal-authority debt.
func ValidatePermanentAuthority(
	snapshot repoanalysis.SourceSnapshot,
	documents []closureledger.Document,
) (AuthorityReport, error) {
	candidates, err := ScanSnapshot(snapshot, nil, CandidateAll)
	if err != nil {
		return AuthorityReport{}, err
	}
	issues, err := validateCandidateBindings(candidates, documents)
	if err != nil {
		return AuthorityReport{}, err
	}
	if len(issues) != 0 {
		return AuthorityReport{}, fmt.Errorf(
			"permanent authority: %d stale active binding(s), first=%s/%s",
			len(issues), issues[0].Kind, issues[0].Name,
		)
	}
	active := compileAuthority(documents)
	for _, document := range documents {
		if document.Status == closureledger.StatusOpen {
			return AuthorityReport{}, fmt.Errorf("permanent authority: open closure row %s", document.Name)
		}
	}
	report := AuthorityReport{ProductionSites: len(candidates)}
	for _, candidate := range candidates {
		found := exactAuthority(active[candidate.DeclarationKey()], candidate)
		if found {
			report.ClassifiedSites++
		}
		if candidate.Policy && !found {
			return AuthorityReport{}, fmt.Errorf(
				"permanent authority: uncatalogued production policy %s at %s:%d",
				candidate.Name, candidate.File, candidate.Line,
			)
		}
	}
	tests, err := CensusTestLiterals(snapshot)
	if err != nil {
		return AuthorityReport{}, err
	}
	report.TestSites = len(tests)
	for _, site := range tests {
		if site.Class == TestPolicyCopy {
			return AuthorityReport{}, fmt.Errorf(
				"permanent authority: test policy copy %s at %s:%d",
				site.Name, site.File, site.Line,
			)
		}
	}
	return report, nil
}

type authorityRecord struct {
	document closureledger.Document
	binding  closureledger.SourceBinding
}

func compileAuthority(documents []closureledger.Document) map[string][]authorityRecord {
	active := make(map[string][]authorityRecord)
	for _, document := range documents {
		for _, binding := range document.Bindings {
			key := (Candidate{
				Kind: binding.Kind, Package: binding.Package, File: binding.File, Scope: binding.Scope,
				Line: binding.Line, Name: binding.Name, StructuralID: binding.StructuralID,
			}).DeclarationKey()
			active[key] = append(active[key], authorityRecord{document: document, binding: binding})
		}
	}
	return active
}

func exactAuthority(records []authorityRecord, candidate Candidate) bool {
	for _, record := range records {
		binding := record.binding
		if binding.Kind == candidate.Kind && binding.Package == candidate.Package &&
			binding.File == candidate.File && binding.Scope == candidate.Scope &&
			binding.Name == candidate.Name && binding.StructuralID == candidate.StructuralID &&
			binding.SourceID == candidate.SourceID && binding.CallsiteID == candidate.CallsiteID &&
			binding.Expression == candidate.Expression && bytes.Equal(record.document.Value, candidate.ValueJSON()) {
			return true
		}
	}
	return false
}
