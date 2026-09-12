package closurescan

import (
	"bytes"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
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
	return validatePermanentAuthority(snapshot, documents, nil)
}

// ValidatePermanentActiveAuthority validates authority per exact canonical
// alias target, including when only part of a multi-binding document remains
// active.
func ValidatePermanentActiveAuthority(
	snapshot repoanalysis.SourceSnapshot,
	documents []closureledger.Document,
	aliases map[string]artifact.ID,
) (AuthorityReport, error) {
	for _, document := range documents {
		for _, binding := range document.Bindings {
			if _, err := closureledger.ActiveAlias(binding); err != nil {
				return AuthorityReport{}, err
			}
		}
	}
	return validatePermanentAuthority(snapshot, documents, func(document closureledger.Document, binding closureledger.SourceBinding) bool {
		alias, _ := closureledger.ActiveAlias(binding)
		return aliases[alias] == document.ID
	})
}

func validatePermanentAuthority(
	snapshot repoanalysis.SourceSnapshot,
	documents []closureledger.Document,
	selected func(closureledger.Document, closureledger.SourceBinding) bool,
) (AuthorityReport, error) {
	candidates, err := ScanSnapshot(snapshot, nil, CandidateAll)
	if err != nil {
		return AuthorityReport{}, err
	}
	issues, err := validateCandidateBindingsSelected(candidates, documents, selected)
	if err != nil {
		return AuthorityReport{}, err
	}
	if len(issues) != 0 {
		return AuthorityReport{}, fmt.Errorf(
			"permanent authority: %d stale active binding(s), first=%s/%s",
			len(issues), issues[0].Kind, issues[0].Name,
		)
	}
	active := compileAuthority(documents, selected)
	for _, document := range documents {
		if selected != nil && !slices.ContainsFunc(document.Bindings, func(binding closureledger.SourceBinding) bool {
			return selected(document, binding)
		}) {
			continue
		}
		if document.Status == closureledger.StatusOpen {
			return AuthorityReport{}, fmt.Errorf("permanent authority: open closure row %s", document.Name)
		}
	}
	report := AuthorityReport{ProductionSites: len(candidates)}
	var missing []error
	for _, candidate := range candidates {
		found := exactAuthority(active[candidate.DeclarationKey()], candidate)
		if found {
			report.ClassifiedSites++
		}
		if candidate.Policy && !found {
			missing = append(missing, fmt.Errorf(
				"permanent authority: uncatalogued production policy %s at %s:%d",
				candidate.Name, candidate.File, candidate.Line,
			))
		}
	}
	if len(missing) != 0 {
		return AuthorityReport{}, errors.Join(missing...)
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

func compileAuthority(
	documents []closureledger.Document,
	selected func(closureledger.Document, closureledger.SourceBinding) bool,
) map[string][]authorityRecord {
	active := make(map[string][]authorityRecord)
	for _, document := range documents {
		for _, binding := range document.Bindings {
			if selected != nil && !selected(document, binding) {
				continue
			}
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
