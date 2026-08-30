package closurescan

import (
	"bytes"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/repoanalysis"
)

type BindingDrift string

const (
	DriftDisposition BindingDrift = "disposition"
	DriftOrphan      BindingDrift = "orphan"
	DriftSource      BindingDrift = "source"
	DriftCallsite    BindingDrift = "callsite"
)

type BindingIssue struct {
	Kind BindingDrift `json:"kind"`
	Name string       `json:"name"`
	File string       `json:"file"`
}

func ValidateBindings(snapshot repoanalysis.SourceSnapshot, documents []closureledger.Document) ([]BindingIssue, error) {
	candidates, err := ScanSnapshot(snapshot, nil, CandidateAll)
	if err != nil {
		return nil, err
	}
	return validateCandidateBindings(candidates, documents)
}

// ValidateActiveBindings checks only bindings whose own canonical active alias
// targets the document. Selecting a multi-binding document through one alias
// does not grant authority to its retired sibling bindings.
func ValidateActiveBindings(
	snapshot repoanalysis.SourceSnapshot,
	documents []closureledger.Document,
	aliases map[string]artifact.ID,
) ([]BindingIssue, error) {
	candidates, err := ScanSnapshot(snapshot, nil, CandidateAll)
	if err != nil {
		return nil, err
	}
	for _, document := range documents {
		for _, binding := range document.Bindings {
			if _, err := closureledger.ActiveAlias(binding); err != nil {
				return nil, err
			}
		}
	}
	return validateCandidateBindingsSelected(candidates, documents, func(document closureledger.Document, binding closureledger.SourceBinding) bool {
		alias, _ := closureledger.ActiveAlias(binding)
		return aliases[alias] == document.ID
	})
}

func validateCandidateBindings(candidates []Candidate, documents []closureledger.Document) ([]BindingIssue, error) {
	return validateCandidateBindingsSelected(candidates, documents, nil)
}

func validateCandidateBindingsSelected(
	candidates []Candidate,
	documents []closureledger.Document,
	selected func(closureledger.Document, closureledger.SourceBinding) bool,
) ([]BindingIssue, error) {
	current := make(map[string][]Candidate, len(candidates))
	for _, candidate := range candidates {
		current["s\x00"+candidate.StructuralID] = append(current["s\x00"+candidate.StructuralID], candidate)
		migration := rebindKey(candidate.Kind, candidate.Package, candidate.File, candidate.Scope, candidate.Name, candidate.Expression)
		current["m\x00"+migration] = append(current["m\x00"+migration], candidate)
	}
	var issues []BindingIssue
	for _, document := range documents {
		if selected != nil && !slices.ContainsFunc(document.Bindings, func(binding closureledger.SourceBinding) bool {
			return selected(document, binding)
		}) {
			continue
		}
		if err := document.ValidateIdentity(); err != nil {
			issues = append(issues, BindingIssue{Kind: DriftDisposition, Name: document.Name})
			continue
		}
		for _, binding := range document.Bindings {
			if selected != nil && !selected(document, binding) {
				continue
			}
			key := "s\x00" + binding.StructuralID
			if binding.StructuralID == "" {
				key = "m\x00" + rebindKey(binding.Kind, binding.Package, binding.File, binding.Scope, binding.Name, binding.Expression)
			}
			matches := current[key]
			if len(matches) != 1 {
				issues = append(issues, BindingIssue{Kind: DriftOrphan, Name: binding.Name, File: binding.File})
				continue
			}
			candidate := matches[0]
			expected, err := candidate.Binding()
			if err != nil {
				return nil, err
			}
			switch {
			case binding.StructuralID != "" && expected.CallsiteID != binding.CallsiteID:
				issues = append(issues, BindingIssue{Kind: DriftCallsite, Name: binding.Name, File: binding.File})
			case binding.StructuralID == "" || expected.StructuralID != binding.StructuralID ||
				expected.SourceID != binding.SourceID || expected.Expression != binding.Expression ||
				!bytes.Equal(candidate.ValueJSON(), document.Value):
				issues = append(issues, BindingIssue{Kind: DriftSource, Name: binding.Name, File: binding.File})
			}
		}
	}
	return issues, nil
}
