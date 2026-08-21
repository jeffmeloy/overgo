package closurescan

import (
	"bytes"

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
	current := make(map[string]Candidate, len(candidates))
	for _, candidate := range candidates {
		current[candidate.DeclarationKey()] = candidate
	}
	var issues []BindingIssue
	for _, document := range documents {
		if err := document.ValidateIdentity(); err != nil {
			issues = append(issues, BindingIssue{Kind: DriftDisposition, Name: document.Name})
			continue
		}
		for _, binding := range document.Bindings {
			key := (Candidate{Kind: binding.Kind, File: binding.File, Scope: binding.Scope, Line: binding.Line, Name: binding.Name}).DeclarationKey()
			candidate, found := current[key]
			if !found {
				issues = append(issues, BindingIssue{Kind: DriftOrphan, Name: binding.Name, File: binding.File})
				continue
			}
			expected, err := candidate.Binding()
			if err != nil {
				return nil, err
			}
			switch {
			case expected.CallsiteID != binding.CallsiteID:
				issues = append(issues, BindingIssue{Kind: DriftCallsite, Name: binding.Name, File: binding.File})
			case expected != binding || !bytes.Equal(candidate.ValueJSON(), document.Value):
				issues = append(issues, BindingIssue{Kind: DriftSource, Name: binding.Name, File: binding.File})
			}
		}
	}
	return issues, nil
}
