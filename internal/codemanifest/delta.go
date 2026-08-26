package codemanifest

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

// ChangeKind identifies how one stable manifest record changed.
type ChangeKind string

const (
	// ChangeAdded identifies a record absent from the base.
	ChangeAdded ChangeKind = "added"
	// ChangeRemoved identifies a record absent from the candidate.
	ChangeRemoved ChangeKind = "removed"
	// ChangeModified identifies a record present with changed authority.
	ChangeModified ChangeKind = "modified"
)

// FileChange records source-byte or file-authority change.
type FileChange struct {
	Path      string     `json:"path"`
	Kind      ChangeKind `json:"kind"`
	Base      *File      `json:"base,omitempty"`
	Candidate *File      `json:"candidate,omitempty"`
}

// SymbolChange records exact declaration and implementation change.
type SymbolChange struct {
	ID               SymbolID   `json:"id"`
	Kind             ChangeKind `json:"kind"`
	Base             *Symbol    `json:"base,omitempty"`
	Candidate        *Symbol    `json:"candidate,omitempty"`
	SignatureChanged bool       `json:"signature_changed,omitempty"`
	BodyChanged      bool       `json:"body_changed,omitempty"`
}

// ExternalInputChange records non-Go authority change.
type ExternalInputChange struct {
	Path      string         `json:"path"`
	Kind      ChangeKind     `json:"kind"`
	Base      *ExternalInput `json:"base,omitempty"`
	Candidate *ExternalInput `json:"candidate,omitempty"`
}

// Delta binds every structural seed to exact base and candidate manifests.
type Delta struct {
	Base           artifact.ID           `json:"base"`
	Candidate      artifact.ID           `json:"candidate"`
	Files          []FileChange          `json:"files,omitempty"`
	Symbols        []SymbolChange        `json:"symbols,omitempty"`
	ExternalInputs []ExternalInputChange `json:"external_inputs,omitempty"`
	Uncertainty    []Uncertainty         `json:"uncertainty,omitempty"`
}

// Diff derives deterministic structural seeds from two canonical manifests.
func Diff(base, candidate Manifest) (Delta, error) {
	if err := base.Validate(); err != nil {
		return Delta{}, errors.New("code manifest delta: invalid base")
	}
	if err := candidate.Validate(); err != nil {
		return Delta{}, errors.New("code manifest delta: invalid candidate")
	}
	delta := Delta{Base: base.ID, Candidate: candidate.ID}
	delta.Files = diffFiles(base.Files, candidate.Files)
	delta.Symbols = diffSymbols(base.Symbols, candidate.Symbols)
	delta.ExternalInputs = diffExternalInputs(base.ExternalInputs, candidate.ExternalInputs)
	delta.Uncertainty = relevantUncertainty(delta, base.Uncertainty, candidate.Uncertainty)
	if base.Analyzer != candidate.Analyzer {
		delta.Uncertainty = append(delta.Uncertainty, Uncertainty{
			Kind:   UncertaintyAnalysis,
			Reason: "base and candidate manifests use different analyzer provenance",
		})
	}
	slices.SortFunc(delta.Uncertainty, compareUncertainty)
	delta.Uncertainty = slices.CompactFunc(delta.Uncertainty, func(left, right Uncertainty) bool {
		return compareUncertainty(left, right) == 0
	})
	return delta, nil
}

func relevantUncertainty(delta Delta, groups ...[]Uncertainty) []Uncertainty {
	paths := make(map[string]bool, len(delta.Files))
	symbols := make(map[string]bool, len(delta.Symbols))
	for _, change := range delta.Files {
		paths[change.Path] = true
	}
	for _, change := range delta.Symbols {
		symbols[symbolKey(change.ID)] = true
	}
	var result []Uncertainty
	for _, group := range groups {
		for _, item := range group {
			if item.Path == "" || paths[item.Path] || item.Symbol != nil && symbols[symbolKey(*item.Symbol)] {
				result = append(result, item)
			}
		}
	}
	return result
}

func diffFiles(base, candidate []File) []FileChange {
	baseIndex := make(map[string]File, len(base))
	candidateIndex := make(map[string]File, len(candidate))
	for _, file := range base {
		baseIndex[file.Path] = file
	}
	for _, file := range candidate {
		candidateIndex[file.Path] = file
	}
	keys := unionKeys(baseIndex, candidateIndex)
	changes := make([]FileChange, 0, len(keys))
	for _, key := range keys {
		before, inBase := baseIndex[key]
		after, inCandidate := candidateIndex[key]
		switch {
		case !inBase:
			changes = append(changes, FileChange{Path: key, Kind: ChangeAdded, Candidate: pointer(after)})
		case !inCandidate:
			changes = append(changes, FileChange{Path: key, Kind: ChangeRemoved, Base: pointer(before)})
		case before.ContentID != after.ContentID || before.Package != after.Package || before.BuildExpression != after.BuildExpression || before.Generated != after.Generated || before.Test != after.Test || !slices.Equal(before.SelectedContexts, after.SelectedContexts):
			changes = append(changes, FileChange{Path: key, Kind: ChangeModified, Base: pointer(before), Candidate: pointer(after)})
		}
	}
	return changes
}

func diffSymbols(base, candidate []Symbol) []SymbolChange {
	baseIndex := make(map[string]Symbol, len(base))
	candidateIndex := make(map[string]Symbol, len(candidate))
	for _, symbol := range base {
		baseIndex[symbolKey(symbol.ID)] = symbol
	}
	for _, symbol := range candidate {
		candidateIndex[symbolKey(symbol.ID)] = symbol
	}
	keys := unionKeys(baseIndex, candidateIndex)
	changes := make([]SymbolChange, 0, len(keys))
	for _, key := range keys {
		before, inBase := baseIndex[key]
		after, inCandidate := candidateIndex[key]
		switch {
		case !inBase:
			changes = append(changes, SymbolChange{ID: after.ID, Kind: ChangeAdded, Candidate: pointer(after), SignatureChanged: true, BodyChanged: after.BodySHA256 != ""})
		case !inCandidate:
			changes = append(changes, SymbolChange{ID: before.ID, Kind: ChangeRemoved, Base: pointer(before), SignatureChanged: true, BodyChanged: before.BodySHA256 != ""})
		default:
			signatureChanged := before.SignatureSHA256 != after.SignatureSHA256
			bodyChanged := before.BodySHA256 != after.BodySHA256
			if signatureChanged || bodyChanged || before.File != after.File || before.Exported != after.Exported {
				changes = append(changes, SymbolChange{
					ID: before.ID, Kind: ChangeModified, Base: pointer(before), Candidate: pointer(after),
					SignatureChanged: signatureChanged, BodyChanged: bodyChanged,
				})
			}
		}
	}
	return changes
}

func diffExternalInputs(base, candidate []ExternalInput) []ExternalInputChange {
	baseIndex := make(map[string]ExternalInput, len(base))
	candidateIndex := make(map[string]ExternalInput, len(candidate))
	for _, input := range base {
		baseIndex[externalKey(input)] = input
	}
	for _, input := range candidate {
		candidateIndex[externalKey(input)] = input
	}
	keys := unionKeys(baseIndex, candidateIndex)
	changes := make([]ExternalInputChange, 0, len(keys))
	for _, key := range keys {
		before, inBase := baseIndex[key]
		after, inCandidate := candidateIndex[key]
		switch {
		case !inBase:
			changes = append(changes, ExternalInputChange{Path: after.Path, Kind: ChangeAdded, Candidate: pointer(after)})
		case !inCandidate:
			changes = append(changes, ExternalInputChange{Path: before.Path, Kind: ChangeRemoved, Base: pointer(before)})
		case before.ContentID != after.ContentID:
			changes = append(changes, ExternalInputChange{Path: before.Path, Kind: ChangeModified, Base: pointer(before), Candidate: pointer(after)})
		}
	}
	return changes
}

func externalKey(input ExternalInput) string {
	return input.Path + "\x00" + input.Kind + "\x00" + input.Owner
}

func unionKeys[T any](left, right map[string]T) []string {
	keys := make(map[string]bool, len(left)+len(right))
	for key := range left {
		keys[key] = true
	}
	for key := range right {
		keys[key] = true
	}
	result := make([]string, 0, len(keys))
	for key := range keys {
		result = append(result, key)
	}
	slices.Sort(result)
	return result
}

func pointer[T any](value T) *T {
	return &value
}

func compareUncertainty(left, right Uncertainty) int {
	return cmp.Or(cmp.Compare(left.Kind, right.Kind), cmp.Compare(left.Path, right.Path), cmp.Compare(left.Context, right.Context), cmp.Compare(optionalSymbolKey(left.Symbol), optionalSymbolKey(right.Symbol)), cmp.Compare(left.Reason, right.Reason))
}
