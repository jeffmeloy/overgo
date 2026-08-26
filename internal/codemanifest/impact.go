package codemanifest

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// Impact is the deterministic transitive surface capable of observing a
// manifest delta.
type Impact struct {
	Base           string                `json:"base"`
	Candidate      string                `json:"candidate"`
	Seeds          []SymbolID            `json:"seeds,omitempty"`
	Reachable      []SymbolID            `json:"reachable,omitempty"`
	Packages       []string              `json:"packages,omitempty"`
	ExternalInputs []ExternalInputChange `json:"external_inputs,omitempty"`
	Uncertainty    []Uncertainty         `json:"uncertainty,omitempty"`
}

// Close computes reverse reachability over the union of the base and
// candidate graphs. This preserves observers of removed as well as added code.
func Close(base, candidate Manifest, delta Delta) (Impact, error) {
	if err := base.Validate(); err != nil {
		return Impact{}, errors.New("code manifest impact: invalid base")
	}
	if err := candidate.Validate(); err != nil {
		return Impact{}, errors.New("code manifest impact: invalid candidate")
	}
	if delta.Base != base.ID || delta.Candidate != candidate.ID {
		return Impact{}, errors.New("code manifest impact: delta identity mismatch")
	}
	seeds := map[string]SymbolID{}
	for _, change := range delta.Symbols {
		seeds[symbolKey(change.ID)] = change.ID
	}
	reverse := map[string]map[string]SymbolID{}
	addReverseReferences(reverse, base.References)
	addReverseReferences(reverse, candidate.References)
	reachable := map[string]SymbolID{}
	queue := make([]SymbolID, 0, len(seeds))
	for _, seed := range seeds {
		reachable[symbolKey(seed)] = seed
		queue = append(queue, seed)
	}
	for len(queue) != 0 {
		current := queue[0]
		queue = queue[1:]
		for key, caller := range reverse[symbolKey(current)] {
			if _, seen := reachable[key]; seen {
				continue
			}
			reachable[key] = caller
			queue = append(queue, caller)
		}
	}
	impact := Impact{
		Base: base.ID.String(), Candidate: candidate.ID.String(),
		Seeds: symbolValues(seeds), Reachable: symbolValues(reachable),
		ExternalInputs: slices.Clone(delta.ExternalInputs),
		Uncertainty:    slices.Clone(delta.Uncertainty),
	}
	packages := map[string]bool{}
	for _, symbol := range impact.Reachable {
		packages[symbol.Package] = true
	}
	for _, change := range delta.Files {
		if change.Base != nil {
			packages[change.Base.Package] = true
		}
		if change.Candidate != nil {
			packages[change.Candidate.Package] = true
		}
	}
	for _, change := range delta.ExternalInputs {
		if change.Base != nil {
			packages[change.Base.Owner] = true
		}
		if change.Candidate != nil {
			packages[change.Candidate.Owner] = true
		}
	}
	impact.Packages = slices.Sorted(maps.Keys(packages))
	impact.Uncertainty = appendReachableUncertainty(impact.Uncertainty, reachable, delta.Files, base.Uncertainty, candidate.Uncertainty)
	return impact, nil
}

// ExclusionAuthority refuses impact with unresolved structural boundaries.
func (i Impact) ExclusionAuthority() error {
	if i.Base == "" || i.Candidate == "" {
		return errors.New("code manifest impact: missing manifest identities")
	}
	if len(i.Uncertainty) != 0 {
		return fmt.Errorf("code manifest impact: %d unresolved boundaries", len(i.Uncertainty))
	}
	return nil
}

func addReverseReferences(reverse map[string]map[string]SymbolID, references []Reference) {
	for _, reference := range references {
		target := symbolKey(reference.To)
		if reverse[target] == nil {
			reverse[target] = map[string]SymbolID{}
		}
		reverse[target][symbolKey(reference.From)] = reference.From
	}
}

func symbolValues(values map[string]SymbolID) []SymbolID {
	result := make([]SymbolID, 0, len(values))
	for _, value := range values {
		result = append(result, value)
	}
	slices.SortFunc(result, func(left, right SymbolID) int { return compareSymbolID(left, right) })
	return result
}

func compareSymbolID(left, right SymbolID) int {
	return strings.Compare(symbolKey(left), symbolKey(right))
}

func appendReachableUncertainty(result []Uncertainty, reachable map[string]SymbolID, changes []FileChange, groups ...[]Uncertainty) []Uncertainty {
	paths := make(map[string]bool, len(changes))
	for _, change := range changes {
		paths[change.Path] = true
	}
	for _, group := range groups {
		for _, item := range group {
			if item.Path == "" || paths[item.Path] || item.Symbol != nil && containsSymbol(reachable, *item.Symbol) {
				result = append(result, item)
			}
		}
	}
	slices.SortFunc(result, compareUncertainty)
	return slices.CompactFunc(result, func(left, right Uncertainty) bool { return compareUncertainty(left, right) == 0 })
}

func containsSymbol(values map[string]SymbolID, symbol SymbolID) bool {
	_, found := values[symbolKey(symbol)]
	return found
}
